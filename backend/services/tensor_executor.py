"""
Tensor Executor — 执行图的 forward pass，返回每个节点的输出统计信息。
复用了统一代码生成器的 SIGNATURES 和构建逻辑。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import sys
import copy
import hashlib
import json
from collections import deque
from functools import lru_cache

from backend.services import codegen_facade


def _graph_key(flow_json: dict) -> str:
    """Compute a cache key from graph structure (excludes positions)."""
    nodes = flow_json.get("nodes", [])
    edges = flow_json.get("edges", [])
    # Only use id, type, data (params) — NOT position
    key_nodes = [{"id": n["id"], "type": n.get("type", ""), "data": n.get("data", {})} for n in sorted(nodes, key=lambda n: n["id"])]
    key_edges = [{"source": e["source"], "target": e["target"]} for e in sorted(edges, key=lambda e: f"{e['source']}->{e['target']}")]
    return hashlib.md5(json.dumps({"nodes": key_nodes, "edges": key_edges}, sort_keys=True).encode()).hexdigest()


# Simple model cache: key → (model, node_var_map)
_model_cache: dict[str, tuple[nn.Module, dict[str, str]]] = {}
_MAX_CACHE = 8


def _validate_flow_json(flow_json: dict) -> None:
    nodes = flow_json.get("nodes")
    edges = flow_json.get("edges")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Invalid graph: nodes must be a non-empty list")
    if not isinstance(edges, list):
        raise ValueError("Invalid graph: edges must be a list")

    node_ids: set[str] = set()
    for node in nodes:
        node_id = node.get("id") if isinstance(node, dict) else None
        if not node_id:
            raise ValueError("Invalid graph: every node must include an id")
        node_ids.add(node_id)

    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("Invalid graph: every edge must be an object")
        source = edge.get("source")
        target = edge.get("target")
        if source not in node_ids or target not in node_ids:
            raise ValueError(f"Invalid graph: edge references missing node ({source} -> {target})")


def _tensor_stats(t: torch.Tensor) -> dict:
    """返回单个张量的统计信息字典。"""
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "mean": round(float(t.mean().item()), 6),
        "std": round(float(t.std().item()), 6),
        "min": round(float(t.min().item()), 6),
        "max": round(float(t.max().item()), 6),
        "numel": t.numel(),
    }


def _build_model_from_flow(flow_json: dict) -> tuple[nn.Module, dict[str, str]]:
    """
    生成模型代码并执行，构造 nn.Module 实例。
    返回 (model, node_id -> output_var) 映射。
    使用 LRU 缓存避免重复编译相同图结构。
    """
    key = _graph_key(flow_json)

    # Cache hit — 直接返回实例（需要重新实例化避免共享状态）
    if key in _model_cache:
        _, node_var_map = _model_cache[key]
        # Re-instantiate to avoid batchnorm/dropout state issues
        code = codegen_facade.generate_model(flow_json)
        namespace = {"__name__": "__tensor_executor__", "torch": torch, "nn": nn}
        exec(compile(code, "<tensor_executor>", "exec"), namespace)
        model_class = namespace["FlowHamsterModel"]
        model = model_class()
        return model, node_var_map

    # 1. 生成模型代码
    code = codegen_facade.generate_model(flow_json)

    # 2. 构造执行命名空间（包含所有需要的 torch 符号）
    namespace = {
        "__name__": "__tensor_executor__",
        "torch": torch,
        "nn": nn,
    }

    # 3. 执行代码，填充 namespace
    exec(compile(code, "<tensor_executor>", "exec"), namespace)

    # 4. 取模型类并实例化
    model_class = namespace["FlowHamsterModel"]
    model = model_class()

    # 5. 构建 node_id → output_var 映射
    sorted_blocks = codegen_facade.build_ast(flow_json)
    node_var_map: dict[str, str] = {}
    for blk in sorted_blocks:
        if blk.node_id and not blk.node_id.startswith("_icat_"):
            node_var_map[blk.node_id] = blk.output_var

    # 6. 缓存（仅缓存 node_var_map，不缓存模型实例以避免状态污染）
    if len(_model_cache) >= _MAX_CACHE:
        _model_cache.pop(next(iter(_model_cache)))
    _model_cache[key] = (model, node_var_map)

    return model, node_var_map


def execute_forward(flow_json: dict, input_shape: list[int] | None = None) -> dict[str, dict]:
    """
    执行图的 forward pass，记录每个（非 internal-concat）节点的输出张量统计。

    Returns:
        {
          "node_id": {
            "shape": [...],
            "dtype": "torch.float32",
            "mean": 0.123,
            "std": 0.456,
            "min": -1.5,
            "max": 2.3,
            "numel": 12345
          }
        }
    """
    if input_shape is None:
        input_shape = [1, 3, 224, 224]
    _validate_flow_json(flow_json)

    # 1. 构建模型
    model, node_var_map = _build_model_from_flow(flow_json)
    model.eval()

    # 2. 准备虚拟输入
    x = torch.randn(*input_shape)

    # 3. 注册前向钩子，捕获每个子模块的输出
    #    注意：FlowHamsterModel.forward() 中，每个操作结果都保存在局部变量，
    #    但 self.{var} 只对 nn.Module 子模块生效（conv2d, linear, batchnorm 等）。
    #    对于 operation 类（add, mul, concat），它们的输出是局部变量，
    #    我们需要在 forward 过程中手动记录。
    #
    #    方案：拦截 forward 本身，用一个 dict 收集输出。
    #    在执行前给模型一个 hook，把所有中间输出映射到 node_id。

    outputs: dict[str, torch.Tensor] = {}
    stored_x = {}

    # 记录每个命名子模块的输出
    def make_hook(node_id: str):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                outputs[node_id] = output.detach()
        return hook_fn

    # 钩住所有 nn 子模块（按 instance_name，即 output_var）
    var_to_node_id: dict[str, str] = {v: k for k, v in node_var_map.items()}

    handles = []
    for name, module in model.named_modules():
        # name 形如 "conv2d_1" 或 "conv2d_1.bn"（带子层）
        # 顶层模块（不含 "."）对应 block output_var
        if "." not in name:
            nid = var_to_node_id.get(name)
            if nid:
                handles.append(module.register_forward_hook(make_hook(nid)))

    # 4. 执行 forward
    try:
        with torch.no_grad():
            model(x)
    except Exception as e:
        # 若 forward 失败，尝试返回已捕获的部分结果
        if not outputs:
            raise RuntimeError(f"Forward pass failed: {e}") from e

    # 5. 卸载钩子
    for h in handles:
        h.remove()

    # 6. 构建结果 dict，只返回有捕获结果的节点
    result = {}
    for node_id, tensor in outputs.items():
        result[node_id] = _tensor_stats(tensor)

    return result
