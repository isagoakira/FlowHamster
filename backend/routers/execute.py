"""
POST /api/execute — 在本地 subprocess 执行训练代码
POST /api/execute/forward — 执行图的 forward pass，返回张量统计
POST /api/execute/gradients — 分析图的梯度流

所有路由共享 /api/execute 前缀。
超时限制为 60 秒，超时后进程会被强制终止。
"""
from fastapi import APIRouter
from pydantic import BaseModel
from copy import deepcopy
import subprocess
import tempfile
import os
import sys

from backend.services.dataflow_compiler import compile_dataflow
from backend.services.unified_code_gen import generate

router = APIRouter(prefix="/execute", tags=["execute"])


class ExecuteRequest(BaseModel):
    """
    本地代码执行请求

    Attributes:
        code: 要执行的 Python 代码字符串（兼容旧调用）
        graph: 可选，模型节点图；提供时后端会生成训练脚本并执行
        data_graph: 可选，数据节点图
        bindings: 可选，数据字段到模型输入 / 训练 target 的绑定
        training_config: 可选，训练配置
        input_shape: 可选，输入张量的形状，用于 forward pass 预览
        target_device: 可选执行设备，提供时覆盖 training_config.runtime.device
        data_loader_config: 可选，数据加载器配置
    """
    code: str | None = None
    graph: dict | None = None
    data_graph: dict | None = None
    bindings: list[dict] | None = None
    training_config: dict | None = None
    input_shape: list[int] | None = None
    target_device: str | None = None  # "cpu" or "cuda"
    data_loader_config: dict | None = None


class ExecuteResponse(BaseModel):
    """
    本地代码执行响应

    Attributes:
        success: 是否成功执行（返回码为 0）
        output: 标准输出 + 标准错误的合并内容
        error: 如果失败，包含错误信息
    """
    success: bool
    output: str
    error: str | None = None


def _build_executable_code(req: ExecuteRequest) -> str:
    if req.code is not None:
        return req.code

    if req.graph is None:
        raise ValueError("Either code or graph must be provided.")

    training_config = deepcopy(req.training_config) if isinstance(req.training_config, dict) else req.training_config
    if req.target_device:
        if training_config is None:
            training_config = {"runtime": {"device": req.target_device}}
        else:
            runtime = training_config.setdefault("runtime", {})
            runtime["device"] = req.target_device

    options: dict = {}
    if training_config is not None:
        options["training_config"] = training_config

    if req.data_graph or req.bindings:
        compiled = compile_dataflow(
            model_graph=req.graph,
            data_graph=req.data_graph,
            bindings=req.bindings,
            training_config=training_config,
        )
        if compiled.has_workflow_runtime:
            options["workflow_scaffold"] = compiled.python_scaffold
            options["workflow_runtime"] = compiled.has_workflow_runtime

    return generate(req.graph, options=options)


@router.post("", response_model=ExecuteResponse)
async def execute_code(req: ExecuteRequest):
    """
    在本地 subprocess 中执行 Python 代码。

    代码会写入临时文件，通过 `python` 命令执行。
    超时限制为 60 秒，超时后进程被强制终止。

    Returns:
        ExecuteResponse: 包含执行结果
    """
    try:
        code = _build_executable_code(req)
    except Exception as e:
        return ExecuteResponse(success=False, output="", error=str(e))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout + result.stderr
        return ExecuteResponse(
            success=result.returncode == 0,
            output=output,
            error=result.stderr if result.returncode != 0 else None,
        )
    except subprocess.TimeoutExpired:
        return ExecuteResponse(success=False, output="", error="Execution timed out (60s limit)")
    except Exception as e:
        return ExecuteResponse(success=False, output="", error=str(e))
    finally:
        os.unlink(tmp_path)


# ── Forward pass tensor preview ────────────────────────────────────────────────

class ForwardRequest(BaseModel):
    """
    Forward pass 执行请求

    Attributes:
        nodes: 节点列表（flow_json.nodes）
        edges: 边列表（flow_json.edges）
        input_shape: 输入张量形状，默认 [1, 3, 224, 224]
    """
    nodes: list
    edges: list
    input_shape: list[int] = [1, 3, 224, 224]


@router.post("/forward")
async def run_forward(req: ForwardRequest):
    """
    执行图的 forward pass，返回各节点输出张量的统计信息。

    用于在浏览器中预览数据流经网络后的张量形状和数值范围。
    不执行梯度计算，仅做前向传播。

    Returns:
        dict: {"success": bool, "outputs": [...]}
    """
    from backend.services.tensor_executor import execute_forward

    flow_json = {"nodes": req.nodes, "edges": req.edges}
    try:
        results = execute_forward(flow_json, input_shape=req.input_shape)
        return {"success": True, "outputs": results}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Gradient analysis ─────────────────────────────────────────────────────────

class GradientRequest(BaseModel):
    """
    梯度分析请求

    Attributes:
        nodes: 节点列表
        edges: 边列表
    """
    nodes: list[dict]
    edges: list[dict]


@router.post("/gradients")
async def get_gradients(req: GradientRequest):
    """
    分析图中各节点的梯度流。

    返回每个参数张量的梯度范数，用于判断梯度消失/爆炸情况。

    Returns:
        dict: {"success": bool, "data": {...}}
    """
    from backend.services.gradient_analyzer import analyze_gradients

    try:
        result = analyze_gradients({"nodes": req.nodes, "edges": req.edges})
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
