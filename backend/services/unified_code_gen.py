"""
FlowHamster 统一代码生成器
基于 ast_core.py 架构 + code_gen_v2.py inline class 模式

关键设计：
- 使用 ast_core.py 的 SIGNATURES、graph pruning、build_ast
- 修复 mamba/crossattention/reshape/lstm bug
- 从 code_gen_v2.py 移植 inline class 定义（避免外部 import 依赖）
- 单一入口：UnifiedCodeGenerator.generate_model() / generate_full()
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict, deque

# ─────────────────────────────────────────────────────────────────
# Inline auxiliary class generators (ported from code_gen_v2.py)
# ─────────────────────────────────────────────────────────────────

def _generate_aux_classes(needed: set[str]) -> str:
    """Generate inline class definitions for SelfAttention, CrossAttention, Mamba, MLP, FFN, DropPath."""
    parts = []
    if "selfattention" in needed:
        parts.append('''
class SelfAttention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)
''')

    if "crossattention" in needed:
        parts.append('''
class CrossAttention(nn.Module):
    """Multi-head cross-attention (Q from target, K/V from source)"""
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q, k, v):
        B, Nq, C = q.shape
        Nk = k.shape[1]
        q = self.q(q).reshape(B, Nq, self.heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(k).reshape(B, Nk, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        return self.proj(x)
''')

    if "mamba" in needed:
        parts.append('''
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MambaBlock(nn.Module):
    """Single Mamba-2 SSM block with full selective scan."""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: str = "auto", dropout: float = 0.0):
        super().__init__()
        d_inner = expand * d_model
        dt_rank_val = max(d_model // 16, 1) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, bias=False)
        self.dt_proj = nn.Linear(d_inner, dt_rank_val, bias=True)
        self.x_proj = nn.Linear(d_inner, dt_rank_val + d_state * 2, bias=False)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x_p, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_p.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)

        select = self.x_proj(x_conv)
        dt_rank_val = self.dt_proj.out_features
        d_state = self.A_log.shape[1]
        dt = F.softplus(self.dt_proj(x_conv))
        B = select[:, :, dt_rank_val:dt_rank_val + d_state]
        C = select[:, :, dt_rank_val + d_state:]

        dA = torch.exp(torch.einsum("bld,dn->bldn", dt, self.A_log.exp()))
        dB = torch.einsum("bld,bln->bldn", dt, B)
        h = torch.zeros(batch, x_conv.shape[-1], d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x_conv[:, t].unsqueeze(-1)
            ys.append(torch.einsum("bn,bn->b", h, C[:, t]))
        y = torch.stack(ys, dim=1)
        y = y + x_conv * self.D
        y = y * torch.sigmoid(z)
        return self.dropout(self.out_proj(y))


class Mamba(nn.Module):
    """
    Mamba-2: multi-layer stack with residual connections and RMSNorm.
    Based on: https://arxiv.org/abs/2405.xxxxx
    """
    def __init__(self, d_model: int = 512, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: str = "auto", dropout: float = 0.0,
                 n_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dt_rank, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x
        return self.norm(x)
''')

    if "mlp" in needed:
        parts.append('''
class MLP(nn.Module):
    """Multi-layer perceptron"""
    def __init__(self, dim: int, hidden_dim: int, depth: int = 2, out_dim: int = None):
        super().__init__()
        out_dim = out_dim or dim
        layers = []
        for i in range(depth):
            in_d = dim if i == 0 else hidden_dim
            out_d = out_dim if i == depth - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < depth - 1:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
''')

    if "ffn" in needed:
        parts.append('''
class FFN(nn.Module):
    """Feed-forward network (GELU nonlinearity)"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))
''')

    if "droppath" in needed:
        parts.append('''
class DropPath(nn.Module):
    """Stochastic depth drop path"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x / keep_prob * random_tensor
        return output
''')

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────
# Node type normalization
# ─────────────────────────────────────────────────────────────────

NodeCategory = str  # "io" | "module" | "operation" | "training" | "evaluation"

NODE_TYPE_ALIASES = {
    "adaptiveavgpool": "adaptiveavgpool2d",
    "avgpool": "avgpool2d",
    "batchnorm": "batchnorm2d",
    "cosineannealing": "cosineannealinglr",
    "maxpool": "maxpool2d",
    "reduceonplateau": "reducelronplateau",
    "slicenode": "slice",
    "splitnode": "split",
    "transpose": "transpose",
    "transposenode": "transpose",
}


def normalize_node_type(node_type: str) -> str:
    return NODE_TYPE_ALIASES.get(node_type, node_type)


def normalize_name_token(value: str) -> str:
    token = value.replace("-", "_").replace(":", "_").replace(".", "_").lower()
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")


def get_block_name_token(block: "NodeBlock") -> str:
    if block.node_id.startswith("_icat_"):
        return "merge_concat"
    return normalize_name_token(block.op_type)


# 操作类型常数
INLINE_AUX_CLASSES = frozenset({"selfattention", "crossattention", "mamba", "mlp", "ffn", "droppath"})

# Edge handle constants
HANDLE_RESULT = "result"
HANDLE_X = "x"
HANDLE_A = "a"
HANDLE_B = "b"
HANDLE_Q = "q"
HANDLE_KV = "kv"
HANDLE_K = "k"
HANDLE_V = "v"
HANDLE_SRC = "src"
HANDLE_TGT = "tgt"
HANDLE_MEMORY = "memory"
HANDLE_PREDICTIONS = "predictions"
HANDLE_TARGETS = "targets"


# ─────────────────────────────────────────────────────────────────
# SIGNATURES — comprehensive node registry
# ─────────────────────────────────────────────────────────────────

SIGNATURES: dict = {}


def register(op_type: str, inputs: dict, outputs: dict, category: NodeCategory) -> None:
    SIGNATURES[op_type] = {"inputs": inputs, "outputs": outputs, "category": category}


# IO
register("input",    {},                              {"result": "tensor"}, "io")
register("output",   {"x": {}},                      {},                    "io")

# 模型组件（__init__ 中声明）
register("conv2d",   {"in_channels": {}, "out_channels": {}}, {"result": "tensor"}, "module")
register("conv1d",   {"in_channels": {}, "out_channels": {}}, {"result": "tensor"}, "module")
register("conv3d",   {"in_channels": {}, "out_channels": {}}, {"result": "tensor"}, "module")
register("linear",   {"in_features": {}, "out_features": {}}, {"result": "tensor"}, "module")
register("embedding", {},                             {"result": "tensor"}, "module")
register("relu",     {},                              {"result": "tensor"}, "module")
register("gelu",     {},                              {"result": "tensor"}, "module")
register("silu",     {},                              {"result": "tensor"}, "module")
register("sigmoid",  {},                              {"result": "tensor"}, "module")
register("tanh",     {},                              {"result": "tensor"}, "module")
register("leakyrelu",{},                             {"result": "tensor"}, "module")
register("maxpool2d",{},                             {"result": "tensor"}, "module")
register("avgpool2d",{},                             {"result": "tensor"}, "module")
register("adaptiveavgpool2d",{},                     {"result": "tensor"}, "module")
register("globalavgpool",{},                        {"result": "tensor"}, "module")
register("batchnorm2d",{},                          {"result": "tensor"}, "module")
register("layernorm",{},                             {"result": "tensor"}, "module")
register("groupnorm",{},                             {"result": "tensor"}, "module")
register("dropout",  {},                              {"result": "tensor"}, "module")
register("softmax",  {},                              {"result": "tensor"}, "module")
register("flatten",  {},                              {"result": "tensor"}, "module")
register("selfattention",{},                        {"result": "tensor"}, "module")
register("crossattention",{"q":{},"kv":{}},         {"result": "tensor"}, "module")
register("multiheadattention",{},                   {"result": "tensor"}, "module")
register("transformerencoder",{"embed_dim": {}, "num_heads": {}, "num_layers": {}, "dim_feedforward": {}}, {"result": "tensor"}, "module")
register("transformerdecoder",{"embed_dim": {}, "num_heads": {}, "num_layers": {}, "dim_feedforward": {}}, {"result": "tensor"}, "module")
register("mamba",   {"d_model": {}},                {"result": "tensor"}, "module")
register("ffn",      {},                              {"result": "tensor"}, "module")
register("mlp",      {},                              {"result": "tensor"}, "module")
register("lstm",     {"input_size": {}, "hidden_size": {}, "num_layers": {}}, {"result": "tensor"}, "module")
register("instnorm", {"num_channels": {}},             {"result": "tensor"}, "module")
register("upsample", {"size": {}},                     {"result": "tensor"}, "operation")

# 纯 tensor 操作（forward 中直接生成代码，不在 __init__ 声明）
register("add",      {"a":{},"b":{}},                {"result": "tensor"}, "operation")
register("mul",      {"a":{},"b":{}},                {"result": "tensor"}, "operation")
register("concat",   {},                              {"result": "tensor"}, "operation")  # 多输入！
register("reshape",  {},                              {"result": "tensor"}, "operation")
register("transpose",{},                              {"result": "tensor"}, "operation")
register("split",    {},                              {"result": "tensor"}, "operation")
register("slice",    {},                              {"result": "tensor"}, "operation")

# 训练组件（只出现在训练脚本，不参与 nn.Module 定义）
register("adam",     {},                              {"result": "tensor"}, "training")
register("adamw",    {},                              {"result": "tensor"}, "training")
register("sgd",      {},                              {"result": "tensor"}, "training")
register("rmsprop",  {},                              {"result": "tensor"}, "training")
register("crossentropyloss",{},                      {"result": "tensor"}, "training")
register("mseloss",  {},                              {"result": "tensor"}, "training")
register("cosineannealinglr",{},                     {"result": "tensor"}, "training")
register("steplr",   {},                              {"result": "tensor"}, "training")
register("reducelronplateau",{},                     {"result": "tensor"}, "training")

# Evaluation 组件（不进入 forward，只在 if __name__ 中独立调用）
register("accuracy",         {"average": {}, "top_k": {}},  {"score": "scalar"}, "evaluation")
register("f1",               {"average": {}},               {"score": "scalar"}, "evaluation")
register("precision",         {"average": {}},               {"score": "scalar"}, "evaluation")
register("recall",           {"average": {}},               {"score": "scalar"}, "evaluation")
register("confusion_matrix",  {},                           {"matrix": "matrix"}, "evaluation")
register("mean_iou",          {},                           {"score": "scalar"}, "evaluation")
register("roc_auc",           {},                           {"score": "scalar"}, "evaluation")


# ─────────────────────────────────────────────────────────────────
# Module import — no more external backend.modules dependency
# All custom classes are inlined via _generate_aux_classes()
# ─────────────────────────────────────────────────────────────────

MODULE_IMPORT_MAP = {
    # torch.nn 内置模块
    "conv2d": {"source": "torch.nn", "class_name": "Conv2d"},
    "conv1d": {"source": "torch.nn", "class_name": "Conv1d"},
    "conv3d": {"source": "torch.nn", "class_name": "Conv3d"},
    "linear": {"source": "torch.nn", "class_name": "Linear"},
    "relu": {"source": "torch.nn", "class_name": "ReLU"},
    "gelu": {"source": "torch.nn", "class_name": "GELU"},
    "silu": {"source": "torch.nn", "class_name": "SiLU"},
    "sigmoid": {"source": "torch.nn", "class_name": "Sigmoid"},
    "tanh": {"source": "torch.nn", "class_name": "Tanh"},
    "leakyrelu": {"source": "torch.nn", "class_name": "LeakyReLU"},
    "maxpool2d": {"source": "torch.nn", "class_name": "MaxPool2d"},
    "avgpool2d": {"source": "torch.nn", "class_name": "AvgPool2d"},
    "adaptiveavgpool2d": {"source": "torch.nn", "class_name": "AdaptiveAvgPool2d"},
    "globalavgpool": {"source": "torch.nn", "class_name": "AdaptiveAvgPool2d"},
    "batchnorm2d": {"source": "torch.nn", "class_name": "BatchNorm2d"},
    "layernorm": {"source": "torch.nn", "class_name": "LayerNorm"},
    "groupnorm": {"source": "torch.nn", "class_name": "GroupNorm"},
    "dropout": {"source": "torch.nn", "class_name": "Dropout"},
    "softmax": {"source": "torch.nn", "class_name": "Softmax"},
    "flatten": {"source": "torch.nn", "class_name": "Flatten"},
    "embedding": {"source": "torch.nn", "class_name": "Embedding"},
    "instnorm": {"source": "torch.nn", "class_name": "InstanceNorm2d"},
    "lstm": {"source": "torch.nn", "class_name": "LSTM"},
    "multiheadattention": {"source": "torch.nn", "class_name": "MultiheadAttention"},
    "transformerencoder": {"source": "torch.nn", "class_name": "TransformerEncoder"},
    "transformerdecoder": {"source": "torch.nn", "class_name": "TransformerDecoder"},
    "selfattention": {"source": "inline", "class_name": "SelfAttention"},
    "crossattention": {"source": "inline", "class_name": "CrossAttention"},
    "mamba": {"source": "inline", "class_name": "Mamba"},
    "ffn": {"source": "inline", "class_name": "FFN"},
    "mlp": {"source": "inline", "class_name": "MLP"},
}


def collect_module_imports(sorted_blocks: list) -> set:
    """收集需要导入的自定义模块（inline 类通过辅助函数生成，无需 import）"""
    custom_modules = set()
    for block in sorted_blocks:
        if block.category != "module":
            continue
        info = MODULE_IMPORT_MAP.get(block.op_type, {})
        if info.get("source") == "inline":
            continue
        if info.get("source") and info.get("source") != "torch.nn":
            custom_modules.add(info["class_name"])
    return custom_modules


def collect_inline_classes(sorted_blocks: list) -> set:
    """收集需要生成的 inline 类（SelfAttention, CrossAttention, Mamba, MLP, FFN, DropPath）"""
    return {block.op_type for block in sorted_blocks if block.op_type in INLINE_AUX_CLASSES}


def gen_module_imports(custom_modules: set) -> str:
    """生成 import 语句"""
    lines = ["import torch", "import torch.nn as nn", "import torch.nn.functional as F"]
    if custom_modules:
        module_list = ", ".join(sorted(custom_modules))
        lines.append(f"from backend.modules import {module_list}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# NodeBlock AST
# ─────────────────────────────────────────────────────────────────

@dataclass
class NodeBlock:
    node_id: str
    op_type: str
    category: NodeCategory
    fields: dict
    # io/module: port -> "node:xxx/port"
    # concat: port -> ["node:a/result", "node:b/result"]
    inputs: dict
    output_var: str = ""
    instance_name: str = ""


# ── Graph pruning ─────────────────────────────────────────────────────────────

def _prune_graph(flow_json: dict, options: dict | None = None) -> tuple[set[str], dict]:
    """剪枝：只保留被 Output 引用且是 Input 下游的节点。
    training 节点：training_nodes=True 时保留（在 if __name__ 中生成训练代码），False 时排除。
    evaluation 节点：evaluation_nodes=True 时保留，False 时排除。"""
    opts = options or {}
    eval_enabled = opts.get("evaluation_nodes", True)
    training_enabled = opts.get("training_nodes", True)
    all_nodes = flow_json.get("nodes", [])
    all_edges = flow_json.get("edges", [])

    # O(1) node lookup by id
    node_by_id = {n["id"]: n for n in all_nodes}

    def _node_category(n):
        node_type = normalize_node_type(n.get("data", {}).get("nodeType", ""))
        return SIGNATURES.get(node_type, {}).get("category", "")

    # 排除 training 类节点（当 training_nodes=False 时）；排除 evaluation 类节点（当 feature 关闭时）
    non_training = {
        n["id"] for n in all_nodes
        if not (_node_category(n) == "training" and not training_enabled)
        and not (_node_category(n) == "evaluation" and not eval_enabled)
    }

    # 一次遍历同时构建反向和正向邻接表
    output_ids: set[str] = set()
    input_ids: set[str] = set()
    rev: dict[str, list[str]] = defaultdict(list)
    fwd: dict[str, list[str]] = defaultdict(list)
    for e in all_edges:
        rev[e["target"]].append(e["source"])
        fwd[e["source"]].append(e["target"])
        src_node = node_by_id.get(e["source"], {})
        tgt_node = node_by_id.get(e["target"], {})
        src_type = normalize_node_type(src_node.get("data", {}).get("nodeType", ""))
        tgt_type = normalize_node_type(tgt_node.get("data", {}).get("nodeType", ""))
        if tgt_type == "output":
            output_ids.add(e["target"])
        if src_type == "input":
            input_ids.add(e["source"])

    # 扩展 input_ids / output_ids 到所有孤立但存在的节点
    for n in all_nodes:
        nt = normalize_node_type(n.get("data", {}).get("nodeType", ""))
        if nt == "input":
            input_ids.add(n["id"])
        if nt == "output":
            output_ids.add(n["id"])

    # 反向 BFS：从 Output 出发
    reachable_from_output: set = set()
    queue = deque(list(output_ids & non_training))
    while queue:
        nid = queue.popleft()
        if nid in reachable_from_output:
            continue
        reachable_from_output.add(nid)
        for src in rev.get(nid, []):
            if src in non_training and src not in reachable_from_output:
                queue.append(src)

    # 正向 BFS：从 Input 出发
    reachable_from_input: set = set()
    queue = deque(list(input_ids & non_training))
    while queue:
        nid = queue.popleft()
        if nid in reachable_from_input:
            continue
        reachable_from_input.add(nid)
        for tgt in fwd.get(nid, []):
            if tgt in non_training and tgt not in reachable_from_input:
                queue.append(tgt)

    # 取差集 = 既是 Input 下游，又能通到 Output
    valid = (reachable_from_output & reachable_from_input) | (output_ids & non_training) | (input_ids & non_training)

    # 补充 evaluation 节点：连接到任意有效节点的 evaluation 节点也要包含
    for e in all_edges:
        src_valid = e["source"] in valid
        tgt_valid = e["target"] in valid
        src_cat = _node_category(node_by_id.get(e["source"], {}))
        tgt_cat = _node_category(node_by_id.get(e["target"], {}))
        if src_valid and tgt_cat == "evaluation":
            valid.add(e["target"])
        if tgt_valid and src_cat == "evaluation":
            valid.add(e["source"])

    # 补充 training 节点：即使不连接也保留（用于 if __name__ 训练代码）
    if training_enabled:
        for n in all_nodes:
            if _node_category(n) == "training":
                valid.add(n["id"])

    pruned_nodes = [n for n in all_nodes if n["id"] in valid]
    pruned_edges = [e for e in all_edges if e["source"] in valid and e["target"] in valid]
    return valid, {"nodes": pruned_nodes, "edges": pruned_edges}


# ── AST build ─────────────────────────────────────────────────────────────────

def build_ast(flow_json: dict, options: dict | None = None) -> list[NodeBlock]:
    _, pruned = _prune_graph(flow_json, options)
    block_map: dict[str, NodeBlock] = {}

    for rf in pruned.get("nodes", []):
        nt = normalize_node_type(rf.get("data", {}).get("nodeType", ""))
        sig = SIGNATURES.get(nt, {})
        cat = sig.get("category", "module")
        fields = rf.get("data", {}).get("params", {})

        if nt == "concat":
            inputs: dict = {}
        else:
            inputs = {}

        block_map[rf["id"]] = NodeBlock(
            node_id=rf["id"],
            op_type=nt,
            category=cat,
            fields=fields,
            inputs=inputs,
        )

    # 处理边
    for edge in pruned.get("edges", []):
        src_id = edge["source"]
        tgt_id = edge["target"]
        tgt = block_map.get(tgt_id)
        if not tgt:
            continue

        src_handle = edge.get("sourceHandle", "result")
        tgt_handle = edge.get("targetHandle", "a")
        ref = f"node:{src_id}/{src_handle}"

        if tgt.op_type == "concat":
            if tgt_handle not in tgt.inputs:
                tgt.inputs[tgt_handle] = []
            tgt.inputs[tgt_handle].append(ref)
        else:
            if tgt.op_type == "output":
                if "x" not in tgt.inputs or tgt.inputs["x"] is None:
                    tgt.inputs["x"] = ref
            elif tgt_handle in tgt.inputs and tgt.inputs[tgt_handle] is not None and tgt.inputs[tgt_handle] != ref:
                concat_id = f"_icat_{tgt_id}_{tgt_handle}"
                concat_ref = f"node:{concat_id}/result"
                if concat_id not in block_map:
                    block_map[concat_id] = NodeBlock(
                        node_id=concat_id,
                        op_type="concat",
                        category="operation",
                        fields={"dim": 1},
                        inputs={"in_0": [tgt.inputs[tgt_handle]], "in_1": [ref]},
                    )
                tgt.inputs[tgt_handle] = concat_ref
            else:
                tgt.inputs[tgt_handle] = ref

    # 拓扑排序（Kahn）
    in_deg: dict[str, int] = {nid: 0 for nid in block_map}
    adj: dict[str, list[str]] = {nid: [] for nid in block_map}

    for block in block_map.values():
        if block.op_type == "concat":
            all_sources: set = set()
            for src_list in block.inputs.values():
                for r in src_list:
                    src_node = r.split(":")[1].split("/")[0]
                    all_sources.add(src_node)
            for s in all_sources:
                if s in in_deg:
                    adj[s].append(block.node_id)
                    in_deg[block.node_id] += 1
        else:
            for ref in block.inputs.values():
                if ref:
                    src_node = ref.split(":")[1].split("/")[0]
                    if src_node in in_deg:
                        in_deg[block.node_id] += 1
                        adj[src_node].append(block.node_id)

    queue = deque([nid for nid, d in in_deg.items() if d == 0])
    sorted_blocks: list[NodeBlock] = []
    while queue:
        nid = queue.popleft()
        sorted_blocks.append(block_map[nid])
        for nb in adj[nid]:
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                queue.append(nb)

    # 分配输出变量名 + 实例命名：x_{module_type}_{index}
    used: set = set()
    name_token_counter: dict = defaultdict(int)
    for block in sorted_blocks:
        name_token = get_block_name_token(block)
        name_token_counter[name_token] += 1
        readable_name = f"{name_token}_{name_token_counter[name_token]}"

        block.instance_name = f"x_{readable_name}"
        block.output_var = f"x_{readable_name}"

        suffix = 1
        while block.output_var in used:
            suffix += 1
            block.instance_name = f"x_{readable_name}_{suffix}"
            block.output_var = f"x_{readable_name}_{suffix}"
        used.add(block.output_var)

    return sorted_blocks


def _resolve(ref, all_blocks):
    if not ref:
        return "x"
    if isinstance(ref, list):
        if ref:
            ref = ref[0]
        else:
            return "x"
    src_id = ref.split(":")[1].split("/")[0]
    for b in all_blocks:
        if b.node_id == src_id:
            return b.output_var
    return "x"


# ── Code generation ───────────────────────────────────────────────────────────


def _gen_init(block: NodeBlock) -> Optional[str]:
    if block.category != "module":
        return None
    f = block.fields
    name = block.output_var

    if block.op_type == "conv2d":
        return "self." + name + " = nn.Conv2d(in_channels=" + str(f.get("in_channels", 0)) + ", out_channels=" + str(f.get("out_channels", 0)) + ", kernel_size=" + str(f.get("kernel_size", 3)) + ", stride=" + str(f.get("stride", 1)) + ", padding=" + str(f.get("padding", 0)) + ", bias=" + str(f.get("bias", False)) + ")"
    if block.op_type == "conv1d":
        return "self." + name + " = nn.Conv1d(in_channels=" + str(f.get("in_channels", 0)) + ", out_channels=" + str(f.get("out_channels", 0)) + ", kernel_size=" + str(f.get("kernel_size", 3)) + ", stride=" + str(f.get("stride", 1)) + ", padding=" + str(f.get("padding", 0)) + ", bias=" + str(f.get("bias", False)) + ")"
    if block.op_type == "conv3d":
        return "self." + name + " = nn.Conv3d(in_channels=" + str(f.get("in_channels", 0)) + ", out_channels=" + str(f.get("out_channels", 0)) + ", kernel_size=" + str(f.get("kernel_size", 3)) + ", stride=" + str(f.get("stride", 1)) + ", padding=" + str(f.get("padding", 0)) + ", bias=" + str(f.get("bias", False)) + ")"
    if block.op_type == "linear":
        return "self." + name + " = nn.Linear(in_features=" + str(f.get("in_features", 0)) + ", out_features=" + str(f.get("out_features", 0)) + ", bias=" + str(f.get("bias", True)) + ")"
    if block.op_type == "relu":
        return "self." + name + " = nn.ReLU()"
    if block.op_type == "gelu":
        return "self." + name + " = nn.GELU()"
    if block.op_type == "silu":
        return "self." + name + " = nn.SiLU()"
    if block.op_type == "sigmoid":
        return "self." + name + " = nn.Sigmoid()"
    if block.op_type == "tanh":
        return "self." + name + " = nn.Tanh()"
    if block.op_type == "leakyrelu":
        return "self." + name + " = nn.LeakyReLU(negative_slope=" + str(f.get("negative_slope", 0.01)) + ")"
    if block.op_type == "maxpool2d":
        return "self." + name + " = nn.MaxPool2d(kernel_size=" + str(f.get("kernel_size", 2)) + ", stride=" + str(f.get("stride", 2)) + ", padding=" + str(f.get("padding", 0)) + ")"
    if block.op_type == "avgpool2d":
        return "self." + name + " = nn.AvgPool2d(kernel_size=" + str(f.get("kernel_size", 2)) + ", stride=" + str(f.get("stride", 2)) + ", padding=" + str(f.get("padding", 0)) + ")"
    if block.op_type == "adaptiveavgpool2d":
        return "self." + name + " = nn.AdaptiveAvgPool2d(output_size=" + str(f.get("output_size", 1)) + ")"
    if block.op_type == "globalavgpool":
        return "self." + name + " = nn.AdaptiveAvgPool2d(1)"
    if block.op_type == "batchnorm2d":
        return "self." + name + " = nn.BatchNorm2d(num_features=" + str(f.get("num_features", 0)) + ")"
    if block.op_type == "layernorm":
        return "self." + name + " = nn.LayerNorm(normalized_shape=" + str(f.get("normalized_shape", 64)) + ")"
    if block.op_type == "groupnorm":
        return "self." + name + " = nn.GroupNorm(num_groups=" + str(f.get("num_groups", 1)) + ", num_channels=" + str(f.get("num_channels", 0)) + ")"
    if block.op_type == "dropout":
        return "self." + name + " = nn.Dropout(p=" + str(f.get("p", 0.5)) + ", inplace=True)"
    if block.op_type == "softmax":
        return "self." + name + " = nn.Softmax(dim=" + str(f.get("dim", -1)) + ")"
    if block.op_type == "flatten":
        return "self." + name + " = nn.Flatten(start_dim=" + str(f.get("start_dim", 1)) + ")"
    if block.op_type == "embedding":
        return "self." + name + " = nn.Embedding(num_embeddings=" + str(f.get("num_embeddings", 0)) + ", embedding_dim=" + str(f.get("embedding_dim", 0)) + ")"
    if block.op_type == "selfattention":
        return "self." + name + " = SelfAttention(dim=" + str(f.get("embed_dim", 512)) + ", heads=" + str(f.get("num_heads", 8)) + ")"
    if block.op_type == "crossattention":
        return "self." + name + " = CrossAttention(dim=" + str(f.get("embed_dim", 512)) + ", heads=" + str(f.get("num_heads", 8)) + ")"
    if block.op_type == "multiheadattention":
        return "self." + name + " = nn.MultiheadAttention(embed_dim=" + str(f.get("embed_dim", 512)) + ", num_heads=" + str(f.get("num_heads", 8)) + ", dropout=" + str(f.get("dropout", 0)) + ", batch_first=True)"
    if block.op_type == "ffn":
        d = str(f.get("dim", 512))
        hd = str(f.get("hidden_dim", 2048))
        return "self." + name + " = FFN(dim=" + d + ", hidden_dim=" + hd + ")"
    if block.op_type == "mlp":
        inf = str(f.get("in_features", 784))
        hdf = str(f.get("hidden_features", 256))
        outf = str(f.get("out_features", 10))
        depth = str(f.get("depth", 2))
        return "self." + name + " = MLP(dim=" + inf + ", hidden_dim=" + hdf + ", depth=" + depth + ", out_dim=" + outf + ")"
    if block.op_type == "transformerencoder":
        d = str(f.get("embed_dim", f.get("d_model", 512))); nh = str(f.get("num_heads", f.get("nhead", 8)))
        dl = str(f.get("num_layers", 6)); dim_ff = str(f.get("dim_feedforward", 2048))
        return "self." + name + " = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=" + d + ", nhead=" + nh + ", dim_feedforward=" + dim_ff + ", batch_first=True), num_layers=" + dl + ")"
    if block.op_type == "transformerdecoder":
        d = str(f.get("embed_dim", f.get("d_model", 512))); nh = str(f.get("num_heads", f.get("nhead", 8)))
        dl = str(f.get("num_layers", 6)); dim_ff = str(f.get("dim_feedforward", 2048))
        return "self." + name + " = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=" + d + ", nhead=" + nh + ", dim_feedforward=" + dim_ff + ", batch_first=True), num_layers=" + dl + ")"
    if block.op_type == "mamba":
        # 修复：完整的 Mamba 参数（从 code_gen_v2.py 移植）
        d = str(f.get("d_model", 512))
        n_layers = str(f.get("n_layers", 1))
        d_state = str(f.get("d_state", 16))
        d_conv = str(f.get("d_conv", 4))
        expand = str(f.get("expand", 2))
        dt_rank = str(f.get("dt_rank", "auto"))
        dropout = str(f.get("dropout", 0.0))
        return ("self." + name + " = Mamba(d_model=" + d + ", d_state=" + d_state + ", d_conv=" + d_conv
                + ", expand=" + expand + ", dt_rank=" + dt_rank + ", dropout=" + dropout + ", n_layers=" + n_layers + ")")
    if block.op_type == "lstm":
        inp = str(f.get("input_size", 512))
        hid = str(f.get("hidden_size", 512))
        lay = str(f.get("num_layers", 2))
        bidirectional = "True" if f.get("bidirectional", False) else "False"
        return ("self." + name + " = nn.LSTM(input_size=" + inp + ", hidden_size=" + hid
                + ", num_layers=" + lay + ", batch_first=True, bidirectional=" + bidirectional + ")")
    if block.op_type == "instnorm":
        nc = str(f.get("num_channels", 64))
        return "self." + name + " = nn.InstanceNorm2d(num_features=" + nc + ")"
    return None


def _gen_forward(block: NodeBlock, all_blocks: list) -> Optional[str]:
    vid = block.output_var

    if block.op_type == "input":
        return "        " + vid + " = x"

    if block.op_type == "output":
        src = _resolve(block.inputs.get("x"), all_blocks)
        return "        " + block.output_var + " = " + src + "\n        return " + block.output_var

    if block.category == "operation":
        if block.op_type == "add":
            a = _resolve(block.inputs.get("a"), all_blocks)
            b = _resolve(block.inputs.get("b"), all_blocks)
            return "        " + vid + " = " + a + " + " + b
        if block.op_type == "mul":
            a = _resolve(block.inputs.get("a"), all_blocks)
            b = _resolve(block.inputs.get("b"), all_blocks)
            return "        " + vid + " = " + a + " * " + b
        if block.op_type == "concat":
            all_srcs = []
            for src_list in block.inputs.values():
                if isinstance(src_list, list):
                    for r in src_list:
                        v = _resolve(r, all_blocks)
                        if v not in all_srcs:
                            all_srcs.append(v)
            if len(all_srcs) >= 2:
                dim = str(block.fields.get("dim", 1))
                return "        " + vid + " = torch.cat([" + ", ".join(all_srcs) + "], dim=" + dim + ")"
            if len(all_srcs) == 1:
                return "        " + vid + " = " + all_srcs[0]
            return None
        if block.op_type in ("reshape", "transpose", "split", "slice", "upsample"):
            first_ref = None
            for v in block.inputs.values():
                if v:
                    first_ref = v[0] if isinstance(v, list) else v
                    break
            up = _resolve(first_ref, all_blocks)
            if block.op_type == "reshape":
                # 修复：使用 .reshape() 替代 .view()（.reshape() 更安全，非连续时自动拷贝）
                shape = block.fields.get("shape", -1)
                return "        " + vid + " = " + up + ".reshape(" + up + ".size(0), " + str(shape) + ")"
            if block.op_type == "transpose":
                return "        " + vid + " = " + up + ".transpose(" + str(block.fields.get("dim0", 0)) + ", " + str(block.fields.get("dim1", 1)) + ")"
            if block.op_type == "split":
                return "        " + vid + " = " + up + ".split(" + str(block.fields.get("split_size", 32)) + ", dim=" + str(block.fields.get("dim", 0)) + ")"
            if block.op_type == "slice":
                s, e, st = str(block.fields.get("start", 0)), str(block.fields.get("end", -1)), str(block.fields.get("step", 1))
                return "        " + vid + " = " + up + "[" + s + ":" + e + ":" + st + "]"
            if block.op_type == "upsample":
                size = block.fields.get("size", None)
                scale_factor = block.fields.get("scale_factor", 2.0)
                mode = block.fields.get("mode", "bilinear")
                if size:
                    return "        " + vid + " = torch.nn.functional.interpolate(" + up + ", size=" + str(size) + ", mode='" + mode + "', align_corners=False)"
                return "        " + vid + " = torch.nn.functional.interpolate(" + up + ", scale_factor=" + str(scale_factor) + ", mode='" + mode + "', align_corners=False)"
            return None
        return None

    if block.category == "module":
        first_ref = None
        for v in block.inputs.values():
            if v:
                first_ref = v[0] if isinstance(v, list) else v
                break
        up = _resolve(first_ref, all_blocks)
        if block.op_type == "selfattention":
            return "        " + vid + " = self." + vid + "(" + up + ")"
        if block.op_type == "crossattention":
            # crossattention 输入：q 和 kv（k=v 来自同一源）
            q = _resolve(block.inputs.get("q"), all_blocks)
            kv = _resolve(block.inputs.get("kv"), all_blocks)
            return "        " + vid + " = self." + vid + "(" + q + ", " + kv + ", " + kv + ")"
        if block.op_type == "multiheadattention":
            q = _resolve(block.inputs.get("q"), all_blocks) if block.inputs.get("q") else up
            k = _resolve(block.inputs.get("k"), all_blocks) if block.inputs.get("k") else up
            v = _resolve(block.inputs.get("v"), all_blocks) if block.inputs.get("v") else up
            return "        " + vid + " = self." + vid + "(" + q + ", " + k + ", " + v + ")"
        if block.op_type == "transformerencoder":
            src = _resolve(block.inputs.get("src"), all_blocks) if block.inputs.get("src") else up
            return "        " + vid + " = self." + vid + "(" + src + ")"
        if block.op_type == "transformerdecoder":
            tgt = _resolve(block.inputs.get("tgt"), all_blocks) if block.inputs.get("tgt") else up
            memory = _resolve(block.inputs.get("memory"), all_blocks) if block.inputs.get("memory") else tgt
            return "        " + vid + " = self." + vid + "(" + tgt + ", " + memory + ")"
        if block.op_type == "lstm":
            # 修复：正确解包 multi-layer / bidirectional LSTM 的输出
            return "        " + vid + ", (hidden, cell) = self." + vid + "(" + up + ")"
        if block.op_type == "reshape":
            # 修复：使用 .reshape() 替代 .view()
            shape = block.fields.get("shape", -1)
            return "        " + vid + " = " + up + ".reshape(" + up + ".size(0), " + str(shape) + ")"
        if block.op_type == "transpose":
            return "        " + vid + " = " + up + ".transpose(" + str(block.fields.get("dim0", 0)) + ", " + str(block.fields.get("dim1", 1)) + ")"
        if block.op_type == "flatten":
            return "        " + vid + " = " + up + ".flatten(start_dim=" + str(block.fields.get("start_dim", 1)) + ")"
        # 统一：self.{instance_name}(x)，如 self.conv2d_1(x)
        return "        " + vid + " = self." + vid + "(" + up + ")"

    return None


def _gen_training(block: NodeBlock, all_blocks: list) -> list[str]:
    """Generate training code for a single training block (goes in if __name__)."""
    lines: list[str] = []
    f = block.fields
    op = block.op_type
    name = block.output_var

    if op == "adam":
        lr = f.get("lr", 0.001)
        betas = f.get("betas", (0.9, 0.999))
        eps = f.get("eps", 1e-8)
        weight_decay = f.get("weight_decay", 0.0)
        lines.append(f"    # Adam optimizer: {name}")
        lines.append(f"    optimizer_{name} = torch.optim.Adam(model.parameters(), lr={lr}, betas={betas}, eps={eps}, weight_decay={weight_decay})")
    elif op == "adamw":
        lr = f.get("lr", 0.001)
        betas = f.get("betas", (0.9, 0.999))
        eps = f.get("eps", 1e-8)
        weight_decay = f.get("weight_decay", 0.01)
        lines.append(f"    # AdamW optimizer: {name}")
        lines.append(f"    optimizer_{name} = torch.optim.AdamW(model.parameters(), lr={lr}, betas={betas}, eps={eps}, weight_decay={weight_decay})")
    elif op == "sgd":
        lr = f.get("lr", 0.01)
        momentum = f.get("momentum", 0.9)
        weight_decay = f.get("weight_decay", 0.0)
        nesterov = f.get("nesterov", True)
        lines.append(f"    # SGD optimizer: {name}")
        lines.append(f"    optimizer_{name} = torch.optim.SGD(model.parameters(), lr={lr}, momentum={momentum}, weight_decay={weight_decay}, nesterov={nesterov})")
    elif op == "rmsprop":
        lr = f.get("lr", 0.01)
        alpha = f.get("alpha", 0.99)
        eps = f.get("eps", 1e-8)
        weight_decay = f.get("weight_decay", 0.0)
        momentum = f.get("momentum", 0.0)
        lines.append(f"    # RMSprop optimizer: {name}")
        lines.append(f"    optimizer_{name} = torch.optim.RMSprop(model.parameters(), lr={lr}, alpha={alpha}, eps={eps}, weight_decay={weight_decay}, momentum={momentum})")
    elif op == "crossentropyloss":
        weight = f.get("weight", None)
        label_smoothing = f.get("label_smoothing", 0.0)
        lines.append(f"    # CrossEntropyLoss: {name}")
        if weight:
            lines.append(f"    criterion_{name} = nn.CrossEntropyLoss(weight={weight}, label_smoothing={label_smoothing})")
        else:
            lines.append(f"    criterion_{name} = nn.CrossEntropyLoss(label_smoothing={label_smoothing})")
    elif op == "mseloss":
        reduction = f.get("reduction", "mean")
        lines.append(f"    # MSELoss: {name}")
        lines.append(f"    criterion_{name} = nn.MSELoss(reduction='{reduction}')")
    elif op == "cosineannealinglr":
        T_max = f.get("T_max", 10)
        eta_min = f.get("eta_min", 0.0)
        lines.append(f"    # CosineAnnealingLR scheduler: {name}")
        lines.append(f"    scheduler_{name} = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_{name}, T_max={T_max}, eta_min={eta_min})")
    elif op == "steplr":
        step_size = f.get("step_size", 5)
        gamma = f.get("gamma", 0.1)
        lines.append(f"    # StepLR scheduler: {name}")
        lines.append(f"    scheduler_{name} = torch.optim.lr_scheduler.StepLR(optimizer_{name}, step_size={step_size}, gamma={gamma})")
    elif op == "reducelronplateau":
        mode = f.get("mode", "min")
        factor = f.get("factor", 0.1)
        patience = f.get("patience", 3)
        threshold = f.get("threshold", 1e-4)
        lines.append(f"    # ReduceLROnPlateau scheduler: {name}")
        lines.append(f"    scheduler_{name} = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_{name}, mode='{mode}', factor={factor}, patience={patience}, threshold={threshold})")

    return lines


def _gen_evaluation(block: NodeBlock, all_blocks: list) -> list:
    """Generate evaluation code for a single evaluation block."""
    lines: list[str] = []
    f = block.fields
    op = block.op_type
    average = f.get("average", "macro")
    top_k = f.get("top_k", 1)

    pred_ref = block.inputs.get("predictions")
    tgt_ref = block.inputs.get("targets")

    def resolve_var(ref) -> str:
        if not ref:
            return "y_pred"
        if isinstance(ref, list):
            ref = ref[0] if ref else None
        if not ref:
            return "y_pred"
        src_id = ref.split(":")[1].split("/")[0]
        for b in all_blocks:
            if b.node_id == src_id:
                return b.output_var
        return "y_pred"

    pred_var = resolve_var(pred_ref) if pred_ref else "y_pred"
    tgt_var = resolve_var(tgt_ref) if tgt_ref else "y_true"

    if op == "accuracy":
        lines.append("    # Accuracy evaluation")
        lines.append("    from sklearn.metrics import accuracy_score")
        lines.append(f"    y_pred_np = {pred_var}.argmax(dim=1).numpy() if {pred_var}.dim() > 1 else {pred_var}.numpy()")
        lines.append(f"    y_true_np = {tgt_var}.numpy() if hasattr({tgt_var}, 'numpy') else {tgt_var}")
        if top_k > 1:
            lines.append(f"    from sklearn.metrics import top_k_accuracy_score")
            lines.append(f"    acc = top_k_accuracy_score(y_true_np, {pred_var}.numpy(), k={top_k})")
        else:
            lines.append(f"    acc = accuracy_score(y_true_np, y_pred_np)")
        lines.append(f'    print(f"Accuracy: {{acc:.4f}}")')
    elif op == "f1":
        lines.append("    # F1 Score evaluation")
        lines.append("    from sklearn.metrics import f1_score")
        lines.append(f"    y_pred_np = {pred_var}.argmax(dim=1).numpy() if {pred_var}.dim() > 1 else {pred_var}.numpy()")
        lines.append(f"    y_true_np = {tgt_var}.numpy() if hasattr({tgt_var}, 'numpy') else {tgt_var}")
        lines.append(f"    f1 = f1_score(y_true_np, y_pred_np, average='{average}')")
        lines.append(f'    print(f"F1 Score ({average}): {{f1:.4f}}")')
    elif op == "precision":
        lines.append("    # Precision evaluation")
        lines.append("    from sklearn.metrics import precision_score")
        lines.append(f"    y_pred_np = {pred_var}.argmax(dim=1).numpy() if {pred_var}.dim() > 1 else {pred_var}.numpy()")
        lines.append(f"    y_true_np = {tgt_var}.numpy() if hasattr({tgt_var}, 'numpy') else {tgt_var}")
        lines.append(f"    prec = precision_score(y_true_np, y_pred_np, average='{average}')")
        lines.append(f'    print(f"Precision ({average}): {{prec:.4f}}")')
    elif op == "recall":
        lines.append("    # Recall evaluation")
        lines.append("    from sklearn.metrics import recall_score")
        lines.append(f"    y_pred_np = {pred_var}.argmax(dim=1).numpy() if {pred_var}.dim() > 1 else {pred_var}.numpy()")
        lines.append(f"    y_true_np = {tgt_var}.numpy() if hasattr({tgt_var}, 'numpy') else {tgt_var}")
        lines.append(f"    rec = recall_score(y_true_np, y_pred_np, average='{average}')")
        lines.append(f'    print(f"Recall ({average}): {{rec:.4f}}")')
    elif op == "confusion_matrix":
        lines.append("    # Confusion Matrix evaluation")
        lines.append("    from sklearn.metrics import confusion_matrix")
        lines.append(f"    y_pred_np = {pred_var}.argmax(dim=1).numpy() if {pred_var}.dim() > 1 else {pred_var}.numpy()")
        lines.append(f"    y_true_np = {tgt_var}.numpy() if hasattr({tgt_var}, 'numpy') else {tgt_var}")
        lines.append(f"    cm = confusion_matrix(y_true_np, y_pred_np)")
        lines.append(f'    print(f"Confusion Matrix:\\n{{cm}}")')
    elif op == "mean_iou":
        lines.append("    # Mean IoU evaluation (semantic segmentation)")
        lines.append("    try:")
        lines.append("        from torchmetrics import MeanIoU")
        lines.append(f"        miou = MeanIoU(num_classes={f.get('num_classes', 10)})")
        lines.append(f"        y_pred_labels = {pred_var}.argmax(dim=1)")
        lines.append(f"        y_true_labels = {tgt_var}")
        lines.append(f"        iou = miou(y_pred_labels, y_true_labels.long())")
        lines.append(f'        print(f"Mean IoU: {{iou:.4f}}")')
        lines.append("    except Exception as e:")
        lines.append(f'        print(f"Mean IoU computation failed: {{e}}")')
    elif op == "roc_auc":
        lines.append("    # ROC AUC evaluation")
        lines.append("    from sklearn.metrics import roc_auc_score")
        lines.append(f"    y_pred_prob = {pred_var}.softmax(dim=1).numpy() if {pred_var}.dim() > 1 else {pred_var}.numpy()")
        lines.append(f"    y_true_np = {tgt_var}.numpy() if hasattr({tgt_var}, 'numpy') else {tgt_var}")
        lines.append(f"    auc = roc_auc_score(y_true_np, y_pred_prob, average='{average}', multi_class='ovr')")
        lines.append(f'    print(f"ROC AUC ({average}): {{auc:.4f}}")')

    return lines


def _py_repr(value):
    if isinstance(value, str):
        if value in ("True", "False", "None"):
            return value
        return repr(value)
    if value is True:
        return "True"
    if value is False:
        return "False"
    if value is None:
        return "None"
    return repr(value)


# ─────────────────────────────────────────────────────────────────
# Loss 函数映射表（与 ast_core.py 保持一致）
# ─────────────────────────────────────────────────────────────────

LOSS_CLASS_MAP = {
    "cross_entropy": "nn.CrossEntropyLoss()",
    "mse": "nn.MSELoss()",
    "bce": "nn.BCEWithLogitsLoss()",
    "bce_logits": "nn.BCEWithLogitsLoss()",
    "dice": "DiceLoss()",
    "focal": "FocalLoss()",
    "lovasz": "LovaszLoss()",
    "tversky": "TverskyLoss()",
    "iou": "IoULoss()",
    "giou": "GIoULoss()",
    "dice_ce": "DiceCELoss()",
    "msssim": "MS_SSIMLoss()",
    "perceptual": "PerceptualLoss()",
    "content": "ContentLoss()",
    "style": "StyleLoss()",
    "smooth_l1": "nn.SmoothL1Loss()",
    "focal_loss": "FocalLoss()",
    "class_balanced": "ClassBalancedLoss()",
    "stft": "STFTLoss()",
    "sdr": "SDRLoss()",
    "sisdr": "SISDRLoss()",
    "mel_spec": "MelSpectrogramLoss()",
    "waveform": "WaveformMSELoss()",
    "multi_res": "MultiResolutionSTFTLoss()",
    "phase": "PhaseLoss()",
    "label_smoothing": "nn.LabelSmoothingLoss()",
    "contrastive": "ContrastiveLoss()",
}

LOSS_CUSTOM_CLASSES = {
    "dice": '''
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0,3,1,2).float()
        intersection = (pred * target_one_hot).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))
        iou = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()
''',
    "focal": '''
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
''',
    "lovasz": '''
class LovaszLoss(nn.Module):
    def forward(self, pred, target):
        return F.cross_entropy(F.softmax(pred, dim=1), target)
''',
    "tversky": '''
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0,3,1,2).float()
        tp = (pred * target_one_hot).sum(dim=(2,3))
        fp = (pred * (1 - target_one_hot)).sum(dim=(2,3))
        fn = ((1 - pred) * target_one_hot).sum(dim=(2,3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()
''',
    "iou": '''
class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0,3,1,2).float()
        intersection = (pred * target_one_hot).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()
''',
    "giou": '''
class GIoULoss(nn.Module):
    def forward(self, pred, target):
        return F.cross_entropy(F.softmax(pred, dim=1), target)
''',
    "dice_ce": '''
class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce = nn.CrossEntropyLoss()
        self.smooth = 1e-6
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0,3,1,2).float()
        intersection = (pred * target_one_hot).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        ce = self.ce(pred, target)
        return self.dice_weight * (1 - dice.mean()) + self.ce_weight * ce
''',
}


def _gen_training_from_config(config: dict | None) -> tuple[list[str], list[str]]:
    """
    Generate training code from training config.
    Returns a tuple of (custom_class_lines, training_lines).
    """
    if not config:
        return [], []

    custom_class_lines: list[str] = []
    lines: list[str] = []
    loss_cfg = config.get("loss", {}) or {}
    optimizer_cfg = config.get("optimizer", {}) or {}
    scheduler_cfg = config.get("scheduler", {}) or {}
    runtime_cfg = config.get("runtime", {}) or {}
    task_type = config.get("taskType", "classification")

    loss_type = loss_cfg.get("type", "cross_entropy")

    if loss_cfg.get("type") == "composite":
        components = loss_cfg.get("params", {}).get("components", [])
        for i, comp in enumerate(components):
            comp_type = comp.get("type", "cross_entropy")
            comp_class = LOSS_CLASS_MAP.get(comp_type, "nn.CrossEntropyLoss()")
            if comp_type in LOSS_CUSTOM_CLASSES:
                custom_class_lines.append(LOSS_CUSTOM_CLASSES[comp_type])
            lines.append(f"    loss_fn_{i} = {comp_class}")
        loss_exprs = []
        for i, comp in enumerate(components):
            weight = comp.get("weight", 1.0)
            loss_exprs.append(f"{weight} * loss_fn_{i}(primary_output, target)")
        lines.append(f"    loss = {' + '.join(loss_exprs)}")
        lines.append("    target = torch.randint(0, max(2, primary_output.shape[-1] if primary_output.dim() > 1 else 2), (primary_output.shape[0],), dtype=torch.long, device=primary_output.device)")
    elif loss_cfg.get("enabled", True):
        if loss_type in LOSS_CUSTOM_CLASSES:
            custom_class_lines.append(LOSS_CUSTOM_CLASSES[loss_type])
        loss_class = LOSS_CLASS_MAP.get(loss_type, "nn.CrossEntropyLoss()")
        lines.append(f"    loss_fn = {loss_class}")
        if loss_type in ("mse",):
            lines.append("    target = torch.randn_like(primary_output)")
        else:
            lines.append("    target = torch.randint(0, max(2, primary_output.shape[-1] if primary_output.dim() > 1 else 2), (primary_output.shape[0],), dtype=torch.long, device=primary_output.device)")

    if optimizer_cfg.get("enabled", True):
        optimizer_type = optimizer_cfg.get("type", "adamw")
        optimizer_params = optimizer_cfg.get("params", {}) or {}
        param_str = ", ".join(f"{key}={_py_repr(value)}" for key, value in optimizer_params.items())
        suffix = f", {param_str}" if param_str else ""
        if optimizer_type == "adam":
            lines.append(f"    optimizer = torch.optim.Adam(model.parameters(){suffix})")
        elif optimizer_type == "sgd":
            lines.append(f"    optimizer = torch.optim.SGD(model.parameters(){suffix})")
        elif optimizer_type == "rmsprop":
            lines.append(f"    optimizer = torch.optim.RMSprop(model.parameters(){suffix})")
        else:
            lines.append(f"    optimizer = torch.optim.AdamW(model.parameters(){suffix})")

    if scheduler_cfg.get("enabled"):
        scheduler_type = scheduler_cfg.get("type", "cosine_annealing")
        scheduler_params = scheduler_cfg.get("params", {}) or {}
        param_str = ", ".join(f"{key}={_py_repr(value)}" for key, value in scheduler_params.items())
        prefix = "optimizer"
        args = f"{prefix}, {param_str}" if param_str else prefix
        if scheduler_type in ("step_lr", "steplr"):
            lines.append(f"    scheduler = torch.optim.lr_scheduler.StepLR({args})")
        elif scheduler_type in ("reduce_on_plateau", "reducelronplateau"):
            lines.append(f"    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau({args})")
        else:
            lines.append(f"    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR({args})")

    if loss_cfg.get("enabled", True) and optimizer_cfg.get("enabled", True):
        lines.append("    model.train()")
        lines.append("    optimizer.zero_grad()")
        if loss_cfg.get("type") != "composite":
            lines.append("    loss = loss_fn(primary_output, target)")
        lines.append("    loss.backward()")
        grad_clip = runtime_cfg.get("gradClip")
        if grad_clip is not None:
            lines.append(f"    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm={grad_clip})")
        lines.append("    optimizer.step()")
        if scheduler_cfg.get("enabled"):
            if scheduler_cfg.get("type") in ("reduce_on_plateau", "reducelronplateau"):
                lines.append("    scheduler.step(loss)")
            else:
                lines.append("    scheduler.step()")
        lines.append(
            f"    print(f\"task={task_type}, epochs={runtime_cfg.get('epochs', 10)}, batch_size={runtime_cfg.get('batchSize', 32)}, device={{device}}\")"
        )

    return custom_class_lines, lines


# ─────────────────────────────────────────────────────────────────
# UnifiedCodeGenerator — single entry point
# ─────────────────────────────────────────────────────────────────

class UnifiedCodeGenerator:
    """
    统一代码生成器。

    generate_model() — 仅生成 nn.Module 类（用于 training_executor）
    generate_full()  — 生成完整文件含训练/评估代码（用于 /api/generate）
    """

    def __init__(self, graph: dict, options: dict | None = None):
        self.graph = graph
        self.options = options or {}
        self.sorted_blocks = build_ast(graph, self.options)
        self._init_lines: list[str] | None = None
        self._fwd_lines: list[str] | None = None
        self._inline_classes: str | None = None

    def _ensure_build(self):
        if self._init_lines is None:
            self._init_lines, self._fwd_lines = self._build_code_sections()
            self._inline_classes = self._collect_inline_classes()

    def _build_code_sections(self) -> tuple[list[str], list[str]]:
        init_lines: list[str] = []
        fwd_lines: list[str] = []

        for block in self.sorted_blocks:
            if block.category in ("training", "evaluation"):
                continue
            il = _gen_init(block)
            if il:
                init_lines.append(f"        {il}")
            fl = _gen_forward(block, self.sorted_blocks)
            if fl:
                fwd_lines.append(fl)

        return init_lines, fwd_lines

    def _collect_inline_classes(self) -> str:
        needed = collect_inline_classes(self.sorted_blocks)
        return _generate_aux_classes(needed)

    def _format_forward_block(self) -> tuple[str, str]:
        """Extract multi-output handling into one place. Returns (fwd_block, init_block)."""
        output_return_lines = []
        non_output_fwd_lines = []
        for line in self._fwd_lines:
            for single_line in line.split("\n"):
                stripped = single_line.strip()
                if stripped.startswith("return "):
                    output_return_lines.append(stripped)
                else:
                    non_output_fwd_lines.append(single_line)

        if len(output_return_lines) > 1:
            output_vars = [ln.replace("return ", "").strip() for ln in output_return_lines]
            final_return = "        return (" + ", ".join(output_vars) + ")"
        elif len(output_return_lines) == 1:
            final_return = "        " + output_return_lines[0]
        else:
            final_return = "        return x"

        fwd_block = "\n".join(non_output_fwd_lines) + "\n" + final_return if non_output_fwd_lines else final_return
        init_block = "\n".join(self._init_lines) if self._init_lines else "        pass"
        return fwd_block, init_block

    def generate_model(self) -> str:
        """
        生成仅包含 nn.Module 的代码（用于嵌入 train.py）。
        """
        self._ensure_build()
        fwd_block, init_block = self._format_forward_block()

        return f"""# Generated by FlowHamster
# DO NOT EDIT — Regenerated from graph editor
import torch
import torch.nn as nn
import torch.nn.functional as F

{self._inline_classes}


class FlowHamsterModel(nn.Module):
    def __init__(self):
        super().__init__()
{init_block}

    def forward(self, x):
{fwd_block}
"""

    def generate_full(self) -> str:
        """
        生成完整文件代码（模型 + 训练 + 评估），用于 /api/generate。
        """
        self._ensure_build()
        opts = self.options
        eval_enabled = opts.get("evaluation_nodes", True)
        training_config = opts.get("training_config")
        fwd_block, init_block = self._format_forward_block()

        # Evaluation code
        eval_main_lines: list[str] = []
        if eval_enabled:
            model_blocks_for_eval = [b for b in self.sorted_blocks if b.category not in ("training", "evaluation")]
            for block in self.sorted_blocks:
                if block.category != "evaluation":
                    continue
                lines = _gen_evaluation(block, model_blocks_for_eval)
                eval_main_lines.extend(lines)

        # Training code
        training_main_lines: list[str] = []
        custom_class_lines: list[str] = []
        training_enabled = opts.get("training_nodes", False)
        if training_enabled:
            training_blocks_for_main = [b for b in self.sorted_blocks if b.category == "training"]
            optimizer_vars = [
                f"optimizer_{block.output_var}"
                for block in training_blocks_for_main
                if block.op_type in ("adam", "adamw", "sgd", "rmsprop")
            ]
            if any(block.op_type in ("cosineannealinglr", "steplr", "reducelronplateau") for block in training_blocks_for_main) and not optimizer_vars:
                training_main_lines.append("    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)")
                optimizer_vars.append("optimizer")
            scheduler_optimizer = optimizer_vars[0] if optimizer_vars else "optimizer"
            for block in training_blocks_for_main:
                lines = _gen_training(block, self.sorted_blocks)
                if block.op_type in ("cosineannealinglr", "steplr", "reducelronplateau"):
                    lines = [line.replace(f"optimizer_{block.output_var}", scheduler_optimizer) for line in lines]
                training_main_lines.extend(lines)
        elif training_config:
            custom_class_lines, training_lines = _gen_training_from_config(training_config)
            training_main_lines.extend(training_lines)

        needs_targets = len([b for b in self.sorted_blocks if b.category == "evaluation"]) > 0
        targets_placeholder = "    # Placeholder labels (replace with real dataset labels)\n    y_true = torch.randint(0, 10, (1,))" if needs_targets else ""

        eval_block = ("\n" + "\n".join(eval_main_lines)) if eval_main_lines else ""
        training_block = ("\n" + "\n".join(training_main_lines)) if training_main_lines else ""
        custom_class_block = ("\n" + "\n".join(custom_class_lines)) if custom_class_lines else ""

        # Determine multi-output flag by checking fwd_block return pattern
        is_multi = "return (" in fwd_block and fwd_block.strip().endswith(")")
        primary_output_line = "    primary_output = output[0]" if is_multi else "    primary_output = output"
        device_name = training_config.get("runtime", {}).get("device", "cpu") if isinstance(training_config, dict) else "cpu"

        return f"""# Generated by FlowHamster
# DO NOT EDIT — Regenerated from graph editor
import torch
import torch.nn as nn
import torch.nn.functional as F

{self._inline_classes}
{custom_class_block}

class FlowHamsterModel(nn.Module):
    def __init__(self):
        super().__init__()
{init_block}

    def forward(self, x):
{fwd_block}


if __name__ == "__main__":
    model = FlowHamsterModel()
    device = "{device_name}"
    model = model.to(device if device != "auto" else "cpu")
    x = torch.randn(1, 3, 224, 224, device=device if device != "auto" else "cpu")
    output = model(x)
{primary_output_line}
    if isinstance(output, tuple):
        print(f"output: {{[tuple(t.shape) if hasattr(t, 'shape') else type(t).__name__ for t in output]}}")
    else:
        print(f"output: {{output.shape}}")
{targets_placeholder}{training_block}{eval_block}
"""


# ─────────────────────────────────────────────────────────────────
# Backward-compatible functional API (used by generate.py router)
# ─────────────────────────────────────────────────────────────────

def generate(flow_json: dict, options: dict | None = None) -> str:
    """
    向后兼容函数 — 直接调用 UnifiedCodeGenerator.generate_full()。
    保留给 /api/generate 端点使用。
    """
    gen = UnifiedCodeGenerator(flow_json, options)
    return gen.generate_full()
