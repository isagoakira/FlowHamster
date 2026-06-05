"""Generated runtime helper classes and import collection."""
from __future__ import annotations

from .metadata import INLINE_AUX_CLASSES

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


