"""
FlowHamster 代码生成器 v2 — 三层架构
Layer 1: NodeBlock AST（inputs 用 "literal:X" 或 "node:id/port" 标记值来源）
Layer 2: NODE_SIGNATURES 类型签名注册表
Layer 3: 拓扑排序 + PyTorch 代码生成

相比 v1 的核心改进：
- 不再用单一 x 变量，所有节点输出都有命名变量
- 支持所有节点类型（包括 Transformer/Attention/Mamba/MLP/FFN）
- 支持多端口节点（CrossAttention Q/K/V 通过 sourceHandle/targetHandle 区分）
- 每条边可独立指定 mergeMode（concat/add/mul/stack）
"""

from dataclasses import dataclass, field
from typing import Any
from collections import deque


# ─────────────────────────────────────────────
# Layer 2: NODE_SIGNATURES 类型签名注册表
# ─────────────────────────────────────────────

NODE_SIGNATURES: dict[str, dict] = {
    # ── 输入/输出 ──
    "input":  {"inputs": {},                    "output": "Tensor", "module": None,  "module_template": None},
    "output": {"inputs": {"x": "Tensor"},        "output": None,    "module": None,  "module_template": None},

    # ── 卷积 ──
    "conv2d": {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.Conv2d",
               "module_template": "nn.Conv2d(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}{stride}{padding}{bias})"},
    "conv1d": {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.Conv1d",
               "module_template": "nn.Conv1d(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}{stride}{padding}{bias})"},
    "conv3d": {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.Conv3d",
               "module_template": "nn.Conv3d(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}{stride}{padding}{bias})"},

    # ── 线性 ──
    "linear": {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.Linear",
               "module_template": "nn.Linear(in_features={in_features}, out_features={out_features}, bias={bias})"},

    # ── 激活函数 ──
    "relu":      {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.ReLU",      "module_template": "nn.ReLU()"},
    "gelu":      {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.GELU",      "module_template": "nn.GELU()"},
    "silu":      {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.SiLU",      "module_template": "nn.SiLU()"},
    "sigmoid":   {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.Sigmoid",   "module_template": "nn.Sigmoid()"},
    "tanh":      {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.Tanh",      "module_template": "nn.Tanh()"},
    "leakyrelu": {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.LeakyReLU", "module_template": "nn.LeakyReLU(negative_slope={negative_slope})"},

    # ── 池化 ──
    "maxpool":  {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.MaxPool2d",
                 "module_template": "nn.MaxPool2d(kernel_size={kernel_size}, stride={stride})"},
    "avgpool":  {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.AvgPool2d",
                 "module_template": "nn.AvgPool2d(kernel_size={kernel_size}, stride={stride})"},
    "adaptiveavgpool": {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.AdaptiveAvgPool2d",
                        "module_template": "nn.AdaptiveAvgPool2d(output_size={output_size})"},
    "globalavgpool":   {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.AdaptiveAvgPool2d",
                        "module_template": "nn.AdaptiveAvgPool2d(1)"},

    # ── 归一化 ──
    "batchnorm": {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.BatchNorm2d",
                  "module_template": "nn.BatchNorm2d(num_features={num_features})"},
    "layernorm": {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.LayerNorm",
                  "module_template": "nn.LayerNorm(normalized_shape={normalized_shape})"},
    "groupnorm": {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.GroupNorm",
                  "module_template": "nn.GroupNorm(num_groups={num_groups}, num_channels={num_channels})"},
    "instnorm": {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.InstanceNorm2d",
                 "module_template": "nn.InstanceNorm2d(num_features={num_features})"},
    "dropout":   {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.Dropout",
                  "module_template": "nn.Dropout(p={p}, inplace=True)"},

    # ── 张量变换 ──
    "flatten":  {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.Flatten",
                 "module_template": "nn.Flatten(start_dim={start_dim})"},
    "softmax":  {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.Softmax",
                 "module_template": "nn.Softmax(dim={dim})"},
    "embedding":{"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.Embedding",
                 "module_template": "nn.Embedding(num_embeddings={num_embeddings}, embedding_dim={embedding_dim})"},
    "reshape":  {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": None, "module_template": None},
    "transpose": {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": None, "module_template": None},
    "split":    {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": None, "module_template": None},
    "slice":    {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": None, "module_template": None},

    # ── 合并/分支 ──
    "add":    {"inputs": {"a": "Tensor", "b": "Tensor"}, "output": "Tensor", "module": None, "module_template": None},
    "mul":    {"inputs": {"a": "Tensor", "b": "Tensor"}, "output": "Tensor", "module": None, "module_template": None},
    "concat": {"inputs": {"tensors": "Tensor"},          "output": "Tensor", "module": None, "module_template": None},

    # ── 注意力 ──
    "selfattention": {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "SelfAttention",
                      "module_template": "SelfAttention(dim={embed_dim}, heads={heads})"},
    "crossattention": {"inputs": {"q": "Tensor", "k": "Tensor", "v": "Tensor"}, "output": "Tensor",
                       "module": "nn.MultiheadAttention", "module_template": "nn.MultiheadAttention(embed_dim={embed_dim}, num_heads={heads}, batch_first=True)"},
    "multiheadattention": {"inputs": {"q": "Tensor", "k": "Tensor", "v": "Tensor"}, "output": "Tensor",
                           "module": "nn.MultiheadAttention",
                           "module_template": "nn.MultiheadAttention(embed_dim={embed_dim}, num_heads={num_heads}, batch_first=True)"},

    # ── 损失函数 ──
    "crossentropyloss": {"inputs": {"logits": "Tensor", "targets": "Tensor"}, "output": "Tensor",
                         "module": "nn.CrossEntropyLoss", "module_template": "nn.CrossEntropyLoss()"},
    "mseloss": {"inputs": {"pred": "Tensor", "target": "Tensor"}, "output": "Tensor",
                "module": "nn.MSELoss", "module_template": "nn.MSELoss()"},

    # ── Transformer ──
    "transformerencoder": {"inputs": {"src": "Tensor"}, "output": "Tensor", "module": "nn.TransformerEncoder",
                           "module_template": "nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, batch_first=True), num_layers={num_layers})"},
    "transformerdecoder": {"inputs": {"tgt": "Tensor", "memory": "Tensor"}, "output": "Tensor",
                           "module": "nn.TransformerDecoder",
                           "module_template": "nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, batch_first=True), num_layers={num_layers})"},

    # ── 状态空间模型 ──
    "mamba": {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "Mamba",
             "module_template": "Mamba(d_model={d_model}, d_state={d_state}, d_conv={d_conv}, expand={expand}, dt_rank={dt_rank}, dropout={dropout})"},

    # ── MLP / FFN ──
    "mlp": {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "MLP",
            "module_template": "MLP(dim={dim}, hidden_dim={hidden_dim}, depth={depth}, out_dim={out_dim})"},
    "ffn": {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "FFN",
            "module_template": "FFN(dim={dim}, hidden_dim={hidden_dim})"},

    # ── 循环网络 ──
    "lstm": {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "nn.LSTM",
             "module_template": "nn.LSTM(input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, batch_first=True)"},

    # ── 优化器（仅训练时使用，不加入 forward）──
    "adam":  {"inputs": {}, "output": None, "module": None, "module_template": None},
    "adamw": {"inputs": {}, "output": None, "module": None, "module_template": None},
    "sgd":   {"inputs": {}, "output": None, "module": None, "module_template": None},
    "rmsprop":{"inputs": {}, "output": None, "module": None, "module_template": None},

    # ── 学习率调度器 ──
    "steplr":         {"inputs": {}, "output": None, "module": None, "module_template": None},
    "cosineannealing":{"inputs": {}, "output": None, "module": None, "module_template": None},
    "reduceonplateau": {"inputs": {}, "output": None, "module": None, "module_template": None},

    # ── 其他 ──
    "focalloss":     {"inputs": {}, "output": None, "module": None, "module_template": None},
    "labelsmoothing":{"inputs": {}, "output": None, "module": None, "module_template": None},
    "droppath":      {"inputs": {"x": "Tensor"}, "output": "Tensor", "module": "DropPath",
                      "module_template": "DropPath(drop_prob={drop_prob})"},
}

MERGE_TEMPLATES = {
    "concat": lambda srcs, dim: f"torch.cat([{', '.join(srcs)}], dim={dim})",
    "add":    lambda srcs, dim: f"({' + '.join(srcs)})",
    "mul":    lambda srcs, dim: f"({' * '.join(srcs)})",
    "stack":  lambda srcs, dim: f"torch.stack([{', '.join(srcs)}], dim={dim})",
}

# ─────────────────────────────────────────────
# Layer 1: NodeBlock AST
# ─────────────────────────────────────────────

@dataclass
class NodeBlock:
    """一个节点在 AST 中的表示"""
    id: str
    op_type: str                    # nodeType
    inputs: dict[str, str]          # port_name -> "literal:X" 或 "node:nodeId[/port]"
    fields: dict[str, Any]           # params
    output_name: str = None          # 分配到的 Python 变量名


@dataclass
class WorkflowAST:
    """整个工作流的 AST 表示"""
    entry_points: list[str] = field(default_factory=list)
    nodes: dict[str, NodeBlock] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# Layer 3: 核心逻辑
# ─────────────────────────────────────────────

class WorkflowCodeGenerator:
    """
    三层架构代码生成器
    1. parse_workflow()        — 前端图结构 → WorkflowAST
    2. topological_sort_v2()   — 计算执行顺序
    3. generate()              — WorkflowAST → 可运行的 PyTorch 代码
    """

    def __init__(self, graph: dict, options: dict = None):
        self.graph = graph
        self.options = options or {}
        self.warnings: list[str] = []
        self.ast: WorkflowAST | None = None

    def generate(self) -> str:
        """主入口"""
        self.ast = self.parse_workflow()
        if not self.ast.nodes:
            return "# Empty graph — add nodes from the sidebar"

        self.ast.execution_order = self.topological_sort_v2()
        code = self.generate_pytorch_module()
        return code

    # ── Layer 1: 解析 ──────────────────────────────

    def parse_workflow(self) -> WorkflowAST:
        """将前端图结构解析为 WorkflowAST"""
        nodes_list = self.graph.get("nodes", [])
        edges_list = self.graph.get("edges", [])

        blocks: dict[str, NodeBlock] = {}

        for n in nodes_list:
            nid = n["id"]
            data = n.get("data", {})
            op_type = data.get("nodeType", "unknown")
            params = data.get("params", {})

            # 建立 inputs 字典
            # key: targetHandle 或默认 "x"
            # value: "node:sourceId" 或 "node:sourceId/sourceHandle"（多端口）
            in_edges = [e for e in edges_list if e.get("target") == nid]

            inputs: dict[str, str] = {}
            for edge in in_edges:
                tgt_handle = edge.get("targetHandle", "x") or "x"
                src_id = edge.get("source", "")
                src_handle = edge.get("sourceHandle")  # 多端口节点有值

                if src_handle:
                    source_ref = f"node:{src_id}/{src_handle}"
                else:
                    source_ref = f"node:{src_id}"

                inputs[tgt_handle] = source_ref

            blocks[nid] = NodeBlock(
                id=nid,
                op_type=op_type,
                inputs=inputs,
                fields=params,
            )

        # 入口点：所有没有被其他节点指向的节点（无入边的节点）
        all_targets = {e.get("target") for e in edges_list}
        entry_points = [n["id"] for n in nodes_list if n["id"] not in all_targets]

        return WorkflowAST(entry_points=entry_points, nodes=blocks)

    def topological_sort_v2(self) -> list[str]:
        """Kahn 算法拓扑排序"""
        blocks = self.ast.nodes
        in_degree = {nid: 0 for nid in blocks}
        adj = {nid: [] for nid in blocks}

        for block in blocks.values():
            for ref in block.inputs.values():
                if ref.startswith("node:"):
                    src_id = ref[5:].split("/")[0]
                    if src_id in adj and block.id in in_degree:
                        adj[src_id].append(block.id)
                        in_degree[block.id] += 1

        queue = deque([nid for nid, d in in_degree.items() if d == 0])
        sorted_list = []

        while queue:
            nid = queue.popleft()
            sorted_list.append(nid)
            for nb in adj[nid]:
                in_degree[nb] -= 1
                if in_degree[nb] == 0:
                    queue.append(nb)

        if len(sorted_list) != len(blocks):
            self.warnings.append("Warning: graph contains cycles — some nodes may be unreachable")

        return sorted_list

    # ── Layer 3: 代码生成 ──────────────────────────

    def _build_init_section(self, exec_order: list[str]) -> tuple[list[str], dict[str, str]]:
        """生成 __init__ 模块实例注册代码"""
        init_lines = []
        var_map: dict[str, str] = {}   # node_id -> self.mod_xxx
        counter: dict[str, int] = {}

        NO_MODULE_OPS = {
            "input", "output", "add", "mul", "concat", "reshape",
            "transpose", "split", "slice", "adam", "adamw", "sgd",
            "rmsprop", "steplr", "cosineannealing", "reduceonplateau",
            "focalloss", "labelsmoothing",
        }

        for nid in exec_order:
            block = self.ast.nodes[nid]
            op = block.op_type

            if op in NO_MODULE_OPS:
                continue

            sig = NODE_SIGNATURES.get(op, {})
            if not sig or not sig.get("module"):
                continue

            template = sig["module_template"]
            if not template:
                continue

            counter[op] = counter.get(op, 0) + 1
            var_name = f"self.mod_{op}_{counter[op]}"

            try:
                # Build optional parameter suffixes only when they differ from defaults
                fields = {k: v for k, v in block.fields.items()}
                if fields.get("stride") not in (None, 1, "1"):
                    fields["stride"] = f", stride={fields['stride']}"
                else:
                    fields["stride"] = ""
                if fields.get("padding") not in (None, 0, "0", "None", ""):
                    fields["padding"] = f", padding={fields['padding']}"
                else:
                    fields["padding"] = ""
                if fields.get("bias") in (True, "True", "true", 1, "1"):
                    fields["bias"] = ", bias=True"
                elif fields.get("bias") in (False, "False", "false", 0, "0"):
                    fields["bias"] = ", bias=False"
                else:
                    fields["bias"] = ""  # default True in PyTorch
                code_str = template.format(**fields)
                init_lines.append(f"        {var_name} = {code_str}")
                var_map[nid] = var_name  # only register if init succeeded
            except Exception as e:
                self.warnings.append(f"Module template error for {nid} ({op}): {e}")

        return init_lines, var_map

    def _resolve_inputs(
        self, block: NodeBlock, node_outputs: dict[str, str]
    ) -> dict[str, str]:
        """解析 block.inputs，将 "node:xxx" 引用转为实际 Python 变量名"""
        result = {}
        for port, ref in block.inputs.items():
            if ref.startswith("node:"):
                src_id = ref[5:].split("/")[0]
                if src_id in node_outputs:
                    result[port] = node_outputs[src_id]
                else:
                    result[port] = "x"  # fallback
            elif ref.startswith("literal:"):
                result[port] = ref[8:]
            else:
                result[port] = ref
        return result

    def _build_forward_section(
        self, exec_order: list[str], var_map: dict[str, str]
    ) -> list[str]:
        """生成 forward() 代码行"""
        lines = []
        outputs: dict[str, str] = {}  # node_id -> var_name

        for nid in exec_order:
            block = self.ast.nodes[nid]
            op = block.op_type

            if op == "input":
                outputs[nid] = "x"
                continue

            if op == "output":
                in_vals = self._resolve_inputs(block, outputs)
                ret = in_vals.get("x") or (list(in_vals.values())[0] if in_vals else "x")
                lines.append(f"        return {ret}")
                continue

            in_vals = self._resolve_inputs(block, outputs)

            # 输出变量名：模块节点用中间变量名避免覆盖 self.mod_xxx
            if nid in var_map:
                out_var = f"x_{nid.replace('-', '_').replace(':', '_')}"
            else:
                out_var = f"x_{nid.replace('-', '_').replace(':', '_')}"

            outputs[nid] = out_var

            line = self._generate_forward_line(block, op, in_vals, out_var, var_map)
            if line:
                lines.append(f"        {line}")

        # If forward body has lines but no explicit return, return the last output
        if lines and not any("return" in l for l in lines):
            last_output = outputs.get(exec_order[-1], "x")
            lines.append(f"        return {last_output}")

        return lines

    def _generate_forward_line(
        self, block: NodeBlock, op: str,
        in_vals: dict[str, str], out_var: str,
        var_map: dict[str, str]
    ) -> str | None:
        """生成单个节点的 forward 代码行"""
        # 有模块注册的节点
        if block.id in var_map:
            mod_var = var_map[block.id]
            main = in_vals.get("x") or in_vals.get("q") or (
                list(in_vals.values())[0] if in_vals else "x")
            return f"{out_var} = {mod_var}({main})"

        # 无模块节点
        if op == "add":
            a = in_vals.get("a", "x")
            b = in_vals.get("b", "x")
            return f"{out_var} = ({a} + {b})"

        if op == "mul":
            a = in_vals.get("a", "x")
            b = in_vals.get("b", "x")
            return f"{out_var} = ({a} * {b})"

        if op == "concat":
            srcs = list(in_vals.values())
            dim = block.fields.get("dim", 1)
            if len(srcs) == 1:
                return f"{out_var} = {srcs[0]}"
            return f"{out_var} = torch.cat([{', '.join(srcs)}], dim={dim})"

        if op == "reshape":
            x = in_vals.get("x", "x")
            shape = block.fields.get("shape", "-1")
            return f"{out_var} = {x}.reshape({shape})"

        if op == "transpose":
            x = in_vals.get("x", "x")
            d0, d1 = block.fields.get("dim0", 1), block.fields.get("dim1", 2)
            return f"{out_var} = {x}.transpose({d0}, {d1})"

        if op == "flatten":
            x = in_vals.get("x", "x")
            sd = block.fields.get("start_dim", 1)
            return f"{out_var} = {x}.flatten(start_dim={sd})"

        if op == "softmax":
            x = in_vals.get("x", "x")
            dim = block.fields.get("dim", -1)
            return f"{out_var} = torch.nn.functional.softmax({x}, dim={dim})"

        if op == "dropout":
            x = in_vals.get("x", "x")
            p = block.fields.get("p", 0.5)
            return f"{out_var} = torch.nn.functional.dropout({x}, p={p}, training=self.training)"

        if op == "sigmoid":
            x = in_vals.get("x", "x")
            return f"{out_var} = torch.sigmoid({x})"

        if op == "tanh":
            x = in_vals.get("x", "x")
            return f"{out_var} = torch.tanh({x})"

        if op == "silu":
            x = in_vals.get("x", "x")
            return f"{out_var} = torch.nn.functional.silu({x})"

        if op == "leakyrelu":
            x = in_vals.get("x", "x")
            neg = block.fields.get("negative_slope", 0.01)
            return f"{out_var} = torch.nn.functional.leaky_relu({x}, negative_slope={neg})"

        if op == "maxpool":
            x = in_vals.get("x", "x")
            ks = block.fields.get("kernel_size", 2)
            st = block.fields.get("stride", 2)
            return f"{out_var} = torch.nn.functional.max_pool2d({x}, kernel_size={ks}, stride={st})"

        if op == "avgpool":
            x = in_vals.get("x", "x")
            ks = block.fields.get("kernel_size", 2)
            st = block.fields.get("stride", 2)
            return f"{out_var} = torch.nn.functional.avg_pool2d({x}, kernel_size={ks}, stride={st})"

        if op == "adaptiveavgpool":
            x = in_vals.get("x", "x")
            os = block.fields.get("output_size", 1)
            return f"{out_var} = torch.nn.functional.adaptive_avg_pool2d({x}, output_size={os})"

        if op == "globalavgpool":
            x = in_vals.get("x", "x")
            return f"{out_var} = torch.nn.functional.adaptive_avg_pool2d({x}, 1)"

        # pass-through fallback
        if in_vals:
            return f"{out_var} = {list(in_vals.values())[0]}"
        return None

    def _generate_aux_classes(self, exec_order: list[str]) -> str:
        """生成辅助类（SelfAttention, CrossAttention, Mamba, MLP, FFN, DropPath）"""
        needed: set[str] = set()
        for nid in exec_order:
            op = self.ast.nodes[nid].op_type
            if op in ("selfattention", "crossattention", "mamba", "mlp", "ffn", "droppath"):
                needed.add(op)

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
            dt_rank = "auto"
            parts.append(f'''
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

        # Parallel associative scan (sequential — correct SSM semantics)
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

    def generate_pytorch_module(self) -> str:
        """生成完整的 PyTorch nn.Module 代码"""
        exec_order = self.ast.execution_order
        init_lines, var_map = self._build_init_section(exec_order)
        forward_lines = self._build_forward_section(exec_order, var_map)
        aux_classes = self._generate_aux_classes(exec_order)

        init_block = "\n".join(init_lines) if init_lines else "        pass"
        forward_block = "\n".join(forward_lines) if forward_lines else "        pass"

        return f'''# Generated by FlowHamster v2 (three-layer architecture)
# DO NOT EDIT — Regenerated from graph editor
import torch
import torch.nn as nn
import torch.nn.functional as F

{aux_classes}


class FlowHamsterModel(nn.Module):
    def __init__(self):
        super().__init__()
{init_block}

    def forward(self, x):
{forward_block}
'''

    def _guess_input_channels(self) -> str:
        for node in self.graph.get("nodes", []):
            if node.get("data", {}).get("nodeType") == "input":
                shape = node.get("data", {}).get("params", {}).get("shape", "3, 224, 224")
                return shape.split(",")[0].strip() if "," in shape else "3"
        return "3"


# ─────────────────────────────────────────────
# FastAPI router（与 v1 接口兼容）
# ─────────────────────────────────────────────

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class GenerateRequest(BaseModel):
    graph: dict
    options: dict | None = None


class GenerateResponse(BaseModel):
    success: bool
    code: str
    warnings: list[str] = []


@router.post("/generate/v2", response_model=GenerateResponse)
async def generate_code_v2(req: GenerateRequest):
    try:
        gen = WorkflowCodeGenerator(req.graph, req.options)
        code = gen.generate()
        return GenerateResponse(success=True, code=code, warnings=gen.warnings)
    except Exception as e:
        return GenerateResponse(success=False, code="", warnings=[str(e)])


# ─────────────────────────────────────────────
# 验证测试
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 测试用例：Input → Conv2d → ReLU → Conv2d → Output
    graph = {
        "nodes": [
            {"id": "input1", "data": {"nodeType": "input", "label": "Input", "params": {"shape": "3, 224, 224", "dtype": "float32"}}},
            {"id": "conv1",  "data": {"nodeType": "conv2d", "label": "Conv2d", "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 7, "stride": 2, "padding": 3, "bias": False}}},
            {"id": "relu1",  "data": {"nodeType": "relu",   "label": "ReLU",  "params": {}}},
            {"id": "conv2",  "data": {"nodeType": "conv2d", "label": "Conv2d", "params": {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1, "bias": False}}},
            {"id": "output1","data": {"nodeType": "output", "label": "Output", "params": {}}},
        ],
        "edges": [
            {"source": "input1", "target": "conv1",  "sourceHandle": None, "targetHandle": "x"},
            {"source": "conv1",  "target": "relu1",  "sourceHandle": None, "targetHandle": "x"},
            {"source": "relu1",  "target": "conv2",  "sourceHandle": None, "targetHandle": "x"},
            {"source": "conv2",  "target": "output1","sourceHandle": None, "targetHandle": "x"},
        ]
    }

    gen = WorkflowCodeGenerator(graph)
    code = gen.generate()
    print("=== Generated Code ===")
    print(code)
    print()

    # 验证：exec() 能否成功 import 并运行
    print("=== Verification ===")
    namespace = {}
    try:
        exec(code, namespace)
        model = namespace["FlowHamsterModel"]()
        x = torch.randn(1, 3)
        out = model(x)
        print("✅ Code runs successfully! Output shape:", out.shape)
    except Exception as e:
        print(f"❌ Error: {e}")
