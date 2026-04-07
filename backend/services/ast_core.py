"""FlowHamster 代码生成器 v2"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict, deque

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

# ─────────────────────────────────────────────────────────────────
# 模块 import 注册表
# 区分 torch.nn 内置模块和需要额外 import 的模块
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
    "selfattention": {"source": "torch.nn", "class_name": "MultiheadAttention"},
    "crossattention": {"source": "torch.nn", "class_name": "MultiheadAttention"},
    # 需要自定义实现的模块
    "mamba": {"source": "backend.modules", "class_name": "Mamba"},
    # ffn 和 mlp 使用 nn.Sequential 组合，不需要额外 import
    "ffn": {"source": None, "class_name": ""},
    "mlp": {"source": None, "class_name": ""},
}


def collect_module_imports(sorted_blocks: list) -> set:
    """收集需要导入的自定义模块"""
    custom_modules = set()
    for block in sorted_blocks:
        if block.category != "module":
            continue
        info = MODULE_IMPORT_MAP.get(block.op_type, {})
        if info.get("source") == "backend.modules":
            custom_modules.add(info["class_name"])
    return custom_modules


def gen_module_imports(custom_modules: set) -> str:
    """生成 import 语句"""
    lines = ["import torch", "import torch.nn as nn"]
    if custom_modules:
        module_list = ", ".join(sorted(custom_modules))
        lines.append(f"from backend.modules import {module_list}")
    return "\n".join(lines)


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
    # 实例命名：{op_type}_{counter}，如 conv2d_1, conv2d_2, linear_1
    instance_name: str = ""


# ── SIGNATURES ────────────────────────────────────────────────────────────────

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
    node_ids = {n["id"] for n in all_nodes}

    def _node_category(n):
        node_type = normalize_node_type(n.get("data", {}).get("nodeType", ""))
        return SIGNATURES.get(node_type, {}).get("category", "")

    # 排除 training 类节点（当 training_nodes=False 时）；排除 evaluation 类节点（当 feature 关闭时）
    non_training = {
        n["id"] for n in all_nodes
        if not (_node_category(n) == "training" and not training_enabled)
        and not (_node_category(n) == "evaluation" and not eval_enabled)
    }

    # 反向 BFS：从 Output 出发
    output_ids = {n["id"] for n in all_nodes
                  if normalize_node_type(n.get("data", {}).get("nodeType", "")) == "output"}
    rev: dict[str, list[str]] = defaultdict(list)
    for e in all_edges:
        rev[e["target"]].append(e["source"])

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
    input_ids = {n["id"] for n in all_nodes
                 if normalize_node_type(n.get("data", {}).get("nodeType", "")) == "input"}
    fwd: dict[str, list[str]] = defaultdict(list)
    for e in all_edges:
        fwd[e["source"]].append(e["target"])

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
        src_cat = _node_category(next((n for n in all_nodes if n["id"] == e["source"]), {}))
        tgt_cat = _node_category(next((n for n in all_nodes if n["id"] == e["target"]), {}))
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

        # Concat 节点：输入是 port -> [sources_list]
        # 其他节点：输入是 port -> source_ref
        if nt == "concat":
            inputs: dict = {}  # port -> list
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
            # Concat: 累积多边到同一端口列表
            if tgt_handle not in tgt.inputs:
                tgt.inputs[tgt_handle] = []
            tgt.inputs[tgt_handle].append(ref)
        else:
            # 其他节点：同一端口多条边 → 自动插入隐式 concat 合并
            if tgt.op_type == "output":
                # output 节点：同一端口只接受第一条边
                if "x" not in tgt.inputs or tgt.inputs["x"] is None:
                    tgt.inputs["x"] = ref
                # 后续边忽略（output 只能有一个输入）
            elif tgt_handle in tgt.inputs and tgt.inputs[tgt_handle] is not None and tgt.inputs[tgt_handle] != ref:
                # 端口已被占用，插入隐式 concat
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
            # concat 的入度 = 所有来源节点数（每个 source 一个入度）
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
        # concat 多输入，取第一个（实际应该都返回同一个变量的多个引用）
        if ref:
            ref = ref[0]
        else:
            return "x"
    src_id = ref.split(":")[1].split("/")[0]
    for b in all_blocks:
        if b.node_id == src_id:
            return b.output_var
    return "x"


# ── Code generation ────────────────────────────────────────────────────────────


def _gen_init(block: NodeBlock) -> Optional[str]:
    if block.category != "module":
        return None
    f = block.fields
    # 实例命名：{op_type}_{counter}，如 conv2d_1, linear_2
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
        return "self." + name + " = nn.MultiheadAttention(embed_dim=" + str(f.get("embed_dim", 512)) + ", num_heads=" + str(f.get("num_heads", 8)) + ", dropout=" + str(f.get("dropout", 0)) + ", batch_first=True)"
    if block.op_type == "crossattention":
        return "self." + name + " = nn.MultiheadAttention(embed_dim=" + str(f.get("query_dim", f.get("embed_dim", 512))) + ", num_heads=" + str(f.get("num_heads", 8)) + ", dropout=" + str(f.get("dropout", 0)) + ", batch_first=True)"
    if block.op_type == "multiheadattention":
        return "self." + name + " = nn.MultiheadAttention(embed_dim=" + str(f.get("embed_dim", 512)) + ", num_heads=" + str(f.get("num_heads", 8)) + ", dropout=" + str(f.get("dropout", 0)) + ", batch_first=True)"
    if block.op_type == "ffn":
        d = str(f.get("dim", 512))
        hd = str(f.get("hidden_dim", 2048))
        dp = str(f.get("dropout", 0))
        return "self." + name + " = nn.Sequential(nn.Linear(" + d + ", " + hd + "), nn.GELU(), nn.Dropout(" + dp + "), nn.Linear(" + hd + ", " + d + "))"
    if block.op_type == "mlp":
        inf = str(f.get("in_features", 784))
        hdf = str(f.get("hidden_features", 256))
        outf = str(f.get("out_features", 10))
        return "self." + name + " = nn.Sequential(nn.Linear(" + inf + ", " + hdf + "), nn.ReLU(), nn.Linear(" + hdf + ", " + outf + "))"
    if block.op_type == "transformerencoder":
        d = str(f.get("embed_dim", f.get("d_model", 512))); nh = str(f.get("num_heads", f.get("nhead", 8)))
        dl = str(f.get("num_layers", 6)); dim_ff = str(f.get("dim_feedforward", 2048))
        return "self." + name + " = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=" + d + ", nhead=" + nh + ", dim_feedforward=" + dim_ff + ", batch_first=True), num_layers=" + dl + ")"
    if block.op_type == "transformerdecoder":
        d = str(f.get("embed_dim", f.get("d_model", 512))); nh = str(f.get("num_heads", f.get("nhead", 8)))
        dl = str(f.get("num_layers", 6)); dim_ff = str(f.get("dim_feedforward", 2048))
        return "self." + name + " = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=" + d + ", nhead=" + nh + ", dim_feedforward=" + dim_ff + ", batch_first=True), num_layers=" + dl + ")"
    if block.op_type == "mamba":
        # 使用从 backend.modules 导入的 Mamba 类
        d = str(f.get("d_model", 512))
        return "self." + name + " = Mamba(d_model=" + d + ")"
    if block.op_type == "lstm":
        inp = str(f.get("input_size", 512))
        hid = str(f.get("hidden_size", 512))
        lay = str(f.get("num_layers", 2))
        return "self." + name + " = nn.LSTM(input_size=" + inp + ", hidden_size=" + hid + ", num_layers=" + lay + ", batch_first=True)"
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
                shape = block.fields.get("shape", -1)
                return "        " + vid + " = " + up + ".view(" + up + ".size(0), " + str(shape) + ")"
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
            return "        " + vid + ", _ = self." + vid + "(" + up + ", " + up + ", " + up + ")"
        if block.op_type == "crossattention":
            q = _resolve(block.inputs.get("q"), all_blocks)
            kv = _resolve(block.inputs.get("kv"), all_blocks)
            return "        " + vid + ", _ = self." + vid + "(" + q + ", " + kv + ", " + kv + ")"
        if block.op_type == "multiheadattention":
            q = _resolve(block.inputs.get("q"), all_blocks) if block.inputs.get("q") else up
            k = _resolve(block.inputs.get("k"), all_blocks) if block.inputs.get("k") else up
            v = _resolve(block.inputs.get("v"), all_blocks) if block.inputs.get("v") else up
            return "        " + vid + ", _ = self." + vid + "(" + q + ", " + k + ", " + v + ")"
        if block.op_type == "transformerencoder":
            src = _resolve(block.inputs.get("src"), all_blocks) if block.inputs.get("src") else up
            return "        " + vid + " = self." + vid + "(" + src + ")"
        if block.op_type == "transformerdecoder":
            tgt = _resolve(block.inputs.get("tgt"), all_blocks) if block.inputs.get("tgt") else up
            memory = _resolve(block.inputs.get("memory"), all_blocks) if block.inputs.get("memory") else tgt
            return "        " + vid + " = self." + vid + "(" + tgt + ", " + memory + ")"
        if block.op_type == "lstm":
            return "        " + vid + ", _ = self." + vid + "(" + up + ")"
        if block.op_type == "reshape":
            shape = block.fields.get("shape", -1)
            return "        " + vid + " = " + up + ".view(" + up + ".size(0), " + str(shape) + ")"
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
    elif op == "labelsmoothing":
        smoothing = f.get("smoothing", 0.1)
        lines.append(f"    # LabelSmoothing loss: {name}")
        lines.append(f"    criterion_{name} = nn.CrossEntropyLoss(label_smoothing={smoothing})")
    elif op == "focalloss":
        alpha = f.get("alpha", 1.0)
        gamma = f.get("gamma", 2.0)
        lines.append(f"    # FocalLoss: {name}")
        lines.append(f"    criterion_{name} = FocalLoss(alpha={alpha}, gamma={gamma})")
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
# Loss 函数映射表
# 统一管理所有 loss 类型的初始化代码和自定义类
# ─────────────────────────────────────────────────────────────────

LOSS_CLASS_MAP = {
    # Basic / Classic
    "cross_entropy": "nn.CrossEntropyLoss()",
    "mse": "nn.MSELoss()",
    "bce": "nn.BCEWithLogitsLoss()",
    "bce_logits": "nn.BCEWithLogitsLoss()",
    # CV - Segmentation
    "dice": "DiceLoss()",
    "focal": "FocalLoss()",
    "lovasz": "LovaszLoss()",
    "tversky": "TverskyLoss()",
    "iou": "IoULoss()",
    "giou": "GIoULoss()",
    "dice_ce": "DiceCELoss()",
    # CV - Metric / Perceptual
    "msssim": "MS_SSIMLoss()",
    "perceptual": "PerceptualLoss()",
    "content": "ContentLoss()",
    "style": "StyleLoss()",
    # CV - Detection
    "smooth_l1": "nn.SmoothL1Loss()",
    "focal_loss": "FocalLoss()",
    "class_balanced": "ClassBalancedLoss()",
    # Audio Enhancement
    "stft": "STFTLoss()",
    "sdr": "SDRLoss()",
    "sisdr": "SISDRLoss()",
    "mel_spec": "MelSpectrogramLoss()",
    "waveform": "WaveformMSELoss()",
    "multi_res": "MultiResolutionSTFTLoss()",
    "phase": "PhaseLoss()",
    # NLP / Other
    "label_smoothing": "nn.LabelSmoothingLoss()",
    "contrastive": "ContrastiveLoss()",
}

# 需要自定义类实现的 loss 类型
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
    "focalloss": '''
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
def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if len(googles) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

class LovaszLoss(nn.Module):
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        # Simplified Lovász-Softmax
        return F.cross_entropy(pred, target)
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
        union = pred.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))
        iou = (intersection + self.smooth) / (union - intersection + self.smooth)
        return 1 - iou.mean()
''',
    "giou": '''
class GIoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        # Simplified GIoU for semantic segmentation
        return F.cross_entropy(pred, target)
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
    "msssim": '''
class MS_SSIMLoss(nn.Module):
    def __init__(self, alpha=0.84):
        super().__init__()
        self.alpha = alpha
    def forward(self, pred, target):
        # Simplified MS-SSIM - use MSE as approximation
        return 1 - self.alpha * (1 - F.mse_loss(pred, target))
''',
    "perceptual": '''
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, pred, target):
        # Simplified perceptual loss using MSE
        return self.mse(pred, target)
''',
    "content": '''
class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, pred, target):
        return self.mse(pred, target)
''',
    "style": '''
class StyleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, pred, target):
        return self.mse(pred, target)
''',
    "class_balanced": '''
class ClassBalancedLoss(nn.Module):
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def forward(self, pred, target):
        return F.cross_entropy(pred, target)
''',
    "stft": '''
class STFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, pred, target):
        # Simplified STFT loss
        return self.mse(pred, target)
''',
    "sdr": '''
class SDRLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        # Signal-to-Distortion Ratio
        signal_power = (target ** 2).sum()
        noise_power = ((pred - target) ** 2).sum()
        return -10 * torch.log10(signal_power / (noise_power + 1e-8))
''',
    "sisdr": '''
class SISDRLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        # Scale-Invariant SDR
        target_mean = target.mean(dim=-1, keepdim=True)
        pred_mean = pred.mean(dim=-1, keepdim=True)
        target = target - target_mean
        pred = pred - pred_mean
        signal = (target * pred).sum()
        noise = ((pred - target) ** 2).sum()
        return -10 * torch.log10(signal ** 2 / (noise + 1e-8))
''',
    "mel_spec": '''
class MelSpectrogramLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, pred, target):
        # Simplified mel spectrogram loss
        return self.mse(pred, target)
''',
    "waveform": '''
class WaveformMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, pred, target):
        return self.mse(pred, target)
''',
    "multi_res": '''
class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, pred, target):
        # Simplified multi-resolution STFT loss
        return self.mse(pred, target)
''',
    "phase": '''
class PhaseLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        # Phase difference loss
        return F.mse_loss(pred, target)
''',
    "contrastive": '''
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    def forward(self, z1, z2):
        # Simplified contrastive loss
        return F.mse_loss(z1, z2)
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

    # Handle composite loss
    if loss_cfg.get("type") == "composite":
        components = loss_cfg.get("params", {}).get("components", [])
        for i, comp in enumerate(components):
            comp_type = comp.get("type", "cross_entropy")
            comp_class = LOSS_CLASS_MAP.get(comp_type, "nn.CrossEntropyLoss()")
            if comp_type in LOSS_CUSTOM_CLASSES:
                custom_class_lines.append(LOSS_CUSTOM_CLASSES[comp_type])
            lines.append(f"    loss_fn_{i} = {comp_class}")
        # Build composite loss expression
        loss_exprs = []
        for i, comp in enumerate(components):
            weight = comp.get("weight", 1.0)
            loss_exprs.append(f"{weight} * loss_fn_{i}(primary_output, target)")
        lines.append(f"    loss = {' + '.join(loss_exprs)}")
        lines.append("    target = torch.randint(0, max(2, primary_output.shape[-1] if primary_output.dim() > 1 else 2), (primary_output.shape[0],), dtype=torch.long, device=primary_output.device)")
    elif loss_cfg.get("enabled", True):
        # Add custom class definition if needed
        if loss_type in LOSS_CUSTOM_CLASSES:
            custom_class_lines.append(LOSS_CUSTOM_CLASSES[loss_type])

        loss_class = LOSS_CLASS_MAP.get(loss_type, "nn.CrossEntropyLoss()")
        lines.append(f"    loss_fn = {loss_class}")

        # Generate appropriate target based on loss type
        if loss_type in ("mse",):
            lines.append("    target = torch.randn_like(primary_output)")
        else:
            # Classification losses need integer targets
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
        # For composite loss, the loss expression is already set
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


def generate(flow_json: dict, options: dict | None = None) -> str:
    opts = options or {}
    eval_enabled = opts.get("evaluation_nodes", True)
    training_config = opts.get("training_config")

    sorted_blocks = build_ast(flow_json, options)

    # 收集模块 imports
    custom_modules = collect_module_imports(sorted_blocks)
    module_imports = gen_module_imports(custom_modules)

    init_lines, fwd_lines = [], []

    for block in sorted_blocks:
        # Skip training and evaluation nodes in forward (they go in if __name__ instead)
        if block.category in ("training", "evaluation"):
            continue
        il = _gen_init(block)
        if il:
            init_lines.append(f"        {il}")
        fl = _gen_forward(block, sorted_blocks)
        if fl:
            fwd_lines.append(fl)

    # ── Multi-output ─────────────────────────────────────────────────────────
    output_return_lines = []
    non_output_fwd_lines = []
    for line in fwd_lines:
        # Handle multi-line strings (e.g. output node generates "x_out = ...\nreturn ...")
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
    init_block = "\n".join(init_lines) if init_lines else "        pass"

    # ── Evaluation code ───────────────────────────────────────────────────────
    eval_main_lines: list[str] = []
    if eval_enabled:
        model_blocks_for_eval = [b for b in sorted_blocks if b.category not in ("training", "evaluation")]
        for block in sorted_blocks:
            if block.category != "evaluation":
                continue
            lines = _gen_evaluation(block, model_blocks_for_eval)
            eval_main_lines.extend(lines)

    # ── Training code (in if __name__ block) ─────────────────────────────────
    training_main_lines: list[str] = []
    custom_class_lines: list[str] = []  # Initialize to avoid UnboundLocalError at line 1140
    training_enabled = opts.get("training_nodes", False)
    if training_enabled:
        training_blocks_for_main = [b for b in sorted_blocks if b.category == "training"]
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
            lines = _gen_training(block, sorted_blocks)
            if block.op_type in ("cosineannealinglr", "steplr", "reducelronplateau"):
                lines = [line.replace(f"optimizer_{block.output_var}", scheduler_optimizer) for line in lines]
            training_main_lines.extend(lines)
    elif training_config:
        custom_class_lines, training_lines = _gen_training_from_config(training_config)
        training_main_lines.extend(training_lines)

    # targets placeholder if evaluation needs it
    needs_targets = len([b for b in sorted_blocks if b.category == "evaluation"]) > 0
    targets_placeholder = "    # Placeholder labels (replace with real dataset labels)\n    y_true = torch.randint(0, 10, (1,))" if needs_targets else ""

    eval_block = ("\n" + "\n".join(eval_main_lines)) if eval_main_lines else ""

    training_block = ("\n" + "\n".join(training_main_lines)) if training_main_lines else ""

    # Custom loss classes (inserted before model class)
    custom_class_block = ("\n" + "\n".join(custom_class_lines)) if custom_class_lines else ""

    primary_output_line = "    primary_output = output[0]" if len(output_return_lines) > 1 else "    primary_output = output"
    device_name = training_config.get("runtime", {}).get("device", "cpu") if isinstance(training_config, dict) else "cpu"

    return f"""# Generated by FlowHamster
# DO NOT EDIT -- Regenerated from graph editor
{module_imports}
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
