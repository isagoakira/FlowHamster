"""Node signatures, aliases, and naming helpers."""
from __future__ import annotations

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


