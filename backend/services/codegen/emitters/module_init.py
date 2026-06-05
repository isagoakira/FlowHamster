"""Emit nn.Module __init__ lines for graph module nodes."""
from __future__ import annotations

from typing import Callable, Optional

from ..graph import NodeBlock

InitEmitter = Callable[[NodeBlock], str]
INIT_EMITTERS: dict[str, InitEmitter] = {}


def init_emitter(*op_types: str) -> Callable[[InitEmitter], InitEmitter]:
    def decorator(fn: InitEmitter) -> InitEmitter:
        for op_type in op_types:
            INIT_EMITTERS[op_type] = fn
        return fn

    return decorator


def _assign(block: NodeBlock, expression: str) -> str:
    return f"self.{block.output_var} = {expression}"


@init_emitter("conv1d", "conv2d", "conv3d")
def _emit_conv(block: NodeBlock) -> str:
    f = block.fields
    dims = block.op_type[-2:]
    cls = f"nn.Conv{dims}"
    return _assign(
        block,
        cls
        + "(in_channels="
        + str(f.get("in_channels", 0))
        + ", out_channels="
        + str(f.get("out_channels", 0))
        + ", kernel_size="
        + str(f.get("kernel_size", 3))
        + ", stride="
        + str(f.get("stride", 1))
        + ", padding="
        + str(f.get("padding", 0))
        + ", bias="
        + str(f.get("bias", False))
        + ")",
    )


@init_emitter("linear")
def _emit_linear(block: NodeBlock) -> str:
    f = block.fields
    return _assign(
        block,
        "nn.Linear(in_features="
        + str(f.get("in_features", 0))
        + ", out_features="
        + str(f.get("out_features", 0))
        + ", bias="
        + str(f.get("bias", True))
        + ")",
    )


@init_emitter("relu", "gelu", "silu", "sigmoid", "tanh")
def _emit_simple_activation(block: NodeBlock) -> str:
    class_name = {
        "relu": "ReLU",
        "gelu": "GELU",
        "silu": "SiLU",
        "sigmoid": "Sigmoid",
        "tanh": "Tanh",
    }[block.op_type]
    return _assign(block, f"nn.{class_name}()")


@init_emitter("leakyrelu")
def _emit_leakyrelu(block: NodeBlock) -> str:
    return _assign(block, "nn.LeakyReLU(negative_slope=" + str(block.fields.get("negative_slope", 0.01)) + ")")


@init_emitter("maxpool2d", "avgpool2d")
def _emit_pool2d(block: NodeBlock) -> str:
    f = block.fields
    cls = "MaxPool2d" if block.op_type == "maxpool2d" else "AvgPool2d"
    return _assign(
        block,
        "nn."
        + cls
        + "(kernel_size="
        + str(f.get("kernel_size", 2))
        + ", stride="
        + str(f.get("stride", 2))
        + ", padding="
        + str(f.get("padding", 0))
        + ")",
    )


@init_emitter("adaptiveavgpool2d")
def _emit_adaptive_avg_pool(block: NodeBlock) -> str:
    return _assign(block, "nn.AdaptiveAvgPool2d(output_size=" + str(block.fields.get("output_size", 1)) + ")")


@init_emitter("globalavgpool")
def _emit_global_avg_pool(block: NodeBlock) -> str:
    return _assign(block, "nn.AdaptiveAvgPool2d(1)")


@init_emitter("batchnorm2d")
def _emit_batchnorm2d(block: NodeBlock) -> str:
    return _assign(block, "nn.BatchNorm2d(num_features=" + str(block.fields.get("num_features", 0)) + ")")


@init_emitter("layernorm")
def _emit_layernorm(block: NodeBlock) -> str:
    return _assign(block, "nn.LayerNorm(normalized_shape=" + str(block.fields.get("normalized_shape", 64)) + ")")


@init_emitter("groupnorm")
def _emit_groupnorm(block: NodeBlock) -> str:
    f = block.fields
    return _assign(
        block,
        "nn.GroupNorm(num_groups="
        + str(f.get("num_groups", 1))
        + ", num_channels="
        + str(f.get("num_channels", 0))
        + ")",
    )


@init_emitter("dropout")
def _emit_dropout(block: NodeBlock) -> str:
    return _assign(block, "nn.Dropout(p=" + str(block.fields.get("p", 0.5)) + ", inplace=True)")


@init_emitter("softmax")
def _emit_softmax(block: NodeBlock) -> str:
    return _assign(block, "nn.Softmax(dim=" + str(block.fields.get("dim", -1)) + ")")


@init_emitter("flatten")
def _emit_flatten(block: NodeBlock) -> str:
    return _assign(block, "nn.Flatten(start_dim=" + str(block.fields.get("start_dim", 1)) + ")")


@init_emitter("embedding")
def _emit_embedding(block: NodeBlock) -> str:
    f = block.fields
    return _assign(
        block,
        "nn.Embedding(num_embeddings="
        + str(f.get("num_embeddings", 0))
        + ", embedding_dim="
        + str(f.get("embedding_dim", 0))
        + ")",
    )


@init_emitter("selfattention", "crossattention")
def _emit_attention(block: NodeBlock) -> str:
    f = block.fields
    class_name = "SelfAttention" if block.op_type == "selfattention" else "CrossAttention"
    return _assign(block, class_name + "(dim=" + str(f.get("embed_dim", 512)) + ", heads=" + str(f.get("num_heads", 8)) + ")")


@init_emitter("multiheadattention")
def _emit_multihead_attention(block: NodeBlock) -> str:
    f = block.fields
    return _assign(
        block,
        "nn.MultiheadAttention(embed_dim="
        + str(f.get("embed_dim", 512))
        + ", num_heads="
        + str(f.get("num_heads", 8))
        + ", dropout="
        + str(f.get("dropout", 0))
        + ", batch_first=True)",
    )


@init_emitter("ffn")
def _emit_ffn(block: NodeBlock) -> str:
    f = block.fields
    return _assign(block, "FFN(dim=" + str(f.get("dim", 512)) + ", hidden_dim=" + str(f.get("hidden_dim", 2048)) + ")")


@init_emitter("mlp")
def _emit_mlp(block: NodeBlock) -> str:
    f = block.fields
    return _assign(
        block,
        "MLP(dim="
        + str(f.get("in_features", 784))
        + ", hidden_dim="
        + str(f.get("hidden_features", 256))
        + ", depth="
        + str(f.get("depth", 2))
        + ", out_dim="
        + str(f.get("out_features", 10))
        + ")",
    )


@init_emitter("transformerencoder", "transformerdecoder")
def _emit_transformer(block: NodeBlock) -> str:
    f = block.fields
    d = str(f.get("embed_dim", f.get("d_model", 512)))
    nh = str(f.get("num_heads", f.get("nhead", 8)))
    dl = str(f.get("num_layers", 6))
    dim_ff = str(f.get("dim_feedforward", 2048))
    layer = "TransformerEncoderLayer" if block.op_type == "transformerencoder" else "TransformerDecoderLayer"
    module = "TransformerEncoder" if block.op_type == "transformerencoder" else "TransformerDecoder"
    return _assign(
        block,
        "nn."
        + module
        + "(nn."
        + layer
        + "(d_model="
        + d
        + ", nhead="
        + nh
        + ", dim_feedforward="
        + dim_ff
        + ", batch_first=True), num_layers="
        + dl
        + ")",
    )


@init_emitter("mamba")
def _emit_mamba(block: NodeBlock) -> str:
    f = block.fields
    return _assign(
        block,
        "Mamba(d_model="
        + str(f.get("d_model", 512))
        + ", d_state="
        + str(f.get("d_state", 16))
        + ", d_conv="
        + str(f.get("d_conv", 4))
        + ", expand="
        + str(f.get("expand", 2))
        + ", dt_rank="
        + str(f.get("dt_rank", "auto"))
        + ", dropout="
        + str(f.get("dropout", 0.0))
        + ", n_layers="
        + str(f.get("n_layers", 1))
        + ")",
    )


@init_emitter("lstm")
def _emit_lstm(block: NodeBlock) -> str:
    f = block.fields
    bidirectional = "True" if f.get("bidirectional", False) else "False"
    return _assign(
        block,
        "nn.LSTM(input_size="
        + str(f.get("input_size", 512))
        + ", hidden_size="
        + str(f.get("hidden_size", 512))
        + ", num_layers="
        + str(f.get("num_layers", 2))
        + ", batch_first=True, bidirectional="
        + bidirectional
        + ")",
    )


@init_emitter("instnorm")
def _emit_instnorm(block: NodeBlock) -> str:
    return _assign(block, "nn.InstanceNorm2d(num_features=" + str(block.fields.get("num_channels", 64)) + ")")


def _gen_init(block: NodeBlock) -> Optional[str]:
    if block.category != "module":
        return None
    emitter = INIT_EMITTERS.get(block.op_type)
    return emitter(block) if emitter else None
