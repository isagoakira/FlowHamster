"""Emit FlowHamsterModel.forward execution lines."""
from __future__ import annotations

from typing import Callable, Optional

from ..graph import NodeBlock

ForwardEmitter = Callable[[NodeBlock, list[NodeBlock]], Optional[str]]
FORWARD_EMITTERS: dict[str, ForwardEmitter] = {}


def forward_emitter(*op_types: str) -> Callable[[ForwardEmitter], ForwardEmitter]:
    def decorator(fn: ForwardEmitter) -> ForwardEmitter:
        for op_type in op_types:
            FORWARD_EMITTERS[op_type] = fn
        return fn

    return decorator


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


def _first_input(block: NodeBlock, all_blocks: list[NodeBlock]) -> str:
    first_ref = None
    for value in block.inputs.values():
        if value:
            first_ref = value[0] if isinstance(value, list) else value
            break
    return _resolve(first_ref, all_blocks)


@forward_emitter("input")
def _emit_input(block: NodeBlock, all_blocks: list[NodeBlock]) -> str:
    return "        " + block.output_var + " = x"


@forward_emitter("output")
def _emit_output(block: NodeBlock, all_blocks: list[NodeBlock]) -> str:
    src = _resolve(block.inputs.get("x"), all_blocks)
    return "        " + block.output_var + " = " + src + "\n        return " + block.output_var


@forward_emitter("add", "mul")
def _emit_binary_op(block: NodeBlock, all_blocks: list[NodeBlock]) -> str:
    a = _resolve(block.inputs.get("a"), all_blocks)
    b = _resolve(block.inputs.get("b"), all_blocks)
    op = "+" if block.op_type == "add" else "*"
    return "        " + block.output_var + " = " + a + " " + op + " " + b


@forward_emitter("concat")
def _emit_concat(block: NodeBlock, all_blocks: list[NodeBlock]) -> Optional[str]:
    all_srcs = []
    for src_list in block.inputs.values():
        if isinstance(src_list, list):
            for ref in src_list:
                value = _resolve(ref, all_blocks)
                if value not in all_srcs:
                    all_srcs.append(value)
    if len(all_srcs) >= 2:
        dim = str(block.fields.get("dim", 1))
        return "        " + block.output_var + " = torch.cat([" + ", ".join(all_srcs) + "], dim=" + dim + ")"
    if len(all_srcs) == 1:
        return "        " + block.output_var + " = " + all_srcs[0]
    return None


@forward_emitter("reshape")
def _emit_reshape(block: NodeBlock, all_blocks: list[NodeBlock]) -> str:
    up = _first_input(block, all_blocks)
    shape = block.fields.get("shape", -1)
    return "        " + block.output_var + " = " + up + ".reshape(" + up + ".size(0), " + str(shape) + ")"


@forward_emitter("transpose")
def _emit_transpose(block: NodeBlock, all_blocks: list[NodeBlock]) -> str:
    up = _first_input(block, all_blocks)
    return (
        "        "
        + block.output_var
        + " = "
        + up
        + ".transpose("
        + str(block.fields.get("dim0", 0))
        + ", "
        + str(block.fields.get("dim1", 1))
        + ")"
    )


@forward_emitter("split")
def _emit_split(block: NodeBlock, all_blocks: list[NodeBlock]) -> str:
    up = _first_input(block, all_blocks)
    return (
        "        "
        + block.output_var
        + " = "
        + up
        + ".split("
        + str(block.fields.get("split_size", 32))
        + ", dim="
        + str(block.fields.get("dim", 0))
        + ")"
    )


@forward_emitter("slice")
def _emit_slice(block: NodeBlock, all_blocks: list[NodeBlock]) -> str:
    up = _first_input(block, all_blocks)
    start = str(block.fields.get("start", 0))
    end = str(block.fields.get("end", -1))
    step = str(block.fields.get("step", 1))
    return "        " + block.output_var + " = " + up + "[" + start + ":" + end + ":" + step + "]"


@forward_emitter("upsample")
def _emit_upsample(block: NodeBlock, all_blocks: list[NodeBlock]) -> str:
    up = _first_input(block, all_blocks)
    size = block.fields.get("size", None)
    scale_factor = block.fields.get("scale_factor", 2.0)
    mode = block.fields.get("mode", "bilinear")
    if size:
        return "        " + block.output_var + " = torch.nn.functional.interpolate(" + up + ", size=" + str(size) + ", mode='" + mode + "', align_corners=False)"
    return "        " + block.output_var + " = torch.nn.functional.interpolate(" + up + ", scale_factor=" + str(scale_factor) + ", mode='" + mode + "', align_corners=False)"


@forward_emitter("selfattention")
def _emit_self_attention(block: NodeBlock, all_blocks: list[NodeBlock]) -> str:
    up = _first_input(block, all_blocks)
    return "        " + block.output_var + " = self." + block.output_var + "(" + up + ")"


@forward_emitter("crossattention")
def _emit_cross_attention(block: NodeBlock, all_blocks: list[NodeBlock]) -> str:
    q = _resolve(block.inputs.get("q"), all_blocks)
    kv = _resolve(block.inputs.get("kv"), all_blocks)
    return "        " + block.output_var + " = self." + block.output_var + "(" + q + ", " + kv + ", " + kv + ")"


@forward_emitter("multiheadattention")
def _emit_multihead_attention(block: NodeBlock, all_blocks: list[NodeBlock]) -> str:
    up = _first_input(block, all_blocks)
    q = _resolve(block.inputs.get("q"), all_blocks) if block.inputs.get("q") else up
    k = _resolve(block.inputs.get("k"), all_blocks) if block.inputs.get("k") else up
    v = _resolve(block.inputs.get("v"), all_blocks) if block.inputs.get("v") else up
    return "        " + block.output_var + " = self." + block.output_var + "(" + q + ", " + k + ", " + v + ")"


@forward_emitter("transformerencoder")
def _emit_transformer_encoder(block: NodeBlock, all_blocks: list[NodeBlock]) -> str:
    up = _first_input(block, all_blocks)
    src = _resolve(block.inputs.get("src"), all_blocks) if block.inputs.get("src") else up
    return "        " + block.output_var + " = self." + block.output_var + "(" + src + ")"


@forward_emitter("transformerdecoder")
def _emit_transformer_decoder(block: NodeBlock, all_blocks: list[NodeBlock]) -> str:
    up = _first_input(block, all_blocks)
    tgt = _resolve(block.inputs.get("tgt"), all_blocks) if block.inputs.get("tgt") else up
    memory = _resolve(block.inputs.get("memory"), all_blocks) if block.inputs.get("memory") else tgt
    return "        " + block.output_var + " = self." + block.output_var + "(" + tgt + ", " + memory + ")"


@forward_emitter("lstm")
def _emit_lstm(block: NodeBlock, all_blocks: list[NodeBlock]) -> str:
    up = _first_input(block, all_blocks)
    return "        " + block.output_var + ", (hidden, cell) = self." + block.output_var + "(" + up + ")"


@forward_emitter("flatten")
def _emit_flatten(block: NodeBlock, all_blocks: list[NodeBlock]) -> str:
    up = _first_input(block, all_blocks)
    return "        " + block.output_var + " = " + up + ".flatten(start_dim=" + str(block.fields.get("start_dim", 1)) + ")"


def _emit_default_module(block: NodeBlock, all_blocks: list[NodeBlock]) -> str:
    up = _first_input(block, all_blocks)
    return "        " + block.output_var + " = self." + block.output_var + "(" + up + ")"


def _gen_forward(block: NodeBlock, all_blocks: list) -> Optional[str]:
    emitter = FORWARD_EMITTERS.get(block.op_type)
    if emitter:
        return emitter(block, all_blocks)
    if block.category == "module":
        return _emit_default_module(block, all_blocks)
    return None
