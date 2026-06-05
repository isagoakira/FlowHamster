"""Graph pruning, topological sorting, and NodeBlock construction."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from .metadata import NodeCategory, SIGNATURES, get_block_name_token, normalize_node_type

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


