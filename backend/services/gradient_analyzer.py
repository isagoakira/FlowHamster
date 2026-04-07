"""
根据图结构分析梯度流动，返回每条边的相对梯度强度（0~1，1=梯度最强）。
不需要真实数据，基于各操作的梯度传播特性估算。
"""
from typing import Optional

from backend.services.ast_core import normalize_node_type

# 各操作的大致梯度衰减因子（相对值，越大梯度越强）
GRADIENT_FACTORS = {
    "relu": 0.5,
    "sigmoid": 0.25,
    "tanh": 0.15,
    "gelu": 0.35,
    "silu": 0.3,
    "leakyrelu": 0.45,
    "softmax": 0.2,
    "dropout": 0.0,
    "maxpool2d": 0.6,
    "avgpool2d": 0.5,
    "adaptiveavgpool2d": 0.5,
    "globalavgpool": 0.5,
    "conv2d": 0.55,
    "conv1d": 0.55,
    "conv3d": 0.55,
    "linear": 0.65,
    "batchnorm2d": 0.3,
    "layernorm": 0.3,
    "groupnorm": 0.3,
    "add": 1.0,
    "mul": 0.5,
    "concat": 1.0,
    "flatten": 1.0,
    "reshape": 1.0,
    "transpose": 1.0,
    "split": 1.0,
    "slice": 1.0,
    "embedding": 0.4,
    # Aliases for node type names used in the frontend
    "input": 1.0,
    "output": 1.0,
    "mseloss": 1.0,
    "crossentropyloss": 1.0,
    "adam": 0.0,
    "adamw": 0.0,
    "sgd": 0.0,
    "rmsprop": 0.0,
    "steplr": 0.0,
    "cosineannealing": 0.0,
    "reduceonplateau": 0.0,
    "ffn": 0.65,
    "mlp": 0.65,
    "selfattention": 0.45,
    "crossattention": 0.45,
    "multiheadattention": 0.45,
    "transformerencoder": 0.4,
    "transformerdecoder": 0.4,
    "mamba": 0.4,
}


def _get_factor(node_type: str) -> float:
    """Look up gradient factor for a node type, with fuzzy aliasing."""
    # Direct match
    normalized = normalize_node_type(node_type.lower())
    if normalized in GRADIENT_FACTORS:
        return GRADIENT_FACTORS[normalized]
    # Try with "node" suffix stripped
    cleaned = node_type.lower().replace("node", "")
    if cleaned in GRADIENT_FACTORS:
        return GRADIENT_FACTORS[cleaned]
    return 0.5  # default conservative


def _topological_sort_reverse(nodes: list, edges: list) -> list:
    """Return nodes in reverse topological order (output → input)."""
    # Build adjacency and in-degree
    in_degree = {n["id"]: 0 for n in nodes}
    adj = {n["id"]: [] for n in nodes}
    for e in edges:
        adj[e["source"]].append(e["target"])
        in_degree[e["target"]] += 1

    # Kahn's algorithm to get forward topo order
    queue = [nid for nid, d in in_degree.items() if d == 0]
    topo = []
    while queue:
        cur = queue.pop(0)
        topo.append(cur)
        for nb in adj[cur]:
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                queue.append(nb)

    return list(reversed(topo))  # reverse = output first


def analyze_gradients(flow_json: dict) -> dict:
    """
    返回：
    {
      "edges": [
        {"source": "conv2d_1", "target": "relu_1", "gradient_strength": 0.55, "direction": "backward"},
        ...
      ],
      "nodes": [
        {"id": "conv2d_1", "gradient_score": 0.8, "risk": "normal"}
      ]
    }
    """
    nodes = flow_json.get("nodes", [])
    edges = flow_json.get("edges", [])

    # Build node type lookup
    node_type_map = {n["id"]: n.get("type", "").replace("Node", "").lower() for n in nodes}
    # Also support data.nodeType
    for n in nodes:
        if "data" in n and "nodeType" in n["data"]:
            node_type_map[n["id"]] = n["data"]["nodeType"].lower()

    # Reverse topological order (output → input)
    rev_order = _topological_sort_reverse(nodes, edges)

    # For each node, compute cumulative gradient score (chain rule product)
    # We propagate backward: start from outputs (score=1.0), work backward
    node_score: dict[str, float] = {}

    # Identify output nodes (nodes with no outgoing edges)
    has_outgoing = {e["source"] for e in edges}
    output_nodes = [nid for nid in rev_order if nid not in has_outgoing]

    # Initialize output nodes with score 1.0
    for nid in output_nodes:
        node_score[nid] = 1.0

    # Backward pass: for each node in reverse topo order,
    # propagate score to incoming edges
    # Accumulate from all outgoing edges that have been computed
    # node_score[target] = sum over incoming edges of (source_score * factor)
    # But we need source score first — so we process in rev_order (output→input)

    # First pass: compute each node's score based on its operation factor
    # and the scores of nodes that depend on it
    # We build a map: for each node, which edges feed into it
    incoming: dict[str, list] = {n["id"]: [] for n in nodes}
    outgoing: dict[str, list] = {n["id"]: [] for n in nodes}
    edge_map: dict[tuple, dict] = {}
    for e in edges:
        incoming[e["target"]].append(e)
        outgoing[e["source"]].append(e)
        edge_map[(e["source"], e["target"])] = e

    # Initialize all scores to 0
    for n in nodes:
        node_score[n["id"]] = 0.0

    # Output nodes get base score 1.0
    for nid in output_nodes:
        node_score[nid] = 1.0

    # Backward propagation: for each node, distribute its score backward
    for nid in rev_order:
        score = node_score[nid]
        factor = _get_factor(node_type_map.get(nid, ""))
        inc_edges = incoming[nid]
        if not inc_edges:
            continue
        # Each incoming edge gets score / num_incoming * factor
        share = score / len(inc_edges)
        for e in inc_edges:
            src = e["source"]
            edge_factor = factor * _get_factor(node_type_map.get(src, ""))
            node_score[src] += share * edge_factor

    # Now compute edge gradient strengths
    result_edges = []
    for e in edges:
        src = e["source"]
        tgt = e["target"]
        src_factor = _get_factor(node_type_map.get(src, ""))
        tgt_factor = _get_factor(node_type_map.get(tgt, ""))
        # Gradient strength along this edge = source node's final score normalized
        # Use the accumulated score at source node, normalized
        max_score = max(node_score.values()) if node_score else 1.0
        src_score = node_score.get(src, 0.0)
        gradient_strength = (src_score / max_score) if max_score > 0 else 0.0
        # Clamp to [0, 1]
        gradient_strength = max(0.0, min(1.0, gradient_strength))
        result_edges.append({
            "source": src,
            "target": tgt,
            "gradient_strength": round(gradient_strength, 3),
            "direction": "backward",
        })

    # Node gradient scores
    max_score = max(node_score.values()) if node_score else 1.0
    result_nodes = []
    for n in nodes:
        nid = n["id"]
        score = node_score.get(nid, 0.0)
        norm_score = (score / max_score) if max_score > 0 else 0.0
        norm_score = max(0.0, min(1.0, norm_score))
        if norm_score > 0.8:
            risk = "exploding"
        elif norm_score < 0.1:
            risk = "vanishing"
        else:
            risk = "normal"
        result_nodes.append({
            "id": nid,
            "gradient_score": round(norm_score, 3),
            "risk": risk,
        })

    return {"edges": result_edges, "nodes": result_nodes}
