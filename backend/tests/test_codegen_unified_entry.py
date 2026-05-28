import base64

import nbformat
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.codegen_facade import generate_full
from backend.services.tensor_executor import execute_forward


client = TestClient(app)


def simple_graph() -> dict:
    return {
        "nodes": [
            {"id": "input", "data": {"nodeType": "input", "params": {}}},
            {"id": "flatten", "data": {"nodeType": "flatten", "params": {"start_dim": 1}}},
            {"id": "linear", "data": {"nodeType": "linear", "params": {"in_features": 48, "out_features": 4}}},
            {"id": "output", "data": {"nodeType": "output", "params": {}}},
        ],
        "edges": [
            {"source": "input", "target": "flatten", "sourceHandle": "result", "targetHandle": "a"},
            {"source": "flatten", "target": "linear", "sourceHandle": "result", "targetHandle": "a"},
            {"source": "linear", "target": "output", "sourceHandle": "result", "targetHandle": "a"},
        ],
    }


def test_generate_export_notebook_and_websocket_share_generated_code():
    graph = simple_graph()

    generate_response = client.post("/api/generate", json={"graph": graph})
    assert generate_response.status_code == 200
    generated = generate_response.json()["code"]
    assert generated == generate_full(graph)

    notebook_response = client.post("/api/export-notebook", json={"graph": graph})
    assert notebook_response.status_code == 200
    notebook_payload = notebook_response.json()
    assert notebook_payload["success"] is True
    notebook = nbformat.reads(base64.b64decode(notebook_payload["content"]).decode(), as_version=4)
    code_cells = [cell["source"] for cell in notebook.cells if cell["cell_type"] == "code"]
    assert code_cells
    for source in code_cells:
        assert source in generated

    with client.websocket_connect("/ws/code") as ws:
        ws.send_json(graph)
        assert ws.receive_json()["code"] == generated


def test_forward_uses_unified_model_generation_success_path():
    outputs = execute_forward(simple_graph(), input_shape=[1, 3, 4, 4])

    assert outputs["linear"]["shape"] == [1, 4]


def test_forward_shape_error_is_readable():
    graph = simple_graph()
    graph["nodes"][2]["data"]["params"]["in_features"] = 16

    with pytest.raises(RuntimeError, match="Forward pass failed"):
        execute_forward(graph, input_shape=[1, 3, 4, 4])


def test_forward_invalid_graph_is_rejected():
    graph = {
        "nodes": [{"id": "input", "data": {"nodeType": "input", "params": {}}}],
        "edges": [{"source": "input", "target": "missing"}],
    }

    with pytest.raises(ValueError, match="edge references missing node"):
        execute_forward(graph, input_shape=[1, 3, 4, 4])


def test_crossattention_frontend_handles_map_to_backend_inputs():
    graph = {
        "nodes": [
            {"id": "input", "data": {"nodeType": "input", "params": {}}},
            {
                "id": "attn",
                "data": {
                    "nodeType": "crossattention",
                    "params": {"query_dim": 256, "kv_dim": 512, "num_heads": 4},
                },
            },
            {"id": "output", "data": {"nodeType": "output", "params": {}}},
        ],
        "edges": [
            {"source": "input", "target": "attn", "sourceHandle": "result", "targetHandle": "tgt"},
            {"source": "input", "target": "attn", "sourceHandle": "result", "targetHandle": "kv"},
            {"source": "attn", "target": "output", "sourceHandle": "result", "targetHandle": "a"},
        ],
    }

    code = generate_full(graph)

    assert "CrossAttention(dim=256, heads=4)" in code
    assert "self.x_crossattention_1(x_input_1, x_input_1, x_input_1)" in code


def test_frontend_tensor_nodes_are_supported_by_backend_generator():
    graph = {
        "nodes": [
            {"id": "input", "data": {"nodeType": "input", "params": {}}},
            {"id": "permute", "data": {"nodeType": "permute", "params": {"dims": "0,2,3,1"}}},
            {"id": "squeeze", "data": {"nodeType": "squeeze", "params": {}}},
            {"id": "expand", "data": {"nodeType": "expand", "params": {"shape": "-1,4,4,3"}}},
            {"id": "output", "data": {"nodeType": "output", "params": {}}},
        ],
        "edges": [
            {"source": "input", "target": "permute", "sourceHandle": "result", "targetHandle": "a"},
            {"source": "permute", "target": "squeeze", "sourceHandle": "result", "targetHandle": "a"},
            {"source": "squeeze", "target": "expand", "sourceHandle": "result", "targetHandle": "a"},
            {"source": "expand", "target": "output", "sourceHandle": "result", "targetHandle": "a"},
        ],
    }

    code = generate_full(graph)

    assert ".permute(0,2,3,1)" in code
    assert ".squeeze()" in code
    assert ".expand(-1,4,4,3)" in code
