import base64
import json
import unittest

import nbformat
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers import export, generate, websocket
from backend.services.code_generation import generate_python


def _linear_graph() -> dict:
    return {
        "nodes": [
            {"id": "input", "data": {"nodeType": "input", "label": "Input", "params": {"name": "features"}}},
            {"id": "fc", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 4, "out_features": 2, "bias": True}}},
            {"id": "output", "data": {"nodeType": "output", "label": "Output", "params": {}}},
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "fc", "sourceHandle": "result", "targetHandle": "x"},
            {"id": "e2", "source": "fc", "target": "output", "sourceHandle": "result", "targetHandle": "x"},
        ],
    }


class CodeGenerationFacadeTest(unittest.TestCase):
    def test_generate_python_returns_ordered_sections_and_full_code(self):
        generated = generate_python(graph=_linear_graph())

        self.assertEqual(generated.warnings, [])
        self.assertIn("import torch", generated.sections.imports)
        self.assertIn("class FlowHamsterModel(nn.Module):", generated.sections.model)
        self.assertIn('if __name__ == "__main__":', generated.sections.main)
        self.assertLess(generated.code.index("import torch"), generated.code.index("class FlowHamsterModel"))
        self.assertLess(generated.code.index("class FlowHamsterModel"), generated.code.index('if __name__ == "__main__":'))

    def test_composite_node_generates_reusable_class(self):
        graph = {
            "nodes": [
                {"id": "input", "data": {"nodeType": "input", "label": "Input", "params": {}}},
                {
                    "id": "block",
                    "data": {
                        "nodeType": "composite",
                        "label": "Reusable Block",
                        "params": {
                            "className": "ReusableLinear",
                            "subgraph": _linear_graph(),
                        },
                    },
                },
                {"id": "output", "data": {"nodeType": "output", "label": "Output", "params": {}}},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "block", "sourceHandle": "result", "targetHandle": "x"},
                {"id": "e2", "source": "block", "target": "output", "sourceHandle": "result", "targetHandle": "x"},
            ],
        }

        generated = generate_python(graph=graph)

        self.assertIn("class ReusableLinear(nn.Module):", generated.sections.reusable_classes)
        self.assertIn("self.x_composite_1 = ReusableLinear()", generated.sections.model)
        self.assertLess(generated.code.index("class ReusableLinear"), generated.code.index("class FlowHamsterModel"))

    def test_websocket_code_matches_http_generate(self):
        app = FastAPI()
        app.include_router(generate.router, prefix="/api")
        app.include_router(websocket.router)
        client = TestClient(app)
        graph = _linear_graph()

        http_body = client.post("/api/generate", json={"graph": graph}).json()
        with client.websocket_connect("/ws/code") as ws:
            ws.send_text(json.dumps({"graph": graph}))
            ws_body = json.loads(ws.receive_text())

        self.assertTrue(http_body["success"])
        self.assertEqual(ws_body["code"], http_body["code"])
        self.assertNotIn("warnings", ws_body)

    def test_export_notebook_uses_unified_inline_classes(self):
        app = FastAPI()
        app.include_router(export.router, prefix="/api")
        graph = {
            "nodes": [
                {"id": "input", "data": {"nodeType": "input", "label": "Input", "params": {}}},
                {"id": "mamba", "data": {"nodeType": "mamba", "label": "Mamba", "params": {"d_model": 512}}},
                {"id": "output", "data": {"nodeType": "output", "label": "Output", "params": {}}},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "mamba", "sourceHandle": "result", "targetHandle": "x"},
                {"id": "e2", "source": "mamba", "target": "output", "sourceHandle": "result", "targetHandle": "x"},
            ],
        }

        response = TestClient(app).post("/api/export-notebook", json={"graph": graph})
        body = response.json()
        notebook_json = base64.b64decode(body["content"]).decode()
        notebook = nbformat.reads(notebook_json, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["success"])
        self.assertIn("class Mamba(nn.Module):", source)
        self.assertIn("class FlowHamsterModel(nn.Module):", source)
        self.assertNotIn("from backend.modules import Mamba", source)


if __name__ == "__main__":
    unittest.main()
