"""
E2E Smoke Tests for FlowHamster ISA-216 refactoring.

Covers:
1. Frontend local code generation vs backend /api/generate semantic consistency
2. WebSocket code generation vs HTTP generation consistency
3. /api/execute graph-first execution (forward + training)
4. Composite workflow: save -> load -> generate -> execute
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers import execute, generate, websocket
from backend.services.code_generation import generate_python


REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_SCRIPT = REPO_ROOT / "scripts" / "e2e_smoke_frontend.ts"


def _run_frontend_generator(graph_type: str) -> dict:
    """Run the frontend code generator via Node.js/tsx and return the result."""
    cmd = f'npx tsx "{FRONTEND_SCRIPT}" {graph_type}'
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["NODE_OPTIONS"] = "--no-warnings"
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO_ROOT),
        timeout=120,
        shell=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Frontend generator failed: {result.stderr}")
    # The last line should be the JSON output
    lines = result.stdout.strip().splitlines()
    json_line = lines[-1] if lines else "{}"
    return json.loads(json_line)


def _extract_layers(code: str) -> list[dict]:
    """Extract layer definitions from generated Python code."""
    layers = []
    # Match self.x_xxx = nn.SomeLayer(...)
    pattern = re.compile(r"self\.(x_\w+)\s*=\s*(nn\.\w+)\((.*?)\)")
    for match in pattern.finditer(code):
        layers.append({
            "var": match.group(1),
            "type": match.group(2),
            "args": match.group(3),
        })
    return layers


def _extract_forward_assignments(code: str) -> list[dict]:
    """Extract forward pass variable assignments from generated code."""
    assignments = []
    in_forward = False
    for line in code.splitlines():
        if "def forward(self, x):" in line:
            in_forward = True
            continue
        if in_forward:
            if line.strip() == "" or (line.startswith("    ") and not line.startswith("        ")):
                break
            stripped = line.strip()
            if "=" in stripped and not stripped.startswith("#"):
                assignments.append(stripped)
    return assignments


def _linear_graph() -> dict:
    return {
        "nodes": [
            {"id": "input", "data": {"nodeType": "input", "label": "Input", "params": {"name": "features"}}},
            {"id": "fc1", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 4, "out_features": 8, "bias": True}}},
            {"id": "relu", "data": {"nodeType": "relu", "label": "ReLU", "params": {}}},
            {"id": "fc2", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 8, "out_features": 2, "bias": True}}},
            {"id": "output", "data": {"nodeType": "output", "label": "Output", "params": {}}},
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "fc1", "sourceHandle": "result", "targetHandle": "x"},
            {"id": "e2", "source": "fc1", "target": "relu", "sourceHandle": "result", "targetHandle": "x"},
            {"id": "e3", "source": "relu", "target": "fc2", "sourceHandle": "result", "targetHandle": "x"},
            {"id": "e4", "source": "fc2", "target": "output", "sourceHandle": "result", "targetHandle": "x"},
        ],
    }


def _lenet5_graph() -> dict:
    return {
        "nodes": [
            {"id": "input", "data": {"nodeType": "input", "label": "Input", "params": {"name": "image"}}},
            {"id": "conv1", "data": {"nodeType": "conv2d", "label": "Conv2d", "params": {"in_channels": 1, "out_channels": 6, "kernel_size": 5, "stride": 1, "padding": 0, "bias": True}}},
            {"id": "relu1", "data": {"nodeType": "relu", "label": "ReLU", "params": {}}},
            {"id": "pool1", "data": {"nodeType": "maxpool2d", "label": "MaxPool2d", "params": {"kernel_size": 2, "stride": 2, "padding": 0}}},
            {"id": "conv2", "data": {"nodeType": "conv2d", "label": "Conv2d", "params": {"in_channels": 6, "out_channels": 16, "kernel_size": 5, "stride": 1, "padding": 0, "bias": True}}},
            {"id": "relu2", "data": {"nodeType": "relu", "label": "ReLU", "params": {}}},
            {"id": "pool2", "data": {"nodeType": "maxpool2d", "label": "MaxPool2d", "params": {"kernel_size": 2, "stride": 2, "padding": 0}}},
            {"id": "flat", "data": {"nodeType": "flatten", "label": "Flatten", "params": {"start_dim": 1}}},
            {"id": "fc1", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 256, "out_features": 120, "bias": True}}},
            {"id": "fc2", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 120, "out_features": 84, "bias": True}}},
            {"id": "fc3", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 84, "out_features": 10, "bias": True}}},
            {"id": "output", "data": {"nodeType": "output", "label": "Output", "params": {}}},
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "conv1", "sourceHandle": "result", "targetHandle": "x"},
            {"id": "e2", "source": "conv1", "target": "relu1", "sourceHandle": "result", "targetHandle": "x"},
            {"id": "e3", "source": "relu1", "target": "pool1", "sourceHandle": "result", "targetHandle": "x"},
            {"id": "e4", "source": "pool1", "target": "conv2", "sourceHandle": "result", "targetHandle": "x"},
            {"id": "e5", "source": "conv2", "target": "relu2", "sourceHandle": "result", "targetHandle": "x"},
            {"id": "e6", "source": "relu2", "target": "pool2", "sourceHandle": "result", "targetHandle": "x"},
            {"id": "e7", "source": "pool2", "target": "flat", "sourceHandle": "result", "targetHandle": "x"},
            {"id": "e8", "source": "flat", "target": "fc1", "sourceHandle": "result", "targetHandle": "x"},
            {"id": "e9", "source": "fc1", "target": "fc2", "sourceHandle": "result", "targetHandle": "x"},
            {"id": "e10", "source": "fc2", "target": "fc3", "sourceHandle": "result", "targetHandle": "x"},
            {"id": "e11", "source": "fc3", "target": "output", "sourceHandle": "result", "targetHandle": "x"},
        ],
    }


def _composite_graph() -> dict:
    subgraph = {
        "nodes": [
            {"id": "sub_input", "data": {"nodeType": "input", "label": "Input", "params": {}}},
            {"id": "sub_fc", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 4, "out_features": 2, "bias": True}}},
            {"id": "sub_output", "data": {"nodeType": "output", "label": "Output", "params": {}}},
        ],
        "edges": [
            {"id": "se1", "source": "sub_input", "target": "sub_fc", "sourceHandle": "result", "targetHandle": "x"},
            {"id": "se2", "source": "sub_fc", "target": "sub_output", "sourceHandle": "result", "targetHandle": "x"},
        ],
    }
    return {
        "nodes": [
            {"id": "input", "data": {"nodeType": "input", "label": "Input", "params": {}}},
            {
                "id": "block",
                "data": {
                    "nodeType": "composite",
                    "label": "Reusable Block",
                    "params": {
                        "className": "ReusableLinear",
                        "subgraph": subgraph,
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


def _training_payload(csv_path: str) -> dict:
    return {
        "graph": _linear_graph(),
        "data_graph": {
            "nodes": [
                {
                    "id": "csv",
                    "data": {
                        "nodeType": "csv_source",
                        "label": "CSV",
                        "params": {
                            "path": csv_path,
                            "delimiter": ",",
                            "feature_columns": "f0,f1,f2,f3",
                            "label_column": "label",
                        },
                    },
                },
                {
                    "id": "loader",
                    "data": {
                        "nodeType": "dataloader",
                        "label": "DataLoader",
                        "params": {"batch_size": 2, "shuffle": False, "num_workers": 0, "pin_memory": False},
                    },
                },
                {"id": "out", "data": {"nodeType": "dataset_output", "label": "Output", "params": {"fields": "features,label"}}},
            ],
            "edges": [
                {"id": "d1", "source": "csv", "target": "loader"},
                {"id": "d2", "source": "loader", "target": "out"},
            ],
        },
        "bindings": [
            {"sourceGraph": "data", "sourceKey": "features", "target": "model_input", "targetKey": "features"},
            {"sourceGraph": "data", "sourceKey": "label", "target": "training_target", "targetKey": "target"},
        ],
        "training_config": {
            "taskType": "classification",
            "loss": {"type": "cross_entropy", "enabled": True, "params": {}},
            "optimizer": {"type": "adam", "enabled": True, "params": {"lr": 0.01}},
            "scheduler": {"type": "none", "enabled": False, "params": {}},
            "runtime": {"device": "cpu", "epochs": 1, "batchSize": 2, "numWorkers": 0, "gradClip": None},
        },
        "target_device": "cpu",
    }


class E2ESmokeTest(unittest.TestCase):
    """End-to-end smoke tests for ISA-216 refactoring."""

    @classmethod
    def setUpClass(cls):
        cls.backend_app = FastAPI()
        cls.backend_app.include_router(generate.router, prefix="/api")
        cls.backend_app.include_router(execute.router, prefix="/api")
        cls.backend_app.include_router(websocket.router)
        cls.client = TestClient(cls.backend_app)

    def test_01_frontend_backend_linear_graph_semantic_consistency(self):
        """Frontend local code and backend /api/generate produce semantically consistent Linear graph code."""
        graph = _linear_graph()

        # Backend generation
        backend_resp = self.client.post("/api/generate", json={"graph": graph})
        self.assertEqual(backend_resp.status_code, 200)
        backend_body = backend_resp.json()
        self.assertTrue(backend_body["success"], backend_body.get("warnings"))
        backend_code = backend_body["code"]

        # Frontend generation
        frontend_result = _run_frontend_generator("linear")
        frontend_code = frontend_result["code"]

        # Both must contain FlowHamsterModel
        self.assertIn("class FlowHamsterModel(nn.Module):", backend_code)
        self.assertIn("class FlowHamsterModel(nn.Module):", frontend_code)

        # Both must contain the same layer types with same params
        backend_layers = _extract_layers(backend_code)
        frontend_layers = _extract_layers(frontend_code)

        backend_types = [layer["type"] for layer in backend_layers]
        frontend_types = [layer["type"] for layer in frontend_layers]
        self.assertEqual(backend_types, frontend_types, "Layer types should match")

        # Check specific params appear in both
        self.assertTrue(any("in_features=4" in layer["args"] for layer in backend_layers))
        self.assertTrue(any("in_features=4" in layer["args"] for layer in frontend_layers))
        self.assertTrue(any("out_features=2" in layer["args"] for layer in backend_layers))
        self.assertTrue(any("out_features=2" in layer["args"] for layer in frontend_layers))

        # Both must have forward pass returning output
        self.assertIn("return", backend_code)
        self.assertIn("return", frontend_code)

    def test_02_frontend_backend_lenet5_graph_semantic_consistency(self):
        """Frontend local code and backend /api/generate produce semantically consistent LeNet-5 graph code."""
        graph = _lenet5_graph()

        backend_resp = self.client.post("/api/generate", json={"graph": graph})
        self.assertEqual(backend_resp.status_code, 200)
        backend_body = backend_resp.json()
        self.assertTrue(backend_body["success"], backend_body.get("warnings"))
        backend_code = backend_body["code"]

        frontend_result = _run_frontend_generator("lenet5")
        frontend_code = frontend_result["code"]

        # Both must contain FlowHamsterModel
        self.assertIn("class FlowHamsterModel(nn.Module):", backend_code)
        self.assertIn("class FlowHamsterModel(nn.Module):", frontend_code)

        # Both must contain Conv2d layers
        backend_layers = _extract_layers(backend_code)
        frontend_layers = _extract_layers(frontend_code)

        backend_conv = [l for l in backend_layers if l["type"] == "nn.Conv2d"]
        frontend_conv = [l for l in frontend_layers if l["type"] == "nn.Conv2d"]
        self.assertEqual(len(backend_conv), 2, "Backend should have 2 Conv2d layers")
        self.assertEqual(len(frontend_conv), 2, "Frontend should have 2 Conv2d layers")

        # Both must contain MaxPool2d
        self.assertTrue(any(l["type"] == "nn.MaxPool2d" for l in backend_layers))
        self.assertTrue(any(l["type"] == "nn.MaxPool2d" for l in frontend_layers))

        # Both must contain Flatten
        self.assertTrue(any(l["type"] == "nn.Flatten" for l in backend_layers))
        self.assertTrue(any(l["type"] == "nn.Flatten" for l in frontend_layers))

        # Both must contain 3 Linear layers
        backend_linear = [l for l in backend_layers if l["type"] == "nn.Linear"]
        frontend_linear = [l for l in frontend_layers if l["type"] == "nn.Linear"]
        self.assertEqual(len(backend_linear), 3)
        self.assertEqual(len(frontend_linear), 3)

    def test_03_websocket_code_matches_http_generate(self):
        """WS /ws/code returns the same code as HTTP /api/generate for the same graph."""
        graph = _linear_graph()

        http_body = self.client.post("/api/generate", json={"graph": graph}).json()
        with self.client.websocket_connect("/ws/code") as ws:
            ws.send_text(json.dumps({"graph": graph}))
            ws_body = json.loads(ws.receive_text())

        self.assertTrue(http_body["success"])
        self.assertEqual(ws_body["code"], http_body["code"])

    def test_04_graph_first_execute_runs_forward_and_training(self):
        """/api/execute with graph payload successfully runs forward pass and training loop."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            csv_path = tmp_dir / "train.csv"
            csv_path.write_text(
                "f0,f1,f2,f3,label\n"
                "0.0,0.1,0.2,0.3,0\n"
                "1.0,1.1,1.2,1.3,1\n"
                "0.4,0.5,0.6,0.7,0\n"
                "1.4,1.5,1.6,1.7,1\n",
                encoding="utf-8",
            )

            payload = _training_payload(str(csv_path))
            response = self.client.post("/api/execute", json=payload)

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertTrue(body["success"], body.get("error"))
            self.assertIn("epoch=1", body["output"])
            self.assertIn("step=1", body["output"])
            self.assertIn("batch_size=2", body["output"])
            self.assertIn("loss=", body["output"])
            self.assertIn("device=cpu", body["output"])

    def test_05_composite_subgraph_save_load_generate_execute(self):
        """Composite workflow: generate code, verify class emission, save and execute."""
        graph = _composite_graph()

        # Generate code
        generated = generate_python(graph=graph)
        self.assertIn("class ReusableLinear(nn.Module):", generated.sections.reusable_classes)
        self.assertIn("self.x_composite_1 = ReusableLinear()", generated.sections.model)
        self.assertIn("class FlowHamsterModel(nn.Module):", generated.code)

        # Save and execute the generated code
        # The default main block uses torch.randn(1, 3, 224, 224) which doesn't match
        # a Linear(in_features=4) composite. Replace with compatible input.
        executable_code = generated.code.replace(
            "x = torch.randn(1, 3, 224, 224, device=runtime_device)",
            "x = torch.randn(1, 4, device=runtime_device)"
        )
        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "composite_test.py"
            script_path.write_text(executable_code, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("output:", result.stdout)

        # Save / load via workflow router
        from backend.routers import workflows
        with tempfile.TemporaryDirectory() as tmp:
            workflows.WORKFLOWS_DIR = (Path(tmp) / "workflows").resolve()
            app = FastAPI()
            app.include_router(workflows.router, prefix="/api")
            client = TestClient(app)

            # Create workflow
            create_resp = client.post("/api/workflows", json={"name": "composite test", "description": "demo"})
            self.assertEqual(create_resp.status_code, 200)
            wf_id = create_resp.json()["id"]

            # Save document with composite graph
            doc = {"modelGraph": graph}
            update_resp = client.put(f"/api/workflows/{wf_id}/document", json=doc)
            self.assertEqual(update_resp.status_code, 200)

            # Load document
            loaded = client.get(f"/api/workflows/{wf_id}").json()
            self.assertEqual(loaded["document"]["modelGraph"], graph)

            # Re-generate from loaded graph
            loaded_graph = loaded["document"]["modelGraph"]
            regenerated = generate_python(graph=loaded_graph)
            self.assertIn("class ReusableLinear(nn.Module):", regenerated.sections.reusable_classes)
            self.assertIn("self.x_composite_1 = ReusableLinear()", regenerated.sections.model)

    def test_06_frontend_backend_composite_semantic_consistency(self):
        """Backend generates reusable composite class; frontend handles composite via custom registry."""
        graph = _composite_graph()

        backend_resp = self.client.post("/api/generate", json={"graph": graph})
        self.assertEqual(backend_resp.status_code, 200)
        backend_code = backend_resp.json()["code"]

        # Backend must contain the composite class and instantiate it
        self.assertIn("class ReusableLinear(nn.Module):", backend_code)
        self.assertIn("ReusableLinear()", backend_code)
        self.assertIn("class FlowHamsterModel(nn.Module):", backend_code)

        # Frontend composite handling uses customCompositeRegistry, which expects
        # pre-registered custom classes. The raw composite node format is backend-native.
        # We verify the frontend at least produces valid model code for the same graph.
        frontend_result = _run_frontend_generator("composite_linear")
        frontend_code = frontend_result["code"]
        self.assertIn("class FlowHamsterModel(nn.Module):", frontend_code)
        self.assertIn("def forward(self, x):", frontend_code)

    def test_07_frontend_backend_training_workflow_semantic_consistency(self):
        """Frontend and backend produce consistent code for a training workflow with data graph."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            csv_path = tmp_dir / "train.csv"
            csv_path.write_text(
                "f0,f1,f2,f3,label\n"
                "0.0,0.1,0.2,0.3,0\n"
                "1.0,1.1,1.2,1.3,1\n"
                "0.4,0.5,0.6,0.7,0\n"
                "1.4,1.5,1.6,1.7,1\n",
                encoding="utf-8",
            )

            payload = _training_payload(str(csv_path))
            backend_resp = self.client.post("/api/generate", json=payload)
            self.assertEqual(backend_resp.status_code, 200)
            backend_code = backend_resp.json()["code"]

            frontend_result = _run_frontend_generator("linear_with_training")
            frontend_code = frontend_result["code"]

            # Both must contain dataloader scaffold
            self.assertIn("build_flowhamster_dataloader", backend_code)
            self.assertIn("build_flowhamster_dataloader", frontend_code)

            # Both must contain training loop
            self.assertIn("for epoch in range", backend_code)
            self.assertIn("for epoch in range", frontend_code)

            # Both must contain optimizer and loss
            self.assertIn("optimizer", backend_code)
            self.assertIn("optimizer", frontend_code)
            self.assertIn("loss", backend_code)
            self.assertIn("loss", frontend_code)

            # Both must resolve bound inputs
            self.assertIn("resolve_bound_inputs", backend_code)
            self.assertIn("resolve_bound_inputs", frontend_code)

    def test_08_graph_first_forward_pass_preview(self):
        """/api/execute/forward returns tensor shapes for a simple CNN graph."""
        graph = {
            "nodes": [
                {"id": "input", "data": {"nodeType": "input", "label": "Input", "params": {"name": "image"}}},
                {"id": "conv", "data": {"nodeType": "conv2d", "label": "Conv2d", "params": {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1, "bias": True}}},
                {"id": "output", "data": {"nodeType": "output", "label": "Output", "params": {}}},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "conv", "sourceHandle": "result", "targetHandle": "x"},
                {"id": "e2", "source": "conv", "target": "output", "sourceHandle": "result", "targetHandle": "x"},
            ],
        }

        response = self.client.post("/api/execute/forward", json={
            "nodes": graph["nodes"],
            "edges": graph["edges"],
            "input_shape": [1, 3, 32, 32],
        })

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"], body.get("error"))
        outputs = body["outputs"]
        # execute_forward returns a dict keyed by node id
        self.assertIsInstance(outputs, dict)
        self.assertIn("conv", outputs)
        conv_output = outputs["conv"]
        self.assertEqual(conv_output.get("shape"), [1, 16, 32, 32])


if __name__ == "__main__":
    unittest.main()
