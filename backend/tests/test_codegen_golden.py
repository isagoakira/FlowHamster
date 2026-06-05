import ast
import base64
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers import generate
from backend.services.codegen_facade import generate_full_code, generate_model_code


def node(node_id: str, node_type: str, params: dict | None = None) -> dict:
    return {
        "id": node_id,
        "data": {"nodeType": node_type, "label": node_type, "params": params or {}},
    }


def edge(
    source: str,
    target: str,
    target_handle: str = "x",
    source_handle: str = "result",
) -> dict:
    return {
        "id": f"{source}_{target}_{target_handle}",
        "source": source,
        "target": target,
        "sourceHandle": source_handle,
        "targetHandle": target_handle,
    }


def linear_classification_graph() -> dict:
    return {
        "nodes": [
            node("input", "input", {"name": "features"}),
            node("fc1", "linear", {"in_features": 4, "out_features": 8, "bias": True}),
            node("relu", "relu"),
            node("fc2", "linear", {"in_features": 8, "out_features": 2, "bias": True}),
            node("output", "output"),
        ],
        "edges": [
            edge("input", "fc1"),
            edge("fc1", "relu"),
            edge("relu", "fc2"),
            edge("fc2", "output"),
        ],
    }


def executable_cnn_graph() -> dict:
    return {
        "nodes": [
            node("input", "input", {"name": "image"}),
            node("conv", "conv2d", {"in_channels": 3, "out_channels": 4, "kernel_size": 3, "padding": 1, "bias": True}),
            node("relu", "relu"),
            node("pool", "globalavgpool"),
            node("flatten", "flatten", {"start_dim": 1}),
            node("fc", "linear", {"in_features": 4, "out_features": 2, "bias": True}),
            node("output", "output"),
        ],
        "edges": [
            edge("input", "conv"),
            edge("conv", "relu"),
            edge("relu", "pool"),
            edge("pool", "flatten"),
            edge("flatten", "fc"),
            edge("fc", "output"),
        ],
    }


def transformer_mamba_graph() -> dict:
    return {
        "nodes": [
            node("input", "input", {"name": "tokens"}),
            node("encoder", "transformerencoder", {"embed_dim": 16, "num_heads": 4, "num_layers": 1, "dim_feedforward": 32}),
            node("mamba", "mamba", {"d_model": 16, "d_state": 4, "d_conv": 2, "n_layers": 1}),
            node("output", "output"),
        ],
        "edges": [
            edge("input", "encoder", "src"),
            edge("encoder", "mamba"),
            edge("mamba", "output"),
        ],
    }


def multi_output_graph() -> dict:
    return {
        "nodes": [
            node("input", "input", {"name": "image"}),
            node("conv_a", "conv2d", {"in_channels": 3, "out_channels": 2, "kernel_size": 1, "bias": True}),
            node("conv_b", "conv2d", {"in_channels": 3, "out_channels": 4, "kernel_size": 1, "bias": True}),
            node("out_a", "output"),
            node("out_b", "output"),
        ],
        "edges": [
            edge("input", "conv_a"),
            edge("input", "conv_b"),
            edge("conv_a", "out_a"),
            edge("conv_b", "out_b"),
        ],
    }


def data_graph() -> dict:
    return {
        "nodes": [
            node("csv", "csv_source", {"path": "./data/train.csv", "feature_columns": "f0,f1,f2,f3", "label_column": "label"}),
            node("loader", "dataloader", {"batch_size": 2, "shuffle": False, "num_workers": 0, "pin_memory": False}),
            node("out", "dataset_output", {"fields": "features,label"}),
        ],
        "edges": [
            {"id": "d1", "source": "csv", "target": "loader"},
            {"id": "d2", "source": "loader", "target": "out"},
        ],
    }


def bindings() -> list[dict]:
    return [
        {"sourceGraph": "data", "sourceKey": "features", "target": "model_input", "targetKey": "features"},
        {"sourceGraph": "data", "sourceKey": "label", "target": "training_target", "targetKey": "target"},
    ]


def training_config() -> dict:
    return {
        "taskType": "classification",
        "loss": {"type": "cross_entropy", "enabled": True, "params": {}},
        "optimizer": {"type": "adamw", "enabled": True, "params": {"lr": 0.001}},
        "scheduler": {"type": "none", "enabled": False, "params": {}},
        "runtime": {"device": "cpu", "epochs": 1, "batchSize": 2, "numWorkers": 0, "gradClip": None},
    }


class CodegenGoldenTest(unittest.TestCase):
    def assert_valid_script(self, code: str) -> None:
        ast.parse(code)
        self.assertIn("import torch", code)
        self.assertIn("import torch.nn as nn", code)
        self.assertIn("class FlowHamsterModel(nn.Module):", code)
        self.assertIn("def forward(self, x):", code)
        self.assertIn('if __name__ == "__main__":', code)

    def test_golden_linear_classification_structure(self):
        code = generate_full_code(linear_classification_graph()).code

        self.assert_valid_script(code)
        self.assertIn("nn.Linear(in_features=4, out_features=8", code)
        self.assertIn("nn.ReLU()", code)
        self.assertIn("primary_output = output", code)

    def test_golden_cnn_script_executes(self):
        code = generate_full_code(executable_cnn_graph()).code
        self.assert_valid_script(code)

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "cnn.py"
            script.write_text(code, encoding="utf-8")
            result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("output: torch.Size([1, 2])", result.stdout)

    def test_golden_transformer_mamba_structure(self):
        code = generate_full_code(transformer_mamba_graph()).code

        self.assert_valid_script(code)
        self.assertIn("nn.TransformerEncoder", code)
        self.assertIn("class Mamba(nn.Module):", code)
        self.assertIn("dt_rank='auto'", code)

    def test_golden_multi_output_returns_tuple(self):
        code = generate_full_code(multi_output_graph()).code

        self.assert_valid_script(code)
        self.assertIn("return (", code)
        self.assertIn("primary_output = output[0]", code)

    def test_golden_data_graph_bindings_scaffold(self):
        generated = generate_full_code(
            linear_classification_graph(),
            training_config=training_config(),
            data_graph=data_graph(),
            bindings=bindings(),
        )

        self.assert_valid_script(generated.code)
        self.assertEqual(generated.warnings, [])
        self.assertIn("def build_flowhamster_dataloader():", generated.code)
        self.assertIn("BOUND_MODEL_INPUTS", generated.code)
        self.assertIn("loader = build_flowhamster_dataloader()", generated.code)

    def test_golden_training_config_block(self):
        code = generate_full_code(executable_cnn_graph(), training_config=training_config()).code

        self.assert_valid_script(code)
        self.assertIn("flowhamster_prepare_loss_inputs", code)
        self.assertIn("torch.optim.AdamW", code)
        self.assertIn("loss.backward()", code)

    def test_golden_notebook_export_matches_generate_code_cells(self):
        try:
            import nbformat
            from backend.routers import export
        except ModuleNotFoundError as exc:
            self.skipTest(f"notebook export dependency missing: {exc}")

        app = FastAPI()
        app.include_router(generate.router, prefix="/api")
        app.include_router(export.router, prefix="/api")
        client = TestClient(app)
        payload = {
            "graph": executable_cnn_graph(),
            "training_config": training_config(),
            "data_graph": data_graph(),
            "bindings": bindings(),
        }

        generated = client.post("/api/generate", json=payload).json()
        exported = client.post("/api/export-notebook", json=payload).json()

        self.assertTrue(generated["success"], generated.get("warnings"))
        self.assertTrue(exported["success"])
        notebook_json = base64.b64decode(exported["content"]).decode()
        notebook = nbformat.reads(notebook_json, as_version=4)
        notebook_sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]

        self.assertEqual(notebook_sources, export._split_code_into_cells(generated["code"]))

    def test_facade_model_and_full_entry_points(self):
        model_code = generate_model_code(executable_cnn_graph())
        full_code = generate_full_code(executable_cnn_graph()).code

        ast.parse(model_code)
        self.assertIn("class FlowHamsterModel(nn.Module):", model_code)
        self.assertNotIn('if __name__ == "__main__":', model_code)
        self.assertIn('if __name__ == "__main__":', full_code)

    def test_user_visible_routes_do_not_import_ast_core(self):
        backend_dir = Path(__file__).resolve().parents[1]
        route_files = [
            backend_dir / "routers" / "generate.py",
            backend_dir / "routers" / "execute.py",
            backend_dir / "routers" / "export.py",
            backend_dir / "routers" / "websocket.py",
        ]

        for route_file in route_files:
            with self.subTest(route=route_file.name):
                content = route_file.read_text(encoding="utf-8")
                self.assertNotIn("backend.services.ast_core", content)
                self.assertIn("codegen_facade", content)


if __name__ == "__main__":
    unittest.main()
