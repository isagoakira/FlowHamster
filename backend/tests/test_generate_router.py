import unittest
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers import generate


class GenerateRouterTest(unittest.TestCase):
    def test_generate_route_appends_dataflow_scaffold(self):
        app = FastAPI()
        app.include_router(generate.router, prefix="/api")
        payload = {
            "graph": {"nodes": [], "edges": []},
            "data_graph": {
                "nodes": [
                    {
                        "id": "loader",
                        "data": {
                            "nodeType": "dataloader",
                            "label": "DataLoader",
                            "params": {"batch_size": 2, "num_workers": 0},
                        },
                    },
                    {
                        "id": "out",
                        "data": {
                            "nodeType": "dataset_output",
                            "label": "Output",
                            "params": {"fields": "image,label"},
                        },
                    },
                ],
                "edges": [{"id": "e1", "source": "loader", "target": "out"}],
            },
            "bindings": [
                {
                    "id": "data:image->model_input:image",
                    "sourceGraph": "data",
                    "sourceKey": "image",
                    "target": "model_input",
                    "targetKey": "image",
                }
            ],
            "training_config": {"taskType": "classification"},
        }

        response = TestClient(app).post("/api/generate", json=payload)
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["success"])
        self.assertIn("def build_flowhamster_dataloader():", body["code"])
        self.assertIn("DATALOADER_CONFIG", body["code"])

    def test_generate_route_returns_executable_dataflow_training_loop(self):
        app = FastAPI()
        app.include_router(generate.router, prefix="/api")

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
            payload = {
                "graph": {
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
                },
                "data_graph": {
                    "nodes": [
                        {
                            "id": "csv",
                            "data": {
                                "nodeType": "csv_source",
                                "label": "CSV",
                                "params": {
                                    "path": str(csv_path),
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
                        {
                            "id": "out",
                            "data": {"nodeType": "dataset_output", "label": "Output", "params": {"fields": "features,label"}},
                        },
                    ],
                    "edges": [
                        {"id": "d1", "source": "csv", "target": "loader"},
                        {"id": "d2", "source": "loader", "target": "out"},
                    ],
                },
                "bindings": [
                    {"id": "data:features->model_input:features", "sourceGraph": "data", "sourceKey": "features", "target": "model_input", "targetKey": "features"},
                    {"id": "data:label->training_target:target", "sourceGraph": "data", "sourceKey": "label", "target": "training_target", "targetKey": "target"},
                ],
                "training_config": {
                    "taskType": "classification",
                    "loss": {"type": "cross_entropy", "enabled": True, "params": {}},
                    "optimizer": {"type": "adam", "enabled": True, "params": {"lr": 0.01}},
                    "scheduler": {"type": "none", "enabled": False, "params": {}},
                    "runtime": {"device": "cpu", "epochs": 1, "batchSize": 2, "numWorkers": 0, "gradClip": None},
                },
            }

            response = TestClient(app).post("/api/generate", json=payload)
            body = response.json()

            self.assertEqual(response.status_code, 200)
            self.assertTrue(body["success"], body.get("warnings"))
            code = body["code"]
            self.assertLess(code.index("def build_flowhamster_dataloader():"), code.index('if __name__ == "__main__":'))
            self.assertIn("loader = build_flowhamster_dataloader()", code)
            self.assertIn("for epoch in range(1):", code)

            script_path = tmp_dir / "generated_train.py"
            script_path.write_text(code, encoding="utf-8")
            result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True, timeout=60)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("epoch=1", result.stdout)
            self.assertIn("step=1", result.stdout)
            self.assertIn("batch_size=2", result.stdout)
            self.assertIn("loss=", result.stdout)


if __name__ == "__main__":
    unittest.main()
