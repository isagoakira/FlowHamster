import unittest
import tempfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers import execute


class ExecuteRouterTest(unittest.TestCase):
    def test_execute_route_runs_submitted_code(self):
        app = FastAPI()
        app.include_router(execute.router, prefix="/api")
        client = TestClient(app)

        response = client.post(
            "/api/execute",
            json={"code": "print('flowhamster-execute-ok')", "target_device": "cpu"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertIn("flowhamster-execute-ok", body["output"])
        self.assertIsNone(body["error"])

    def test_execute_route_runs_generated_dataflow_training_loop(self):
        app = FastAPI()
        app.include_router(execute.router, prefix="/api")
        client = TestClient(app)

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

            response = client.post("/api/execute", json=payload)

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertTrue(body["success"], body.get("error"))
            self.assertIn("epoch=1", body["output"])
            self.assertIn("step=1", body["output"])
            self.assertIn("batch_size=2", body["output"])
            self.assertIn("loss=", body["output"])
            self.assertIn("device=cpu", body["output"])

    def test_execute_route_pads_missing_and_non_numeric_csv_features(self):
        app = FastAPI()
        app.include_router(execute.router, prefix="/api")
        client = TestClient(app)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            csv_path = tmp_dir / "ragged.csv"
            csv_path.write_text(
                "f0,f1,label\n"
                "0.0,bad,0\n"
                "1.0,1.1,1\n",
                encoding="utf-8",
            )

            payload = {
                "graph": {
                    "nodes": [
                        {"id": "input", "data": {"nodeType": "input", "label": "Input", "params": {"name": "features"}}},
                        {"id": "fc", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 4, "out_features": 2, "bias": True}}},
                        {"id": "output", "data": {"nodeType": "output", "label": "Output", "params": {}}},
                    ],
                    "edges": [
                        {"id": "e1", "source": "input", "target": "fc", "sourceHandle": "result", "targetHandle": "x"},
                        {"id": "e2", "source": "fc", "target": "output", "sourceHandle": "result", "targetHandle": "x"},
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
                        {"id": "loader", "data": {"nodeType": "dataloader", "label": "DataLoader", "params": {"batch_size": 2, "shuffle": False, "num_workers": 0}}},
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
            }

            response = client.post("/api/execute", json=payload)

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertTrue(body["success"], body.get("error"))
            self.assertIn("output=(2, 2)", body["output"])
            self.assertIn("loss=", body["output"])

    def test_execute_route_uses_fallback_batch_for_empty_csv_and_missing_label(self):
        app = FastAPI()
        app.include_router(execute.router, prefix="/api")
        client = TestClient(app)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            csv_path = tmp_dir / "empty.csv"
            csv_path.write_text("f0,f1,f2,f3\n", encoding="utf-8")

            payload = {
                "graph": {
                    "nodes": [
                        {"id": "input", "data": {"nodeType": "input", "label": "Input", "params": {"name": "features"}}},
                        {"id": "fc", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 4, "out_features": 2, "bias": True}}},
                        {"id": "output", "data": {"nodeType": "output", "label": "Output", "params": {}}},
                    ],
                    "edges": [
                        {"id": "e1", "source": "input", "target": "fc", "sourceHandle": "result", "targetHandle": "x"},
                        {"id": "e2", "source": "fc", "target": "output", "sourceHandle": "result", "targetHandle": "x"},
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
                        {"id": "loader", "data": {"nodeType": "dataloader", "label": "DataLoader", "params": {"batch_size": 2, "shuffle": False, "num_workers": 0}}},
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
            }

            response = client.post("/api/execute", json=payload)

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertTrue(body["success"], body.get("error"))
            self.assertIn("epoch=1", body["output"])
            self.assertIn("batch_size=1", body["output"])
            self.assertIn("output=(1, 2)", body["output"])
            self.assertIn("loss=", body["output"])

    def test_execute_route_uses_available_bound_tensor_when_model_input_key_is_mismatched(self):
        app = FastAPI()
        app.include_router(execute.router, prefix="/api")
        client = TestClient(app)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            csv_path = tmp_dir / "train.csv"
            csv_path.write_text(
                "f0,f1,f2,f3,label\n"
                "0.0,0.1,0.2,0.3,0\n"
                "1.0,1.1,1.2,1.3,1\n",
                encoding="utf-8",
            )

            payload = {
                "graph": {
                    "nodes": [
                        {"id": "input", "data": {"nodeType": "input", "label": "Input", "params": {"name": "features"}}},
                        {"id": "fc", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 4, "out_features": 2, "bias": True}}},
                        {"id": "output", "data": {"nodeType": "output", "label": "Output", "params": {}}},
                    ],
                    "edges": [
                        {"id": "e1", "source": "input", "target": "fc", "sourceHandle": "result", "targetHandle": "x"},
                        {"id": "e2", "source": "fc", "target": "output", "sourceHandle": "result", "targetHandle": "x"},
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
                        {"id": "loader", "data": {"nodeType": "dataloader", "label": "DataLoader", "params": {"batch_size": 2, "shuffle": False, "num_workers": 0}}},
                        {"id": "out", "data": {"nodeType": "dataset_output", "label": "Output", "params": {"fields": "features,label"}}},
                    ],
                    "edges": [
                        {"id": "d1", "source": "csv", "target": "loader"},
                        {"id": "d2", "source": "loader", "target": "out"},
                    ],
                },
                "bindings": [
                    {"sourceGraph": "data", "sourceKey": "features", "target": "model_input", "targetKey": "typo"},
                    {"sourceGraph": "data", "sourceKey": "label", "target": "training_target", "targetKey": "target"},
                ],
                "training_config": {
                    "taskType": "classification",
                    "loss": {"type": "cross_entropy", "enabled": True, "params": {}},
                    "optimizer": {"type": "adam", "enabled": True, "params": {"lr": 0.01}},
                    "scheduler": {"type": "none", "enabled": False, "params": {}},
                    "runtime": {"device": "cpu", "epochs": 1, "batchSize": 2, "numWorkers": 0, "gradClip": None},
                },
            }

            response = client.post("/api/execute", json=payload)

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertTrue(body["success"], body.get("error"))
            self.assertIn("output=(2, 2)", body["output"])
            self.assertIn("loss=", body["output"])

    def test_execute_route_pools_spatial_logits_for_classification_labels(self):
        app = FastAPI()
        app.include_router(execute.router, prefix="/api")
        client = TestClient(app)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            csv_path = tmp_dir / "train.csv"
            csv_path.write_text(
                "f0,f1,f2,f3,label\n"
                "0.0,0.1,0.2,0.3,0\n"
                "1.0,1.1,1.2,1.3,1\n",
                encoding="utf-8",
            )

            payload = {
                "graph": {
                    "nodes": [
                        {"id": "input", "data": {"nodeType": "input", "label": "Input", "params": {"name": "image"}}},
                        {
                            "id": "conv",
                            "data": {
                                "nodeType": "conv2d",
                                "label": "Conv2d",
                                "params": {"in_channels": 3, "out_channels": 2, "kernel_size": 3, "stride": 1, "padding": 1, "bias": True},
                            },
                        },
                        {"id": "output", "data": {"nodeType": "output", "label": "Output", "params": {}}},
                    ],
                    "edges": [
                        {"id": "e1", "source": "input", "target": "conv", "sourceHandle": "result", "targetHandle": "x"},
                        {"id": "e2", "source": "conv", "target": "output", "sourceHandle": "result", "targetHandle": "x"},
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
                        {"id": "loader", "data": {"nodeType": "dataloader", "label": "DataLoader", "params": {"batch_size": 2, "shuffle": False, "num_workers": 0}}},
                        {"id": "out", "data": {"nodeType": "dataset_output", "label": "Output", "params": {"fields": "features,label"}}},
                    ],
                    "edges": [
                        {"id": "d1", "source": "csv", "target": "loader"},
                        {"id": "d2", "source": "loader", "target": "out"},
                    ],
                },
                "bindings": [
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

            response = client.post("/api/execute", json=payload)

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertTrue(body["success"], body.get("error"))
            self.assertIn("output=(2, 2, 224, 224)", body["output"])
            self.assertIn("loss=", body["output"])

    def test_execute_route_returns_api_error_when_code_and_graph_are_missing(self):
        app = FastAPI()
        app.include_router(execute.router, prefix="/api")
        client = TestClient(app)

        response = client.post("/api/execute", json={"target_device": "cpu"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["output"], "")
        self.assertIn("Either code or graph must be provided", body["error"])

    def test_execute_route_returns_error_for_invalid_graph_structure(self):
        app = FastAPI()
        app.include_router(execute.router, prefix="/api")
        client = TestClient(app)

        response = client.post(
            "/api/execute",
            json={
                "graph": {
                    "nodes": "not-a-node-list",
                    "edges": [],
                },
                "target_device": "cpu",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["output"], "")
        self.assertIn("string indices must be integers", body["error"])

    def test_execute_route_returns_readable_pytorch_shape_error(self):
        app = FastAPI()
        app.include_router(execute.router, prefix="/api")
        client = TestClient(app)

        response = client.post(
            "/api/execute",
            json={
                "graph": {
                    "nodes": [
                        {"id": "input", "data": {"nodeType": "input", "label": "Input", "params": {"name": "features"}}},
                        {"id": "fc", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 4, "out_features": 2, "bias": True}}},
                        {"id": "output", "data": {"nodeType": "output", "label": "Output", "params": {}}},
                    ],
                    "edges": [
                        {"id": "e1", "source": "input", "target": "fc", "sourceHandle": "result", "targetHandle": "x"},
                        {"id": "e2", "source": "fc", "target": "output", "sourceHandle": "result", "targetHandle": "x"},
                    ],
                },
                "target_device": "cpu",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertIn("RuntimeError", body["output"])
        self.assertIn("mat1 and mat2 shapes cannot be multiplied", body["error"])


if __name__ == "__main__":
    unittest.main()
