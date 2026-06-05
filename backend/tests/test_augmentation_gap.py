import unittest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.routers import generate

class AugmentationGapTest(unittest.TestCase):
    def test_backend_generate_drops_augmentation_nodes_with_unknown_comment(self):
        """Demonstrate that backend dataflow_compiler does not handle augmentation nodes."""
        app = FastAPI()
        app.include_router(generate.router, prefix="/api")
        client = TestClient(app)

        payload = {
            "graph": {
                "nodes": [
                    {"id": "input", "data": {"nodeType": "input", "label": "Input", "params": {"name": "image"}}},
                    {"id": "conv", "data": {"nodeType": "conv2d", "label": "Conv2d", "params": {"in_channels": 3, "out_channels": 2, "kernel_size": 3, "padding": 1}}},
                    {"id": "output", "data": {"nodeType": "output", "label": "Output", "params": {}}},
                ],
                "edges": [
                    {"id": "e1", "source": "input", "target": "conv", "sourceHandle": "result", "targetHandle": "x"},
                    {"id": "e2", "source": "conv", "target": "output", "sourceHandle": "result", "targetHandle": "x"},
                ],
            },
            "data_graph": {
                "nodes": [
                    {"id": "folder", "data": {"nodeType": "folder_source", "label": "Folder", "params": {"path": "./data", "pattern": "*.jpg"}}},
                    {"id": "flip", "data": {"nodeType": "random_horizontal_flip", "label": "Random H-Flip", "params": {"p": 0.5}}},
                    {"id": "crop", "data": {"nodeType": "random_crop", "label": "Random Crop", "params": {"size": 224, "padding": 4}}},
                    {"id": "norm", "data": {"nodeType": "normalize", "label": "Normalize", "params": {"mean": "0.5,0.5,0.5", "std": "0.5,0.5,0.5"}}},
                    {"id": "loader", "data": {"nodeType": "dataloader", "label": "DataLoader", "params": {"batch_size": 2, "num_workers": 0}}},
                    {"id": "out", "data": {"nodeType": "dataset_output", "label": "Output", "params": {"fields": "image,label"}}},
                ],
                "edges": [
                    {"id": "d1", "source": "folder", "target": "flip"},
                    {"id": "d2", "source": "flip", "target": "crop"},
                    {"id": "d3", "source": "crop", "target": "norm"},
                    {"id": "d4", "source": "norm", "target": "loader"},
                    {"id": "d5", "source": "loader", "target": "out"},
                ],
            },
            "bindings": [
                {"sourceGraph": "data", "sourceKey": "image", "target": "model_input", "targetKey": "image"},
                {"sourceGraph": "data", "sourceKey": "label", "target": "training_target", "targetKey": "target"},
            ],
            "training_config": {"taskType": "classification"},
        }

        response = client.post("/api/generate", json=payload)
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["success"])
        # Backend compiler does not know random_horizontal_flip, random_crop, normalize
        # so it emits "# Unknown node type" comments for them.
        self.assertIn("# Unknown node type: random_horizontal_flip", body["code"])
        self.assertIn("# Unknown node type: random_crop", body["code"])
        self.assertIn("# Unknown node type: normalize", body["code"])

if __name__ == "__main__":
    unittest.main()
