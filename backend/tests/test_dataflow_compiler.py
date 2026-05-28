import unittest

from backend.services.dataflow_compiler import compile_dataflow


class DataflowCompilerTest(unittest.TestCase):
    def test_dataloader_node_builds_real_dataloader_scaffold(self):
        data_graph = {
            "nodes": [
                {
                    "id": "csv",
                    "data": {
                        "nodeType": "csv_source",
                        "label": "CSV",
                        "params": {
                            "path": "./data/train.csv",
                            "delimiter": ",",
                            "feature_columns": "f0,f2",
                            "label_column": "class_id",
                        },
                    },
                },
                {
                    "id": "loader",
                    "data": {
                        "nodeType": "dataloader",
                        "label": "DataLoader",
                        "params": {
                            "batch_size": 4,
                            "shuffle": "false",
                            "num_workers": 0,
                            "pin_memory": "false",
                            "drop_last": "true",
                        },
                    },
                },
                {
                    "id": "out",
                    "data": {
                        "nodeType": "dataset_output",
                        "label": "Output",
                        "params": {"fields": "features,label"},
                    },
                },
            ],
            "edges": [
                {"id": "d1", "source": "csv", "target": "loader"},
                {"id": "d2", "source": "loader", "target": "out"},
            ],
        }
        bindings = [
            {
                "id": "data:image->model_input:image",
                "sourceGraph": "data",
                "sourceKey": "image",
                "target": "model_input",
                "targetKey": "image",
            },
            {
                "id": "data:label->training_target:target",
                "sourceGraph": "data",
                "sourceKey": "label",
                "target": "training_target",
                "targetKey": "target",
            },
        ]

        compiled = compile_dataflow(data_graph=data_graph, bindings=bindings, training_config={"taskType": "classification"})

        self.assertTrue(compiled.has_workflow_runtime)
        self.assertIn("def build_flowhamster_dataloader():", compiled.python_scaffold)
        self.assertIn("torch.utils.data.DataLoader(", compiled.python_scaffold)
        self.assertIn("'batch_size': 4", compiled.python_scaffold)
        self.assertIn("'shuffle': False", compiled.python_scaffold)
        self.assertIn("'num_workers': 0", compiled.python_scaffold)
        self.assertIn("'pin_memory': False", compiled.python_scaffold)
        self.assertIn("'drop_last': True", compiled.python_scaffold)
        self.assertIn("_field_features", compiled.python_scaffold)
        self.assertIn("self._csv_feature_columns = ['f0', 'f2']", compiled.python_scaffold)
        self.assertIn("self._csv_label_column = 'class_id'", compiled.python_scaffold)
        self.assertIn("batch = next(iter(loader))", compiled.python_scaffold)

    def test_binding_mismatches_emit_actionable_warnings(self):
        model_graph = {
            "nodes": [
                {"id": "input", "data": {"nodeType": "input", "label": "Input", "params": {"name": "features"}}},
            ],
            "edges": [],
        }
        data_graph = {
            "nodes": [
                {
                    "id": "out",
                    "data": {
                        "nodeType": "dataset_output",
                        "label": "Output",
                        "params": {"fields": "features,label"},
                    },
                }
            ],
            "edges": [],
        }
        bindings = [
            {
                "id": "data:features->model_input:typo",
                "sourceGraph": "data",
                "sourceKey": "features",
                "target": "model_input",
                "targetKey": "typo",
            },
            {
                "id": "data:missing->training_target:target",
                "sourceGraph": "data",
                "sourceKey": "missing",
                "target": "training_target",
                "targetKey": "target",
            },
        ]

        compiled = compile_dataflow(model_graph=model_graph, data_graph=data_graph, bindings=bindings)

        self.assertIn("模型输入 features 尚未绑定数据字段", "\n".join(compiled.warnings))
        self.assertIn("绑定源字段 missing 未在 Dataset Output 中声明", "\n".join(compiled.warnings))


if __name__ == "__main__":
    unittest.main()
