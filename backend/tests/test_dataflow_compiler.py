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

    def test_advanced_augmentation_nodes_generate_code(self):
        data_graph = {
            "nodes": [
                {
                    "id": "folder",
                    "data": {
                        "nodeType": "folder_source",
                        "label": "Images",
                        "params": {"path": "./data/images", "pattern": "*.jpg"},
                    },
                },
                {
                    "id": "mixup",
                    "data": {
                        "nodeType": "mixup",
                        "label": "MixUp",
                        "params": {"alpha": 0.2},
                    },
                },
                {
                    "id": "cutmix",
                    "data": {
                        "nodeType": "cutmix",
                        "label": "CutMix",
                        "params": {"alpha": 1.0},
                    },
                },
                {
                    "id": "cutout",
                    "data": {
                        "nodeType": "cutout",
                        "label": "CutOut",
                        "params": {"hole_size": 16},
                    },
                },
                {
                    "id": "posterize",
                    "data": {
                        "nodeType": "posterize",
                        "label": "Posterize",
                        "params": {"bits": 4},
                    },
                },
                {
                    "id": "solarize",
                    "data": {
                        "nodeType": "solarize",
                        "label": "Solarize",
                        "params": {"threshold": 128},
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
            "edges": [
                {"id": "e1", "source": "folder", "target": "mixup"},
                {"id": "e2", "source": "mixup", "target": "cutmix"},
                {"id": "e3", "source": "cutmix", "target": "cutout"},
                {"id": "e4", "source": "cutout", "target": "posterize"},
                {"id": "e5", "source": "posterize", "target": "solarize"},
                {"id": "e6", "source": "solarize", "target": "out"},
            ],
        }
        compiled = compile_dataflow(data_graph=data_graph)
        scaffold = compiled.python_scaffold
        self.assertIn("# MixUp", scaffold)
        self.assertIn("# CutMix", scaffold)
        self.assertIn("# CutOut", scaffold)
        self.assertIn("# Posterize", scaffold)
        self.assertIn("# Solarize", scaffold)
        self.assertIn("torch.distributions.Beta", scaffold)
        self.assertIn("F.posterize", scaffold)
        self.assertIn("F.solarize", scaffold)

    def test_feature_engineering_nodes_generate_code(self):
        data_graph = {
            "nodes": [
                {
                    "id": "csv",
                    "data": {
                        "nodeType": "csv_source",
                        "label": "CSV",
                        "params": {"path": "./data/train.csv", "feature_columns": "f0,f1", "label_column": "label"},
                    },
                },
                {
                    "id": "scaler",
                    "data": {
                        "nodeType": "standard_scaler",
                        "label": "Standard Scaler",
                        "params": {"with_mean": True, "with_std": True},
                    },
                },
                {
                    "id": "mm_scaler",
                    "data": {
                        "nodeType": "minmax_scaler",
                        "label": "MinMax Scaler",
                        "params": {"feature_range": "0,1"},
                    },
                },
                {
                    "id": "pca",
                    "data": {
                        "nodeType": "pca",
                        "label": "PCA",
                        "params": {"n_components": 2},
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
                {"id": "e1", "source": "csv", "target": "scaler"},
                {"id": "e2", "source": "scaler", "target": "mm_scaler"},
                {"id": "e3", "source": "mm_scaler", "target": "pca"},
                {"id": "e4", "source": "pca", "target": "out"},
            ],
        }
        compiled = compile_dataflow(data_graph=data_graph)
        scaffold = compiled.python_scaffold
        self.assertIn("# Standard Scaler", scaffold)
        self.assertIn("# MinMax Scaler", scaffold)
        self.assertIn("# PCA", scaffold)
        self.assertIn("self._scaler_mean_", scaffold)
        self.assertIn("self._minmax_min_", scaffold)
        self.assertIn("self._pca_n_components_", scaffold)

    def test_multi_source_nodes_generate_code(self):
        data_graph = {
            "nodes": [
                {
                    "id": "zip",
                    "data": {
                        "nodeType": "zip_datasets",
                        "label": "Zip",
                        "params": {},
                    },
                },
                {
                    "id": "interleave",
                    "data": {
                        "nodeType": "interleave_datasets",
                        "label": "Interleave",
                        "params": {"cycle_length": 2},
                    },
                },
                {
                    "id": "sample",
                    "data": {
                        "nodeType": "sample_from_datasets",
                        "label": "Sample",
                        "params": {"weights": "0.5,0.5"},
                    },
                },
                {
                    "id": "out",
                    "data": {
                        "nodeType": "dataset_output",
                        "label": "Output",
                        "params": {"fields": "image"},
                    },
                },
            ],
            "edges": [
                {"id": "e1", "source": "zip", "target": "interleave"},
                {"id": "e2", "source": "interleave", "target": "sample"},
                {"id": "e3", "source": "sample", "target": "out"},
            ],
        }
        compiled = compile_dataflow(data_graph=data_graph)
        scaffold = compiled.python_scaffold
        self.assertIn("# Zip Datasets", scaffold)
        self.assertIn("# Interleave Datasets", scaffold)
        self.assertIn("# Sample From Datasets", scaffold)

    def test_existing_augmentation_nodes_generate_code(self):
        data_graph = {
            "nodes": [
                {
                    "id": "folder",
                    "data": {
                        "nodeType": "folder_source",
                        "label": "Images",
                        "params": {"path": "./data/images", "pattern": "*.jpg"},
                    },
                },
                {
                    "id": "hflip",
                    "data": {
                        "nodeType": "random_horizontal_flip",
                        "label": "H-Flip",
                        "params": {"p": 0.5},
                    },
                },
                {
                    "id": "vflip",
                    "data": {
                        "nodeType": "random_vertical_flip",
                        "label": "V-Flip",
                        "params": {"p": 0.5},
                    },
                },
                {
                    "id": "rcrop",
                    "data": {
                        "nodeType": "random_crop",
                        "label": "Random Crop",
                        "params": {"size": 224, "padding": 4},
                    },
                },
                {
                    "id": "rrot",
                    "data": {
                        "nodeType": "random_rotation",
                        "label": "Random Rotation",
                        "params": {"degrees": 15},
                    },
                },
                {
                    "id": "cjitter",
                    "data": {
                        "nodeType": "color_jitter",
                        "label": "Color Jitter",
                        "params": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
                    },
                },
                {
                    "id": "rerase",
                    "data": {
                        "nodeType": "random_erasing",
                        "label": "Random Erasing",
                        "params": {"p": 0.5, "scale": "0.02,0.33", "ratio": "0.3,3.3"},
                    },
                },
                {
                    "id": "gblur",
                    "data": {
                        "nodeType": "gaussian_blur",
                        "label": "Gaussian Blur",
                        "params": {"kernel_size": 5, "sigma": "1.0,2.0"},
                    },
                },
                {
                    "id": "gray",
                    "data": {
                        "nodeType": "grayscale",
                        "label": "Grayscale",
                        "params": {},
                    },
                },
                {
                    "id": "out",
                    "data": {
                        "nodeType": "dataset_output",
                        "label": "Output",
                        "params": {"fields": "image"},
                    },
                },
            ],
            "edges": [
                {"id": "e1", "source": "folder", "target": "hflip"},
                {"id": "e2", "source": "hflip", "target": "vflip"},
                {"id": "e3", "source": "vflip", "target": "rcrop"},
                {"id": "e4", "source": "rcrop", "target": "rrot"},
                {"id": "e5", "source": "rrot", "target": "cjitter"},
                {"id": "e6", "source": "cjitter", "target": "rerase"},
                {"id": "e7", "source": "rerase", "target": "gblur"},
                {"id": "e8", "source": "gblur", "target": "gray"},
                {"id": "e9", "source": "gray", "target": "out"},
            ],
        }
        compiled = compile_dataflow(data_graph=data_graph)
        scaffold = compiled.python_scaffold
        self.assertIn("# Random Horizontal Flip", scaffold)
        self.assertIn("# Random Vertical Flip", scaffold)
        self.assertIn("# Random Crop", scaffold)
        self.assertIn("# Random Rotation", scaffold)
        self.assertIn("# Color Jitter", scaffold)
        self.assertIn("# Random Erasing", scaffold)
        self.assertIn("# Gaussian Blur", scaffold)
        self.assertIn("# Grayscale", scaffold)
        self.assertNotIn("Unknown node type", scaffold)

    def test_new_feature_engineering_nodes_generate_code(self):
        data_graph = {
            "nodes": [
                {
                    "id": "csv",
                    "data": {
                        "nodeType": "csv_source",
                        "label": "CSV",
                        "params": {"path": "./data/train.csv", "feature_columns": "f0,f1", "label_column": "label"},
                    },
                },
                {
                    "id": "norm",
                    "data": {
                        "nodeType": "normalize_features",
                        "label": "Normalize Features",
                        "params": {"mean": "0,0", "std": "1,1"},
                    },
                },
                {
                    "id": "fill",
                    "data": {
                        "nodeType": "fill_missing_values",
                        "label": "Fill Missing",
                        "params": {"strategy": "mean", "fill_value": 0},
                    },
                },
                {
                    "id": "ohe",
                    "data": {
                        "nodeType": "one_hot_encode",
                        "label": "One-Hot",
                        "params": {"columns": "category", "num_classes": "3"},
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
                {"id": "e1", "source": "csv", "target": "norm"},
                {"id": "e2", "source": "norm", "target": "fill"},
                {"id": "e3", "source": "fill", "target": "ohe"},
                {"id": "e4", "source": "ohe", "target": "out"},
            ],
        }
        compiled = compile_dataflow(data_graph=data_graph)
        scaffold = compiled.python_scaffold
        self.assertIn("# Normalize Features", scaffold)
        self.assertIn("# Fill Missing Values", scaffold)
        self.assertIn("# One-Hot Encode", scaffold)
        self.assertIn("torch.where(_mask", scaffold)
        self.assertNotIn("Unknown node type", scaffold)

    def test_mixup_cutmix_label_blending(self):
        data_graph = {
            "nodes": [
                {
                    "id": "folder",
                    "data": {
                        "nodeType": "folder_source",
                        "label": "Images",
                        "params": {"path": "./data/images", "pattern": "*.jpg"},
                    },
                },
                {
                    "id": "mixup",
                    "data": {
                        "nodeType": "mixup",
                        "label": "MixUp",
                        "params": {"alpha": 0.2},
                    },
                },
                {
                    "id": "cutmix",
                    "data": {
                        "nodeType": "cutmix",
                        "label": "CutMix",
                        "params": {"alpha": 1.0},
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
            "edges": [
                {"id": "e1", "source": "folder", "target": "mixup"},
                {"id": "e2", "source": "mixup", "target": "cutmix"},
                {"id": "e3", "source": "cutmix", "target": "out"},
            ],
        }
        compiled = compile_dataflow(data_graph=data_graph)
        scaffold = compiled.python_scaffold
        # Verify label blending exists for both mixup and cutmix
        self.assertIn("_field_label = _lam * _field_label + (1 - _lam) * _sample2['label']", scaffold)
        self.assertIn("torch.distributions.Beta", scaffold)


if __name__ == "__main__":
    unittest.main()
