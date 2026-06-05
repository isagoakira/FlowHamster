import unittest

from backend.services.codegen.emitters.evaluation import _gen_evaluation
from backend.services.codegen.emitters.forward import _gen_forward
from backend.services.codegen.emitters.module_init import _gen_init
from backend.services.codegen.emitters.training import (
    _gen_training,
    _gen_training_from_config_sections,
)
from backend.services.codegen.graph import NodeBlock, build_ast
from backend.services.codegen.sections import CodeSection, CodeWriter
from backend.services.unified_code_gen import UnifiedCodeGenerator


class CodegenEmitterTest(unittest.TestCase):
    def test_code_writer_renders_named_sections_in_order(self):
        writer = CodeWriter()
        writer.section("imports", ["import torch"])
        writer.section("model", ["class FlowHamsterModel:", "    pass"])

        self.assertEqual(
            writer.render(),
            "import torch\n\nclass FlowHamsterModel:\n    pass\n",
        )

    def test_graph_module_and_forward_emitters_cover_linear_relu_output(self):
        graph = {
            "nodes": [
                {"id": "input", "data": {"nodeType": "input", "params": {}}},
                {
                    "id": "fc",
                    "data": {
                        "nodeType": "linear",
                        "params": {"in_features": 4, "out_features": 2, "bias": False},
                    },
                },
                {"id": "act", "data": {"nodeType": "relu", "params": {}}},
                {"id": "out", "data": {"nodeType": "output", "params": {}}},
            ],
            "edges": [
                {"source": "input", "target": "fc", "sourceHandle": "result", "targetHandle": "x"},
                {"source": "fc", "target": "act", "sourceHandle": "result", "targetHandle": "x"},
                {"source": "act", "target": "out", "sourceHandle": "result", "targetHandle": "x"},
            ],
        }

        blocks = build_ast(graph)
        linear = next(block for block in blocks if block.op_type == "linear")
        relu = next(block for block in blocks if block.op_type == "relu")
        output = next(block for block in blocks if block.op_type == "output")

        self.assertEqual(
            _gen_init(linear),
            "self.x_linear_1 = nn.Linear(in_features=4, out_features=2, bias=False)",
        )
        self.assertEqual(_gen_init(relu), "self.x_relu_1 = nn.ReLU()")
        self.assertIn("x_relu_1 = self.x_relu_1(x_linear_1)", _gen_forward(relu, blocks))
        self.assertIn("return x_output_1", _gen_forward(output, blocks))

    def test_training_and_evaluation_emitters_cover_core_nodes(self):
        optimizer = NodeBlock(
            node_id="opt",
            op_type="adam",
            category="training",
            fields={"lr": 0.01},
            inputs={},
            output_var="x_adam_1",
        )
        metric = NodeBlock(
            node_id="metric",
            op_type="accuracy",
            category="evaluation",
            fields={"top_k": 1},
            inputs={},
            output_var="x_accuracy_1",
        )

        self.assertIn(
            "torch.optim.Adam(model.parameters(), lr=0.01",
            "\n".join(_gen_training(optimizer, [])),
        )
        self.assertIn("accuracy_score", "\n".join(_gen_evaluation(metric, [])))

        custom_classes, setup_lines, step_lines = _gen_training_from_config_sections(
            {
                "taskType": "classification",
                "loss": {"type": "cross_entropy", "enabled": True, "params": {}},
                "optimizer": {"type": "adam", "enabled": True, "params": {"lr": 0.001}},
                "scheduler": {"enabled": False},
                "runtime": {"gradClip": 1.0},
            }
        )

        self.assertTrue(any("flowhamster_prepare_loss_inputs" in line for line in custom_classes))
        setup_block = "\n".join(setup_lines)
        self.assertIn("loss_fn = nn.CrossEntropyLoss()", setup_block)
        self.assertIn("optimizer = torch.optim.Adam(model.parameters(), lr=0.001)", setup_block)
        self.assertIn("torch.nn.utils.clip_grad_norm_", "\n".join(step_lines))

    def test_unified_generator_facade_keeps_public_model_api(self):
        code = UnifiedCodeGenerator({"nodes": [], "edges": []}).generate_model()

        self.assertIn("class FlowHamsterModel(nn.Module):", code)
        self.assertIn("def forward(self, x):", code)
        self.assertIn("return x", code)
        self.assertIsInstance(CodeSection("empty").render(), str)


if __name__ == "__main__":
    unittest.main()
