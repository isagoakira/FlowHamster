import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from backend.services.unified_code_gen import UnifiedCodeGenerator


def _node(node_id, node_type, params=None):
    return {
        "id": node_id,
        "data": {
            "nodeType": node_type,
            "label": node_type,
            "params": params or {},
        },
    }


def _edge(edge_id, source, target, target_handle="x"):
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "sourceHandle": "result",
        "targetHandle": target_handle,
    }


def _run_generated_code(code: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "generated.py"
        script_path.write_text(code, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )


class UnifiedCodeGeneratorTest(unittest.TestCase):
    def test_tensor_operation_nodes_generate_executable_forward_code(self):
        graph = {
            "nodes": [
                _node("input", "input"),
                _node("permute", "permute", {"dims": "0, 2, 3, 1"}),
                _node("squeeze", "squeeze", {"dim": 0}),
                _node("expand", "expand", {"shape": "2, 224, 224, 3"}),
                _node("output", "output"),
            ],
            "edges": [
                _edge("e1", "input", "permute"),
                _edge("e2", "permute", "squeeze"),
                _edge("e3", "squeeze", "expand"),
                _edge("e4", "expand", "output"),
            ],
        }

        code = UnifiedCodeGenerator(graph, {"training_nodes": False}).generate_full()
        result = _run_generated_code(code)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(".permute(0, 2, 3, 1)", code)
        self.assertIn(".squeeze(0)", code)
        self.assertIn(".expand(2, 224, 224, 3)", code)
        self.assertIn("torch.Size([2, 224, 224, 3])", result.stdout)

    def test_focal_and_label_smoothing_training_nodes_generate_executable_setup(self):
        graph = {
            "nodes": [
                _node("input", "input"),
                _node("output", "output"),
                _node("focal", "focalloss", {"alpha": 0.5, "gamma": 1.5}),
                _node("smooth", "labelsmoothing", {"smoothing": 0.2}),
            ],
            "edges": [_edge("e1", "input", "output")],
        }

        code = UnifiedCodeGenerator(graph, {"training_nodes": True}).generate_full()
        result = _run_generated_code(code)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("class FocalLoss(nn.Module):", code)
        self.assertIn("criterion_x_focalloss_1 = FocalLoss(alpha=0.5, gamma=1.5)", code)
        self.assertIn("criterion_x_labelsmoothing_1 = nn.CrossEntropyLoss(label_smoothing=0.2)", code)
        self.assertNotIn("self.x_focalloss_1", code)
        self.assertNotIn("self.x_labelsmoothing_1", code)


if __name__ == "__main__":
    unittest.main()
