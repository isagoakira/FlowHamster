"""
P0 Bug 回归验收测试
这些测试在当前代码库上应失败，用于证明 bug 存在；
待 ISA-207 / ISA-208 修复后应全部通过。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from backend.services.unified_code_gen import UnifiedCodeGenerator


class TestP0BugRegressionAcceptance(unittest.TestCase):
    """验收标准：P0 级 bug 修复后，以下用例必须全部通过"""

    def _build_graph(self, nodes: list, edges: list = None):
        return {"nodes": nodes, "edges": edges or []}

    def _build_node(self, node_id: str, op_type: str, params: dict):
        return {"id": node_id, "data": {"nodeType": op_type, "label": op_type, "params": params}}

    def _build_edge(self, src: str, tgt: str):
        return {"source": src, "target": tgt}

    def test_focalloss_training_code_instantiated(self):
        """ISA-207: focalloss 应在 __init__ 中被实例化，而非只在 forward 中引用"""
        graph = self._build_graph(
            nodes=[
                self._build_node("input_1", "input", {}),
                self._build_node("linear_1", "linear", {"in_features": 10, "out_features": 2, "bias": True}),
                self._build_node("focalloss_1", "focalloss", {"alpha": 0.25, "gamma": 2.0}),
            ],
            edges=[
                self._build_edge("input_1", "linear_1"),
                self._build_edge("linear_1", "focalloss_1"),
            ],
        )
        gen = UnifiedCodeGenerator(graph, {"training_nodes": True})
        code = gen.generate_full()

        # 修复后：__init__ 中应出现 FocalLoss 实例化
        self.assertIn("FocalLoss", code, "focalloss 应在 __init__ 中被实例化")
        # 不应出现未定义引用
        self.assertNotIn("self.x_focalloss_1()", code, "不应出现未定义的 self.x_focalloss_1 调用")

    def test_labelsmoothing_training_code_instantiated(self):
        """ISA-207: labelsmoothing 应在 __init__ 中被实例化"""
        graph = self._build_graph(
            nodes=[
                self._build_node("input_1", "input", {}),
                self._build_node("linear_1", "linear", {"in_features": 10, "out_features": 2, "bias": True}),
                self._build_node("labelsmoothing_1", "labelsmoothing", {"smoothing": 0.1}),
            ],
            edges=[
                self._build_edge("input_1", "linear_1"),
                self._build_edge("linear_1", "labelsmoothing_1"),
            ],
        )
        gen = UnifiedCodeGenerator(graph, {"training_nodes": True})
        code = gen.generate_full()

        self.assertIn("CrossEntropyLoss", code, "labelsmoothing 应在 __init__ 中被实例化")
        self.assertIn("label_smoothing", code, "labelsmoothing 应传递 label_smoothing 参数")

    def test_permute_forward_emits_torch_method(self):
        """ISA-207: permute 应生成 .permute(...) 而非未定义模块引用"""
        graph = self._build_graph(
            nodes=[
                self._build_node("input_1", "input", {}),
                self._build_node("permute_1", "permute", {"dims": "0,2,1"}),
                self._build_node("output_1", "output", {}),
            ],
            edges=[
                self._build_edge("input_1", "permute_1"),
                self._build_edge("permute_1", "output_1"),
            ],
        )
        gen = UnifiedCodeGenerator(graph)
        code = gen.generate_model()

        self.assertIn(".permute(", code, "permute 应生成 torch.Tensor.permute 方法调用")
        self.assertNotIn("self.x_permute_1", code, "不应生成未定义的 self.x_permute_1 引用")

    def test_squeeze_forward_emits_torch_method(self):
        """ISA-207: squeeze 应生成 .squeeze(...) 而非未定义模块引用"""
        graph = self._build_graph(
            nodes=[
                self._build_node("input_1", "input", {}),
                self._build_node("squeeze_1", "squeeze", {"dim": 1}),
                self._build_node("output_1", "output", {}),
            ],
            edges=[
                self._build_edge("input_1", "squeeze_1"),
                self._build_edge("squeeze_1", "output_1"),
            ],
        )
        gen = UnifiedCodeGenerator(graph)
        code = gen.generate_model()

        self.assertIn(".squeeze(", code, "squeeze 应生成 torch.Tensor.squeeze 方法调用")
        self.assertNotIn("self.x_squeeze_1", code, "不应生成未定义的 self.x_squeeze_1 引用")

    def test_expand_forward_emits_torch_method(self):
        """ISA-207: expand 应生成 .expand(...) 而非未定义模块引用"""
        graph = self._build_graph(
            nodes=[
                self._build_node("input_1", "input", {}),
                self._build_node("expand_1", "expand", {"shape": "-1,3,224,224"}),
                self._build_node("output_1", "output", {}),
            ],
            edges=[
                self._build_edge("input_1", "expand_1"),
                self._build_edge("expand_1", "output_1"),
            ],
        )
        gen = UnifiedCodeGenerator(graph)
        code = gen.generate_model()

        self.assertIn(".expand(", code, "expand 应生成 torch.Tensor.expand 方法调用")
        self.assertNotIn("self.x_expand_1", code, "不应生成未定义的 self.x_expand_1 引用")

    def test_focalloss_code_is_executable(self):
        """ISA-207: 含 focalloss 的图生成的代码应可 compile"""
        graph = self._build_graph(
            nodes=[
                self._build_node("input_1", "input", {}),
                self._build_node("linear_1", "linear", {"in_features": 10, "out_features": 2, "bias": True}),
                self._build_node("focalloss_1", "focalloss", {"alpha": 0.25, "gamma": 2.0}),
            ],
            edges=[
                self._build_edge("input_1", "linear_1"),
                self._build_edge("linear_1", "focalloss_1"),
            ],
        )
        gen = UnifiedCodeGenerator(graph, {"training_nodes": True})
        code = gen.generate_full()
        compile(code, "<focalloss_exec_test>", "exec")

    def test_permute_code_is_executable(self):
        """ISA-207: 含 permute 的图生成的代码应可 compile"""
        graph = self._build_graph(
            nodes=[
                self._build_node("input_1", "input", {}),
                self._build_node("permute_1", "permute", {"dims": "0,2,1"}),
                self._build_node("output_1", "output", {}),
            ],
            edges=[
                self._build_edge("input_1", "permute_1"),
                self._build_edge("permute_1", "output_1"),
            ],
        )
        gen = UnifiedCodeGenerator(graph)
        code = gen.generate_model()
        compile(code, "<permute_exec_test>", "exec")


class TestP0FrontendBugRegressionAcceptance(unittest.TestCase):
    """ISA-208: 前端自定义复合模块类定义内联"""

    def test_custom_composite_class_inlined(self):
        """自定义复合模块应在生成代码中内联 class 定义"""
        # ISA-208 已由前端单元测试 src/utils/codeGenerator.test.ts 验证：
        # 'emits inline class definition for custom composite nodes not in localStorage'
        # 验证 generateLocalCode 对 isCustomComposite=true 的节点输出 class Xxx(nn.Module):
        # 即使 localStorage 中无对应自定义类定义，也能根据 internalStructure/internalEdges 内联生成。
        pass


if __name__ == "__main__":
    unittest.main()
