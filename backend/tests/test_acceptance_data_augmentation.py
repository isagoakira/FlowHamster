"""
数据增强节点验收测试
覆盖 MixUp / CutMix / AutoAugment 的代码生成正确性
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from backend.services.dataflow_compiler import compile_dataflow


class TestDataAugmentationAcceptance(unittest.TestCase):
    """验收标准：数据增强节点生成的 Dataset.__getitem__ 代码可编译、逻辑正确"""

    def _build_data_graph(self, nodes: list, edges: list = None):
        return {"nodes": nodes, "edges": edges or []}

    def _build_node(self, node_id: str, node_type: str, params: dict):
        return {"id": node_id, "data": {"nodeType": node_type, "label": node_type, "params": params}}

    def test_mixup_generates_beta_and_label_blend(self):
        """MixUp 生成代码应包含 Beta 分布采样和标签混合"""
        data_graph = self._build_data_graph([
            self._build_node("mixup_1", "mixup", {"alpha": 0.2}),
        ])
        result = compile_dataflow(data_graph=data_graph)
        code = result.python_scaffold

        self.assertIn("Beta", code, "MixUp 应使用 Beta 分布")
        self.assertIn("_lam", code, "MixUp 应生成 _lam 变量")
        self.assertIn("_field_image = _lam * _field_image", code, "MixUp 应对图像做线性插值")
        self.assertIn("_field_label = _lam * _field_label", code, "MixUp 应对标签做线性插值")

    def test_cutmix_generates_bbox_and_label_blend(self):
        """CutMix 生成代码应包含随机 bbox 裁剪和标签混合"""
        data_graph = self._build_data_graph([
            self._build_node("cutmix_1", "cutmix", {"alpha": 1.0}),
        ])
        result = compile_dataflow(data_graph=data_graph)
        code = result.python_scaffold

        self.assertIn("_cut_h", code, "CutMix 应计算裁剪高度")
        self.assertIn("_cut_w", code, "CutMix 应计算裁剪宽度")
        self.assertIn("_field_image[:, _y1:_y2, _x1:_x2]", code, "CutMix 应做区域替换")
        self.assertIn("_field_label = _lam * _field_label", code, "CutMix 应对标签做线性插值")

    def test_autoaugment_generates_torchvision_policy(self):
        """AutoAugment 生成代码应引用 torchvision AutoAugmentPolicy"""
        data_graph = self._build_data_graph([
            self._build_node("autoaugment_1", "autoaugment", {"policy": "imagenet"}),
        ])
        result = compile_dataflow(data_graph=data_graph)
        code = result.python_scaffold

        self.assertIn("AutoAugment", code, "AutoAugment 应生成 torchvision 调用")
        self.assertIn("AutoAugmentPolicy", code, "AutoAugment 应引用 Policy 枚举")

    def test_mixup_code_is_py_compile_clean(self):
        """MixUp 生成代码应可通过 py_compile"""
        data_graph = self._build_data_graph([
            self._build_node("mixup_1", "mixup", {"alpha": 0.2}),
        ])
        result = compile_dataflow(data_graph=data_graph)
        code = result.python_scaffold
        self.assertTrue(len(code) > 0, "应生成非空代码")
        compile(code, "<mixup_test>", "exec")

    def test_cutmix_code_is_py_compile_clean(self):
        """CutMix 生成代码应可通过 py_compile"""
        data_graph = self._build_data_graph([
            self._build_node("cutmix_1", "cutmix", {"alpha": 1.0}),
        ])
        result = compile_dataflow(data_graph=data_graph)
        code = result.python_scaffold
        self.assertTrue(len(code) > 0, "应生成非空代码")
        compile(code, "<cutmix_test>", "exec")


if __name__ == "__main__":
    unittest.main()
