"""
Backend code generator golden sample tests
为 ast_core (code_gen_v2) 编写最小化输入，断言生成代码包含预期的
class 定义和 forward 调用链。
"""
import pytest

from backend.services.ast_core import generate as ast_generate, build_ast
from backend.services.code_gen_v2 import WorkflowCodeGenerator


def _make_linear_relu_linear_graph() -> dict:
    """最小化输入：Input -> Linear -> ReLU -> Linear -> Output"""
    return {
        "nodes": [
            {"id": "input1", "data": {"nodeType": "input", "label": "Input", "params": {"shape": "784"}}},
            {"id": "linear1", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 784, "out_features": 128, "bias": True}}},
            {"id": "relu1", "data": {"nodeType": "relu", "label": "ReLU", "params": {}}},
            {"id": "linear2", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 128, "out_features": 10, "bias": True}}},
            {"id": "output1", "data": {"nodeType": "output", "label": "Output", "params": {}}},
        ],
        "edges": [
            {"source": "input1", "target": "linear1", "sourceHandle": None, "targetHandle": "x"},
            {"source": "linear1", "target": "relu1", "sourceHandle": None, "targetHandle": "x"},
            {"source": "relu1", "target": "linear2", "sourceHandle": None, "targetHandle": "x"},
            {"source": "linear2", "target": "output1", "sourceHandle": None, "targetHandle": "x"},
        ],
    }


def _make_conv_bn_relu_pool_graph() -> dict:
    """CNN 最小化输入：Input -> Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Output"""
    return {
        "nodes": [
            {"id": "input1", "data": {"nodeType": "input", "label": "Input", "params": {"shape": "3, 32, 32"}}},
            {"id": "conv1", "data": {"nodeType": "conv2d", "label": "Conv2d", "params": {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1, "bias": False}}},
            {"id": "bn1", "data": {"nodeType": "batchnorm2d", "label": "BatchNorm", "params": {"num_features": 16}}},
            {"id": "relu1", "data": {"nodeType": "relu", "label": "ReLU", "params": {}}},
            {"id": "pool1", "data": {"nodeType": "maxpool", "label": "MaxPool", "params": {"kernel_size": 2, "stride": 2, "padding": 0}}},
            {"id": "output1", "data": {"nodeType": "output", "label": "Output", "params": {}}},
        ],
        "edges": [
            {"source": "input1", "target": "conv1", "sourceHandle": None, "targetHandle": "x"},
            {"source": "conv1", "target": "bn1", "sourceHandle": None, "targetHandle": "x"},
            {"source": "bn1", "target": "relu1", "sourceHandle": None, "targetHandle": "x"},
            {"source": "relu1", "target": "pool1", "sourceHandle": None, "targetHandle": "x"},
            {"source": "pool1", "target": "output1", "sourceHandle": None, "targetHandle": "x"},
        ],
    }


def test_ast_core_linear_relu_linear():
    """ast_core: Linear -> ReLU -> Linear 应生成包含预期 nn.Module 和 forward 链的代码"""
    graph = _make_linear_relu_linear_graph()
    code = ast_generate(graph)

    # class 定义
    assert "class FlowHamsterModel(nn.Module):" in code

    # __init__ 中注册模块
    assert "nn.Linear(in_features=784, out_features=128" in code
    assert "nn.ReLU()" in code
    assert "nn.Linear(in_features=128, out_features=10" in code

    # forward 调用链
    assert "x_linear_1 = self.x_linear_1(" in code
    assert "x_relu_1 = self.x_relu_1(" in code
    assert "x_linear_2 = self.x_linear_2(" in code
    assert "return x_linear_2" in code or "return x_output" in code


def test_ast_core_conv_bn_relu_pool():
    """ast_core: CNN 流水线应生成 Conv2d + BatchNorm2d + ReLU + MaxPool2d"""
    graph = _make_conv_bn_relu_pool_graph()
    code = ast_generate(graph)

    assert "class FlowHamsterModel(nn.Module):" in code
    assert "nn.Conv2d(in_channels=3, out_channels=16" in code
    assert "nn.BatchNorm2d(num_features=16)" in code
    assert "nn.ReLU()" in code
    assert "nn.MaxPool2d(kernel_size=2" in code

    # forward 链
    assert "x_conv2d_1 = self.x_conv2d_1(" in code
    assert "x_batchnorm2d_1 = self.x_batchnorm2d_1(" in code
    assert "x_relu_1 = self.x_relu_1(" in code
    assert "x_maxpool2d_1 = self.x_maxpool2d_1(" in code


def test_code_gen_v2_linear_relu_linear():
    """code_gen_v2: 同样的图应生成可执行的 PyTorch 代码"""
    graph = _make_linear_relu_linear_graph()
    gen = WorkflowCodeGenerator(graph)
    code = gen.generate()

    assert "class FlowHamsterModel(nn.Module):" in code
    assert "nn.Linear" in code
    assert "def forward(self, x):" in code

    # 验证 exec 能成功 import 并运行
    namespace = {}
    exec(code, namespace)
    model = namespace["FlowHamsterModel"]()
    import torch
    x = torch.randn(1, 784)
    out = model(x)
    assert out.shape == (1, 10)


def test_code_gen_v2_conv_forward_execution():
    """code_gen_v2: CNN 图生成代码后可执行并输出正确 shape"""
    graph = _make_conv_bn_relu_pool_graph()
    gen = WorkflowCodeGenerator(graph)
    code = gen.generate()

    namespace = {}
    exec(code, namespace)
    model = namespace["FlowHamsterModel"]()
    import torch
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    # 经过 Conv(3->16, pad=1) 保持 32x32，再 MaxPool2d(2,2) -> 16x16
    assert out.shape == (1, 16, 16, 16)


def test_ast_core_training_config_generates_optimizer():
    """ast_core: 传入 training_config 时应在 if __name__ 中生成 optimizer"""
    graph = _make_linear_relu_linear_graph()
    training_config = {
        "taskType": "classification",
        "loss": {"type": "cross_entropy", "enabled": True, "params": {}},
        "optimizer": {"type": "adamw", "enabled": True, "params": {"lr": 0.001, "weight_decay": 0.01}},
        "scheduler": {"type": "cosine_annealing", "enabled": False, "params": {}},
        "metrics": [],
        "runtime": {"device": "cpu", "epochs": 2, "batchSize": 32, "amp": False, "gradClip": None, "numWorkers": 0},
        "checkpoint": {"enabled": False},
    }
    code = ast_generate(graph, options={"training_config": training_config})

    assert "torch.optim.AdamW(model.parameters()" in code
    assert "loss_fn = nn.CrossEntropyLoss()" in code
    assert "loss.backward()" in code
    assert "optimizer.step()" in code


def test_code_gen_v2_empty_graph():
    """code_gen_v2: 空图应返回提示性注释而非崩溃"""
    gen = WorkflowCodeGenerator({"nodes": [], "edges": []})
    code = gen.generate()
    assert "Empty graph" in code or code.startswith("#")


def test_ast_core_build_ast_topology_order():
    """build_ast: 拓扑排序应保证输入在输出之前"""
    graph = _make_linear_relu_linear_graph()
    blocks = build_ast(graph)
    ids = [b.node_id for b in blocks]
    assert ids.index("input1") < ids.index("linear1")
    assert ids.index("linear1") < ids.index("relu1")
    assert ids.index("relu1") < ids.index("linear2")
    assert ids.index("linear2") < ids.index("output1")


def test_ast_core_multi_output():
    """ast_core: 多输出节点应生成 tuple return"""
    graph = {
        "nodes": [
            {"id": "input1", "data": {"nodeType": "input", "label": "Input", "params": {}}},
            {"id": "linear1", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 10, "out_features": 5}}},
            {"id": "linear2", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 10, "out_features": 3}}},
            {"id": "output1", "data": {"nodeType": "output", "label": "Out1", "params": {}}},
            {"id": "output2", "data": {"nodeType": "output", "label": "Out2", "params": {}}},
        ],
        "edges": [
            {"source": "input1", "target": "linear1", "sourceHandle": None, "targetHandle": "x"},
            {"source": "input1", "target": "linear2", "sourceHandle": None, "targetHandle": "x"},
            {"source": "linear1", "target": "output1", "sourceHandle": None, "targetHandle": "x"},
            {"source": "linear2", "target": "output2", "sourceHandle": None, "targetHandle": "x"},
        ],
    }
    code = ast_generate(graph)
    assert "return (" in code or "return x_output" in code
    assert "class FlowHamsterModel" in code
