"""
双图联动语义对齐测试
验证 dataGraph + modelGraph + bindings 生成的完整脚本能在后端执行，
且 bindings 正确将数据字段映射到模型输入。
"""
import pytest
from backend.services.dataflow_compiler import compile_dataflow
from backend.services.code_gen_v2 import WorkflowCodeGenerator
from backend.services.ast_core import generate as ast_generate


def _make_model_graph() -> dict:
    """模型图：Input(image) -> Linear(784->10) -> Output"""
    return {
        "nodes": [
            {"id": "input1", "data": {"nodeType": "input", "label": "Input", "params": {"name": "image"}}},
            {"id": "linear1", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 784, "out_features": 10}}},
            {"id": "output1", "data": {"nodeType": "output", "label": "Output", "params": {}}},
        ],
        "edges": [
            {"source": "input1", "target": "linear1", "sourceHandle": None, "targetHandle": "x"},
            {"source": "linear1", "target": "output1", "sourceHandle": None, "targetHandle": "x"},
        ],
    }


def _make_data_graph() -> dict:
    """数据图：csv_source -> select_fields -> dataset_output"""
    return {
        "nodes": [
            {"id": "csv1", "data": {"nodeType": "csv_source", "label": "CSV", "params": {"path": "./data/train.csv", "delimiter": ","}}},
            {"id": "select1", "data": {"nodeType": "select_fields", "label": "Select", "params": {"fields": "image,label"}}},
            {"id": "dsout1", "data": {"nodeType": "dataset_output", "label": "Dataset Output", "params": {"fields": "image,label"}}},
        ],
        "edges": [
            {"source": "csv1", "target": "select1", "sourceHandle": None, "targetHandle": "x"},
            {"source": "select1", "target": "dsout1", "sourceHandle": None, "targetHandle": "x"},
        ],
    }


def test_compile_dataflow_bindings_mapping():
    """compile_dataflow 应正确解析 bindings 并生成 BOUND_MODEL_INPUTS"""
    model_graph = _make_model_graph()
    data_graph = _make_data_graph()
    bindings = [
        {"target": "model_input", "targetKey": "image", "sourceGraph": "data", "sourceKey": "image"},
        {"target": "training_target", "sourceKey": "label"},
    ]
    compiled = compile_dataflow(
        model_graph=model_graph,
        data_graph=data_graph,
        bindings=bindings,
        training_config={"taskType": "classification"},
    )

    assert compiled.has_workflow_runtime is True
    assert compiled.primary_model_input_key == "image"
    assert compiled.target_binding_source == "label"
    assert len(compiled.model_input_bindings) == 1
    assert compiled.model_input_bindings[0]["target_key"] == "image"
    assert compiled.model_input_bindings[0]["source_key"] == "image"

    # scaffold 中应包含绑定映射
    assert "BOUND_MODEL_INPUTS" in compiled.python_scaffold
    assert '"image": "image"' in compiled.python_scaffold
    assert "BOUND_TRAINING_TARGET" in compiled.python_scaffold
    assert "FlowHamsterDataset" in compiled.python_scaffold


def test_compile_dataflow_missing_binding_warning():
    """模型输入未绑定数据字段时应发出警告"""
    model_graph = _make_model_graph()
    data_graph = _make_data_graph()
    # 故意不提供 bindings
    compiled = compile_dataflow(
        model_graph=model_graph,
        data_graph=data_graph,
        bindings=[],
        training_config={"taskType": "classification"},
    )

    assert any("尚未绑定" in w for w in compiled.warnings)


def test_compile_dataflow_source_field_not_in_output_warning():
    """绑定源字段未在 Dataset Output 中声明时应发出警告"""
    model_graph = _make_model_graph()
    data_graph = _make_data_graph()
    bindings = [
        {"target": "model_input", "targetKey": "image", "sourceGraph": "data", "sourceKey": "nonexistent_field"},
    ]
    compiled = compile_dataflow(
        model_graph=model_graph,
        data_graph=data_graph,
        bindings=bindings,
        training_config={"taskType": "classification"},
    )

    assert any("未在 Dataset Output 中声明" in w for w in compiled.warnings)


def test_full_script_runnable_with_bindings():
    """dataGraph + modelGraph + bindings 生成的完整脚本应能在 Python 中执行"""
    model_graph = _make_model_graph()
    data_graph = _make_data_graph()
    bindings = [
        {"target": "model_input", "targetKey": "image", "sourceGraph": "data", "sourceKey": "image"},
        {"target": "training_target", "sourceKey": "label"},
    ]
    training_config = {
        "taskType": "classification",
        "loss": {"type": "cross_entropy", "enabled": True, "params": {}},
        "optimizer": {"type": "adamw", "enabled": True, "params": {"lr": 0.001}},
        "scheduler": {"type": "step_lr", "enabled": False, "params": {}},
        "metrics": [],
        "runtime": {"device": "cpu", "epochs": 1, "batchSize": 8},
        "checkpoint": {"enabled": False},
    }

    # 生成模型代码
    model_code = ast_generate(model_graph, options={"training_config": training_config})

    # 编译数据流
    compiled = compile_dataflow(
        model_graph=model_graph,
        data_graph=data_graph,
        bindings=bindings,
        training_config=training_config,
    )

    full_code = model_code + "\n\n" + compiled.python_scaffold

    # 执行验证：至少能 import 并实例化模型
    namespace = {}
    exec(full_code, namespace)
    model = namespace["FlowHamsterModel"]()
    import torch
    x = torch.randn(1, 784)
    out = model(x)
    assert out.shape == (1, 10)

    # 验证 resolve_bound_inputs 存在并可调用
    assert "resolve_bound_inputs" in namespace
    batch = {"image": torch.randn(1, 784), "label": torch.tensor([0])}
    model_feed, target = namespace["resolve_bound_inputs"](batch, "cpu")
    assert "image" in model_feed
    assert target is not None


def test_code_gen_v2_with_dataflow_scaffold():
    """code_gen_v2 模型代码 + dataflow scaffold 组合后可执行"""
    model_graph = _make_model_graph()
    data_graph = _make_data_graph()
    bindings = [
        {"target": "model_input", "targetKey": "image", "sourceGraph": "data", "sourceKey": "image"},
    ]

    gen = WorkflowCodeGenerator(model_graph)
    model_code = gen.generate()
    compiled = compile_dataflow(
        model_graph=model_graph,
        data_graph=data_graph,
        bindings=bindings,
        training_config={"taskType": "classification"},
    )

    full_code = model_code + "\n\n" + compiled.python_scaffold
    namespace = {}
    exec(full_code, namespace)
    model = namespace["FlowHamsterModel"]()
    import torch
    x = torch.randn(1, 784)
    out = model(x)
    assert out.shape == (1, 10)
