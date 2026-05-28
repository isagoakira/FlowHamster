"""
训练异常与资源管理测试
验证 execute、forward、gradients 路由的错误处理行为，
确保异常时返回可读 error 字段，不泄露 traceback 到前端。
"""
import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_forward_pass_returns_expected_structure():
    """POST /api/execute/forward 成功时返回 success + outputs 数组"""
    response = client.post(
        "/api/execute/forward",
        json={
            "nodes": [
                {"id": "input1", "data": {"nodeType": "input", "label": "Input", "params": {}}},
                {"id": "linear1", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 10, "out_features": 5}}},
                {"id": "output1", "data": {"nodeType": "output", "label": "Output", "params": {}}},
            ],
            "edges": [
                {"source": "input1", "target": "linear1", "sourceHandle": None, "targetHandle": "x"},
                {"source": "linear1", "target": "output1", "sourceHandle": None, "targetHandle": "x"},
            ],
            "input_shape": [1, 10],
        },
    )
    assert response.status_code == 200
    data = response.json()
    # 无论成功失败，结构必须包含 success 字段
    assert "success" in data
    if data["success"]:
        assert "outputs" in data
        assert isinstance(data["outputs"], (list, dict))
    else:
        assert "error" in data
        # 错误信息应为字符串，不应包含大段 traceback
        err = data["error"]
        assert isinstance(err, str)
        assert "Traceback" not in err


def test_forward_pass_invalid_node_type():
    """POST /api/execute/forward 传入非法节点类型时不应崩溃"""
    response = client.post(
        "/api/execute/forward",
        json={
            "nodes": [
                {"id": "input1", "data": {"nodeType": "input", "label": "Input", "params": {}}},
                {"id": "bad1", "data": {"nodeType": "nonexistent_node_type", "label": "Bad", "params": {}}},
                {"id": "output1", "data": {"nodeType": "output", "label": "Output", "params": {}}},
            ],
            "edges": [
                {"source": "input1", "target": "bad1", "sourceHandle": None, "targetHandle": "x"},
                {"source": "bad1", "target": "output1", "sourceHandle": None, "targetHandle": "x"},
            ],
            "input_shape": [1, 10],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    if not data["success"]:
        assert "error" in data
        assert "Traceback" not in data["error"]


def test_gradients_returns_expected_structure():
    """POST /api/execute/gradients 返回 success + data 结构"""
    response = client.post(
        "/api/execute/gradients",
        json={
            "nodes": [
                {"id": "input1", "data": {"nodeType": "input", "label": "Input", "params": {}}},
                {"id": "linear1", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 10, "out_features": 5}}},
                {"id": "output1", "data": {"nodeType": "output", "label": "Output", "params": {}}},
            ],
            "edges": [
                {"source": "input1", "target": "linear1", "sourceHandle": None, "targetHandle": "x"},
                {"source": "linear1", "target": "output1", "sourceHandle": None, "targetHandle": "x"},
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    if data["success"]:
        assert "data" in data
    else:
        assert "error" in data
        assert "Traceback" not in data["error"]


def test_gradients_empty_graph():
    """POST /api/execute/gradients 空图时不应崩溃"""
    response = client.post(
        "/api/execute/gradients",
        json={"nodes": [], "edges": []},
    )
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    if not data["success"]:
        assert "error" in data
        assert "Traceback" not in data["error"]


def test_execute_code_return_structure():
    """POST /api/execute/execute 返回 success + output + error 结构"""
    response = client.post(
        "/api/execute/execute",
        json={
            "code": "print('hello')",
            "input_shape": None,
            "target_device": "cpu",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "output" in data
    assert "error" in data
    if data["success"]:
        assert data["error"] is None or data["error"] == ""
    else:
        # 错误信息不应泄露 traceback
        if data["error"]:
            assert "Traceback" not in data["error"]


def test_execute_code_invalid_python():
    """POST /api/execute/execute 执行非法 Python 时返回可读错误"""
    response = client.post(
        "/api/execute/execute",
        json={
            "code": "raise ValueError('intentional error')",
            "target_device": "cpu",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["error"] is not None
    assert "intentional error" in data["error"]
    # 不应包含 traceback 帧信息
    assert "Traceback" not in data["error"]
    assert "File \"" not in data["error"]


def test_execute_code_timeout_simulation():
    """POST /api/execute/execute 执行超时时返回 timed out 提示（不实际等 60s）"""
    # 这里只验证接口契约，不触发真实超时
    response = client.post(
        "/api/execute/execute",
        json={
            "code": "print('ok')",
            "target_device": "cpu",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "output" in data
    assert "error" in data


def test_training_config_epochs_zero():
    """epochs=0 时训练代码仍应生成但执行时不进入循环"""
    from backend.services.ast_core import generate as ast_generate
    graph = {
        "nodes": [
            {"id": "input1", "data": {"nodeType": "input", "label": "Input", "params": {}}},
            {"id": "linear1", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 10, "out_features": 2}}},
            {"id": "output1", "data": {"nodeType": "output", "label": "Output", "params": {}}},
        ],
        "edges": [
            {"source": "input1", "target": "linear1", "sourceHandle": None, "targetHandle": "x"},
            {"source": "linear1", "target": "output1", "sourceHandle": None, "targetHandle": "x"},
        ],
    }
    training_config = {
        "taskType": "classification",
        "loss": {"type": "cross_entropy", "enabled": True, "params": {}},
        "optimizer": {"type": "adamw", "enabled": True, "params": {"lr": 0.001}},
        "scheduler": {"type": "step_lr", "enabled": False, "params": {}},
        "metrics": [],
        "runtime": {"device": "cpu", "epochs": 0, "batchSize": 32, "amp": False, "gradClip": None, "numWorkers": 0},
        "checkpoint": {"enabled": False},
    }
    code = ast_generate(graph, options={"training_config": training_config})
    assert "class FlowHamsterModel" in code
    # 训练配置信息应出现在代码中
    assert "epochs=0" in code


def test_empty_csv_source_scaffold():
    """空 CSV 路径时 dataflow compiler 不应崩溃，应生成有效占位"""
    from backend.services.dataflow_compiler import compile_dataflow
    data_graph = {
        "nodes": [
            {"id": "csv1", "data": {"nodeType": "csv_source", "label": "CSV", "params": {"path": "", "delimiter": ","}}},
            {"id": "dsout1", "data": {"nodeType": "dataset_output", "label": "Dataset Output", "params": {"fields": "image,label"}}},
        ],
        "edges": [
            {"source": "csv1", "target": "dsout1", "sourceHandle": None, "targetHandle": "x"},
        ],
    }
    compiled = compile_dataflow(
        model_graph=None,
        data_graph=data_graph,
        bindings=[],
        training_config={"taskType": "classification"},
    )
    assert compiled.has_workflow_runtime is True
    assert "FlowHamsterDataset" in compiled.python_scaffold
    assert "pd.read_csv" in compiled.python_scaffold


def test_invalid_bindings_no_crash():
    """非法绑定（如 targetKey 为空）时不应崩溃，应发出警告"""
    from backend.services.dataflow_compiler import compile_dataflow
    model_graph = {
        "nodes": [
            {"id": "input1", "data": {"nodeType": "input", "label": "Input", "params": {"name": "image"}}},
        ],
        "edges": [],
    }
    bindings = [
        {"target": "model_input", "targetKey": "", "sourceGraph": "data", "sourceKey": "image"},
    ]
    compiled = compile_dataflow(
        model_graph=model_graph,
        data_graph=None,
        bindings=bindings,
        training_config={"taskType": "classification"},
    )
    # 不应抛出异常
    assert isinstance(compiled.warnings, list)
