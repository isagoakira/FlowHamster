"""
Workflow save/load consistency tests
验证工作流序列化 -> 保存 -> 重新加载 -> 再次生成代码 的一致性。
"""
import json
import tempfile
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schema.workflow_document import WorkflowDocument, WorkflowGraphSnapshot, WorkflowTrainingConfig
from backend.services.ast_core import generate as ast_generate
from backend.services.code_gen_v2 import WorkflowCodeGenerator

client = TestClient(app)


def _build_complete_workflow() -> dict:
    """构造含 modelGraph + dataGraph + bindings + trainingConfig 的完整工作流"""
    return {
        "version": "2.0.0",
        "metadata": {
            "name": "Test Workflow",
            "description": "ISA-136 consistency test",
            "schemaVersion": "2.0.0",
            "createdAt": 1716854400.0,
            "updatedAt": 1716854400.0,
            "exportSource": "flowhamster",
        },
        "modelGraph": {
            "kind": "model",
            "nodes": [
                {"id": "input1", "data": {"nodeType": "input", "label": "Input", "params": {"shape": "1, 28, 28"}}},
                {"id": "linear1", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 784, "out_features": 128}}},
                {"id": "relu1", "data": {"nodeType": "relu", "label": "ReLU", "params": {}}},
                {"id": "linear2", "data": {"nodeType": "linear", "label": "Linear", "params": {"in_features": 128, "out_features": 10}}},
                {"id": "output1", "data": {"nodeType": "output", "label": "Output", "params": {}}},
            ],
            "edges": [
                {"source": "input1", "target": "linear1", "sourceHandle": "result", "targetHandle": "x"},
                {"source": "linear1", "target": "relu1", "sourceHandle": "result", "targetHandle": "x"},
                {"source": "relu1", "target": "linear2", "sourceHandle": "result", "targetHandle": "x"},
                {"source": "linear2", "target": "output1", "sourceHandle": "result", "targetHandle": "x"},
            ],
            "contract": {"inputs": [{"name": "image", "dtype": "float32", "shape_hint": "1,28,28"}], "outputs": [{"name": "logits", "dtype": "float32", "shape_hint": "10"}]},
        },
        "dataGraph": {
            "kind": "data",
            "nodes": [
                {"id": "csv1", "data": {"nodeType": "csv_source", "label": "CSV", "params": {"path": "./data/mnist.csv", "delimiter": ","}}},
                {"id": "select1", "data": {"nodeType": "select_fields", "label": "Select", "params": {"fields": "image,label"}}},
                {"id": "dsout1", "data": {"nodeType": "dataset_output", "label": "Dataset Output", "params": {"fields": "image,label"}}},
            ],
            "edges": [
                {"source": "csv1", "target": "select1", "sourceHandle": "result", "targetHandle": "x"},
                {"source": "select1", "target": "dsout1", "sourceHandle": "result", "targetHandle": "x"},
            ],
            "contract": {"inputs": [], "outputs": [{"name": "image", "dtype": "float32"}, {"name": "label", "dtype": "int64"}]},
        },
        "trainingConfig": {
            "taskType": "classification",
            "loss": {"type": "cross_entropy", "enabled": True, "params": {}},
            "optimizer": {"type": "adamw", "enabled": True, "params": {"lr": 0.001, "weight_decay": 0.01}},
            "scheduler": {"type": "step_lr", "enabled": False, "params": {"step_size": 10, "gamma": 0.1}},
            "metrics": [{"type": "accuracy", "enabled": True, "params": {}}],
            "runtime": {"device": "cpu", "epochs": 5, "batchSize": 64, "amp": False, "gradClip": None, "numWorkers": 0},
            "checkpoint": {"enabled": False, "saveTopK": 3, "monitor": "val_loss", "mode": "min", "earlyStopPatience": None},
        },
        "bindings": [
            {"target": "model_input", "targetKey": "image", "sourceGraph": "data", "sourceKey": "image"},
            {"target": "training_target", "sourceKey": "label"},
        ],
    }


def test_workflow_document_schema_roundtrip():
    """WorkflowDocument dataclass 序列化/反序列化不丢字段"""
    doc = WorkflowDocument()
    doc.model_graph = WorkflowGraphSnapshot(
        kind="model",
        nodes=[{"id": "n1", "data": {"nodeType": "linear"}}],
        edges=[{"source": "a", "target": "b"}],
    )
    doc.data_graph = WorkflowGraphSnapshot(
        kind="data",
        nodes=[{"id": "d1", "data": {"nodeType": "csv_source"}}],
        edges=[],
    )
    doc.training_config = WorkflowTrainingConfig(
        task_type="classification",
        runtime={"epochs": 20, "batch_size": 16},
    )
    doc.bindings = [{"target": "model_input", "targetKey": "x"}]

    # dataclass -> dict -> JSON -> dict -> dataclass
    from dataclasses import asdict
    d = asdict(doc)
    json_str = json.dumps(d)
    loaded = json.loads(json_str)

    assert loaded["version"] == "2.0.0"
    assert loaded["model_graph"]["kind"] == "model"
    assert len(loaded["model_graph"]["nodes"]) == 1
    assert loaded["data_graph"]["kind"] == "data"
    assert loaded["training_config"]["task_type"] == "classification"
    assert loaded["training_config"]["runtime"]["epochs"] == 20
    assert loaded["bindings"][0]["targetKey"] == "x"


def test_workflow_save_reload_api_consistency():
    """通过 API 保存工作流后再加载，document 内容一致"""
    # 创建工作流
    create_resp = client.post(
        "/api/workflows",
        json={"name": "ISA136Consistency", "description": "test"},
    )
    assert create_resp.status_code == 200
    wf = create_resp.json()
    wf_id = wf["id"]

    try:
        document = _build_complete_workflow()

        # 保存文档
        save_resp = client.put(
            f"/api/workflows/{wf_id}/document",
            json=document,
        )
        assert save_resp.status_code == 200

        # 重新加载
        load_resp = client.get(f"/api/workflows/{wf_id}/document")
        assert load_resp.status_code == 200
        loaded = load_resp.json()

        # 关键字段一致
        assert loaded["version"] == document["version"]
        assert loaded["metadata"]["name"] == document["metadata"]["name"]
        assert len(loaded["modelGraph"]["nodes"]) == len(document["modelGraph"]["nodes"])
        assert len(loaded["modelGraph"]["edges"]) == len(document["modelGraph"]["edges"])
        assert len(loaded["dataGraph"]["nodes"]) == len(document["dataGraph"]["nodes"])
        assert len(loaded["bindings"]) == len(document["bindings"])
        assert loaded["trainingConfig"]["taskType"] == document["trainingConfig"]["taskType"]
        assert loaded["trainingConfig"]["runtime"]["epochs"] == document["trainingConfig"]["runtime"]["epochs"]
    finally:
        # 清理
        client.delete(f"/api/workflows/{wf_id}")


def test_workflow_reload_generates_identical_code():
    """保存前后代码生成结果一致"""
    document = _build_complete_workflow()

    # 第一次生成
    code_before = ast_generate(document["modelGraph"], options={"training_config": document["trainingConfig"]})
    assert "class FlowHamsterModel" in code_before
    assert "nn.Linear" in code_before

    # 模拟保存到临时目录（绕过 API，直接测试序列化一致性）
    with tempfile.TemporaryDirectory() as tmpdir:
        doc_path = Path(tmpdir) / "document.json"
        doc_path.write_text(json.dumps(document, indent=2), encoding="utf-8")

        # 重新加载
        loaded = json.loads(doc_path.read_text(encoding="utf-8"))

    # 第二次生成
    code_after = ast_generate(loaded["modelGraph"], options={"training_config": loaded["trainingConfig"]})

    # 核心结构一致（忽略时间戳等无关差异）
    assert code_before == code_after


def test_workflow_update_does_not_corrupt_bindings():
    """部分更新工作流时 bindings 不被意外清空"""
    create_resp = client.post(
        "/api/workflows",
        json={"name": "ISA136Bindings", "description": "test"},
    )
    assert create_resp.status_code == 200
    wf_id = create_resp.json()["id"]

    try:
        document = _build_complete_workflow()
        client.put(f"/api/workflows/{wf_id}/document", json=document)

        # 只更新 name
        update_resp = client.put(
            f"/api/workflows/{wf_id}",
            json={"name": "Renamed"},
        )
        assert update_resp.status_code == 200

        # 检查 bindings 仍然保留
        loaded = client.get(f"/api/workflows/{wf_id}/document").json()
        assert len(loaded["bindings"]) == 2
        assert loaded["bindings"][0]["targetKey"] == "image"
    finally:
        client.delete(f"/api/workflows/{wf_id}")


def test_workflow_code_gen_v2_roundtrip():
    """code_gen_v2 在保存/加载前后生成一致代码"""
    document = _build_complete_workflow()
    gen = WorkflowCodeGenerator(document["modelGraph"])
    code_v2_before = gen.generate()

    # 序列化并重新加载
    json_str = json.dumps(document)
    loaded = json.loads(json_str)

    gen_after = WorkflowCodeGenerator(loaded["modelGraph"])
    code_v2_after = gen_after.generate()

    assert code_v2_before == code_v2_after
    assert "class FlowHamsterModel" in code_v2_after
