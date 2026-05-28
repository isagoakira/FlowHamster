"""
FlowHamster Workflow Router
后端工作流持久化 API
"""
import os
import json
import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/workflows", tags=["workflows"])

# 工作流存储根目录
WORKFLOWS_DIR = (Path(__file__).parent.parent.parent / "workflows").resolve()

class WorkflowCreate(BaseModel):
    name: str
    description: Optional[str] = ""

class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    document: Optional[dict] = None

class WorkflowInfo(BaseModel):
    id: str
    name: str
    description: str
    created_at: float
    updated_at: float

def get_workflow_dir(name: str) -> Path:
    """获取工作流目录路径（sanitized name）"""
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return _resolve_workflow_dir(safe_name, must_exist=False)

def ensure_workflows_dir():
    """确保工作流根目录存在"""
    WORKFLOWS_DIR.mkdir(exist_ok=True)

def _resolve_workflow_dir(workflow_id: str, *, must_exist: bool = True) -> Path:
    """Resolve a workflow id to a directory guaranteed to stay under WORKFLOWS_DIR."""
    if not workflow_id or workflow_id in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid workflow id")

    candidate_id = Path(workflow_id)
    if candidate_id.is_absolute() or any(part == ".." for part in candidate_id.parts):
        raise HTTPException(status_code=400, detail="Invalid workflow id")

    resolved = (WORKFLOWS_DIR / candidate_id).resolve()
    try:
        resolved.relative_to(WORKFLOWS_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid workflow id") from exc

    if resolved == WORKFLOWS_DIR:
        raise HTTPException(status_code=400, detail="Invalid workflow id")
    if must_exist and not resolved.exists():
        raise HTTPException(status_code=404, detail="Workflow not found")
    return resolved

def get_default_document(workflow_id: str, name: str, description: str) -> dict:
    """创建默认的工作流文档结构"""
    import time
    now = time.time()
    return {
        "version": "2.0.0",
        "metadata": {
            "name": name,
            "description": description,
            "schemaVersion": "2.0.0",
            "createdAt": now,
            "updatedAt": now,
            "exportSource": "flowhamster"
        },
        "modelGraph": {
            "kind": "model",
            "nodes": [],
            "edges": [],
            "contract": {"inputs": [], "outputs": []}
        },
        "dataGraph": {
            "kind": "data",
            "nodes": [],
            "edges": [],
            "contract": {"inputs": [], "outputs": []}
        },
        "trainingConfig": {
            "taskType": "classification",
            "loss": {"type": "cross_entropy", "enabled": True, "params": {}},
            "optimizer": {"type": "adamw", "enabled": True, "params": {"lr": 0.001, "weight_decay": 0}},
            "scheduler": {"type": "step_lr", "enabled": False, "params": {"step_size": 10, "gamma": 0.1}},
            "metrics": [],
            "runtime": {"device": "auto", "epochs": 10, "batchSize": 32, "amp": False, "gradClip": None, "numWorkers": 0},
            "checkpoint": {"enabled": False, "saveTopK": 3, "monitor": "val_loss", "mode": "min", "earlyStopPatience": None}
        },
        "bindings": []
    }

@router.get("")
async def list_workflows():
    """列出所有工作流"""
    ensure_workflows_dir()
    workflows = []
    for item in sorted(WORKFLOWS_DIR.iterdir()):
        if item.is_dir():
            meta_file = item / "metadata.json"
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                    workflows.append(meta)
                except:
                    workflows.append({
                        "id": item.name,
                        "name": item.name,
                        "description": "",
                        "created_at": item.stat().st_ctime,
                        "updated_at": item.stat().st_mtime,
                    })
    return workflows

@router.post("")
async def create_workflow(workflow: WorkflowCreate):
    """创建新工作流"""
    ensure_workflows_dir()
    wf_dir = get_workflow_dir(workflow.name)

    if wf_dir.exists():
        raise HTTPException(status_code=400, detail=f"Workflow '{workflow.name}' already exists")

    wf_dir.mkdir(parents=True, exist_ok=True)

    import time
    now = time.time()
    meta = {
        "id": wf_dir.name,
        "name": workflow.name,
        "description": workflow.description or "",
        "created_at": now,
        "updated_at": now,
    }
    (wf_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # 创建完整的工作流文档
    document = get_default_document(wf_dir.name, workflow.name, workflow.description or "")
    document["metadata"]["createdAt"] = now
    document["metadata"]["updatedAt"] = now
    (wf_dir / "document.json").write_text(json.dumps(document, indent=2))

    return meta

@router.get("/{workflow_id}")
async def get_workflow(workflow_id: str):
    """获取工作流完整文档"""
    wf_dir = _resolve_workflow_dir(workflow_id)

    meta_file = wf_dir / "metadata.json"
    doc_file = wf_dir / "document.json"

    meta = {}
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())

    document = {}
    if doc_file.exists():
        document = json.loads(doc_file.read_text())

    return {**meta, "document": document}

@router.put("/{workflow_id}")
async def update_workflow(workflow_id: str, update: WorkflowUpdate):
    """更新工作流"""
    wf_dir = _resolve_workflow_dir(workflow_id)

    meta_file = wf_dir / "metadata.json"
    doc_file = wf_dir / "document.json"

    meta = {}
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())

    document = {}
    if doc_file.exists():
        document = json.loads(doc_file.read_text())

    import time
    now = time.time()

    if update.name is not None:
        meta["name"] = update.name
        document.setdefault("metadata", {})["name"] = update.name
    if update.description is not None:
        meta["description"] = update.description
        document.setdefault("metadata", {})["description"] = update.description

    meta["updated_at"] = now
    document["metadata"]["updatedAt"] = now

    if update.document is not None:
        # 合并更新文档（保留 metadata）
        incoming = update.document
        if "metadata" in incoming:
            del incoming["metadata"]
        document = {**document, **incoming}

    meta_file.write_text(json.dumps(meta, indent=2))
    doc_file.write_text(json.dumps(document, indent=2))

    return meta

@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """删除工作流"""
    wf_dir = _resolve_workflow_dir(workflow_id)

    shutil.rmtree(wf_dir)
    return {"status": "ok"}

@router.get("/{workflow_id}/document")
async def get_workflow_document(workflow_id: str):
    """获取工作流文档（不含 metadata）"""
    wf_dir = _resolve_workflow_dir(workflow_id)

    doc_file = wf_dir / "document.json"
    if not doc_file.exists():
        return get_default_document(workflow_id, workflow_id, "")

    return json.loads(doc_file.read_text())

@router.put("/{workflow_id}/document")
async def save_workflow_document(workflow_id: str, document: dict):
    """保存完整工作流文档"""
    wf_dir = _resolve_workflow_dir(workflow_id)

    doc_file = wf_dir / "document.json"
    meta_file = wf_dir / "metadata.json"

    # 更新 metadata 的 updated_at
    import time
    now = time.time()

    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        meta["updated_at"] = now
        meta_file.write_text(json.dumps(meta, indent=2))

    # 保存文档
    document.setdefault("metadata", {})["updatedAt"] = now
    doc_file.write_text(json.dumps(document, indent=2))

    return {"status": "ok"}
