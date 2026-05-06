"""
FlowHamster Runs API — CRUD for training runs, checkpoints, and metrics.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Literal

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.run_database import (
    init_db,
    list_runs,
    get_run as db_get_run,
    delete_run as db_delete_run,
    list_checkpoints,
    get_metrics as db_get_metrics,
)
from services.training_executor import (
    submit_training_run,
    cancel_training_run,
    list_training_runs,
)

# Ensure DB schema exists
init_db()

router = APIRouter(prefix="/runs", tags=["runs"])


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class TrainingConfig(BaseModel):
    workflowId: Optional[str] = "local"
    epochs: int = 10
    batchSize: int = 32
    optimizer: str = "adam"
    learningRate: float = 0.001
    momentum: float = 0.9
    weightDecay: float = 0.0
    scheduler: str = "cosine"
    stepSize: int = 10
    gamma: float = 0.1
    warmupEpochs: int = 0
    checkpoint: dict = {"enabled": True, "saveTopK": 3, "monitor": "val_loss", "mode": "min"}
    dataConfig: dict = {"trainDir": "", "valDir": "", "numWorkers": 4}


class SubmitRunRequest(BaseModel):
    workflowDoc: dict          # full WorkflowDocument snapshot
    trainingConfig: TrainingConfig


class RunResponse(BaseModel):
    id: str
    workflow_id: str
    status: str
    config_json: str
    started_at: Optional[str]
    ended_at: Optional[str]
    best_metric: Optional[str]
    best_metric_value: Optional[float]
    output_dir: str
    error_message: Optional[str]


class CheckpointResponse(BaseModel):
    id: str
    run_id: str
    step: int
    metric_value: Optional[float]
    filepath: str
    created_at: str


class MetricResponse(BaseModel):
    id: int
    run_id: str
    step: int
    phase: str
    loss: Optional[float]
    accuracy: Optional[float]
    learning_rate: Optional[float]
    epoch: Optional[int]
    created_at: str


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=List[RunResponse])
async def get_runs(
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = None,
):
    """List all runs, optionally filtered by status."""
    rows = list_runs(limit=limit, offset=offset, status=status)
    return [RunResponse(**row) for row in rows]


@router.post("", response_model=dict)
async def create_run_endpoint(req: SubmitRunRequest):
    """
    Submit a new training run.
    Immediately returns a run_id; training runs asynchronously in background.
    """
    run_id = submit_training_run(req.workflowDoc, req.trainingConfig.model_dump())
    return {"run_id": run_id, "status": "pending"}


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(run_id: str):
    """Get a single run by ID."""
    row = db_get_run(run_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return RunResponse(**row)


@router.delete("/{run_id}")
async def delete_run_endpoint(run_id: str):
    """Delete a run and its associated records."""
    ok = db_delete_run(run_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return {"deleted": True}


@router.get("/{run_id}/cancel", response_model=dict)
async def cancel_run(run_id: str):
    """Cancel a running training run."""
    ok = cancel_training_run(run_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Run not found or not in 'running' state")
    return {"cancelled": True}


@router.get("/{run_id}/checkpoints", response_model=List[CheckpointResponse])
async def get_run_checkpoints(run_id: str):
    """List checkpoints for a run."""
    return [CheckpointResponse(**cp) for cp in list_checkpoints(run_id)]


@router.get("/{run_id}/metrics", response_model=List[MetricResponse])
async def get_run_metrics(run_id: str, phase: Optional[Literal["train", "val"]] = None):
    """Get metrics timeline for a run, optionally filtered by phase."""
    rows = db_get_metrics(run_id, phase=phase)
    return [MetricResponse(**row) for row in rows]


@router.get("/{run_id}/export", response_model=dict)
async def export_run_project(run_id: str):
    """Return the path to the exported project directory."""
    run_info = db_get_run(run_id)
    if not run_info:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    output_dir = run_info.get("output_dir", "")
    if not output_dir:
        raise HTTPException(status_code=404, detail="Output directory not set")
    return {
        "output_dir": output_dir,
        "train_py": f"{output_dir}/train.py",
        "project_json": f"{output_dir}/project.json",
        "checkpoints": f"{output_dir}/checkpoints/",
        "best_model": f"{output_dir}/best_model.pt",
    }