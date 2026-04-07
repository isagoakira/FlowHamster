from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

WORKFLOW_DOCUMENT_VERSION = "2.0.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class WorkflowPortContract:
    name: str
    dtype: str | None = None
    shape_hint: str | None = None
    semantic_role: str | None = None
    description: str | None = None


@dataclass
class WorkflowGraphSnapshot:
    kind: str
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    contract: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: {"inputs": [], "outputs": []})


@dataclass
class WorkflowComponentConfig:
    type: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowTrainingConfig:
    task_type: str = "classification"
    loss: WorkflowComponentConfig = field(default_factory=lambda: WorkflowComponentConfig(type="cross_entropy"))
    optimizer: WorkflowComponentConfig = field(
        default_factory=lambda: WorkflowComponentConfig(type="adamw", params={"lr": 1e-3, "weight_decay": 1e-2})
    )
    scheduler: WorkflowComponentConfig = field(default_factory=lambda: WorkflowComponentConfig(type="cosine_annealing", enabled=False))
    metrics: list[WorkflowComponentConfig] = field(default_factory=lambda: [WorkflowComponentConfig(type="accuracy")])
    runtime: dict[str, Any] = field(default_factory=lambda: {
        "device": "auto",
        "epochs": 10,
        "batch_size": 32,
        "amp": False,
        "grad_clip": None,
        "num_workers": 4,
    })
    checkpoint: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "save_top_k": 1,
        "monitor": "val_loss",
        "mode": "min",
        "early_stop_patience": None,
    })


@dataclass
class WorkflowMetadata:
    name: str = "FlowHamster Workflow"
    description: str = "Model graph driven workflow document"
    schema_version: str = WORKFLOW_DOCUMENT_VERSION
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    export_source: str = "flowhamster"


@dataclass
class WorkflowDocument:
    version: str = WORKFLOW_DOCUMENT_VERSION
    metadata: WorkflowMetadata = field(default_factory=WorkflowMetadata)
    model_graph: WorkflowGraphSnapshot = field(default_factory=lambda: WorkflowGraphSnapshot(kind="model"))
    data_graph: WorkflowGraphSnapshot = field(default_factory=lambda: WorkflowGraphSnapshot(kind="data"))
    training_config: WorkflowTrainingConfig = field(default_factory=WorkflowTrainingConfig)
    bindings: list[dict[str, Any]] = field(default_factory=list)


def create_workflow_document_from_legacy_graph(
    nodes: list[dict[str, Any]] | None = None,
    edges: list[dict[str, Any]] | None = None,
    *,
    name: str = "FlowHamster Workflow",
    description: str = "Migrated from legacy graph JSON",
) -> WorkflowDocument:
    metadata = WorkflowMetadata(name=name, description=description)
    return WorkflowDocument(
        metadata=metadata,
        model_graph=WorkflowGraphSnapshot(kind="model", nodes=nodes or [], edges=edges or []),
    )

