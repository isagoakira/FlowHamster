"""
POST /api/execute — 在本地 subprocess 执行训练代码
POST /api/execute/forward — 执行图的 forward pass，返回张量统计
"""
from fastapi import APIRouter
from pydantic import BaseModel
import subprocess
import tempfile
import os

router = APIRouter(prefix="/api/execute", tags=["execute"])


class ExecuteRequest(BaseModel):
    code: str
    input_shape: list[int] | None = None
    target_device: str = "cpu"  # "cpu" or "cuda"
    data_loader_config: dict | None = None


class ExecuteResponse(BaseModel):
    success: bool
    output: str
    error: str | None = None


@router.post("/execute", response_model=ExecuteResponse)
async def execute_code(req: ExecuteRequest):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(req.code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout + result.stderr
        return ExecuteResponse(
            success=result.returncode == 0,
            output=output,
            error=result.stderr if result.returncode != 0 else None,
        )
    except subprocess.TimeoutExpired:
        return ExecuteResponse(success=False, output="", error="Execution timed out (60s limit)")
    except Exception as e:
        return ExecuteResponse(success=False, output="", error=str(e))
    finally:
        os.unlink(tmp_path)


# ── Forward pass tensor preview ────────────────────────────────────────────────

class ForwardRequest(BaseModel):
    nodes: list
    edges: list
    input_shape: list[int] = [1, 3, 224, 224]


@router.post("/forward")
async def run_forward(req: ForwardRequest):
    from backend.services.tensor_executor import execute_forward

    flow_json = {"nodes": req.nodes, "edges": req.edges}
    try:
        results = execute_forward(flow_json, input_shape=req.input_shape)
        return {"success": True, "outputs": results}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Gradient analysis ─────────────────────────────────────────────────────────

class GradientRequest(BaseModel):
    nodes: list[dict]
    edges: list[dict]


@router.post("/gradients")
async def get_gradients(req: GradientRequest):
    from backend.services.gradient_analyzer import analyze_gradients

    try:
        result = analyze_gradients({"nodes": req.nodes, "edges": req.edges})
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
