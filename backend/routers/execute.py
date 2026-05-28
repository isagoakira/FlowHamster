"""
POST /api/execute — 在本地 subprocess 执行训练代码
POST /api/execute/forward — 执行图的 forward pass，返回张量统计
POST /api/execute/gradients — 分析图的梯度流

所有路由共享 /api/execute 前缀。
超时限制为 60 秒，超时后进程会被强制终止。
"""
from fastapi import APIRouter
from pydantic import BaseModel
import subprocess
import tempfile
import os

router = APIRouter(prefix="/execute", tags=["execute"])


class ExecuteRequest(BaseModel):
    """
    本地代码执行请求

    Attributes:
        code: 要执行的 Python 代码字符串
        input_shape: 可选，输入张量的形状，用于 forward pass 预览
        target_device: 执行设备，"cpu" 或 "cuda"（默认 "cpu"）
        data_loader_config: 可选，数据加载器配置
    """
    code: str
    input_shape: list[int] | None = None
    target_device: str = "cpu"  # "cpu" or "cuda"
    data_loader_config: dict | None = None


class ExecuteResponse(BaseModel):
    """
    本地代码执行响应

    Attributes:
        success: 是否成功执行（返回码为 0）
        output: 标准输出 + 标准错误的合并内容
        error: 如果失败，包含错误信息
    """
    success: bool
    output: str
    error: str | None = None


@router.post("/execute", response_model=ExecuteResponse)
async def execute_code(req: ExecuteRequest):
    """
    在本地 subprocess 中执行 Python 代码。

    代码会写入临时文件，通过 `python` 命令执行。
    超时限制为 60 秒，超时后进程被强制终止。

    Returns:
        ExecuteResponse: 包含执行结果
    """
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
        error = result.stderr if result.returncode != 0 else None
        # 避免将 traceback 泄露到前端，只保留最后一行异常信息
        if error and "Traceback" in error:
            lines = [ln for ln in error.strip().splitlines() if ln.strip()]
            error = lines[-1] if lines else error
        return ExecuteResponse(
            success=result.returncode == 0,
            output=output,
            error=error,
        )
    except subprocess.TimeoutExpired:
        return ExecuteResponse(success=False, output="", error="Execution timed out (60s limit)")
    except Exception as e:
        return ExecuteResponse(success=False, output="", error=str(e))
    finally:
        os.unlink(tmp_path)


# ── Forward pass tensor preview ────────────────────────────────────────────────

class ForwardRequest(BaseModel):
    """
    Forward pass 执行请求

    Attributes:
        nodes: 节点列表（flow_json.nodes）
        edges: 边列表（flow_json.edges）
        input_shape: 输入张量形状，默认 [1, 3, 224, 224]
    """
    nodes: list
    edges: list
    input_shape: list[int] = [1, 3, 224, 224]


@router.post("/forward")
async def run_forward(req: ForwardRequest):
    """
    执行图的 forward pass，返回各节点输出张量的统计信息。

    用于在浏览器中预览数据流经网络后的张量形状和数值范围。
    不执行梯度计算，仅做前向传播。

    Returns:
        dict: {"success": bool, "outputs": [...]}
    """
    from backend.services.tensor_executor import execute_forward

    flow_json = {"nodes": req.nodes, "edges": req.edges}
    try:
        results = execute_forward(flow_json, input_shape=req.input_shape)
        return {"success": True, "outputs": results}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Gradient analysis ─────────────────────────────────────────────────────────

class GradientRequest(BaseModel):
    """
    梯度分析请求

    Attributes:
        nodes: 节点列表
        edges: 边列表
    """
    nodes: list[dict]
    edges: list[dict]


@router.post("/gradients")
async def get_gradients(req: GradientRequest):
    """
    分析图中各节点的梯度流。

    返回每个参数张量的梯度范数，用于判断梯度消失/爆炸情况。

    Returns:
        dict: {"success": bool, "data": {...}}
    """
    from backend.services.gradient_analyzer import analyze_gradients

    try:
        result = analyze_gradients({"nodes": req.nodes, "edges": req.edges})
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
