"""
POST /api/execute — 在本地 subprocess 执行后端生成的训练代码
POST /api/execute/forward — 执行图的 forward pass，返回张量统计
POST /api/execute/gradients — 分析图的梯度流

所有路由共享 /api/execute 前缀。
超时限制为 60 秒，超时后进程会被强制终止。
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from copy import deepcopy
from pathlib import Path
import ast
import subprocess
import tempfile
import os
import signal
import sys

from backend.services.codegen_facade import generate_full_code

router = APIRouter(prefix="/execute", tags=["execute"])

EXECUTION_TIMEOUT_SECONDS = 60
RAW_EXECUTION_ENV_FLAG = "FLOWHAMSTER_ALLOW_RAW_EXECUTE"
REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "backend",
    "json",
    "math",
    "numpy",
    "pandas",
    "pathlib",
    "sklearn",
    "torch",
    "torchmetrics",
    "torchvision",
    "typing",
}
ENV_ALLOWLIST = {
    "APPDATA",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "HOME",
    "LOGNAME",
    "PATH",
    "PROGRAMFILES",
    "PYTHONPATH",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TORCH_HOME",
    "USER",
    "USERPROFILE",
    "USERNAME",
    "VIRTUAL_ENV",
    "WINDIR",
}


class ExecuteRequest(BaseModel):
    """
    本地代码执行请求

    Attributes:
        code: 要执行的 Python 代码字符串（兼容旧调用）
        graph: 可选，模型节点图；提供时后端会生成训练脚本并执行
        data_graph: 可选，数据节点图
        bindings: 可选，数据字段到模型输入 / 训练 target 的绑定
        training_config: 可选，训练配置
        input_shape: 可选，输入张量的形状，用于 forward pass 预览
        target_device: 可选执行设备，提供时覆盖 training_config.runtime.device
        data_loader_config: 可选，数据加载器配置
    """
    code: str | None = None
    graph: dict | None = None
    data_graph: dict | None = None
    bindings: list[dict] | None = None
    training_config: dict | None = None
    input_shape: list[int] | None = None
    target_device: str | None = None  # "cpu" or "cuda"
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


def _build_executable_code(req: ExecuteRequest) -> str:
    if req.code is not None:
        if os.getenv(RAW_EXECUTION_ENV_FLAG) != "1":
            raise ValueError(
                "Raw code execution is disabled. Submit graph/data_graph for backend-generated execution, "
                f"or set {RAW_EXECUTION_ENV_FLAG}=1 only in a trusted local development environment."
            )
        return req.code

    if req.graph is None:
        raise ValueError("Either code or graph must be provided.")

    training_config = deepcopy(req.training_config) if isinstance(req.training_config, dict) else req.training_config
    if req.target_device:
        if training_config is None:
            training_config = {"runtime": {"device": req.target_device}}
        else:
            runtime = training_config.setdefault("runtime", {})
            runtime["device"] = req.target_device

    return generate_full_code(
        req.graph,
        training_config=training_config,
        data_graph=req.data_graph,
        bindings=req.bindings,
    ).code


def _validate_allowed_imports(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python syntax: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("Relative imports are not allowed during execution.")
            modules = [node.module or ""]
        else:
            continue

        for module in modules:
            root = module.split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                raise ValueError(f"Import '{root}' is not allowed during execution.")


def _build_execution_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key.upper() in ENV_ALLOWLIST}
    python_path = str(REPO_ROOT)
    if env.get("PYTHONPATH"):
        python_path = os.pathsep.join([python_path, env["PYTHONPATH"]])
    env["PYTHONPATH"] = python_path
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _limit_child_resources() -> None:
    if os.name == "nt":
        return
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (EXECUTION_TIMEOUT_SECONDS + 5, EXECUTION_TIMEOUT_SECONDS + 5))
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    except Exception:
        return


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return
    os.killpg(process.pid, signal.SIGKILL)


def _run_python_script(script_path: Path, cwd: Path) -> tuple[int | None, str, str, bool]:
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(cwd),
        env=_build_execution_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
        start_new_session=os.name != "nt",
        preexec_fn=None if os.name == "nt" else _limit_child_resources,
    )
    try:
        stdout, stderr = process.communicate(timeout=EXECUTION_TIMEOUT_SECONDS)
        return process.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired:
        _terminate_process_tree(process)
        stdout, stderr = process.communicate()
        return None, stdout, stderr, True


@router.post("", response_model=ExecuteResponse)
async def execute_code(req: ExecuteRequest):
    """
    在本地 subprocess 中执行 Python 代码。

    代码会写入临时文件，通过 `python` 命令执行。
    超时限制为 60 秒，超时后进程被强制终止。

    Returns:
        ExecuteResponse: 包含执行结果
    """
    try:
        code = _build_executable_code(req)
        _validate_allowed_imports(code)
    except Exception as e:
        return ExecuteResponse(success=False, output="", error=str(e))

    with tempfile.TemporaryDirectory(prefix="flowhamster_exec_") as tmp:
        tmp_dir = Path(tmp)
        script_path = tmp_dir / "main.py"
        script_path.write_text(code, encoding="utf-8")

        try:
            returncode, stdout, stderr, timed_out = _run_python_script(script_path, tmp_dir)
            output = stdout + stderr
            if timed_out:
                return ExecuteResponse(
                    success=False,
                    output=output,
                    error=f"Execution timed out ({EXECUTION_TIMEOUT_SECONDS}s limit)",
                )
            return ExecuteResponse(
                success=returncode == 0,
                output=output,
                error=stderr if returncode != 0 else None,
            )
        except Exception as e:
            return ExecuteResponse(success=False, output="", error=str(e))


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
