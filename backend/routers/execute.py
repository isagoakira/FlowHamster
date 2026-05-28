"""
POST /api/execute — 在本地 subprocess 执行训练代码
POST /api/execute/forward — 执行图的 forward pass，返回张量统计
POST /api/execute/gradients — 分析图的梯度流

所有路由共享 /api/execute 前缀。
超时限制为 60 秒，超时后进程会被强制终止。
"""
import ast
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config.security import ALLOW_RAW_EXECUTION, ALLOWED_IMPORTS, ALLOWED_ENV_VARS

router = APIRouter(prefix="/execute", tags=["execute"])


# ── Request / Response Models ─────────────────────────────────────────────────

class ExecuteRequest(BaseModel):
    """
    本地代码执行请求

    Attributes:
        code: 要执行的 Python 代码字符串（raw 执行，需开启 dev 模式）
        graph: 前端 flow_json（nodes + edges），用于编译执行路径
        input_shape: 可选，输入张量的形状，用于 forward pass 预览
        target_device: 执行设备，"cpu" 或 "cuda"（默认 "cpu"）
        data_loader_config: 可选，数据加载器配置
    """
    code: str | None = None
    graph: dict | None = None
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


# ── Sandboxing helpers ────────────────────────────────────────────────────────

# 内存限制: 512 MB
MEMORY_LIMIT_BYTES = 512 * 1024 * 1024

# 执行超时: 60 秒
EXEC_TIMEOUT = 60


def _validate_imports(code: str) -> list[str]:
    """
    解析代码中的 import 语句，返回不在白名单中的模块列表。
    """
    violations = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # 语法错误会在执行阶段捕获
        return []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if alias.name not in ALLOWED_IMPORTS and top not in ALLOWED_IMPORTS:
                    violations.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if node.module not in ALLOWED_IMPORTS and top not in ALLOWED_IMPORTS:
                    violations.append(node.module)

    return violations


def _build_sandbox_env() -> dict[str, str]:
    """
    构建白名单环境变量，仅保留允许的变量。
    """
    env: dict[str, str] = {}
    for key in ALLOWED_ENV_VARS:
        if key in os.environ:
            env[key] = os.environ[key]
    # 强制关闭用户 site-packages，增强隔离
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run_sandboxed(
    code: str,
    timeout: int = EXEC_TIMEOUT,
    memory_limit: int = MEMORY_LIMIT_BYTES,
) -> ExecuteResponse:
    """
    在隔离的临时目录中以 subprocess 运行代码。

    安全措施：
    1. 工作目录隔离（在临时目录中运行）
    2. 环境变量白名单
    3. 允许导入白名单校验
    4. 超时后强制终止进程树
    5. 内存限制监控（best-effort，依赖 psutil）
    """
    # 1. 导入白名单校验
    violations = _validate_imports(code)
    if violations:
        return ExecuteResponse(
            success=False,
            output="",
            error=f"Disallowed imports detected: {', '.join(violations)}",
        )

    # 2. 创建隔离临时目录
    with tempfile.TemporaryDirectory(prefix="fh_sandbox_") as tmpdir:
        script_path = Path(tmpdir) / "script.py"
        script_path.write_text(code, encoding="utf-8")

        env = _build_sandbox_env()
        cmd = [sys.executable, str(script_path)]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=tmpdir,
            )
        except Exception as e:
            return ExecuteResponse(success=False, output="", error=str(e))

        # 3. 监控线程：超时 + 内存限制
        start_time = time.time()
        killed = False
        kill_reason = ""

        def _watchdog():
            nonlocal killed, kill_reason
            try:
                import psutil  # type: ignore[import-untyped]
            except ImportError:
                psutil = None  # type: ignore[misc]

            while proc.poll() is None and not killed:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    killed = True
                    kill_reason = f"Execution timed out ({timeout}s limit)"
                    _kill_proc_tree(proc.pid)
                    return

                if psutil:
                    try:
                        p = psutil.Process(proc.pid)
                        mem = p.memory_info().rss
                        if mem > memory_limit:
                            killed = True
                            kill_reason = (
                                f"Memory limit exceeded ({mem // (1024 * 1024)}MB > "
                                f"{memory_limit // (1024 * 1024)}MB)"
                            )
                            _kill_proc_tree(proc.pid)
                            return
                    except Exception:
                        pass

                time.sleep(0.5)

        watcher = threading.Thread(target=_watchdog, daemon=True)
        watcher.start()

        try:
            stdout, stderr = proc.communicate(timeout=timeout + 5)
        except subprocess.TimeoutExpired:
            _kill_proc_tree(proc.pid)
            stdout, stderr = "", ""
            killed = True
            kill_reason = f"Execution timed out ({timeout}s limit)"
        finally:
            killed = True  # 通知 watchdog 退出
            watcher.join(timeout=2)
            if proc.poll() is None:
                _kill_proc_tree(proc.pid)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass

        output = stdout + stderr
        if kill_reason:
            return ExecuteResponse(success=False, output=output, error=kill_reason)

        return ExecuteResponse(
            success=proc.returncode == 0,
            output=output,
            error=stderr if proc.returncode != 0 else None,
        )


def _kill_proc_tree(pid: int) -> None:
    """
    递归终止进程树。
    先尝试优雅终止，再强制 kill。
    """
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError:
        psutil = None  # type: ignore[misc]

    if psutil:
        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                try:
                    child.terminate()
                except Exception:
                    pass
            parent.terminate()
            gone, alive = psutil.wait_procs([parent] + list(parent.children(recursive=True)), timeout=2)
            for p in alive:
                try:
                    p.kill()
                except Exception:
                    pass
        except Exception:
            pass
    else:
        # fallback: 仅终止主进程
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
            else:
                os.kill(pid, 9)
        except Exception:
            pass


# ── Compile execution helper ──────────────────────────────────────────────────

def _generate_code_from_graph(graph: dict, options: dict | None = None) -> str:
    """
    使用后端代码生成器将 graph 编译为 Python 代码。
    """
    from backend.services.unified_code_gen import generate
    return generate(graph, options=options or {})


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/", response_model=ExecuteResponse)
async def execute_code(req: ExecuteRequest):
    """
    在本地 subprocess 中执行 Python 代码。

    支持两种模式：
    1. 编译执行（推荐，生产环境）：传入 graph，后端生成代码后执行
    2. Raw 执行（开发模式）：传入 code，需开启 ALLOW_RAW_EXECUTION 开关

    超时限制为 60 秒，超时后进程被强制终止。

    Returns:
        ExecuteResponse: 包含执行结果
    """
    # 编译执行路径：优先处理 graph
    if req.graph is not None:
        try:
            code = _generate_code_from_graph(req.graph)
        except Exception as e:
            return ExecuteResponse(success=False, output="", error=f"Code generation failed: {e}")
        return _run_sandboxed(code)

    # Raw 执行路径：需要 code 且 ALLOW_RAW_EXECUTION 开启
    if req.code is None:
        raise HTTPException(status_code=400, detail="Either 'code' or 'graph' must be provided")

    if not ALLOW_RAW_EXECUTION:
        raise HTTPException(
            status_code=403,
            detail="Raw code execution is disabled. Set FLOWHAMSTER_ALLOW_RAW_EXECUTION=1 to enable (dev only).",
        )

    return _run_sandboxed(req.code)


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
