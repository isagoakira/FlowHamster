"""
SSH Training Router

Provides API endpoints for SSH-based remote training control.
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import json

from backend.modules.ssh_client import get_ssh_manager, PARAMIKO_AVAILABLE

router = APIRouter(prefix="/ssh", tags=["ssh"])


# Request/Response Models
class SSHConnectRequest(BaseModel):
    host: str
    username: str
    password: Optional[str] = None
    key_filename: Optional[str] = None
    port: int = 22
    timeout: float = 10.0


class SSHConnectResponse(BaseModel):
    success: bool
    message: str
    connection_key: Optional[str] = None


class SSHExecuteRequest(BaseModel):
    command: str
    cwd: Optional[str] = None
    timeout: Optional[float] = None


class SSHExecuteResponse(BaseModel):
    success: bool
    stdout: str
    stderr: str


class SubmitTrainingRequest(BaseModel):
    script_path: str
    python_env: Optional[str] = None
    gpu_id: Optional[int] = None
    cwd: Optional[str] = None


class SubmitTrainingResponse(BaseModel):
    success: bool
    message: str
    pid: int


class TrainingStatusResponse(BaseModel):
    pid: int
    script_path: str
    status: str
    start_time: float
    gpu_id: Optional[int]
    error: Optional[str]


# Error handling
class SSHError(Exception):
    def __init__(self, message: str, code: str = "ssh_error"):
        self.message = message
        self.code = code
        super().__init__(self.message)


# Check paramiko availability
def check_paramiko():
    if not PARAMIKO_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="paramiko is not installed. Install with: pip install paramiko"
        )


# Endpoints
@router.post("/connect", response_model=SSHConnectResponse)
async def ssh_connect(request: SSHConnectRequest):
    """Establish SSH connection to remote host."""
    check_paramiko()

    manager = get_ssh_manager()
    success, message = manager.connect(
        host=request.host,
        username=request.username,
        password=request.password,
        key_filename=request.key_filename,
        port=request.port,
        timeout=request.timeout,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    key = f"{request.username}@{request.host}:{request.port}"
    return SSHConnectResponse(
        success=True,
        message=message,
        connection_key=key,
    )


@router.post("/disconnect")
async def ssh_disconnect(host: str, username: str):
    """Disconnect SSH connection."""
    check_paramiko()

    manager = get_ssh_manager()
    success, message = manager.disconnect(host=host, username=username)

    return {"success": success, "message": message}


@router.post("/execute", response_model=SSHExecuteResponse)
async def ssh_execute(request: SSHExecuteRequest):
    """Execute command on connected SSH host."""
    check_paramiko()

    manager = get_ssh_manager()
    success, stdout, stderr = manager.execute(
        command=request.command,
        cwd=request.cwd,
        timeout=request.timeout,
    )

    return SSHExecuteResponse(
        success=success,
        stdout=stdout,
        stderr=stderr,
    )


@router.post("/submit-training", response_model=SubmitTrainingResponse)
async def submit_training(request: SubmitTrainingRequest):
    """Submit training job to run on remote host."""
    check_paramiko()

    manager = get_ssh_manager()
    success, message, pid = manager.submit_training(
        script_path=request.script_path,
        python_env=request.python_env,
        gpu_id=request.gpu_id,
        cwd=request.cwd,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return SubmitTrainingResponse(
        success=True,
        message=message,
        pid=pid,
    )


@router.get("/training-status/{pid}", response_model=TrainingStatusResponse)
async def get_training_status(pid: int):
    """Get status of a training job."""
    check_paramiko()

    manager = get_ssh_manager()
    status = manager.get_training_status(pid)

    if status.get("error") == "PID not found":
        raise HTTPException(status_code=404, detail="Training job not found")

    return TrainingStatusResponse(**status)


@router.get("/training-logs/{pid}")
async def get_training_logs(pid: int, max_lines: int = 100):
    """Get recent training log lines."""
    check_paramiko()

    manager = get_ssh_manager()
    logs = manager.get_training_logs(pid, max_lines=max_lines)

    return {"pid": pid, "logs": logs, "count": len(logs)}


@router.post("/stop-training/{pid}")
async def stop_training(pid: int):
    """Stop a running training job."""
    check_paramiko()

    manager = get_ssh_manager()
    success, message = manager.stop_training(pid)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {"success": True, "message": message}


@router.get("/connections")
async def list_connections():
    """List active SSH connections."""
    check_paramiko()

    manager = get_ssh_manager()
    connections = manager.get_active_connections()

    return {"connections": connections, "count": len(connections)}


# WebSocket for real-time log streaming
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass


manager_ws = ConnectionManager()


@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket, pid: int = -1):
    """
    WebSocket endpoint for real-time training log streaming.

    Connect with: ws://host:port/api/ssh/ws/logs?pid=12345
    """
    await websocket.accept()

    if pid > 0:
        ssh_manager = get_ssh_manager()

        async def send_logs():
            last_count = 0
            while True:
                try:
                    status = ssh_manager.get_training_status(pid)
                    logs = ssh_manager.get_training_logs(pid, max_lines=50)

                    if len(logs) > last_count:
                        # Send new logs
                        for log in logs[last_count:]:
                            await websocket.send_json({
                                "type": "log",
                                "pid": pid,
                                "line": log,
                            })
                        last_count = len(logs)

                    if status.get("status") in ("completed", "killed", "error"):
                        await websocket.send_json({
                            "type": "status",
                            "pid": pid,
                            "status": status.get("status"),
                        })
                        break

                    await asyncio.sleep(1)

                except Exception:
                    break

        try:
            await send_logs()
        except WebSocketDisconnect:
            pass

    await websocket.close()