"""
SSH Client Module for FlowHamster

Provides SSH connection management, remote command execution,
and training job submission with logging.
"""

import threading
import queue
import time
from typing import Optional, Dict, Any, Callable, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    paramiko = None


class ConnectionState(Enum):
    """SSH connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class SSHConnection:
    """SSH connection wrapper with keepalive and reconnect support."""
    client: Optional['paramiko.SSHClient'] = None
    state: ConnectionState = ConnectionState.DISCONNECTED
    host: str = ""
    port: int = 22
    username: str = ""
    last_active: float = field(default_factory=time.time)
    reconnect_attempts: int = 0
    max_reconnect_attempts: int = 3
    keepalive_interval: int = 30  # seconds


@dataclass
class TrainingJob:
    """Training job info."""
    pid: Optional[int] = None
    script_path: str = ""
    working_dir: str = ""
    python_env: str = ""
    gpu_id: Optional[int] = None
    status: str = "pending"
    start_time: float = field(default_factory=time.time)
    output_buffer: List[str] = field(default_factory=list)
    error: Optional[str] = None


class SSHClientManager:
    """
    SSH connection manager with keepalive and auto-reconnect.

    Usage:
        manager = SSHClientManager()
        manager.connect(host="192.168.1.100", username="user", password="pass")
        stdout = manager.execute("python train.py")
        manager.disconnect()
    """

    def __init__(self):
        if not PARAMIKO_AVAILABLE:
            raise ImportError(
                "paramiko is required for SSH functionality. "
                "Install with: pip install paramiko"
            )
        self._connections: Dict[str, SSHConnection] = {}
        self._training_jobs: Dict[int, TrainingJob] = {}
        self._log_queues: Dict[int, queue.Queue] = {}
        self._lock = threading.Lock()

    def _get_connection_key(self, host: str, port: int, username: str) -> str:
        return f"{username}@{host}:{port}"

    def connect(
        self,
        host: str,
        username: str,
        password: Optional[str] = None,
        key_filename: Optional[str] = None,
        port: int = 22,
        timeout: float = 10.0,
    ) -> Tuple[bool, str]:
        """
        Establish SSH connection.

        Args:
            host: Remote host
            username: SSH username
            password: Password (if key-based auth not used)
            key_filename: Path to private key file
            port: SSH port (default 22)
            timeout: Connection timeout in seconds

        Returns:
            (success, message) tuple
        """
        key = self._get_connection_key(host, port, username)

        with self._lock:
            # Check if already connected
            if key in self._connections:
                conn = self._connections[key]
                if conn.state == ConnectionState.CONNECTED:
                    return True, f"Already connected to {key}"

            # Create new connection
            conn = SSHConnection(
                host=host,
                port=port,
                username=username,
                state=ConnectionState.CONNECTING,
            )
            self._connections[key] = conn

        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Try key-based auth first, then password
            if key_filename:
                client.connect(
                    hostname=host,
                    port=port,
                    username=username,
                    key_filename=key_filename,
                    timeout=timeout,
                    look_for_keys=True,
                    allow_agent=True,
                )
            elif password:
                client.connect(
                    hostname=host,
                    port=port,
                    username=username,
                    password=password,
                    timeout=timeout,
                    look_for_keys=True,
                    allow_agent=True,
                )
            else:
                # Try with default keys
                client.connect(
                    hostname=host,
                    port=port,
                    username=username,
                    timeout=timeout,
                    look_for_keys=True,
                    allow_agent=True,
                )

            # Enable keepalive
            client.get_transport().set_keepalive(conn.keepalive_interval)

            with self._lock:
                conn.client = client
                conn.state = ConnectionState.CONNECTED
                conn.last_active = time.time()
                conn.reconnect_attempts = 0

            return True, f"Connected to {host}:{port} as {username}"

        except paramiko.AuthenticationException as e:
            with self._lock:
                conn.state = ConnectionState.ERROR
            return False, f"Authentication failed: {e}"

        except paramiko.SSHException as e:
            with self._lock:
                conn.state = ConnectionState.ERROR
            return False, f"SSH error: {e}"

        except Exception as e:
            with self._lock:
                conn.state = ConnectionState.ERROR
            return False, f"Connection failed: {e}"

    def _ensure_connected(self, key: str) -> Optional[paramiko.SSHClient]:
        """Check connection and attempt reconnect if needed."""
        with self._lock:
            conn = self._connections.get(key)
            if not conn or conn.state != ConnectionState.CONNECTED:
                return None

            # Check if keepalive needed
            if time.time() - conn.last_active > conn.keepalive_interval:
                try:
                    transport = conn.client.get_transport()
                    if transport and transport.is_active():
                        transport.send_ignore()
                        conn.last_active = time.time()
                except Exception:
                    pass

            return conn.client if conn.state == ConnectionState.CONNECTED else None

    def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Tuple[bool, str, str]:
        """
        Execute command on remote host.

        Args:
            command: Command to execute
            cwd: Working directory (optional)
            timeout: Execution timeout (optional)

        Returns:
            (success, stdout, stderr) tuple
        """
        key = None
        for k, conn in self._connections.items():
            if conn.state == ConnectionState.CONNECTED:
                key = k
                break

        if not key:
            return False, "", "No active SSH connection"

        client = self._ensure_connected(key)
        if not client:
            return False, "", "Connection lost or not connected"

        try:
            # Wrap command with cd if cwd specified
            full_cmd = f"cd {cwd} && {command}" if cwd else command

            stdin, stdout, stderr = client.exec_command(
                full_cmd,
                timeout=timeout,
                get_pty=False,
            )

            stdout_data = stdout.read().decode('utf-8', errors='replace')
            stderr_data = stderr.read().decode('utf-8', errors='replace')
            exit_status = stdout.channel.recv_exit_status()

            with self._lock:
                self._connections[key].last_active = time.time()

            return exit_status == 0, stdout_data, stderr_data

        except Exception as e:
            return False, "", str(e)

    def submit_training(
        self,
        script_path: str,
        python_env: Optional[str] = None,
        gpu_id: Optional[int] = None,
        cwd: Optional[str] = None,
        on_output: Optional[Callable[[str], None]] = None,
    ) -> Tuple[bool, str, int]:
        """
        Submit a training job to run in background.

        Args:
            script_path: Path to training script on remote
            python_env: Python environment to activate (optional)
            gpu_id: GPU ID to use (optional)
            cwd: Working directory
            on_output: Callback for log output

        Returns:
            (success, message, pid) tuple
        """
        key = None
        conn = None
        for k, c in self._connections.items():
            if c.state == ConnectionState.CONNECTED:
                key = k
                conn = c
                break

        if not key or not conn:
            return False, "No active SSH connection", -1

        try:
            # Build command
            env_setup = f"source {python_env}/bin/activate && " if python_env else ""
            gpu_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} " if gpu_id is not None else ""
            full_cmd = f"{env_setup}{gpu_cmd}python {script_path}"

            # Submit with nohup and redirect output
            log_file = f"/tmp/flowhamster_training_{int(time.time())}.log"
            remote_cmd = f"cd {cwd or '.'} && nohup {full_cmd} > {log_file} 2>&1 & echo $!"

            stdin, stdout, stderr = conn.client.exec_command(remote_cmd, timeout=10)
            pid_str = stdout.read().decode('utf-8', errors='replace').strip()
            stderr_data = stderr.read().decode('utf-8', errors='replace').strip()

            if not pid_str or not pid_str.isdigit():
                return False, f"Failed to get PID: {stderr_data}", -1

            pid = int(pid_str)

            job = TrainingJob(
                pid=pid,
                script_path=script_path,
                working_dir=cwd or "",
                python_env=python_env or "",
                gpu_id=gpu_id,
                status="running",
            )
            self._training_jobs[pid] = job

            # Start log tailing in background
            q = queue.Queue()
            self._log_queues[pid] = q
            self._start_log_tailing(pid, log_file, q, on_output)

            with self._lock:
                conn.last_active = time.time()

            return True, f"Training job submitted with PID {pid}", pid

        except Exception as e:
            return False, f"Failed to submit training: {e}", -1

    def _start_log_tailing(
        self,
        pid: int,
        log_file: str,
        q: queue.Queue,
        on_output: Optional[Callable[[str], None]],
    ):
        """Background thread to tail training logs."""
        def tail_logs():
            last_pos = 0
            while True:
                # Check if process is still running
                if pid in self._training_jobs:
                    job = self._training_jobs[pid]
                    # Check if process still exists
                    ret = self._check_process(pid)
                    if not ret:
                        job.status = "completed" if job.status == "running" else job.status
                        break

                try:
                    # Read new log lines
                    with open(log_file, 'r') as f:
                        f.seek(last_pos)
                        new_lines = f.readlines()
                        last_pos = f.tell()

                    for line in new_lines:
                        q.put(line)
                        if on_output:
                            on_output(line.rstrip())

                except FileNotFoundError:
                    time.sleep(1)
                    continue
                except Exception:
                    time.sleep(1)
                    continue

                time.sleep(0.5)

        thread = threading.Thread(target=tail_logs, daemon=True)
        thread.start()

    def _check_process(self, pid: int) -> bool:
        """Check if process is still running."""
        key = None
        for k, conn in self._connections.items():
            if conn.state == ConnectionState.CONNECTED:
                key = k
                break

        if not key:
            return False

        conn = self._connections[key]
        try:
            stdin, stdout, stderr = conn.client.exec_command(
                f"ps -p {pid} -o pid=,state= 2>/dev/null | grep -v Z",
                timeout=5,
            )
            output = stdout.read().decode('utf-8', errors='replace')
            return str(pid) in output
        except Exception:
            return False

    def get_training_status(self, pid: int) -> Dict[str, Any]:
        """Get training job status."""
        if pid not in self._training_jobs:
            return {"status": "unknown", "error": "PID not found"}

        job = self._training_jobs[pid]
        is_running = self._check_process(pid)

        return {
            "pid": job.pid,
            "script_path": job.script_path,
            "status": job.status if not is_running else "running",
            "start_time": job.start_time,
            "gpu_id": job.gpu_id,
            "error": job.error,
        }

    def get_training_logs(self, pid: int, max_lines: int = 100) -> List[str]:
        """Get recent training log lines."""
        if pid not in self._log_queues:
            return []

        q = self._log_queues[pid]
        logs = []
        while not q.empty() and len(logs) < max_lines:
            try:
                logs.append(q.get_nowait())
            except queue.Empty:
                break

        return logs

    def stop_training(self, pid: int) -> Tuple[bool, str]:
        """Stop a training job."""
        if pid not in self._training_jobs:
            return False, "Job not found"

        key = None
        for k, conn in self._connections.items():
            if conn.state == ConnectionState.CONNECTED:
                key = k
                break

        if not key:
            return False, "No active connection"

        conn = self._connections[key]
        try:
            # Kill the process
            stdin, stdout, stderr = conn.client.exec_command(
                f"kill -9 {pid} 2>/dev/null; wait {pid} 2>/dev/null",
                timeout=5,
            )
            self._training_jobs[pid].status = "killed"
            return True, f"Job {pid} killed"
        except Exception as e:
            return False, f"Failed to kill job: {e}"

    def disconnect(self, host: Optional[str] = None, username: Optional[str] = None) -> Tuple[bool, str]:
        """
        Disconnect SSH connection(s).

        If host and username provided, disconnect specific connection.
        Otherwise disconnect all.
        """
        if host and username:
            key = self._get_connection_key(host, 22, username)
            with self._lock:
                if key in self._connections:
                    conn = self._connections[key]
                    if conn.client:
                        conn.client.close()
                    conn.state = ConnectionState.DISCONNECTED
                    del self._connections[key]
            return True, f"Disconnected from {key}"
        else:
            with self._lock:
                for conn in self._connections.values():
                    if conn.client:
                        conn.client.close()
                    conn.state = ConnectionState.DISCONNECTED
                self._connections.clear()
            return True, "Disconnected all connections"

    def get_active_connections(self) -> List[Dict[str, Any]]:
        """Get list of active connections."""
        result = []
        with self._lock:
            for key, conn in self._connections.items():
                result.append({
                    "key": key,
                    "host": conn.host,
                    "port": conn.port,
                    "username": conn.username,
                    "state": conn.state.value,
                    "last_active": conn.last_active,
                })
        return result


# Global instance
_ssh_manager: Optional[SSHClientManager] = None


def get_ssh_manager() -> SSHClientManager:
    """Get or create global SSH manager instance."""
    global _ssh_manager
    if _ssh_manager is None:
        _ssh_manager = SSHClientManager()
    return _ssh_manager