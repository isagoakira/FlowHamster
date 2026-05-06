"""
Connection Pool Module for FlowHamster

Manages SSH and WebSocket connection pools with:
- Connection limits and timeout handling
- Heartbeat/keepalive mechanisms
- Automatic cleanup of stale connections
"""

import asyncio
import threading
import time
import logging
from typing import Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager, contextmanager

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Connection status enum."""
    IDLE = "idle"
    ACTIVE = "active"
    STALE = "stale"
    CLOSED = "closed"


@dataclass
class PoolConfig:
    """Configuration for a connection pool."""
    max_connections: int = 10
    max_idle_seconds: float = 300.0       # 5 minutes
    stale_check_interval: float = 60.0   # 1 minute
    acquire_timeout: float = 30.0        # seconds
    name: str = "pool"


@dataclass
class PooledConnection:
    """Base class for pooled connections."""
    conn_id: str
    status: ConnectionStatus = ConnectionStatus.IDLE
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# SSH Connection Pool
# ─────────────────────────────────────────────────────────────────────────────

class SSHConnectionPool:
    """
    SSH connection pool with limits and automatic cleanup.

    Usage:
        pool = SSHConnectionPool(max_connections=5)
        with pool.acquire(host, username) as client:
            client.exec_command("ls")
    """

    def __init__(
        self,
        max_connections: int = 10,
        max_idle_seconds: float = 300.0,
        stale_check_interval: float = 60.0,
    ):
        self.config = PoolConfig(
            max_connections=max_connections,
            max_idle_seconds=max_idle_seconds,
            stale_check_interval=stale_check_interval,
            name="ssh",
        )
        self._connections: Dict[str, Any] = {}  # key -> SSH connection
        self._statuses: Dict[str, PooledConnection] = {}  # key -> PooledConnection
        self._lock = threading.RLock()
        self._semaphore = threading.Semaphore(max_connections)
        self._stale_task: Optional[threading.Thread] = None
        self._running = False

    def _make_key(self, host: str, port: int, username: str) -> str:
        return f"{username}@{host}:{port}"

    def _is_stale(self, key: str) -> bool:
        """Check if a connection is stale (idle too long)."""
        if key not in self._statuses:
            return True
        status = self._statuses[key]
        idle_time = time.time() - status.last_used
        return idle_time > self.config.max_idle_seconds

    def _cleanup_stale(self):
        """Remove stale connections."""
        with self._lock:
            stale_keys = [
                k for k in self._statuses
                if self._statuses[k].status == ConnectionStatus.STALE
                or (self._statuses[k].status == ConnectionStatus.IDLE and self._is_stale(k))
            ]
            for key in stale_keys:
                if key in self._connections:
                    try:
                        self._connections[key].close()
                    except Exception:
                        pass
                del self._connections[key]
                del self._statuses[key]
                logger.debug(f"Cleaned up stale SSH connection: {key}")

    def _start_stale_checker(self):
        """Start background thread to check for stale connections."""
        def checker():
            while self._running:
                time.sleep(self.config.stale_check_interval)
                if not self._running:
                    break
                self._cleanup_stale()

        self._running = True
        self._stale_task = threading.Thread(target=checker, daemon=True)
        self._stale_task.start()

    def _stop_stale_checker(self):
        """Stop the stale connection checker."""
        self._running = False
        if self._stale_task:
            self._stale_task.join(timeout=5)
            self._stale_task = None

    def connect(
        self,
        host: str,
        username: str,
        password: Optional[str] = None,
        key_filename: Optional[str] = None,
        port: int = 22,
        timeout: float = 10.0,
    ) -> tuple[bool, str, Any]:
        """
        Establish an SSH connection from the pool.

        Returns:
            (success, message, client)
        """
        key = self._make_key(host, port, username)

        with self._lock:
            # Return existing if still valid
            if key in self._connections:
                status = self._statuses.get(key)
                if status and status.status == ConnectionStatus.IDLE:
                    status.last_used = time.time()
                    status.use_count += 1
                    return True, f"Reused existing connection to {key}", self._connections[key]

        # Check semaphore (limits max concurrent)
        acquired = self._semaphore.acquire(timeout=self.config.acquire_timeout)
        if not acquired:
            return False, f"Connection pool exhausted (max={self.config.max_connections})", None

        try:
            import paramiko
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            if key_filename:
                client.connect(
                    hostname=host, port=port, username=username,
                    key_filename=key_filename, timeout=timeout,
                    look_for_keys=True, allow_agent=True,
                )
            elif password:
                client.connect(
                    hostname=host, port=port, username=username,
                    password=password, timeout=timeout,
                    look_for_keys=True, allow_agent=True,
                )
            else:
                client.connect(
                    hostname=host, port=port, username=username,
                    timeout=timeout, look_for_keys=True, allow_agent=True,
                )

            # Enable keepalive
            transport = client.get_transport()
            if transport:
                transport.set_keepalive(30)

            with self._lock:
                self._connections[key] = client
                self._statuses[key] = PooledConnection(
                    conn_id=key,
                    status=ConnectionStatus.IDLE,
                )

            return True, f"Connected to {key}", client

        except Exception as e:
            self._semaphore.release()
            return False, f"Connection failed: {e}", None

    def release(self, host: str, port: int, username: str):
        """Mark a connection as idle (released back to pool)."""
        key = self._make_key(host, port, username)
        with self._lock:
            if key in self._statuses:
                self._statuses[key].status = ConnectionStatus.IDLE
                self._statuses[key].last_used = time.time()

    def close(self, host: str, port: int, username: str):
        """Close and remove a specific connection."""
        key = self._make_key(host, port, username)
        with self._lock:
            if key in self._connections:
                try:
                    self._connections[key].close()
                except Exception:
                    pass
                del self._connections[key]
                del self._statuses[key]
                self._semaphore.release()

    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            for key, conn in self._connections.items():
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()
            self._statuses.clear()
            # Release all semaphores
            for _ in range(self.config.max_connections):
                try:
                    self._semaphore.release()
                except Exception:
                    pass

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "total": len(self._connections),
                "active": sum(1 for s in self._statuses.values() if s.status == ConnectionStatus.ACTIVE),
                "idle": sum(1 for s in self._statuses.values() if s.status == ConnectionStatus.IDLE),
                "max": self.config.max_connections,
            }


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Connection Manager
# ─────────────────────────────────────────────────────────────────────────────

class WebSocketConnectionManager:
    """
    WebSocket connection manager with heartbeat and auto-cleanup.

    Usage:
        manager = WebSocketConnectionManager()

        # Accept connection
        await manager.connect(websocket, client_id)

        # Broadcast to all
        await manager.broadcast({"type": "update"})

        # Remove on disconnect
        manager.disconnect(client_id)
    """

    def __init__(
        self,
        heartbeat_interval: float = 30.0,
        stale_check_interval: float = 60.0,
    ):
        self.heartbeat_interval = heartbeat_interval
        self.stale_check_interval = stale_check_interval
        self._connections: Dict[str, Any] = {}  # client_id -> WebSocket
        self._last_ping: Dict[str, float] = {}   # client_id -> last ping time
        self._lock = asyncio.Lock()
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: Any, client_id: str):
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self._connections[client_id] = websocket
            self._last_ping[client_id] = time.time()
        logger.debug(f"WebSocket connected: {client_id}")

    async def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        async with self._lock:
            self._connections.pop(client_id, None)
            self._last_ping.pop(client_id, None)
        logger.debug(f"WebSocket disconnected: {client_id}")

    async def send_to(self, client_id: str, message: str | Dict[str, Any]):
        """Send message to a specific client."""
        async with self._lock:
            websocket = self._connections.get(client_id)

        if websocket is None:
            return False

        try:
            if isinstance(message, dict):
                import json
                await websocket.send_text(json.dumps(message))
            else:
                await websocket.send_text(message)
            return True
        except Exception:
            # Connection is dead, mark for cleanup
            await self.disconnect(client_id)
            return False

    async def broadcast(self, message: str | Dict[str, Any]):
        """
        Broadcast message to all connected clients.

        Dead connections are automatically removed.
        """
        if isinstance(message, dict):
            import json
            msg_str = json.dumps(message)
        else:
            msg_str = message

        dead_ids = []
        async with self._lock:
            client_ids = list(self._connections.keys())

        for client_id in client_ids:
            websocket = self._connections.get(client_id)
            if websocket is None:
                dead_ids.append(client_id)
                continue

            try:
                await websocket.send_text(msg_str)
            except Exception:
                dead_ids.append(client_id)

        # Cleanup dead connections
        for client_id in dead_ids:
            await self.disconnect(client_id)

        return len(client_ids) - len(dead_ids)

    async def ping(self, client_id: str) -> bool:
        """Send a ping to a client and update last_ping."""
        async with self._lock:
            websocket = self._connections.get(client_id)

        if websocket is None:
            return False

        try:
            import json
            await websocket.send_text(json.dumps({"type": "ping", "timestamp": time.time()}))
            self._last_ping[client_id] = time.time()
            return True
        except Exception:
            await self.disconnect(client_id)
            return False

    async def start_heartbeat(self):
        """Start background heartbeat task."""
        self._running = True

        async def heartbeat_loop():
            while self._running:
                await asyncio.sleep(self.heartbeat_interval)
                if not self._running:
                    break

                async with self._lock:
                    client_ids = list(self._connections.keys())

                for client_id in client_ids:
                    # Check if client is stale (no activity)
                    last = self._last_ping.get(client_id, 0)
                    if time.time() - last > self.heartbeat_interval * 3:
                        await self.disconnect(client_id)
                        continue
                    await self.ping(client_id)

        self._cleanup_task = asyncio.create_task(heartbeat_loop())

    async def stop_heartbeat(self):
        """Stop the heartbeat task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": len(self._connections),
            "heartbeat_interval": self.heartbeat_interval,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Global instances
# ─────────────────────────────────────────────────────────────────────────────

_ssh_pool: Optional[SSHConnectionPool] = None
_ws_manager: Optional[WebSocketConnectionManager] = None


def get_ssh_pool() -> SSHConnectionPool:
    """Get or create the global SSH connection pool."""
    global _ssh_pool
    if _ssh_pool is None:
        _ssh_pool = SSHConnectionPool()
    return _ssh_pool


def get_ws_manager() -> WebSocketConnectionManager:
    """Get or create the global WebSocket connection manager."""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketConnectionManager()
    return _ws_manager
