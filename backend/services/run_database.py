"""
FlowHamster Run Database — SQLite persistence layer for training runs, checkpoints, and metrics.
"""

import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

# Path to the SQLite database (project root/backend/)
DB_PATH = Path(__file__).parent.parent.parent / "backend" / "runs.db"

# Ensure backend directory exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Connection context manager
# ---------------------------------------------------------------------------

@contextmanager
def get_db():
    """Yield a sqlite3 connection with Row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------

def init_db():
    """Create all tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id              TEXT PRIMARY KEY,
                workflow_id     TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'pending',
                config_json     TEXT NOT NULL,
                started_at      TEXT,
                ended_at        TEXT,
                best_metric     TEXT,
                best_metric_value REAL,
                output_dir      TEXT NOT NULL,
                error_message   TEXT
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                id          TEXT PRIMARY KEY,
                run_id      TEXT NOT NULL,
                step        INTEGER NOT NULL,
                metric_value REAL,
                filepath    TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT NOT NULL,
                step        INTEGER NOT NULL,
                phase       TEXT NOT NULL,
                loss        REAL,
                accuracy    REAL,
                learning_rate REAL,
                epoch       INTEGER,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_metrics_run_step ON metrics(run_id, step);
            CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON checkpoints(run_id);
        """)

# ---------------------------------------------------------------------------
# Run CRUD
# ---------------------------------------------------------------------------

def create_run(workflow_id: str, config_json: str, output_dir: str) -> str:
    """Create a new run and return its ID."""
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    with get_db() as conn:
        conn.execute(
            """INSERT INTO runs (id, workflow_id, status, config_json, output_dir)
               VALUES (?, ?, 'pending', ?, ?)""",
            (run_id, workflow_id, config_json, output_dir)
        )
    return run_id


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Get a single run by ID."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


def update_run_output_dir(run_id: str, output_dir: str):
    """Update the output directory for an existing run."""
    with get_db() as conn:
        conn.execute("UPDATE runs SET output_dir = ? WHERE id = ?", (output_dir, run_id))


def list_runs(limit: int = 20, offset: int = 0, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """List runs with optional status filter."""
    with get_db() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM runs WHERE status = ? ORDER BY started_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset)
            )
        else:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
        return [dict(row) for row in rows.fetchall()]


def update_run_status(
    run_id: str,
    status: str,
    error_message: Optional[str] = None,
    best_metric: Optional[str] = None,
    best_metric_value: Optional[float] = None,
):
    """Update run status, optionally setting end time and best metric."""
    now = datetime.now(timezone.utc).isoformat()
    ended_at = now if status in ('completed', 'failed', 'cancelled') else None
    with get_db() as conn:
        conn.execute(
            """UPDATE runs
               SET status = ?, ended_at = COALESCE(?, ended_at),
                   error_message = COALESCE(?, error_message),
                   best_metric = COALESCE(?, best_metric),
                   best_metric_value = COALESCE(?, best_metric_value)
               WHERE id = ?""",
            (status, ended_at, error_message, best_metric, best_metric_value, run_id)
        )


def set_run_started(run_id: str):
    """Mark run as started (transition pending → running)."""
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            "UPDATE runs SET status = 'running', started_at = ? WHERE id = ?",
            (now, run_id)
        )


def delete_run(run_id: str) -> bool:
    """Delete a run and its associated checkpoints/metrics."""
    with get_db() as conn:
        cur = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Checkpoint CRUD
# ---------------------------------------------------------------------------

def save_checkpoint(
    run_id: str,
    step: int,
    metric_value: Optional[float],
    filepath: str,
) -> str:
    """Save a checkpoint record and return its ID."""
    checkpoint_id = f"ckpt_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO checkpoints (id, run_id, step, metric_value, filepath, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (checkpoint_id, run_id, step, metric_value, filepath, now)
        )
    return checkpoint_id


def list_checkpoints(run_id: str) -> List[Dict[str, Any]]:
    """List checkpoints for a run, ordered by step."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY step ASC",
            (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Metrics CRUD
# ---------------------------------------------------------------------------

def save_metric(
    run_id: str,
    step: int,
    phase: str,
    loss: Optional[float],
    accuracy: Optional[float],
    learning_rate: Optional[float],
    epoch: Optional[int],
) -> int:
    """Save a metric record and return its auto-increment ID."""
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO metrics (run_id, step, phase, loss, accuracy, learning_rate, epoch, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, step, phase, loss, accuracy, learning_rate, epoch, now)
        )
        return cur.lastrowid


def get_metrics(run_id: str, phase: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
    """Get metrics for a run, optionally filtered by phase."""
    with get_db() as conn:
        if phase:
            rows = conn.execute(
                "SELECT * FROM metrics WHERE run_id = ? AND phase = ? ORDER BY step ASC LIMIT ?",
                (run_id, phase, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM metrics WHERE run_id = ? ORDER BY step ASC LIMIT ?",
                (run_id, limit)
            ).fetchall()
        return [dict(row) for row in rows]