"""
FlowHamster Training Executor — submits and manages async training runs.

Each submission:
1. Writes workflow snapshot to runs/<run_id>/project.json
2. Generates full train.py (model + epoch loop + DataLoader)
3. Runs train.py in a background thread (subprocess)
4. Persists checkpoints + metrics to SQLite via run_database.py
"""

import os
import sys
import json
import signal
import subprocess
import threading
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Resolve project root (backend/ is one level below project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
BACKEND_DIR = PROJECT_ROOT / "backend"
RUNS_DIR = BACKEND_DIR / "runs"

# Ensure runs/ directory exists
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Find a Python interpreter that has torch installed (needed for training scripts)
_PYTHON_BIN: str | None = None

def _get_python_with_torch() -> str:
    """Return the path to a Python interpreter that has torch installed."""
    global _PYTHON_BIN
    if _PYTHON_BIN:
        return _PYTHON_BIN
    candidates = [
        "/opt/anaconda3/bin/python3",
        "/usr/bin/python3",
        "python3",
    ]
    for cand in candidates:
        try:
            import subprocess as _sub
            r = _sub.run([cand, "-c", "import torch"], capture_output=True, timeout=5)
            if r.returncode == 0:
                _PYTHON_BIN = cand
                return cand
        except Exception:
            pass
    # Fallback to current interpreter (will fail with clear error)
    _PYTHON_BIN = sys.executable
    return _PYTHON_BIN

# Import after project root is set
sys.path.insert(0, str(BACKEND_DIR))
from services.run_database import (
    init_db,
    create_run,
    get_run as _get_run,
    set_run_started,
    update_run_status,
    delete_run,
    save_checkpoint,
    save_metric,
    get_db,
    update_run_output_dir,
)

# Alias to avoid name collision with local function
get_run = _get_run
from services.codegen_facade import generate_model

init_db()

# ---------------------------------------------------------------------------
# Process management (for cancellation)
# ---------------------------------------------------------------------------

_processes: dict[str, subprocess.Popen] = {}   # run_id → Popen handle
_process_lock = threading.Lock()


def _signal_cancel(run_id: str):
    """Send SIGINT to a running training process."""
    with _process_lock:
        proc = _processes.get(run_id)
    if proc:
        proc.send_signal(signal.SIGINT)


# ---------------------------------------------------------------------------
# Train.py generation
# ---------------------------------------------------------------------------

def generate_train_script(
    workflow_doc: dict,
    training_config: dict,
    output_dir: Path,
) -> Path:
    """
    Generate a complete train.py for the given workflow document.

    training_config is expected to have keys like:
      epochs, batch_size, optimizer (sgd|adam|adamw),
      learning_rate, momentum, weight_decay,
      scheduler (cosine|step|None), step_size, gamma,
      checkpoint (enabled, save_top_k, monitor, mode, early_stop_patience),
      data_config (train_dir, val_dir, batch_size, num_workers)
    """
    # Build graph from workflow
    graph = workflow_doc.get("modelGraph", workflow_doc.get("graph", {}))

    # Generate model code
    model_code = generate_model(graph)

    # Extract dataset path info
    data_config = training_config.get("dataConfig", {})
    train_dir = data_config.get("trainDir", "")
    val_dir = data_config.get("valDir", "")
    num_workers = data_config.get("numWorkers", 4)
    batch_size = training_config.get("batchSize", 32)

    # Optimizer config
    opt_type = training_config.get("optimizer", "adam").lower()
    lr = training_config.get("learningRate", 0.001)
    momentum = training_config.get("momentum", 0.9)
    weight_decay = training_config.get("weightDecay", 0.0)

    # Scheduler config
    sched_type = training_config.get("scheduler", "cosine").lower()
    step_size = training_config.get("stepSize", 10)
    gamma = training_config.get("gamma", 0.1)
    warmup_epochs = training_config.get("warmupEpochs", 0)

    # Checkpoint config
    ckpt_enabled = training_config.get("checkpoint", {}).get("enabled", True)
    save_top_k = training_config.get("checkpoint", {}).get("saveTopK", 3)
    monitor = training_config.get("checkpoint", {}).get("monitor", "val_loss")
    monitor_mode = training_config.get("checkpoint", {}).get("mode", "min")

    # Build train.py
    train_py = output_dir / "train.py"
    raw_script = _BUILD_TRAIN_SCRIPT(
        model_code=model_code,
        epochs=training_config.get("epochs", 10),
        batch_size=batch_size,
        opt_type=opt_type,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        sched_type=sched_type,
        step_size=step_size,
        gamma=gamma,
        warmup_epochs=warmup_epochs,
        train_dir=train_dir,
        val_dir=val_dir,
        num_workers=num_workers,
        ckpt_enabled=ckpt_enabled,
        save_top_k=save_top_k,
        monitor=monitor,
        monitor_mode=monitor_mode,
        output_dir=str(output_dir),
    )
    # The template uses f-strings that conflict with .format() placeholder syntax.
    # fix: the literal __EPOCHS__ placeholder is protected from .format() and replaced
    # after formatting to produce a correct Python f-string constant.
    # Also fix lowercase bool literals that str(False/True).lower() produces.
    script = (raw_script
        .replace("__EPOCHS__", "EPOCHS")
        .replace("= false", "= False")
        .replace("= true", "= True"))
    train_py.write_text(script, encoding="utf-8")
    return train_py


def _BUILD_TRAIN_SCRIPT(
    model_code: str,
    epochs: int,
    batch_size: int,
    opt_type: str,
    lr: float,
    momentum: float,
    weight_decay: float,
    sched_type: str,
    step_size: int,
    gamma: float,
    warmup_epochs: int,
    train_dir: str,
    val_dir: str,
    num_workers: int,
    ckpt_enabled: bool,
    save_top_k: int,
    monitor: str,
    monitor_mode: str,
    output_dir: str,
) -> str:
    return f'''\
#!/usr/bin/env python3
"""
FlowHamster generated training script.
Generated at: {datetime.now(timezone.utc).isoformat()}
"""
import os, sys, json, time, signal, traceback
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────────
EPOCHS              = {epochs}
BATCH_SIZE          = {batch_size}
OPT_TYPE            = "{opt_type}"
LR                 = {lr}
MOMENTUM            = {momentum}
WEIGHT_DECAY        = {weight_decay}
SCHED_TYPE          = "{sched_type}"
STEP_SIZE           = {step_size}
GAMMA               = {gamma}
WARMUP_EPOCHS       = {warmup_epochs}
TRAIN_DIR           = "{train_dir}"
VAL_DIR             = "{val_dir}"
NUM_WORKERS         = {num_workers}
CKPT_ENABLED        = {str(ckpt_enabled).lower()}
SAVE_TOP_K          = {save_top_k}
MONITOR             = "{monitor}"
MONITOR_MODE        = "{monitor_mode}"
OUTPUT_DIR          = Path("{output_dir}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR      = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ── Metrics logging ─────────────────────────────────────────────────────────────
METRICS_FILE = OUTPUT_DIR / "metrics.jsonl"
METRICS_FILE_WRITE = open(METRICS_FILE, "a", encoding="utf-8")

def log_metric(phase, epoch, step, loss, acc, lr):
    row = json.dumps({{"phase":phase,"epoch":epoch,"step":step,"loss":loss,"acc":acc,"lr":lr,"ts":time.time()}})
    METRICS_FILE_WRITE.write(row + "\\n")
    METRICS_FILE_WRITE.flush()

# ── Load Data ───────────────────────────────────────────────────────────────────
def make_dataset(root):
    if not root or not Path(root).exists():
        # Fallback: random placeholder data
        return None
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    ds = ImageFolder(root, transform=transform)
    return ds

train_ds = make_dataset(TRAIN_DIR)
val_ds   = make_dataset(VAL_DIR)
has_real_data = train_ds is not None

if not has_real_data:
    print("[FlowHamster] WARNING: No real data found. Using random placeholder tensors.", flush=True)
    class RandomDataset(Dataset):
        def __len__(self): return 100
        def __getitem__(self, idx): return torch.randn(3,224,224), idx % 10
    train_ds = RandomDataset()
    val_ds   = RandomDataset()

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

# ── Model ────────────────────────────────────────────────────────────────────────
{model_code}

model = FlowHamsterModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ── Loss & Optimizer ────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

if OPT_TYPE == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)
elif OPT_TYPE == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Scheduler
if SCHED_TYPE == "step":
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
elif SCHED_TYPE == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
else:
    scheduler = None

# ── Checkpoint helpers ──────────────────────────────────────────────────────────
def get_best():
    best_file = OUTPUT_DIR / "best_model.pt"
    if best_file.exists():
        return torch.load(best_file, map_location=device)
    return None

def save_checkpoint(epoch, val_loss, val_acc, is_best):
    torch.save({{"epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss, "val_acc": val_acc}},
               CHECKPOINT_DIR / f"epoch_{{epoch}}.pt")
    if is_best:
        torch.save({{"epoch": epoch, "model_state": model.state_dict(),
                    "val_loss": val_loss, "val_acc": val_acc}},
                   OUTPUT_DIR / "best_model.pt")

def is_better(val):
    best = get_best()
    if best is None: return True
    if MONITOR_MODE == "min": return val < best.get("val_loss", float("inf"))
    return val > best.get("val_acc", 0.0)

# ── Training loop ───────────────────────────────────────────────────────────────
best_val_loss = float("inf")
for epoch in range(EPOCHS):
    # ── Train ────────────────────────────────────────────────────────────
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        log_metric("train", epoch, step, loss.item(),
                   100.*correct/total, optimizer.param_groups[0]["lr"])

    train_loss = running_loss / len(train_loader)
    train_acc  = 100.*correct/total

    # ── Validate ────────────────────────────────────────────────────────
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    val_loss_val = val_loss / len(val_loader)
    val_acc_val  = 100.*val_correct / val_total

    log_metric("val", epoch, 0, val_loss_val, val_acc_val,
               optimizer.param_groups[0]["lr"])
    print(f"[Epoch {{epoch+1}}/__EPOCHS__] train_loss={{train_loss:.4f}} "
          f"train_acc={{train_acc:.2f}}% | val_loss={{val_loss_val:.4f}} val_acc={{val_acc_val:.2f}}%", flush=True)

    # Checkpoint
    is_best = is_better(val_loss_val)
    if CKPT_ENABLED:
        save_checkpoint(epoch, val_loss_val, val_acc_val, is_best)

    if scheduler: scheduler.step()

    # Early stopping
    patience = 10  # TODO: wire from trainingConfig.checkpoint.earlyStopPatience
    if val_loss_val < best_val_loss:
        best_val_loss = val_loss_val
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {{epoch+1}}", flush=True)
            break

METRICS_FILE_WRITE.close()
print("Training complete!", flush=True)
'''


# ---------------------------------------------------------------------------
# Background training thread
# ---------------------------------------------------------------------------

def _run_training_loop(run_id: str, train_py_path: Path):
    """Execute train.py in a subprocess, parse metrics, and persist to DB."""
    run_info = get_run(run_id)
    if not run_info:
        return

    set_run_started(run_id)
    proc = None
    try:
        proc = subprocess.Popen(
            [_get_python_with_torch(), str(train_py_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(train_py_path.parent),
        )
        with _process_lock:
            _processes[run_id] = proc

        # Stream stdout to DB metrics (simple approach: parse after finish)
        # For real-time updates we'd need a Unix pipe; here we parse on completion
        output_lines = []
        for line in proc.stdout:
            output_lines.append(line)
            print(line, end="")  # visible in backend logs

        proc.wait()

        # Parse metrics from metrics.jsonl
        metrics_file = train_py_path.parent / "metrics.jsonl"
        step = 0
        if metrics_file.exists():
            for raw_line in metrics_file.read_text().splitlines():
                try:
                    m = json.loads(raw_line)
                    phase = m.get("phase", "train")
                    epoch = m.get("epoch", 0)
                    save_metric(
                        run_id=run_id,
                        step=step,
                        phase=phase,
                        loss=m.get("loss"),
                        accuracy=m.get("acc"),
                        learning_rate=m.get("lr"),
                        epoch=epoch,
                    )
                    step += 1
                except Exception:
                    pass

        # Determine final status from return code
        if proc.returncode == 0:
            update_run_status(run_id, "completed")
        else:
            error_msg = "\\n".join(output_lines[-20:])  # last 20 lines
            update_run_status(run_id, "failed", error_message=error_msg)

    except Exception as e:
        tb = traceback.format_exc()
        update_run_status(run_id, "failed", error_message=tb)
    finally:
        with _process_lock:
            _processes.pop(run_id, None)
        if proc and proc.poll() is None:
            proc.terminate()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def submit_training_run(workflow_doc: dict, training_config: dict) -> str:
    """
    Create a new training run:
    - Persist run record to SQLite
    - Write project.json + generate train.py
    - Launch background training thread

    Returns the run_id.
    """
    workflow_id = training_config.get("workflowId", "local")
    config_json = json.dumps(training_config, ensure_ascii=False)
    run_id = create_run(workflow_id=workflow_id, config_json=config_json, output_dir="")

    # Create run directory
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Update output_dir in DB record
    from services.run_database import update_run_output_dir
    update_run_output_dir(run_id, str(run_dir))

    # Write project snapshot
    (run_dir / "project.json").write_text(
        json.dumps(workflow_doc, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Generate train.py
    train_py = generate_train_script(workflow_doc, training_config, run_dir)

    # Start background thread
    t = threading.Thread(target=_run_training_loop, args=(run_id, train_py), daemon=True)
    t.start()

    return run_id


def cancel_training_run(run_id: str) -> bool:
    """Send SIGINT to a running training process and mark it cancelled."""
    run_info = get_run(run_id)
    if not run_info:
        return False
    if run_info["status"] != "running":
        return False

    _signal_cancel(run_id)
    update_run_status(run_id, "cancelled")
    return True


def get_training_run(run_id: str) -> Optional[dict]:
    return get_run(run_id)


def list_training_runs(limit: int = 20, offset: int = 0) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        return [dict(row) for row in rows]
