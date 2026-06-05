"""Emit optimizer, loss, scheduler, and training-loop code."""
from __future__ import annotations

from ..graph import NodeBlock

def _gen_training(block: NodeBlock, all_blocks: list) -> list[str]:
    """Generate training code for a single training block (goes in if __name__)."""
    lines: list[str] = []
    f = block.fields
    op = block.op_type
    name = block.output_var

    if op == "adam":
        lr = f.get("lr", 0.001)
        betas = f.get("betas", (0.9, 0.999))
        eps = f.get("eps", 1e-8)
        weight_decay = f.get("weight_decay", 0.0)
        lines.append(f"    # Adam optimizer: {name}")
        lines.append(f"    optimizer_{name} = torch.optim.Adam(model.parameters(), lr={lr}, betas={betas}, eps={eps}, weight_decay={weight_decay})")
    elif op == "adamw":
        lr = f.get("lr", 0.001)
        betas = f.get("betas", (0.9, 0.999))
        eps = f.get("eps", 1e-8)
        weight_decay = f.get("weight_decay", 0.01)
        lines.append(f"    # AdamW optimizer: {name}")
        lines.append(f"    optimizer_{name} = torch.optim.AdamW(model.parameters(), lr={lr}, betas={betas}, eps={eps}, weight_decay={weight_decay})")
    elif op == "sgd":
        lr = f.get("lr", 0.01)
        momentum = f.get("momentum", 0.9)
        weight_decay = f.get("weight_decay", 0.0)
        nesterov = f.get("nesterov", True)
        lines.append(f"    # SGD optimizer: {name}")
        lines.append(f"    optimizer_{name} = torch.optim.SGD(model.parameters(), lr={lr}, momentum={momentum}, weight_decay={weight_decay}, nesterov={nesterov})")
    elif op == "rmsprop":
        lr = f.get("lr", 0.01)
        alpha = f.get("alpha", 0.99)
        eps = f.get("eps", 1e-8)
        weight_decay = f.get("weight_decay", 0.0)
        momentum = f.get("momentum", 0.0)
        lines.append(f"    # RMSprop optimizer: {name}")
        lines.append(f"    optimizer_{name} = torch.optim.RMSprop(model.parameters(), lr={lr}, alpha={alpha}, eps={eps}, weight_decay={weight_decay}, momentum={momentum})")
    elif op == "crossentropyloss":
        weight = f.get("weight", None)
        label_smoothing = f.get("label_smoothing", 0.0)
        lines.append(f"    # CrossEntropyLoss: {name}")
        if weight:
            lines.append(f"    criterion_{name} = nn.CrossEntropyLoss(weight={weight}, label_smoothing={label_smoothing})")
        else:
            lines.append(f"    criterion_{name} = nn.CrossEntropyLoss(label_smoothing={label_smoothing})")
    elif op == "mseloss":
        reduction = f.get("reduction", "mean")
        lines.append(f"    # MSELoss: {name}")
        lines.append(f"    criterion_{name} = nn.MSELoss(reduction='{reduction}')")
    elif op == "cosineannealinglr":
        T_max = f.get("T_max", 10)
        eta_min = f.get("eta_min", 0.0)
        lines.append(f"    # CosineAnnealingLR scheduler: {name}")
        lines.append(f"    scheduler_{name} = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_{name}, T_max={T_max}, eta_min={eta_min})")
    elif op == "steplr":
        step_size = f.get("step_size", 5)
        gamma = f.get("gamma", 0.1)
        lines.append(f"    # StepLR scheduler: {name}")
        lines.append(f"    scheduler_{name} = torch.optim.lr_scheduler.StepLR(optimizer_{name}, step_size={step_size}, gamma={gamma})")
    elif op == "reducelronplateau":
        mode = f.get("mode", "min")
        factor = f.get("factor", 0.1)
        patience = f.get("patience", 3)
        threshold = f.get("threshold", 1e-4)
        lines.append(f"    # ReduceLROnPlateau scheduler: {name}")
        lines.append(f"    scheduler_{name} = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_{name}, mode='{mode}', factor={factor}, patience={patience}, threshold={threshold})")

    return lines




def _py_repr(value):
    if isinstance(value, str):
        if value in ("True", "False", "None"):
            return value
        return repr(value)
    if value is True:
        return "True"
    if value is False:
        return "False"
    if value is None:
        return "None"
    return repr(value)


# ─────────────────────────────────────────────────────────────────
# Loss 函数映射表（与 ast_core.py 保持一致）
# ─────────────────────────────────────────────────────────────────

LOSS_CLASS_MAP = {
    "cross_entropy": "nn.CrossEntropyLoss()",
    "mse": "nn.MSELoss()",
    "bce": "nn.BCEWithLogitsLoss()",
    "bce_logits": "nn.BCEWithLogitsLoss()",
    "dice": "DiceLoss()",
    "focal": "FocalLoss()",
    "lovasz": "LovaszLoss()",
    "tversky": "TverskyLoss()",
    "iou": "IoULoss()",
    "giou": "GIoULoss()",
    "dice_ce": "DiceCELoss()",
    "msssim": "MS_SSIMLoss()",
    "perceptual": "PerceptualLoss()",
    "content": "ContentLoss()",
    "style": "StyleLoss()",
    "smooth_l1": "nn.SmoothL1Loss()",
    "focal_loss": "FocalLoss()",
    "class_balanced": "ClassBalancedLoss()",
    "stft": "STFTLoss()",
    "sdr": "SDRLoss()",
    "sisdr": "SISDRLoss()",
    "mel_spec": "MelSpectrogramLoss()",
    "waveform": "WaveformMSELoss()",
    "multi_res": "MultiResolutionSTFTLoss()",
    "phase": "PhaseLoss()",
    "label_smoothing": "nn.LabelSmoothingLoss()",
    "contrastive": "ContrastiveLoss()",
}

LOSS_SPATIAL_TYPES = {"segmentation", "dice", "lovasz", "tversky", "iou", "giou", "dice_ce"}
LOSS_CLASSIFICATION_TYPES = {
    "cross_entropy",
    "label_smoothing",
    "focal",
    "focal_loss",
    "class_balanced",
    "detection",
    "nlp",
}

FLOWHAMSTER_LOSS_INPUT_HELPER = '''
def flowhamster_prepare_loss_inputs(primary_output, target, loss_type="cross_entropy", task_type="classification"):
    if not torch.is_tensor(primary_output):
        return primary_output, target

    loss_type = str(loss_type or "cross_entropy").lower()
    task_type = str(task_type or "classification").lower()
    class_loss_types = {"cross_entropy", "label_smoothing", "focal", "focal_loss", "class_balanced", "detection", "nlp"}
    spatial_loss_types = {"segmentation", "dice", "lovasz", "tversky", "iou", "giou", "dice_ce"}

    if torch.is_tensor(target):
        target = target.to(primary_output.device)

    if loss_type in {"mse", "regression", "waveform"}:
        if torch.is_tensor(target):
            target = target.float()
            if target.shape != primary_output.shape and target.numel() == primary_output.numel():
                target = target.reshape_as(primary_output)
        return primary_output, target

    is_class_loss = loss_type in class_loss_types or task_type in {"classification", "segmentation"}
    if is_class_loss:
        if torch.is_tensor(target):
            target = target.long()
            if primary_output.dim() > 2 and target.dim() == 1 and task_type != "segmentation" and loss_type not in spatial_loss_types:
                primary_output = primary_output.flatten(2).mean(dim=2)
            elif primary_output.dim() > 2 and target.dim() == primary_output.dim() and target.shape[1] == 1:
                target = target.squeeze(1)
            elif primary_output.dim() <= 2 and target.dim() > 1:
                target = target.reshape(target.shape[0], -1)
                if target.shape[1] == 1:
                    target = target.squeeze(1)
        elif primary_output.dim() > 2 and task_type != "segmentation" and loss_type not in spatial_loss_types:
            primary_output = primary_output.flatten(2).mean(dim=2)

    if torch.is_tensor(target) and primary_output.dim() > 0 and target.dim() > 0 and target.shape[0] != primary_output.shape[0]:
        out_batch = primary_output.shape[0]
        if target.shape[0] > out_batch:
            target = target[:out_batch]
        elif target.shape[0] > 0:
            repeats = (out_batch + target.shape[0] - 1) // target.shape[0]
            target = target.repeat((repeats,) + (1,) * (target.dim() - 1))[:out_batch]

    return primary_output, target
'''

LOSS_CUSTOM_CLASSES = {
    "dice": '''
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0,3,1,2).float()
        intersection = (pred * target_one_hot).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))
        iou = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()
''',
    "focal": '''
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
''',
    "lovasz": '''
class LovaszLoss(nn.Module):
    def forward(self, pred, target):
        return F.cross_entropy(F.softmax(pred, dim=1), target)
''',
    "tversky": '''
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0,3,1,2).float()
        tp = (pred * target_one_hot).sum(dim=(2,3))
        fp = (pred * (1 - target_one_hot)).sum(dim=(2,3))
        fn = ((1 - pred) * target_one_hot).sum(dim=(2,3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()
''',
    "iou": '''
class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0,3,1,2).float()
        intersection = (pred * target_one_hot).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()
''',
    "giou": '''
class GIoULoss(nn.Module):
    def forward(self, pred, target):
        return F.cross_entropy(F.softmax(pred, dim=1), target)
''',
    "dice_ce": '''
class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce = nn.CrossEntropyLoss()
        self.smooth = 1e-6
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0,3,1,2).float()
        intersection = (pred * target_one_hot).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        ce = self.ce(pred, target)
        return self.dice_weight * (1 - dice.mean()) + self.ce_weight * ce
''',
}


def _gen_training_from_config_sections(config: dict | None) -> tuple[list[str], list[str], list[str]]:
    """
    Generate training code from training config.
    Returns a tuple of (custom_class_lines, setup_lines, step_lines).
    """
    if not config:
        return [], [], []

    custom_class_lines: list[str] = []
    setup_lines: list[str] = []
    loss_lines: list[str] = []
    step_lines: list[str] = []
    loss_cfg = config.get("loss", {}) or {}
    optimizer_cfg = config.get("optimizer", {}) or {}
    scheduler_cfg = config.get("scheduler", {}) or {}
    runtime_cfg = config.get("runtime", {}) or {}

    raw_loss_type = loss_cfg.get("type", "cross_entropy")
    loss_type = (
        loss_cfg.get("params", {}).get("lossType", "cross_entropy")
        if raw_loss_type == "single"
        else raw_loss_type
    )
    task_type = str(config.get("taskType", "classification") or "classification")

    def _target_line_for(loss_name: str) -> str:
        normalized = str(loss_name or "cross_entropy").lower()
        if normalized in {"mse", "regression", "waveform"}:
            return "        target = torch.randn_like(primary_output)"
        if normalized in LOSS_SPATIAL_TYPES or task_type == "segmentation":
            return "        target = torch.randint(0, primary_output.shape[1] if primary_output.dim() > 1 else 10, (primary_output.shape[0], *primary_output.shape[2:]), dtype=torch.long, device=primary_output.device)"
        return "        target = torch.randint(0, max(2, primary_output.shape[1] if primary_output.dim() > 1 else 2), (primary_output.shape[0],), dtype=torch.long, device=primary_output.device)"

    def _prepare_loss_line(loss_name: str) -> str:
        return f"    loss_input, loss_target = flowhamster_prepare_loss_inputs(primary_output, target, loss_type={_py_repr(str(loss_name or 'cross_entropy'))}, task_type={_py_repr(task_type)})"

    if loss_cfg.get("enabled", True):
        custom_class_lines.append(FLOWHAMSTER_LOSS_INPUT_HELPER)

    if raw_loss_type == "composite":
        components = loss_cfg.get("params", {}).get("components", [])
        component_types = [str(comp.get("type", "cross_entropy") or "cross_entropy") for comp in components]
        prepare_loss_type = next(
            (name for name in component_types if name in LOSS_CLASSIFICATION_TYPES or name in LOSS_SPATIAL_TYPES),
            component_types[0] if component_types else "cross_entropy",
        )
        for i, comp in enumerate(components):
            comp_type = comp.get("type", "cross_entropy")
            comp_class = LOSS_CLASS_MAP.get(comp_type, "nn.CrossEntropyLoss()")
            if comp_type in LOSS_CUSTOM_CLASSES:
                custom_class_lines.append(LOSS_CUSTOM_CLASSES[comp_type])
            setup_lines.append(f"    loss_fn_{i} = {comp_class}")
        loss_exprs = []
        for i, comp in enumerate(components):
            weight = comp.get("weight", 1.0)
            loss_exprs.append(f"{weight} * loss_fn_{i}(loss_input, loss_target)")
        if loss_exprs:
            loss_lines.append("    if target is None:")
            loss_lines.append(_target_line_for(prepare_loss_type))
            loss_lines.append(_prepare_loss_line(prepare_loss_type))
            loss_lines.append(f"    loss = {' + '.join(loss_exprs)}")
    elif loss_cfg.get("enabled", True):
        if loss_type in LOSS_CUSTOM_CLASSES:
            custom_class_lines.append(LOSS_CUSTOM_CLASSES[loss_type])
        loss_class = LOSS_CLASS_MAP.get(loss_type, "nn.CrossEntropyLoss()")
        setup_lines.append(f"    loss_fn = {loss_class}")
        loss_lines.append("    if target is None:")
        loss_lines.append(_target_line_for(str(loss_type)))
        loss_lines.append(_prepare_loss_line(str(loss_type)))
        loss_lines.append("    loss = loss_fn(loss_input, loss_target)")

    if optimizer_cfg.get("enabled", True):
        optimizer_type = optimizer_cfg.get("type", "adamw")
        optimizer_params = optimizer_cfg.get("params", {}) or {}
        param_str = ", ".join(f"{key}={_py_repr(value)}" for key, value in optimizer_params.items())
        suffix = f", {param_str}" if param_str else ""
        if optimizer_type == "adam":
            setup_lines.append(f"    optimizer = torch.optim.Adam(model.parameters(){suffix})")
        elif optimizer_type == "sgd":
            setup_lines.append(f"    optimizer = torch.optim.SGD(model.parameters(){suffix})")
        elif optimizer_type == "rmsprop":
            setup_lines.append(f"    optimizer = torch.optim.RMSprop(model.parameters(){suffix})")
        else:
            setup_lines.append(f"    optimizer = torch.optim.AdamW(model.parameters(){suffix})")

    if scheduler_cfg.get("enabled"):
        scheduler_type = scheduler_cfg.get("type", "cosine_annealing")
        scheduler_params = scheduler_cfg.get("params", {}) or {}
        param_str = ", ".join(f"{key}={_py_repr(value)}" for key, value in scheduler_params.items())
        prefix = "optimizer"
        args = f"{prefix}, {param_str}" if param_str else prefix
        if scheduler_type in ("step_lr", "steplr"):
            setup_lines.append(f"    scheduler = torch.optim.lr_scheduler.StepLR({args})")
        elif scheduler_type in ("reduce_on_plateau", "reducelronplateau"):
            setup_lines.append(f"    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau({args})")
        else:
            setup_lines.append(f"    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR({args})")

    if loss_cfg.get("enabled", True) and optimizer_cfg.get("enabled", True):
        has_explicit_loss = any(line.strip().startswith("loss = ") for line in loss_lines)
        step_lines.append("    model.train()")
        step_lines.append("    optimizer.zero_grad()")
        step_lines.extend(loss_lines)
        if not has_explicit_loss:
            step_lines.append(_prepare_loss_line(str(loss_type)))
            step_lines.append("    loss = loss_fn(loss_input, loss_target)")
        step_lines.append("    loss.backward()")
        grad_clip = runtime_cfg.get("gradClip")
        if grad_clip is not None:
            step_lines.append(f"    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm={grad_clip})")
        step_lines.append("    optimizer.step()")
        if scheduler_cfg.get("enabled"):
            if scheduler_cfg.get("type") in ("reduce_on_plateau", "reducelronplateau"):
                step_lines.append("    scheduler.step(loss)")
            else:
                step_lines.append("    scheduler.step()")

    return custom_class_lines, setup_lines, step_lines


def _gen_training_from_config(config: dict | None) -> tuple[list[str], list[str]]:
    """
    Backward-compatible training code generator.
    Returns custom class definitions and a one-shot setup+step block.
    """
    custom_class_lines, setup_lines, step_lines = _gen_training_from_config_sections(config)
    return custom_class_lines, setup_lines + step_lines


def _indent_lines(lines: list[str], spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" if line.strip() else line for line in lines)


def _without_legacy_training_summary(lines: list[str]) -> list[str]:
    return [line for line in lines if not line.strip().startswith('print(f"task=')]


