from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Optional


@dataclass
class CompiledDataflow:
    output_fields: list[str]
    model_input_bindings: list[dict[str, str]]
    target_binding_source: str | None
    primary_model_input_key: str | None
    warnings: list[str]
    summary_lines: list[str]
    python_scaffold: str
    has_workflow_runtime: bool


def _safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value).strip("_").lower()
    return token or "field"


def _python_value(value) -> str:
    if isinstance(value, str):
        if value in ("True", "False", "None"):
            return value
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    return str(value)


def _python_bool(value, fallback: bool = False) -> str:
    if value is None:
        value = fallback
    if isinstance(value, str):
        return "True" if value.strip().lower() in {"1", "true", "yes", "on"} else "False"
    return "True" if bool(value) else "False"


def _infer_tensor_expression(field: str, task_type: str | None = None) -> str:
    lowered = field.lower()
    if any(keyword in lowered for keyword in ("label", "target", "class")):
        if task_type == "regression":
            return "torch.randn(1, 1, device=device)"
        return "torch.randint(0, 10, (1,), dtype=torch.long, device=device)"
    if "mask" in lowered:
        return "torch.randint(0, 2, (1, 1, 224, 224), dtype=torch.long, device=device)"
    if any(keyword in lowered for keyword in ("token", "text", "ids")):
        return "torch.randint(0, 1000, (1, 32), dtype=torch.long, device=device)"
    if any(keyword in lowered for keyword in ("image", "input", "feature")):
        return "torch.randn(1, 3, 224, 224, device=device)"
    return "torch.randn(1, 8, device=device)"


# ─────────────────────────────────────────────────────────────────
# 数据节点代码生成器
# ─────────────────────────────────────────────────────────────────

@dataclass
class DataNodeCodeGen:
    init_code: list[str] = field(default_factory=list)
    getitem_code: list[str] = field(default_factory=list)
    field_tracking: dict[str, str] = field(default_factory=dict)


def _generate_data_node_code(node: dict, node_index: int) -> DataNodeCodeGen:
    node_type = node.get("data", {}).get("nodeType", "")
    params = node.get("data", {}).get("params", {}) or {}
    result = DataNodeCodeGen()

    node_id = _safe_token(node.get("id", f"node_{node_index}"))

    if node_type == "folder_source":
        path = _python_value(params.get("path", "./data/images"))
        pattern = _python_value(params.get("pattern", "*.jpg"))
        result.init_code.extend([
            f"        self._{node_id}_paths = sorted(Path({path}).glob({pattern}))",
            f"        self._{node_id}_cache = {{}}"
        ])
        result.getitem_code.extend([
            f"        # Folder Source: {path}",
            f"        img_path = self._{node_id}_paths[index % len(self._{node_id}_paths)]",
            f"        img_tensor = torchvision.io.read_image(str(img_path)).float() / 255.0",
            f"        _field_image = img_tensor",
            f"        _field_path = str(img_path)"
        ])
        result.field_tracking["image"] = "_field_image"
        result.field_tracking["path"] = "_field_path"

    elif node_type == "csv_source":
        path = _python_value(params.get("path", "./data/train.csv"))
        delimiter = _python_value(params.get("delimiter", ","))
        label_column = str(params.get("label_column", "label") or "label").strip() or "label"
        feature_columns = [
            column.strip()
            for column in str(params.get("feature_columns", "") or "").split(",")
            if column.strip()
        ]
        feature_columns_expr = (
            f"[{', '.join(_python_value(column) for column in feature_columns)}]"
            if feature_columns
            else f"[col for col in self._{node_id}_columns if col != self._{node_id}_label_column and col.lower() not in ('label', 'class', 'target')]"
        )
        result.init_code.extend([
            f"        self._{node_id}_df = pd.read_csv({path}, delimiter={delimiter})",
            f"        self._{node_id}_columns = list(self._{node_id}_df.columns)",
            f"        self._{node_id}_label_column = {_python_value(label_column)}",
            f"        self._{node_id}_feature_columns = {feature_columns_expr}",
            f"        self._{node_id}_cache = {{}}"
        ])
        result.getitem_code.extend([
            f"        # CSV Source: {path}",
            f"        row = self._{node_id}_df.iloc[index % len(self._{node_id}_df)] if len(self._{node_id}_df) > 0 else {{}}",
            f"        _field_data = row.to_dict() if hasattr(row, 'to_dict') else dict(row)",
            f"        _feature_values = []",
            f"        if self._{node_id}_label_column in row and not pd.isna(row[self._{node_id}_label_column]):",
            f"            _field_label = int(float(row[self._{node_id}_label_column]))",
            f"        for col in self._{node_id}_feature_columns:",
            f"            numeric_val = 0.0",
            f"            val = row[col] if col in row else None",
            f"            if val is None or pd.isna(val):",
            f"                _feature_values.append(numeric_val)",
            f"                continue",
            f"            try:",
            f"                numeric_val = float(val)",
            f"            except Exception:",
            f"                numeric_val = 0.0",
            f"            _feature_values.append(numeric_val)",
            f"            locals()[f'_field_{{col}}'] = torch.tensor(numeric_val, dtype=torch.float32)",
            f"            _field_names.append(f'_field_{{col}}')",
            f"        _field_features = torch.tensor(_feature_values, dtype=torch.float32) if _feature_values else torch.zeros(1, dtype=torch.float32)",
        ])
        result.field_tracking["label"] = "_field_label"
        result.field_tracking["features"] = "_field_features"
        result.field_tracking["data"] = "_field_data"

    elif node_type == "jsonl_source":
        path = _python_value(params.get("path", "./data/train.jsonl"))
        result.init_code.extend([
            f"        self._{node_id}_path = {path}",
            f"        self._{node_id}_cache = {{}}"
        ])
        result.getitem_code.extend([
            f"        # JSONL Source: {path}",
            f"        with open(self._{node_id}_path, 'r') as f:",
            f"            lines = f.readlines()",
            f"        record = json.loads(lines[index % len(lines)])",
            f"        _field_record = record"
        ])
        result.field_tracking["record"] = "_field_record"

    elif node_type == "read_image":
        mode = _python_value(params.get("mode", "RGB"))
        result.getitem_code.extend([
            f"        # Read Image (mode={mode})",
            f"        if '_field_image' in dir():",
            f"            _field_image = torchvision.io.read_image(str(_field_image)).float() / 255.0",
            f"            if {mode} == 'RGB' and _field_image.shape[0] != 3:",
            f"                _field_image = _field_image.repeat(3, 1, 1)[:3, :, :]"
        ])
        result.field_tracking["image"] = "_field_image"
        result.field_tracking["path"] = "_field_path"

    elif node_type == "read_lines":
        encoding = _python_value(params.get("encoding", "utf-8"))
        result.getitem_code.extend([
            f"        # Read Lines (encoding={encoding})",
            f"        if '_field_path' in dir():",
            f"            with open(_field_path, 'r', encoding={encoding}) as f:",
            f"                _field_text = f.read().splitlines()"
        ])
        result.field_tracking["text"] = "_field_text"

    elif node_type == "unpack":
        key = _python_value(params.get("key", "payload"))
        result.getitem_code.extend([
            f"        # Unpack: {key}",
            f"        if '_field_record' in dir():",
            f"            _unpacked = _field_record.get({key}, {{}})",
            f"            for k, v in _unpacked.items():",
            f"                locals()[f'_field_{{k}}'] = v"
        ])

    elif node_type == "select_fields":
        fields = str(params.get("fields", "image,label")).split(",")
        fields = [f.strip() for f in fields if f.strip()]
        result.getitem_code.append(f"        # Select Fields: {', '.join(fields)}")
        for fld in fields:
            result.field_tracking[fld] = f"_field_{fld}"
            result.getitem_code.append(f"        _selected_{fld} = locals().get('_field_{fld}')")

    elif node_type == "rename_fields":
        mapping = str(params.get("mapping", "")).split(",")
        result.getitem_code.append("        # Rename Fields")
        for pair in mapping:
            parts = pair.split(":")
            if len(parts) == 2:
                from_field, to_field = parts[0].strip(), parts[1].strip()
                if from_field and to_field:
                    result.getitem_code.append(
                        f"        if '_field_{from_field}' in dir(): _field_{to_field} = _field_{from_field}"
                    )
                    result.field_tracking[to_field] = f"_field_{to_field}"

    elif node_type == "map_transform":
        transform = str(params.get("transform", ""))
        result.getitem_code.append(f"        # Map Transform: {transform}")
        transforms = [t.strip() for t in transform.split("|") if t.strip()]
        for t in transforms:
            t_lower = t.lower()
            if t_lower == "resize":
                result.getitem_code.extend([
                    "        if '_field_image' in dir():",
                    "            _field_image = torch.nn.functional.interpolate(_field_image.unsqueeze(0), size=(224, 224)).squeeze(0)"
                ])
            elif t_lower == "normalize":
                result.getitem_code.extend([
                    "        if '_field_image' in dir():",
                    "            _field_image = (_field_image - 0.5) / 0.5"
                ])
            elif t_lower == "random_flip":
                result.getitem_code.extend([
                    "        if '_field_image' in dir() and torch.rand(1) > 0.5:",
                    "            _field_image = torch.flip(_field_image, dims=[2])"
                ])
            elif t_lower == "random_crop":
                result.getitem_code.extend([
                    "        if '_field_image' in dir():",
                    "            h, w = _field_image.shape[1:];",
                    "            if h > 224 and w > 224:",
                    "                top = torch.randint(0, h - 224, (1,)).item()",
                    "                left = torch.randint(0, w - 224, (1,)).item()",
                    "                _field_image = _field_image[:, top:top+224, left:left+224]"
                ])

    elif node_type == "train_val_split":
        train_ratio = params.get("train_ratio", 0.8)
        result.getitem_code.append(f"        # Train/Val Split (train_ratio={train_ratio})")

    elif node_type == "shuffle":
        enabled = params.get("enabled", True)
        seed = params.get("seed", 42)
        if enabled:
            result.getitem_code.append(f"        # Shuffle (seed={seed}) - apply at DataLoader level")

    elif node_type == "batch":
        batch_size = params.get("batch_size", 32)
        result.getitem_code.append(f"        # Batch (batch_size={batch_size}) - handled by DataLoader")

    elif node_type == "collate":
        strategy = _python_value(params.get("strategy", "default"))
        result.getitem_code.append(f"        # Collate (strategy={strategy}) - handled by DataLoader collate_fn")

    elif node_type == "dataloader":
        batch_size = int(params.get("batch_size", 32))
        shuffle = params.get("shuffle", True)
        num_workers = int(params.get("num_workers", 4))
        pin_memory = params.get("pin_memory", True)
        drop_last = params.get("drop_last", False)
        result.getitem_code.append(
            f"        # DataLoader (batch_size={batch_size}, shuffle={shuffle}, num_workers={num_workers}, pin_memory={pin_memory})"
        )
        result.init_code.extend([
            f"        self._dataloader_config_{node_id} = {{",
            f"            'batch_size': {batch_size},",
            f"            'shuffle': {_python_bool(shuffle, True)},",
            f"            'num_workers': {num_workers},",
            f"            'pin_memory': {_python_bool(pin_memory, True)},",
            f"            'drop_last': {_python_bool(drop_last, False)},",
            f"        }}",
        ])
        result.field_tracking["dataloader"] = "_dataloader"

    # === Existing Augmentation (frontend registry already has these) ===
    elif node_type == "random_horizontal_flip":
        p = float(params.get("p", 0.5))
        result.getitem_code.extend([
            f"        # Random Horizontal Flip (p={p})",
            f"        if '_field_image' in dir() and torch.rand(1) < {p}:",
            f"            _field_image = torch.flip(_field_image, dims=[2])",
        ])
        result.field_tracking["image"] = "_field_image"

    elif node_type == "random_vertical_flip":
        p = float(params.get("p", 0.5))
        result.getitem_code.extend([
            f"        # Random Vertical Flip (p={p})",
            f"        if '_field_image' in dir() and torch.rand(1) < {p}:",
            f"            _field_image = torch.flip(_field_image, dims=[1])",
        ])
        result.field_tracking["image"] = "_field_image"

    elif node_type == "random_crop":
        size = int(params.get("size", 224))
        padding = int(params.get("padding", 4))
        result.getitem_code.extend([
            f"        # Random Crop (size={size}, padding={padding})",
            f"        if '_field_image' in dir():",
            f"            c, h, w = _field_image.shape",
            f"            if h > {size} and w > {size}:",
            f"                top = torch.randint(0, h - {size}, (1,)).item()",
            f"                left = torch.randint(0, w - {size}, (1,)).item()",
            f"                _field_image = _field_image[:, top:top+{size}, left:left+{size}]",
            f"            elif {padding} > 0:",
            f"                pad = torch.nn.functional.pad(_field_image, [{padding}]*4, mode='reflect')",
            f"                top = torch.randint(0, pad.shape[1] - {size}, (1,)).item()",
            f"                left = torch.randint(0, pad.shape[2] - {size}, (1,)).item()",
            f"                _field_image = pad[:, top:top+{size}, left:left+{size}]",
        ])
        result.field_tracking["image"] = "_field_image"

    elif node_type == "random_rotation":
        degrees = float(params.get("degrees", 15))
        result.getitem_code.extend([
            f"        # Random Rotation (degrees={degrees})",
            f"        if '_field_image' in dir():",
            f"            angle = torch.rand(1).item() * 2 * {degrees} - {degrees}",
            f"            theta = torch.tensor([[torch.cos(torch.tensor(angle * 3.14159/180)), -torch.sin(torch.tensor(angle * 3.14159/180)), 0],",
            f"                                 [torch.sin(torch.tensor(angle * 3.14159/180)), torch.cos(torch.tensor(angle * 3.14159/180)), 0],",
            f"                                 [0, 0, 1]], dtype=torch.float32)",
            f"            grid = torch.nn.functional.affine_grid(theta.unsqueeze(0), _field_image.unsqueeze(0).shape, align_corners=False)",
            f"            _field_image = torch.nn.functional.grid_sample(_field_image.unsqueeze(0), grid, align_corners=False).squeeze(0)",
        ])
        result.field_tracking["image"] = "_field_image"

    elif node_type == "color_jitter":
        b = float(params.get("brightness", 0.2))
        c = float(params.get("contrast", 0.2))
        s = float(params.get("saturation", 0.2))
        h = float(params.get("hue", 0.1))
        result.getitem_code.extend([
            f"        # Color Jitter (b={b}, c={c}, s={s}, h={h})",
            f"        if '_field_image' in dir() and _field_image.shape[0] == 3:",
            f"            _field_image = torch.clamp(_field_image * (1 + torch.rand(1) * {b * 2} - {b}), 0, 1)",
        ])
        result.field_tracking["image"] = "_field_image"

    elif node_type == "random_erasing":
        p = float(params.get("p", 0.5))
        scale = str(params.get("scale", "0.02,0.33"))
        ratio = str(params.get("ratio", "0.3,3.3"))
        scale_parts = [p.strip() for p in scale.split(",")]
        ratio_parts = [p.strip() for p in ratio.split(",")]
        scale_min = float(scale_parts[0]) if scale_parts else 0.02
        scale_max = float(scale_parts[1]) if len(scale_parts) > 1 else 0.33
        ratio_min = float(ratio_parts[0]) if ratio_parts else 0.3
        ratio_max = float(ratio_parts[1]) if len(ratio_parts) > 1 else 3.3
        result.getitem_code.extend([
            f"        # Random Erasing (p={p}, scale=[{scale_min},{scale_max}], ratio=[{ratio_min},{ratio_max}])",
            f"        if '_field_image' in dir() and torch.rand(1) < {p}:",
            f"            _area = _field_image.shape[1] * _field_image.shape[2]",
            f"            _erase_area = torch.empty(1).uniform_({scale_min}, {scale_max}).item() * _area",
            f"            _aspect_ratio = torch.empty(1).uniform_({ratio_min}, {ratio_max}).item()",
            f"            _eh = int(round((_erase_area * _aspect_ratio) ** 0.5))",
            f"            _ew = int(round((_erase_area / _aspect_ratio) ** 0.5))",
            f"            if _eh < _field_image.shape[1] and _ew < _field_image.shape[2]:",
            f"                _ey = torch.randint(0, _field_image.shape[1] - _eh, (1,)).item()",
            f"                _ex = torch.randint(0, _field_image.shape[2] - _ew, (1,)).item()",
            f"                _field_image[:, _ey:_ey+_eh, _ex:_ex+_ew] = torch.randn(_field_image.shape[0], _eh, _ew) * 0.5",
        ])
        result.field_tracking["image"] = "_field_image"

    elif node_type == "gaussian_blur":
        kernel_size = int(params.get("kernel_size", 5))
        sigma = str(params.get("sigma", "1.0,2.0"))
        sigma_parts = [p.strip() for p in sigma.split(",")]
        sigma_min = float(sigma_parts[0]) if sigma_parts else 1.0
        sigma_max = float(sigma_parts[1]) if len(sigma_parts) > 1 else 2.0
        result.getitem_code.extend([
            f"        # Gaussian Blur (kernel_size={kernel_size}, sigma=[{sigma_min},{sigma_max}])",
            f"        if '_field_image' in dir():",
            f"            _sigma = torch.empty(1).uniform_({sigma_min}, {sigma_max}).item()",
            f"            _ks = {kernel_size} if {kernel_size} % 2 == 1 else {kernel_size} + 1",
            f"            _field_image = torchvision.transforms.functional.gaussian_blur(_field_image, _ks, [_sigma])",
        ])
        result.field_tracking["image"] = "_field_image"

    elif node_type == "grayscale":
        result.getitem_code.extend([
            f"        # Grayscale",
            f"        if '_field_image' in dir() and _field_image.shape[0] == 3:",
            f"            _field_image = 0.299 * _field_image[0:1] + 0.587 * _field_image[1:2] + 0.114 * _field_image[2:3]",
        ])
        result.field_tracking["image"] = "_field_image"

    # === Advanced Augmentation ===
    elif node_type == "mixup":
        alpha = float(params.get("alpha", 0.2))
        result.getitem_code.extend([
            f"        # MixUp (alpha={alpha})",
            f"        if '_field_image' in dir():",
            f"            _lam = torch.distributions.Beta({alpha}, {alpha}).sample().item()",
            f"            _idx2 = torch.randint(0, len(self), (1,)).item()",
            f"            _sample2 = self[_idx2]",
            f"            if 'image' in _sample2:",
            f"                _field_image = _lam * _field_image + (1 - _lam) * _sample2['image']",
            f"            if 'label' in _sample2 and '_field_label' in dir():",
            f"                _field_label = _lam * _field_label + (1 - _lam) * _sample2['label']",
        ])
        result.field_tracking["image"] = "_field_image"
        result.field_tracking["label"] = "_field_label"

    elif node_type == "cutmix":
        alpha = float(params.get("alpha", 1.0))
        result.getitem_code.extend([
            f"        # CutMix (alpha={alpha})",
            f"        if '_field_image' in dir():",
            f"            _lam = torch.distributions.Beta({alpha}, {alpha}).sample().item()",
            f"            _idx2 = torch.randint(0, len(self), (1,)).item()",
            f"            _sample2 = self[_idx2]",
            f"            if 'image' in _sample2:",
            f"                _h, _w = _field_image.shape[1:]",
            f"                _cut_ratio = (1 - _lam) ** 0.5",
            f"                _cut_h = int(_h * _cut_ratio)",
            f"                _cut_w = int(_w * _cut_ratio)",
            f"                _cy = torch.randint(0, _h, (1,)).item()",
            f"                _cx = torch.randint(0, _w, (1,)).item()",
            f"                _y1 = max(0, _cy - _cut_h // 2)",
            f"                _y2 = min(_h, _cy + _cut_h // 2)",
            f"                _x1 = max(0, _cx - _cut_w // 2)",
            f"                _x2 = min(_w, _cx + _cut_w // 2)",
            f"                _field_image[:, _y1:_y2, _x1:_x2] = _sample2['image'][:, _y1:_y2, _x1:_x2]",
            f"                _lam = 1 - ((_y2 - _y1) * (_x2 - _x1) / (_h * _w))",
            f"            if 'label' in _sample2 and '_field_label' in dir():",
            f"                _field_label = _lam * _field_label + (1 - _lam) * _sample2['label']",
        ])
        result.field_tracking["image"] = "_field_image"
        result.field_tracking["label"] = "_field_label"

    elif node_type == "autoaugment":
        policy = str(params.get("policy", "imagenet")).upper()
        result.getitem_code.extend([
            f"        # AutoAugment (policy={policy})",
            f"        if '_field_image' in dir():",
            f"            from torchvision.transforms import autoaugment",
            f"            _aa_policy = autoaugment.AutoAugmentPolicy.{policy}",
            f"            _aa_transform = autoaugment.AutoAugment(policy=_aa_policy)",
            f"            from PIL import Image",
            f"            _pil_img = Image.fromarray((_field_image.permute(1,2,0).cpu().numpy() * 255).astype('uint8'))",
            f"            _pil_aug = _aa_transform(_pil_img)",
            f"            _field_image = torch.from_numpy(np.array(_pil_aug)).permute(2,0,1).float() / 255.0",
        ])
        result.field_tracking["image"] = "_field_image"

    elif node_type == "randaugment":
        num_ops = int(params.get("num_ops", 2))
        magnitude = int(params.get("magnitude", 9))
        result.getitem_code.extend([
            f"        # RandAugment (num_ops={num_ops}, magnitude={magnitude})",
            f"        if '_field_image' in dir():",
            f"            from torchvision.transforms import RandAugment",
            f"            _ra_transform = RandAugment(num_ops={num_ops}, magnitude={magnitude})",
            f"            from PIL import Image",
            f"            _pil_img = Image.fromarray((_field_image.permute(1,2,0).cpu().numpy() * 255).astype('uint8'))",
            f"            _pil_aug = _ra_transform(_pil_img)",
            f"            _field_image = torch.from_numpy(np.array(_pil_aug)).permute(2,0,1).float() / 255.0",
        ])
        result.field_tracking["image"] = "_field_image"

    elif node_type == "cutout":
        hole_size = int(params.get("hole_size", 16))
        result.getitem_code.extend([
            f"        # CutOut (hole_size={hole_size})",
            f"        if '_field_image' in dir():",
            f"            _h, _w = _field_image.shape[1:]",
            f"            _y = torch.randint(0, max(1, _h - {hole_size}), (1,)).item()",
            f"            _x = torch.randint(0, max(1, _w - {hole_size}), (1,)).item()",
            f"            _field_image[:, _y:_y+{hole_size}, _x:_x+{hole_size}] = 0",
        ])
        result.field_tracking["image"] = "_field_image"

    elif node_type == "posterize":
        bits = int(params.get("bits", 4))
        result.getitem_code.extend([
            f"        # Posterize (bits={bits})",
            f"        if '_field_image' in dir():",
            f"            import torchvision.transforms.functional as F",
            f"            _field_image = F.posterize((_field_image * 255).to(torch.uint8), {bits}).float() / 255.0",
        ])
        result.field_tracking["image"] = "_field_image"

    elif node_type == "solarize":
        threshold = int(params.get("threshold", 128))
        result.getitem_code.extend([
            f"        # Solarize (threshold={threshold})",
            f"        if '_field_image' in dir():",
            f"            import torchvision.transforms.functional as F",
            f"            _field_image = F.solarize((_field_image * 255).to(torch.uint8), {threshold}).float() / 255.0",
        ])
        result.field_tracking["image"] = "_field_image"

    # === Multi-source Synthesis ===
    elif node_type == "zip_datasets":
        result.getitem_code.append("        # Zip Datasets - fields merged from upstream sources")

    elif node_type == "interleave_datasets":
        cycle_length = int(params.get("cycle_length", 2))
        result.getitem_code.append(f"        # Interleave Datasets (cycle_length={cycle_length}) - applied at dataset construction level")

    elif node_type == "sample_from_datasets":
        weights = str(params.get("weights", "0.5,0.5"))
        result.getitem_code.append(f"        # Sample From Datasets (weights={weights}) - applied at dataset construction level")

    # === Feature Engineering ===
    elif node_type == "standard_scaler":
        with_mean = _python_bool(params.get("with_mean", True), True)
        with_std = _python_bool(params.get("with_std", True), True)
        result.init_code.extend([
            f"        self._scaler_mean_{node_id} = 0.0",
            f"        self._scaler_std_{node_id} = 1.0",
        ])
        result.getitem_code.extend([
            f"        # Standard Scaler (with_mean={with_mean}, with_std={with_std})",
            f"        if '_field_features' in dir():",
        ])
        if with_mean == "True":
            result.getitem_code.append(f"            _field_features = _field_features - self._scaler_mean_{node_id}")
        if with_std == "True":
            result.getitem_code.append(f"            _field_features = _field_features / (self._scaler_std_{node_id} + 1e-8)")
        result.field_tracking["features"] = "_field_features"

    elif node_type == "minmax_scaler":
        feature_range = str(params.get("feature_range", "0,1"))
        parts = [p.strip() for p in feature_range.split(",")]
        range_min = float(parts[0]) if parts else 0.0
        range_max = float(parts[1]) if len(parts) > 1 else 1.0
        result.init_code.extend([
            f"        self._minmax_min_{node_id} = 0.0",
            f"        self._minmax_max_{node_id} = 1.0",
            f"        self._minmax_range_min_{node_id} = {range_min}",
            f"        self._minmax_range_max_{node_id} = {range_max}",
        ])
        result.getitem_code.extend([
            f"        # MinMax Scaler (feature_range=[{range_min}, {range_max}])",
            f"        if '_field_features' in dir():",
            f"            _denom = self._minmax_max_{node_id} - self._minmax_min_{node_id} + 1e-8",
            f"            _field_features = (_field_features - self._minmax_min_{node_id}) / _denom",
            f"            _field_features = _field_features * (self._minmax_range_max_{node_id} - self._minmax_range_min_{node_id}) + self._minmax_range_min_{node_id}",
        ])
        result.field_tracking["features"] = "_field_features"

    elif node_type == "pca":
        n_components = int(params.get("n_components", 2))
        result.init_code.extend([
            f"        self._pca_n_components_{node_id} = {n_components}",
            f"        self._pca_mean_{node_id} = None",
            f"        self._pca_components_{node_id} = None",
        ])
        result.getitem_code.extend([
            f"        # PCA (n_components={n_components})",
            f"        if '_field_features' in dir():",
            f"            if self._pca_components_{node_id} is not None:",
            f"                _field_features = _field_features - self._pca_mean_{node_id}",
            f"                _field_features = torch.matmul(_field_features, self._pca_components_{node_id})",
        ])
        result.field_tracking["features"] = "_field_features"

    elif node_type == "normalize_features":
        mean = str(params.get("mean", "0,0,0,0"))
        std = str(params.get("std", "1,1,1,1"))
        mean_parts = [p.strip() for p in mean.split(",")]
        std_parts = [p.strip() for p in std.split(",")]
        mean_vals = ", ".join(mean_parts) if mean_parts else "0"
        std_vals = ", ".join(std_parts) if std_parts else "1"
        result.getitem_code.extend([
            f"        # Normalize Features (mean=[{mean_vals}], std=[{std_vals}])",
            f"        if '_field_features' in dir():",
            f"            _mean_t = torch.tensor([{mean_vals}], dtype=torch.float32)",
            f"            _std_t = torch.tensor([{std_vals}], dtype=torch.float32)",
            f"            _std_t = torch.where(_std_t == 0, torch.ones_like(_std_t), _std_t)",
            f"            _field_features = (_field_features - _mean_t) / _std_t",
        ])
        result.field_tracking["features"] = "_field_features"

    elif node_type == "fill_missing_values":
        strategy = _python_value(params.get("strategy", "mean"))
        fill_value = float(params.get("fill_value", 0))
        result.getitem_code.extend([
            f"        # Fill Missing Values (strategy={strategy}, fill_value={fill_value})",
            f"        if '_field_features' in dir():",
            f"            _mask = torch.isnan(_field_features) | torch.isinf(_field_features)",
            f"            if _mask.any():",
            f"                if {strategy} == 'mean':",
            f"                    _fill = _field_features[~_mask].mean()",
            f"                elif {strategy} == 'median':",
            f"                    _fill = _field_features[~_mask].median()",
            f"                elif {strategy} == 'constant':",
            f"                    _fill = {fill_value}",
            f"                else:",
            f"                    _fill = 0.0",
            f"                _field_features = torch.where(_mask, torch.tensor(_fill, dtype=_field_features.dtype), _field_features)",
        ])
        result.field_tracking["features"] = "_field_features"

    elif node_type == "one_hot_encode":
        columns = str(params.get("columns", "category"))
        num_classes = str(params.get("num_classes", "3"))
        result.getitem_code.extend([
            f"        # One-Hot Encode (columns={columns}, num_classes={num_classes})",
            f"        # One-hot encoding is typically applied during dataset initialization",
        ])

    elif node_type == "dataset_output":
        fields = str(params.get("fields", "")).split(",")
        fields = [f.strip() for f in fields if f.strip()]
        result.getitem_code.append(f"        # Dataset Output: {', '.join(fields)}")
        for fld in fields:
            result.field_tracking[fld] = f"_field_{fld}"

    else:
        result.getitem_code.append(f"        # Unknown node type: {node_type}")

    return result


def _collect_data_output_fields(nodes: list[dict]) -> list[str]:
    seen: set[str] = set()
    fields: list[str] = []
    for node in nodes:
        data = node.get("data", {})
        if data.get("nodeType") != "dataset_output":
            continue
        for raw_field in str(data.get("params", {}).get("fields", "")).split(","):
            field = raw_field.strip()
            if not field or field in seen:
                continue
            seen.add(field)
            fields.append(field)
    return fields


def _collect_model_input_names(nodes: list[dict]) -> list[str]:
    names: list[str] = []
    for index, node in enumerate(nodes):
        data = node.get("data", {})
        if data.get("nodeType") != "input":
            continue
        params = data.get("params", {}) or {}
        name = str(params.get("name", "")).strip()
        names.append(name or f"input_{index + 1}")
    return names


def _source_length_expression(node: dict) -> str | None:
    data = node.get("data", {}) or {}
    node_type = data.get("nodeType")
    node_id = _safe_token(str(node.get("id", "")))
    if node_type == "folder_source":
        return f"len(self._{node_id}_paths)"
    if node_type in {"csv_source", "parquet_source"}:
        return f"len(self._{node_id}_df)"
    if node_type == "huggingface_source":
        return f"len(self._{node_id}_dataset)"
    return None


def _dataloader_config_block(nodes: list[dict], training_config: dict) -> str:
    loader_node = next((node for node in nodes if (node.get("data", {}) or {}).get("nodeType") == "dataloader"), None)
    params = ((loader_node or {}).get("data", {}) or {}).get("params", {}) or {}
    runtime = training_config.get("runtime", {}) or {}
    batch_size = int(params.get("batch_size", runtime.get("batchSize", 1)))
    shuffle = params.get("shuffle", True)
    num_workers = int(params.get("num_workers", runtime.get("numWorkers", 0)))
    pin_memory = params.get("pin_memory", False)
    drop_last = params.get("drop_last", False)
    return "\n".join([
        "DATALOADER_CONFIG = {",
        f"    'batch_size': {batch_size},",
        f"    'shuffle': {_python_bool(shuffle, True)},",
        f"    'num_workers': {num_workers},",
        f"    'pin_memory': {_python_bool(pin_memory, False)},",
        f"    'drop_last': {_python_bool(drop_last, False)},",
        "}",
    ])


def _format_node_summary(node: dict, index: int) -> str:
    data = node.get("data", {}) or {}
    params = data.get("params", {}) or {}
    params_text = ", ".join(f"{key}={value}" for key, value in params.items())
    label = data.get("label", data.get("nodeType", "node"))
    node_type = data.get("nodeType", "unknown")
    if params_text:
        return f"{index + 1}. {label} [{node_type}] — {params_text}"
    return f"{index + 1}. {label} [{node_type}]"


def _topological_sort(nodes: list[dict], edges: list[dict]) -> tuple[list[dict], list[str]]:
    node_map = {node["id"]: node for node in nodes if "id" in node}
    in_degree = {node_id: 0 for node_id in node_map}
    adjacency = {node_id: [] for node_id in node_map}

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source not in node_map or target not in node_map:
            continue
        adjacency[source].append(target)
        in_degree[target] += 1

    queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
    ordered_ids: list[str] = []
    while queue:
        current = queue.popleft()
        ordered_ids.append(current)
        for nxt in adjacency[current]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    if len(ordered_ids) == len(node_map):
        return [node_map[node_id] for node_id in ordered_ids], []

    ordered_set = set(ordered_ids)
    fallback = [node_map[node_id] for node_id in ordered_ids] + [node for node_id, node in node_map.items() if node_id not in ordered_set]
    return fallback, ["数据图存在环或断裂，已退回到'尽量稳定'的节点顺序生成脚手架。"]


def compile_dataflow(
    model_graph: dict | None = None,
    data_graph: dict | None = None,
    bindings: list[dict] | None = None,
    training_config: dict | None = None,
) -> CompiledDataflow:
    model_nodes = (model_graph or {}).get("nodes", []) if isinstance(model_graph, dict) else []
    data_nodes = (data_graph or {}).get("nodes", []) if isinstance(data_graph, dict) else []
    data_edges = (data_graph or {}).get("edges", []) if isinstance(data_graph, dict) else []
    bindings = bindings or []
    training_config = training_config or {}

    sorted_nodes, graph_warnings = _topological_sort(data_nodes, data_edges)
    output_fields = _collect_data_output_fields(data_nodes)
    output_field_set = set(output_fields)
    model_input_names = _collect_model_input_names(model_nodes)
    model_input_bindings = [
        {"target_key": str(binding.get("targetKey", "")), "source_key": str(binding.get("sourceKey", ""))}
        for binding in bindings
        if binding.get("target") == "model_input"
    ]
    target_binding_source = next(
        (str(binding.get("sourceKey", "")) for binding in bindings if binding.get("target") == "training_target" and binding.get("sourceKey")),
        None,
    )
    primary_model_input_key = model_input_names[0] if model_input_names else (model_input_bindings[0]["target_key"] if model_input_bindings else None)

    warnings = list(graph_warnings)
    for input_name in model_input_names:
        if not any(binding["target_key"] == input_name for binding in model_input_bindings):
            warnings.append(f"模型输入 {input_name} 尚未绑定数据字段，运行时将回退为随机张量。")

    for binding in bindings:
        source_key = str(binding.get("sourceKey", ""))
        if binding.get("sourceGraph") == "data" and source_key and source_key not in output_field_set:
            warnings.append(f"绑定源字段 {source_key} 未在 Dataset Output 中声明。")

    summary_lines = [_format_node_summary(node, index) for index, node in enumerate(sorted_nodes)]
    if not summary_lines:
        summary_lines = ["尚未配置数据图节点，当前导出使用占位 batch。"]

    has_real_data_pipeline = len(sorted_nodes) > 0

    # 收集所有 init 代码和 getitem 代码
    all_init_code = ["        # Auto-generated data pipeline initialization"]
    all_getitem_code = ["        # Auto-generated data pipeline"]

    for i, node in enumerate(sorted_nodes):
        node_code = _generate_data_node_code(node, i)
        all_init_code.extend(node_code.init_code)
        all_getitem_code.extend(node_code.getitem_code)

    sample_fields = output_fields if output_fields else ["image", "label"]
    task_type = training_config.get("taskType")
    sample_lines = [f'        "{field}": {_infer_tensor_expression(field, task_type)},' for field in sample_fields]
    step_comment_lines = [f"        # {line}" for line in summary_lines]
    binding_map_lines = [f'    "{binding["target_key"]}": "{binding["source_key"]}",' for binding in model_input_bindings]

    # 构建真正的 Dataset 类
    init_code_block = "\n".join(all_init_code)
    getitem_code_block = "\n".join(all_getitem_code)
    # For __getitem__, use actual field tracking variables instead of re-generating random tensors
    # Each field maps to its local variable name (e.g. "image" -> "_field_image")
    # Note: fallback expressions don't use device= since __getitem__ runs in DataLoader worker (CPU tensors)
    def _getitem_fallback_expr(field: str, task_type: str | None = None) -> str:
        lowered = field.lower()
        if any(kw in lowered for kw in ("label", "target", "class")):
            if task_type == "regression":
                return "torch.tensor(0.0, dtype=torch.float32)"
            return "torch.tensor(0, dtype=torch.long)"
        if "mask" in lowered:
            return "torch.randint(0, 2, (1, 1, 224, 224), dtype=torch.long)"
        if any(kw in lowered for kw in ("token", "text", "ids")):
            return "torch.randint(0, 1000, (1, 32), dtype=torch.long)"
        if any(kw in lowered for kw in ("image", "input", "feature")):
            return "torch.randn(1, 3, 224, 224)"
        return "torch.randn(1, 8)"

    sample_return_block = "\n".join(
        f'            "{field}": locals().get("_field_{field}", {_getitem_fallback_expr(field, task_type)}),'
        for field in sample_fields
    )
    source_length_expressions = [expr for expr in (_source_length_expression(node) for node in sorted_nodes) if expr]
    dataset_size_line = (
        f"        self._size = max(1, {', '.join(source_length_expressions)})"
        if source_length_expressions
        else "        self._size = 1"
    )
    dataloader_config = _dataloader_config_block(sorted_nodes, training_config)

    if has_real_data_pipeline:
        real_dataset_class = f'''
class FlowHamsterDataset(torch.utils.data.Dataset):
    def __init__(self):
{init_code_block}
{dataset_size_line}

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        _field_names = []
{getitem_code_block}

        # 构建输出字典，使用 field_tracking 中填充的真实变量
        sample = {{
{sample_return_block}
        }}

        return sample
'''
    else:
        fallback_sample_fields = "\n".join(f'            "{field}": {_infer_tensor_expression(field, task_type)},' for field in sample_fields)
        real_dataset_class = f'''class FlowHamsterDataset(torch.utils.data.Dataset):
    def __init__(self):
        self._size = 1

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        sample = {{
{fallback_sample_fields}
        }}
        return sample
'''

    summary_block = "\n".join(f"    {line!r}," for line in summary_lines)
    binding_block = "\n".join(binding_map_lines)
    sample_lines_block = "\n".join(sample_lines)

    # Collect required imports based on node types present
    required_imports = ["import torch"]
    seen_node_types = {node.get("data", {}).get("nodeType", "") for node in sorted_nodes}
    image_nodes = {"folder_source", "read_image", "random_horizontal_flip", "random_vertical_flip",
                   "random_crop", "random_rotation", "color_jitter", "random_erasing",
                   "gaussian_blur", "grayscale", "autoaugment", "randaugment", "cutout",
                   "posterize", "solarize", "mixup", "cutmix"}
    if seen_node_types & image_nodes:
        required_imports.append("import torchvision")
    if "csv_source" in seen_node_types:
        required_imports.append("import pandas as pd")
    if "jsonl_source" in seen_node_types:
        required_imports.append("import json")
    if seen_node_types & {"folder_source"}:
        required_imports.insert(0, "from pathlib import Path")
    if seen_node_types & {"autoaugment", "randaugment"}:
        required_imports.append("import numpy as np")
    imports_block = "\n".join(required_imports)

    python_scaffold = f"""
{imports_block}

DATA_PIPELINE_SUMMARY = [
{summary_block}
]

BOUND_MODEL_INPUTS = {{
{binding_block}
}}
BOUND_TRAINING_TARGET = {target_binding_source!r}
PRIMARY_MODEL_INPUT_KEY = {primary_model_input_key!r}
{dataloader_config}

{real_dataset_class}

def build_demo_batch(device):
    try:
        loader = build_flowhamster_dataloader()
        batch = next(iter(loader))
        if isinstance(batch, dict):
            return batch
    except Exception as exc:
        print(f"Data pipeline warning: {{exc}}")

    return build_fallback_batch(device)

def build_fallback_batch(device):
    return {{
{sample_lines_block}
    }}

def build_flowhamster_dataset():
    return FlowHamsterDataset()

def build_flowhamster_dataloader():
    dataset = build_flowhamster_dataset()
    return torch.utils.data.DataLoader(dataset, **DATALOADER_CONFIG)

def _coerce_bound_value(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, (int, float)):
        return torch.tensor(value, device=device)
    return None

def resolve_bound_inputs(batch, device):
    model_feed = {{}}
    for target_key, source_key in BOUND_MODEL_INPUTS.items():
        value = _coerce_bound_value(batch.get(source_key), device)
        if value is not None:
            model_feed[target_key] = value

    target = None
    if BOUND_TRAINING_TARGET is not None:
        target = _coerce_bound_value(batch.get(BOUND_TRAINING_TARGET), device)

    return model_feed, target

def select_primary_model_input(model_feed, device, batch_size=None):
    if PRIMARY_MODEL_INPUT_KEY and PRIMARY_MODEL_INPUT_KEY in model_feed:
        return model_feed[PRIMARY_MODEL_INPUT_KEY]
    for value in model_feed.values():
        if torch.is_tensor(value):
            return value
    fallback_batch_size = int(batch_size) if batch_size else 1
    return torch.randn(fallback_batch_size, 3, 224, 224, device=device)"""

    return CompiledDataflow(
        output_fields=output_fields,
        model_input_bindings=model_input_bindings,
        target_binding_source=target_binding_source,
        primary_model_input_key=primary_model_input_key,
        warnings=warnings,
        summary_lines=summary_lines,
        python_scaffold=python_scaffold,
        has_workflow_runtime=bool(data_nodes or bindings),
    )
