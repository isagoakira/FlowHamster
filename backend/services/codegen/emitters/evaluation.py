"""Emit evaluation metric code for generated scripts."""
from __future__ import annotations

from ..graph import NodeBlock

def _gen_evaluation(block: NodeBlock, all_blocks: list) -> list:
    """Generate evaluation code for a single evaluation block."""
    lines: list[str] = []
    f = block.fields
    op = block.op_type
    average = f.get("average", "macro")
    top_k = f.get("top_k", 1)

    pred_ref = block.inputs.get("predictions")
    tgt_ref = block.inputs.get("targets")

    def resolve_var(ref) -> str:
        if not ref:
            return "y_pred"
        if isinstance(ref, list):
            ref = ref[0] if ref else None
        if not ref:
            return "y_pred"
        src_id = ref.split(":")[1].split("/")[0]
        for b in all_blocks:
            if b.node_id == src_id:
                return b.output_var
        return "y_pred"

    pred_var = resolve_var(pred_ref) if pred_ref else "y_pred"
    tgt_var = resolve_var(tgt_ref) if tgt_ref else "y_true"

    if op == "accuracy":
        lines.append("    # Accuracy evaluation")
        lines.append("    from sklearn.metrics import accuracy_score")
        lines.append(f"    y_pred_np = {pred_var}.argmax(dim=1).numpy() if {pred_var}.dim() > 1 else {pred_var}.numpy()")
        lines.append(f"    y_true_np = {tgt_var}.numpy() if hasattr({tgt_var}, 'numpy') else {tgt_var}")
        if top_k > 1:
            lines.append(f"    from sklearn.metrics import top_k_accuracy_score")
            lines.append(f"    acc = top_k_accuracy_score(y_true_np, {pred_var}.numpy(), k={top_k})")
        else:
            lines.append(f"    acc = accuracy_score(y_true_np, y_pred_np)")
        lines.append(f'    print(f"Accuracy: {{acc:.4f}}")')
    elif op == "f1":
        lines.append("    # F1 Score evaluation")
        lines.append("    from sklearn.metrics import f1_score")
        lines.append(f"    y_pred_np = {pred_var}.argmax(dim=1).numpy() if {pred_var}.dim() > 1 else {pred_var}.numpy()")
        lines.append(f"    y_true_np = {tgt_var}.numpy() if hasattr({tgt_var}, 'numpy') else {tgt_var}")
        lines.append(f"    f1 = f1_score(y_true_np, y_pred_np, average='{average}')")
        lines.append(f'    print(f"F1 Score ({average}): {{f1:.4f}}")')
    elif op == "precision":
        lines.append("    # Precision evaluation")
        lines.append("    from sklearn.metrics import precision_score")
        lines.append(f"    y_pred_np = {pred_var}.argmax(dim=1).numpy() if {pred_var}.dim() > 1 else {pred_var}.numpy()")
        lines.append(f"    y_true_np = {tgt_var}.numpy() if hasattr({tgt_var}, 'numpy') else {tgt_var}")
        lines.append(f"    prec = precision_score(y_true_np, y_pred_np, average='{average}')")
        lines.append(f'    print(f"Precision ({average}): {{prec:.4f}}")')
    elif op == "recall":
        lines.append("    # Recall evaluation")
        lines.append("    from sklearn.metrics import recall_score")
        lines.append(f"    y_pred_np = {pred_var}.argmax(dim=1).numpy() if {pred_var}.dim() > 1 else {pred_var}.numpy()")
        lines.append(f"    y_true_np = {tgt_var}.numpy() if hasattr({tgt_var}, 'numpy') else {tgt_var}")
        lines.append(f"    rec = recall_score(y_true_np, y_pred_np, average='{average}')")
        lines.append(f'    print(f"Recall ({average}): {{rec:.4f}}")')
    elif op == "confusion_matrix":
        lines.append("    # Confusion Matrix evaluation")
        lines.append("    from sklearn.metrics import confusion_matrix")
        lines.append(f"    y_pred_np = {pred_var}.argmax(dim=1).numpy() if {pred_var}.dim() > 1 else {pred_var}.numpy()")
        lines.append(f"    y_true_np = {tgt_var}.numpy() if hasattr({tgt_var}, 'numpy') else {tgt_var}")
        lines.append(f"    cm = confusion_matrix(y_true_np, y_pred_np)")
        lines.append(f'    print(f"Confusion Matrix:\\n{{cm}}")')
    elif op == "mean_iou":
        lines.append("    # Mean IoU evaluation (semantic segmentation)")
        lines.append("    try:")
        lines.append("        from torchmetrics import MeanIoU")
        lines.append(f"        miou = MeanIoU(num_classes={f.get('num_classes', 10)})")
        lines.append(f"        y_pred_labels = {pred_var}.argmax(dim=1)")
        lines.append(f"        y_true_labels = {tgt_var}")
        lines.append(f"        iou = miou(y_pred_labels, y_true_labels.long())")
        lines.append(f'        print(f"Mean IoU: {{iou:.4f}}")')
        lines.append("    except Exception as e:")
        lines.append(f'        print(f"Mean IoU computation failed: {{e}}")')
    elif op == "roc_auc":
        lines.append("    # ROC AUC evaluation")
        lines.append("    from sklearn.metrics import roc_auc_score")
        lines.append(f"    y_pred_prob = {pred_var}.softmax(dim=1).numpy() if {pred_var}.dim() > 1 else {pred_var}.numpy()")
        lines.append(f"    y_true_np = {tgt_var}.numpy() if hasattr({tgt_var}, 'numpy') else {tgt_var}")
        lines.append(f"    auc = roc_auc_score(y_true_np, y_pred_prob, average='{average}', multi_class='ovr')")
        lines.append(f'    print(f"ROC AUC ({average}): {{auc:.4f}}")')

    return lines


