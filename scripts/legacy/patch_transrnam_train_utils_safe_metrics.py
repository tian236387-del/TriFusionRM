#!/usr/bin/env python
"""Patch TransRNAm train_utils.py to avoid missing safe metric helpers.

The official TransRNAm metric code assumes every class slice contains both
positive and negative examples. Strict splits or small classes can violate that
assumption, and some adapted copies also call _safe_roc_auc_score without
defining it. This patch inserts safe helpers and rewrites the fragile calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path


HELPERS = '''

def _safe_roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float("nan")


def _safe_average_precision_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0 or np.sum(y_true) == 0:
        return float("nan")
    try:
        return average_precision_score(y_true, y_score)
    except Exception:
        return float("nan")


def _safe_curve(curve_fn, y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]), np.asarray([0.5])
    try:
        return curve_fn(y_true, y_score)
    except Exception:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]), np.asarray([0.5])


def _safe_confusion_counts(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return float(tn), float(fp), float(fn), float(tp)


def _safe_div(num, den):
    return float(num) / float(den) if float(den) != 0.0 else 0.0
'''


def patch(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text

    if "def _safe_roc_auc_score" not in text:
        marker = "def cal_metrics(model_out,label,plot=True,class_names=None,plot_name=None):"
        text = text.replace(marker, HELPERS + "\n" + marker)

    replacements = {
        "auc = roc_auc_score(y_true[i*100:(i+1)*100],y_score[i*100:(i+1)*100])":
            "auc = _safe_roc_auc_score(y_true[i*100:(i+1)*100], y_score[i*100:(i+1)*100])",
        "ap = average_precision_score(y_true[i*100:(i+1)*100],y_score[i*100:(i+1)*100])":
            "ap = _safe_average_precision_score(y_true[i*100:(i+1)*100], y_score[i*100:(i+1)*100])",
        "auc_2 = roc_auc_score(y_true,y_score)":
            "auc_2 = _safe_roc_auc_score(y_true, y_score)",
        "ap_2 = average_precision_score(y_true,y_score)":
            "ap_2 = _safe_average_precision_score(y_true, y_score)",
        "fpr[i], tpr[i], thresholds = roc_curve(y_true[i*100:(i+1)*100], y_score[i*100:(i+1)*100])":
            "fpr[i], tpr[i], thresholds = _safe_curve(roc_curve, y_true[i*100:(i+1)*100], y_score[i*100:(i+1)*100])",
        "fpr_2[i], tpr_2[i], thresholds_2 = roc_curve(y_true, y_score)":
            "fpr_2[i], tpr_2[i], thresholds_2 = _safe_curve(roc_curve, y_true, y_score)",
        "precisions[i], recalls[i], _ = precision_recall_curve(y_true[i*100:(i+1)*100], y_score[i*100:(i+1)*100])":
            "precisions[i], recalls[i], _ = _safe_curve(precision_recall_curve, y_true[i*100:(i+1)*100], y_score[i*100:(i+1)*100])",
        "precisions_m[i], recalls_m[i], _ = precision_recall_curve(y_true, y_score)":
            "precisions_m[i], recalls_m[i], _ = _safe_curve(precision_recall_curve, y_true, y_score)",
        "tn, fp, fn, tp = confusion_matrix(y_true[i*100:(i+1)*100], y_pred_new[i*100:(i+1)*100]).ravel()":
            "tn, fp, fn, tp = _safe_confusion_counts(y_true[i*100:(i+1)*100], y_pred_new[i*100:(i+1)*100])",
        "sensitivity = tp / pp":
            "sensitivity = _safe_div(tp, pp)",
        "specificity = tn / pn":
            "specificity = _safe_div(tn, pn)",
        "precision = tp / (tp + fp)":
            "precision = _safe_div(tp, (tp + fp))",
        "acc_2 = (tp+tn) / (pp+pn)":
            "acc_2 = _safe_div((tp + tn), (pp + pn))",
        "mcc = ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))":
            "mcc = matthews_corrcoef(y_true[i*100:(i+1)*100], y_pred_new[i*100:(i+1)*100]) if len(np.unique(y_true[i*100:(i+1)*100])) > 1 else 0.0",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    if text != original:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="rmbase12_work/official_code_reproduction/multirm12_101nt/TransRNAm-main/Scripts/train_utils.py",
    )
    args = parser.parse_args()
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"missing file: {path}")
    changed = patch(path)
    print(f"{'patched' if changed else 'already patched'}: {path}")


if __name__ == "__main__":
    main()
