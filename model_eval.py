from __future__ import annotations
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd

import sys
sys.path.append(".")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from src.data.loader import DataConfig, make_loaders
from src.models.frame_lstm import FrameLSTM


# ── Metric helpers ────────────────────────────────────────────────────────────

def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_prob)


def safe_pr_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return average_precision_score(y_true, y_prob)


def extract_sources(meta, batch_size):
    if isinstance(meta, dict):
        sources = meta.get("source", ["unknown"] * batch_size)
    else:
        sources = ["unknown"] * batch_size
    if isinstance(sources, (list, tuple)):
        return list(sources)
    return ["unknown"] * batch_size


def find_best_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.10, 0.91, 0.01):
        preds = (probs >= t).astype(int)
        f = f1_score(labels, preds, average="macro", zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = float(t)
    return best_t


def nan_str(v) -> str:
    try:
        if np.isnan(float(v)):
            return "nan"
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "nan"


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    split_name: str = "eval",
    threshold: float = 0.5,
    per_source_csv=None,
    print_per_source: bool = True,
):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    n = 0

    all_labels  = []
    all_probs   = []
    all_sources = []

    for x, y, meta in tqdm(loader, desc=f"Evaluating ({split_name})"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        logits = model(x).view(-1)
        y      = y.view(-1)

        loss = loss_fn(logits, y)
        total_loss += loss.item() * y.size(0)
        n          += y.size(0)

        probs = torch.sigmoid(logits)
        all_labels.extend(y.detach().cpu().numpy().astype(int).tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_sources.extend(extract_sources(meta, y.size(0)))

    all_labels = np.asarray(all_labels)
    all_probs  = np.asarray(all_probs)

    swept_threshold = find_best_threshold(all_labels, all_probs)
    all_preds_swept = (all_probs >= swept_threshold).astype(int)
    all_preds_fixed = (all_probs >= threshold).astype(int)

    avg_loss = total_loss / max(n, 1)

    metrics = {
        "loss":           avg_loss,
        "threshold_used": threshold,
        "best_threshold": swept_threshold,

        "accuracy":  accuracy_score(all_labels, all_preds_fixed),
        "precision": precision_score(all_labels, all_preds_fixed, zero_division=0),
        "recall":    recall_score(all_labels, all_preds_fixed, zero_division=0),
        "f1":        f1_score(all_labels, all_preds_fixed, zero_division=0),
        "macro_f1":  f1_score(all_labels, all_preds_fixed, average="macro", zero_division=0),

        "accuracy_swept":  accuracy_score(all_labels, all_preds_swept),
        "f1_swept":        f1_score(all_labels, all_preds_swept, zero_division=0),
        "macro_f1_swept":  f1_score(all_labels, all_preds_swept, average="macro", zero_division=0),

        "roc_auc":          safe_auc(all_labels, all_probs),
        "pr_auc":           safe_pr_auc(all_labels, all_probs),
        "confusion_matrix": confusion_matrix(all_labels, all_preds_fixed).tolist(),
    }

    # ── Per-source breakdown ──────────────────────────────────────────────────
    per_source_rows = []
    by_source_idx   = defaultdict(list)
    for i, s in enumerate(all_sources):
        by_source_idx[s].append(i)

    for source in sorted(by_source_idx.keys()):
        idx      = np.asarray(by_source_idx[source])
        ys       = all_labels[idx]
        ps       = all_probs[idx]
        yh       = all_preds_fixed[idx]
        yh_swept = all_preds_swept[idx]

        row = {
            "split":              split_name,
            "source":             source,
            "true_label":         "fake" if float(np.mean(ys)) >= 0.5 else "real",
            "total":              int(len(idx)),
            "correct":            int((yh == ys).sum()),
            "accuracy":           accuracy_score(ys, yh),
            "precision":          precision_score(ys, yh, zero_division=0),
            "recall":             recall_score(ys, yh, zero_division=0),
            "f1":                 f1_score(ys, yh, zero_division=0),
            "macro_f1":           f1_score(ys, yh, average="macro", zero_division=0),
            "f1_swept":           f1_score(ys, yh_swept, zero_division=0),
            "macro_f1_swept":     f1_score(ys, yh_swept, average="macro", zero_division=0),
            "mean_prob":          float(np.mean(ps)),
            "pred_positive_rate": float(np.mean(yh)),
            "true_positive_rate": float(np.mean(ys)),
        }
        per_source_rows.append(row)

    per_source_df = pd.DataFrame(per_source_rows)
    metrics["per_source_df"] = per_source_df

    if print_per_source and not per_source_df.empty:
        print(f"\n  Per-source metrics ({split_name})  "
              f"[threshold={threshold:.2f} | swept={swept_threshold:.2f}]")
        print(f"  {'Source':<24} {'Label':<6} {'Acc':>7} {'F1':>7} "
              f"{'MacroF1':>9} {'F1@t*':>8} {'MeanProb':>9} {'Correct/Total':>15}")
        print(f"  {'─' * 105}")
        for _, row in per_source_df.sort_values("source").iterrows():
            print(
                f"  {row['source']:<24} "
                f"{row['true_label']:<6} "
                f"{row['accuracy']:>7.4f} "
                f"{row['f1']:>7.4f} "
                f"{row['macro_f1']:>9.4f} "
                f"{row['f1_swept']:>8.4f} "
                f"{row['mean_prob']:>9.4f} "
                f"  ({int(row['correct'])}/{int(row['total'])})"
            )
        print(
            f"\n  Overall @ t={threshold:.2f} → "
            f"Acc={metrics['accuracy']:.4f}  MacroF1={metrics['macro_f1']:.4f}  "
            f"AUC={nan_str(metrics['roc_auc'])}  PR-AUC={nan_str(metrics['pr_auc'])}\n"
            f"  Overall @ t*={swept_threshold:.2f} → "
            f"Acc={metrics['accuracy_swept']:.4f}  MacroF1={metrics['macro_f1_swept']:.4f}\n"
        )

    if per_source_csv is not None:
        per_source_df.to_csv(per_source_csv, index=False)
        print(f"  Saved per-source metrics -> {per_source_csv}")

    return metrics


def _print_test_summary(metrics: dict, label: str = "test") -> None:
    print(f"\n── Test Results ({label}) ─────────────────────────────────")
    print(f"  threshold   : {metrics['threshold_used']:.2f}")
    print(f"  loss        : {metrics['loss']:.4f}")
    print(f"  accuracy    : {metrics['accuracy']:.4f}")
    print(f"  precision   : {metrics['precision']:.4f}")
    print(f"  recall      : {metrics['recall']:.4f}")
    print(f"  f1          : {metrics['f1']:.4f}")
    print(f"  macro_f1    : {metrics['macro_f1']:.4f}")
    print(f"  roc_auc     : {nan_str(metrics['roc_auc'])}")
    print(f"  pr_auc      : {nan_str(metrics['pr_auc'])}")
    print(f"  conf_matrix : {metrics['confusion_matrix']}")
    print("──────────────────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",       required=True, help="Path to .pt checkpoint")
    ap.add_argument("--dataset_root",     default="/scratch/sc22as2/IVY-Fake")
    ap.add_argument("--train_csv",        default="/scratch/sc22as2/IVY-Fake/splits/standard_train_clean.csv")
    ap.add_argument("--val_csv",          default="/scratch/sc22as2/IVY-Fake/splits/standard_val_clean.csv")
    ap.add_argument("--test_csv",         default="/scratch/sc22as2/IVY-Fake/splits/standard_test_clean.csv")
    ap.add_argument("--image_size",       type=int, default=224)
    ap.add_argument("--n_frames",         type=int, default=16)
    ap.add_argument("--batch_size",       type=int, default=4)
    ap.add_argument("--num_workers",      type=int, default=8)
    ap.add_argument("--val_metrics_csv",  default="runs/val_per_source_metrics.csv")
    ap.add_argument("--test_metrics_csv", default="runs/test_per_source_metrics.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = DataConfig(
        dataset_root=args.dataset_root,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        image_size=args.image_size,
    )

    _, val_loader, test_loader = make_loaders(
        cfg,
        mode="frames",          # FrameLSTM is always frame-based
        n_frames=args.n_frames,
        clip_len=16,            # unused in frames mode but required by make_loaders
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # ── Model: FrameLSTM ─────────────────────────────────────────────────────
    model = FrameLSTM(
        backbone="resnet18",
        pretrained=False,       # weights come from the checkpoint
        lstm_hidden_size=512,
        lstm_num_layers=1,
        lstm_bidirectional=False,
        lstm_dropout=0.0,
    )

    # ── Load checkpoint ───────────────────────────────────────────────────────
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        ckpt_threshold = checkpoint.get("threshold", 0.5)
    else:
        model.load_state_dict(checkpoint)
        ckpt_threshold = 0.5

    model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Using threshold  : {ckpt_threshold:.2f}")

    # ── Val evaluation ────────────────────────────────────────────────────────
    val_metrics = evaluate(
        model, val_loader, device,
        split_name="val",
        threshold=ckpt_threshold,
        per_source_csv=args.val_metrics_csv,
        print_per_source=True,
    )

    # ── Test evaluation ───────────────────────────────────────────────────────
    # Use the best threshold found on the val set for final test evaluation
    best_val_threshold = val_metrics["best_threshold"]
    print(f"\nUsing val-swept threshold {best_val_threshold:.2f} for test evaluation")

    test_metrics = evaluate(
        model, test_loader, device,
        split_name="test",
        threshold=best_val_threshold,
        per_source_csv=args.test_metrics_csv,
        print_per_source=True,
    )

    _print_test_summary(test_metrics, label="FrameLSTM")


if __name__ == "__main__":
    main()