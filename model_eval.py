from __future__ import annotations
import argparse, time
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from collections import defaultdict

from src.models.residual2plus1dcnn import Residual2Plus1DCNN

import numpy as np
import pandas as pd

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
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
from src.models.frame_baseline import FrameMeanPoolBaseline
from src.models.clip_baseline import R3D18Baseline


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
    """
    Sweep thresholds 0.1–0.9 in steps of 0.01.
    Return the threshold that maximises macro-averaged F1 on the provided
    labels/probs (intended for use on the validation set).
    """
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.10, 0.91, 0.01):
        preds = (probs >= t).astype(int)
        f = f1_score(labels, preds, average="macro", zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = float(t)
    return best_t


def nan_str(v) -> str:
    """Safely format a float that may be nan or non-numeric."""
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
    """
    Evaluate model on loader.

    """
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    n = 0

    all_labels  = []
    all_probs   = []
    all_sources = []

    for x, y, meta in loader:
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

    # ── Threshold sweep on this split ────────────────────────────────────────
    swept_threshold = find_best_threshold(all_labels, all_probs)
    all_preds_swept = (all_probs >= swept_threshold).astype(int)
    all_preds_fixed = (all_probs >= threshold).astype(int)

    avg_loss = total_loss / max(n, 1)

    metrics = {
        "loss":           avg_loss,
        "threshold_used": threshold,
        "best_threshold": swept_threshold,

        # ── metrics at supplied threshold ─────────────────────────────────────
        "accuracy":  accuracy_score(all_labels, all_preds_fixed),
        "precision": precision_score(all_labels, all_preds_fixed, zero_division=0),
        "recall":    recall_score(all_labels, all_preds_fixed, zero_division=0),
        "f1":        f1_score(all_labels, all_preds_fixed, zero_division=0),
        "macro_f1":  f1_score(all_labels, all_preds_fixed, average="macro", zero_division=0),

        # ── metrics at swept threshold ────────────────────────────────────────
        "accuracy_swept":  accuracy_score(all_labels, all_preds_swept),
        "f1_swept":        f1_score(all_labels, all_preds_swept, zero_division=0),
        "macro_f1_swept":  f1_score(all_labels, all_preds_swept, average="macro", zero_division=0),

        # ── threshold-independent (computed across all sources combined) ───────
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",         choices=["frames", "clip"], required=True)
    ap.add_argument("--dataset_root", default="/scratch/sc22as2/IVY-Fake")
    ap.add_argument("--train_csv",    default="/scratch/sc22as2/IVY-Fake/splits/standard_train_clean.csv")
    ap.add_argument("--val_csv",      default="/scratch/sc22as2/IVY-Fake/splits/standard_val_clean.csv")
    ap.add_argument("--test_csv",     default="/scratch/sc22as2/IVY-Fake/splits/standard_test_clean.csv")
    ap.add_argument("--image_size",   type=int,   default=224)
    ap.add_argument("--n_frames",     type=int,   default=16)
    ap.add_argument("--clip_len",     type=int,   default=16)
    ap.add_argument("--backbone",     default="resnet18")
    ap.add_argument("--batch_size",   type=int,   default=8)
    ap.add_argument("--num_workers",  type=int,   default=4)
    ap.add_argument("--epochs",       type=int,   default=10)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--ckpt", default="runs/clip_r3d18_w4_flash.pt")

    # pos_weight for BCEWithLogitsLoss.
    # Label convention: 0 = real (negative), 1 = fake (positive).
    # pos_weight < 1 decreases the penalty for missing a fake video,
    # which reduces the model's tendency to over-predict fake —
    # n_fake / n_real = 45375 / 55000 ≈ 0.825
    ap.add_argument("--n_real", type=int, default=55000,
                    help="Number of real (negative) samples in training split")
    ap.add_argument("--n_fake", type=int, default=45375,
                    help="Number of fake (positive) samples in training split")

    ap.add_argument("--out",             default="runs/baseline.pt")
    ap.add_argument("--out_best_f1",     default="runs/baseline_best_f1.pt",
                    help="Checkpoint saved when val macro-F1 improves")
    ap.add_argument("--val_metrics_csv", default="runs/val_per_source_metrics.csv")
    ap.add_argument("--test_metrics_csv",default="runs/test_per_source_metrics.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── pos_weight ────────────────────────────────────────────────────────────
    # n_fake / n_real < 1 → slightly down-weights the fake (positive) class,
    # reducing the model's tendency to predict everything as fake.
    pos_weight_val = args.n_fake / args.n_real          # ≈ 0.818 with defaults
    pos_weight     = torch.tensor([pos_weight_val], device=device)
    loss_fn        = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"pos_weight = {pos_weight_val:.4f}  "
          f"(n_fake={args.n_fake}, n_real={args.n_real})  "
          f"[down-weights fake class to reduce over-prediction bias]")

    # ── Data ──────────────────────────────────────────────────────────────────
    cfg = DataConfig(
        dataset_root=args.dataset_root,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        image_size=args.image_size,
    )
    train_loader, val_loader, test_loader = make_loaders(
        cfg,
        mode=args.mode,
        n_frames=args.n_frames,
        clip_len=args.clip_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.mode == "frames":
        model = FrameMeanPoolBaseline(backbone=args.backbone, pretrained=True)
    else:
        model = Residual2Plus1DCNN()
    model.to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ── Evaluation-only mode ─────────────────────────────────────
    if args.eval_only:
        print(f"\nLoading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)

        #model.load_state_dict(ckpt["model"])
        #ckpt_threshold = ckpt.get("threshold", 0.5)

        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            ckpt_threshold = ckpt.get("threshold", 0.5)
        else:
            model.load_state_dict(ckpt)
            ckpt_threshold = 0.5

        print(f"Using threshold: {ckpt_threshold:.2f}")

        _ = evaluate(
            model, val_loader, device,
            split_name="val_eval_only",
            threshold=ckpt_threshold,
            per_source_csv=args.val_metrics_csv,
            print_per_source=True,
        )

        test_metrics = evaluate(
            model, test_loader, device,
            split_name="test_eval_only",
            threshold=ckpt_threshold,
            per_source_csv=args.test_metrics_csv,
            print_per_source=True,
        )

        _print_test_summary(test_metrics, label="eval-only")

        return   #stop before training

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss     = float("inf")
    best_val_macro_f1 = -1.0
    best_threshold    = 0.5     # updated each epoch from val sweep

    print(
        f"\n{'Epoch':>5} {'TrainLoss':>10} {'ValLoss':>10} {'Acc':>8} "
        f"{'MacroF1':>9} {'MacroF1@t*':>11} {'AUC':>8} {'BestT':>7} {'Time':>8}"
    )
    print("─" * 95)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, n = 0.0, 0

        for x, y, _ in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().view(-1)

            opt.zero_grad(set_to_none=True)
            logits = model(x).view(-1)
            loss   = loss_fn(logits, y)
            loss.backward()
            opt.step()

            running += loss.item() * y.size(0)
            n       += y.size(0)

        train_loss = running / max(n, 1)

        val_metrics = evaluate(
            model, val_loader, device,
            split_name=f"val_epoch_{epoch}",
            threshold=best_threshold,
            per_source_csv=None,
            print_per_source=True,
        )

        best_threshold = val_metrics["best_threshold"]

        elapsed = time.time() - t0
        print(
            f"{epoch:>5} "
            f"{train_loss:>10.4f} "
            f"{val_metrics['loss']:>10.4f} "
            f"{val_metrics['accuracy']:>8.4f} "
            f"{val_metrics['macro_f1']:>9.4f} "
            f"{val_metrics['macro_f1_swept']:>11.4f} "
            f"{nan_str(val_metrics['roc_auc']):>8} "
            f"{best_threshold:>7.2f} "
            f"{elapsed:>7.1f}s"
        )

        # ── Checkpoint: best val loss ─────────────────────────────────────────
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {"model": model.state_dict(), "args": vars(args),
                 "threshold": best_threshold},
                args.out,
            )
            print(f"  ✓ saved best-loss checkpoint  -> {args.out}  "
                  f"(loss={best_val_loss:.4f})")

        # ── Checkpoint: best val macro-F1 (swept threshold) ───────────────────
        if val_metrics["macro_f1_swept"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1_swept"]
            torch.save(
                {"model": model.state_dict(), "args": vars(args),
                 "threshold": best_threshold},
                args.out_best_f1,
            )
            print(f"  ✓ saved best-macroF1 checkpoint -> {args.out_best_f1}  "
                  f"(macro_f1={best_val_macro_f1:.4f}, t*={best_threshold:.2f})")

    # ── Final evaluation: best-loss checkpoint ────────────────────────────────
    print(f"\n{'='*60}")
    print("Loading best-loss checkpoint for final evaluation ...")
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model"])
    ckpt_threshold = ckpt.get("threshold", 0.5)
    print(f"  checkpoint threshold = {ckpt_threshold:.2f}")

    _ = evaluate(
        model, val_loader, device,
        split_name="val_best_loss",
        threshold=ckpt_threshold,
        per_source_csv=args.val_metrics_csv,
        print_per_source=True,
    )
    test_metrics = evaluate(
        model, test_loader, device,
        split_name="test_best_loss",
        threshold=ckpt_threshold,
        per_source_csv=args.test_metrics_csv,
        print_per_source=True,
    )
    _print_test_summary(test_metrics, label="best-loss checkpoint")

    # ── Final evaluation: best-macro-F1 checkpoint ────────────────────────────
    print(f"\n{'='*60}")
    print("Loading best-macroF1 checkpoint for final evaluation ...")
    ckpt_f1 = torch.load(args.out_best_f1, map_location=device)
    model.load_state_dict(ckpt_f1["model"])
    ckpt_f1_threshold = ckpt_f1.get("threshold", 0.5)
    print(f"  checkpoint threshold = {ckpt_f1_threshold:.2f}")

    test_metrics_f1 = evaluate(
        model, test_loader, device,
        split_name="test_best_f1",
        threshold=ckpt_f1_threshold,
        per_source_csv=args.test_metrics_csv.replace(".csv", "_best_f1.csv"),
        print_per_source=True,
    )
    _print_test_summary(test_metrics_f1, label="best-macroF1 checkpoint")


def _print_test_summary(metrics: dict, label: str = "test") -> None:
    print(f"\n── Test Results ({label}) ─────────────────────────────────")
    print(f"  threshold   : {metrics['threshold_used']:.2f}  (swept best on val)")
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


if __name__ == "__main__":
    main()