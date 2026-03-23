from __future__ import annotations
import argparse, time
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

import torch.multiprocessing as mp #
mp.set_start_method("spawn", force=True) #
import sys
sys.path.append(".")

from src.data.loader import DataConfig, make_loaders
from src.models.frame_baseline import FrameMeanPoolBaseline
from src.models.frame_lstm import FrameLSTM
from src.models.clip_baseline import R3D18Baseline

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    n = 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * y.size(0)

        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == y.long()).sum().item()
        n += y.size(0)

    return total_loss / max(n, 1), correct / max(n, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["frames","clip"], required=True)
    ap.add_argument("--dataset_root", default="/scratch/sc22as2/IVY-Fake")
    ap.add_argument("--train_csv", default="/scratch/sc22as2/IVY-Fake/splits/standard_train_clean.csv")
    ap.add_argument("--val_csv",   default="/scratch/sc22as2/IVY-Fake/splits/standard_val_clean.csv")
    ap.add_argument("--test_csv",  default="/scratch/sc22as2/IVY-Fake/splits/standard_test_clean.csv")
    ap.add_argument("--image_size", type=int, default=224)

    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--clip_len", type=int, default=16)

    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--out", default="runs/baseline.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    if args.mode == "frames":
        if args.backbone == "frame_lstm":
            model = FrameLSTM(
                backbone="resnet18",  # internal ResNet feature extractor (resnet18/resnet50)
                pretrained=True,
                lstm_hidden_size=512,
                lstm_num_layers=1,
                lstm_bidirectional=False,
                lstm_dropout=0.0,
            )
        else:
            model = FrameMeanPoolBaseline(backbone=args.backbone, pretrained=True)
    else:
        model = R3D18Baseline(pretrained=True)

    model.to(device)

    # Partial Freezing: Freeze early layers (stem, layer1, layer2) of the backbone
    if hasattr(model, "backbone"):
        for name, param in model.backbone.named_parameters():
            if any(layer_name in name for layer_name in ["conv1", "bn1", "layer1", "layer2"]):
                param.requires_grad = False

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    opt = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    best_val = 1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n = 0

        for x, y, _ in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)
            
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()

            running += loss.item() * y.size(0)
            n += y.size(0)

        train_loss = running / max(n, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={time.time()-t0:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.out)
            print(f"  saved best -> {args.out}")

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"[test] loss={test_loss:.4f} acc={test_acc:.4f}")

if __name__ == "__main__":
    main()
