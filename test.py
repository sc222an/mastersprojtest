from __future__ import annotations
import argparse, time
import torch
import torch.nn as nn
from tqdm import tqdm

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

    for x, y, _ in tqdm(loader, desc="Evaluating"):
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
    ap.add_argument("--checkpoint", required=True, help="Path to the trained model checkpoint")
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
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = DataConfig(
        dataset_root=args.dataset_root,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        image_size=args.image_size,
    )

    # Note: We only need the test loader for evaluation
    _, _, test_loader = make_loaders(
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
                backbone="resnet18",
                pretrained=False,  # No need to load pretrained weights, we are loading a checkpoint
                lstm_hidden_size=512,
                lstm_num_layers=1,
                lstm_bidirectional=False,
                lstm_dropout=0.0,
            )
        else:
            model = FrameMeanPoolBaseline(backbone=args.backbone, pretrained=False)
    else:
        model = R3D18Baseline(pretrained=False)
        
    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    print(f"Loaded model from {args.checkpoint}")

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"[test] loss={test_loss:.4f} acc={test_acc:.4f}")

if __name__ == "__main__":
    main()
