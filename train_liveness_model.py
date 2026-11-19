#!/usr/bin/env python3
"""
train_liveness_model.py — OptiPulse-Pay Liveness Model Training
Binary classifier: MobileNetV3-Small backbone → Sigmoid output
  Input:  64x64 RGB skin patch
  Output: scalar ∈ [0, 1]  (>0.5 → LIVE, <=0.5 → SPOOF)
Saves trained weights to models/optipulse_liveness_mobilenet.pth
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import Tuple

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import Image
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_LIVE_DIR  = Path("data/train/live")
TRAIN_SPOOF_DIR = Path("data/train/spoof")
VAL_LIVE_DIR    = Path("data/val/live")
VAL_SPOOF_DIR   = Path("data/val/spoof")
MODEL_SAVE_PATH = Path("models/optipulse_liveness_mobilenet.pth")
METRICS_PATH    = Path("models/training_metrics.json")

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE    = 64
NUM_EPOCHS    = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
PATCH_SIZE    = 64
NUM_WORKERS   = 0          # set to 0 for Windows compatibility
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ────────────────────────────────────────────────────────────────────
class LivenessDataset(Dataset):
    """
    Binary liveness dataset.
    Label: 1 = LIVE, 0 = SPOOF
    """
    def __init__(self, live_dir: Path, spoof_dir: Path, transform=None):
        live_images  = sorted(live_dir.glob("*.png"))  + sorted(live_dir.glob("*.jpg"))
        spoof_images = sorted(spoof_dir.glob("*.png")) + sorted(spoof_dir.glob("*.jpg"))

        if not live_images:
            raise FileNotFoundError(
                f"No live images found in {live_dir}. "
                "Run synthetic_data_gen.py first."
            )
        if not spoof_images:
            raise FileNotFoundError(
                f"No spoof images found in {spoof_dir}. "
                "Run synthetic_data_gen.py first."
            )

        self.samples = (
            [(str(p), 1) for p in live_images] +
            [(str(p), 0) for p in spoof_images]
        )
        self.transform = transform
        print(f"  Dataset: {len(live_images)} live + {len(spoof_images)} spoof "
              f"= {len(self.samples)} total")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# ── Transforms ─────────────────────────────────────────────────────────────────
def get_transforms(is_train: bool):
    if is_train:
        return transforms.Compose([
            transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomRotation(degrees=15),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ── Model Architecture ─────────────────────────────────────────────────────────
class OptiPulseLivenessNet(nn.Module):
    """
    MobileNetV3-Small backbone with custom binary classification head.
    Backbone: pretrained MobileNetV3-Small (ImageNet)
    Head:     Dropout(0.3) → Linear(576→128) → HardSwish → Linear(128→1) → Sigmoid
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base    = models.mobilenet_v3_small(weights=weights)

        # Extract feature backbone (all layers except classifier)
        self.features   = base.features
        self.avgpool    = base.avgpool

        # Custom lightweight classification head
        in_features = 576  # MobileNetV3-Small avgpool output
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.squeeze(1)   # (B,)


# ── Training Loop ──────────────────────────────────────────────────────────────
def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: optim.Optimizer, criterion: nn.Module,
                    epoch: int) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:02d} [Train]", leave=False,
                ncols=90, unit="batch")
    for imgs, labels in pbar:
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        preds = model(imgs)
        loss  = criterion(preds, labels)
        loss.backward()

        # Gradient clipping to stabilize training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        predicted   = (preds >= 0.5).float()
        correct    += (predicted == labels).sum().item()
        total      += imgs.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{correct/total:.3f}")

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module) -> Tuple[float, float, float, float]:
    """
    Run validation pass. Returns (avg_loss, accuracy, precision, recall).
    """
    model.eval()
    total_loss, tp, fp, fn, tn = 0.0, 0, 0, 0, 0

    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        preds  = model(imgs)
        loss   = criterion(preds, labels)
        total_loss += loss.item() * imgs.size(0)

        predicted = (preds >= 0.5).float()
        tp += ((predicted == 1) & (labels == 1)).sum().item()
        fp += ((predicted == 1) & (labels == 0)).sum().item()
        fn += ((predicted == 0) & (labels == 1)).sum().item()
        tn += ((predicted == 0) & (labels == 0)).sum().item()

    total    = tp + fp + fn + tn
    avg_loss = total_loss / max(total, 1)
    accuracy  = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    return avg_loss, accuracy, precision, recall


def train(num_epochs: int = NUM_EPOCHS, batch_size: int = BATCH_SIZE,
          lr: float = LEARNING_RATE, pretrained: bool = True) -> None:
    """Full training pipeline with learning rate scheduling and best-model checkpointing."""
    print("=" * 60)
    print("  OptiPulse-Pay — Liveness Model Training")
    print("=" * 60)
    print(f"  Device   : {DEVICE}")
    print(f"  Backbone : MobileNetV3-Small (pretrained={pretrained})")
    print(f"  Epochs   : {num_epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print()

    # Datasets
    print("[1/4] Loading datasets...")
    train_ds = LivenessDataset(TRAIN_LIVE_DIR, TRAIN_SPOOF_DIR, get_transforms(True))
    val_ds   = LivenessDataset(VAL_LIVE_DIR,   VAL_SPOOF_DIR,   get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
                               drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))

    # Model
    print("[2/4] Building model...")
    model     = OptiPulseLivenessNet(pretrained=pretrained).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # Cosine annealing LR schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc  = 0.0
    history       = []

    print(f"[3/4] Training for {num_epochs} epochs...\n")
    for epoch in range(1, num_epochs + 1):
        t0 = time.perf_counter()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        vl_loss, vl_acc, vl_prec, vl_rec = validate(model, val_loader, criterion)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        f1 = 2 * vl_prec * vl_rec / max(vl_prec + vl_rec, 1e-8)

        print(
            f"  Epoch {epoch:02d}/{num_epochs} | "
            f"TR loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"VL loss={vl_loss:.4f} acc={vl_acc:.3f} "
            f"prec={vl_prec:.3f} rec={vl_rec:.3f} F1={f1:.3f} | "
            f"{elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 6),
            "train_acc":  round(tr_acc, 6),
            "val_loss":   round(vl_loss, 6),
            "val_acc":    round(vl_acc, 6),
            "val_prec":   round(vl_prec, 6),
            "val_rec":    round(vl_rec, 6),
            "val_f1":     round(f1, 6),
        })

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc":        vl_acc,
                "val_f1":         f1,
                "config": {
                    "patch_size": PATCH_SIZE,
                    "threshold":  0.5,
                    "backbone":   "mobilenet_v3_small",
                },
            }, MODEL_SAVE_PATH)
            print(f"    ✓ Best model saved (val_acc={vl_acc:.4f})")

    # Persist training metrics
    with open(METRICS_PATH, "w") as f:
        json.dump({"history": history, "best_val_acc": best_val_acc}, f, indent=2)

    print(f"\n[4/4] Training complete.")
    print(f"  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Model saved to    : {MODEL_SAVE_PATH}")
    print(f"  Metrics saved to  : {METRICS_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="OptiPulse-Pay Liveness Model Training"
    )
    parser.add_argument("--epochs",     type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)
    parser.add_argument("--no-pretrain", action="store_true",
                        help="Train from scratch without ImageNet weights")
    args = parser.parse_args()

    train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pretrained=not args.no_pretrain,
    )


if __name__ == "__main__":
    main()
