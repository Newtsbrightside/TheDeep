"""
src/train.py
============
Training loop for one (model_name, seed) combination.

Features
--------
* Deterministic seed-fixing (Python, NumPy, PyTorch CPU & CUDA).
* Checkpoint: saves the best model (by validation loss) to
  outputs/checkpoints/<model>_seed<seed>_best.pt
* Logging: prints per-epoch metrics and saves them to a CSV.
* Returns training / validation loss histories for curve plotting.

Usage (direct)
--------------
    python -m src.train --model baseline --seed 42 --epochs 30
    python -m src.train --model method_a --seed 42 --epochs 30
    python -m src.train --model method_b --seed 42 --epochs 30

Called automatically by main.py.
"""

import argparse
import csv
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import (
    NUM_CLASSES, get_dataloaders, get_class_weights,
)
from src.models import (
    build_model, count_parameters,
    get_criterion_baseline, get_criterion_a, get_criterion_b,
)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
CKPT_DIR    = Path("outputs/checkpoints")
RESULTS_DIR = Path("outputs/results")


# ──────────────────────────────────────────────────────────────────────────────
# Seed fixing
# ──────────────────────────────────────────────────────────────────────────────

def fix_seed(seed: int) -> None:
    """
    Fix ALL sources of randomness for full reproducibility.

    Sources fixed
    -------------
    * Python built-in `random` module
    * NumPy global RNG
    * PyTorch CPU RNG
    * PyTorch CUDA RNG (all GPUs)
    * cuDNN deterministic mode (may slow down training slightly)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)
    print(f"[seed] Fixed all seeds → {seed}")


# ──────────────────────────────────────────────────────────────────────────────
# Training / validation step helpers
# ──────────────────────────────────────────────────────────────────────────────

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    phase: str = "train",
) -> Tuple[float, float]:
    """
    Run one epoch of training or validation.

    Returns
    -------
    (avg_loss, accuracy) over the entire loader.
    """
    is_train = phase == "train"
    model.train(is_train)

    running_loss   = 0.0
    correct        = 0
    total          = 0

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            # MedMNIST labels shape: (B, 1) → flatten to (B,)
            labels = labels.to(device, non_blocking=True).squeeze(1).long()

            logits = model(imgs)                       # (B, C)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds         = logits.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += imgs.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train(
    model_name: str,
    seed:       int,
    epochs:     int    = 30,
    lr:         float  = 1e-3,
    batch_size: int    = 64,
    img_size:   int    = 224,
    patience:   int    = 7,
    device_str: str    = "auto",
) -> Dict:
    """
    Full training loop for one (model_name, seed) run.

    Parameters
    ----------
    model_name  : "baseline" | "method_a" | "method_b"
    seed        : Random seed.
    epochs      : Maximum training epochs (same for all models – fair comparison).
    lr          : Initial learning rate.
    batch_size  : Mini-batch size.
    img_size    : Spatial resolution (224 for ResNet; 224 also for baseline).
    patience    : Early-stopping patience (epochs without val-loss improvement).
    device_str  : "auto" | "cpu" | "cuda".

    Returns
    -------
    dict with keys:
        model_name, seed, best_val_loss, best_epoch,
        train_losses, val_losses, train_accs, val_accs,
        n_params, train_time_sec, ckpt_path
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    fix_seed(seed)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Model : {model_name.upper()}   Seed : {seed}   Device : {device}")
    print(f"{'='*65}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        method=model_name, img_size=img_size, batch_size=batch_size
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(model_name).to(device)
    n_params = count_parameters(model)
    print(f"  Parameters (trainable) : {n_params:,}")

    # Print layer shapes for Baseline
    if model_name == "baseline":
        model.describe_shapes()

    # ── Loss function ─────────────────────────────────────────────────────────
    if model_name == "baseline":
        criterion = get_criterion_baseline()
    elif model_name == "method_a":
        criterion = get_criterion_a()
    elif model_name == "method_b":
        class_weights = get_class_weights(device)
        criterion = get_criterion_b(class_weights)
        print(f"  Class weights : {class_weights.cpu().numpy().round(3)}")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    # Adam with weight decay for all models (fair comparison).
    # ResNet fine-tuning typically benefits from a lower LR on the backbone;
    # here we use the same LR for all layers for simplicity and fairness.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing LR schedule (T_max = epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses, val_losses   = [], []
    train_accs,   val_accs     = [], []

    best_val_loss = float("inf")
    best_epoch    = 0
    epochs_no_improve = 0
    ckpt_path = CKPT_DIR / f"{model_name}_seed{seed}_best.pt"

    # CSV log
    log_path = RESULTS_DIR / f"{model_name}_seed{seed}_log.csv"
    log_header = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_header).writeheader()

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t_ep = time.time()

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, phase="train"
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, device, phase="val"
        )
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        ep_time    = time.time() - t_ep

        print(
            f"  Epoch {epoch:03d}/{epochs}  "
            f"TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}  "
            f"TrainAcc={train_acc:.3f}  ValAcc={val_acc:.3f}  "
            f"LR={current_lr:.2e}  [{ep_time:.1f}s]"
        )

        # Log to CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_header)
            writer.writerow({
                "epoch": epoch, "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_acc": round(val_acc, 6),
                "lr": current_lr,
            })

        # ── Checkpoint on best val_loss ───────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch":      epoch,
                "model_name": model_name,
                "seed":       seed,
                "state_dict": model.state_dict(),
                "val_loss":   val_loss,
                "val_acc":    val_acc,
                "n_params":   n_params,
            }, ckpt_path)
            print(f"  ✓ Checkpoint saved (best val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  [early stop] No improvement for {patience} epochs → stopping.")
                break

    train_time = time.time() - t_start

    print(f"\n  Training complete  |  Best epoch: {best_epoch}  "
          f"|  Best val_loss: {best_val_loss:.4f}  "
          f"|  Time: {train_time/60:.1f} min")

    return {
        "model_name":     model_name,
        "seed":           seed,
        "best_val_loss":  best_val_loss,
        "best_epoch":     best_epoch,
        "train_losses":   train_losses,
        "val_losses":     val_losses,
        "train_accs":     train_accs,
        "val_accs":       val_accs,
        "n_params":       n_params,
        "train_time_sec": train_time,
        "ckpt_path":      str(ckpt_path),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one model / seed")
    parser.add_argument("--model",   type=str, default="baseline",
                        choices=["baseline", "method_a", "method_b"])
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--epochs",  type=int, default=30)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--batch",   type=int, default=64)
    parser.add_argument("--device",  type=str, default="auto")
    args = parser.parse_args()

    result = train(
        model_name  = args.model,
        seed        = args.seed,
        epochs      = args.epochs,
        lr          = args.lr,
        batch_size  = args.batch,
        device_str  = args.device,
    )
    print(f"\nDone. Checkpoint: {result['ckpt_path']}")
