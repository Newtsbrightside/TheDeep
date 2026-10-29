"""
src/dataset.py
==============
Pipeline for DermaMNIST (MedMNIST package).

Responsibilities
----------------
1. Download the dataset via medmnist (official train/val/test split used as-is).
2. Save split metadata (index → label) to CSV for traceability.
3. Leakage check: confirm zero intersection of sample indices across splits.
4. Visualise class distribution per split and a grid of sample images.
5. Provide get_dataloaders() factory used by training scripts.

Usage (standalone)
------------------
    python -m src.dataset
"""

import os
import csv
import hashlib
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless-friendly backend
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

import medmnist
from medmnist import DermaMNIST, INFO

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
DATASET_NAME  = "dermamnist"
IMG_SIZE      = 28                    # native MedMNIST resolution
BATCH_SIZE    = 64
NUM_WORKERS   = 2
DATA_ROOT     = Path("data")          # downloaded dataset cached here
DATA_ROOT.mkdir(parents=True, exist_ok=True)   # ensure it exists before medmnist uses it
SPLITS_DIR    = Path("splits")        # CSV files saved here
PLOTS_DIR     = Path("outputs/plots")

# ImageNet statistics (used for pretrained ResNet normalisation)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────────────────────────────────────
# Label names (DermaMNIST has 7 skin-lesion categories)
# ──────────────────────────────────────────────────────────────────────────────
INFO_DICT     = INFO[DATASET_NAME]
NUM_CLASSES   = len(INFO_DICT["label"])
CLASS_NAMES   = [INFO_DICT["label"][str(i)] for i in range(NUM_CLASSES)]


# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────

def get_base_transform(size: int = 224) -> T.Compose:
    """Resize + normalise only – used for Baseline CNN and Method A."""
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_augmented_transform(size: int = 224) -> T.Compose:
    """
    Augmented pipeline used for Method B.

    Design decision
    ---------------
    RandAugment (Cubuk et al., 2020) applies N randomly chosen photometric /
    geometric transformations at each iteration.  It is particularly effective
    for small dermatology datasets because it diversifies texture and colour
    presentation without hand-crafting individual augmentations.
    RandomHorizontalFlip and RandomVerticalFlip are safe for skin lesion images
    because orientation is not diagnostically meaningful.
    """
    return T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),   # N=2, M=9 (standard default)
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_test_transform(size: int = 224) -> T.Compose:
    """Deterministic transform for validation / test sets (no augmentation)."""
    return get_base_transform(size)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_split(split: str, transform: T.Compose, download: bool = True) -> DermaMNIST:
    """Return a DermaMNIST dataset object for the given split."""
    return DermaMNIST(
        split=split,
        transform=transform,
        download=download,
        root=str(DATA_ROOT),
        size=IMG_SIZE,          # raw 28×28; transforms will resize later
    )


def _labels_array(dataset: DermaMNIST) -> np.ndarray:
    """Extract flat integer label array from a MedMNIST dataset."""
    return dataset.labels.flatten().astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# Split CSV – traceability
# ──────────────────────────────────────────────────────────────────────────────

def save_split_csvs() -> None:
    """
    Save index→label mappings for train/val/test to CSV.

    These CSVs serve two purposes:
      - They document which samples belong to which split (traceability).
      - They are used in the leakage check below.

    NOTE: MedMNIST provides an official, pre-defined split, so we use it
    without any modification, satisfying the guideline requirement.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    identity = T.Compose([T.ToTensor()])   # minimal transform just to load

    for split in ("train", "val", "test"):
        ds = _load_split(split, identity)
        labels = _labels_array(ds)
        rows = [{"index": i, "label": labels[i], "label_name": CLASS_NAMES[labels[i]]}
                for i in range(len(ds))]
        path = SPLITS_DIR / f"{split}_split.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "label", "label_name"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[split] Saved {split} split → {path}  ({len(ds)} samples)")


# ──────────────────────────────────────────────────────────────────────────────
# Leakage check
# ──────────────────────────────────────────────────────────────────────────────

def _md5_sample(img_tensor: torch.Tensor) -> str:
    """Return MD5 hash of a tensor's byte representation."""
    return hashlib.md5(img_tensor.numpy().tobytes()).hexdigest()


def check_data_leakage(verbose: bool = True) -> bool:
    """
    Verify no sample appears in more than one split.

    Strategy
    --------
    DermaMNIST provides an official split; index spaces are disjoint by
    construction.  We verify this by computing per-sample MD5 hashes and
    checking for collisions across splits.  A hash collision would indicate
    a duplicate image appearing in multiple splits.

    Returns
    -------
    True  – no leakage detected
    False – leakage found (raises ValueError as well)
    """
    print("\n[leakage] Computing sample hashes …")
    identity = T.Compose([T.ToTensor()])
    hash_sets: Dict[str, set] = {}

    for split in ("train", "val", "test"):
        ds = _load_split(split, identity)
        hashes = set()
        for img, _ in ds:
            hashes.add(_md5_sample(img))
        hash_sets[split] = hashes
        print(f"  {split}: {len(hashes)} unique hashes (total {len(ds)} samples)")

    # Pairwise intersection checks
    leakage_found = False
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for s1, s2 in pairs:
        overlap = hash_sets[s1] & hash_sets[s2]
        if overlap:
            print(f"  ⚠️  LEAKAGE: {len(overlap)} duplicate(s) between {s1} and {s2}!")
            leakage_found = True
        else:
            if verbose:
                print(f"  ✓  No overlap between {s1} and {s2}")

    if leakage_found:
        raise ValueError("Data leakage detected! Review dataset preparation.")

    print("[leakage] PASSED – no cross-split duplicates found.\n")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Visualisations
# ──────────────────────────────────────────────────────────────────────────────

def plot_class_distribution() -> None:
    """
    Bar chart of class frequency per split.

    Saves to outputs/plots/class_distribution.png.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    identity = T.Compose([T.ToTensor()])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle("DermaMNIST – Class Distribution per Split", fontsize=14, y=1.02)

    palette = sns.color_palette("tab10", NUM_CLASSES)

    for ax, split in zip(axes, ("train", "val", "test")):
        ds = _load_split(split, identity)
        labels = _labels_array(ds)
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        bars = ax.bar(range(NUM_CLASSES), counts, color=palette)
        ax.set_title(f"{split.capitalize()} ({len(ds)} samples)")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_xticklabels(
            [c[:12] for c in CLASS_NAMES],   # truncate long names
            rotation=40, ha="right", fontsize=8
        )
        # Annotate bar heights
        for bar, cnt in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                str(cnt), ha="center", va="bottom", fontsize=7
            )

    plt.tight_layout()
    out_path = PLOTS_DIR / "class_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved class distribution → {out_path}")


def plot_sample_grid(n_per_class: int = 4) -> None:
    """
    Grid of n_per_class sample images per class from the training set.

    Saves to outputs/plots/sample_grid.png.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    identity = T.Compose([T.ToTensor()])
    ds = _load_split("train", identity)
    labels = _labels_array(ds)

    # Collect indices per class
    class_indices: Dict[int, list] = {c: [] for c in range(NUM_CLASSES)}
    for idx, lbl in enumerate(labels):
        if len(class_indices[lbl]) < n_per_class:
            class_indices[lbl].append(idx)

    fig, axes = plt.subplots(
        NUM_CLASSES, n_per_class,
        figsize=(n_per_class * 2, NUM_CLASSES * 2)
    )
    fig.suptitle("DermaMNIST – Sample Images (train set)", fontsize=13)

    for c in range(NUM_CLASSES):
        for j, idx in enumerate(class_indices[c]):
            img_tensor, _ = ds[idx]           # (3, 28, 28), [0,1]
            img_np = img_tensor.permute(1, 2, 0).numpy()
            axes[c][j].imshow(img_np)
            axes[c][j].axis("off")
            if j == 0:
                axes[c][j].set_ylabel(
                    CLASS_NAMES[c][:14], rotation=0,
                    labelpad=60, fontsize=8, va="center"
                )

    plt.tight_layout()
    out_path = PLOTS_DIR / "sample_grid.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved sample grid → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    method: str = "baseline",
    img_size: int = 224,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader, test_loader) for the given method.

    Parameters
    ----------
    method      : "baseline" | "method_a" | "method_b"
                  Controls which transform is applied to the training set.
                  Validation and test sets always use the deterministic transform.
    img_size    : Spatial resolution fed to the model (default 224 for ResNet).
    batch_size  : Mini-batch size.
    num_workers : DataLoader worker processes.
    """
    if method == "method_b":
        train_transform = get_augmented_transform(img_size)
    else:
        train_transform = get_base_transform(img_size)

    val_test_transform = get_val_test_transform(img_size)

    train_ds = _load_split("train", train_transform)
    val_ds   = _load_split("val",   val_test_transform)
    test_ds  = _load_split("test",  val_test_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_class_weights(device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the training set.

    Used by Method B's weighted cross-entropy loss to handle class imbalance.
    Formula: w_c = N / (C * n_c)  where N = total samples, C = num classes,
    n_c = samples in class c.
    """
    identity = T.Compose([T.ToTensor()])
    ds = _load_split("train", identity)
    labels = _labels_array(ds)
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    weights = len(labels) / (NUM_CLASSES * counts)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights


def print_dataset_stats() -> None:
    """Print a summary table of dataset statistics."""
    identity = T.Compose([T.ToTensor()])
    print(f"\n{'='*60}")
    print(f"  Dataset : DermaMNIST (medmnist {medmnist.__version__})")
    print(f"  Classes : {NUM_CLASSES}  →  {CLASS_NAMES}")
    print(f"{'='*60}")
    print(f"  {'Split':<10} {'Samples':>8}  Class counts")
    print(f"  {'-'*55}")
    for split in ("train", "val", "test"):
        ds = _load_split(split, identity)
        labels = _labels_array(ds)
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        print(f"  {split:<10} {len(ds):>8}  {counts.tolist()}")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Standalone entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  DermaMNIST – Dataset Pipeline")
    print("=" * 60)

    # 1. Stats
    print_dataset_stats()

    # 2. Save CSVs
    save_split_csvs()

    # 3. Leakage check
    check_data_leakage()

    # 4. Visualisations
    plot_class_distribution()
    plot_sample_grid()

    # 5. Smoke-test a DataLoader
    print("[smoke] Testing DataLoader …")
    train_loader, val_loader, test_loader = get_dataloaders("method_b")
    imgs, labels = next(iter(train_loader))
    print(f"  Batch shape : {imgs.shape}   Labels shape : {labels.shape}")
    print(f"  Pixel range : [{imgs.min():.3f}, {imgs.max():.3f}]")
    print("\n[dataset.py] All checks passed ✓")
