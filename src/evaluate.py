"""
src/evaluate.py
===============
Multi-seed evaluation, plotting, and failure-case export.

Functions
---------
evaluate_seed()        – Load best checkpoint, run on test set, compute metrics.
aggregate_results()    – Compute mean ± std across seeds for a model.
plot_learning_curves() – Loss/acc curves for one representative seed per model.
plot_confusion_matrix()– Confusion matrix heatmap for the best model on test set.
export_failure_cases() – Save 3 test samples where the best model fails.
run_full_evaluation()  – Orchestrate all of the above.

Usage
-----
    python -m src.evaluate
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.dataset import (
    NUM_CLASSES, CLASS_NAMES, get_dataloaders, get_class_weights
)
from src.models import (
    build_model, get_criterion_baseline, get_criterion_a, get_criterion_b
)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
CKPT_DIR    = Path("outputs/checkpoints")
PLOTS_DIR   = Path("outputs/plots")
RESULTS_DIR = Path("outputs/results")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ──────────────────────────────────────────────────────────────────────────────

def _collect_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model on loader and collect ground-truth labels, predicted labels,
    and prediction probabilities.

    Returns
    -------
    (y_true, y_pred, y_prob)
    y_true : (N,) int
    y_pred : (N,) int
    y_prob : (N, C) float  [softmax probabilities]
    """
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).squeeze(1).long()
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)

            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    return y_true, y_pred, y_prob


# ──────────────────────────────────────────────────────────────────────────────
# Single-seed evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_seed(
    model_name: str,
    seed: int,
    batch_size: int = 64,
    img_size: int = 224,
    device_str: str = "auto",
) -> Dict:
    """
    Load the best checkpoint for (model_name, seed) and evaluate on the test set.

    Returns
    -------
    dict with keys: model_name, seed, balanced_accuracy, macro_f1,
                    y_true, y_pred, y_prob
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    ckpt_path = CKPT_DIR / f"{model_name}_seed{seed}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run training first with main.py or src/train.py."
        )

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[eval] Loaded {ckpt_path.name}  "
          f"(epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    # Test loader (always deterministic transform)
    _, _, test_loader = get_dataloaders(
        method=model_name, img_size=img_size, batch_size=batch_size
    )

    y_true, y_pred, y_prob = _collect_predictions(model, test_loader, device)

    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"  Balanced Accuracy : {bal_acc:.4f}")
    print(f"  Macro-F1          : {macro_f1:.4f}")

    return {
        "model_name":        model_name,
        "seed":              seed,
        "n_params":          ckpt.get("n_params", 0),
        "balanced_accuracy": bal_acc,
        "macro_f1":          macro_f1,
        "y_true":            y_true,
        "y_pred":            y_pred,
        "y_prob":            y_prob,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Multi-seed aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_results(
    model_name: str,
    seeds: List[int],
    **eval_kwargs,
) -> Dict:
    """
    Evaluate all seeds and compute mean ± std for each metric.

    Returns
    -------
    dict with keys: model_name, n_params, seeds,
                    bal_acc_mean, bal_acc_std, f1_mean, f1_std,
                    per_seed (list of individual results)
    """
    print(f"\n{'─'*55}")
    print(f"  Aggregating results for: {model_name.upper()}")
    print(f"{'─'*55}")

    per_seed = []
    for seed in seeds:
        res = evaluate_seed(model_name, seed, **eval_kwargs)
        per_seed.append(res)

    bal_accs = [r["balanced_accuracy"] for r in per_seed]
    f1s      = [r["macro_f1"]          for r in per_seed]
    n_params = per_seed[0]["n_params"]

    summary = {
        "model_name":   model_name,
        "n_params":     n_params,
        "seeds":        seeds,
        "bal_acc_mean": np.mean(bal_accs),
        "bal_acc_std":  np.std(bal_accs),
        "f1_mean":      np.mean(f1s),
        "f1_std":       np.std(f1s),
        "per_seed":     per_seed,
    }
    print(
        f"\n  {model_name.upper():12s}  "
        f"Bal-Acc = {summary['bal_acc_mean']:.4f} ± {summary['bal_acc_std']:.4f}  "
        f"Macro-F1 = {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}"
    )
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Learning curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(
    model_results: Dict[str, Dict],
    representative_seed: int = 42,
) -> None:
    """
    Plot train/val loss and accuracy curves for one representative seed per model.

    Parameters
    ----------
    model_results : dict  {model_name: train() return dict}
                   (from training with representative_seed)
    """
    n_models = len(model_results)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 4 * n_models))
    if n_models == 1:
        axes = [axes]

    colors = {"train": "#4C9BE8", "val": "#E8854C"}

    for row, (model_name, res) in enumerate(model_results.items()):
        epochs_range = range(1, len(res["train_losses"]) + 1)

        # Loss
        ax_loss = axes[row][0]
        ax_loss.plot(epochs_range, res["train_losses"],
                     color=colors["train"], lw=2, label="Train Loss")
        ax_loss.plot(epochs_range, res["val_losses"],
                     color=colors["val"], lw=2, linestyle="--", label="Val Loss")
        ax_loss.axvline(x=res["best_epoch"], color="green",
                        linestyle=":", lw=1.5, label=f"Best epoch ({res['best_epoch']})")
        ax_loss.set_title(f"{model_name.replace('_', ' ').title()} – Loss (seed {representative_seed})")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(fontsize=8)
        ax_loss.grid(True, alpha=0.3)

        # Accuracy
        ax_acc = axes[row][1]
        ax_acc.plot(epochs_range, res["train_accs"],
                    color=colors["train"], lw=2, label="Train Acc")
        ax_acc.plot(epochs_range, res["val_accs"],
                    color=colors["val"], lw=2, linestyle="--", label="Val Acc")
        ax_acc.axvline(x=res["best_epoch"], color="green",
                       linestyle=":", lw=1.5, label=f"Best epoch ({res['best_epoch']})")
        ax_acc.set_title(f"{model_name.replace('_', ' ').title()} – Accuracy (seed {representative_seed})")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend(fontsize=8)
        ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = PLOTS_DIR / "learning_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[plot] Saved learning curves → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Confusion matrix
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    seed: int,
    normalize: bool = True,
) -> None:
    """
    Heatmap confusion matrix for the test set predictions.

    Saves to outputs/plots/confusion_matrix_<model>_seed<seed>.png.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, vmax = ".2f", 1.0
        title_suffix = " (row-normalised)"
    else:
        cm_plot = cm
        fmt, vmax = "d", None
        title_suffix = " (counts)"

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, vmin=0, vmax=vmax, linewidths=0.5, linecolor="grey",
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(
        f"Confusion Matrix – {model_name.replace('_',' ').title()}"
        f" (seed {seed}){title_suffix}",
        fontsize=12, pad=12
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()
    out_path = PLOTS_DIR / f"confusion_matrix_{model_name}_seed{seed}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved confusion matrix → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Failure cases
# ──────────────────────────────────────────────────────────────────────────────

def export_failure_cases(
    model_name: str,
    seed: int,
    n_cases: int = 3,
    batch_size: int = 64,
    img_size: int = 224,
    device_str: str = "auto",
) -> None:
    """
    Find and visualise n_cases test samples where the model is wrong.

    For each failure case, we show:
      - The image (de-normalised)
      - True label
      - Predicted label
      - Max softmax probability (confidence)
      - Hypothesis for why the model failed

    Saves to outputs/plots/failure_cases_<model>_seed<seed>.png.
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    ckpt_path = CKPT_DIR / f"{model_name}_seed{seed}_best.pt"
    ckpt      = torch.load(ckpt_path, map_location=device)
    model     = build_model(model_name).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Need raw images (for display, we undo normalisation)
    import torchvision.transforms as T
    from src.dataset import IMAGENET_MEAN, IMAGENET_STD, _load_split

    # Display transform: resize only (no normalisation)
    display_transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    test_ds_raw = _load_split("test", display_transform)

    # Eval transform (normalised) for inference
    from src.dataset import get_val_test_transform
    eval_transform = get_val_test_transform(img_size)
    test_ds_norm   = _load_split("test", eval_transform)

    # Find failures (iterate sequentially for determinism)
    failures = []
    with torch.no_grad():
        for idx in range(len(test_ds_norm)):
            img_norm, label = test_ds_norm[idx]
            label = int(label.item()) if hasattr(label, "item") else int(label)
            logits = model(img_norm.unsqueeze(0).to(device))
            probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred   = int(probs.argmax())
            if pred != label:
                img_raw, _ = test_ds_raw[idx]
                failures.append({
                    "idx":       idx,
                    "img_raw":   img_raw.permute(1, 2, 0).numpy(),  # (H, W, 3)
                    "true_lbl":  label,
                    "pred_lbl":  pred,
                    "confidence": float(probs[pred]),
                    "true_conf":  float(probs[label]),
                })
                if len(failures) >= n_cases:
                    break

    if not failures:
        print("[failure] No failures found on test set (model is perfect!).")
        return

    # Pre-generated hypotheses for common DermaMNIST confusions
    _HYPOTHESES = {
        (0, 1): "Melanocytic nevi and melanoma share similar pigmentation patterns; "
                "the model may confuse subtle textural differences.",
        (1, 0): "Melanoma vs. nevi confusion; the model under-weighs lesion asymmetry cues.",
        (4, 6): "Vascular lesions and dermatofibromas can both appear as flat, reddish spots.",
    }

    def _get_hypothesis(true_c: int, pred_c: int) -> str:
        key = (true_c, pred_c)
        if key in _HYPOTHESES:
            return _HYPOTHESES[key]
        # Generic hypothesis
        return (
            f"The model predicted '{CLASS_NAMES[pred_c]}' instead of "
            f"'{CLASS_NAMES[true_c]}'. Both classes may share visual "
            f"features (colour, texture) at this resolution, and the "
            f"training set for the true class may be under-represented."
        )

    # Plot
    fig, axes = plt.subplots(1, n_cases, figsize=(6 * n_cases, 6))
    if n_cases == 1:
        axes = [axes]

    fig.suptitle(
        f"Failure Cases – {model_name.replace('_', ' ').title()} (seed {seed})",
        fontsize=13, y=1.02
    )

    for ax, fc in zip(axes, failures):
        ax.imshow(fc["img_raw"], interpolation="bilinear")
        ax.axis("off")

        true_name = CLASS_NAMES[fc["true_lbl"]]
        pred_name = CLASS_NAMES[fc["pred_lbl"]]
        hypo      = _get_hypothesis(fc["true_lbl"], fc["pred_lbl"])

        title = (
            f"Sample #{fc['idx']}\n"
            f"True  : {true_name}  (conf={fc['true_conf']:.2f})\n"
            f"Pred  : {pred_name}  (conf={fc['confidence']:.2f})"
        )
        ax.set_title(title, fontsize=8, loc="center", pad=4)

        # Hypothesis as figure text below image
        ax.text(
            0.5, -0.02, f"Hypothesis:\n{hypo}",
            transform=ax.transAxes, fontsize=6.5,
            ha="center", va="top", style="italic",
            wrap=True, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.7)
        )

    plt.tight_layout()
    out_path = PLOTS_DIR / f"failure_cases_{model_name}_seed{seed}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved failure cases → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Results table
# ──────────────────────────────────────────────────────────────────────────────

def save_results_table(summaries: List[Dict]) -> None:
    """
    Save the main results table as CSV and Markdown.

    Table format matches the guideline:
    | Model | Params | Macro-F1 (mean ± std) | Bal. Acc (mean ± std) |
    """
    rows = []
    for s in summaries:
        rows.append({
            "Model":   s["model_name"].replace("_", " ").title(),
            "Params":  f"{s['n_params']:,}",
            "Macro-F1 (mean ± std)":
                f"{s['f1_mean']:.4f} ± {s['f1_std']:.4f}",
            "Bal. Acc (mean ± std)":
                f"{s['bal_acc_mean']:.4f} ± {s['bal_acc_std']:.4f}",
        })

    df = pd.DataFrame(rows)

    # CSV
    csv_path = RESULTS_DIR / "results_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[results] Saved results table (CSV) → {csv_path}")

    # Markdown
    md_path = RESULTS_DIR / "results_table.md"
    with open(md_path, "w") as f:
        f.write("## Main Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n*Metrics computed on the test set over 3 seeds (42, 123, 999).*\n")
    print(f"[results] Saved results table (MD)  → {md_path}")

    # Print to console
    print("\n" + df.to_markdown(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# Full evaluation orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(
    seeds: List[int] = [42, 123, 999],
    model_names: List[str] = ["baseline", "method_a", "method_b"],
    representative_seed: int = 42,
    training_curves: Dict[str, Dict] | None = None,
    best_model: str = "method_b",
    device_str: str = "auto",
) -> List[Dict]:
    """
    Full evaluation pipeline.

    Steps
    -----
    1. Multi-seed evaluation (Balanced Accuracy + Macro-F1) for all models.
    2. Save results table (CSV + Markdown).
    3. Plot confusion matrix for best model (representative seed).
    4. Export 3 failure cases for best model.
    5. (Optional) Plot learning curves if training_curves is provided.

    Parameters
    ----------
    training_curves : {model_name: train() result dict} for the representative
                      seed.  If None, learning curves are skipped.
    best_model      : Name of the best-performing model (used for confusion
                      matrix and failure cases).

    Returns
    -------
    List of per-model summary dicts.
    """
    print("\n" + "═" * 65)
    print("  EVALUATION PIPELINE")
    print("═" * 65)

    summaries = []
    for model_name in model_names:
        summary = aggregate_results(model_name, seeds, device_str=device_str)
        summaries.append(summary)

    save_results_table(summaries)

    # Confusion matrix – best model, representative seed
    print(f"\n[eval] Plotting confusion matrix for {best_model} …")
    res_best = evaluate_seed(best_model, representative_seed, device_str=device_str)
    plot_confusion_matrix(
        res_best["y_true"], res_best["y_pred"],
        model_name=best_model, seed=representative_seed,
        normalize=True,
    )

    # Failure cases – best model, representative seed
    print(f"\n[eval] Exporting failure cases for {best_model} …")
    export_failure_cases(best_model, representative_seed, device_str=device_str)

    # Learning curves (if supplied)
    if training_curves:
        plot_learning_curves(training_curves, representative_seed)

    # Per-model classification report (printed to console)
    print("\n\n" + "─" * 65)
    print("  Per-class Classification Report (best model, rep. seed)")
    print("─" * 65)
    print(classification_report(
        res_best["y_true"], res_best["y_pred"],
        target_names=CLASS_NAMES, zero_division=0
    ))

    return summaries


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run full evaluation pipeline")
    parser.add_argument("--seeds",  nargs="+", type=int, default=[42, 123, 999])
    parser.add_argument("--best",   type=str, default="method_b")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_full_evaluation(
        seeds       = args.seeds,
        best_model  = args.best,
        device_str  = args.device,
    )
