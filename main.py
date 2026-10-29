"""
main.py
=======
End-to-end runner: trains all 3 models × 3 seeds, then evaluates.

Usage
-----
    python main.py                         # full pipeline
    python main.py --epochs 20             # shorter run
    python main.py --seeds 42 123 999      # specify seeds
    python main.py --skip_train            # evaluate only (checkpoints must exist)
    python main.py --model method_b        # train a single model across all seeds

Steps
-----
1.  Dataset smoke-test, leakage check, visualisations.
2.  Train baseline / method_a / method_b × {42, 123, 999}.
    (Saves checkpoints; training curves from seed 42 are kept for plotting.)
3.  Evaluate all models, aggregate mean ± std.
4.  Plot learning curves, confusion matrix, failure cases.
5.  Print final results table.
"""

import argparse
import sys
from pathlib import Path

# ── project root on path (allows `python main.py` from dl_project/) ──────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import print_dataset_stats, save_split_csvs, check_data_leakage
from src.dataset import plot_class_distribution, plot_sample_grid
from src.train   import train
from src.evaluate import run_full_evaluation, plot_learning_curves

# ──────────────────────────────────────────────────────────────────────────────
SEEDS       = [42, 123, 999]
MODEL_NAMES = ["baseline", "method_a", "method_b"]
EPOCHS      = 30          # same budget for all models → fair comparison
LR          = 1e-3
BATCH_SIZE  = 64
IMG_SIZE    = 224
PATIENCE    = 7
REP_SEED    = 42          # representative seed for plots


# ──────────────────────────────────────────────────────────────────────────────
def main(args):
    print("\n" + "█" * 65)
    print("  DermaMNIST Deep Learning Project – Full Pipeline")
    print("█" * 65)

    # ── Ensure all output directories exist (important for Colab) ─────────────
    from pathlib import Path
    for d in ["data", "splits", "outputs/checkpoints", "outputs/plots", "outputs/results"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("[init] Output directories ready.")

    # ── Step 1: Dataset preparation ───────────────────────────────────────────
    if not args.skip_train:
        print("\n── Step 1: Dataset Preparation ──")
        print_dataset_stats()
        save_split_csvs()
        check_data_leakage()
        plot_class_distribution()
        plot_sample_grid()

    # ── Step 2: Training ──────────────────────────────────────────────────────
    training_curves = {}   # {model_name: result dict from rep seed}

    models_to_train = (
        [args.model] if args.model else MODEL_NAMES
    )

    if not args.skip_train:
        print("\n── Step 2: Training ──")
        for model_name in models_to_train:
            for seed in args.seeds:
                result = train(
                    model_name  = model_name,
                    seed        = seed,
                    epochs      = args.epochs,
                    lr          = LR,
                    batch_size  = BATCH_SIZE,
                    img_size    = IMG_SIZE,
                    patience    = PATIENCE,
                    device_str  = args.device,
                )
                # Keep the representative seed result for curve plotting
                if seed == REP_SEED:
                    training_curves[model_name] = result
    else:
        print("\n[main] Skipping training (--skip_train flag set).")
        training_curves = None

    # ── Step 3: Evaluation ────────────────────────────────────────────────────
    print("\n── Step 3: Evaluation ──")
    summaries = run_full_evaluation(
        seeds               = args.seeds,
        model_names         = models_to_train,
        representative_seed = REP_SEED,
        training_curves     = training_curves,
        best_model          = "method_b",
        device_str          = args.device,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "█" * 65)
    print("  PIPELINE COMPLETE")
    print("█" * 65)
    print("\nOutputs:")
    print("  outputs/plots/class_distribution.png")
    print("  outputs/plots/sample_grid.png")
    print("  outputs/plots/learning_curves.png")
    print("  outputs/plots/confusion_matrix_method_b_seed42.png")
    print("  outputs/plots/failure_cases_method_b_seed42.png")
    print("  outputs/results/results_table.csv")
    print("  outputs/results/results_table.md")
    print("  outputs/checkpoints/*.pt")
    print("  splits/*.csv")
    print()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end DermaMNIST deep learning pipeline"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="List of random seeds (default: 42 123 999)"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Max training epochs (default: {EPOCHS})"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["baseline", "method_a", "method_b"],
        help="Train only one model (default: all)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device (default: auto)"
    )
    parser.add_argument(
        "--skip_train", action="store_true",
        help="Skip training and only run evaluation (checkpoints must exist)"
    )
    args = parser.parse_args()
    main(args)
