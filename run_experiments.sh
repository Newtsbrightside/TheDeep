#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
# ------------------
# Shell runner – trains all 3 models × 3 seeds sequentially and evaluates.
# Equivalent to: python main.py
#
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh                   # full run (30 epochs, GPU auto-detect)
#   ./run_experiments.sh --epochs 10       # quick smoke-test
# =============================================================================

set -euo pipefail   # exit on error, unset variable, or pipe failure

# ── Configuration (edit here) ─────────────────────────────────────────────────
EPOCHS=30
SEEDS="42 123 999"
DEVICE="auto"
# ──────────────────────────────────────────────────────────────────────────────

# Parse optional --epochs flag
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        *)        echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "========================================================="
echo "  DermaMNIST Experiments"
echo "  Epochs : $EPOCHS  |  Seeds : $SEEDS  |  Device : $DEVICE"
echo "========================================================="

# Activate virtual-env if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "[env] Activated .venv"
fi

# Install dependencies
pip install -q -r requirements.txt

# Run full pipeline via main.py
python main.py --epochs "$EPOCHS" --seeds $SEEDS --device "$DEVICE"

echo ""
echo "========================================================="
echo "  All experiments complete."
echo "  Results: outputs/results/results_table.md"
echo "========================================================="
