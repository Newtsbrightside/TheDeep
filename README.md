# DermaMNIST Deep Learning Project

> **Skin Lesion Classification with Baseline CNN, Transfer Learning, and Focal Loss**  
> IEEE Conference Paper Format · PyTorch · Google Colab T4 Compatible

---

## Quick Start

```bash
# 1. Clone / unzip the project
cd dl_project/

# 2. Create virtual environment (optional but recommended)
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run full pipeline (train all 3 models × 3 seeds, then evaluate)
python main.py
```

For a quick smoke-test (fewer epochs):
```bash
python main.py --epochs 5
```

---

## Project Structure

```
dl_project/
├── main.py                   # ← END-TO-END ENTRY POINT
├── run_experiments.sh        # Shell runner (wraps main.py)
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── dataset.py            # Data pipeline, leakage check, visualisations
│   ├── models.py             # Baseline CNN, Method A, Method B, FocalLoss
│   ├── train.py              # Training loop (seed, checkpoint, CSV log)
│   └── evaluate.py           # Multi-seed eval, confusion matrix, failure cases
│
├── data/                     # Auto-created: MedMNIST raw files (auto-downloaded)
├── splits/                   # Auto-created: train/val/test CSVs
└── outputs/
    ├── checkpoints/          # Best .pt files per model × seed
    ├── plots/                # All figures
    └── results/              # results_table.csv, results_table.md, per-epoch logs
```

---

## Dataset

| Property          | Value                                  |
|-------------------|----------------------------------------|
| Name              | DermaMNIST (MedMNIST v2)               |
| Task              | 7-class skin lesion classification     |
| Input             | 3-channel RGB image, resized to 224×224|
| Official split    | Train / Val / Test (no modification)   |
| Train samples     | 7,007                                  |
| Val samples       | 1,003                                  |
| Test samples      | 2,005                                  |
| License           | CC BY 4.0                              |
| Source            | [MedMNIST](https://medmnist.com/)      |

Classes (7): `actinic keratosis`, `basal cell carcinoma`, `benign keratosis`,
`dermatofibroma`, `melanoma`, `melanocytic nevi`, `vascular lesion`

---

## Experiments

| Model       | Description                                               |
|-------------|-----------------------------------------------------------|
| **Baseline**| 3-block CNN, trained from scratch, standard CE            |
| **Method A**| ResNet-18 (ImageNet pretrained), full fine-tune, std CE   |
| **Method B**| ResNet-18 (ImageNet pretrained) + RandAugment + Focal Loss|

All models use:
- **Same data split** (official DermaMNIST)
- **Same training budget** (up to 30 epochs + early stopping patience=7)
- **Same optimizer**: Adam (lr=1e-3, weight_decay=1e-4)
- **Same scheduler**: Cosine Annealing (T_max=30, eta_min=1e-6)
- **3 seeds**: 42, 123, 999

---

## CLI Reference

```bash
# Train a single model (one seed)
python -m src.train --model method_b --seed 42 --epochs 30

# Evaluate only (checkpoints must already exist)
python main.py --skip_train

# Train only one model type (all seeds)
python main.py --model method_a

# Force CPU
python main.py --device cpu

# Shell runner
./run_experiments.sh --epochs 30 --device auto
```

---

## Reproducing Results

1. All random seeds are fixed at the **top of every training call** via `fix_seed()`:
   - `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`
   - `cudnn.deterministic = True`, `cudnn.benchmark = False`, `PYTHONHASHSEED`
2. The official DermaMNIST split is used as-is (no re-splitting).
3. The split CSVs in `splits/` document exactly which samples belong to each set.

---

## Outputs After Full Run

| File | Description |
|------|-------------|
| `outputs/plots/class_distribution.png` | Class counts per split |
| `outputs/plots/sample_grid.png` | Example training images per class |
| `outputs/plots/learning_curves.png` | Train/val loss & acc per model |
| `outputs/plots/confusion_matrix_method_b_seed42.png` | Confusion matrix (best model) |
| `outputs/plots/failure_cases_method_b_seed42.png` | 3 misclassified samples |
| `outputs/results/results_table.md` | Main results table (mean ± std) |
| `outputs/results/results_table.csv` | Same table in CSV |
| `outputs/results/<model>_seed<N>_log.csv` | Per-epoch training log |
| `outputs/checkpoints/<model>_seed<N>_best.pt` | Best checkpoints |
| `splits/<split>_split.csv` | Index→label mapping per split |

---

## Hardware Requirements

| Setting         | Value                              |
|-----------------|------------------------------------|
| GPU             | Free Colab T4 (or equivalent)      |
| VRAM needed     | ~4 GB (batch_size=64, img=224)     |
| Approx. runtime | ~60–90 min (3 models × 3 seeds × 30 ep) |
| CPU fallback    | Supported (slower)                 |

---

## Dependencies

See [`requirements.txt`](requirements.txt). Key packages:

```
torch==2.2.0        torchvision==0.17.0
medmnist==3.0.2     scikit-learn==1.4.1.post1
matplotlib==3.8.3   seaborn==0.13.2
numpy==1.26.4       pandas==2.2.1
```

---

## AI Tool Usage

This project was developed with assistance from Google Antigravity AI for code scaffolding and structure. All methods, justifications, and experimental decisions were reviewed and approved by the team. See Appendix B of the report for the full AI usage statement.
