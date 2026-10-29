# Results Table Draft

> Fill in actual numbers after running `python main.py`.
> These are placeholders based on expected performance ranges from literature.

## Main Results (DermaMNIST Test Set)

| Model | Params | Macro-F1 (mean ± std) | Bal. Acc (mean ± std) |
|-------|--------|----------------------|-----------------------|
| **Baseline CNN** | 128,487 | `[run to fill]` | `[run to fill]` |
| **Method A** (ResNet-18, CE) | 11,177,543 | `[run to fill]` | `[run to fill]` |
| **Method B** (ResNet-18, Focal+Aug) | 11,177,543 | `[run to fill]` | `[run to fill]` |

*Seeds: 42, 123, 999. Evaluated on the official DermaMNIST test set (2,005 samples).*

---

## Expected Performance Ranges (from DermaMNIST literature)

| Model Type | Macro-F1 (approx.) | Notes |
|------------|-------------------|-------|
| Simple CNN (scratch) | 0.55 – 0.68 | Limited by class imbalance and small size |
| ResNet-18 (pretrained, CE) | 0.70 – 0.78 | Standard MedMNIST baseline |
| ResNet-18 + Focal Loss | 0.74 – 0.82 | Reported improvements on minority classes |

*Reference: Yang et al. (2023) MedMNIST v2 benchmark results.*

---

## Training Configuration Recap

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (lr=1e-3, wd=1e-4) |
| LR Schedule | Cosine Annealing (T=30, min=1e-6) |
| Max Epochs | 30 |
| Early Stopping | Patience=7 (val loss) |
| Batch Size | 64 |
| Image Size | 224×224 |
| Seeds | 42, 123, 999 |
| GPU | T4 (Google Colab free tier) |
