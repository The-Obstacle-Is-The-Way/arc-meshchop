# Evaluation Metrics & Benchmarks

> Extracted from: "State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters"

## Metrics Used

The paper evaluates models using three complementary metrics:

### 1. DICE Coefficient (Sørensen–Dice Index)

**Formula:**
```
DICE = 2 × |A ∩ B| / (|A| + |B|)
```

Where:
- A = predicted lesion voxels
- B = ground truth lesion voxels
- |A ∩ B| = number of overlapping voxels

**Interpretation:**
- Range: [0, 1]
- 0 = no overlap
- 1 = perfect overlap
- **Threshold for "reliable segmentation": 0.85** (per Liew et al.)

**Properties:**
- Most widely used segmentation metric
- Equally weights precision and recall
- Sensitive to both over- and under-segmentation

---

### 2. Average Volume Difference (AVD)

**Formula:**
```
AVD = |V_pred - V_true| / V_true
```

Where:
- V_pred = predicted lesion volume (voxel count)
- V_true = ground truth lesion volume

**Interpretation:**
- Range: [0, ∞)
- 0 = perfect volume match
- 0.25 = 25% volume difference
- **Lower is better (↓)**

**Properties:**
- Measures volumetric accuracy
- Important for clinical applications (lesion volume correlates with outcomes)
- Does not penalize shape errors (only total volume)

---

### 3. Matthews Correlation Coefficient (MCC)

**Formula:**
```
MCC = (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

Where:
- TP = True Positives (correctly predicted lesion voxels)
- TN = True Negatives (correctly predicted background voxels)
- FP = False Positives (background predicted as lesion)
- FN = False Negatives (lesion predicted as background)

**Interpretation:**
- Range: [-1, 1]
- -1 = total disagreement
- 0 = random prediction
- 1 = perfect prediction
- **Higher is better (↑)**

**Properties:**
- Balanced metric for imbalanced datasets
- Accounts for all four confusion matrix quadrants
- More informative than accuracy for class-imbalanced problems

---

## Statistical Testing

### Method: Wilcoxon Signed-Rank Test

- Non-parametric test (doesn't assume normal distribution)
- Paired test (compares same subjects across models)
- Tests if median difference between paired observations is zero

### Multiple Comparison Correction: Holm-Bonferroni

- Controls family-wise error rate
- Less conservative than Bonferroni
- Significance threshold: **p < 0.05** after correction

### Reference Model: MeshNet-26

All statistical comparisons are made against MeshNet-26 as the reference.

---

## Complete Results Table

| Model | Parameters | DICE (↑) | Sig? | AVD (↓) | Sig? | MCC (↑) | Sig? |
|-------|------------|----------|------|---------|------|---------|------|
| **MeshNet-26** | **147,474** | **0.876 (0.016)** | N/A | **0.245 (0.036)** | N/A | **0.760 (0.030)** | N/A |
| MeshNet-16 | 56,194 | 0.873 (0.007) | | 0.249 (0.033) | | 0.757 (0.013) | |
| U-MAMBA-BOT | 7,351,400 | 0.870 (0.012) | | 0.266 (0.053) | | 0.750 (0.023) | |
| MedNeXt-M | 17,548,963 | 0.868 (0.017) | | 0.288 (0.094) | | 0.745 (0.033) | |
| SegResNet | 1,176,186 | 0.867 (0.005) | | 0.268 (0.039) | * | 0.743 (0.011) | |
| U-MAMBA-ENC | 7,514,280 | 0.865 (0.010) | | 0.309 (0.080) | * | 0.740 (0.022) | |
| MedNeXt-S | 5,201,315 | 0.861 (0.012) | | 0.257 (0.054) | | 0.729 (0.021) | |
| Swin-UNETR | 18,346,844 | 0.859 (0.025) | | 0.425 (0.283) | * | 0.730 (0.048) | |
| MedNeXt-B | 10,526,307 | 0.855 (0.016) | * | 0.260 (0.039) | * | 0.718 (0.030) | * |
| U-KAN | 44,070,082 | 0.851 (0.013) | * | 0.321 (0.027) | * | 0.717 (0.025) | * |
| MeshNet-5 | 5,682 | 0.848 (0.023) | * | 0.280 (0.060) | * | 0.708 (0.042) | * |
| UNETR | 95,763,682 | 0.847 (0.014) | * | 0.397 (0.029) | * | 0.700 (0.027) | * |
| Residual U-Net | 1,979,610 | 0.836 (0.005) | * | 0.562 (0.495) | * | 0.685 (0.010) | * |
| V-Net | 45,597,898 | 0.798 (0.018) | * | 0.387 (0.047) | * | 0.614 (0.033) | * |

**Format:** Mean (Standard Deviation)
**\*** = Statistically significantly different from MeshNet-26 (p < 0.05, Holm-corrected)

---

## Key Findings

### 1. MeshNet Achieves State-of-the-Art

- **MeshNet-26 has the highest DICE score (0.876)**
- Outperforms models with 1000× more parameters
- Only MeshNet-16 is not statistically different from MeshNet-26

### 2. Parameter Count ≠ Performance

| Observation | Evidence |
|-------------|----------|
| Largest model (UNETR, 96M params) | DICE = 0.847 (below average) |
| V-Net (46M params) | DICE = 0.798 (worst) |
| MeshNet-26 (147K params) | DICE = 0.876 (best) |

### 3. Volume Estimation

- MeshNet-26: AVD = 0.245 (24.5% average volume error)
- Best AVD overall (tied with MeshNet-16)
- Some large models (Swin-UNETR, Residual U-Net) have high AVD variance

### 4. Consistency (Standard Deviation)

Most consistent models:
1. **MeshNet-16:** σ = 0.007 DICE
2. SegResNet: σ = 0.005 DICE
3. Residual U-Net: σ = 0.005 DICE

Least consistent:
1. Swin-UNETR: σ = 0.025 DICE
2. MeshNet-5: σ = 0.023 DICE
3. V-Net: σ = 0.018 DICE

---

## Efficiency Metrics

### DICE per Million Parameters

| Model | DICE | Params (M) | DICE/M Params |
|-------|------|------------|---------------|
| MeshNet-5 | 0.848 | 0.006 | 149.30 |
| MeshNet-16 | 0.873 | 0.056 | 15.59 |
| MeshNet-26 | 0.876 | 0.147 | 5.96 |
| SegResNet | 0.867 | 1.18 | 0.73 |
| MedNeXt-S | 0.861 | 5.20 | 0.17 |
| U-MAMBA-BOT | 0.870 | 7.35 | 0.12 |
| MedNeXt-M | 0.868 | 17.55 | 0.05 |
| UNETR | 0.847 | 95.76 | 0.009 |

**MeshNet-5 achieves 149× more DICE per parameter than MeshNet-26, and 16,500× more than UNETR.**

---

## Qualitative Observations (Figure 3)

### Boundary Alignment

| Model | Boundary Quality |
|-------|-----------------|
| MeshNet-16, -26 | Best alignment, fewest issues |
| MeshNet-5 | Variable, struggles with complex shapes |
| MedNeXt-M | Tends to over-segment |
| U-MAMBA-BOT | Inconsistent alignment |

### Common Error Patterns

**Over-segmentation (predicts lesion where none exists):**
- MedNeXt-M frequently over-segments boundaries
- Captures non-lesion areas

**Under-segmentation (misses lesion):**
- MeshNet-16/26 occasionally under-segment irregular regions
- Some lesion extensions missed

**Boundary deviation:**
- U-MAMBA-BOT shows inconsistent alignment
- Deviates from actual lesion boundaries

---

## Benchmark Implementation Guide

### Computing DICE

```python
def dice_coefficient(pred, target):
    """
    Compute DICE coefficient.
    pred, target: binary tensors of shape (B, D, H, W)
    """
    smooth = 1e-6  # Numerical stability

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()
```

### Computing AVD

```python
def average_volume_difference(pred, target):
    """
    Compute Average Volume Difference.
    pred, target: binary tensors
    """
    pred_volume = pred.sum().item()
    target_volume = target.sum().item()

    if target_volume == 0:
        return float('inf') if pred_volume > 0 else 0.0

    avd = abs(pred_volume - target_volume) / target_volume
    return avd
```

### Computing MCC

```python
def matthews_correlation_coefficient(pred, target):
    """
    Compute Matthews Correlation Coefficient.
    pred, target: binary tensors
    """
    pred_flat = pred.view(-1).bool()
    target_flat = target.view(-1).bool()

    tp = (pred_flat & target_flat).sum().float()
    tn = (~pred_flat & ~target_flat).sum().float()
    fp = (pred_flat & ~target_flat).sum().float()
    fn = (~pred_flat & target_flat).sum().float()

    numerator = tp * tn - fp * fn
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if denominator == 0:
        return 0.0

    mcc = numerator / denominator
    return mcc.item()
```

### Statistical Testing

```python
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

def compare_to_reference(reference_scores, model_scores_dict, alpha=0.05):
    """
    Compare all models to reference using Wilcoxon test with Holm correction.
    """
    p_values = []
    model_names = []

    for name, scores in model_scores_dict.items():
        stat, p = wilcoxon(reference_scores, scores)
        p_values.append(p)
        model_names.append(name)

    # Holm-Bonferroni correction
    reject, corrected_p, _, _ = multipletests(p_values, alpha=alpha, method='holm')

    results = {}
    for name, p, sig in zip(model_names, corrected_p, reject):
        results[name] = {'p_value': p, 'significant': sig}

    return results
```

---

## Reproducing the Benchmark

### Required Models

| Model | Source |
|-------|--------|
| MeshNet | Implement from paper / BrainChop |
| SegResNet | MONAI |
| U-Net variants | MONAI |
| MedNeXt | Official repo |
| U-MAMBA | Official repo |
| Swin-UNETR | MONAI |
| UNETR | MONAI |
| V-Net | MONAI |
| U-KAN | Official repo |

### Evaluation Protocol

1. Train each model using nested 3-fold CV
2. For MeshNet: hyperparameter search on inner folds of fold 1
3. For baselines: use fixed hyperparameters
4. Evaluate on hold-out test sets
5. Report mean ± std across outer folds
6. Perform Wilcoxon test with Holm correction

---

## References

- DICE coefficient: Sørensen (1948), Dice (1945)
- AVD: Commonly used in lesion segmentation literature
- MCC: Matthews (1975), Chicco & Jurman (2020)
- Reliability threshold: Liew et al., Scientific Data, 2022
