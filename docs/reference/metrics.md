# Evaluation Metrics & Benchmarks

> Reference: "State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters"

## Metrics Used

The paper evaluates models using three complementary metrics:

### 1. DICE Coefficient (Sorensen-Dice Index)

**Formula:**
```
DICE = 2 * |A intersection B| / (|A| + |B|)
```

Where:
- A = predicted lesion voxels
- B = ground truth lesion voxels
- |A intersection B| = number of overlapping voxels

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
- Range: [0, infinity)
- 0 = perfect volume match
- 0.25 = 25% volume difference
- **Lower is better**

**Properties:**
- Measures volumetric accuracy
- Important for clinical applications (lesion volume correlates with outcomes)
- Does not penalize shape errors (only total volume)

---

### 3. Matthews Correlation Coefficient (MCC)

**Formula:**
```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
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
- **Higher is better**

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

| Model | Parameters | DICE | AVD | MCC |
|-------|------------|------|-----|-----|
| **MeshNet-26** | **147,474** | **0.876 (0.016)** | **0.245 (0.036)** | **0.760 (0.030)** |
| MeshNet-16 | 56,194 | 0.873 (0.007) | 0.249 (0.033) | 0.757 (0.013) |
| U-MAMBA-BOT | 7,351,400 | 0.870 (0.012) | 0.266 (0.053) | 0.750 (0.023) |
| MedNeXt-M | 17,548,963 | 0.868 (0.017) | 0.288 (0.094) | 0.745 (0.033) |
| SegResNet | 1,176,186 | 0.867 (0.005) | 0.268 (0.039)* | 0.743 (0.011) |
| U-MAMBA-ENC | 7,514,280 | 0.865 (0.010) | 0.309 (0.080)* | 0.740 (0.022) |
| MedNeXt-S | 5,201,315 | 0.861 (0.012) | 0.257 (0.054) | 0.729 (0.021) |
| Swin-UNETR | 18,346,844 | 0.859 (0.025) | 0.425 (0.283)* | 0.730 (0.048) |
| MedNeXt-B | 10,526,307 | 0.855 (0.016)* | 0.260 (0.039)* | 0.718 (0.030)* |
| U-KAN | 44,070,082 | 0.851 (0.013)* | 0.321 (0.027)* | 0.717 (0.025)* |
| MeshNet-5 | 5,682 | 0.848 (0.023)* | 0.280 (0.060)* | 0.708 (0.042)* |
| UNETR | 95,763,682 | 0.847 (0.014)* | 0.397 (0.029)* | 0.700 (0.027)* |
| Residual U-Net | 1,979,610 | 0.836 (0.005)* | 0.562 (0.495)* | 0.685 (0.010)* |
| V-Net | 45,597,898 | 0.798 (0.018)* | 0.387 (0.047)* | 0.614 (0.033)* |

**Format:** Mean (Standard Deviation)
**\*** = Statistically significantly different from MeshNet-26 (p < 0.05, Holm-corrected)

---

## Key Findings

### 1. MeshNet Achieves State-of-the-Art

- **MeshNet-26 has the highest DICE score (0.876)**
- Outperforms models with 1000x more parameters
- Only MeshNet-16 is not statistically different from MeshNet-26

### 2. Parameter Count != Performance

| Observation | Evidence |
|-------------|----------|
| Largest model (UNETR, 96M params) | DICE = 0.847 (below average) |
| V-Net (46M params) | DICE = 0.798 (worst) |
| MeshNet-26 (147K params) | DICE = 0.876 (best) |

### 3. Consistency (Standard Deviation)

Most consistent models:
1. **MeshNet-16:** std = 0.007 DICE
2. SegResNet: std = 0.005 DICE
3. Residual U-Net: std = 0.005 DICE

---

## Metric Implementations

### Computing DICE

```python
def dice_coefficient(pred, target, smooth=1e-6):
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
    pred_volume = pred.sum().item()
    target_volume = target.sum().item()
    if target_volume == 0:
        return float('inf') if pred_volume > 0 else 0.0
    return abs(pred_volume - target_volume) / target_volume
```

### Computing MCC

```python
def matthews_correlation_coefficient(pred, target):
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
    return (numerator / denominator).item()
```

---

## References

- DICE coefficient: Sorensen (1948), Dice (1945)
- AVD: Commonly used in lesion segmentation literature
- MCC: Matthews (1975), Chicco & Jurman (2020)
- Reliability threshold: Liew et al., Scientific Data, 2022
