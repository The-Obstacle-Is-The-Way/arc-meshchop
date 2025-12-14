# BUG-002: Test Metric Aggregation Does Not Match Paper Protocol

**Date**: 2025-12-13
**Status**: CONFIRMED
**Priority**: P0 (Affects paper parity claims)
**Affects**: `src/arc_meshchop/experiment/runner.py`

---

## Summary

The experiment runner computes test metrics as **mean-of-fold-means (n=3)** instead of **pooled per-subject scores (n≈224)** as the paper does. This causes incorrect std calculation and prevents proper Wilcoxon statistical testing.

---

## Paper Evidence

### 1. Figure 1 is a Per-Subject Boxplot

![Figure 1](_literature/markdown/stroke_lesion_segmentation/_page_2_Figure_0.jpeg)

Figure 1 shows a **boxplot** with:
- Median line
- IQR box (25th-75th percentile)
- Whiskers and outliers

**A boxplot requires a distribution.** You cannot create a meaningful boxplot from n=3 values. This is clearly a distribution of ~224 per-subject DICE scores.

### 2. Table 1 Uses Paired Wilcoxon Test

From paper Table 1 caption:
> "(\*) indicate models statistically significantly different from MeshNet-26 (p < 0.05, Holm-corrected Wilcoxon test)"

A Wilcoxon signed-rank test is a **paired** non-parametric test comparing matched observations (same subject, different models). With n=3 (fold means), this test is meaningless. With n≈224 (per-subject paired scores), it works correctly.

### 3. Figure 2 Shows Median with IQR

From paper Figure 2 caption:
> "median DICE score with interquartile range (IQR) error bars"

IQR requires at least a distribution of values to compute 25th/75th percentiles. This only makes sense with ~224 per-subject scores.

### 4. Standard Deviation Magnitudes

Paper Table 1 reports:
| Model | DICE (mean ± std) |
|-------|------------------|
| MeshNet-26 | 0.876 (0.016) |
| MeshNet-16 | 0.873 (0.007) |
| SegResNet | 0.867 (0.005) |

Std values of 0.005-0.02 are consistent with per-subject variation across n≈224 subjects (typical DICE variance). If computed from n=3 fold means, std would be more variable and depend on fold-level differences.

---

## Current Implementation (WRONG)

```python
# runner.py: _evaluate_on_test_sets (lines 478-510)

# STEP 1: Collect per-subject scores (CORRECT)
for idx in range(len(test_dataset)):
    scores = metrics_calculator.compute_single(pred, mask)
    dice_scores.append(scores["dice"])  # Per-subject DICE

# STEP 2: Average to fold-level mean (LOSES INFORMATION)
test_dice = float(np.mean(dice_scores))  # ~75 values → 1 value

# STEP 3: Store only the fold mean
test_results.append({"test_dice": test_dice, ...})
```

```python
# runner.py: ExperimentResult.test_std_dice (lines 96-102)

def test_std_dice(self) -> float:
    test_dices = [t["test_dice"] for t in self.test_results]  # 3 values!
    return float(np.std(test_dices))  # Std over n=3, WRONG
```

### What Current Code Produces

```
Fold 1 test (75 subjects): mean=0.87 → stores 0.87
Fold 2 test (75 subjects): mean=0.88 → stores 0.88
Fold 3 test (74 subjects): mean=0.86 → stores 0.86

test_mean_dice = mean([0.87, 0.88, 0.86]) = 0.87  ← Correct (unbiased estimator)
test_std_dice = std([0.87, 0.88, 0.86]) = 0.01   ← WRONG (std over 3 values)
```

### What Paper Does

```
Fold 1 test: 75 per-subject scores [0.91, 0.83, 0.88, ...]
Fold 2 test: 75 per-subject scores [0.85, 0.90, 0.87, ...]
Fold 3 test: 74 per-subject scores [0.89, 0.82, 0.86, ...]

Pool all: 224 per-subject scores
mean = 0.876
std = 0.016  ← Std over 224 values (matches paper!)
median = for boxplot
IQR = for error bars
Wilcoxon = paired comparison using 224 scores
```

---

## Impact

1. **Incorrect std**: Current implementation computes std over n=3, paper uses n≈224
2. **No Wilcoxon support**: Cannot do paired per-subject comparison without preserving scores
3. **No boxplot data**: Cannot reproduce Figure 1 without per-subject distribution
4. **No IQR data**: Cannot reproduce Figure 2 error bars without per-subject scores

---

## Required Fix

### 1. Store Per-Subject Scores

```python
# In _evaluate_on_test_sets, change:
test_results.append({
    "outer_fold": outer_fold,
    "test_dice": test_dice,  # Keep for backward compat
    "per_subject_dice": dice_scores,  # ADD: raw scores
    "per_subject_avd": avd_scores,    # ADD
    "per_subject_mcc": mcc_scores,    # ADD
    "subject_indices": list(test_indices),  # ADD: for Wilcoxon pairing
    ...
})
```

### 2. Compute Stats from Pooled Scores

```python
@property
def test_std_dice(self) -> float:
    """Std of TEST DICE across all subjects (paper protocol)."""
    all_scores = []
    for t in self.test_results:
        all_scores.extend(t.get("per_subject_dice", [t["test_dice"]]))
    return float(np.std(all_scores))

@property
def test_median_dice(self) -> float:
    """Median TEST DICE (for Figure 2)."""
    all_scores = []
    for t in self.test_results:
        all_scores.extend(t.get("per_subject_dice", []))
    return float(np.median(all_scores)) if all_scores else 0.0

@property
def test_iqr_dice(self) -> tuple[float, float]:
    """IQR of TEST DICE (for Figure 2 error bars)."""
    all_scores = []
    for t in self.test_results:
        all_scores.extend(t.get("per_subject_dice", []))
    if not all_scores:
        return (0.0, 0.0)
    return (float(np.percentile(all_scores, 25)),
            float(np.percentile(all_scores, 75)))
```

### 3. Support Wilcoxon Testing

```python
def get_per_subject_scores(self) -> dict[int, dict[str, float]]:
    """Get per-subject scores indexed by subject ID for Wilcoxon pairing."""
    scores_by_subject = {}
    for t in self.test_results:
        indices = t.get("subject_indices", [])
        dices = t.get("per_subject_dice", [])
        for idx, dice in zip(indices, dices):
            scores_by_subject[idx] = {"dice": dice, ...}
    return scores_by_subject
```

---

## Verification

After fix, verify:
1. `test_std_dice` computed from ~224 values, not 3
2. Can generate boxplot matching Figure 1 shape
3. Can generate scatter with IQR error bars matching Figure 2
4. Can run Wilcoxon test between models using paired per-subject data

---

## References

- Paper: Figure 1 (boxplot), Figure 2 (IQR error bars), Table 1 (Wilcoxon test)
- Code: `src/arc_meshchop/experiment/runner.py:88-134, 401-520`
