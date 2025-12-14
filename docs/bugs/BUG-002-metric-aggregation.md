# BUG-002: Test Metric Aggregation Does Not Match Paper Protocol

**Date**: 2025-12-13
**Status**: ✅ FIXED
**Last Updated**: 2025-12-14
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

## Fix Applied (2025-12-14)

### 1. Per-Subject Scores Now Stored in RunResult

```python
# runner.py:58-60 - RunResult dataclass now includes per-subject scores
@dataclass
class RunResult:
    ...
    per_subject_dice: list[float]
    per_subject_avd: list[float]
    per_subject_mcc: list[float]
    test_indices: list[int]

# runner.py:542-544 - Scores are populated during evaluation
RunResult(
    per_subject_dice=dice_scores,
    per_subject_avd=avd_scores,
    per_subject_mcc=mcc_scores,
    ...
)
```

### 2. Pooled Score Methods Implemented

```python
# runner.py:138-146 - get_all_per_subject_scores()
def get_all_per_subject_scores(self) -> dict[int, dict[str, float]]:
    """Get per-subject scores from best run per fold."""
    ...

# runner.py:283 - get_per_subject_scores() for Wilcoxon pairing
def get_per_subject_scores(self) -> dict[int, dict[str, float]]:
    ...
```

### 3. Restart Aggregation Mode Added

```python
# runner.py:189,291 - Configurable aggregation
mode = self.config.get("restart_aggregation", "mean")
```

---

## Verification

✅ `per_subject_dice`, `per_subject_avd`, `per_subject_mcc` stored in RunResult
✅ `get_all_per_subject_scores()` returns pooled scores for std/IQR computation
✅ `get_per_subject_scores()` returns indexed scores for Wilcoxon pairing
✅ `restart_aggregation` config supports "mean", "median", "best" modes

---

## References

- Paper: Figure 1 (boxplot), Figure 2 (IQR error bars), Table 1 (Wilcoxon test)
- Code: `src/arc_meshchop/experiment/runner.py:58-60, 138-146, 283, 542-544`

---

## Last Updated

2025-12-14
