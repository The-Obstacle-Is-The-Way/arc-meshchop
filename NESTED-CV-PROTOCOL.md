# Nested CV Protocol: Definitive Answer

**Date**: 2025-12-13
**Status**: CONFIRMED
**Impact**: Major implementation change required

---

## TL;DR

The paper **retrains on full outer-training data** after hyperparameter selection. Our current implementation incorrectly evaluates inner-fold checkpoints directly.

| Approach | Training Data | % of Total | Total Runs |
|----------|--------------|------------|------------|
| **Our implementation (WRONG)** | Inner-train (2/3 × 2/3) | 44% | 90 |
| **Paper protocol (CORRECT)** | Full outer-train (2/3) | 67% | 30 |

---

## Paper Evidence

### Direct Quote

From Section 2 (Training):

> "Hyperparameter optimization was conducted on the inner folds of **the first outer fold**. The optimized hyperparameters were then applied to **train models on all outer folds**."

This is unambiguous:
1. HP search happens **only on Outer Fold 1**
2. After finding HPs, they "**train models**" (new training) on "**all outer folds**"

### Logical Proof

| Step | Paper Statement | Implication |
|------|-----------------|-------------|
| 1 | HP search on "first outer fold" only | Outer Folds 2 & 3 have no inner-loop runs |
| 2 | No inner loops for Folds 2 & 3 | No checkpoints exist to select from |
| 3 | No checkpoints | Must train NEW models for Folds 2 & 3 |
| 4 | "Train models on all outer folds" | Retrain for Fold 1 too (consistency) |
| 5 | Fixed epochs (50) from HP search | No validation-based early stopping needed |
| **Conclusion** | | **Train on FULL outer-train for each fold** |

### Supporting Evidence

1. **Fixed Epochs (50)**: The paper says epochs were part of HP search space (15, 50). By selecting 50, they established a fixed stopping criterion, eliminating the need for validation-based early stopping. This enables training on 100% of outer-train data.

2. **Varma & Simon (2006)**: The standard nested CV protocol estimates performance of the *algorithm* (Data → Find HPs → Train Final Model), not of a specific checkpoint. The final model should be trained on maximum available data.

3. **10 Restarts**: The paper says "we trained the model with 10 restarts." These restarts are for the final evaluation phase, not for HP search.

---

## Correct Protocol

### Phase 1: Hyperparameter Search (Outer Fold 1 Only)

**Paper did this with Orion/ASHA. We skip this and use their final HPs.**

```
Outer Fold 1 Training Data:
├── Inner Fold 0: train/val split → test HP config A
├── Inner Fold 1: train/val split → test HP config B
└── Inner Fold 2: train/val split → test HP config C

Best HP configuration: H* = {lr=0.001, epochs=50, ...}
```

### Phase 2: Final Evaluation (All Outer Folds)

```
for outer_fold in [0, 1, 2]:
    outer_train = all_data - test_fold[outer_fold]  # 67% of data
    outer_test = test_fold[outer_fold]               # 33% of data

    for restart in range(10):
        model = MeshNet(channels=26)
        model.train(outer_train, epochs=50, **H*)  # Fixed epochs, no validation
        scores[restart] = model.evaluate(outer_test)

    fold_scores[outer_fold] = aggregate(scores)  # Mean or ensemble

final_score = pool_all_per_subject_scores(fold_scores)
```

### Run Count

```
Phase 1 (HP Search): Skip - using paper's final HPs
Phase 2 (Final Eval): 3 outer folds × 10 restarts = 30 runs

Total: 30 runs (not 90!)
```

---

## What Changes

### runner.py

**Before (WRONG):**
```python
for outer_fold in range(3):
    for inner_fold in range(3):      # ← REMOVE THIS LOOP
        for restart in range(10):
            train_on_inner_fold()    # ← 44% of data

# Evaluation: Use inner-fold checkpoint  # ← WRONG
```

**After (CORRECT):**
```python
for outer_fold in range(3):
    for restart in range(10):
        train_on_full_outer_train()  # ← 67% of data
        evaluate_on_outer_test()

# Evaluation: Use the just-trained model  # ← CORRECT
```

### config.py

```python
@property
def total_runs(self) -> int:
    # OLD: return num_outer_folds * num_inner_folds * num_restarts  # 90
    return self.num_outer_folds * self.num_restarts  # 30
```

### Training Data Flow

```
Before (WRONG):
Total Data (224 subjects)
└── Outer Fold 1 (149 train / 75 test)
    └── Inner Fold 1 (99 train / 50 val)  ← Training on 99 subjects (44%)

After (CORRECT):
Total Data (224 subjects)
└── Outer Fold 1 (149 train / 75 test)    ← Training on 149 subjects (67%)
```

---

## Restart Aggregation (Minor Ambiguity)

The paper doesn't explicitly state how 10 restarts aggregate. Options:

1. **Average metrics** across 10 restarts → Report mean DICE per subject
2. **Ensemble predictions** from 10 models → Compute DICE on ensembled logits
3. **Select best** restart → Report best DICE

Given paper reports "mean ± std," likely option 1 (average metrics). But this is implementation detail, not protocol.

---

## Validation

After implementing the correct protocol, verify:

1. **Run count**: 30 total runs (not 90)
2. **Training data**: Each run trains on ~149 subjects (67% of 224)
3. **No validation split**: Full outer-train used (no inner fold held out)
4. **Fixed epochs**: Train for exactly 50 epochs per run
5. **DICE target**: Should achieve ~0.876 (paper's MeshNet-26 score)

---

## References

- Paper Section 2: "Hyperparameter optimization was conducted on the inner folds of the first outer fold. The optimized hyperparameters were then applied to train models on all outer folds."
- Varma & Simon (2006): "Bias in error estimation when using cross-validation for model selection"
- Our analysis: BUG-002-metric-aggregation.md (per-subject pooling)
