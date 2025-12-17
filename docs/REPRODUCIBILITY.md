# Reproducibility Guide

Exact protocol for replicating the MeshNet paper results.

---

## Target Results

| Metric | Target Value | Tolerance |
|--------|--------------|-----------|
| DICE | 0.876 | +/- 0.016 |
| AVD | 0.245 | +/- 0.036 |
| MCC | 0.760 | +/- 0.030 |

---

## Full Protocol: 30 Runs

The paper uses 3 outer folds x 10 restarts = 30 total training runs.

### Experiment Structure

```
Outer Fold 0 (67% train, 33% test)
├── Restart 0 (seed=42)
├── Restart 1 (seed=43)
├── ...
└── Restart 9 (seed=51)

Outer Fold 1 (67% train, 33% test)
├── Restart 0 (seed=42)
└── ...

Outer Fold 2 (67% train, 33% test)
├── Restart 0 (seed=42)
└── ...
```

### Running the Full Experiment

```bash
# Download data
uv run arc-meshchop download --output data/arc

# Run all 30 training runs
uv run arc-meshchop experiment \
    --data-dir data/arc \
    --output experiments/meshnet26 \
    --variant meshnet_26 \
    --restarts 10 \
    --epochs 50
```

### Output Structure

```
experiments/meshnet26/
├── experiment_results.json      # Aggregated results
├── fold_0_restart_0/
│   ├── results.json             # Per-run results
│   └── final.pt                 # Checkpoint
├── fold_0_restart_1/
│   └── ...
└── fold_2_restart_9/
    └── ...
```

---

## Aggregation Method

### Per-Fold Aggregation

For each outer fold, the paper:
1. Runs 10 restarts with different seeds
2. Evaluates each restart on the test set
3. Reports mean +/- std across restarts

### Final Aggregation

The final reported metrics are:
1. Pool all per-subject scores from all outer folds
2. Compute mean and std across all test subjects (223 total)

This is different from computing mean-of-means across folds.

---

## Seed Handling

### Seed Assignment

```python
# Base seeds for each restart
base_seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# Same seeds used across all outer folds
for outer_fold in range(3):
    for restart, seed in enumerate(base_seeds):
        train(outer_fold=outer_fold, seed=seed)
```

### What Seeds Control

- Model weight initialization
- Data shuffling order
- Dropout (if used)

---

## Hardware Requirements

See [TRAIN-001](issues/TRAIN-001-runtime-estimates.md) for detailed timing.

### Estimated Runtime

| Hardware | Per Run | Full 30 Runs |
|----------|---------|--------------|
| NVIDIA A100 (paper) | ~1-2 hours | ~30-60 hours |
| NVIDIA RTX 4090 | ~6-12 hours | ~7-15 days |
| Apple M-series (MPS) | ~62 hours | ~77 days |
| CPU only | Not recommended | - |

### Memory Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 4 GB | 8+ GB |
| System RAM | 32 GB | 64 GB |
| Storage | 50 GB | 100 GB |

---

## Verification Checklist

### Before Training

- [ ] Dataset downloaded: `data/arc/dataset_info.json` exists
- [ ] Sample count: 223 samples (115 SPACE-2x + 108 SPACE)
- [ ] TSE excluded: No turbo-spin-echo samples
- [ ] Preprocessing: 256^3 @ 1mm isotropic

### During Training

- [ ] Loss decreasing
- [ ] No NaN/Inf values
- [ ] Checkpoints saving

### After Training

- [ ] 30 `results.json` files
- [ ] Per-subject DICE scores recorded
- [ ] Aggregated results in `experiment_results.json`

### Result Validation

- [ ] Mean DICE within paper range (0.876 +/- 0.016)
- [ ] Mean AVD within paper range (0.245 +/- 0.036)
- [ ] Mean MCC within paper range (0.760 +/- 0.030)

---

## Quick Validation Run

For testing the pipeline without full training:

```bash
# Single fold, single restart, few epochs
uv run arc-meshchop train \
    --data-dir data/arc \
    --output outputs/quick-test \
    --outer-fold 0 \
    --inner-fold 0 \
    --epochs 5

# Evaluate
uv run arc-meshchop evaluate \
    outputs/quick-test/fold_0_0/final.pt \
    --data-dir data/arc \
    --outer-fold 0
```

---

## Troubleshooting

### Results Don't Match Paper

1. **Check sample count**: Must be exactly 223
2. **Check preprocessing**: 256^3 @ 1mm isotropic
3. **Check hyperparameters**: lr=0.001, weight_decay=3e-5
4. **Check loss weights**: [0.5, 1.0] for [background, lesion]
5. **Run more restarts**: Variance is expected

### Training Instabilities

1. **NaN loss**: Reduce learning rate, check for bad samples
2. **Low DICE**: Ensure class weights are correct
3. **High variance**: Normal with 10 restarts

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more issues.

---

## Citation

When using this implementation:

```bibtex
@article{fedorov2024meshnet,
  title={State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters},
  author={Fedorov, Alexander and others},
  year={2024}
}
```
