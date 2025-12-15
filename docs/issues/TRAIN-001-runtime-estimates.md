# TRAIN-001: Training Runtime Estimates & Hardware Options

> **Status:** OPEN - Awaiting Senior Approval
>
> **Date:** 2025-12-15
>
> **Impact:** Full paper replication takes ~77 days on MPS, ~7-15 days on RTX 4090 (est.)

---

## Summary

We successfully started MeshNet-26 training on the ARC dataset (223 SPACE samples).
The paper states **3 outer folds** and **10 restarts** for MeshNet; our implementation
interprets this as **10 restarts per outer fold** → 30 runs total.
but runtime on Apple Silicon is prohibitive for full replication.

**Key Finding:** You only need **1 run** to get a working model with usable weights.
The extra runs are for stability and paper-style reporting; they are not ensembling.

---

## Current Training Status

```
Platform: MacBook Pro (Apple Silicon MPS)
Run: 1 of 30
Epoch: 1 of 50
Step rate: ~30 seconds/step
```

---

## Runtime Breakdown

### Per-Run Math

| Component | Value |
|-----------|-------|
| Steps per epoch | 148 (1 per sample, batch=1) |
| Time per step (MPS) | ~30 seconds |
| Time per epoch | 148 × 30s = **74 minutes** |
| Epochs per run | 50 |
| Time per run (MPS) | 50 × 74 min = **~62 hours (~2.5 days)** |

Notes:
- This math is for **training steps only**. Per-run overhead (checkpoint save + test-set evaluation)
  exists but is small relative to multi-day training on MPS.
- `steps/epoch` comes from the **outer-train** split size (≈2/3 of 223), not the full cohort.

### Full Replication (30 Runs)

| Hardware | Time per Step | Time per Run | Total (30 runs) |
|----------|---------------|--------------|-----------------|
| **MPS (M-series Mac)** | ~30s | ~62 hours | **~77 days** |
| **RTX 4090** | ~3-6s (est.) | ~6-12 hours | **~7-15 days** |
| **A100 80GB** | ~2-4s (est.) | ~4-8 hours | **~5-10 days** |

### Estimation Basis

RTX 4090 estimates based on:
- Paper trained on **A100 80GB** (benchmark reference)
- CUDA kernels + mixed precision (FP16) are typically much faster than MPS FP32 for 3D conv
- For memory-heavy 3D conv, throughput often tracks memory bandwidth (4090 ≈1 TB/s vs A100 ≈2 TB/s)

Important caveats:
- **Speedup is workload-dependent.** Small batch size (1), CPU preprocessing, and disk I/O can limit GPU scaling.
- **Validate with a quick benchmark** on the 4090: run 20-50 steps and measure avg seconds/step.

Sources:
- [BIZON GPU Benchmarks](https://bizon-tech.com/gpu-benchmarks/NVIDIA-A100-80-GB-(PCIe)-vs-NVIDIA-RTX-4090/624vs637)
- [NVIDIA Forums: A100 vs 4090](https://forums.developer.nvidia.com/t/difference-between-a100-vs-rtx-4090-in-training-deep-learning-models/315232)
- [Vast.ai Comparison](https://vast.ai/article/nvidia-rtx-4090-vs-a100-two-powerhouses-two-purposes)

---

## Options

### Option 1: Single Run on MPS (Practical Model)

**Timeline:** ~2.5 days

**What you get:**
- 1 fully trained MeshNet-26 model
- Checkpoint weights (`final.pt`)
- DICE score on held-out test fold

**What you lose:**
- Statistical variance estimate (std dev)
- Cannot claim "paper replication" (need 30 runs for that)

**Where outputs land (default):**
- `experiments/fold_0_restart_0/final.pt`
- `experiments/fold_0_restart_0/results.json`

**Action:** Let current run complete, then stop the experiment (e.g., `tmux kill-session -t arc-train`).

### Option 2: Full Replication on RTX 4090

**Timeline:** ~7-15 days

**What you get:**
- Full paper replication
- Mean ± std across 30 runs
- Direct comparison to paper's 0.876 (0.016) DICE

**Requirements:**
- Transfer code to NVIDIA box
- Ensure CUDA + PyTorch setup
- Run: `uv run arc-meshchop experiment --data-dir data/arc`

**Note:** No code changes needed. The codebase auto-detects CUDA and uses FP16 on CUDA.

### Option 3: Reduced Restarts (Compromise)

**Timeline:** ~15-25 days on MPS, ~2-5 days on 4090

Run fewer restarts:
```bash
uv run arc-meshchop experiment --restarts 3
```

This gives 3 folds × 3 restarts = 9 runs (vs 30).

**Tradeoff:** Less statistical power, but still defensible for a replication.

---

## Practical Recommendations

### For "I just want a working model"

Let **Option 1** complete on MPS. After ~2.5 days you'll have:
```
experiments/fold_0_restart_0/final.pt
```

This is a fully trained MeshNet-26 that can segment stroke lesions.

### For "I want full paper replication"

Transfer to RTX 4090 (**Option 2**). Steps:

1. Clone repo on NVIDIA box
2. `uv sync --all-extras`
3. Run download: `uv run arc-meshchop download --paper-parity`
4. Run experiment: `uv run arc-meshchop experiment --data-dir data/arc`
5. ~7-15 days later, collect results

### For "I want to validate the pipeline first"

Run 1 epoch on MPS to confirm no crashes:
```bash
uv run arc-meshchop experiment --restarts 1 --epochs 1

# Or wait for Epoch 1 to complete (~74 min on MPS)
# Then check loss curve looks reasonable
```

---

## Why 10 Restarts? (Paper Context)

From the paper (Section 2, Training):

> "we trained the model with 10 restarts"

Also from the paper (Section 2, Dataset/Training):

> "We implemented a nested cross-validation approach with three outer folds..."

**Purpose:**
1. Neural networks are stochastic (random init, batch order)
2. 10 runs gives statistical confidence in reported metrics
3. Standard ML practice for small datasets (223 samples)

**The 30 runs breakdown:**
- 3 outer folds (different train/test splits)
- 10 restarts per fold (different random seeds)
- Reporting: paper reports aggregate test statistics; our implementation tracks both
  per-subject pooled stats (paper Table 1 style) and per-run stats (n=30)

**You don't ensemble them.** Each is independent. 1 model = 1 run.

---

## Decision Needed

- [ ] **Option 1:** Accept ~2.5 day MPS run for single model (practical)
- [ ] **Option 2:** Transfer to RTX 4090 for full replication (~7-15 days)
- [ ] **Option 3:** Reduce restarts to 3 for compromise timeline
- [ ] **Other:** Cloud GPU (Lambda, Vast.ai, etc.)

---

## Current Status

Training is running in tmux (`arc-train`):
```
tmux attach -t arc-train
```

To kill after Run 1 completes:
```bash
# Wait for "Run 1/30 complete" in logs, then:
tmux kill-session -t arc-train
```

---

## Related

- Paper: "State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters"
- Config: `src/arc_meshchop/experiment/config.py`
- Runner: `src/arc_meshchop/experiment/runner.py`
