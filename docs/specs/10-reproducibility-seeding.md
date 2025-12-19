# Spec 10: Reproducibility Seeding

**Status:** Implemented

## Summary

Implement a single, consistent seeding strategy across CLI commands and
DataLoader workers to reduce run-to-run variance.

---

## Goals

- Deterministic behavior when a seed is provided (subject to backend nondeterminism).
- Reproducible CV splits and DataLoader shuffling.
- Seed metadata captured in run outputs.

---

## Non-Goals

- Full bitwise determinism across all hardware backends.
- Changing model or training logic.

---

## Implementation

### 1. Global Seeding Utility

Added `arc_meshchop.utils.seeding.seed_everything(seed, deterministic=False)`:

- Seeds `random`, `numpy`, `torch`, `torch.cuda`.
- Sets `PYTHONHASHSEED`.
- Optionally toggles cuDNN deterministic/benchmark flags.

### 2. DataLoader Seeding

- `worker_init_fn` seeds numpy/random per worker based on `torch.initial_seed`.
- `get_generator` returns a seeded `torch.Generator`.
- `create_dataloaders(..., seed=...)` wires the generator and `worker_init_fn`.

### 3. CLI and HPO Integration

- `train`, `experiment`, and `hpo` call `seed_everything`.
- `run_hpo_trial` seeds splits and loaders using the fixed HPO seed.
- `create_study` seeds the Optuna sampler when a seed is provided.
- Seeds are persisted in outputs (`results.json`, `best_params.json`, experiment run results).

---

## Verification

- Added unit tests for deterministic DataLoader ordering and RNG seeding.
- Verified HPO reproducibility via fixed trial seed + sampler seed.
