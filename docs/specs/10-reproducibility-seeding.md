# Spec 10: Reproducibility Seeding

## Summary

Implement a single, consistent seeding strategy across CLI commands and
DataLoader workers to reduce run-to-run variance.

---

## Goals

- Deterministic behavior when a seed is provided.
- Reproducible CV splits and DataLoader shuffling.
- Seed metadata captured in run outputs.

---

## Non-Goals

- Full bitwise determinism across all hardware backends.
- Changing model or training logic.

---

## Proposed Design

### 1. Global Seeding Utility

Add `arc_meshchop.utils.seed.set_global_seed(seed, deterministic)`:

- `random.seed`, `numpy.random.seed`
- `torch.manual_seed`, `torch.cuda.manual_seed_all`
- `PYTHONHASHSEED`
- Optional `torch.backends.cudnn.deterministic/benchmark` toggles

### 2. DataLoader Seeding

- Add `worker_init_fn` that seeds numpy/random per worker.
- Use a `torch.Generator` with the base seed for DataLoader shuffling.

### 3. CLI Integration

- Call `set_global_seed` at the start of `train`, `experiment`, and `hpo`.
- Record the seed and deterministic flag in `results.json` or `run.json`.

---

## Verification

- Run the same configuration twice and compare:
  - CV split indices
  - Metrics within a tight tolerance
- Add unit tests that verify deterministic DataLoader ordering.
