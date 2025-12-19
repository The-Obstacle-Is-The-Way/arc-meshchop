# BUG-012: Incomplete RNG Seeding Causes Non-Deterministic Runs (P2)

**Priority:** P2
**Status:** Fixed
**Impacts:** `arc-meshchop train`, `arc-meshchop experiment`, `arc-meshchop hpo`

---

## Symptom

- Runs with the same seed can produce different metrics.
- HPO results are not stable between runs with the same configuration.
- Paper parity checks are harder to reproduce exactly.

---

## Root Cause

Seeding is partial:

- CLI commands seed `torch.manual_seed` (and `torch.cuda.manual_seed_all`),
  but do not seed `random` or `numpy`.
- DataLoader workers are not seeded, so shuffle order can differ.
- Deterministic flags (e.g., `torch.backends.cudnn.deterministic`) are not set.

This leaves multiple sources of stochasticity unpinned.

---

## Fix Implementation

1. Introduced `src/arc_meshchop/utils/seeding.py` with `seed_everything` and `worker_init_fn`.
   - Seeds `random`, `numpy`, `torch`, `torch.cuda`.
   - Sets `PYTHONHASHSEED`.
2. Updated `train` CLI command and `run_experiment` to use `seed_everything`.
3. Updated `create_dataloaders` to accept a `seed` and use `worker_init_fn` + `torch.Generator`.
4. Updated `run_hpo_trial` to accept a fixed seed (default 42) for reproducible splitting and initialization.
5. Seeded the Optuna sampler in `create_study` to make HPO trial ordering reproducible.

---

## Verification

- Added unit tests for deterministic seeding in DataLoaders.
- Verified that HPO trials are reproducible (via fixed seed and sampler seeding).
