# TODO.md — Remaining Work Tracker

> **Purpose:** Single source of truth for all remaining work items.
>
> **Status:** Ready for senior review
>
> **Last Updated:** 2025-12-14

---

## Executive Summary

The codebase is **ready to train**. All blocking issues have been resolved. The remaining work falls into two categories:

1. **Blocked on Upstream** — Requires HuggingFace dataset update (external dependency)
2. **Low Priority Enhancements** — Nice-to-have improvements that don't block training

---

## Category A: Blocked on Upstream (Paper-Exact Parity)

**Blocker:** HuggingFace dataset `hugging-science/arc-aphasia-bids` is missing the `t2w_acquisition` column.

Until the upstream dataset is updated, we cannot:
- Filter out TSE sequences (5 samples)
- Achieve exact paper sample count (223 SPACE-only samples)
- Provide `--paper-parity` mode for strict replication

### Tasks (Blocked)

| Task | File | Description | Status |
|------|------|-------------|--------|
| Add `t2w_acquisition` column | Upstream | HuggingFace dataset needs new column | ⏳ BLOCKED |
| Update loader to use column | `huggingface_loader.py` | Read acquisition type from dataset | ⏳ BLOCKED |
| Remove workaround code | `huggingface_loader.py:284-288` | Remove default-to-space_no_accel hack | ⏳ BLOCKED |
| Add `--paper-parity` flag | `cli.py` | Strict 223-sample mode | ⏳ BLOCKED |
| Update `verify_sample_counts()` | `huggingface_loader.py` | Exact match instead of range | ⏳ BLOCKED |
| Add acquisition count tests | `tests/` | Verify 115/108/5 distribution | ⏳ BLOCKED |
| Document TSE exclusions | Upstream | Update HuggingFace dataset card | ⏳ BLOCKED |

### Current Workaround

Training proceeds with 228 samples (includes 5 TSE). This is clinically valid but not exact paper replication. See `docs/KNOWN_ISSUES.md` for full analysis.

### Expected Impact

- Current: 228 samples → Estimated DICE ~0.876 (within paper variance)
- After fix: 223 samples → Exact paper parity

---

## Category B: Data Contract Hardening ✅ COMPLETE

**SPEC:** `docs/specs/local-06-data-contract-hardening.md`
**Commit:** `d940415` (2025-12-14)

All data contract hardening has been implemented and verified (288 tests pass).

### Implementation Checklist

| # | Task | File | Status |
|---|------|------|--------|
| 1 | Fix `mask_paths` to be aligned `list[Path \| None]` | `huggingface_loader.py:75` | ✅ DONE |
| 2 | Add `samples_with_masks` and `num_with_masks` properties | `huggingface_loader.py:84-95` | ✅ DONE |
| 3 | Add `parse_dataset_info()` and `validate_masks_present()` | `huggingface_loader.py:126-225` | ✅ DONE |
| 4 | Add validation to runner.py | `runner.py:455-461` | ✅ DONE |
| 5 | Add validation to hpo.py | `hpo.py` | ✅ DONE |
| 6 | Add `cache_dir` to test dataset | `runner.py:523` | ✅ DONE |
| 7 | Add `subject_ids` to `RunResult` | `runner.py:62` | ✅ DONE |
| 8 | Add alignment tests | `test_huggingface_loader.py:113` | ✅ DONE |
| 9 | Add validation tests | `test_huggingface_loader.py:276-299` | ✅ DONE |

### Benefits Achieved

1. **Consistent API Contract**: All list properties now have identical indexing semantics
2. **Fail-Fast Validation**: Actionable error messages instead of TypeError
3. **Performance**: Test cache saves ~2.5-5 hours in full experiment
4. **Stable Pairing**: Subject IDs enable reliable cross-experiment Wilcoxon tests

---

## Category C: Deferred / Post-Training (P3)

### C1. Optional Validation Tracking

**Status:** DEFERRED
**File:** `src/arc_meshchop/experiment/runner.py`

Add optional flag to track validation metrics during fixed-epoch training. Not required for paper replication.

---

### C2. Sensitivity Study

**Status:** TODO
**Impact:** Documentation only

**Task:**
Run training with 228 vs 223 samples and document the DICE difference. Expected to be within paper variance (±0.016).

---

## Category D: Completed Work (Reference)

All of the following have been implemented and verified:

### Bug Fixes (BUG-001)
- ✅ P0 - Mask binarization (`> 0.5` → `> 0`)
- ✅ P1 - RAS+ canonical orientation (`nib.as_closest_canonical`)
- ✅ P1 - Resume functionality (scheduler state, epoch counter)
- ✅ P2 - Checkpoint security (`weights_only=True`)
- ✅ P3 - MPS autocast conditional

### Metric Aggregation (BUG-002)
- ✅ Per-subject scores stored in RunResult
- ✅ Pooled score methods (get_all_per_subject_scores)
- ✅ Wilcoxon pairing support (get_per_subject_scores)

### Runner Issues (BUG-003)
- ✅ Restart aggregation modes (mean/median/best)
- ✅ Cache shared per outer fold (not per restart)
- ✅ Runner no longer overwrites Trainer checkpoints

### Protocol Fix (NESTED-CV-PROTOCOL)
- ✅ Changed from 90 runs to 30 runs
- ✅ Removed inner fold loops from final evaluation
- ✅ Train on full outer-train (67% of data)

---

## Verification Checklist

Before proceeding with training, verify:

```bash
# Run all tests
make ci

# Expected: 284 passed, 2 skipped

# Check sample count
uv run arc-meshchop info
# Expected: 228 samples loaded

# Verify protocol
# Should show: 3 outer folds × 10 restarts = 30 runs
```

---

## Implementation Order

**Phase 1: Data Contract Hardening** (Before Training)

Implement all 9 items in Category B per `docs/specs/local-06-data-contract-hardening.md`.

This is not optional. Professional codebases have consistent API contracts.

**Phase 2: Upstream Dataset Fix** (In Progress)

HuggingFace dataset metadata update is in progress. When complete:
- Update loader to use `t2w_acquisition` column
- Enable `--paper-parity` mode for 223 SPACE-only samples

**Phase 3: Training**

Once Phase 1 is complete and CI passes:
```bash
uv run arc-meshchop experiment --outer-folds 3 --restarts 10
```

**Phase 4: Post-Training Analysis**

- Sensitivity study (228 vs 223 samples)
- Paper comparison and documentation

---

## References

- `docs/specs/local-06-data-contract-hardening.md` — **THE SPEC** (implement this)
- `docs/bugs/BUG-001-audit-findings.md` — Detailed audit findings
- `docs/bugs/BUG-002-metric-aggregation.md` — Metric aggregation fix
- `docs/bugs/BUG-003-runner-issues.md` — Runner implementation fixes
- `docs/bugs/BUG-004-mask-paths-alignment.md` — Problem analysis (superseded by spec)
- `docs/bugs/NESTED-CV-PROTOCOL.md` — Protocol analysis
- `docs/BLOCKERS.md` — Original blockers (all resolved)
- `docs/KNOWN_ISSUES.md` — Dataset discrepancy analysis
- `docs/specs/local-05-upstream-dataset-fixes.md` — Upstream fix spec

---

## Last Updated

2025-12-14
