# Paper to Implementation Audit

This audit compares the MeshNet paper implementation details to this codebase.
Source paper: `_literature/markdown/stroke_lesion_segmentation/stroke_lesion_segmentation.md`

---

## Matches (Paper Parity)

- **Architecture**: 10-layer MeshNet with dilation pattern 1-2-4-8-16-16-8-4-2-1.
- **Input/Output**: full 256^3 volumes, 2-class output (background + lesion).
- **Dataset filtering**: SPACE 2x and SPACE no-accel; TSE excluded in parity mode.
- **Preprocessing**: resample to 1mm isotropic, 256^3, min-max normalization.
- **Training**: AdamW, OneCycleLR (1% warmup, div_factor=100), label smoothing 0.01.
- **Precision**: FP16 on CUDA, FP32 on MPS/CPU.
- **Evaluation**: DICE, AVD, MCC; pooled per-subject statistics for parity.
- **Protocol**: 3 outer folds and 10 restarts for replication.

---

## Deviations or Assumptions

- **HPO framework**: Optuna + ASHA replaces Orion (paper). Functional parity, not exact tooling.
- **Resampling**: SciPy-based resampling replaces FreeSurfer `mri_convert --conform`.
- **Layer checkpointing**: referenced in paper for large models, not implemented here.
- **Config system**: Hydra configs exist but the CLI is the primary interface.
- **Hyperparameter final values**: code uses the paper defaults but does not persist the
  actual HPO-selected values for MeshNet variants in outputs.

---

## Implementation Notes

- `src/arc_meshchop/models/meshnet.py` implements the architecture.
- `src/arc_meshchop/data/preprocessing.py` implements resampling + normalization.
- `src/arc_meshchop/training/*` implements optimizer, scheduler, and loss.
- `src/arc_meshchop/experiment/runner.py` implements outer folds and restarts.
- `src/arc_meshchop/evaluation/metrics.py` implements DICE, AVD, MCC.

---

## Open Questions

- Are the final HPO-selected MeshNet hyperparameters identical to the defaults
  used in `ExperimentConfig`? If not, we should persist and surface them.
- Should the code record dataset version/hash for full parity audits?
- Do any ARC samples have missing or unknown acquisition metadata that could
  affect stratification?

---

## Risks

- Small numerical differences from SciPy resampling may lead to slight metric drift.
- Non-deterministic seeding can obscure parity when comparing runs.

---

## Suggested Next Steps

- Implement deterministic seeding (see BUG-012 and Spec 10).
- Improve HPO reporting for fold-based pruning (see BUG-011 and Spec 11).
- Add a run manifest and IO registry in code (see BUG-014 and Spec 09).
