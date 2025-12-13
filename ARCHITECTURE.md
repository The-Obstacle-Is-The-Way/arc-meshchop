# Architecture: ARC-MeshChop Data Pipeline

> **Purpose:** Document the end-to-end data flow from HuggingFace Hub to training.

---

## System Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ARC-MESHCHOP ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────┐
                    │      HuggingFace Hub            │
                    │  hugging-science/arc-aphasia-   │
                    │         bids                    │
                    │  (230 subjects, 902 sessions)   │
                    └───────────────┬─────────────────┘
                                    │
                                    │ datasets.load_dataset()
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        huggingface_loader.py                              │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  bids_hub.validation.arc.ARC_VALIDATION_CONFIG                      │  │
│  │  (Full dataset counts: 230 subjects, 902 sessions, 447 t2w)         │  │
│  │  Used for: Logging context, sanity checking                         │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                      │
│                                    ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  Paper-Specific Filtering (OUR CODE)                                │  │
│  │  • strict_t2w=True (no FLAIR fallback)                              │  │
│  │  • include_space_2x=True (115 scans)                                │  │
│  │  • include_space_no_accel=True (109 scans)                          │  │
│  │  • exclude_turbo_spin_echo=True (5 scans excluded)                  │  │
│  │  • require_lesion_mask=True                                         │  │
│  │  → Output: 224 training samples                                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                      │
│                                    ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  Domain Models (OUR CODE)                                           │  │
│  │  • ARCSample: subject_id, session_id, image_path, mask_path,        │  │
│  │               lesion_volume, acquisition_type                       │  │
│  │  • ARCDatasetInfo: List of samples + convenience accessors          │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ arc-meshchop download
                                    ▼
                    ┌───────────────────────────────┐
                    │      data/arc/                │
                    │  ├── dataset_info.json        │
                    │  │   (paths, volumes, types)  │
                    │  └── cache/                   │
                    │      (HuggingFace cache)      │
                    └───────────────┬───────────────┘
                                    │
                                    │ arc-meshchop train
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                          Training Pipeline                                │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  Preprocessing (preprocessing.py)                                   │  │
│  │  • Resample to 256³ @ 1mm isotropic                                 │  │
│  │  • Normalize to 0-1 range                                           │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                      │
│                                    ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  Nested Cross-Validation (splits.py)                                │  │
│  │  • 3 outer folds × 3 inner folds = 9 configurations                 │  │
│  │  • Stratified by: lesion size quartile + acquisition type           │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                      │
│                                    ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  MeshNet Training (trainer.py)                                      │  │
│  │  • MeshNet-26 (147,474 params)                                      │  │
│  │  • AdamW + OneCycleLR + WeightedCrossEntropy                        │  │
│  │  • 10 random restarts per fold (paper methodology)                  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                      │
│                                    ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  Evaluation (metrics.py, evaluator.py)                              │  │
│  │  • DICE coefficient (target: 0.876 ± 0.016)                         │  │
│  │  • Average Volume Difference (target: 0.245 ± 0.036)                │  │
│  │  • Matthews Correlation Coefficient (target: 0.760 ± 0.030)         │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Cache Location

When you run `arc-meshchop download`, data is cached in the **project-local** directory:

```text
data/arc/
├── dataset_info.json           # Paths, volumes, acquisition types
└── cache/                      # HuggingFace cache (project-local)
    └── hugging-science--arc-aphasia-bids/
        ├── blobs/              # Raw NIfTI files
        └── snapshots/          # Dataset versions
```

This project-local cache (instead of global `~/.cache/huggingface/`) ensures:
- Reproducibility: All data is self-contained in the project
- Easy cleanup: Delete `data/arc/` to remove everything
- Explicit storage: You know exactly where ~25GB is being stored

### Customizing Cache Location (Optional)

You can override with environment variables if needed:

```bash
# Force offline mode (use cached data only)
export HF_HUB_OFFLINE=1
```

---

## Component Responsibilities

### 1. bids_hub (neuroimaging-go-brrrr) - External Dependency

**What it provides:**
- `ARC_VALIDATION_CONFIG`: Expected counts for full ARC dataset validation
- `validate_arc_download()`: Validate local BIDS directories (optional)

**What we DON'T use:**
- `build_arc_file_table()`: For UPLOADING to HuggingFace (not our use case)
- `get_arc_features()`: Schema definition for uploads (not needed for consumption)
- SIGKILL workaround: Only relevant for push operations

**Why this architecture:**
The `bids_hub` package was designed for **uploading** BIDS datasets to HuggingFace.
For **consuming** (training), we use `datasets.load_dataset()` directly and only
pull validation constants for sanity checking.

### 2. huggingface_loader.py - Our Data Loading

**Responsibilities:**
- Load ARC dataset from HuggingFace Hub via `datasets.load_dataset()`
- Apply paper-specific filtering (SPACE sequences only, exclude TSE)
- Extract file paths and metadata into domain models
- Verify sample counts match paper (224 = 115 + 109)

**Key functions:**
```python
load_arc_from_huggingface(
    repo_id="hugging-science/arc-aphasia-bids",
    include_space_2x=True,       # 115 scans
    include_space_no_accel=True, # 109 scans
    exclude_turbo_spin_echo=True,# 5 excluded
    strict_t2w=True,             # No FLAIR fallback
    verify_counts=True,          # Enforce 224 total
) -> ARCDatasetInfo
```

### 3. Domain Models

```python
@dataclass
class ARCSample:
    subject_id: str           # e.g., "sub-M2001"
    session_id: str           # e.g., "ses-1"
    image_path: Path          # T2-weighted NIfTI
    mask_path: Path | None    # Lesion mask NIfTI
    lesion_volume: int        # Voxels
    acquisition_type: str     # "space_2x" or "space_no_accel"

@dataclass
class ARCDatasetInfo:
    samples: list[ARCSample]
    # + convenience properties: image_paths, mask_paths, etc.
```

---

## Count Clarification

| Source | Count | Purpose |
|--------|-------|---------|
| **ARC_VALIDATION_CONFIG** | 230 subjects, 902 sessions, 447 t2w | Full dataset integrity |
| **Paper training subset** | 224 samples (115 + 109) | MeshNet training data |

These are DIFFERENT. The paper uses a SUBSET of ARC:
- Only T2w images (no other modalities)
- Only SPACE acquisition (not TSE)
- Only sessions with lesion masks

---

## CLI Workflow

```bash
# Step 1: Download and filter dataset
arc-meshchop download --output data/arc

# Step 2: Train MeshNet-26
arc-meshchop train --data-dir data/arc --output outputs/train

# Step 3: Evaluate
arc-meshchop evaluate outputs/train/best.pt --data-dir data/arc
```

---

## Parity-Critical Behaviors

These behaviors ensure paper replication accuracy:

1. **`strict_t2w=True`**: Only use T2w images, NO FLAIR fallback
2. **Filename-first acquisition parsing**: Extract `acq-*` from BIDS filename
3. **Paper-specific count verification**: 224 = 115 + 109 (hardcoded, not from bids_hub)
4. **Unknown acquisition rejection**: Skip unrecognizable acquisition types

---

## File Structure

```text
src/arc_meshchop/
├── data/
│   ├── huggingface_loader.py  # HuggingFace → ARCDatasetInfo
│   ├── dataset.py             # PyTorch Dataset wrapper
│   ├── preprocessing.py       # 256³ resampling, normalization
│   ├── splits.py              # Nested CV with stratification
│   └── config.py              # Data pipeline configuration
├── models/
│   └── meshnet.py             # 10-layer dilated CNN (147K params)
├── training/
│   ├── trainer.py             # Training loop
│   ├── loss.py                # Weighted CrossEntropy
│   ├── optimizer.py           # AdamW + OneCycleLR
│   └── hpo.py                 # Hyperparameter optimization
├── evaluation/
│   ├── metrics.py             # DICE, AVD, MCC
│   ├── evaluator.py           # Batch evaluation
│   └── statistics.py          # Wilcoxon, Holm-Bonferroni
├── experiment/
│   └── runner.py              # Full experiment orchestration
├── export/
│   ├── onnx_export.py         # PyTorch → ONNX
│   └── pipeline.py            # Export orchestration
└── cli.py                     # Typer CLI commands
```

---

## Ready to Train

With this architecture:

1. **Data loading** is correct: Uses `datasets.load_dataset()` with paper-specific filtering
2. **bids_hub integration** is correct: Only validation constants, not upload utilities
3. **Specs are aligned**: All docs now accurately reflect the architecture
4. **Tests pass**: 276 tests verify the implementation

```bash
# Full training run (paper methodology)
arc-meshchop download
arc-meshchop train --channels 26 --epochs 50
```

Target: **DICE 0.876 ± 0.016** with 147K parameters.
