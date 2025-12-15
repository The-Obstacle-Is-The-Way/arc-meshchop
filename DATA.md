# Data Pipeline

> **The Problem:** HuggingFace `datasets` doesn't natively support NIfTI/BIDS neuroimaging formats.
> **The Solution:** We extend it with domain-specific tooling.

---

## What is neuroimaging-go-brrrr?

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              neuroimaging-go-brrrr IS AN EXTENSION OF HUGGINGFACE               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   pip install datasets              pip install neuroimaging-go-brrrr           │
│   ───────────────────────           ─────────────────────────────────           │
│   Standard HuggingFace              EXTENDS datasets with:                      │
│   • Images, text, audio             • NIfTI file support (.nii.gz)              │
│   • Parquet/Arrow storage           • BIDS directory structure                  │
│   • Hub integration                 • Neuroimaging validation                   │
│                                     • Upload utilities for BIDS→Hub             │
│                                                                                 │
│   neuroimaging-go-brrrr automatically installs datasets as a dependency.        │
│   It's NOT a replacement — it's a domain-specific extension.                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Key insight:** When you `pip install neuroimaging-go-brrrr`, you get:
- `datasets` (the standard HuggingFace library)
- `huggingface-hub` (for Hub interactions)
- `bids_hub` module (neuroimaging-specific extensions)

This is the canonical pattern for domain-specific HuggingFace extensions.

---

## Why This Document Exists

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    THE NEUROIMAGING DATA PROBLEM                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Standard ML Datasets          vs       Neuroimaging (BIDS)                    │
│   ─────────────────────                  ──────────────────────                 │
│   • Images: JPEG, PNG                    • Images: NIfTI (.nii.gz)              │
│   • Format: Simple 2D arrays             • Format: 3D/4D volumes + headers      │
│   • Size: KB per image                   • Size: 50-200 MB per scan             │
│   • Metadata: filename                   • Metadata: BIDS sidecar JSONs         │
│   • Structure: flat folders              • Structure: sub-*/ses-*/anat/         │
│                                                                                 │
│   HuggingFace `datasets` was built for the left column.                         │
│   We're working with the right column.                                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## The Two Pipelines

### Pipeline 1: PRODUCTION (Uploading to HuggingFace)

**This is what we already did.** The ARC dataset is now on HuggingFace.

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Local BIDS     │     │  neuroimaging-go-    │     │   HuggingFace Hub   │
│  Directory      │ ──► │  brrrr (bids_hub)    │ ──► │   hugging-science/  │
│  (OpenNeuro)    │     │                      │     │   arc-aphasia-bids  │
└─────────────────┘     │  • build_arc_file_   │     └─────────────────────┘
                        │    table()           │
                        │  • get_arc_features()│
                        │  • Parquet upload    │
                        └──────────────────────┘
```

**Tools used:** `bids_hub.build_arc_file_table()`, `bids_hub.get_arc_features()`

### Pipeline 2: CONSUMPTION (Training from HuggingFace)

**This is what we do now.** Download and train.

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│   HuggingFace Hub   │     │  datasets.load_      │     │  ~/.cache/          │
│   hugging-science/  │ ──► │  dataset()           │ ──► │  huggingface/hub/   │
│   arc-aphasia-bids  │     │  (standard HF)       │     │  (local cache)      │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
                                     │
                                     │ + Paper-specific filtering
                                     ▼
                            ┌──────────────────────┐
                            │  arc_meshchop.data.  │
                            │  huggingface_loader  │
                            │                      │
                            │  • 223 samples       │
                            │  • SPACE only        │
                            │  • With masks        │
                            └──────────────────────┘
```

**Tools used:** `datasets.load_dataset()` (standard), our filtering code

---

## Where Does the Data Go?

By default, the `arc-meshchop` CLI caches data locally within the project to ensure reproducibility:

```text
data/arc/
├── dataset_info.json    # Metadata and paths
└── cache/               # HuggingFace cache (local)
    ├── hub/             # Raw downloads
    └── datasets/        # Processed Arrow files
```

If you use `datasets.load_dataset()` directly in Python (without the CLI), it uses the standard location:

```text
~/.cache/huggingface/
├── hub/
│   └── datasets--hugging-science--arc-aphasia-bids/
└── datasets/
    └── hugging-science___arc-aphasia-bids/
```

### Customizing Cache Location

You can override the CLI's default cache location:
```bash
arc-meshchop download --output /path/to/data
# Data goes to /path/to/data/cache
```

Or for standard HuggingFace usage:
```bash
export HF_HOME=/Volumes/ExternalDrive/huggingface
```

---

## What bids_hub Provides (and What We Use)

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    neuroimaging-go-brrrr (bids_hub)                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   FOR UPLOADING (Production):              FOR CONSUMING (Us):                  │
│   ─────────────────────────────            ───────────────────                  │
│   ✗ build_arc_file_table()                 ✓ ARC_VALIDATION_CONFIG              │
│   ✗ get_arc_features()                       └── expected_counts["subjects"]    │
│   ✗ validate_arc_download()                  └── expected_counts["sessions"]    │
│                                              └── expected_counts["t2w"]         │
│   We DON'T use these.                                                           │
│   Dataset is already uploaded.             We use this for sanity checking:     │
│                                            "Full dataset has 230 subjects,      │
│                                             902 sessions, 447 T2w scans"        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Why Import from bids_hub at All?

```python
from bids_hub.validation.arc import ARC_VALIDATION_CONFIG

# Provides ground truth for the FULL dataset
# Used for logging context, NOT for filtering
logger.info(
    "Full ARC dataset: %d subjects, %d sessions, %d t2w",
    ARC_VALIDATION_CONFIG.expected_counts["subjects"],   # 230
    ARC_VALIDATION_CONFIG.expected_counts["sessions"],   # 902
    ARC_VALIDATION_CONFIG.expected_counts["t2w"],        # 447
)
```

This is different from the **SPACE subset** (223 samples) which we filter ourselves.

---

## The Filtering Logic

The paper uses a specific subset of ARC:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PAPER FILTERING RULES                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Full ARC T2w: 447 scans                                                       │
│        │                                                                        │
│        ├── SPACE 2x acceleration ─────────────────────► 115 samples             │
│        │                                                                        │
│        ├── SPACE no acceleration ─────────────────────► 108 samples             │
│        │                                                                        │
│        ├── Turbo Spin Echo (TSE) ─────────────────────► EXCLUDED (5)            │
│        │                                                                        │
│        └── No lesion mask ────────────────────────────► EXCLUDED                │
│                                                                                 │
│   SPACE training subset: 223 samples (115 + 108)                                │
│   Note: Paper cites 224 (115+109), OpenNeuro has 223 (115+108).                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### In Code

```python
from arc_meshchop.data.huggingface_loader import load_arc_from_huggingface

# Load with paper-specific filtering
arc_data = load_arc_from_huggingface(
    include_space_2x=True,         # 115 scans
    include_space_no_accel=True,   # 108 scans
    exclude_turbo_spin_echo=True,  # Remove 5 TSE
    require_lesion_mask=True,      # Must have ground truth
    verify_counts=True,            # Enforce 223 total
)

# Returns ARCDatasetInfo with 223 samples
print(len(arc_data))  # 223
```

---

## Count Clarification

| Source | Count | Purpose |
|--------|-------|---------|
| `ARC_VALIDATION_CONFIG` | 230 subjects, 902 sessions, 447 T2w | Full dataset integrity |
| SPACE training subset | 223 samples (115 + 108) | What we actually train on |

These are **different numbers** for **different purposes**:
- 230/902/447 = "Does the full dataset look right?"
- 223 = "How many samples are we training on?"

---

## Dependency Relationship

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PACKAGE DEPENDENCIES                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

   arc-meshchop (this repo)
        │
        ├── neuroimaging-go-brrrr (bids_hub)
        │        │
        │        └── datasets ◄─────────────────┐
        │        └── huggingface-hub ◄──────────┤
        │                                       │
        ├── datasets ◄──────────────────────────┤  Standard HuggingFace
        │                                       │  libraries
        └── huggingface-hub ◄───────────────────┘

   We list datasets and huggingface-hub explicitly for clarity,
   but they'd be installed anyway via bids_hub.
```

---

## Workflow Summary

```bash
# 1. Install (brings in datasets, huggingface-hub, bids_hub)
uv sync --all-extras

# 2. Download dataset (uses datasets.load_dataset under the hood)
uv run arc-meshchop download

# 3. Data goes to data/arc/cache/ (project-local)
ls data/arc/cache/

# 4. Train (loads from cache, filters to 223 samples)
uv run arc-meshchop train --channels 26 --epochs 50
```

---

## FAQ

### Q: Why not just use `datasets.load_dataset()` directly?

You can! But you'd need to:
1. Filter for T2w only
2. Filter for SPACE acquisition only
3. Exclude turbo-spin-echo
4. Require lesion masks
5. Verify you got 223 samples

Our `load_arc_from_huggingface()` does all this for you.

### Q: Does bids_hub download the data?

No. `datasets.load_dataset()` does. bids_hub just provides validation constants.

### Q: Can I change the cache location?

Yes. Set `HF_HOME=/your/path` before running any commands.

### Q: How big is the dataset?

The full ARC dataset is ~50GB. Our filtered subset uses ~25GB (223 T2w + masks).

### Q: Can I use this offline?

Yes, once cached. Set `HF_HUB_OFFLINE=1` to force offline mode.
