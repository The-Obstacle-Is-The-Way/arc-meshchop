# Data Directory

This directory is **gitignored** except for this README. When you clone the repo, this directory will be empty except for documentation files.

## Quick Start

```bash
# Download the ARC dataset (~25GB for paper subset)
uv run arc-meshchop download

# Data will be cached here:
# data/arc/cache/
```

## Expected Structure After Download

```
data/
├── README.md          # This file (tracked in git)
├── .gitignore         # Keeps directory tracked, ignores contents
└── arc/
    ├── dataset_info.json
    └── cache/
        ├── hugging-science___arc-aphasia-bids/  # HuggingFace cache
        └── nifti_cache/                         # Decoded NIfTI volumes
```

## Data Source

The ARC (Aphasia Recovery Cohort) dataset is hosted on HuggingFace:
- **Repository:** [hugging-science/arc-aphasia-bids](https://huggingface.co/datasets/hugging-science/arc-aphasia-bids)
- **Paper subset:** 223 T2w FLAIR scans (SPACE acquisition only)
- **Full dataset:** 447 T2w scans, 230 subjects, 902 sessions

## Customizing Cache Location

### Option 1: CLI flag
```bash
uv run arc-meshchop download --output /path/to/data
```

### Option 2: Environment variable
```bash
export HF_HOME=/Volumes/ExternalDrive/huggingface
uv run arc-meshchop download
```

## Offline Mode

Once downloaded, you can train offline:
```bash
export HF_HUB_OFFLINE=1
uv run arc-meshchop train --fold 0
```

## Disk Space Requirements

| Component | Size | Location |
|-----------|------|----------|
| HuggingFace hub cache | ~273 GB | `~/.cache/huggingface/` |
| Extracted NIfTI cache | ~50 GB | `data/arc/cache/` |
| Paper subset (223 samples) | ~25 GB | (subset of above) |
| Per sample (T2w + mask) | ~100 MB | — |

**Notes:**
- During download, ~273GB is needed for the HuggingFace hub cache
- After download, you can clear it: `huggingface-cli cache clean`
- Final project usage: ~50GB in `data/arc/cache/`
- WSL2: Use `--hf-cache /mnt/d/huggingface` to redirect hub cache

## See Also

- [DATA.md](../DATA.md) - Full data pipeline documentation
- [TRAINING.md](../TRAINING.md) - Training guide
- [docs/REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md) - Paper replication protocol
