# What We're Training

> **TL;DR:** MeshNet-26 with 147K parameters. That's it. Runs on M1 MacBook.

---

## The Model: MeshNet-26

```text
┌─────────────────────────────────────────────────────────────────┐
│                         MeshNet-26                              │
│                                                                 │
│  Parameters: 147,474  (~600 KB of weights)                      │
│  Input:      256 × 256 × 256 × 1  (T2-weighted MRI)             │
│  Output:     256 × 256 × 256 × 2  (background + lesion)         │
│                                                                 │
│  Architecture:                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Layer 1:  Conv3D(1→26, k=3, d=1)  → BN → ReLU           │   │
│  │  Layer 2:  Conv3D(26→26, k=3, d=2)  → BN → ReLU          │   │
│  │  Layer 3:  Conv3D(26→26, k=3, d=4)  → BN → ReLU          │   │
│  │  Layer 4:  Conv3D(26→26, k=3, d=8)  → BN → ReLU          │   │
│  │  Layer 5:  Conv3D(26→26, k=3, d=16) → BN → ReLU          │   │
│  │  Layer 6:  Conv3D(26→26, k=3, d=16) → BN → ReLU          │   │
│  │  Layer 7:  Conv3D(26→26, k=3, d=8)  → BN → ReLU          │   │
│  │  Layer 8:  Conv3D(26→26, k=3, d=4)  → BN → ReLU          │   │
│  │  Layer 9:  Conv3D(26→26, k=3, d=2)  → BN → ReLU          │   │
│  │  Layer 10: Conv3D(26→2, k=1, d=1)   (no BN, no ReLU)     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Dilation pattern: 1 → 2 → 4 → 8 → 16 → 16 → 8 → 4 → 2 → 1      │
│                    └─── encoder ───┘    └─── decoder ───┘       │
│                                                                 │
│  No skip connections. No downsampling. No upsampling.           │
│  Just dilated convolutions.                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why This is Perfect for M1 MacBook

| Factor | Value | Why It's Fine |
|--------|-------|---------------|
| **Parameters** | 147,474 | ~600 KB - fits in L2 cache |
| **Model size** | ~0.6 MB | Trivial |
| **Input volume** | 256³ × 1 | ~67 MB per sample |
| **Batch size** | 1 | One volume at a time |
| **Precision** | FP32 on MPS | M1 doesn't need FP16 |

**The whole point of the paper** is that MeshNet runs on resource-limited environments like web browsers (brainchop.org). Your M1 MacBook is MORE powerful than a browser.

---

## What We're NOT Training

The paper compared MeshNet against these models to show it BEATS them:

| Model | Parameters | We Train? |
|-------|------------|-----------|
| **MeshNet-26** | **147,474** | **YES** |
| MeshNet-16 | 56,194 | Optional |
| MeshNet-5 | 5,682 | Optional |
| U-MAMBA-BOT | 7,351,400 | NO |
| MedNeXt-M | 17,548,963 | NO |
| Swin-UNETR | 18,346,844 | NO |
| U-KAN | 44,070,082 | NO |
| UNETR | 95,763,682 | NO |

Those big models need A100 GPUs. We don't need them - **MeshNet-26 already beats them!**

---

## Training Details

```python
# From the paper - Section 2
optimizer = AdamW(lr=0.001, weight_decay=3e-5, eps=1e-4)
scheduler = OneCycleLR(max_lr=0.001, pct_start=0.01)  # 1% warmup
loss = CrossEntropyLoss(weight=[0.5, 1.0], label_smoothing=0.01)
epochs = 50
batch_size = 1
restarts = 10  # Train 10 times, pick best
```

---

## Dataset

```text
ARC Dataset (hugging-science/arc-aphasia-bids)
├── Full dataset: 230 subjects, 902 sessions, 228 lesion masks
├── SPACE subset: 223 samples (after filtering)
│   ├── 115 SPACE with 2x acceleration
│   └── 108 SPACE without acceleration
│   (Excludes 5 turbo-spin-echo sequences)
├── 256³ @ 1mm isotropic (after preprocessing)
└── Expert lesion masks for each sample

Note: Paper cites 224 (115+109), but OpenNeuro has 223 (115+108).
```

---

## Training Time Estimate

See [TRAIN-001: Runtime Estimates](docs/issues/TRAIN-001-runtime-estimates.md) for detailed timing by hardware.

| Platform | Per-Run | Full (30 runs) |
|----------|---------|----------------|
| MPS (M-series Mac) | ~62 hours | ~77 days |
| RTX 4090 (est.) | ~6-12 hours | ~7-15 days |

**Note:** You only need **1 run** to get a working model. The 30-run protocol is for paper-style statistical reporting.

---

## Commands

```bash
# Step 1: Download dataset (once)
arc-meshchop download --output data/arc

# Step 2: Train MeshNet-26
arc-meshchop train \
  --data-dir data/arc \
  --output outputs/meshnet26 \
  --channels 26 \
  --resample-method nibabel_conform \
  --epochs 50

# Step 3: Evaluate
arc-meshchop evaluate outputs/meshnet26/fold_0_0/best.pt \
  --data-dir data/arc \
  --resample-method nibabel_conform
```

---

## Target Performance

From the paper (Table 1), MeshNet-26 achieves:

| Metric | Value | Description |
|--------|-------|-------------|
| **DICE** | **0.876 ± 0.016** | Overlap with ground truth |
| AVD | 0.245 ± 0.036 | Volume difference |
| MCC | 0.760 ± 0.030 | Correlation coefficient |

This BEATS:
- U-MAMBA-BOT (0.870 DICE) with 50× more parameters
- MedNeXt-M (0.868 DICE) with 119× more parameters
- Swin-UNETR (0.859 DICE) with 124× more parameters

---

## Summary

```text
┌─────────────────────────────────────────────────────────────────┐
│                     WHAT WE'RE DOING                            │
├─────────────────────────────────────────────────────────────────┤
│  Model:      MeshNet-26 (147K params)                           │
│  Dataset:    ARC (223 SPACE samples from 230-subject dataset)   │
│  Hardware:   M1 MacBook (MPS backend)                           │
│  Time:       ~62 hours/run (MPS)                                │
│  Target:     DICE 0.876                                         │
│                                                                 │
│  NOT training U-MAMBA, MedNeXt, or any other big model.         │
│  MeshNet is the whole point - it's tiny AND better.             │
└─────────────────────────────────────────────────────────────────┘
```
