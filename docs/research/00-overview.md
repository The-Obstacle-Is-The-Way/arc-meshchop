# Stroke Lesion Segmentation Research Overview

> Technical research documentation extracted from:
> **"State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters"**
> Fedorov et al. (Emory University, Georgia State University, University of South Carolina)

> **Documentation Standard:** These docs distinguish between facts stated in the paper and reasonable inferences. Items marked with "inferred" or "estimated" are not explicitly stated in the paper but are derived from standard practices or the original MeshNet architecture.

## Paper Summary

This paper presents a revisited **MeshNet** architecture that achieves state-of-the-art stroke lesion segmentation using **only 147,474 parameters** - approximately 1/1000th the parameters of competing models like U-MAMBA (7.3M), MedNeXt (17.5M), and UNETR (95.8M).

### Key Results

| Model | Parameters | DICE Score |
|-------|------------|------------|
| **MeshNet-26** | **147K** | **0.876** |
| MeshNet-16 | 56K | 0.873 |
| U-MAMBA-BOT | 7.3M | 0.870 |
| MedNeXt-M | 17.5M | 0.868 |
| UNETR | 95.8M | 0.847 |

---

## Research Documentation Index

### [01 - Architecture](./01-architecture.md)

Core MeshNet design and technical details:

- 10-layer fully convolutional structure
- Symmetric dilation pattern (1→2→4→8→16→16→8→4→2→1)
- How it mimics encoder-decoder without downsampling
- Receptive field calculations
- Memory efficiency analysis

### [02 - Dataset & Preprocessing](./02-dataset-preprocessing.md)

Data preparation pipeline:

- ARC (Aphasia Recovery Cohort) dataset specifications
- Acquisition protocol filtering (SPACE sequences)
- Resampling to 256³ @ 1mm isotropic
- Intensity normalization
- Nested cross-validation structure
- Stratification strategy

### [03 - Training Configuration](./03-training-configuration.md)

Complete training setup:

- Optimizer settings (AdamW)
- Learning rate scheduling (OneCycleLR)
- Loss function (weighted cross-entropy + label smoothing)
- Hyperparameter search with Orion/ASHA (our impl uses Optuna for Python 3.12+)
- Half-precision (FP16) training
- Hardware requirements

### [04 - Model Variants](./04-model-variants.md)

Comparison of MeshNet variants:

- MeshNet-5 (5.7K params, 0.848 DICE)
- MeshNet-16 (56K params, 0.873 DICE)
- MeshNet-26 (147K params, 0.876 DICE)
- Efficiency ratios vs. baselines
- Pareto frontier analysis
- PyTorch implementation skeleton

### [05 - Evaluation Metrics](./05-evaluation-metrics.md)

Performance measurement:

- DICE coefficient
- Average Volume Difference (AVD)
- Matthews Correlation Coefficient (MCC)
- Statistical testing (Wilcoxon + Holm correction)
- Complete benchmark results table
- Metric implementation code

### [06 - Implementation Checklist](./06-implementation-checklist.md)

Step-by-step reproduction guide:

- Environment setup
- Data acquisition and preprocessing
- Model implementation
- Training configuration
- Hyperparameter optimization
- Evaluation protocol
- Deployment considerations

---

## Why This Matters

### Clinical Relevance

- **Chronic stroke affects millions** - accurate lesion segmentation aids diagnosis and treatment planning
- **Manual delineation is subjective and time-consuming**
- Automated segmentation provides rapid, objective analysis

### Technical Innovation

1. **Parameter Efficiency:** 50-1000× fewer parameters than alternatives
2. **Memory Efficiency:** No skip connections = less memory for high-res 3D volumes
3. **Browser Deployment:** Enables client-side inference (BrainChop)
4. **Full-Volume Processing:** 256³ whole-brain analysis in single pass

### Integration Potential

This architecture is well-suited for:

- **BrainChop** - Browser-based MRI segmentation
- **NiiVue** - WebGL neuroimaging visualization
- **Edge devices** - Mobile/embedded deployment
- **Resource-limited environments** - Clinics without GPU infrastructure

---

## Key Technical Takeaways

### Architecture

```
Dilation Pattern: 1 → 2 → 4 → 8 → 16 → 16 → 8 → 4 → 2 → 1
                  ↑___encoder___↑     ↑___decoder___↑
```

- Symmetric dilation mimics encoder-decoder
- No pooling, upsampling, or skip connections
- Constant spatial resolution (256³)

### Training

- **Loss:** CrossEntropy(weights=[0.5, 1.0], label_smoothing=0.01)
- **Optimizer:** AdamW(lr=0.001, weight_decay=3e-5)
- **Scheduler:** OneCycleLR with 1% warmup
- **Precision:** FP16

### Data

- **Dataset:** ARC (230 subjects, 224 scans after filtering)
- **Resolution:** 256³ @ 1mm isotropic
- **Validation:** Nested 3-fold cross-validation
- **Stratification:** Lesion size quintiles + acquisition type

---

## Quick Reference

### Model Selection

| Use Case | Recommended Variant |
|----------|---------------------|
| Maximum accuracy | MeshNet-26 |
| Browser deployment | MeshNet-16 |
| Ultra-low resources | MeshNet-5 |

### Critical Hyperparameters

| Parameter | Value |
|-----------|-------|
| Channels | 26 (best), 16 (balanced), 5 (minimal) |
| Dilation | [1, 2, 4, 8, 16, 16, 8, 4, 2, 1] |
| Batch size | 1 |
| Epochs | 50 |
| Class weights | [0.5, 1.0] |

---

## Implementation Strategy: BrainChop Relationship

### Key Finding

The BrainChop reference implementation (`_references/brainchop/py2tfjs/meshnet.py`) uses the **OLD 2017 MeshNet architecture**. This paper introduces an **improved 2024 architecture**. The difference is minimal but critical:

| Component | BrainChop (2017) | This Paper (2024) |
|-----------|------------------|-------------------|
| Layers | 8 | 10 |
| Dilation pattern | `1,1,2,4,8,16,1,1` | `1,2,4,8,16,16,8,4,2,1` |
| Conv/BN/ReLU | ✅ Same | ✅ Same |
| Kernel size | ✅ Same (3×3×3) | ✅ Same (3×3×3) |
| Final layer | ✅ Same (1×1×1) | ✅ Same (1×1×1) |

### What We Must Do

**We need to implement our own MeshNet** based on the paper's 10-layer symmetric pattern. BrainChop provides the verified building blocks (Conv→BN→ReLU→Dropout, Xavier init, etc.), but we cannot use BrainChop's architecture as-is because:

1. It uses the old 8-layer pattern
2. It would NOT reproduce the paper's 0.876 DICE score
3. The symmetric "encoder-decoder" dilation is the key innovation

### The Change Is Minimal

```python
# OLD (BrainChop 2017) - DO NOT USE
dilations = [1, 1, 2, 4, 8, 16, 1, 1]  # 8 layers, abrupt drop

# NEW (This Paper 2024) - USE THIS
dilations = [1, 2, 4, 8, 16, 16, 8, 4, 2, 1]  # 10 layers, symmetric
```

Everything else (Conv3D, BatchNorm3d, ReLU, Xavier init, final 1×1×1 layer) stays identical.

### Recommendation

✅ **Use BrainChop as a reference** for verified architecture details
✅ **Implement our own MeshNet** with the new 10-layer pattern
❌ **Do NOT use BrainChop's model directly** - wrong dilation pattern

---

## Next Steps for Implementation

1. **Read architecture doc** → Understand the model design
2. **Review dataset doc** → Plan data pipeline
3. **Study training config** → Set up training loop
4. **Follow implementation checklist** → Build the system
5. **Validate with metrics** → Verify reproduction

---

## External Resources

### Referenced Implementations

- **BrainChop:** https://brainchop.org (browser-based MRI segmentation)
- **MONAI:** https://monai.io (medical imaging framework)
- **Orion:** https://orion.readthedocs.io (paper's HPO tool; we use Optuna for Python 3.12+)

### Dataset

- **OpenNeuro ARC:** Search for "Aphasia Recovery Cohort" on https://openneuro.org

### Original Paper Authors

- Alex Fedorov (Emory University)
- Yutong Bu (Emory University)
- Xiao Hu (Emory University)
- Chris Rorden (University of South Carolina)
- Sergey Plis (Georgia State University)
