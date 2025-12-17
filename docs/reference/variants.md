# MeshNet Model Variants

> Reference: "State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters"

## Overview

MeshNet-X variants differ only in **channel count (X)**. All other architectural choices remain constant.

---

## Variant Specifications

### MeshNet-5 (Ultra-Compact)

| Property | Value |
|----------|-------|
| Channels | 5 |
| Parameters | 5,682 |
| DICE Score | 0.848 (+/-0.023) |
| AVD | 0.280 (+/-0.060) |
| MCC | 0.708 (+/-0.042) |

**Use Case:** Extreme resource constraints, proof-of-concept, ultra-low-memory devices

**Notable:**
- Just 5,682 parameters (comparable to a small fully-connected network)
- Achieves 0.848 DICE, just below the 0.85 threshold for "reliable segmentation"
- 1,300x more efficient than U-MAMBA-BOT
- 3,089x more efficient than MedNeXt-M

**Trade-offs:**
- Shows over- and under-segmentation in harder areas
- Struggles with boundary precision in complex lesion shapes
- Statistically significantly worse than MeshNet-26 (p < 0.05)

---

### MeshNet-16 (Balanced)

| Property | Value |
|----------|-------|
| Channels | 16 |
| Parameters | 56,194 |
| DICE Score | 0.873 (+/-0.007) |
| AVD | 0.249 (+/-0.033) |
| MCC | 0.757 (+/-0.013) |

**Use Case:** Production browser deployment (BrainChop), balanced accuracy/efficiency

**Notable:**
- Only 56K parameters
- No statistically significant difference from MeshNet-26
- Lowest standard deviation (most consistent performance)
- 130x more efficient than U-MAMBA-BOT
- 310x more efficient than MedNeXt-M

**Trade-offs:**
- Slightly lower peak performance than MeshNet-26
- May slightly under-segment in highly irregular regions

---

### MeshNet-26 (Best Performance)

| Property | Value |
|----------|-------|
| Channels | 26 |
| Parameters | 147,474 |
| DICE Score | 0.876 (+/-0.016) |
| AVD | 0.245 (+/-0.036) |
| MCC | 0.760 (+/-0.030) |

**Use Case:** Maximum accuracy when ~150K parameters is acceptable

**Notable:**
- **Highest DICE score** among all tested models
- 50x more efficient than U-MAMBA-BOT
- 120x more efficient than MedNeXt-M
- Best lesion boundary alignment

**Trade-offs:**
- 2.6x more parameters than MeshNet-16 for only +0.003 DICE improvement
- Marginally higher memory usage

---

## Comparative Analysis

### Efficiency Ratios (vs. MeshNet Variants)

| Model | Parameters | Relative to MeshNet-26 | Relative to MeshNet-16 | Relative to MeshNet-5 |
|-------|------------|------------------------|------------------------|----------------------|
| MeshNet-26 | 147,474 | 1x | 2.6x | 26x |
| MeshNet-16 | 56,194 | 0.4x | 1x | 10x |
| MeshNet-5 | 5,682 | 0.04x | 0.1x | 1x |
| SegResNet | 1,176,186 | 8x | 21x | 207x |
| MedNeXt-S | 5,201,315 | 35x | 93x | 915x |
| U-MAMBA-BOT | 7,351,400 | 50x | 131x | 1,294x |
| MedNeXt-M | 17,548,963 | 119x | 312x | 3,089x |
| Swin-UNETR | 18,346,844 | 124x | 326x | 3,229x |
| U-KAN | 44,070,082 | 299x | 784x | 7,756x |
| V-Net | 45,597,898 | 309x | 812x | 8,025x |
| UNETR | 95,763,682 | 649x | 1,704x | 16,854x |

---

## Pareto Frontier Analysis

MeshNet variants lie on the **Pareto frontier**, meaning:
- No other model achieves better DICE at the same parameter count
- No other model achieves same DICE with fewer parameters

---

## Memory Requirements by Variant

Estimated for 256^3 input, FP16, batch size 1:

| Variant | Model Size | Activations (peak) | Total VRAM |
|---------|------------|-------------------|------------|
| MeshNet-5 | ~11 KB | ~200 MB | ~1 GB |
| MeshNet-16 | ~112 KB | ~650 MB | ~2 GB |
| MeshNet-26 | ~295 KB | ~1.1 GB | ~4 GB |

**Comparison with baselines:**
| Model | Model Size | Total VRAM (estimated) |
|-------|------------|------------------------|
| SegResNet | ~2.4 MB | ~20 GB |
| MedNeXt-M | ~35 MB | ~40 GB |
| U-MAMBA-BOT | ~15 MB | ~30 GB |
| UNETR | ~191 MB | ~60 GB+ |

---

## Which Variant to Choose?

### Decision Matrix

| Constraint | Recommended | Rationale |
|------------|-------------|-----------|
| Maximum accuracy | MeshNet-26 | Highest DICE (0.876) |
| Browser deployment | MeshNet-16 | Best accuracy/memory trade-off |
| Mobile/edge device | MeshNet-5 | Only 5K params, usable accuracy |
| Research baseline | MeshNet-26 | Compare against best performance |
| Real-time inference | MeshNet-5 or -16 | Faster inference |
| Consistency matters | MeshNet-16 | Lowest std (0.007) |

### Gap Analysis: MeshNet-5 to MeshNet-16

The paper notes:
> "This also suggests a potential for an intermediate model between MeshNet-5 and MeshNet-16 to optimize parameter efficiency further while maintaining robust accuracy."

Potential variants to explore:
- MeshNet-8 (~15K params)
- MeshNet-10 (~25K params)
- MeshNet-12 (~35K params)

---

## Segmentation Quality by Variant

Based on Figure 3 observations:

### MeshNet-26
- Best boundary alignment
- Fewest over/under-segmentation issues
- Captures complex lesion structures well

### MeshNet-16
- Generally follows boundaries closely
- Occasional under-segmentation in highly irregular regions
- Some lesion extensions may be missed

### MeshNet-5
- Over- and under-segmentation in harder areas
- Struggles with boundary precision
- More frequent deviations in specific regions
- Variable performance on finer lesion details

---

## References

- Original MeshNet: Fedorov et al., IJCNN 2017
- DICE threshold (0.85): Liew et al., "A large, curated, open-source stroke neuroimaging dataset," Scientific Data, 2022
