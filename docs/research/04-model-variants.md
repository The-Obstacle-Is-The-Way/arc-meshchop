# MeshNet Model Variants

> Extracted from: "State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters"

## Overview

MeshNet-X variants differ only in **channel count (X)**. All other architectural choices remain constant.

---

## Variant Specifications

### MeshNet-5 (Ultra-Compact)

| Property | Value |
|----------|-------|
| Channels | 5 |
| Parameters | 5,682 |
| DICE Score | 0.848 (±0.023) |
| AVD | 0.280 (±0.060) |
| MCC | 0.708 (±0.042) |

**Use Case:** Extreme resource constraints, proof-of-concept, ultra-low-memory devices

**Notable:**
- Just 5,682 parameters (comparable to a small fully-connected network)
- Achieves 0.848 DICE, just below the 0.85 threshold for "reliable segmentation"
- 1,300× more efficient than U-MAMBA-BOT
- 3,089× more efficient than MedNeXt-M

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
| DICE Score | 0.873 (±0.007) |
| AVD | 0.249 (±0.033) |
| MCC | 0.757 (±0.013) |

**Use Case:** Production browser deployment (BrainChop), balanced accuracy/efficiency

**Notable:**
- Only 56K parameters
- No statistically significant difference from MeshNet-26
- Lowest standard deviation (most consistent performance)
- 130× more efficient than U-MAMBA-BOT
- 310× more efficient than MedNeXt-M

**Trade-offs:**
- Slightly lower peak performance than MeshNet-26
- May slightly under-segment in highly irregular regions

---

### MeshNet-26 (Best Performance)

| Property | Value |
|----------|-------|
| Channels | 26 |
| Parameters | 147,474 |
| DICE Score | 0.876 (±0.016) |
| AVD | 0.245 (±0.036) |
| MCC | 0.760 (±0.030) |

**Use Case:** Maximum accuracy when ~150K parameters is acceptable

**Notable:**
- **Highest DICE score** among all tested models
- 50× more efficient than U-MAMBA-BOT
- 120× more efficient than MedNeXt-M
- Best lesion boundary alignment

**Trade-offs:**
- 2.6× more parameters than MeshNet-16 for only +0.003 DICE improvement
- Marginally higher memory usage

---

## Comparative Analysis

### Parameter Count vs. Performance

```
DICE
0.88 |                              * MeshNet-26
     |                          * MeshNet-16
0.87 |                      * U-MAMBA-BOT
     |                  * MedNeXt-M
0.86 |              * SegResNet  * U-MAMBA-ENC
     |          * MedNeXt-S
0.85 |      * Swin-UNETR  * MedNeXt-B
     |  * U-KAN  * MeshNet-5  * UNETR
0.84 |
     |
0.80 |                                          * V-Net
     |
     +------------------------------------------------→ Parameters
       5K   50K   150K   1M   10M   50M   100M
```

### Efficiency Ratios (vs. MeshNet Variants)

| Model | Parameters | Relative to MeshNet-26 | Relative to MeshNet-16 | Relative to MeshNet-5 |
|-------|------------|------------------------|------------------------|----------------------|
| MeshNet-26 | 147,474 | 1× | 2.6× | 26× |
| MeshNet-16 | 56,194 | 0.4× | 1× | 10× |
| MeshNet-5 | 5,682 | 0.04× | 0.1× | 1× |
| SegResNet | 1,176,186 | 8× | 21× | 207× |
| MedNeXt-S | 5,201,315 | 35× | 93× | 915× |
| U-MAMBA-BOT | 7,351,400 | 50× | 131× | 1,294× |
| MedNeXt-M | 17,548,963 | 119× | 312× | 3,089× |
| Swin-UNETR | 18,346,844 | 124× | 326× | 3,229× |
| U-KAN | 44,070,082 | 299× | 784× | 7,756× |
| V-Net | 45,597,898 | 309× | 812× | 8,025× |
| UNETR | 95,763,682 | 649× | 1,704× | 16,854× |

---

## Pareto Frontier Analysis

MeshNet variants lie on the **Pareto frontier**, meaning:
- No other model achieves better DICE at the same parameter count
- No other model achieves same DICE with fewer parameters

```
DICE
0.88 |  ★ MeshNet-26 (Pareto optimal)
     | ★ MeshNet-16 (Pareto optimal)
0.87 |  ● U-MAMBA-BOT
     |      ● MedNeXt-M
0.86 |
     |
0.85 |
     |★ MeshNet-5 (Pareto optimal)
0.84 |          ● U-KAN
     |                  ● UNETR
     +→ 1/Parameters (log scale) →

★ = Pareto optimal
● = Dominated (better exists)
```

---

## Memory Requirements by Variant

Estimated for 256³ input, FP16, batch size 1:

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

## PyTorch Implementation Skeleton

```python
import torch
import torch.nn as nn

class MeshNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, channels=26):
        super().__init__()

        # Dilation pattern: encoder-decoder style
        dilations = [1, 2, 4, 8, 16, 16, 8, 4, 2, 1]

        layers = []
        prev_channels = in_channels

        for i, dilation in enumerate(dilations):
            is_last = (i == len(dilations) - 1)
            out_ch = num_classes if is_last else channels

            layers.append(
                nn.Conv3d(
                    prev_channels, out_ch,
                    kernel_size=3,
                    padding=dilation,  # Same padding with dilation
                    dilation=dilation,
                    bias=True
                )
            )

            if not is_last:
                layers.append(nn.BatchNorm3d(out_ch))
                layers.append(nn.ReLU(inplace=True))

            prev_channels = out_ch

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Instantiate variants
meshnet_5 = MeshNet(channels=5)    # 5,682 params
meshnet_16 = MeshNet(channels=16)  # 56,194 params
meshnet_26 = MeshNet(channels=26)  # 147,474 params

# Verify parameter counts
for name, model in [('MeshNet-5', meshnet_5), ('MeshNet-16', meshnet_16), ('MeshNet-26', meshnet_26)]:
    params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {params:,} parameters")
```

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
