# MeshNet Architecture

> Extracted from: "State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters"
> Authors: Fedorov et al. (Emory University, Georgia State University, University of South Carolina)

## Overview

MeshNet is a **fully convolutional segmentation architecture** that achieves state-of-the-art stroke lesion segmentation using only ~147K parameters (compared to 7-95M for alternatives). The key innovation is a **multi-scale dilation pattern** that mimics encoder-decoder behavior without traditional downsampling, upsampling, or skip connections.

## Core Design Principles

### 1. Dilated Convolutions for Receptive Field Expansion

Instead of using pooling/strided convolutions to increase receptive field (like U-Net), MeshNet uses **dilated (atrous) convolutions**:
- Maintains full spatial resolution throughout the network
- Expands receptive field without increasing parameters
- Avoids information loss from downsampling

### 2. No Skip Connections

Unlike U-Net architectures that require skip connections to recover spatial detail:
- MeshNet maintains spatial consistency through controlled dilation and padding
- Eliminates need to store high-dimensional feature maps from earlier layers
- Significantly reduces memory requirements during inference

### 3. No Feature Concatenation

Traditional encoder-decoder models concatenate encoder features with decoder features:
- MeshNet avoids this entirely
- More memory-efficient (critical for browser deployment)
- Simpler architecture with fewer failure modes

---

## Architecture Details

### Layer Structure: 10-Layer Fully Convolutional Network

**From Paper:** "10-layer structure with an adaptive dilation pattern"

```
Layer 1:  Input → Conv3D (dilation=1)  → [?] → [?]
Layer 2:  → Conv3D (dilation=2)  → [?] → [?]
Layer 3:  → Conv3D (dilation=4)  → [?] → [?]
Layer 4:  → Conv3D (dilation=8)  → [?] → [?]
Layer 5:  → Conv3D (dilation=16) → [?] → [?]  [Maximum receptive field]
Layer 6:  → Conv3D (dilation=16) → [?] → [?]
Layer 7:  → Conv3D (dilation=8)  → [?] → [?]
Layer 8:  → Conv3D (dilation=4)  → [?] → [?]
Layer 9:  → Conv3D (dilation=2)  → [?] → [?]
Layer 10: → Conv3D (dilation=1)  → Output (2 classes: background, lesion)
```

> **⚠️ NOT IN PAPER:** The paper does NOT specify activation functions, normalization layers, or their ordering. The original MeshNet (2017) used BatchNorm + ReLU. For strict reproduction, consult the original MeshNet paper [9] or BrainChop implementation.

### Dilation Pattern (Key Innovation)

**Original MeshNet (2017):**
```
1 → 1 → 2 → 4 → 8 → 16 → 1 → 1  (8 layers)
```

The original pattern increased dilation progressively but then **abruptly dropped** back to 1 in the final layers. The paper notes this "did not decrease the dilation rate in later layers" gradually, which "restricted the model's flexibility in refining segmentation boundaries."

**Revisited MeshNet (This Paper):**
```
1 → 2 → 4 → 8 → 16 → 16 → 8 → 4 → 2 → 1  (10 layers, symmetric)
```

The symmetric pattern:
- **Encoder phase (layers 1-5):** Progressive dilation increase captures broad contextual information
- **Decoder phase (layers 6-10):** Progressive dilation decrease recovers fine-grained spatial details
- Mimics U-Net's encoder-decoder information flow without the architectural complexity

### Convolution Specifications

| Property | Value | Source |
|----------|-------|--------|
| Kernel size | 3×3×3 | **⚠️ NOT IN PAPER** - inferred from original MeshNet [9] |
| Padding | Dilation-dependent | **⚠️ NOT IN PAPER** - inferred to preserve 256³ |
| Activation | ReLU | **⚠️ NOT IN PAPER** - from original MeshNet [9] |
| Normalization | Batch Normalization | **⚠️ NOT IN PAPER** - from original MeshNet [9] |
| Input shape | 256×256×256×1 | **FROM PAPER:** "256³ MRI volumes" |
| Output shape | 256×256×256×2 | **FROM PAPER:** binary segmentation (background/lesion) |

> **⚠️ CRITICAL:** The paper does NOT specify kernel size, padding, activation functions, or normalization. These details must be obtained from the original MeshNet paper [9] or the BrainChop reference implementation. The parameter counts reported (5,682 / 56,194 / 147,474) can be used to reverse-engineer the architecture.

### Channel Count (X in MeshNet-X)

The number of channels is constant across all layers:

| Variant | Channels | Parameters |
|---------|----------|------------|
| MeshNet-5 | 5 | 5,682 |
| MeshNet-16 | 16 | 56,194 |
| MeshNet-26 | 26 | 147,474 |

---

## Why This Works: Technical Analysis

### Receptive Field Calculation

For a 3×3×3 kernel with dilation `d`, the effective kernel size is:
```
effective_size = kernel_size + (kernel_size - 1) × (dilation - 1)
                = 3 + 2 × (d - 1)
```

| Dilation | Effective Kernel Size |
|----------|----------------------|
| 1 | 3×3×3 |
| 2 | 5×5×5 |
| 4 | 9×9×9 |
| 8 | 17×17×17 |
| 16 | 33×33×33 |

The cumulative receptive field after all layers covers the entire 256³ volume.

### Memory Efficiency

Traditional U-Net stores intermediate feature maps for skip connections:
- Memory: O(batch × channels × D × H × W) per skip connection level
- For 256³ volumes at multiple scales: **prohibitively expensive**

MeshNet processes layer-by-layer with constant memory:
- Only needs current and previous layer activations
- Memory: O(batch × channels × D × H × W) total
- **Critical for browser deployment (WebGL memory limits)**

### Parameter Efficiency

The 3×3×3 kernel with C channels has:
```
params_per_layer = C × C × 27 + C (bias) + 2C (BatchNorm)
                 ≈ 27C² + 3C
```

For 10 layers with C=26:
```
≈ 10 × (27 × 676 + 78) = 10 × 18,330 ≈ 183,300
```

Actual: 147,474 (some layers may differ in input channels)

---

## Comparison with Original MeshNet

| Aspect | Original (2017) | Revisited (This Paper) |
|--------|-----------------|------------------------|
| Layers | 8 | 10 |
| Dilation pattern | Monotonic increase | Symmetric (encoder-decoder) |
| Max dilation | 16 | 16 |
| Detail recovery | Limited (no decrease) | Good (symmetric decrease) |
| Use case | Brain parcellation | Stroke lesion segmentation |

---

## Key Implementation Notes

**From Paper:**
1. **Half-precision (FP16) training** is used for memory efficiency
2. **Layer checkpointing** used for models that don't fit in GPU memory

**Inferred (NOT in paper):**
1. **Padding must be carefully calculated** to maintain 256³ output dimensions with varying dilations
2. **Batch normalization** - assumed based on original MeshNet
3. **No dropout mentioned** in this paper
4. **No data augmentation mentioned** in this paper

---

## References

- Original MeshNet: Fedorov et al., "End-to-end learning of brain tissue segmentation from imperfect labeling," IJCNN 2017
- BrainChop implementation: https://brainchop.org
- Dilated convolutions: Yu & Koltun, "Multi-Scale Context Aggregation by Dilated Convolutions"
