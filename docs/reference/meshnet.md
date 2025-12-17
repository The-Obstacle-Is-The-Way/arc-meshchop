# MeshNet Architecture

> Reference: "State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters" (Fedorov et al.)

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
Layer 1:  Input -> Conv3D(k=3, d=1, p=1)  -> BN -> ReLU
Layer 2:  -> Conv3D(k=3, d=2, p=2)  -> BN -> ReLU
Layer 3:  -> Conv3D(k=3, d=4, p=4)  -> BN -> ReLU
Layer 4:  -> Conv3D(k=3, d=8, p=8)  -> BN -> ReLU
Layer 5:  -> Conv3D(k=3, d=16, p=16) -> BN -> ReLU  [Maximum receptive field]
Layer 6:  -> Conv3D(k=3, d=16, p=16) -> BN -> ReLU
Layer 7:  -> Conv3D(k=3, d=8, p=8)  -> BN -> ReLU
Layer 8:  -> Conv3D(k=3, d=4, p=4)  -> BN -> ReLU
Layer 9:  -> Conv3D(k=3, d=2, p=2)  -> BN -> ReLU
Layer 10: -> Conv3D(k=1, d=1, p=0)  -> Output (2 classes)

k=kernel_size, d=dilation, p=padding, BN=BatchNorm3d
```

### Dilation Pattern (Key Innovation)

**Original MeshNet (2017):**
```
1 -> 1 -> 2 -> 4 -> 8 -> 16 -> 1 -> 1  (8 layers)
```

**Revisited MeshNet (This Paper):**
```
1 -> 2 -> 4 -> 8 -> 16 -> 16 -> 8 -> 4 -> 2 -> 1  (10 layers, symmetric)
```

The symmetric pattern:
- **Encoder phase (layers 1-5):** Progressive dilation increase captures broad contextual information
- **Decoder phase (layers 6-10):** Progressive dilation decrease recovers fine-grained spatial details
- Mimics U-Net's encoder-decoder information flow without the architectural complexity

### Convolution Specifications

| Property | Value |
|----------|-------|
| Kernel size | 3x3x3 (final layer: 1x1x1) |
| Padding | Dilation-dependent (padding=dilation) |
| Activation | ReLU |
| Normalization | BatchNorm3d |
| Dropout | Optional (Dropout3d) |
| Bias | True (init to 0.0) |
| Weight init | Xavier normal |
| Input shape | 256x256x256x1 |
| Output shape | 256x256x256x2 |

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

For a 3x3x3 kernel with dilation `d`, the effective kernel size is:
```
effective_size = kernel_size + (kernel_size - 1) * (dilation - 1)
               = 3 + 2 * (d - 1)
```

| Dilation | Effective Kernel Size |
|----------|----------------------|
| 1 | 3x3x3 |
| 2 | 5x5x5 |
| 4 | 9x9x9 |
| 8 | 17x17x17 |
| 16 | 33x33x33 |

The cumulative receptive field after all layers covers the entire 256^3 volume.

### Memory Efficiency

Traditional U-Net stores intermediate feature maps for skip connections:
- Memory: O(batch x channels x D x H x W) per skip connection level
- For 256^3 volumes at multiple scales: **prohibitively expensive**

MeshNet processes layer-by-layer with constant memory:
- Only needs current and previous layer activations
- Memory: O(batch x channels x D x H x W) total
- **Critical for browser deployment (WebGL memory limits)**

---

## Key Implementation Notes

1. **Half-precision (FP16) training** is used for memory efficiency on CUDA
2. **Cross-platform AMP:** Uses `torch.amp` (not `torch.cuda.amp`) for CUDA/MPS/CPU support
3. **No data augmentation** mentioned in the paper
4. **Padding = dilation** to maintain spatial dimensions
5. **Final layer: 1x1x1 conv** (no activation, no BatchNorm)

---

## References

- Original MeshNet: Fedorov et al., "End-to-end learning of brain tissue segmentation from imperfect labeling," IJCNN 2017
- BrainChop implementation: https://brainchop.org
- Dilated convolutions: Yu & Koltun, "Multi-Scale Context Aggregation by Dilated Convolutions"
