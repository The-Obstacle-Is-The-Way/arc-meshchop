# ARC MeshChop: TDD Specification Index

> **MeshNet Stroke Lesion Segmentation Paper Replication**
>
> Goal: Replicate "State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters" (Fedorov et al.) and deploy to Hugging Face Spaces.

---

## Specification Status

### ✅ IMPLEMENTED (Archived)

All core specs have been implemented and are archived in `../archive/specs/`:

| Phase | Document | Status |
|-------|----------|--------|
| 1 | [01-project-setup.md](../archive/specs/01-project-setup.md) | ✅ Archived |
| 2 | [02-meshnet-architecture.md](../archive/specs/02-meshnet-architecture.md) | ✅ Archived |
| 3 | [03-data-pipeline.md](../archive/specs/03-data-pipeline.md) | ✅ Archived |
| 4 | [04-training-infrastructure.md](../archive/specs/04-training-infrastructure.md) | ✅ Archived |
| 5 | [05-evaluation-metrics.md](../archive/specs/05-evaluation-metrics.md) | ✅ Archived |
| 6 | [06-model-export.md](../archive/specs/06-model-export.md) | ✅ Archived |
| 8 | [08-cross-platform.md](../archive/specs/08-cross-platform.md) | ✅ Archived |

Local specs (also archived):
- [FIX-001-bids-hub-integration.md](../archive/specs/FIX-001-bids-hub-integration.md)
- [local-01-hf-data-loader.md](../archive/specs/local-01-hf-data-loader.md)
- [local-02-training-cli.md](../archive/specs/local-02-training-cli.md)
- [local-03-experiment-runner.md](../archive/specs/local-03-experiment-runner.md)
- [local-04-paper-parity-validation.md](../archive/specs/local-04-paper-parity-validation.md)
- [local-05-upstream-dataset-fixes.md](../archive/specs/local-05-upstream-dataset-fixes.md)
- [local-06-data-contract-hardening.md](../archive/specs/local-06-data-contract-hardening.md)

Additional archived specs:
- [IO-DIRECTORY-STRUCTURE.md](../archive/specs/IO-DIRECTORY-STRUCTURE.md)

### 🚧 PENDING IMPLEMENTATION

| Phase | Document | Description | Blocker |
|-------|----------|-------------|---------|
| 7 | [07-huggingface-spaces.md](./07-huggingface-spaces.md) | HF Spaces deployment | Needs trained model |
| 9 | [09-io-registry-and-logging.md](./09-io-registry-and-logging.md) | IO registry + run logs | None |
| 10 | [10-reproducibility-seeding.md](./10-reproducibility-seeding.md) | Deterministic seeding | None |
| 11 | [11-hpo-trial-metric-reporting.md](./11-hpo-trial-metric-reporting.md) | Stable HPO reporting | None |
| 12 | [12-stratification-guardrails.md](./12-stratification-guardrails.md) | Stratification fallbacks | None |

---

## Quick Reference

### Target Performance (FROM PAPER)

| Model | Parameters | DICE | AVD | MCC |
|-------|------------|------|-----|-----|
| MeshNet-5 | 5,682 | 0.848 | 0.280 | 0.708 |
| MeshNet-16 | 56,194 | 0.873 | 0.249 | 0.757 |
| **MeshNet-26** | **147,474** | **0.876** | **0.245** | **0.760** |

### Critical Architecture Details

```
Dilation Pattern (10 layers):
  1 → 2 → 4 → 8 → 16 → 16 → 8 → 4 → 2 → 1
  └─── encoder ───┘     └─── decoder ───┘

Layer Structure:
  Layers 1-9: Conv3D(k=3, p=d, d=d) → BatchNorm3d → ReLU
  Layer 10:   Conv3D(k=1) → Output (2 classes)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=0.001, wd=3e-5, eps=1e-4) |
| Scheduler | OneCycleLR (1% warmup) |
| Loss | CrossEntropy([0.5, 1.0], smooth=0.01) |
| Precision | FP16 on CUDA, FP32 on MPS/CPU |
| Batch size | 1 |
| Epochs | 50 |

---

## Current Status

**Ready to train:** MeshNet-26 on 223 samples (default, excludes TSE)

All blockers resolved:
- ✅ Acquisition type metadata (`t2w_acquisition`) now available in HuggingFace dataset
- ✅ Paper parity mode (`--paper-parity`) for strict 223-sample replication
- ✅ All tests passing (run `make ci` to verify)

---

## Getting Started

```bash
# Install dependencies
uv sync --all-extras

# Check device availability
uv run arc-meshchop info

# Run tests
make ci

# Start training
uv run arc-meshchop train --variant meshnet-26 --epochs 50
```

---

## Documentation

For detailed technical documentation, see:
- [docs/reference/](../reference/) - Deep technical reference
- [docs/REPRODUCIBILITY.md](../REPRODUCIBILITY.md) - Paper replication protocol
- [docs/TROUBLESHOOTING.md](../TROUBLESHOOTING.md) - Common issues

---

## Last Updated

2025-12-16
