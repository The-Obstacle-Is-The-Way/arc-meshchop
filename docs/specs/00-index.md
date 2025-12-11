# ARC MeshChop: Phased TDD Specification Index

> **MeshNet Stroke Lesion Segmentation Paper Replication**
>
> Goal: Replicate "State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters" (Fedorov et al.) and deploy to Hugging Face Spaces.

---

## Specification Documents

| Phase | Document | Description | Dependencies |
|-------|----------|-------------|--------------|
| 1 | [01-project-setup.md](./01-project-setup.md) | UV, pyproject.toml, Makefile, CI/CD | None |
| 2 | [02-meshnet-architecture.md](./02-meshnet-architecture.md) | MeshNet 10-layer implementation | Phase 1 |
| 3 | [03-data-pipeline.md](./03-data-pipeline.md) | ARC dataset, preprocessing, splits | Phase 1 |
| 4 | [04-training-infrastructure.md](./04-training-infrastructure.md) | Training loop, loss, HPO | Phases 1-3 |
| 5 | [05-evaluation-metrics.md](./05-evaluation-metrics.md) | DICE, AVD, MCC, statistics | Phases 1-4 |
| 6 | [06-model-export.md](./06-model-export.md) | ONNX, TensorFlow.js export | Phases 1-5 |
| 7 | [07-huggingface-spaces.md](./07-huggingface-spaces.md) | HF Spaces deployment (no Gradio) | Phases 1-6 |
| 8 | [08-cross-platform.md](./08-cross-platform.md) | Mac/Linux/Windows device support | Phase 1 |

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
| Precision | FP16 |
| Batch size | 1 |
| Epochs | 50 |

---

## Implementation Order

### Phase 1: Foundation (Est. 1-2 days)

```bash
# Create project structure
mkdir -p src/arc_meshchop/{models,data,training,evaluation,export}
mkdir -p tests/test_{models,data,training,evaluation,export}
mkdir -p configs/{model,training,experiment}

# Initialize with UV
uv init
uv sync --all-extras

# Setup pre-commit
uv run pre-commit install

# Verify setup
make ci
```

### Phase 2: Model (Est. 1 day)

```bash
# Implement MeshNet
# Run: uv run pytest tests/test_models/ -v

# Verify parameter counts:
# MeshNet-5:  5,682
# MeshNet-16: 56,194
# MeshNet-26: 147,474
```

### Phase 3: Data (Est. 2-3 days)

```bash
# Implement preprocessing pipeline
# Download ARC dataset via HuggingFace datasets
# Generate CV splits

# Run: uv run pytest tests/test_data/ -v
```

### Phase 4: Training (Est. 2-3 days)

```bash
# Implement training loop
# Implement loss function
# Optional: Implement HPO with Orion

# Run: uv run pytest tests/test_training/ -v
```

### Phase 5: Evaluation (Est. 1 day)

```bash
# Implement DICE, AVD, MCC
# Implement statistical testing

# Run: uv run pytest tests/test_evaluation/ -v
```

### Phase 6: Export (Est. 1 day)

```bash
# Export to ONNX
# Optional: Convert to TensorFlow.js

# Run: uv run pytest tests/test_export/ -v
```

### Phase 7: Deployment (Est. 1-2 days)

```bash
# Create HF Space
# Deploy model to HF Hub
# Test deployment
```

---

## Key Dependencies

### Production

```toml
[project.dependencies]
torch = ">=2.0.0"
torchvision = ">=0.15.0"
nibabel = ">=5.0.0"
monai = ">=1.3.0"
numpy = ">=1.24.0"
pandas = ">=2.0.0"
scipy = ">=1.10.0"
statsmodels = ">=0.14.0"  # Holm-Bonferroni correction
hydra-core = ">=1.3.0"
optuna = ">=3.5.0"  # Replaces orion (broken on Python 3.12)
typer = ">=0.12.0"
```

### Development

```toml
[project.optional-dependencies.dev]
pytest = ">=8.0.0"
pytest-cov = ">=4.0.0"
pytest-xdist = ">=3.5.0"
pytest-sugar = ">=1.0.0"
mypy = ">=1.11.0"
ruff = ">=0.8.0"
pre-commit = ">=3.0.0"
```

### Export

```toml
[project.optional-dependencies.export]
onnx = ">=1.14.0"
onnxruntime = ">=1.16.0"
```

---

## Reference Implementations

> **Note:** The `_references/` directory contains local copies for **reading and study only**.
> These are NOT Python imports - they are gitignored reference material.

| Reference | GitHub | Use |
|-----------|--------|-----|
| BrainChop | [neuroneural/brainchop](https://github.com/neuroneural/brainchop) | Verified architecture details (JS-based, read-only) |
| NiiVue | [niivue/niivue](https://github.com/niivue/niivue) | Visualization (JS-based, read-only) |

### HuggingFace Dependencies

```toml
# ARC dataset access via HuggingFace (neuroimaging-go-brrrr removed due to submodule issues)
"datasets>=3.4.0",
"huggingface-hub>=0.32.0",
```

---

## Research Documentation

| Document | Content |
|----------|---------|
| [00-overview.md](../research/00-overview.md) | Paper summary, BrainChop relationship |
| [01-architecture.md](../research/01-architecture.md) | Verified architecture details |
| [02-dataset-preprocessing.md](../research/02-dataset-preprocessing.md) | ARC dataset, preprocessing |
| [03-training-configuration.md](../research/03-training-configuration.md) | Training hyperparameters |
| [04-model-variants.md](../research/04-model-variants.md) | MeshNet-5/16/26 comparison |
| [05-evaluation-metrics.md](../research/05-evaluation-metrics.md) | DICE, AVD, MCC definitions |
| [06-implementation-checklist.md](../research/06-implementation-checklist.md) | Step-by-step guide |

---

## Getting Started

```bash
# Clone repository
git clone https://github.com/username/arc-meshchop.git
cd arc-meshchop

# Install dependencies
uv sync --all-extras

# Run tests
make test

# Start implementing from Spec 01
# Follow each phase in order
```

---

## Success Criteria

The replication is successful when:

1. **Parameter counts match exactly:**
   - MeshNet-5: 5,682
   - MeshNet-16: 56,194
   - MeshNet-26: 147,474

2. **Performance is within paper bounds:**
   - MeshNet-26 DICE: 0.876 ± 0.016
   - MeshNet-26 AVD: 0.245 ± 0.036
   - MeshNet-26 MCC: 0.760 ± 0.030

3. **All tests pass:**
   - `make ci` succeeds
   - Coverage ≥ 80%

4. **Deployment works:**
   - Model exported to ONNX
   - HF Space accessible
   - Inference produces valid segmentations

---

## Sources

- [UV Package Manager](https://realpython.com/python-uv/)
- [GitHub Actions Python 2025](https://ber2.github.io/posts/2025_github_actions_python/)
- [MONAI Medical Imaging](https://monai.io/)
- [Hugging Face Spaces Docker](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [Ruff Configuration](https://docs.astral.sh/ruff/configuration/)
