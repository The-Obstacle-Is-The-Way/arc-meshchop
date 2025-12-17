# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Documentation overhaul with canonical structure
- `docs/reference/` - Deep technical documentation
- `docs/REPRODUCIBILITY.md` - Exact paper replication protocol
- `docs/TROUBLESHOOTING.md` - Common issues and solutions
- `docs/archive/` - Historical documentation preserved

### Changed
- Fixed stale documentation references (export CLI, Hydra claims)
- Updated checkpoint paths in examples to match fold-scoped output
- Consolidated runtime estimates to TRAIN-001 issue

## [0.1.0] - 2024-12-16

### Added
- Initial MeshNet paper replication implementation
- MeshNet-5/16/26 architecture variants (10-layer symmetric dilation)
- HuggingFace dataset integration for ARC dataset
- Cross-platform training support (CUDA, MPS, CPU)
- Nested 3-fold cross-validation with stratification
- 10-restart training protocol per fold
- DICE, AVD, MCC evaluation metrics
- CLI commands: `download`, `train`, `evaluate`, `experiment`, `hpo`, `validate`
- Python API for ONNX export (`arc_meshchop.export`)

### Target
- DICE: 0.876 +/- 0.016
- AVD: 0.245 +/- 0.036
- MCC: 0.760 +/- 0.030

### Technical
- 147,474 parameters (MeshNet-26)
- 223 training samples (SPACE acquisitions only)
- 256^3 @ 1mm isotropic preprocessing
- AdamW optimizer with OneCycleLR scheduler
- Cross-entropy loss with class weighting [0.5, 1.0]

[unreleased]: https://github.com/The-Obstacle-Is-The-Way/arc-meshchop/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/The-Obstacle-Is-The-Way/arc-meshchop/releases/tag/v0.1.0
