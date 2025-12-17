# Documentation Index

Navigation guide to ARC-MeshChop documentation.

---

## Quick Start

| Goal | Document |
|------|----------|
| Understand the project | [`README.md`](../README.md) |
| Set up development | [`CONTRIBUTING.md`](../CONTRIBUTING.md) |
| Train a model | [`TRAINING.md`](../TRAINING.md) |
| Understand data flow | [`DATA.md`](../DATA.md) |

---

## Reference Documentation

Deep technical documentation extracted from the paper:

| Document | Content |
|----------|---------|
| [`reference/meshnet.md`](reference/meshnet.md) | MeshNet architecture (10-layer dilated CNN) |
| [`reference/dataset.md`](reference/dataset.md) | ARC dataset and preprocessing (256³ @ 1mm) |
| [`reference/training.md`](reference/training.md) | Training configuration (AdamW, OneCycleLR) |
| [`reference/metrics.md`](reference/metrics.md) | Evaluation metrics (DICE, AVD, MCC) |
| [`reference/variants.md`](reference/variants.md) | Model variants (MeshNet-5/16/26) |

---

## Guides

| Document | Content |
|----------|---------|
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Exact paper replication protocol |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Common issues and solutions |

---

## Specifications

| Document | Status |
|----------|--------|
| [`specs/07-huggingface-spaces.md`](specs/07-huggingface-spaces.md) | Future work |

---

## Open Issues

| Issue | Description |
|-------|-------------|
| [`issues/TRAIN-001-runtime-estimates.md`](issues/TRAIN-001-runtime-estimates.md) | Hardware runtime benchmarks needed |
| [`issues/OPT-001-download-performance.md`](issues/OPT-001-download-performance.md) | Download performance (deferred) |

---

## For Claude Code

Agent instructions are in [`CLAUDE.md`](../CLAUDE.md) at the repo root.
