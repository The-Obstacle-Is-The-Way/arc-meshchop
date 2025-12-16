# BUG-007: `arc-meshchop experiment --variant` Not Validated (Can Run Wrong Model)

> **Severity:** P3 (misconfiguration risk)
>
> **Status:** OPEN
>
> **Date:** 2025-12-16
>
> **Affected:** `src/arc_meshchop/cli.py`, `src/arc_meshchop/experiment/config.py`

## Summary

The `arc-meshchop experiment` CLI accepts `--variant` as a free-form string and then `cast(...)`s it to a `Literal[...]` without runtime validation. `ExperimentConfig.__post_init__()` treats any unknown value as the default case and sets `channels=26`.

## Why This Can Ruin Results

A typo like `--variant meshnet26` (missing underscore) can silently run MeshNet-26 (or another unintended configuration), leading to incorrect conclusions about model size/performance.

## Evidence

- `src/arc_meshchop/cli.py`: `variant` is `str`, then `cast(Literal[...], variant)` (no check).
- `src/arc_meshchop/experiment/config.py`: unknown `model_variant` falls through to `channels = 26`.

## Mitigations

- **Operational:** rely on the printed `Model: MeshNet-{channels}` line and validate before long runs.
- **Code fix:** validate `variant` against allowed values and `raise typer.BadParameter` on unknown input (or use an Enum/Literal type in the Typer option).
