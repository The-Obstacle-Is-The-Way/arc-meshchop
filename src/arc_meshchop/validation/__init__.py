"""Paper parity validation for MeshNet replication."""

from arc_meshchop.validation.parity import (
    PAPER_RESULTS,
    ParityResult,
    generate_benchmark_table,
    run_statistical_comparison,
    validate_parity,
)

__all__ = [
    "PAPER_RESULTS",
    "ParityResult",
    "generate_benchmark_table",
    "run_statistical_comparison",
    "validate_parity",
]
