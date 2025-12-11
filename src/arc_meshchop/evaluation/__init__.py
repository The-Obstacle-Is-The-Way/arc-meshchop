"""Evaluation metrics and statistical testing."""

from arc_meshchop.evaluation.evaluator import EvaluationResult, Evaluator
from arc_meshchop.evaluation.metrics import (
    MetricResult,
    SegmentationMetrics,
    average_volume_difference,
    compute_confusion_matrix,
    dice_coefficient,
    matthews_correlation_coefficient,
)
from arc_meshchop.evaluation.statistics import (
    ComparisonResult,
    compare_models_to_reference,
    format_results_table,
    holm_bonferroni_correction,
    wilcoxon_test,
)

__all__ = [
    "ComparisonResult",
    "EvaluationResult",
    "Evaluator",
    "MetricResult",
    "SegmentationMetrics",
    "average_volume_difference",
    "compare_models_to_reference",
    "compute_confusion_matrix",
    "dice_coefficient",
    "format_results_table",
    "holm_bonferroni_correction",
    "matthews_correlation_coefficient",
    "wilcoxon_test",
]
