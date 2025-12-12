"""Complete evaluation pipeline for model assessment."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from arc_meshchop.evaluation.metrics import MetricResult, SegmentationMetrics
from arc_meshchop.evaluation.statistics import ComparisonResult, compare_models_to_reference

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation results for a model."""

    model_name: str
    dice: MetricResult
    avd: MetricResult
    mcc: MetricResult
    num_samples: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "dice": asdict(self.dice),
            "avd": asdict(self.avd),
            "mcc": asdict(self.mcc),
            "num_samples": self.num_samples,
        }

    def save(self, path: Path | str) -> None:
        """Save results to JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> EvaluationResult:
        """Load results from JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(
            model_name=data["model_name"],
            dice=MetricResult(**data["dice"]),
            avd=MetricResult(**data["avd"]),
            mcc=MetricResult(**data["mcc"]),
            num_samples=data["num_samples"],
        )


class Evaluator:
    """Evaluator for stroke lesion segmentation models.

    Implements the evaluation protocol from the paper:
    - DICE, AVD, MCC metrics
    - Per-sample and aggregate statistics
    - Statistical comparison between models
    """

    def __init__(self, device: torch.device | None = None) -> None:
        """Initialize evaluator.

        Args:
            device: Device for inference (default: auto-detect).
        """
        from arc_meshchop.utils.device import get_device

        # Cross-platform device selection (CUDA > MPS > CPU)
        self.device = device or get_device()
        self.metrics_calculator = SegmentationMetrics()

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        model_name: str = "model",
        use_fp16: bool = True,
    ) -> EvaluationResult:
        """Evaluate model on dataset.

        Args:
            model: Model to evaluate.
            dataloader: DataLoader for evaluation data.
            model_name: Name for logging and results.
            use_fp16: Whether to use FP16 inference.

        Returns:
            EvaluationResult with all metrics.
        """
        model = model.to(self.device)
        model.eval()

        logger.info("Evaluating %s on %d batches", model_name, len(dataloader))

        dice_values: list[float] = []
        avd_values: list[float] = []
        mcc_values: list[float] = []

        for images, masks in tqdm(dataloader, desc=f"Evaluating {model_name}"):
            images = images.to(self.device)

            # Inference with platform-aware mixed precision
            # Only use FP16 autocast on CUDA (MPS/CPU don't benefit)
            if use_fp16 and self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):  # type: ignore[attr-defined]
                    outputs = model(images)
            else:
                outputs = model(images)

            # Get predictions (ensure CPU for metrics computation)
            preds = outputs.argmax(dim=1).cpu()
            masks_cpu = masks.cpu()

            # Compute metrics for batch (incrementally)
            batch_metrics = self.metrics_calculator.compute_with_stats(preds, masks_cpu)
            dice_values.extend(batch_metrics["dice"].values)
            avd_values.extend(batch_metrics["avd"].values)
            mcc_values.extend(batch_metrics["mcc"].values)

        # Helper to create MetricResult
        def create_result(name: str, values: list[float]) -> MetricResult:
            return MetricResult(
                name=name,
                mean=float(np.mean(values)) if values else 0.0,
                std=float(np.std(values)) if values else 0.0,
                values=values,
            )

        result = EvaluationResult(
            model_name=model_name,
            dice=create_result("DICE", dice_values),
            avd=create_result("AVD", avd_values),
            mcc=create_result("MCC", mcc_values),
            num_samples=len(dice_values),
        )

        logger.info(
            "%s: DICE=%s, AVD=%s, MCC=%s",
            model_name,
            result.dice,
            result.avd,
            result.mcc,
        )

        return result

    def compare_to_reference(
        self,
        reference_result: EvaluationResult,
        other_results: list[EvaluationResult],
        metric: str = "dice",
    ) -> list[ComparisonResult]:
        """Compare models to reference using statistical tests.

        Args:
            reference_result: Results from reference model (e.g., MeshNet-26).
            other_results: Results from other models to compare.
            metric: Metric to use for comparison ("dice", "avd", "mcc").

        Returns:
            List of ComparisonResult objects.

        Raises:
            ValueError: If metric is not one of "dice", "avd", "mcc".
        """
        valid_metrics = {"dice", "avd", "mcc"}
        if metric not in valid_metrics:
            msg = f"Invalid metric '{metric}'. Must be one of {valid_metrics}"
            raise ValueError(msg)

        # Get reference scores
        reference_metric = getattr(reference_result, metric)
        reference_scores = np.array(reference_metric.values)

        # Build comparison dict
        model_scores_dict: dict[str, np.ndarray] = {}
        for result in other_results:
            result_metric = getattr(result, metric)
            scores = np.array(result_metric.values)
            model_scores_dict[result.model_name] = scores

        return compare_models_to_reference(reference_scores, model_scores_dict)
