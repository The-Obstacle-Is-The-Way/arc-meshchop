"""Command-line interface for arc-meshchop.

Provides commands for downloading data, training models, running experiments,
and evaluating results for MeshNet stroke lesion segmentation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Literal, cast

import typer
from rich.console import Console
from rich.table import Table

from arc_meshchop.utils.paths import resolve_dataset_path

app = typer.Typer(
    name="arc-meshchop",
    help="MeshNet stroke lesion segmentation - paper replication",
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _validate_paper_parity(
    num_samples: int,
    acquisition_types: list[str],
    *,
    context: str = "Paper Parity",
) -> None:
    """Validate dataset meets paper parity requirements (223 samples, 0 TSE).

    Args:
        num_samples: Number of samples in the dataset.
        acquisition_types: List of acquisition type strings.
        context: Context string for error messages.

    Raises:
        typer.Exit: If validation fails.
    """
    if num_samples != 223:
        err_console.print(f"[red]{context} Error: Expected 223 samples, got {num_samples}.[/red]")
        err_console.print(
            "Re-run 'arc-meshchop download --paper-parity' to ensure correct dataset."
        )
        raise typer.Exit(1)

    # Use exact matching for TSE detection (not substring)
    tse_count = sum(1 for acq in acquisition_types if acq == "turbo_spin_echo")
    if tse_count > 0:
        err_console.print(f"[red]{context} Error: Found {tse_count} TSE samples.[/red]")
        err_console.print("Re-run 'arc-meshchop download --paper-parity' to exclude TSE.")
        raise typer.Exit(1)

    console.print(f"[yellow]{context} mode: Verified 223 samples (0 TSE).[/yellow]")


@app.command()
def version() -> None:
    """Show version information."""
    from arc_meshchop import __version__

    console.print(f"arc-meshchop v{__version__}")


@app.command()
def info() -> None:
    """Show project and device information."""
    import torch

    from arc_meshchop.utils.device import get_device_info

    # Project info
    console.print("[bold]ARC MeshChop[/bold]")
    console.print("MeshNet stroke lesion segmentation - paper replication")
    console.print()
    console.print("[bold]Paper:[/bold]")
    console.print("  State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters")
    console.print("  Fedorov et al. (Emory, Georgia State, USC)")
    console.print()
    console.print("[bold]Model Variants:[/bold]")
    console.print("  MeshNet-5:  5,682 params  (0.848 DICE)")
    console.print("  MeshNet-16: 56,194 params (0.873 DICE)")
    console.print("  MeshNet-26: 147,474 params (0.876 DICE)")
    console.print()

    # Device info
    device_info = get_device_info()

    table = Table(title="Device Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("PyTorch Version", torch.__version__)
    table.add_row("CUDA Available", str(device_info["cuda_available"]))
    table.add_row("MPS Available", str(device_info["mps_available"]))
    table.add_row("CPU Cores", str(device_info["cpu_count"]))

    if device_info["cuda_available"]:
        table.add_row("CUDA Device", str(device_info.get("cuda_device_name", "N/A")))
        table.add_row("CUDA Memory (GB)", str(device_info.get("cuda_memory_gb", "N/A")))

    if device_info["mps_available"]:
        table.add_row("MPS Functional", str(device_info.get("mps_functional", "N/A")))

    console.print(table)


@app.command()
def download(
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for dataset"),
    ] = Path("data/arc"),
    repo_id: Annotated[
        str,
        typer.Option("--repo", help="HuggingFace repository ID"),
    ] = "hugging-science/arc-aphasia-bids",
    include_space_2x: Annotated[
        bool,
        typer.Option("--include-space-2x/--no-space-2x", help="Include SPACE 2x accel"),
    ] = True,
    include_space_no_accel: Annotated[
        bool,
        typer.Option("--include-space-no-accel/--no-space-no-accel", help="Include SPACE no accel"),
    ] = True,
    exclude_turbo_spin_echo: Annotated[
        bool,
        typer.Option("--exclude-tse/--include-tse", help="Exclude turbo-spin echo"),
    ] = True,
    require_lesion_mask: Annotated[
        bool,
        typer.Option("--require-mask/--no-require-mask", help="Require lesion mask"),
    ] = True,
    verify_counts: Annotated[
        bool,
        typer.Option("--verify-counts/--no-verify-counts", help="Verify paper sample counts"),
    ] = True,
    paper_parity: Annotated[
        bool,
        typer.Option("--paper-parity", help="Use strict 223-sample mode (excludes TSE)"),
    ] = False,
) -> None:
    """Download ARC dataset from HuggingFace Hub.

    Downloads the ARC dataset with specified filters and saves
    file paths for training. By default, downloads ~224 samples
    matching the paper's methodology.
    """
    from arc_meshchop.data import load_arc_from_huggingface

    console.print(f"Downloading ARC dataset to {output_dir}...")

    if paper_parity:
        console.print("[yellow]Paper parity mode enabled: Using strict 223-sample cohort.[/yellow]")
        exclude_turbo_spin_echo = True
        verify_counts = True

    try:
        arc_info = load_arc_from_huggingface(
            repo_id=repo_id,
            cache_dir=output_dir / "cache",
            include_space_2x=include_space_2x,
            include_space_no_accel=include_space_no_accel,
            exclude_turbo_spin_echo=exclude_turbo_spin_echo,
            require_lesion_mask=require_lesion_mask,
            verify_counts=verify_counts,
            paper_parity=paper_parity,
        )

        # Save dataset info for later use
        output_dir.mkdir(parents=True, exist_ok=True)
        info_path = output_dir / "dataset_info.json"

        # Store paths relative to output_dir for portability.
        # Note: Some older dataset_info.json files stored repo-root relative paths like
        # "data/arc/cache/...". New downloads should use "cache/...".
        output_dir_resolved = output_dir.resolve()

        def _rel_path(p: Path) -> str:
            try:
                return str(p.resolve().relative_to(output_dir_resolved))
            except ValueError:
                # If the path isn't under output_dir (unexpected), fall back to raw string.
                return str(p)

        def _rel_path_optional(p: Path | None) -> str | None:
            return _rel_path(p) if p is not None else None

        # Write aligned lists - mask_paths must align 1:1 with image_paths
        # (BUG-001: arc_info.mask_paths filters out None, causing misalignment)
        image_paths_rel = [_rel_path(s.image_path) for s in arc_info.samples]
        mask_paths_aligned = [_rel_path_optional(s.mask_path) for s in arc_info.samples]

        dataset_info = {
            "num_samples": len(arc_info),
            "image_paths": image_paths_rel,
            "mask_paths": mask_paths_aligned,
            "lesion_volumes": arc_info.lesion_volumes,
            "acquisition_types": arc_info.acquisition_types,
            "subject_ids": arc_info.subject_ids,
        }

        info_path.write_text(json.dumps(dataset_info, indent=2))

        console.print(f"[green]Downloaded {len(arc_info)} samples[/green]")
        console.print(f"Dataset info saved to {info_path}")
        if paper_parity:
            console.print("Expected: 223 samples (paper parity subset)")
        else:
            console.print("Expected: ~224 samples (from paper)")

    except Exception as e:
        err_console.print(f"[red]Error downloading dataset: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def train(
    # Data options
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Directory with dataset_info.json"),
    ] = Path("data/arc"),
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for checkpoints"),
    ] = Path("outputs/train"),
    # Model options
    channels: Annotated[
        int,
        typer.Option("--channels", "-c", help="MeshNet channels (5, 16, or 26)"),
    ] = 26,
    # Fold options
    outer_fold: Annotated[
        int,
        typer.Option("--outer-fold", help="Outer CV fold (0-2)"),
    ] = 0,
    inner_fold: Annotated[
        int,
        typer.Option("--inner-fold", help="Inner CV fold (0-2)"),
    ] = 0,
    # Training options (FROM PAPER)
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Number of epochs"),
    ] = 50,
    learning_rate: Annotated[
        float,
        typer.Option("--lr", help="Max learning rate (OneCycleLR max_lr)"),
    ] = 0.001,
    weight_decay: Annotated[
        float,
        typer.Option("--weight-decay", "--wd", help="Weight decay"),
    ] = 3e-5,
    background_weight: Annotated[
        float,
        typer.Option("--bg-weight", help="Background class weight"),
    ] = 0.5,
    warmup_pct: Annotated[
        float,
        typer.Option("--warmup", help="Warmup percentage for OneCycleLR"),
    ] = 0.01,
    div_factor: Annotated[
        float,
        typer.Option("--div-factor", help="OneCycleLR div_factor (initial_lr = max_lr/div_factor)"),
    ] = 100.0,  # FROM PAPER: "starts at 1/100th of the max learning rate"
    use_fp16: Annotated[
        bool,
        typer.Option("--fp16/--no-fp16", help="Use FP16 mixed precision"),
    ] = True,
    # Reproducibility
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed"),
    ] = 42,
    # Other
    num_workers: Annotated[
        int,
        typer.Option("--workers", help="DataLoader workers"),
    ] = 4,
    resume: Annotated[
        Path | None,
        typer.Option("--resume", help="Resume from checkpoint"),
    ] = None,
    paper_parity: Annotated[
        bool,
        typer.Option("--paper-parity", help="Verify strict 223-sample mode"),
    ] = False,
) -> None:
    """Train MeshNet on ARC dataset.

    Trains a single model configuration on specified CV folds.
    Note: `arc-meshchop experiment` runs the paper replication protocol
    (3 outer folds x N restarts) using full outer-train per fold.

    FROM PAPER:
    - AdamW optimizer (lr=0.001, weight_decay=3e-5, eps=1e-4)
    - OneCycleLR scheduler (1% warmup, starts at 1/100th of max_lr)
    - CrossEntropyLoss (weights=[0.5, 1.0], label_smoothing=0.01)
    - 50 epochs, batch size 1
    """
    import torch

    from arc_meshchop.data import (
        ARCDataset,
        create_dataloaders,
        create_stratification_labels,
        generate_nested_cv_splits,
        get_lesion_quintile,
    )
    from arc_meshchop.models import MeshNet
    from arc_meshchop.training import Trainer, TrainingConfig
    from arc_meshchop.utils.device import get_device

    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load dataset info
    info_path = data_dir / "dataset_info.json"
    if not info_path.exists():
        err_console.print(f"[red]Dataset info not found: {info_path}[/red]")
        err_console.print("Run 'arc-meshchop download' first.")
        raise typer.Exit(1)

    with info_path.open() as f:
        dataset_info = json.load(f)

    from arc_meshchop.data.huggingface_loader import parse_dataset_info, validate_masks_present

    try:
        (
            image_paths_str,
            mask_paths_raw,
            lesion_volumes,
            acquisition_types,
            _subject_ids,
        ) = parse_dataset_info(dataset_info, context="Training")
        mask_paths_str = validate_masks_present(mask_paths_raw, context="Training")
    except ValueError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    data_dir = data_dir.resolve()
    image_paths = [cast(Path, resolve_dataset_path(data_dir, p)) for p in image_paths_str]
    mask_paths = [cast(Path, resolve_dataset_path(data_dir, p)) for p in mask_paths_str]

    # Verify paper parity counts
    if paper_parity:
        _validate_paper_parity(len(image_paths), acquisition_types)

    console.print(f"Loaded {len(image_paths)} samples from {info_path}")

    # Generate CV splits
    quintiles = [get_lesion_quintile(v) for v in lesion_volumes]
    strat_labels = create_stratification_labels(quintiles, acquisition_types)

    splits = generate_nested_cv_splits(
        n_samples=len(image_paths),
        stratification_labels=strat_labels,
        num_outer_folds=3,
        num_inner_folds=3,
        random_seed=42,  # Fixed seed for reproducible splits
    )

    # Get fold indices
    split = splits.get_split(outer_fold, inner_fold)
    train_indices = split.train_indices
    val_indices = split.val_indices

    n_train, n_val = len(train_indices), len(val_indices)
    console.print(f"Fold {outer_fold}.{inner_fold}: {n_train} train, {n_val} val")

    # Create datasets
    train_dataset = ARCDataset(
        image_paths=[image_paths[i] for i in train_indices],
        mask_paths=[mask_paths[i] for i in train_indices],
        cache_dir=data_dir / "cache" / f"fold_{outer_fold}_{inner_fold}" / "train",
    )

    val_dataset = ARCDataset(
        image_paths=[image_paths[i] for i in val_indices],
        mask_paths=[mask_paths[i] for i in val_indices],
        cache_dir=data_dir / "cache" / f"fold_{outer_fold}_{inner_fold}" / "val",
    )

    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=1,  # FROM PAPER: batch size 1
        num_workers=num_workers,
    )

    # Create model
    device = get_device()
    console.print(f"Using device: {device}")

    model = MeshNet(channels=channels)
    console.print(f"MeshNet-{channels}: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create training config
    # NOTE: div_factor=100 is critical for paper parity
    # FROM PAPER: "starts at 1/100th of the max learning rate"
    config = TrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        max_lr=learning_rate,
        weight_decay=weight_decay,
        pct_start=warmup_pct,
        div_factor=div_factor,  # Paper requires 100 (start at max_lr/100)
        background_weight=background_weight,
        use_fp16=use_fp16,
        checkpoint_dir=output_dir / f"fold_{outer_fold}_{inner_fold}",
        random_seed=seed,
    )

    # Create trainer
    trainer = Trainer(model, config, device=device)

    # Resume if specified
    if resume and resume.exists():
        console.print(f"Resuming from checkpoint: {resume}")
        trainer.load_checkpoint(resume)

    # Train
    console.print(f"Training for {epochs} epochs...")
    results = trainer.train(train_loader, val_loader)

    # Save results
    results_path = output_dir / f"fold_{outer_fold}_{inner_fold}" / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results_data = {
        "channels": channels,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "epochs": epochs,
        "seed": seed,
        **results,
    }

    results_path.write_text(json.dumps(results_data, indent=2))

    console.print("[green]Training complete![/green]")
    console.print(f"Best DICE: {results['best_dice']:.4f} (epoch {results['best_epoch']})")
    console.print(f"Results saved to {results_path}")


@app.command()
def evaluate(
    checkpoint: Annotated[
        Path,
        typer.Argument(help="Path to model checkpoint"),
    ],
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Directory with dataset_info.json"),
    ] = Path("data/arc"),
    outer_fold: Annotated[
        int,
        typer.Option("--outer-fold", help="Outer CV fold for test set"),
    ] = 0,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output JSON file for results"),
    ] = None,
    channels: Annotated[
        int,
        typer.Option("--channels", "-c", help="MeshNet channels"),
    ] = 26,
) -> None:
    """Evaluate trained model on test set.

    Loads a checkpoint and evaluates on the held-out test fold.
    Reports DICE, AVD, and MCC metrics.
    """
    import torch

    from arc_meshchop.data import (
        ARCDataset,
        create_stratification_labels,
        generate_nested_cv_splits,
        get_lesion_quintile,
    )
    from arc_meshchop.evaluation import SegmentationMetrics
    from arc_meshchop.models import MeshNet
    from arc_meshchop.utils.device import get_device

    # Load dataset info
    info_path = data_dir / "dataset_info.json"
    if not info_path.exists():
        err_console.print(f"[red]Dataset info not found: {info_path}[/red]")
        raise typer.Exit(1)

    with info_path.open() as f:
        dataset_info = json.load(f)

    from arc_meshchop.data.huggingface_loader import parse_dataset_info

    try:
        (
            image_paths_str,
            mask_paths_str_or_none,
            lesion_volumes,
            acquisition_types,
            _subject_ids,
        ) = parse_dataset_info(dataset_info, context="Evaluation")
    except ValueError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    data_dir = data_dir.resolve()
    image_paths = [cast(Path, resolve_dataset_path(data_dir, p)) for p in image_paths_str]
    # Handle None values in mask_paths (BUG-001 fix: --no-require-mask may have None entries)
    mask_paths_raw = [resolve_dataset_path(data_dir, p) for p in mask_paths_str_or_none]

    # Generate splits to get test indices
    quintiles = [get_lesion_quintile(v) for v in lesion_volumes]
    strat_labels = create_stratification_labels(quintiles, acquisition_types)

    splits = generate_nested_cv_splits(
        n_samples=len(image_paths),
        stratification_labels=strat_labels,
        num_outer_folds=3,
        num_inner_folds=3,
        random_seed=42,
    )

    # Get test indices (outer fold test set)
    split = splits.get_split(outer_fold, inner_fold=None)
    test_indices = split.test_indices

    # Validate all test samples have masks (evaluation requires ground truth)
    test_mask_paths = [mask_paths_raw[i] for i in test_indices]
    missing_masks = [i for i, m in zip(test_indices, test_mask_paths, strict=True) if m is None]
    if missing_masks:
        err_console.print(
            f"[red]Error: {len(missing_masks)} test samples have no ground truth masks.[/red]\n"
            f"[yellow]Evaluation requires masks. Re-download with 'arc-meshchop download' "
            f"(default requires masks).[/yellow]"
        )
        raise typer.Exit(1)

    # Cast to non-None list (validated above)
    mask_paths: list[Path] = cast(list[Path], test_mask_paths)

    console.print(f"Evaluating on {len(test_indices)} test samples (outer fold {outer_fold})")

    # Create test dataset
    test_dataset = ARCDataset(
        image_paths=[image_paths[i] for i in test_indices],
        mask_paths=mask_paths,
    )

    # Load model
    device = get_device()
    model = MeshNet(channels=channels)
    model = model.to(device)

    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    console.print(f"Loaded checkpoint from {checkpoint}")

    # Evaluate
    dice_scores = []
    avd_scores = []
    mcc_scores = []

    # Use SegmentationMetrics class
    metrics_calculator = SegmentationMetrics()

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image, mask = test_dataset[idx]
            image = image.unsqueeze(0).to(device)

            model_output = model(image)
            pred = model_output.argmax(dim=1).squeeze(0).cpu()

            # Compute metrics per-sample (no OOM)
            scores = metrics_calculator.compute_single(pred, mask)
            dice_scores.append(scores["dice"])
            avd_scores.append(scores["avd"])
            mcc_scores.append(scores["mcc"])

            if (idx + 1) % 10 == 0:
                console.print(f"Processed {idx + 1}/{len(test_dataset)} samples")

    # Compute mean metrics
    import numpy as np

    metrics = {
        "dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "avd": float(np.mean(avd_scores)) if avd_scores else 0.0,
        "mcc": float(np.mean(mcc_scores)) if mcc_scores else 0.0,
    }

    console.print()
    console.print("[bold]Results:[/bold]")
    console.print(f"  DICE: {metrics['dice']:.4f}")
    console.print(f"  AVD:  {metrics['avd']:.4f}")
    console.print(f"  MCC:  {metrics['mcc']:.4f}")

    # Save results if output specified
    if output:
        results = {
            "checkpoint": str(checkpoint),
            "outer_fold": outer_fold,
            "num_samples": len(test_indices),
            **metrics,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2))
        console.print(f"\nResults saved to {output}")


@app.command()
def experiment(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Directory with dataset_info.json"),
    ] = Path("data/arc"),
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for experiment"),
    ] = Path("experiments"),
    variant: Annotated[
        str,
        typer.Option("--variant", "-v", help="Model variant: meshnet_5, meshnet_16, meshnet_26"),
    ] = "meshnet_26",
    num_restarts: Annotated[
        int,
        typer.Option("--restarts", "-r", help="Number of restarts per fold"),
    ] = 10,
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Number of epochs per run"),
    ] = 50,
    # Hyperparameters (FROM PAPER or from HPO)
    learning_rate: Annotated[
        float,
        typer.Option("--lr", help="Max learning rate"),
    ] = 0.001,
    weight_decay: Annotated[
        float,
        typer.Option("--weight-decay", "--wd", help="Weight decay"),
    ] = 3e-5,
    background_weight: Annotated[
        float,
        typer.Option("--bg-weight", help="Background class weight"),
    ] = 0.5,
    warmup_pct: Annotated[
        float,
        typer.Option("--warmup", help="Warmup percentage for OneCycleLR"),
    ] = 0.01,
    div_factor: Annotated[
        float,
        typer.Option("--div-factor", help="OneCycleLR div_factor"),
    ] = 100.0,  # FROM PAPER: "starts at 1/100th of the max learning rate"
    skip_completed: Annotated[
        bool,
        typer.Option("--skip-completed/--no-skip", help="Skip completed runs"),
    ] = True,
    paper_parity: Annotated[
        bool,
        typer.Option("--paper-parity", help="Verify strict 223-sample mode"),
    ] = False,
) -> None:
    """Run paper replication experiment (outer-fold evaluation + restarts).

    Runs 3 outer folds x N restarts (default 30 runs) and produces
    paper-comparable results. Training uses full outer-train data per fold.

    Hyperparameters can be overridden (e.g., from HPO results).

    FROM PAPER:
    - MeshNet-26: 0.876 (0.016) DICE
    - MeshNet-16: 0.873 (0.007) DICE
    - MeshNet-5: 0.848 (0.023) DICE
    """
    from arc_meshchop.experiment.config import ExperimentConfig
    from arc_meshchop.experiment.runner import run_experiment

    # Validate variant
    valid_variants = {"meshnet_5", "meshnet_16", "meshnet_26"}
    if variant not in valid_variants:
        err_console.print(
            f"[red]Invalid variant '{variant}'. Must be one of: {valid_variants}[/red]"
        )
        raise typer.Exit(1)

    # Verify paper parity counts if requested
    if paper_parity:
        info_path = data_dir / "dataset_info.json"
        if not info_path.exists():
            err_console.print(f"[red]Dataset info not found: {info_path}[/red]")
            err_console.print("Run 'arc-meshchop download' first.")
            raise typer.Exit(1)

        with info_path.open() as f:
            dataset_info = json.load(f)

        _validate_paper_parity(
            len(dataset_info["image_paths"]),
            dataset_info["acquisition_types"],
        )

    # Cast variant to Literal type for ExperimentConfig
    model_variant = cast(Literal["meshnet_5", "meshnet_16", "meshnet_26"], variant)
    config = ExperimentConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        model_variant=model_variant,
        num_restarts=num_restarts,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        background_weight=background_weight,
        warmup_pct=warmup_pct,
        div_factor=div_factor,
        skip_completed=skip_completed,
    )

    console.print(f"Running experiment: {variant}")
    console.print(f"Total runs: {config.total_runs}")
    console.print(f"Hyperparams: lr={learning_rate}, wd={weight_decay}, bg={background_weight}")
    console.print(f"Output: {output_dir}")

    result = run_experiment(config)

    # Report TEST metrics (not validation) for paper parity
    console.print()
    console.print("=" * 60)
    console.print("[bold]EXPERIMENT COMPLETE[/bold]")
    console.print("=" * 60)
    console.print(f"Model: MeshNet-{config.channels}")
    console.print(f"Test DICE: {result.test_mean_dice:.4f} +/- {result.test_std_dice:.4f}")
    console.print(f"Test AVD:  {result.test_mean_avd:.4f} +/- {result.test_std_avd:.4f}")
    console.print(f"Test MCC:  {result.test_mean_mcc:.4f} +/- {result.test_std_mcc:.4f}")
    console.print("Paper Target: DICE 0.876 (MeshNet-26)")
    console.print(f"Duration: {result.total_duration_hours:.1f} hours")
    console.print(f"Results: {output_dir / 'experiment_results.json'}")


@app.command()
def hpo(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Directory with dataset_info.json"),
    ] = Path("data/arc"),
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for HPO results"),
    ] = Path("experiments/hpo"),
    outer_fold: Annotated[
        int,
        typer.Option("--outer-fold", help="Outer fold for HPO (paper uses 0)"),
    ] = 0,
    max_trials: Annotated[
        int,
        typer.Option("--max-trials", "-n", help="Maximum HPO trials"),
    ] = 100,
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Epochs per trial"),
    ] = 50,
    pruning: Annotated[
        bool,
        typer.Option("--pruning/--no-pruning", help="Enable ASHA-style pruning"),
    ] = True,
) -> None:
    """Run hyperparameter optimization.

    FROM PAPER Section 2:
    "HPO was conducted on the inner folds of the first outer fold"
    "We employed the Asynchronous Successive Halving Algorithm (ASHA)"

    This replicates the paper's HPO process using Optuna with ASHA pruning.
    Results can be passed to the `experiment` command.
    """
    from arc_meshchop.training.hpo import create_study, run_hpo_trial

    console.print(f"Running HPO on outer fold {outer_fold}")
    console.print(f"Max trials: {max_trials}, Epochs: {epochs}")

    study = create_study(
        study_name=f"meshnet_hpo_outer{outer_fold}",
        pruning=pruning,
    )

    study.optimize(
        lambda trial: run_hpo_trial(
            trial,
            data_dir=data_dir,
            outer_fold=outer_fold,
            epochs=epochs,
        ),
        n_trials=max_trials,
    )

    # Save best params
    output_dir.mkdir(parents=True, exist_ok=True)
    best_params = {
        "learning_rate": study.best_params["learning_rate"],
        "weight_decay": study.best_params["weight_decay"],
        "bg_weight": study.best_params["bg_weight"],
        "warmup_pct": study.best_params["warmup_pct"],
        "best_dice": study.best_value,
        "n_trials": len(study.trials),
    }

    params_path = output_dir / "best_params.json"
    params_path.write_text(json.dumps(best_params, indent=2))

    console.print()
    console.print("[green]HPO Complete![/green]")
    console.print(f"Best DICE: {study.best_value:.4f}")
    console.print(f"Best params: {params_path}")
    console.print()
    console.print("To run full experiment with these params:")
    console.print("  uv run arc-meshchop experiment \\")
    console.print(f"    --lr {best_params['learning_rate']:.6f} \\")
    console.print(f"    --wd {best_params['weight_decay']:.6f} \\")
    console.print(f"    --bg-weight {best_params['bg_weight']:.4f} \\")
    console.print(f"    --warmup {best_params['warmup_pct']:.4f}")


@app.command()
def validate(
    results_file: Annotated[
        Path,
        typer.Argument(help="Path to experiment_results.json"),
    ],
    variant: Annotated[
        str,
        typer.Option("--variant", "-v", help="Model variant"),
    ] = "meshnet_26",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output path for report"),
    ] = None,
) -> None:
    """Validate experiment results against paper.

    Checks if trained model achieves paper-comparable performance.
    Reports parity level: strict, acceptable, minimum, or failed.

    FROM PAPER:
    - MeshNet-26: DICE 0.876 (0.016)
    - MeshNet-16: DICE 0.873 (0.007)
    - MeshNet-5: DICE 0.848 (0.023)
    """
    from arc_meshchop.validation.parity import generate_benchmark_table, validate_parity

    if not results_file.exists():
        err_console.print(f"[red]Results file not found: {results_file}[/red]")
        raise typer.Exit(1)

    result = validate_parity(results_file, variant)

    console.print()
    console.print("=" * 60)
    console.print("[bold]PAPER PARITY VALIDATION[/bold]")
    console.print("=" * 60)
    console.print(str(result))
    console.print()

    console.print("[bold]Detailed Comparison:[/bold]")
    console.print(
        f"  DICE: {result.dice_mean:.4f} vs {result.details['dice']['paper']:.4f} "
        f"({result.details['dice']['pct_diff']:+.1f}%)"
    )
    console.print(
        f"  AVD:  {result.avd_mean:.4f} vs {result.details['avd']['paper']:.4f} "
        f"({result.details['avd']['pct_diff']:+.1f}%)"
    )
    console.print(
        f"  MCC:  {result.mcc_mean:.4f} vs {result.details['mcc']['paper']:.4f} "
        f"({result.details['mcc']['pct_diff']:+.1f}%)"
    )

    console.print()
    console.print(f"[bold]Parity Level:[/bold] {result.parity_level.upper()}")

    if result.parity_level == "strict":
        console.print("[green]Congratulations! Strict paper parity achieved![/green]")
    elif result.parity_level == "acceptable":
        console.print("[green]Good! Acceptable paper parity achieved.[/green]")
    elif result.parity_level == "minimum":
        console.print("[yellow]Warning: Only minimum viable parity achieved.[/yellow]")
    else:
        console.print("[red]Failed: Results do not match paper.[/red]")

    if output:
        with results_file.open() as f:
            results_data = json.load(f)
        generate_benchmark_table([results_data], output)
        console.print(f"\nBenchmark table saved to {output}")


if __name__ == "__main__":
    app()
