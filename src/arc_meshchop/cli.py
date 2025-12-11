"""Command-line interface for arc-meshchop."""

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="arc-meshchop",
    help="MeshNet stroke lesion segmentation - paper replication",
    add_completion=False,
)
console = Console()


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


if __name__ == "__main__":
    app()
