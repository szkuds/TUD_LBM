"""The ``tud-lbm benchmark`` command."""

from __future__ import annotations
import click
from src.cli._console import cli_command
from src.cli._console import success
from src.cli.app import cli
from src.cli.benchmarking import run_benchmark


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--steps",
    type=click.IntRange(min=1),
    default=100,
    show_default=True,
    help="Time steps advanced per timed execution (the scan length that gets compiled).",
)
@click.option(
    "--warmup",
    type=click.IntRange(min=0),
    default=1,
    show_default=True,
    help="Untimed executions before sampling, to settle allocators and caches.",
)
@click.option(
    "--repeats",
    type=click.IntRange(min=1),
    default=3,
    show_default=True,
    help="Timed samples. Every repeat restarts from the same state, so all samples do equal work.",
)
@click.option(
    "--breakdown",
    is_flag=True,
    help=(
        "Also time an ablation ladder (optimiser off, then plain multiphase step) "
        "to attribute the per-step cost to each layer."
    ),
)
@click.option(
    "--io",
    "with_io",
    is_flag=True,
    help="Also time the loop with streaming snapshots attached, to price the ordered host callback.",
)
@click.option(
    "--profile",
    "profile_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Wrap the steady-state measurement in jax.profiler.trace, writing to this directory.",
)
@click.option(
    "--json",
    "json_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Destination for the JSON record (default: <results_dir>/benchmarks/<label>_<backend>.json).",
)
@click.option(
    "--label",
    default=None,
    help="Name used in the default JSON filename (default: slugified simulation_name).",
)
@click.option(
    "--override",
    "overrides",
    multiple=True,
    help="Override config values using path=value (repeatable), e.g. --override grid_shape=[512,512]",
)
@cli_command(title="Benchmark", interrupt_message="Benchmark interrupted.")
def benchmark(
    config_path: str,
    steps: int,
    warmup: int,
    repeats: int,
    breakdown: bool,
    with_io: bool,
    profile_dir: str | None,
    json_path: str | None,
    label: str | None,
    overrides: tuple[str, ...],
) -> None:
    """Measure where CONFIG_PATH's time goes, on whichever backend JAX resolves.

    Reports setup, trace and compile time separately from steady-state
    throughput, so a slow number can be attributed rather than guessed at, and
    writes a JSON record carrying the backend and device identity — which is
    what makes a CPU run and a GPU run comparable after the fact.

    Nothing here runs the production plotting or surface-tension paths; only the
    time loop is measured.
    """
    run_benchmark(
        config_path,
        overrides,
        steps=steps,
        warmup=warmup,
        repeats=repeats,
        breakdown=breakdown,
        with_io=with_io,
        profile_dir=profile_dir,
        json_path=json_path,
        label=label,
    )
    success("Benchmark complete!")
