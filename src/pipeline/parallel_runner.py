"""Parallel simulation runner for parameter sweeps.

Executes multiple SimulationConfig objects in parallel using process
pooling, with progress tracking and error aggregation.

Example usage::

    from config.adapter_toml import TomlAdapter
    from config.array_expansion import expand_config
    from runner.parallel_runner import run_parallel_simulations

    adapter = TomlAdapter()
    config_dict = adapter.load_raw("config_parallel.toml")
    configs, metadata = expand_config(config_dict)

    results = run_parallel_simulations(
        configs,
        max_workers=4,
        verbose=True,
    )

    for result in results:
        print(f"Simulation {result.index}: {result.status}")
"""

from __future__ import annotations
import json
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from dataclasses import replace
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from src.config.simulation_config import SimulationConfig


@dataclass(frozen=True)
class SimulationResult:
    """Result of a single simulation execution.

    Attributes:
        index: Position in the parameter sweep (0-based).
        config: The :class:`SimulationConfig` that was run.
        status: One of ``"success"``, ``"failed"``, ``"skipped"``.
        output_dir: Path to results directory (if successful).
        parameters: Dict of parameter values for this simulation (if array sweep).
        error: Exception message (if status == "failed").
        duration: Elapsed time in seconds (if completed).
    """

    index: int
    config: SimulationConfig
    status: str
    output_dir: str | None = None
    parameters: dict[str, Any] | None = None
    error: str | None = None
    duration: float = 0.0


def run_single_simulation(
    index: int,
    config: SimulationConfig,
    parameters: dict[str, Any] | None = None,
    setup_fn: Callable | None = None,
    run_fn: Callable | None = None,
) -> SimulationResult:
    """Execute a single simulation in a worker process.

    This function is pickled and executed in a separate process pool.

    Args:
        index: Position in the sweep (for identification).
        config: The :class:`SimulationConfig` to execute.
        parameters: Dict of parameter values (used for logging/naming).
        setup_fn: Callable that builds setup from config; defaults to
            :func:`setup.simulation_setup.build_setup`.
        run_fn: Callable that runs the simulation; defaults to the
            functional runner from :mod:`runner.run`.

    Returns:
        :class:`SimulationResult` with status and output path.
    """
    import time
    from src.config.jax_config import configure_jax
    from src.pipeline.runner import init_state
    from src.pipeline.runner import run
    from src.pipeline.setup import build_setup
    from src.simulation_io import SimulationIO

    # Configure JAX in this worker process
    configure_jax()

    start = time.perf_counter()

    try:
        # Use provided functions or defaults
        if setup_fn is None:
            setup_fn = build_setup
        if run_fn is None:
            run_fn = run

        # Build simulation setup
        setup = setup_fn(config)

        # Initialize state
        state = init_state(setup)

        # Create unique simulation name including parameter values
        if parameters:
            params_str = ", ".join(f"{k}={v}" for k, v in sorted(parameters.items()))
            unique_sim_name = f"{config.simulation_name} [{params_str}]"
        else:
            unique_sim_name = f"{config.simulation_name} [sim_{index}]"

        # Create I/O handler for this simulation with unique name
        io = SimulationIO(
            base_dir=config.results_dir,
            config=config,
            simulation_name=unique_sim_name,
            output_format=config.output_format,
        )

        from src.simulation_io.analysis.surface_tension import record_surface_tension

        config = record_surface_tension(config, io.run_dir)

        # Run simulation with streaming I/O
        _, _ = run_fn(
            setup,
            state,
            nt=config.nt,
            save_interval=config.save_interval,
            io_handler=io,
            skip_interval=config.skip_interval,
            save_fields=tuple(config.save_fields) if config.save_fields else None,
        )

        # Store the resolved output directory in the config
        config = replace(config, output_dir=str(io.run_dir))

        duration = time.perf_counter() - start

        return SimulationResult(
            index=index,
            config=config,
            status="success",
            output_dir=str(io.run_dir),
            parameters=parameters,
            duration=duration,
        )

    except Exception as e:  # noqa: BLE001
        duration = time.perf_counter() - start
        error_msg = f"{type(e).__name__}: {e!s}\n{traceback.format_exc()}"
        return SimulationResult(
            index=index,
            config=config,
            status="failed",
            parameters=parameters,
            error=error_msg,
            duration=duration,
        )


def run_parallel_simulations(
    configs: list[SimulationConfig],
    *,
    max_workers: int | None = None,
    parameters_list: list[dict[str, Any]] | None = None,
    verbose: bool = False,
    continue_on_error: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
    setup_fn: Callable | None = None,
    run_fn: Callable | None = None,
) -> list[SimulationResult]:
    """Execute multiple simulations in parallel.

    Args:
        configs: List of :class:`SimulationConfig` objects to execute.
        max_workers: Max number of worker processes. Defaults to CPU count.
        parameters_list: Optional list of dicts mapping parameter names to
            values (for sweep identification/logging). Should match
            length of *configs*.
        verbose: If True, print progress and status updates.
        continue_on_error: If False, stop execution on first failure.
        progress_callback: Optional callable(completed, total) for
            progress tracking.
        setup_fn: Optional custom setup function (default: build_setup).
        run_fn: Optional custom run function (default: run).

    Returns:
        List of :class:`SimulationResult` objects, one per config.

    Raises:
        RuntimeError: If any simulation fails and *continue_on_error*
            is False.
    """
    if not configs:
        return []

    n_configs = len(configs)
    params_iter: list[dict[str, Any] | None]
    if parameters_list is None:
        params_iter = [None] * n_configs
    elif len(parameters_list) != n_configs:
        msg = f"parameters_list length ({len(parameters_list)}) must match configs length ({n_configs})"
        raise ValueError(
            msg,
        )
    else:
        params_iter = cast("list[dict[str, Any] | None]", parameters_list)

    results: list[SimulationResult] = []
    futures_to_idx: dict[Any, int] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, (config, params) in enumerate(zip(configs, params_iter, strict=False)):
            future = executor.submit(
                run_single_simulation,
                index=idx,
                config=config,
                parameters=params,
                setup_fn=setup_fn,
                run_fn=run_fn,
            )
            futures_to_idx[future] = idx

        results = _collect_results(
            executor,
            futures_to_idx,
            n_configs=n_configs,
            verbose=verbose,
            progress_callback=progress_callback,
            continue_on_error=continue_on_error,
        )

    results.sort(key=lambda r: r.index)
    generate_plots(results, verbose=verbose)

    if verbose:
        n_success = sum(1 for r in results if r.status == "success")
        n_failed = sum(1 for r in results if r.status == "failed")
        print(f"Completed {n_configs} simulations: {n_success} succeeded, {n_failed} failed.")

    return results


def _collect_results(
    executor: ProcessPoolExecutor,
    futures_to_idx: dict[Any, int],
    *,
    n_configs: int,
    verbose: bool,
    progress_callback: Callable[[int, int], None] | None,
    continue_on_error: bool,
) -> list[SimulationResult]:
    """Collect futures as they complete, with optional progress reporting."""
    results: list[SimulationResult] = []
    for completed, future in enumerate(as_completed(futures_to_idx.keys()), start=1):
        result = future.result()
        results.append(result)

        if verbose:
            print_result_line(result, completed, n_configs)
        if progress_callback:
            progress_callback(completed, n_configs)
        if result.status == "failed" and not continue_on_error:
            executor.shutdown(wait=False)
            msg = f"Simulation {result.index} failed (continue_on_error=False): {result.error}"
            raise RuntimeError(
                msg,
            )

    return results


def print_result_line(result: SimulationResult, completed: int, total: int) -> None:
    """Print a single progress line for a completed simulation."""
    status_icon = "✓" if result.status == "success" else "✗"
    params_str = f" [{', '.join(f'{k}={v}' for k, v in result.parameters.items())}]" if result.parameters else ""
    error_str = f" - {result.error.split(chr(10))[0]}" if result.status == "failed" and result.error is not None else ""

    print(f"{status_icon} Sim {result.index}{params_str} ({completed}/{total}) [{result.duration:.2f}s]{error_str}")


def generate_plots(results: list[SimulationResult], *, verbose: bool) -> None:
    """Generate plots for all successful simulations that request them."""
    from src.simulation_io.plotting import FigureBuilder

    for result in results:
        if result.status != "success" or not result.config.plot_fields:
            continue
        try:
            if result.output_dir is None:
                continue
            FigureBuilder(result.config, result.output_dir).build_all()
            if verbose:
                print(f"Generated plots for simulation {result.index}")
        except Exception as e:  # noqa: BLE001
            print(f"Failed to generate plots for simulation {result.index}: {e!s}")


def save_sweep_log(
    results: list[SimulationResult],
    output_dir: str | Path,
) -> None:
    """Save a JSON manifest describing the parameter sweep.

    Args:
        results: List of :class:`SimulationResult` objects.
        output_dir: Directory to save ``sweep_manifest.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "sweep_id": str(uuid.uuid4()),
        "timestamp": datetime.now(UTC).isoformat(),
        "total_simulations": len(results),
        "successful": sum(1 for r in results if r.status == "success"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "simulations": [
            {
                "index": r.index,
                "status": r.status,
                "output_dir": r.output_dir,
                "parameters": r.parameters,
                "duration_sec": r.duration,
                "error": r.error,
            }
            for r in results
        ],
    }

    manifest_path = output_dir / f"sweep_log_{manifest['sweep_id']}.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
