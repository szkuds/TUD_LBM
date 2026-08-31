"""Timing harness behind ``tud-lbm benchmark``.

The whole ``nt``-step time loop compiles to a single XLA program
(:func:`src.pipeline.runner.run`), so there is nothing per-operator left for
Python to time.  This module therefore measures the four things that *are*
separable, and attributes the rest by ablation:

1. **setup** — ``build_setup`` + ``init_state``, host-side and backend-independent.
2. **trace / compile** — timed as two distinct stages via ``lower()`` then
   ``compile()``.  Compile cost is disproportionate for hysteresis runs, whose
   graph carries six ``lax.while_loop``s.
3. **steady state** — repeated executions of the compiled program, from which
   MLUPS follows.
4. **I/O** — the same measurement with a real :class:`SimulationIO` attached, so
   the delta prices the ordered host callback in
   :mod:`src.simulation_io.callbacks`.

``--breakdown`` then peels layers off the *same* setup rather than timing
separate configs, so the differences are attributable.  See
:func:`_breakdown_variants` for what each rung removes.
"""

from __future__ import annotations
import json
import os
import platform
import statistics
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
import jax
import jax.numpy as jnp
from src.cli._console import console

if TYPE_CHECKING:
    from collections.abc import Callable
    from src.config import SimulationConfig
    from src.pipeline.setup import SimulationSetup
    from src.pipeline.state.state import State

#: Sub-directory of ``results_dir`` that JSON records are written to.  Never
#: inside the repo — ``results_dir`` defaults to ``TUD_LBM_DATA_DIR``.
_BENCHMARK_SUBDIR = "benchmarks"

_MICRO = 1e6


# ── Result containers ────────────────────────────────────────────────


@dataclass(frozen=True)
class Measurement:
    """Wall-clock samples for one repeated measurement.

    Attributes:
        label: Human-readable name of what was measured.
        samples: Per-repeat elapsed seconds, in acquisition order.
        steps: Time steps advanced by a single sample.
        cells: ``nx * ny * nz`` for the benchmarked grid.
    """

    label: str
    samples: tuple[float, ...]
    steps: int
    cells: int

    @property
    def best(self) -> float:
        """Fastest sample in seconds — the least noise-contaminated estimate."""
        return min(self.samples)

    @property
    def median(self) -> float:
        """Median sample in seconds."""
        return statistics.median(self.samples)

    @property
    def spread(self) -> float:
        """``(max - min) / min``, the relative spread across repeats."""
        return (max(self.samples) - self.best) / self.best if self.best > 0 else 0.0

    @property
    def per_step_ms(self) -> float:
        """Milliseconds per time step, from the fastest sample."""
        return 1e3 * self.best / self.steps

    @property
    def mlups(self) -> float:
        """Million lattice updates per second, from the fastest sample."""
        return self.cells * self.steps / (self.best * _MICRO) if self.best > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of this measurement."""
        return {
            "label": self.label,
            "samples_s": list(self.samples),
            "steps": self.steps,
            "cells": self.cells,
            "best_s": self.best,
            "median_s": self.median,
            "spread": self.spread,
            "per_step_ms": self.per_step_ms,
            "mlups": self.mlups,
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """Everything one ``tud-lbm benchmark`` invocation measured."""

    environment: dict[str, Any]
    config: dict[str, Any]
    setup_s: float
    trace_s: float
    compile_s: float
    steady: Measurement
    cost: dict[str, Any] | None = None
    memory: dict[str, Any] | None = None
    io: Measurement | None = None
    breakdown: tuple[Measurement, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the whole benchmark."""
        return {
            "environment": self.environment,
            "config": self.config,
            "setup_s": self.setup_s,
            "trace_s": self.trace_s,
            "compile_s": self.compile_s,
            "steady": self.steady.to_dict(),
            "cost": self.cost,
            "memory": self.memory,
            "io": self.io.to_dict() if self.io is not None else None,
            "breakdown": [m.to_dict() for m in self.breakdown],
        }


# ── Provenance ───────────────────────────────────────────────────────


def _environment() -> dict[str, Any]:
    """Capture the machine and backend facts needed to compare two records.

    Without these a CPU JSON and a GPU JSON are not comparable — the backend,
    device kind and x64 flag are exactly what the comparison is about.
    """
    devices = jax.devices()
    return {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in devices],
        "device_kind": devices[0].device_kind if devices else "unknown",
        "device_count": len(devices),
        # read() rather than the dynamic jax.config.jax_enable_x64 attribute,
        # which the type checker cannot resolve.
        "x64": bool(jax.config.read("jax_enable_x64")),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }


def _fingerprint(config: SimulationConfig) -> dict[str, Any]:
    """Capture the config fields that determine how much work a step is."""
    hysteresis = config.hysteresis_config or {}
    return {
        "simulation_name": config.simulation_name,
        "sim_type": config.sim_type,
        "grid_shape": list(config.grid_shape),
        "lattice_type": config.lattice_type,
        "collision_scheme": config.collision_scheme,
        "eos": config.eos,
        "tau": config.tau,
        "max_iterations": hysteresis.get("max_iterations"),
        "trial_steps": hysteresis.get("trial_steps"),
        "loss_tol": hysteresis.get("loss_tol"),
    }


# ── Compilation and timing primitives ────────────────────────────────


def _make_stepper(
    setup: SimulationSetup,
    steps: int,
    do_save: Callable[[State, jnp.ndarray], None] | None = None,
) -> Any:  # noqa: ANN401 — jax.jit returns an untyped wrapper
    """Build a jitted ``state -> final_state`` advancing *steps* time steps.

    Deliberately mirrors :func:`src.pipeline.runner.run`'s streaming path — the
    same :func:`~src.pipeline.runner._make_scan_body` body and the same
    ``jnp.arange(steps)`` scan driver — so what is timed is what production
    executes, wart for wart.

    Args:
        setup: The (possibly ablated) simulation setup to time.
        steps: Scan length.  Static, so each distinct value costs a compile.
        do_save: Optional snapshot callback, for the ``--io`` measurement.

    Returns:
        A jitted callable taking the initial state and returning the final one.

    Raises:
        TypeError: If *setup* has no ``step_fn``.
    """
    from src.pipeline.runner import _make_scan_body

    step_fn = setup.step_fn
    if step_fn is None:
        msg = "step_fn is required in SimulationSetup to benchmark it"
        raise TypeError(msg)

    body = _make_scan_body(partial(step_fn, setup), do_save=do_save, collect=False)

    def stepper(state: State) -> State:
        final_state, _ = jax.lax.scan(body, state, jnp.arange(steps))
        return final_state

    return jax.jit(stepper)


def _compile_timed(stepper: Any, state: State) -> tuple[Any, float, float]:  # noqa: ANN401
    """Lower then compile *stepper*, timing the two stages separately.

    Returns the ``Compiled`` object rather than the jitted wrapper so that the
    program subsequently executed is provably the one whose compilation was
    just timed, and so ``cost_analysis``/``memory_analysis`` are available.

    Returns:
        ``(compiled, trace_seconds, compile_seconds)``.
    """
    start = time.perf_counter()
    lowered = stepper.lower(state)
    trace_s = time.perf_counter() - start

    start = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - start

    return compiled, trace_s, compile_s


def _measure(
    compiled: Any,  # noqa: ANN401
    state: State,
    *,
    label: str,
    steps: int,
    cells: int,
    repeats: int,
    warmup: int,
) -> Measurement:
    """Time *repeats* executions, each starting from the same *state*.

    Repeats restart from the identical initial state rather than chaining. That
    is load-bearing for hysteresis: its ``while_loop`` trip count depends on the
    physics state (``hysteresis.py`` ``cond_fn`` exits on ``loss <= loss_tol``),
    so chained repeats would each measure a *different* amount of work and the
    spread would be physics, not noise.

    Args:
        compiled: The compiled program from :func:`_compile_timed`.
        state: Initial state; reused unchanged for every repeat.
        label: Name recorded on the resulting :class:`Measurement`.
        steps: Time steps per execution.
        cells: ``nx * ny * nz``, for the MLUPS derivation.
        repeats: Number of timed samples.
        warmup: Untimed executions run first, to settle allocators and caches.

    Returns:
        The collected :class:`Measurement`.
    """
    for _ in range(warmup):
        jax.block_until_ready(compiled(state))

    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(compiled(state))
        samples.append(time.perf_counter() - start)

    return Measurement(label=label, samples=tuple(samples), steps=steps, cells=cells)


def _static_analysis(compiled: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:  # noqa: ANN401
    """Extract XLA's own FLOP/byte and peak-memory estimates, if available.

    These are noise-free and require no execution, so they place the case on a
    roofline (arithmetic intensity = FLOPs / bytes) *before* a GPU is involved:
    a bandwidth-bound case is one a GPU rewards, a latency-bound one may not be.

    Both analyses are backend- and version-dependent and are documented as
    best-effort, so failure degrades to ``None`` rather than failing the run.
    """
    cost: dict[str, Any] | None = None
    memory: dict[str, Any] | None = None

    try:
        raw = compiled.cost_analysis()
    except Exception:  # noqa: BLE001 — best-effort diagnostic, never fatal
        raw = None
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if isinstance(raw, dict):
        flops = raw.get("flops")
        bytes_accessed = raw.get("bytes accessed")
        cost = {"flops": flops, "bytes_accessed": bytes_accessed}
        if flops and bytes_accessed:
            cost["arithmetic_intensity"] = flops / bytes_accessed

    try:
        analysis = compiled.memory_analysis()
    except Exception:  # noqa: BLE001 — best-effort diagnostic, never fatal
        analysis = None
    if analysis is not None:
        memory = {
            name: getattr(analysis, name, None)
            for name in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes")
        }

    return cost, memory


# ── Ablation ─────────────────────────────────────────────────────────


def _breakdown_variants(setup: SimulationSetup) -> list[tuple[str, SimulationSetup, str]]:
    """Build the ablation ladder below the configured step, cheapest last.

    Each rung removes exactly one layer from the *same* setup, so successive
    differences in wall time attribute cost to that layer:

    * ``no_optimiser`` — ``wetting_fn=None``.  ``WettingExtraStatePlugin.update_state``
      returns early on that, so the Adam/AD hysteresis optimiser — and with it
      ``_trial_step``'s chained ``_multiphase_pipeline`` passes — never enters
      the graph.  Nothing else changes.
    * ``plain_step`` — additionally swaps in the plain ``multiphase`` step
      operator, dropping the per-step contact-angle/contact-line measurement and
      the live re-binding of wetting parameters.  Note the wetting *correction*
      itself survives: it is baked into ``gradient_density``/``laplacian_density``
      at setup time (``build_diff_ops``), so this rung is not "wetting off".

    Rungs identical to the configured setup are omitted, so a plain
    ``multiphase`` config yields an empty ladder and a single-phase config never
    reaches here.

    Args:
        setup: The setup as configured.

    Returns:
        ``(label, ablated_setup, what_it_removes)`` triples.
    """
    from src.operators.step import build_step_fn

    if "multiphase" not in setup.config.sim_type:
        return []

    variants: list[tuple[str, SimulationSetup, str]] = []

    if setup.wetting_fn is not None:
        variants.append(
            (
                "no_optimiser",
                setup._replace(wetting_fn=None),
                "hysteresis Adam/AD optimiser",
            )
        )

    if setup.config.sim_type != "multiphase":
        variants.append(
            (
                "plain_step",
                setup._replace(wetting_fn=None, step_fn=build_step_fn("multiphase")),
                "contact-angle/contact-line measurement",
            )
        )

    return variants


# ── Orchestration ────────────────────────────────────────────────────


def _load_benchmark_config(config_path: str, overrides: tuple[str, ...]) -> SimulationConfig:
    """Load and expand *config_path*, rejecting sweeps.

    A sweep expands to several configs with different work per step, which the
    single-number output of this command cannot represent honestly.

    Raises:
        click.UsageError: If the config expands to more than one simulation.
    """
    import click
    from src.cli.config_loading import _expand_raw_config
    from src.cli.config_loading import _load_raw_config

    raw_config = _load_raw_config(config_path, overrides)
    _configs, config, sweep_metadata, _params = _expand_raw_config(raw_config)
    if sweep_metadata is not None or config is None:
        msg = "benchmark does not support parameter sweeps; benchmark one grid at a time"
        raise click.UsageError(msg)
    return config


def _resolve_json_path(config: SimulationConfig, json_path: str | None, label: str, backend: str) -> Path:
    """Decide where the JSON record goes.

    Defaults under ``results_dir`` (i.e. ``TUD_LBM_DATA_DIR``) so a benchmark
    never dirties the checkout, matching the rule the surface-tension cache
    already follows for its field snapshots.
    """
    if json_path is not None:
        return Path(json_path).expanduser()
    base = Path(config.results_dir).expanduser() / _BENCHMARK_SUBDIR
    return base / f"{label}_{backend}.json"


def _slugify(name: str) -> str:
    """Reduce *name* to a filesystem-safe lowercase slug."""
    slug = "".join(char if char.isalnum() else "_" for char in name.lower())
    return "_".join(part for part in slug.split("_") if part) or "benchmark"


def _render_phases(result: BenchmarkResult) -> None:
    """Print the one-off costs: setup, tracing and XLA compilation.

    Kept separate from throughput because compile time is a single up-front cost
    for the whole loop and must never be amortised into a per-step figure.
    """
    from rich.table import Table

    phases = Table(title="Phases", header_style="bold cyan")
    phases.add_column("Phase")
    phases.add_column("Seconds", justify="right")
    phases.add_row("setup (build + init)", f"{result.setup_s:.3f}")
    phases.add_row("trace (lower)", f"{result.trace_s:.3f}")
    phases.add_row("compile (XLA)", f"{result.compile_s:.3f}")
    console.print(phases)


def _render_throughput(result: BenchmarkResult) -> None:
    """Print per-step cost and MLUPS for the full loop and every variant."""
    from rich.table import Table

    throughput = Table(title="Throughput", header_style="bold cyan")
    throughput.add_column("Variant")
    throughput.add_column("ms/step", justify="right")
    throughput.add_column("MLUPS", justify="right")
    throughput.add_column("spread", justify="right")
    throughput.add_column("vs full", justify="right")

    baseline = result.steady.best
    rows = [result.steady, *result.breakdown]
    if result.io is not None:
        rows.insert(1, result.io)
    for measurement in rows:
        ratio = measurement.best / baseline if baseline > 0 else 0.0
        throughput.add_row(
            measurement.label,
            f"{measurement.per_step_ms:.4g}",
            f"{measurement.mlups:.4g}",
            f"{measurement.spread:.1%}",
            f"{ratio:.3g}x",
        )
    console.print(throughput)

    if result.io is not None and baseline > 0:
        console.print(f"[dim]Streaming I/O overhead: {result.io.best / baseline - 1.0:+.1%} of the no-I/O loop[/dim]")


def _render_attribution(result: BenchmarkResult) -> None:
    """Print what each ablation rung removed, in ms/step and as a share."""
    from rich.table import Table

    if not result.breakdown:
        return

    console.print()
    attribution = Table(title="Attribution", header_style="bold cyan")
    attribution.add_column("Layer removed")
    attribution.add_column("ms/step saved", justify="right")
    attribution.add_column("share of full", justify="right")

    previous = result.steady
    for measurement in result.breakdown:
        saved = previous.per_step_ms - measurement.per_step_ms
        share = saved / result.steady.per_step_ms if result.steady.per_step_ms > 0 else 0.0
        attribution.add_row(measurement.label, f"{saved:.4g}", f"{share:.1%}")
        previous = measurement
    console.print(attribution)

    cheapest = result.breakdown[-1]
    if cheapest.best > 0:
        console.print(
            f"[dim]Configured step costs {result.steady.best / cheapest.best:.0f}x "
            f"the bare '{cheapest.label}' step.[/dim]"
        )


def _render_static(result: BenchmarkResult) -> None:
    """Print XLA's own FLOP/byte and peak-memory estimates.

    Arithmetic intensity is the roofline position: a low value means
    bandwidth-bound, which is the profile a GPU rewards.
    """
    if result.cost:
        console.print()
        intensity = result.cost.get("arithmetic_intensity")
        detail = f" · intensity {intensity:.2f} FLOP/byte" if intensity else ""
        console.print(
            f"[dim]XLA cost model: {result.cost.get('flops')} FLOPs, "
            f"{result.cost.get('bytes_accessed')} bytes accessed{detail}[/dim]"
        )
    if result.memory:
        peak = result.memory.get("temp_size_in_bytes")
        if peak:
            console.print(f"[dim]Peak temporaries: {peak / 1e9:.2f} GB[/dim]")


def _render(result: BenchmarkResult) -> None:
    """Print the benchmark result as rich tables."""
    env = result.environment
    console.print(
        f"[bold]{env['backend']}[/bold] · {env['device_kind']} × {env['device_count']} · "
        f"jax {env['jax_version']} · x64={env['x64']}"
    )
    cfg = result.config
    console.print(
        f"[dim]{cfg['sim_type']} · grid {cfg['grid_shape']} · {cfg['lattice_type']}/{cfg['collision_scheme']}[/dim]"
    )
    console.print()
    _render_phases(result)
    console.print()
    _render_throughput(result)
    _render_attribution(result)
    _render_static(result)


def run_benchmark(
    config_path: str,
    overrides: tuple[str, ...] = (),
    *,
    steps: int,
    warmup: int,
    repeats: int,
    breakdown: bool = False,
    with_io: bool = False,
    profile_dir: str | None = None,
    json_path: str | None = None,
    label: str | None = None,
) -> BenchmarkResult:
    """Benchmark one simulation config and report where its time goes.

    Args:
        config_path: TOML config to benchmark.  Must not be a sweep.
        overrides: ``--override KEY=VALUE`` strings, e.g. to vary ``grid_shape``.
        steps: Time steps per timed execution.
        warmup: Untimed executions before sampling begins.
        repeats: Timed samples, all from the same initial state.
        breakdown: Also time the ablation ladder from :func:`_breakdown_variants`.
        with_io: Also time the loop with streaming snapshots attached.
        profile_dir: If given, wrap the steady-state measurement in
            ``jax.profiler.trace`` for a per-HLO-op deep dive.
        json_path: Explicit destination for the JSON record.
        label: Name used in the default JSON filename.

    Returns:
        The populated :class:`BenchmarkResult`.
    """
    import contextlib
    from src.config.jax_config import configure_jax

    configure_jax()

    from src.pipeline.runner import init_state
    from src.pipeline.setup import build_setup

    config = _load_benchmark_config(config_path, overrides)

    start = time.perf_counter()
    setup = build_setup(config)
    state = init_state(setup)
    jax.block_until_ready(state.f)
    setup_s = time.perf_counter() - start

    nx, ny, nz = setup.grid_shape[0], setup.grid_shape[1], setup.grid_shape[2]
    cells = nx * ny * nz

    console.print(f"[cyan]Compiling[/cyan] {steps} steps on {jax.default_backend()}...")
    compiled, trace_s, compile_s = _compile_timed(_make_stepper(setup, steps), state)
    cost, memory = _static_analysis(compiled)

    console.print(f"[cyan]Measuring[/cyan] {repeats} repeats of {steps} steps ({warmup} warmup)...")
    tracer = jax.profiler.trace(profile_dir) if profile_dir is not None else contextlib.nullcontext()
    with tracer:
        steady = _measure(compiled, state, label="full", steps=steps, cells=cells, repeats=repeats, warmup=warmup)

    io_measurement = _measure_io(setup, state, steps=steps, cells=cells, repeats=repeats) if with_io else None

    breakdown_measurements = (
        _measure_breakdown(setup, state, steps=steps, cells=cells, repeats=repeats, warmup=warmup) if breakdown else ()
    )

    result = BenchmarkResult(
        environment=_environment(),
        config=_fingerprint(config),
        setup_s=setup_s,
        trace_s=trace_s,
        compile_s=compile_s,
        steady=steady,
        cost=cost,
        memory=memory,
        io=io_measurement,
        breakdown=breakdown_measurements,
    )

    console.print()
    _render(result)
    _persist(result, config, config_path, json_path=json_path, label=label)

    if profile_dir is not None:
        console.print(f"[dim]Profile trace written to {profile_dir}[/dim]")

    return result


def _measure_breakdown(
    setup: SimulationSetup,
    state: State,
    *,
    steps: int,
    cells: int,
    repeats: int,
    warmup: int,
) -> tuple[Measurement, ...]:
    """Time each rung of :func:`_breakdown_variants`, cheapest last."""
    measurements: list[Measurement] = []
    for name, variant, removes in _breakdown_variants(setup):
        console.print(f"[cyan]Ablating[/cyan] {name} (removes {removes})...")
        compiled, _trace_s, _compile_s = _compile_timed(_make_stepper(variant, steps), state)
        measurements.append(
            _measure(compiled, state, label=name, steps=steps, cells=cells, repeats=repeats, warmup=warmup)
        )
    return tuple(measurements)


def _persist(
    result: BenchmarkResult,
    config: SimulationConfig,
    config_path: str,
    *,
    json_path: str | None,
    label: str | None,
) -> Path:
    """Write the JSON record and report where it went."""
    destination = _resolve_json_path(
        config,
        json_path,
        label or _slugify(config.simulation_name or Path(config_path).stem),
        result.environment["backend"],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    console.print()
    console.print(f"[dim]Record written to {destination}[/dim]")
    return destination


def _measure_io(
    setup: SimulationSetup,
    state: State,
    *,
    steps: int,
    cells: int,
    repeats: int,
) -> Measurement:
    """Time the loop with streaming snapshots attached.

    The delta against the ``full`` measurement prices the ordered host callback
    in :mod:`src.simulation_io.callbacks`, which threads a token through every
    scan iteration whether or not that step saves.

    ``save_interval`` is clamped to *steps* so at least one snapshot is actually
    written; otherwise the measurement would price the token threading alone and
    silently omit the device-to-host copy.
    """
    from src.simulation_io import SimulationIO
    from src.simulation_io.callbacks import make_save_callback

    config = setup.config
    save_interval = min(config.save_interval, steps) if config.save_interval > 0 else steps
    console.print(f"[cyan]Measuring[/cyan] streaming I/O at save_interval={save_interval}...")

    io_handler = SimulationIO(
        base_dir=config.results_dir,
        config=config,
        simulation_name=f"{config.simulation_name} [benchmark-io]",
        output_format=config.output_format,
    )
    do_save = make_save_callback(
        io_handler,
        save_interval=save_interval,
        skip_interval=0,
        save_fields=tuple(config.save_fields) if config.save_fields else None,
    )
    compiled, _trace, _compile = _compile_timed(_make_stepper(setup, steps, do_save=do_save), state)
    console.print(f"[dim]I/O snapshots written to {io_handler.data_dir}[/dim]")
    return _measure(compiled, state, label="with I/O", steps=steps, cells=cells, repeats=repeats, warmup=1)
