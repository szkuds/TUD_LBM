"""Behavioural tests for ``tud-lbm benchmark``.

The command exists to make CPU and GPU numbers comparable after the fact, so
what matters is that the JSON record carries the backend identity, that the
ablation ladder isolates the layer it claims to, and that nothing lands in the
repo.
"""

from __future__ import annotations
import json
from typing import TYPE_CHECKING
import pytest
from src.cli.benchmarking import Measurement
from src.cli.benchmarking import _breakdown_variants
from src.cli.benchmarking import _environment
from src.cli.benchmarking import _fingerprint
from src.cli.benchmarking import _slugify
from src.cli.benchmarking import run_benchmark

if TYPE_CHECKING:
    from pathlib import Path

_KERNEL = """
[simulation_type]
simulation_name = "bench test"
type = "multiphase"
grid_shape = [32, 32]
lattice_type = "D2Q9"
tau = 0.99
nt = 20
init_type = "multiphase_bubbles"

[initialisation]
centres = [[0.5, 0.5]]
radii = [0.3]

[multiphase]
kappa = 0.017
rho_l = 1.0
rho_v = 0.33
interface_width = 4
eos = "double-well"

[output]
results_dir = "{results_dir}"
save_fields = ["rho", "u"]
"""

_HYSTERESIS = """
[simulation_type]
simulation_name = "bench hysteresis test"
type = "multiphase_hysteresis"
grid_shape = [32, 16]
lattice_type = "D2Q9"
tau = 0.99
nt = 20
init_type = "multiphase_bubbles"

[initialisation]
centres = [[0.5, 0.0]]
radii = [0.5]
dispersed = "liquid"

[multiphase]
kappa = 0.017
rho_l = 1.0
rho_v = 0.33
interface_width = 4
eos = "double-well"

[boundary_conditions]
left = "periodic"
right = "periodic"
top = "bounce-back"
bottom = "wetting"

[hysteresis]
ca_advancing = 120.0
ca_receding = 60.0
learning_rate = 0.01
max_iterations = 2
loss_tol = 0.0

[output]
results_dir = "{results_dir}"
save_fields = ["rho", "u"]
"""


def _write_config(tmp_path: Path, template: str, name: str = "config.toml") -> Path:
    path = tmp_path / name
    path.write_text(template.format(results_dir=tmp_path.as_posix()), encoding="utf-8")
    return path


# ── Derived metrics ──────────────────────────────────────────────────


def test_measurement_reports_the_fastest_sample():
    """The headline numbers come from the best sample, not the mean."""
    m = Measurement(label="full", samples=(0.4, 0.2, 0.3), steps=10, cells=100)

    assert m.best == pytest.approx(0.2)
    assert m.median == pytest.approx(0.3)
    assert m.per_step_ms == pytest.approx(20.0)
    assert m.mlups == pytest.approx(100 * 10 / (0.2 * 1e6))
    assert m.spread == pytest.approx((0.4 - 0.2) / 0.2)


def test_measurement_survives_a_zero_sample():
    """A clock that cannot resolve the work must not raise ZeroDivisionError."""
    m = Measurement(label="full", samples=(0.0,), steps=1, cells=1)

    assert m.mlups == 0.0
    assert m.spread == 0.0


def test_slugify_is_filesystem_safe():
    assert _slugify("Test Multiphase [64x64]") == "test_multiphase_64x64"
    assert _slugify("!!!") == "benchmark"


# ── Provenance ───────────────────────────────────────────────────────


def test_environment_records_what_makes_runs_comparable():
    """Backend and precision are the whole point of the record."""
    env = _environment()

    assert env["backend"]
    assert env["devices"]
    assert isinstance(env["x64"], bool)
    assert env["jax_version"]


def test_fingerprint_captures_hysteresis_work_per_step(tmp_path):
    """max_iterations and trial_steps determine step cost, so they must be recorded."""
    from src.cli.config_loading import _load_single_config

    config = _load_single_config(str(_write_config(tmp_path, _HYSTERESIS)))
    fingerprint = _fingerprint(config)

    assert fingerprint["sim_type"] == "multiphase_hysteresis"
    assert fingerprint["max_iterations"] == 2
    assert fingerprint["loss_tol"] == 0.0


# ── Ablation ladder ──────────────────────────────────────────────────


def test_breakdown_isolates_the_optimiser_for_hysteresis(tmp_path):
    """no_optimiser must null wetting_fn and change nothing else."""
    from src.cli.config_loading import _load_single_config
    from src.pipeline.setup import build_setup

    setup = build_setup(_load_single_config(str(_write_config(tmp_path, _HYSTERESIS))))
    variants = {name: ablated for name, ablated, _why in _breakdown_variants(setup)}

    assert set(variants) == {"no_optimiser", "plain_step"}
    assert variants["no_optimiser"].wetting_fn is None
    assert variants["no_optimiser"].step_fn is setup.step_fn
    assert variants["plain_step"].step_fn is not setup.step_fn


def test_breakdown_is_empty_for_a_plain_multiphase_run(tmp_path):
    """Every rung would equal the configured setup, so none are offered."""
    from src.cli.config_loading import _load_single_config
    from src.pipeline.setup import build_setup

    setup = build_setup(_load_single_config(str(_write_config(tmp_path, _KERNEL))))

    assert _breakdown_variants(setup) == []


# ── End to end ───────────────────────────────────────────────────────


def test_benchmark_writes_a_record_under_results_dir(tmp_path):
    """The JSON must land under results_dir — never in the checkout."""
    config_path = _write_config(tmp_path, _KERNEL)

    result = run_benchmark(str(config_path), steps=2, warmup=0, repeats=2)

    destination = tmp_path / "benchmarks" / f"bench_test_{result.environment['backend']}.json"
    assert destination.exists()

    record = json.loads(destination.read_text(encoding="utf-8"))
    assert record["steady"]["steps"] == 2
    assert len(record["steady"]["samples_s"]) == 2
    assert record["steady"]["cells"] == 32 * 32
    assert record["environment"]["backend"] == result.environment["backend"]
    assert record["config"]["grid_shape"] == [32, 32, 1]
    assert record["compile_s"] > 0


def test_benchmark_honours_an_explicit_json_destination(tmp_path):
    config_path = _write_config(tmp_path, _KERNEL)
    destination = tmp_path / "elsewhere" / "record.json"

    run_benchmark(str(config_path), steps=2, warmup=0, repeats=1, json_path=str(destination))

    assert destination.exists()


def test_benchmark_applies_overrides(tmp_path):
    """Grid sweeps are driven by --override, so it must reach the config."""
    config_path = _write_config(tmp_path, _KERNEL)

    result = run_benchmark(str(config_path), ("grid_shape=[16,16]",), steps=2, warmup=0, repeats=1)

    assert result.config["grid_shape"] == [16, 16, 1]
    assert result.steady.cells == 16 * 16


def test_benchmark_rejects_sweeps(tmp_path):
    """A sweep does different work per config; one number cannot describe it."""
    import click

    config_path = _write_config(tmp_path, _KERNEL)

    with pytest.raises(click.UsageError, match="does not support parameter sweeps"):
        run_benchmark(str(config_path), ("tau=[0.7,0.8]",), steps=2, warmup=0, repeats=1)


@pytest.mark.slow
def test_benchmark_breakdown_attributes_cost_to_the_optimiser(tmp_path):
    """The whole point: the hysteresis optimiser must dominate the step."""
    config_path = _write_config(tmp_path, _HYSTERESIS)

    result = run_benchmark(str(config_path), steps=2, warmup=0, repeats=1, breakdown=True)

    labels = [m.label for m in result.breakdown]
    assert labels == ["no_optimiser", "plain_step"]
    assert result.breakdown[0].best < result.steady.best
