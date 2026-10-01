"""The ``tud-lbm calibration`` commands, end to end on synthetic run directories."""

from __future__ import annotations
import shutil
from typing import TYPE_CHECKING
import numpy as np
import pytest
from src.cli.commands import cli
from src.config.adapter_toml import TomlAdapter
from src.config.run_config import CONFIG_FILENAME
from src.config.run_config import DATA_DIRNAME
from src.config.run_config import PHYSICAL_PARAMETERS_FILENAME
from src.config.simulation_config import SimulationConfig
from src.simulation_io.analysis.surface_tension import surface_tension as st

if TYPE_CHECKING:
    from pathlib import Path
    from click.testing import CliRunner

_SIGMA = 0.0123


@pytest.fixture(autouse=True)
def _isolate_field_cache(tmp_path, monkeypatch):
    """Keep the density fields a collection stores out of the real data root."""
    monkeypatch.setattr(st, "_FIELDS_CACHE_DIR", tmp_path / "field_cache")


def _config(**overrides: object) -> SimulationConfig:
    params: dict[str, object] = {
        "sim_type": "multiphase",
        "simulation_name": "drop",
        "grid_shape": (48, 32),
        "tau": 0.99,
        "nt": 3,
        "eos": "carnahan-starling",
        "kappa": 0.01,
        "rho_l": 0.4,
        "rho_v": 0.02,
        "interface_width": 4,
        "a_eos": 0.5,
        "b_eos": 4.0,
        "r_eos": 1.0,
        "t_eos": 0.05,
    }
    params.update(overrides)
    return SimulationConfig(**params)  # ty: ignore[invalid-argument-type]


def _write_config(path: Path, config: SimulationConfig) -> Path:
    TomlAdapter().save(config, str(path))
    return path


def _finish_sweep_run(root: Path, staged_toml: Path) -> None:
    """Turn a staged sweep config into the run directory a finished run leaves."""
    config = TomlAdapter().load(str(staged_toml))
    run_dir = root / "2026-01-01" / f"00-00-00_{config.simulation_name}"
    (run_dir / DATA_DIRNAME).mkdir(parents=True)
    _write_config(run_dir / CONFIG_FILENAME, config)

    side = st._CALIBRATION_SIDE
    radius = config.initialisation["radii"][0] * side
    offsets = np.arange(side) - side // 2
    inside = np.hypot(offsets[:, None], offsets[None, :]) <= radius
    np.savez(
        run_dir / DATA_DIRNAME / f"timestep_{config.nt}.npz",
        pressure=np.where(inside, _SIGMA / radius, 0.0)[:, :, None, None, None],
        rho=np.where(inside, 0.4, 0.02)[:, :, None, None, None],
    )


def test_stage_writes_one_sweep_per_fluid(runner: CliRunner, tmp_path: Path):
    """Two configs of one fluid share a sweep, whatever their grids."""
    first = _write_config(tmp_path / "a.toml", _config(grid_shape=(48, 32)))
    second = _write_config(tmp_path / "b.toml", _config(grid_shape=(96, 32), simulation_name="other"))
    staged = tmp_path / "staged"

    result = runner.invoke(cli, ["calibration", "stage", str(first), str(second), "--out-dir", str(staged)])

    assert result.exit_code == 0, result.output
    digest = st.sweep_digest(_config())
    assert sorted(p.name for p in staged.iterdir()) == [f"st_{digest}_r{i}.toml" for i in range(st._N_RADII)]


def test_stage_skips_a_calibrated_fluid_and_a_closed_form_eos(runner: CliRunner, tmp_path: Path):
    calibrated = _config()
    radii = st.sweep_radii()
    st._store_cache(st._cache_key(calibrated), radii, _SIGMA / radii, _SIGMA, st._CALIBRATION_GRID_SHAPE)
    double_well = _config(eos="double-well", a_eos=None, b_eos=None, r_eos=None, t_eos=None)
    paths = [_write_config(tmp_path / "a.toml", calibrated), _write_config(tmp_path / "b.toml", double_well)]
    staged = tmp_path / "staged"

    result = runner.invoke(cli, ["calibration", "stage", *map(str, paths), "--out-dir", str(staged)])

    assert result.exit_code == 0, result.output
    assert "nothing staged" in result.output
    assert not staged.exists()


def test_stage_covers_every_fluid_of_a_parameter_sweep(runner: CliRunner, tmp_path: Path):
    path = _write_config(tmp_path / "sweep.toml", _config())
    path.write_text(path.read_text().replace("kappa = 0.01", "kappa = [0.01, 0.02]"))
    staged = tmp_path / "staged"

    result = runner.invoke(cli, ["calibration", "stage", str(path), "--out-dir", str(staged)])

    assert result.exit_code == 0, result.output
    digests = {st.sweep_digest(_config(kappa=kappa)) for kappa in (0.01, 0.02)}
    assert {p.name.split("_")[1] for p in staged.iterdir()} == digests
    assert len(list(staged.iterdir())) == 2 * st._N_RADII


def test_stage_closed_form_flag_stages_a_verification_sweep(runner: CliRunner, tmp_path: Path):
    double_well = _config(eos="double-well", a_eos=None, b_eos=None, r_eos=None, t_eos=None)
    path = _write_config(tmp_path / "dw.toml", double_well)
    staged = tmp_path / "staged"

    result = runner.invoke(cli, ["calibration", "stage", str(path), "--out-dir", str(staged), "--closed-form"])

    assert result.exit_code == 0, result.output
    assert len(list(staged.iterdir())) == st._N_RADII


def test_stage_collect_refresh_round_trip(runner: CliRunner, tmp_path: Path):
    """The whole loop: stage, (run), collect into the cache, refresh an earlier run."""
    config = _config()
    source = _write_config(tmp_path / "source.toml", config)
    staged = tmp_path / "staged"
    results = tmp_path / "results"

    # A run of that fluid that finished before its sigma was known.
    earlier_run = results / "2026-01-01" / "00-00-00_drop"
    earlier_run.mkdir(parents=True)
    _write_config(earlier_run / CONFIG_FILENAME, config)
    config_before = (earlier_run / CONFIG_FILENAME).read_text()

    assert runner.invoke(cli, ["calibration", "stage", str(source), "--out-dir", str(staged)]).exit_code == 0
    for staged_toml in sorted(staged.iterdir()):
        _finish_sweep_run(results, staged_toml)

    collected = runner.invoke(cli, ["calibration", "collect", str(results)])
    assert collected.exit_code == 0, collected.output
    assert st.cached_surface_tension(config) == pytest.approx(_SIGMA, rel=1e-9)

    # A second collection leaves the stored measurement alone.
    again = runner.invoke(cli, ["calibration", "collect", str(results)])
    assert again.exit_code == 0, again.output
    assert "already in the cache" in again.output

    # The density fields are machine-local and feed only the snapshot figures.
    # Refresh as a machine that pulled the cache from git would: without them.
    shutil.rmtree(tmp_path / "field_cache")

    refreshed = runner.invoke(cli, ["calibration", "refresh", str(results)])
    assert refreshed.exit_code == 0, refreshed.output
    overview = (earlier_run / PHYSICAL_PARAMETERS_FILENAME).read_text()
    assert "measured, Young–Laplace" in overview
    # Sigma is linked through the cache; the run's config is not rewritten.
    assert (earlier_run / CONFIG_FILENAME).read_text() == config_before


def test_collect_reports_an_incomplete_sweep_without_caching(runner: CliRunner, tmp_path: Path):
    config = _config()
    source = _write_config(tmp_path / "source.toml", config)
    staged = tmp_path / "staged"
    results = tmp_path / "results"
    runner.invoke(cli, ["calibration", "stage", str(source), "--out-dir", str(staged)])
    for staged_toml in sorted(staged.iterdir())[:2]:
        _finish_sweep_run(results, staged_toml)

    result = runner.invoke(cli, ["calibration", "collect", str(results)])

    assert result.exit_code == 0, result.output
    assert "2/5" in result.output
    assert st.cached_surface_tension(config) is None
