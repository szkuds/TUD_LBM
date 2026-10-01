"""End-to-end surface-tension calibration over a real droplet sweep.

``tests/io/test_surface_tension_calibration.py`` covers staging, collection, the
caches and the artefact tree on synthetic run directories. This module closes
the loop those tests fake: the staged sweep configs are run through the
ordinary run pipeline (``build_setup`` -> ``init_state`` -> ``run`` with
streaming snapshots), their run directories are collected into the cache, and a
later run of the same fluid picks the measured sigma up from it.

**The value of sigma is deliberately not asserted.** A trustworthy number needs
the production sweep — a 301x301 box equilibrated for 200_000 steps. On a box
small enough to run in a test the droplet compresses the vapour it shares the
periodic domain with, and that finite-box pressure offset is larger than the
Laplace jump itself: the fitted slope comes out negative even though nothing is
broken. Shrinking the sweep is what makes this test affordable, so it asserts
the wiring plus the physics invariants a short run does satisfy — mass
conservation, a droplet that stays a droplet, finite fields — and leaves the
calibrated number to a real run.
"""

from __future__ import annotations
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
import numpy as np
import pytest
from src.config.run_config import DATA_DIRNAME
from src.config.run_config import PHYSICAL_PARAMETERS_FILENAME
from src.config.run_config import PLOTS_DIRNAME
from src.config.run_config import SNAPSHOTS_DIRNAME
from src.config.simulation_config import SimulationConfig
from src.simulation_io.analysis.surface_tension import surface_tension as st

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.integration, pytest.mark.slow]

#: Small enough to equilibrate three droplets in seconds, large enough that the
#: vapour corner sample points sit outside the biggest droplet. The fixture
#: pins the calibration box to this.
_GRID = 48
_N_RADII = 3
_N_ITERATIONS = 200


def _cs_config(**overrides) -> SimulationConfig:
    """A real Carnahan-Starling config whose own grid is *not* the calibration box.

    The EOS parameters are the coexistence pair from
    ``examples/config_cs_simple.toml`` — synthetic values equilibrate to a
    droplet that is barely denser than its vapour, which would make the
    "still a droplet" assertions below vacuous.
    """
    base: dict[str, Any] = {
        "sim_type": "multiphase",
        "grid_shape": (64, 32),
        "tau": 0.99,
        "nt": 3,
        "eos": "carnahan-starling",
        "kappa": 0.01,
        "rho_l": 12.18,
        "rho_v": 0.015,
        "interface_width": 5,
        "a_eos": 0.00031459670905604266,
        "b_eos": 0.1490857142857143,
        "r_eos": 1.0,
        "t_eos": 0.00039808421247983624,
    }
    base.update(overrides)
    return SimulationConfig(**base)


def _rho_2d(snapshot: Path) -> np.ndarray:
    """The (nx, ny) density field of a saved snapshot."""
    return np.asarray(np.load(snapshot)["rho"]).reshape(_GRID, _GRID)


def _initial_mass(config: SimulationConfig) -> float:
    """Total mass of the state the run pipeline starts *config* from."""
    from src.pipeline.runner import init_state
    from src.pipeline.setup import build_setup

    return float(np.asarray(init_state(build_setup(config)).rho).sum())


@pytest.fixture(scope="module")
def calibration(tmp_path_factory) -> SimpleNamespace:
    """Stage, run and collect one sweep for the whole module.

    Both caches are redirected into ``tmp``: ``_SHARED_CACHE_PATH`` is a
    git-tracked file and ``_FIELDS_CACHE_DIR`` resolves under the developer's
    real data root, and a test must dirty neither.
    """
    from src.cli.execution import _run_simulation

    tmp = tmp_path_factory.mktemp("surface_tension_e2e")
    root = tmp / "surface_tension"
    config = _cs_config()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(st, "_SHARED_CACHE_PATH", tmp / st._CACHE_FILENAME)
        mp.setattr(st, "_FIELDS_CACHE_DIR", tmp / "field_cache")
        mp.setattr(st, "SURFACE_TENSION_ROOT", root)
        mp.setattr(st, "_N_RADII", _N_RADII)
        mp.setattr(st, "_N_ITERATIONS", _N_ITERATIONS)
        # Production equilibrates 301x301 droplets; the confinement problem the
        # module docstring describes is why it does not size down, and why sigma
        # is not asserted here.
        mp.setattr(st, "_CALIBRATION_SIDE", _GRID)
        mp.setattr(st, "_CALIBRATION_GRID_SHAPE", (_GRID, _GRID, 1))

        # Before the sweep: the fluid is unknown, and nothing is measured.
        miss_dir = tmp / "miss"
        unchanged = st.record_surface_tension(config, miss_dir)

        sweep = st.calibration_configs(config)
        initial_masses = [_initial_mass(c) for c in sweep]
        for sweep_config in sweep:
            _run_simulation(sweep_config)

        # Read inside the patch context, where the box is the test grid.
        fluid_dir = root / st.fluid_label(config)
        sweep_run_dirs = st.find_sweep_runs(root)[fluid_dir]
        sigma = st.collect_calibration(sweep_run_dirs, out_dir=fluid_dir)

        # After: a run of the same fluid, on its own grid, resolves it.
        run_dir = tmp / "run"
        updated = st.record_surface_tension(config, run_dir)
        cache_key = st._cache_key(config)
        radii = st.sweep_radii()

    return SimpleNamespace(
        config=config,
        unchanged=unchanged,
        miss_dir=miss_dir,
        sweep=sweep,
        sweep_run_dirs=sweep_run_dirs,
        initial_masses=initial_masses,
        sigma=sigma,
        updated=updated,
        run_dir=run_dir,
        cache_key=cache_key,
        radii=radii,
        fields_cache_dir=tmp / "field_cache",
    )


def _final_snapshot(run_dir: Path) -> Path:
    return run_dir / DATA_DIRNAME / f"timestep_{_N_ITERATIONS}.npz"


def test_an_uncalibrated_fluid_runs_without_a_sweep(calibration):
    """The miss is reported, not measured: no artefact, the config untouched."""
    assert calibration.unchanged is calibration.config
    assert not calibration.miss_dir.exists()


def test_each_sweep_run_is_an_ordinary_run_directory(calibration):
    """One run per radius, each leaving exactly the final snapshot the fit reads."""
    assert len(calibration.sweep_run_dirs) == _N_RADII
    for run_dir in calibration.sweep_run_dirs:
        snapshots = sorted(p.name for p in (run_dir / DATA_DIRNAME).iterdir())
        assert snapshots == [f"timestep_{_N_ITERATIONS}.npz"]
        saved = np.load(_final_snapshot(run_dir))
        assert {"rho", "pressure"} <= set(saved.files)
        assert saved["pressure"].shape == (_GRID, _GRID, 1, 1, 1)


def test_collected_sigma_is_the_fit_of_the_runs_pressure_jumps(calibration):
    """The cached number is the slope through the jumps read off the snapshots."""
    delta_p = [
        st._pressure_jump(np.asarray(np.load(_final_snapshot(run_dir))["pressure"])[:, :, 0, 0, 0])
        for run_dir in calibration.sweep_run_dirs
    ]

    assert calibration.sigma is not None
    assert np.isfinite(calibration.sigma)
    assert st._fit_sigma(calibration.radii, np.asarray(delta_p)) == pytest.approx(calibration.sigma, rel=1e-12)


def test_a_later_run_of_the_fluid_resolves_the_collected_sigma(calibration):
    """``record_surface_tension`` publishes sigma without mutating the input config."""
    assert calibration.config.extra.get("surface_tension") is None
    assert calibration.updated.extra["surface_tension"] == pytest.approx(calibration.sigma, rel=1e-12)

    text = (calibration.run_dir / PHYSICAL_PARAMETERS_FILENAME).read_text(encoding="utf-8")
    assert f"{calibration.sigma:.6g}" in text
    assert "measured, Young–Laplace" in text


def test_every_artefact_lands_under_the_run_directory(calibration):
    """The fit, its data and one figure per droplet, all nested, nothing flat."""
    assert sorted(p.name for p in calibration.run_dir.iterdir()) == sorted(
        [st._OUTPUT_DIRNAME, PHYSICAL_PARAMETERS_FILENAME]
    )
    assert sorted(p.name for p in st.surface_tension_dir(calibration.run_dir).iterdir()) == [
        DATA_DIRNAME,
        PLOTS_DIRNAME,
    ]

    data = json.loads((st.surface_tension_data_dir(calibration.run_dir) / st._DATA_FILENAME).read_text())
    np.testing.assert_allclose(data["radii"], calibration.radii)
    assert data["sigma"] == pytest.approx(calibration.sigma, rel=1e-12)

    plots_dir = st.surface_tension_plots_dir(calibration.run_dir)
    assert (plots_dir / st._PLOT_FILENAME).stat().st_size > 0
    snapshots = plots_dir / SNAPSHOTS_DIRNAME
    assert sorted(p.name for p in snapshots.iterdir()) == sorted(f"R_{r:.2f}.png" for r in calibration.radii)


def test_mass_is_conserved_across_every_droplet_run(calibration):
    """The LBM sweep neither creates nor destroys mass over its equilibration."""
    for run_dir, initial in zip(calibration.sweep_run_dirs, calibration.initial_masses, strict=True):
        assert _rho_2d(_final_snapshot(run_dir)).sum() == pytest.approx(initial, rel=1e-9), run_dir.name


def test_each_equilibrated_droplet_is_still_a_droplet(calibration):
    """Liquid at the centre, vapour at the corners the pressure jump is read from."""
    config = calibration.config
    for run_dir in calibration.sweep_run_dirs:
        rho = _rho_2d(_final_snapshot(run_dir))
        inside, outside = st.sample_points(*rho.shape)
        centre = float(rho[inside])
        corners = float(np.mean([rho[point] for point in outside]))

        assert np.all(np.isfinite(rho)), run_dir.name
        assert rho.min() > 0.0, run_dir.name
        assert rho.max() <= config.rho_l, run_dir.name
        assert centre > 0.5 * config.rho_l, (run_dir.name, centre)
        assert corners < 0.1 * centre, (run_dir.name, corners, centre)


def test_cached_density_fields_are_the_measured_ones(calibration, monkeypatch):
    """A later run redraws its figures from the fields the sweep produced."""
    monkeypatch.setattr(st, "_FIELDS_CACHE_DIR", calibration.fields_cache_dir)
    fields = st._load_fields(calibration.cache_key, _N_RADII)

    assert fields is not None
    for run_dir, field in zip(calibration.sweep_run_dirs, fields, strict=True):
        np.testing.assert_allclose(field, _rho_2d(_final_snapshot(run_dir)))
