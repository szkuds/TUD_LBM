"""End-to-end surface-tension calibration over the real droplet sweep.

``tests/io/test_surface_tension_calibration.py`` covers the fit, the caches and
the artefact tree with ``_measure_pressure_jumps`` faked out. This module
exercises exactly the seam those tests stub: ``record_surface_tension`` driving
real droplets through ``build_setup`` -> ``init_state`` -> ``step_fn``, read
back with the real Carnahan-Starling bulk pressure, fitted, cached, and written
to disk.

**The value of sigma is deliberately not asserted.** A trustworthy number needs
the production sweep — a 201x201 domain equilibrated for 200_000 steps, as in
``examples/config_cs_simple.toml``. On a box small enough to run in a test the
droplet compresses the vapour it shares the periodic domain with, and that
finite-box pressure offset is larger than the Laplace jump itself: the fitted
slope comes out negative even though nothing is broken. Shrinking the sweep is
what makes this test affordable, so it asserts the wiring plus the physics
invariants a short run does satisfy — mass conservation, a droplet that stays a
droplet, finite fields — and leaves the calibrated number to a real run.
"""

from __future__ import annotations
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
import numpy as np
import pytest
from src.config.run_config import DATA_DIRNAME
from src.config.run_config import PLOTS_DIRNAME
from src.config.run_config import SNAPSHOTS_DIRNAME
from src.config.simulation_config import SimulationConfig
from src.simulation_io.analysis.surface_tension import surface_tension as st

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.integration, pytest.mark.slow]

#: Small enough to equilibrate three droplets in seconds, large enough that the
#: vapour corners sample points sit outside the biggest droplet.
_GRID = 48
_N_RADII = 3
_N_ITERATIONS = 200


def _cs_config(**overrides) -> SimulationConfig:
    """A real Carnahan-Starling droplet config on a test-sized grid.

    The EOS parameters are the coexistence pair from
    ``examples/config_cs_simple.toml`` — synthetic values equilibrate to a
    droplet that is barely denser than its vapour, which would make the
    "still a droplet" assertions below vacuous.
    """
    base: dict[str, Any] = {
        "sim_type": "multiphase",
        "grid_shape": (_GRID, _GRID),
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


def _expected_radii() -> np.ndarray:
    return np.linspace(_GRID * st._RADIUS_MIN_FRACTION, _GRID * st._RADIUS_MAX_FRACTION, _N_RADII)


def _rho_2d(path: Path) -> np.ndarray:
    """The (nx, ny) density field of a saved calibration state."""
    return np.asarray(np.load(path)["rho"]).reshape(_GRID, _GRID)


@pytest.fixture(scope="module")
def calibration(tmp_path_factory) -> SimpleNamespace:
    """Calibrate once for the whole module; every test below reads the result.

    Both caches are redirected into ``tmp_path``: ``_SHARED_CACHE_PATH`` is a
    git-tracked file and ``_FIELDS_CACHE_DIR`` resolves under the developer's
    real data root, and a test must dirty neither.
    """
    tmp = tmp_path_factory.mktemp("surface_tension_e2e")
    config = _cs_config()
    sweeps: list[Path | None] = []
    measure = st._measure_pressure_jumps

    def counting_measure(cfg, states_dir=None):
        sweeps.append(states_dir)
        return measure(cfg, states_dir=states_dir)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(st, "_SHARED_CACHE_PATH", tmp / st._CACHE_FILENAME)
        mp.setattr(st, "_FIELDS_CACHE_DIR", tmp / "field_cache")
        mp.setattr(st, "_N_RADII", _N_RADII)
        mp.setattr(st, "_N_ITERATIONS", _N_ITERATIONS)
        mp.setattr(st, "_measure_pressure_jumps", counting_measure)

        run_dir = tmp / "run"
        run_dir.mkdir()
        updated = st.record_surface_tension(config, run_dir)

        cached_run_dir = tmp / "cached_run"
        cached_run_dir.mkdir()
        cached_sigma = st.calibrate_surface_tension(config, cached_run_dir)

    return SimpleNamespace(
        config=config,
        updated=updated,
        run_dir=run_dir,
        cached_run_dir=cached_run_dir,
        sigma=float(updated.extra["surface_tension"]),
        cached_sigma=cached_sigma,
        sweeps=sweeps,
        fields_cache_dir=tmp / "field_cache",
    )


def test_the_sweep_runs_once_and_the_repeat_is_served_from_cache(calibration):
    """The second calibration of identical parameters runs no droplets at all."""
    assert calibration.sweeps == [st.surface_tension_data_dir(calibration.run_dir)]
    assert calibration.cached_sigma == pytest.approx(calibration.sigma, rel=1e-12)
    assert not list(st.surface_tension_data_dir(calibration.cached_run_dir).glob("radius_*.npz"))


def test_returned_sigma_is_the_fit_of_the_points_that_were_written(calibration):
    """The number handed back is the slope through the measured points on disk."""
    data = json.loads((st.surface_tension_data_dir(calibration.run_dir) / st._DATA_FILENAME).read_text())
    radii = np.asarray(data["radii"], dtype=float)
    delta_p = np.asarray(data["delta_p"], dtype=float)

    np.testing.assert_allclose(radii, _expected_radii())
    assert np.all(np.isfinite(delta_p))
    assert np.isfinite(calibration.sigma)
    assert data["sigma"] == pytest.approx(calibration.sigma, rel=1e-12)
    assert st._fit_sigma(radii, delta_p) == pytest.approx(calibration.sigma, rel=1e-12)


def test_every_artefact_lands_under_the_run_directory(calibration):
    """One init/final state and one figure per droplet, all nested, nothing flat."""
    assert sorted(p.name for p in calibration.run_dir.iterdir()) == sorted(
        [st._OUTPUT_DIRNAME, "physical_parameters.txt"]
    )
    assert sorted(p.name for p in st.surface_tension_dir(calibration.run_dir).iterdir()) == [
        DATA_DIRNAME,
        PLOTS_DIRNAME,
    ]

    radii = _expected_radii()
    data_dir = st.surface_tension_data_dir(calibration.run_dir)
    expected_states = sorted(f"radius_{r:.2f}_{stage}.npz" for r in radii for stage in ("init", "final"))
    assert sorted(p.name for p in data_dir.iterdir()) == sorted([st._DATA_FILENAME, *expected_states])

    plots_dir = st.surface_tension_plots_dir(calibration.run_dir)
    assert (plots_dir / st._PLOT_FILENAME).stat().st_size > 0
    snapshots = plots_dir / SNAPSHOTS_DIRNAME
    assert sorted(p.name for p in snapshots.iterdir()) == sorted(f"R_{r:.2f}.png" for r in radii)


def test_mass_is_conserved_across_every_droplet_run(calibration):
    """The LBM sweep neither creates nor destroys mass over its equilibration."""
    data_dir = st.surface_tension_data_dir(calibration.run_dir)
    for radius in _expected_radii():
        initial = _rho_2d(data_dir / f"radius_{radius:.2f}_init.npz")
        final = _rho_2d(data_dir / f"radius_{radius:.2f}_final.npz")
        assert final.sum() == pytest.approx(initial.sum(), rel=1e-9), radius


def test_each_equilibrated_droplet_is_still_a_droplet(calibration):
    """Liquid at the centre, vapour at the corners the pressure jump is read from."""
    config = calibration.config
    data_dir = st.surface_tension_data_dir(calibration.run_dir)
    for radius in _expected_radii():
        rho = _rho_2d(data_dir / f"radius_{radius:.2f}_final.npz")
        inside, outside = st.sample_points(*rho.shape)
        centre = float(rho[inside])
        corners = float(np.mean([rho[point] for point in outside]))

        assert np.all(np.isfinite(rho)), radius
        assert rho.min() > 0.0, radius
        assert rho.max() <= config.rho_l, radius
        assert centre > 0.5 * config.rho_l, (radius, centre)
        assert corners < 0.1 * centre, (radius, corners, centre)


def test_cached_density_fields_are_the_measured_ones(calibration, monkeypatch):
    """A later cache hit redraws its figures from the fields the sweep produced."""
    monkeypatch.setattr(st, "_FIELDS_CACHE_DIR", calibration.fields_cache_dir)
    fields = st._load_fields(st._cache_key(calibration.config), _N_RADII)

    assert fields is not None
    data_dir = st.surface_tension_data_dir(calibration.run_dir)
    for radius, field in zip(_expected_radii(), fields, strict=True):
        np.testing.assert_allclose(field, _rho_2d(data_dir / f"radius_{radius:.2f}_final.npz"))


def test_physical_parameters_reports_the_measured_sigma(calibration):
    """``record_surface_tension`` publishes sigma without mutating the input config."""
    assert calibration.config.extra.get("surface_tension") is None
    assert calibration.updated.extra["surface_tension"] == pytest.approx(calibration.sigma)

    text = (calibration.run_dir / "physical_parameters.txt").read_text(encoding="utf-8")
    assert f"{calibration.sigma:.6g}" in text
    assert "measured, Young–Laplace" in text
