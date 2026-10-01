"""Tests for numerical surface-tension calibration.

These cover the pure logic — fitting, caching, plot output, staging the sweep
as run configs and collecting finished sweep runs — on synthetic run
directories. No droplet is equilibrated here: the sweep is a set of ordinary
runs, so the solver side is covered by the run pipeline's own tests.
"""

from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from src.config.adapter_toml import TomlAdapter
from src.config.config_overview import BASE_RESULTS_DIR
from src.config.run_config import CONFIG_FILENAME
from src.config.run_config import DATA_DIRNAME
from src.config.run_config import PLOTS_DIRNAME
from src.config.run_config import SNAPSHOTS_DIRNAME
from src.simulation_io.analysis.surface_tension import surface_tension as st

# Captured before the autouse fixture below redirects it, so the "never inside
# the checkout" invariant can be asserted against the value a real run uses.
_REAL_FIELDS_CACHE_DIR = st._FIELDS_CACHE_DIR


@pytest.fixture(autouse=True)
def _isolate_field_cache(tmp_path, monkeypatch):
    """Keep the density-field cache out of the developer's real data root.

    ``_FIELDS_CACHE_DIR`` resolves to ``$TUD_LBM_DATA_DIR`` at import time, so
    without this every test that measures would leave multi-megabyte archives
    in the user's actual results directory. (The JSON cache is redirected for
    every test by ``tests/conftest.py``.)
    """
    monkeypatch.setattr(st, "_FIELDS_CACHE_DIR", tmp_path / "field_cache")


def _stub_config(**overrides):
    """Minimal object exposing the attributes the calibration reads."""
    base = {
        "eos": "carnahan-starling",
        "kappa": 0.01,
        "rho_l": 0.4,
        "rho_v": 0.02,
        "interface_width": 4,
        "grid_shape": (64, 64, 1),
        "a_eos": 0.5,
        "b_eos": 4.0,
        "r_eos": 1.0,
        "t_eos": 0.05,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_fit_sigma_recovers_slope():
    radii = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    sigma_true = 0.0123
    delta_p = sigma_true / radii
    assert st._fit_sigma(radii, delta_p) == pytest.approx(sigma_true, rel=1e-9)


def test_pressure_jump_centre_minus_corners():
    # Constant-pressure field => zero jump, independent of margin.
    pressure = np.full((40, 40), 3.0)
    assert st._pressure_jump(pressure) == pytest.approx(0.0)


def test_pressure_jump_reads_the_shared_sample_points():
    """The measurement and the markers must always read the same pixels."""
    nx, ny = 40, 44
    inside, outside = st.sample_points(nx, ny)
    pressure = np.zeros((nx, ny))
    pressure[inside] = 5.0
    for i, point in enumerate(outside):
        pressure[point] = float(i)  # mean of 0, 1, 2, 3

    assert st._pressure_jump(pressure) == pytest.approx(5.0 - 1.5)


def test_sample_points_geometry():
    # margin = round(min(40, 44) / 8) = 5
    inside, outside = st.sample_points(40, 44)

    assert inside == (20, 22)
    assert outside == [(5, 5), (5, 38), (34, 5), (34, 38)]


def test_sample_points_stay_outside_the_largest_droplet():
    """The vapour corners must be bulk vapour at every resolution.

    This is what an inset measured in interface widths could not guarantee: on
    a 32x32 grid ``3 * W`` put the corners inside the largest droplet.
    """
    for n in (32, 64, 101, 401):
        inside, outside = st.sample_points(n, n)
        r_max = n * st._RADIUS_MAX_FRACTION
        distances = [np.hypot(px - inside[0], py - inside[1]) for px, py in outside]
        assert min(distances) > r_max, (n, min(distances), r_max)


def test_cache_round_trip(tmp_path, monkeypatch):
    config = _stub_config()
    path = tmp_path / st._CACHE_FILENAME
    monkeypatch.setattr(st, "_SHARED_CACHE_PATH", path)
    key = st._cache_key(config)
    radii = np.array([1.0, 2.0])
    delta_p = np.array([0.5, 0.25])

    st._store_cache(key, radii, delta_p, sigma=0.5, grid_shape=config.grid_shape)

    stored = st._load_cache(path)
    assert key in stored
    assert stored[key]["sigma"] == 0.5
    assert stored[key]["grid_shape"] == [64, 64, 1]
    assert json.loads(path.read_text())[key]["radii"] == [1.0, 2.0]


def test_cache_key_changes_with_eos_params(tmp_path):
    base = _stub_config()
    changed = _stub_config(a_eos=base.a_eos + 1.0)
    assert st._cache_key(base) != st._cache_key(changed)


def test_cache_key_does_not_depend_on_the_run_grid():
    """Sigma is a property of the fluid, so every grid of one fluid shares an entry.

    The box used to grow with the run's grid, so a 401x101 run missed the entry
    a 201x101 run of the same fluid had measured.
    """
    keys = {
        st._cache_key(_stub_config(grid_shape=grid))
        for grid in [(64, 64, 1), (201, 101, 1), (401, 101, 1), (701, 701, 1)]
    }

    assert len(keys) == 1
    assert json.loads(keys.pop())["grid_shape"] == [st._CALIBRATION_SIDE, st._CALIBRATION_SIDE, 1]


def test_load_cache_drops_malformed_entries(tmp_path):
    path = tmp_path / st._CACHE_FILENAME
    good_entry = {"sigma": 0.5, "radii": [1.0], "delta_p": [0.5], "grid_shape": [64, 64, 1]}
    good_key = st._cache_key(_stub_config())
    none_field_key = st._cache_key(_stub_config(a_eos=None))
    bad_entry_key = st._cache_key(_stub_config(kappa=0.02))
    missing_field_key = st._cache_key(_stub_config(kappa=0.03))
    bad_grid_shape_key = st._cache_key(_stub_config(kappa=0.04))
    raw = {
        good_key: good_entry,
        none_field_key: good_entry,
        "not-json{": good_entry,
        json.dumps({"unexpected": 1}): good_entry,
        json.dumps(dict(json.loads(good_key), eos="unknown-eos")): good_entry,
        json.dumps(dict(json.loads(good_key), kappa="0.1")): good_entry,
        json.dumps(dict(json.loads(good_key), kappa=True)): good_entry,
        json.dumps(dict(json.loads(good_key), grid_shape=[64, 0, 1])): good_entry,
        bad_entry_key: "not a dict",
        missing_field_key: {"radii": [1.0], "delta_p": [0.5]},
        bad_grid_shape_key: {"sigma": 0.5, "radii": [1.0], "delta_p": [0.5], "grid_shape": [64, 0, 1]},
    }
    path.write_text(json.dumps(raw))

    cache = st._load_cache(path)

    assert set(cache) == {good_key, none_field_key}
    assert cache[good_key] == good_entry


def test_load_cache_rejects_non_dict_file(tmp_path):
    path = tmp_path / st._CACHE_FILENAME
    path.write_text(json.dumps([1, 2, 3]))
    assert st._load_cache(path) == {}


def test_store_cache_preserves_existing_valid_entries(tmp_path, monkeypatch):
    path = tmp_path / st._CACHE_FILENAME
    monkeypatch.setattr(st, "_SHARED_CACHE_PATH", path)
    old_config = _stub_config(kappa=0.5, grid_shape=(64, 64, 1))
    new_config = _stub_config(grid_shape=(128, 128, 1))
    old_key = st._cache_key(old_config)
    new_key = st._cache_key(new_config)

    st._store_cache(old_key, np.array([1.0]), np.array([0.1]), sigma=0.7, grid_shape=old_config.grid_shape)
    st._store_cache(new_key, np.array([2.0]), np.array([0.2]), sigma=0.9, grid_shape=new_config.grid_shape)

    stored = st._load_cache(path)
    assert stored[old_key]["sigma"] == 0.7
    assert stored[new_key]["sigma"] == 0.9
    assert stored[old_key]["grid_shape"] == [64, 64, 1]
    assert stored[new_key]["grid_shape"] == [128, 128, 1]


def _droplet_field(config, radius):
    """A crude equilibrated-looking droplet: liquid disc in vapour."""
    nx, ny = int(config.grid_shape[0]), int(config.grid_shape[1])
    xs = np.arange(nx)[:, None] - nx // 2
    ys = np.arange(ny)[None, :] - ny // 2
    inside = np.hypot(xs, ys) <= radius
    return np.where(inside, config.rho_l, config.rho_v).astype(float)


_SEED_RADII = np.array([10.0, 20.0, 30.0])
_SEED_SIGMA = 0.02


def _seed_cache(config, *, fields: bool = True) -> None:
    """Store a measurement for *config*'s fluid, as a collected sweep would."""
    key = st._cache_key(config)
    st._store_cache(key, _SEED_RADII, _SEED_SIGMA / _SEED_RADII, _SEED_SIGMA, st._CALIBRATION_GRID_SHAPE)
    if fields:
        st._store_fields(key, [_droplet_field(config, r) for r in _SEED_RADII])


def test_calibrate_reads_cache_and_writes_plot(tmp_path):
    config = _cs_config()
    _seed_cache(config)
    run_dir = tmp_path / "run"

    sigma = st.calibrate_surface_tension(config, run_dir)

    assert sigma == pytest.approx(_SEED_SIGMA, rel=1e-9)
    assert (st.surface_tension_plots_dir(run_dir) / st._PLOT_FILENAME).exists()
    data = json.loads((st.surface_tension_data_dir(run_dir) / st._DATA_FILENAME).read_text())
    assert data["sigma"] == pytest.approx(_SEED_SIGMA, rel=1e-9)
    assert data["radii"] == [10.0, 20.0, 30.0]
    snapshots = st.surface_tension_plots_dir(run_dir) / SNAPSHOTS_DIRNAME
    assert sorted(p.name for p in snapshots.iterdir()) == ["R_10.00.png", "R_20.00.png", "R_30.00.png"]


def test_calibrate_skips_snapshots_without_cached_fields(tmp_path):
    """An entry measured on another machine still calibrates, minus the figures."""
    config = _cs_config()
    _seed_cache(config, fields=False)

    run_dir = tmp_path / "run"
    sigma = st.calibrate_surface_tension(config, run_dir)

    assert sigma == pytest.approx(_SEED_SIGMA, rel=1e-9)
    plots_dir = st.surface_tension_plots_dir(run_dir)
    assert (plots_dir / st._PLOT_FILENAME).exists()
    assert not (plots_dir / SNAPSHOTS_DIRNAME).exists()


def test_calibrate_nests_all_outputs_in_subdirectory(tmp_path):
    """No artefact is dumped flat into the run directory, or flat into its own.

    The tree mirrors a run directory: fitted numbers under ``data/``, figures
    under ``plots/``.
    """
    config = _cs_config()
    _seed_cache(config)

    run_dir = tmp_path / "run"
    st.calibrate_surface_tension(config, run_dir)

    assert sorted(p.name for p in run_dir.iterdir()) == [st._OUTPUT_DIRNAME]
    out_dir = st.surface_tension_dir(run_dir)
    assert sorted(p.name for p in out_dir.iterdir()) == [DATA_DIRNAME, PLOTS_DIRNAME]
    assert sorted(p.name for p in st.surface_tension_data_dir(run_dir).iterdir()) == [st._DATA_FILENAME]
    assert sorted(p.name for p in st.surface_tension_plots_dir(run_dir).iterdir()) == [
        st._PLOT_FILENAME,
        SNAPSHOTS_DIRNAME,
    ]


def test_calibrate_miss_measures_nothing_and_writes_nothing(tmp_path):
    """An uncalibrated fluid no longer equilibrates five droplets before the run starts."""
    config = _cs_config()
    run_dir = tmp_path / "run"

    assert st.calibrate_surface_tension(config, run_dir) is None
    assert not run_dir.exists()
    assert st._load_cache(st._cache_path()) == {}


def test_record_returns_the_config_unchanged_on_a_miss(tmp_path):
    config = _cs_config()

    assert st.record_surface_tension(config, tmp_path / "run") is config


def test_record_attaches_cached_sigma_and_rewrites_the_overview(tmp_path):
    config = _cs_config()
    _seed_cache(config, fields=False)
    run_dir = tmp_path / "run"

    updated = st.record_surface_tension(config, run_dir)

    assert updated.extra["surface_tension"] == pytest.approx(_SEED_SIGMA, rel=1e-9)
    assert "measured, Young–Laplace" in (run_dir / "physical_parameters.txt").read_text()


def test_cached_surface_tension_links_a_config_to_the_cache():
    """A run's config.toml never stores sigma; the cache is looked up by the fluid."""
    config = _cs_config(grid_shape=(48, 32))
    assert st.cached_surface_tension(config) is None

    _seed_cache(_cs_config(grid_shape=(32, 32)))

    assert st.cached_surface_tension(config) == pytest.approx(_SEED_SIGMA, rel=1e-9)
    assert st.is_calibrated(config)


def test_cached_surface_tension_ignores_the_cache_for_a_closed_form_eos():
    """A double-well entry is a verification measurement, never the run's sigma."""
    config = _cs_config(eos="double-well", a_eos=None, b_eos=None, r_eos=None, t_eos=None)
    _seed_cache(config)

    assert st.is_calibrated(config)
    assert not st.needs_calibration(config)
    assert st.cached_surface_tension(config) is None


# ── The sweep as ordinary runs ────────────────────────────────────────


_FLUID = "cs_kappa0.01_rho0.4_0.02"
_RADIUS_TAGS = ["R75", "R82", "R88", "R94", "R100"]


def test_fluid_label_and_run_names_are_readable():
    config = _cs_config()

    assert st.fluid_label(config) == _FLUID
    assert (
        st.fluid_label(_cs_config(eos="double-well", kappa=0.04, rho_l=1.0, rho_v=0.001)) == "dw_kappa0.04_rho1_0.001"
    )
    assert [c.simulation_name for c in st.calibration_configs(config)] == [
        f"surface_tension_{_FLUID}_{tag}" for tag in _RADIUS_TAGS
    ]


def test_calibration_configs_are_one_plain_run_per_radius():
    config = _cs_config(
        sim_type="multiphase_wetting",
        grid_shape=(64, 48),
        bc_config={"top": "wetting", "bottom": "bounce-back"},
        gravity_force={"force_g": 1e-6, "inclination_angle_deg": 0.0},
        results_dir="/somewhere/else",
    )

    sweep = st.calibration_configs(config)

    radii = [c.initialisation["radii"][0] * st._CALIBRATION_SIDE for c in sweep]
    np.testing.assert_allclose(radii, st.sweep_radii())
    fluid_dir = st.SURFACE_TENSION_ROOT / _FLUID
    assert [st.sweep_config_path(c) for c in sweep] == [fluid_dir / "configs" / f"{tag}.toml" for tag in _RADIUS_TAGS]
    for calib in sweep:
        assert calib.sim_type == "multiphase"
        assert tuple(calib.grid_shape) == st._CALIBRATION_GRID_SHAPE
        assert calib.gravity_force is None
        assert calib.bc_config is not None
        assert calib.bc_config["top"] == "periodic"
        assert calib.nt == st._N_ITERATIONS
        assert calib.save_interval == st._N_ITERATIONS
        assert calib.save_fields is not None
        assert {"rho", "pressure"} <= set(calib.save_fields)
        # Every run of a fluid writes into that fluid's own folder.
        assert calib.results_dir == str(fluid_dir)
        # The whole point: the sweep's result lands where the source config looks.
        assert st._cache_key(calib) == st._cache_key(config)


def test_a_sweep_config_is_recognised_by_its_name():
    config = _cs_config(simulation_name="surface_tension_notes")

    assert not st.is_sweep_config(config)
    assert not st.is_sweep_config(_cs_config())
    assert all(st.is_sweep_config(c) for c in st.calibration_configs(config))


def test_a_sweep_run_does_not_ask_for_its_own_calibration(tmp_path):
    """It *is* the measurement: no cache lookup, no "stage a sweep" hint."""
    sweep_config = st.calibration_configs(_cs_config())[0]
    _seed_cache(sweep_config, fields=False)

    assert st.record_surface_tension(sweep_config, tmp_path / "run") is sweep_config
    assert not (tmp_path / "run").exists()


def _write_sweep_run(sweep_config, sigma: float, *, stamp: str = "2026-01-01/00-00-00") -> Path:
    """A finished sweep run directory in its fluid folder: config.toml and one final snapshot."""
    run_dir = Path(sweep_config.results_dir) / f"{stamp}_{sweep_config.simulation_name}"
    data_dir = run_dir / DATA_DIRNAME
    data_dir.mkdir(parents=True)
    TomlAdapter().save(sweep_config, str(run_dir / CONFIG_FILENAME))

    side = st._CALIBRATION_SIDE
    radius = sweep_config.initialisation["radii"][0] * side
    offsets = np.arange(side) - side // 2
    inside = np.hypot(offsets[:, None], offsets[None, :]) <= radius
    pressure = np.where(inside, sigma / radius, 0.0)
    rho = np.where(inside, sweep_config.rho_l, sweep_config.rho_v)
    np.savez(
        data_dir / f"timestep_{sweep_config.nt}.npz",
        pressure=pressure[:, :, None, None, None],
        rho=rho[:, :, None, None, None],
    )
    return run_dir


def _sweep_runs() -> list[Path]:
    """The run directories of the one fluid these tests stage."""
    groups = st.find_sweep_runs(st.SURFACE_TENSION_ROOT)
    assert list(groups) == [st.SURFACE_TENSION_ROOT / _FLUID]
    return groups[st.SURFACE_TENSION_ROOT / _FLUID]


def test_find_sweep_runs_groups_by_fluid_folder_and_skips_other_runs():
    for kappa in (0.01, 0.02):
        for sweep_config in st.calibration_configs(_cs_config(kappa=kappa))[:2]:
            _write_sweep_run(sweep_config, 0.0123)
    # An ordinary run kept directly under the root, like the cs_eos_test runs.
    plain = st.SURFACE_TENSION_ROOT / "2026-08-05" / "10-08-21_cs_eos_test"
    plain.mkdir(parents=True)
    TomlAdapter().save(_cs_config(), str(plain / CONFIG_FILENAME))

    groups = st.find_sweep_runs(st.SURFACE_TENSION_ROOT)

    assert sorted(d.name for d in groups) == ["cs_kappa0.01_rho0.4_0.02", "cs_kappa0.02_rho0.4_0.02"]
    assert all(len(runs) == 2 for runs in groups.values())


def test_collect_fits_sigma_from_finished_sweep_runs():
    config = _cs_config()
    sigma_true = 0.0123
    for sweep_config in st.calibration_configs(config):
        _write_sweep_run(sweep_config, sigma_true)
    fluid_dir = st.SURFACE_TENSION_ROOT / _FLUID

    sigma = st.collect_calibration(_sweep_runs(), out_dir=fluid_dir)

    assert sigma == pytest.approx(sigma_true, rel=1e-9)
    # Stored under the key the *source* config resolves, with the fields.
    assert st.cached_surface_tension(config) == pytest.approx(sigma_true, rel=1e-9)
    fields = st._load_fields(st._cache_key(config), st._N_RADII)
    assert fields is not None
    assert fields[0].shape == (st._CALIBRATION_SIDE, st._CALIBRATION_SIDE)
    # The fit sits beside the runs it came from.
    assert (fluid_dir / st._PLOT_FILENAME).stat().st_size > 0
    data = json.loads((fluid_dir / st._DATA_FILENAME).read_text())
    assert data["sigma"] == pytest.approx(sigma_true, rel=1e-9)
    np.testing.assert_allclose(data["radii"], st.sweep_radii())


def test_collect_orders_runs_by_radius_not_by_directory_name():
    """R100 sorts before R75 as a string; the fit must not care."""
    config = _cs_config()
    for sweep_config in st.calibration_configs(config):
        _write_sweep_run(sweep_config, 0.0123)

    st.collect_calibration(_sweep_runs())

    stored = st._load_cache(st._cache_path())[st._cache_key(config)]
    np.testing.assert_allclose(stored["radii"], st.sweep_radii())
    np.testing.assert_allclose(stored["delta_p"], 0.0123 / st.sweep_radii())


def test_collect_waits_for_every_radius():
    config = _cs_config()
    for sweep_config in st.calibration_configs(config)[:-1]:
        _write_sweep_run(sweep_config, 0.0123)

    assert st.collect_calibration(_sweep_runs(), out_dir=st.SURFACE_TENSION_ROOT / _FLUID) is None
    assert st._load_cache(st._cache_path()) == {}
    assert not (st.SURFACE_TENSION_ROOT / _FLUID / st._PLOT_FILENAME).exists()


def test_collect_ignores_a_run_cut_short():
    """A snapshot from before the last step is not an equilibrated droplet."""
    config = _cs_config()
    sweep = st.calibration_configs(config)
    for sweep_config in sweep:
        run_dir = _write_sweep_run(sweep_config, 0.0123)
    final = run_dir / DATA_DIRNAME / f"timestep_{sweep[-1].nt}.npz"
    final.rename(final.with_name("timestep_100.npz"))

    assert st.collect_calibration(_sweep_runs()) is None


def test_collect_prefers_the_newest_run_of_a_radius():
    config = _cs_config()
    for sweep_config in st.calibration_configs(config):
        _write_sweep_run(sweep_config, 0.5, stamp="2026-01-01/00-00-00")
        _write_sweep_run(sweep_config, 0.0123, stamp="2026-01-02/00-00-00")

    sigma = st.collect_calibration(_sweep_runs())

    assert sigma == pytest.approx(0.0123, rel=1e-9)


def test_collect_rejects_two_fluids_sharing_one_label():
    """The folder name leaves parameters out; identity comes from the configs."""
    first = st.calibration_configs(_cs_config())
    second = st.calibration_configs(_cs_config(a_eos=0.6))
    assert st.fluid_label(first[0]) == st.fluid_label(second[0])
    for sweep_config in (first[0], second[1]):
        _write_sweep_run(sweep_config, 0.0123)

    with pytest.raises(ValueError, match="one fluid at a time"):
        st.collect_calibration(_sweep_runs())


def test_field_cache_never_lands_in_the_checkout():
    """The density fields are simulation output; the repo must stay clean.

    They were once written into ``src/.../surface_tension/data/fields/``, which
    dirtied the working tree with a multi-megabyte archive on every fresh
    calibration.
    """
    fields_dir = _REAL_FIELDS_CACHE_DIR.resolve()

    assert not fields_dir.is_relative_to(st._PACKAGE_ROOT)
    assert fields_dir.is_relative_to(Path(BASE_RESULTS_DIR).resolve())


def test_store_fields_refuses_to_write_into_the_checkout(monkeypatch):
    """The guard fires even if the cache directory is pointed back at the repo."""
    in_repo = st._PACKAGE_ROOT / "simulation_io" / "analysis" / "surface_tension" / "data" / "fields"
    monkeypatch.setattr(st, "_FIELDS_CACHE_DIR", in_repo)

    with pytest.raises(RuntimeError, match="inside the repository"):
        st._store_fields(st._cache_key(_stub_config()), [np.zeros((4, 4))])

    # Not `not in_repo.exists()`: the directory may survive from an old
    # checkout. What must hold is that nothing was written into it.
    assert not list(in_repo.glob("*"))


def test_fields_cache_round_trip():
    key = st._cache_key(_stub_config())
    densities = [np.full((8, 6), 0.4), np.full((8, 6), 0.02)]

    st._store_fields(key, densities)
    loaded = st._load_fields(key, n_radii=2)

    assert loaded is not None
    np.testing.assert_allclose(np.stack(loaded), np.stack(densities))
    assert st._load_fields(key, n_radii=3) is None  # stale entry: wrong count


def test_load_fields_missing_or_corrupt_is_a_miss():
    key = st._cache_key(_stub_config())

    assert st._load_fields(key, n_radii=2) is None  # nothing stored yet

    path = st._fields_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not an npz archive")
    assert st._load_fields(key, n_radii=2) is None


def test_store_fields_ignores_empty_sweep():
    key = st._cache_key(_stub_config())

    st._store_fields(key, [])

    assert not st._fields_path(key).exists()


def test_save_snapshot_figures_writes_one_per_radius(tmp_path):
    from src.simulation_io.analysis.surface_tension.snapshot_figures import save_snapshot_figures

    config = _cs_config()
    radii = np.array([6.0, 9.0])
    densities = [_droplet_field(config, r) for r in radii]

    save_snapshot_figures(config, tmp_path, radii, 0.02 / radii, densities, timestep=100)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["R_6.00.png", "R_9.00.png"]
    assert all(p.stat().st_size > 0 for p in tmp_path.iterdir())


def _multiphase_params(**overrides):
    from typing import Any
    from src.config.multiphase_params import MultiphaseParams

    base: dict[str, Any] = {
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
    base.update(overrides)
    return MultiphaseParams(**base)


def test_bulk_pressure_fn_carnahan_starling_matches_reference():
    import jax.numpy as jnp
    from src.operators.macroscopic.eos import build_pressure_fn
    from src.operators.macroscopic.eos._carnahan_starling import _pressure_carnahan_starling

    mp = _multiphase_params()
    assert mp.a_eos is not None
    assert mp.b_eos is not None
    assert mp.r_eos is not None
    assert mp.t_eos is not None
    pressure_fn = build_pressure_fn(mp)
    rho = jnp.linspace(mp.rho_v, mp.rho_l, 20)

    expected = _pressure_carnahan_starling(rho, mp.a_eos, mp.b_eos, mp.r_eos, mp.t_eos)
    np.testing.assert_allclose(np.asarray(pressure_fn(rho)), np.asarray(expected))


def test_bulk_pressure_fn_double_well_matches_reference():
    import jax.numpy as jnp
    from src.operators.macroscopic.eos import build_pressure_fn
    from src.operators.macroscopic.eos._double_well import _pressure_double_well

    mp = _multiphase_params(eos="double-well", a_eos=None, b_eos=None, r_eos=None, t_eos=None)
    pressure_fn = build_pressure_fn(mp)
    rho = jnp.linspace(mp.rho_v, mp.rho_l, 20)

    beta = 8.0 * mp.kappa / (float(mp.interface_width) ** 2 * (mp.rho_l - mp.rho_v) ** 2)
    expected = _pressure_double_well(rho, beta, mp.rho_l, mp.rho_v)
    np.testing.assert_allclose(np.asarray(pressure_fn(rho)), np.asarray(expected))


def test_bulk_pressure_fn_cs_missing_params_raises():
    from src.operators.macroscopic.eos import build_pressure_fn

    mp = _multiphase_params(a_eos=None)
    with pytest.raises(ValueError, match="required for Carnahan-Starling"):
        build_pressure_fn(mp)


def _cs_config(**overrides):
    """A real, valid Carnahan-Starling multiphase SimulationConfig."""
    from typing import Any
    from src.config.simulation_config import SimulationConfig

    base: dict[str, Any] = {
        "sim_type": "multiphase",
        "grid_shape": (32, 32),
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
    base.update(overrides)
    return SimulationConfig(**base)


def test_calibration_config_isolates_single_droplet():
    cfg = _cs_config(
        simulation_name="drop",
        gravity_force={"force_g": 1e-6},
        save_fields=["rho"],
        save_interval=10,
    )

    calib = st._calibration_config(cfg)

    assert calib.sim_type == "multiphase"
    assert calib.nt == st._N_ITERATIONS
    # save_interval=0 is falsy, so validation re-applies the nt // 10 default.
    assert calib.save_interval == calib.nt // 10
    assert calib.skip_interval == 0
    assert calib.bc_config is not None
    for face in ("top", "bottom", "left", "right"):
        assert calib.bc_config[face] == "periodic"
    for name in (
        "save_fields",
        "plot_fields",
        "animate_fields",
        "overlay_fields",
        "g",
        "gravity_force",
        "gravity_masked_force",
        "electric_force",
        "wetting_config",
        "hysteresis_config",
        "chemical_step_config",
    ):
        assert getattr(calib, name) is None, name
    assert calib.init_type == "multiphase_bubbles"
    assert calib.initialisation == {"centres": [[0.5, 0.5]], "radii": [0.2], "dispersed": "liquid"}
    assert calib.simulation_name == "drop_surface_tension"
    assert calib.init_dir is None
    # Thermodynamic parameters that determine sigma are preserved.
    for name in ("eos", "kappa", "rho_l", "rho_v", "interface_width", "a_eos", "b_eos", "r_eos", "t_eos"):
        assert getattr(calib, name) == getattr(cfg, name), name
