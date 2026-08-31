"""Tests for numerical surface-tension calibration.

These cover the pure logic — fitting, caching, plot output, and the cache-hit
path — without running the expensive droplet sweep (that is exercised by the
physics integration tests, not here).
"""

from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
import numpy as np
import pytest
from src.config.config_overview import BASE_RESULTS_DIR
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
    in the user's actual results directory.
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


def test_cache_key_changes_with_grid_shape():
    base = _stub_config(grid_shape=(64, 64, 1))
    changed = _stub_config(grid_shape=(128, 128, 1))
    assert st._cache_key(base) != st._cache_key(changed)


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


def test_calibrate_uses_cache_and_writes_plot(tmp_path, monkeypatch):
    config = _cs_config()

    calls = {"n": 0}
    seen_states_dirs: list[Path | None] = []

    def fake_measure(_config, states_dir=None):
        calls["n"] += 1
        seen_states_dirs.append(states_dir)
        radii = np.array([10.0, 20.0, 30.0])
        densities = [_droplet_field(config, r) for r in radii]
        return radii, 0.02 / radii, densities

    monkeypatch.setattr(st, "_measure_pressure_jumps", fake_measure)
    monkeypatch.setattr(st, "_SHARED_CACHE_PATH", tmp_path / st._CACHE_FILENAME)

    run_dir_a = tmp_path / "run_a"
    run_dir_b = tmp_path / "run_b"

    sigma_a = st.calibrate_surface_tension(config, run_dir_a)
    sigma_b = st.calibrate_surface_tension(config, run_dir_b)

    assert sigma_a == pytest.approx(0.02, rel=1e-9)
    assert sigma_b == pytest.approx(sigma_a)
    assert calls["n"] == 1  # second call served from cache
    assert seen_states_dirs == [st.surface_tension_data_dir(run_dir_a)]
    for run_dir in (run_dir_a, run_dir_b):
        # Plot and data are written on a cache hit too.
        assert (st.surface_tension_plots_dir(run_dir) / st._PLOT_FILENAME).exists()
    data_a = json.loads((st.surface_tension_data_dir(run_dir_a) / st._DATA_FILENAME).read_text())
    data_b = json.loads((st.surface_tension_data_dir(run_dir_b) / st._DATA_FILENAME).read_text())
    assert data_a["sigma"] == pytest.approx(0.02, rel=1e-9)
    assert data_a["radii"] == [10.0, 20.0, 30.0]
    assert data_b == data_a

    # Snapshot figures come from the cached density fields on the second run,
    # which ran no droplets at all.
    expected_figures = ["R_10.00.png", "R_20.00.png", "R_30.00.png"]
    for run_dir in (run_dir_a, run_dir_b):
        snapshots = st.surface_tension_plots_dir(run_dir) / SNAPSHOTS_DIRNAME
        assert sorted(p.name for p in snapshots.iterdir()) == expected_figures


def test_calibrate_skips_snapshots_without_cached_fields(tmp_path, monkeypatch):
    """A cache entry predating the field cache still calibrates, minus the figures."""
    config = _cs_config()
    monkeypatch.setattr(st, "_SHARED_CACHE_PATH", tmp_path / st._CACHE_FILENAME)
    radii = np.array([10.0, 20.0, 30.0])
    st._store_cache(st._cache_key(config), radii, 0.02 / radii, sigma=0.02, grid_shape=config.grid_shape)

    run_dir = tmp_path / "run"
    sigma = st.calibrate_surface_tension(config, run_dir)

    assert sigma == pytest.approx(0.02, rel=1e-9)
    plots_dir = st.surface_tension_plots_dir(run_dir)
    assert (plots_dir / st._PLOT_FILENAME).exists()
    assert not (plots_dir / SNAPSHOTS_DIRNAME).exists()


def test_calibrate_nests_all_outputs_in_subdirectory(tmp_path, monkeypatch):
    """No artefact is dumped flat into the run directory, or flat into its own.

    The tree mirrors a run directory: arrays and fitted numbers under ``data/``,
    figures under ``plots/``.
    """
    config = _cs_config()

    def fake_measure(_config, states_dir=None):
        if states_dir is not None:
            states_dir.mkdir(parents=True, exist_ok=True)
            (states_dir / "radius_10.00_final.npz").touch()
        radii = np.array([10.0, 20.0, 30.0])
        return radii, 0.02 / radii, [_droplet_field(config, r) for r in radii]

    monkeypatch.setattr(st, "_measure_pressure_jumps", fake_measure)
    monkeypatch.setattr(st, "_SHARED_CACHE_PATH", tmp_path / st._CACHE_FILENAME)

    run_dir = tmp_path / "run"
    st.calibrate_surface_tension(config, run_dir)

    assert [p.name for p in run_dir.iterdir()] == [st._OUTPUT_DIRNAME]
    out_dir = st.surface_tension_dir(run_dir)
    assert sorted(p.name for p in out_dir.iterdir()) == [DATA_DIRNAME, PLOTS_DIRNAME]
    assert sorted(p.name for p in st.surface_tension_data_dir(run_dir).iterdir()) == [
        st._DATA_FILENAME,
        "radius_10.00_final.npz",
    ]
    assert sorted(p.name for p in st.surface_tension_plots_dir(run_dir).iterdir()) == [
        st._PLOT_FILENAME,
        SNAPSHOTS_DIRNAME,
    ]


def test_calibrate_cache_is_grid_specific(tmp_path, monkeypatch):
    config_a = _cs_config(grid_shape=(32, 32))
    config_b = _cs_config(grid_shape=(48, 48))

    seen_grid_shapes: list[tuple[int, ...]] = []

    def fake_measure(config, states_dir=None):
        del states_dir
        seen_grid_shapes.append(tuple(config.grid_shape))
        radii = np.array([10.0, 20.0, 30.0])
        sigma = 0.02 if tuple(config.grid_shape) == (32, 32, 1) else 0.03
        return radii, sigma / radii, [_droplet_field(config, r) for r in radii]

    monkeypatch.setattr(st, "_measure_pressure_jumps", fake_measure)
    monkeypatch.setattr(st, "_SHARED_CACHE_PATH", tmp_path / st._CACHE_FILENAME)

    sigma_a = st.calibrate_surface_tension(config_a, tmp_path / "run_a")
    sigma_b = st.calibrate_surface_tension(config_b, tmp_path / "run_b")
    sigma_a_cached = st.calibrate_surface_tension(config_a, tmp_path / "run_c")

    assert sigma_a == pytest.approx(0.02, rel=1e-9)
    assert sigma_b == pytest.approx(0.03, rel=1e-9)
    assert sigma_a_cached == pytest.approx(sigma_a)
    assert seen_grid_shapes == [(32, 32, 1), (48, 48, 1)]


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


def test_save_state_writes_array_fields_and_skips_none(tmp_path):
    from typing import NamedTuple

    class FakeState(NamedTuple):
        f: np.ndarray
        rho: np.ndarray
        t: np.ndarray
        force: np.ndarray | None
        wetting: object | None

    state = FakeState(
        f=np.ones((4, 4, 1, 9, 1)),
        rho=np.full((4, 4, 1, 1, 1), 0.4),
        t=np.asarray(7),
        force=None,
        wetting=None,
    )
    path = tmp_path / "radius_10.00_final.npz"

    st._save_state(path, state)  # ty: ignore[invalid-argument-type]

    saved = np.load(path)
    assert set(saved.files) == {"f", "rho", "t"}
    np.testing.assert_array_equal(saved["rho"], state.rho)
    assert int(saved["t"]) == 7


def _multiphase_params(**overrides):
    from typing import Any
    from src.operators.macroscopic import MultiphaseParams

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
    from src.operators.macroscopic.eos import build_pressure_fn
    from src.operators.macroscopic.eos._carnahan_starling import _pressure_carnahan_starling

    mp = _multiphase_params()
    pressure_fn = build_pressure_fn(mp)
    rho = np.linspace(mp.rho_v, mp.rho_l, 20)

    expected = _pressure_carnahan_starling(rho, mp.a_eos, mp.b_eos, mp.r_eos, mp.t_eos)
    np.testing.assert_allclose(pressure_fn(rho), np.asarray(expected))


def test_bulk_pressure_fn_double_well_matches_reference():
    from src.operators.macroscopic.eos import build_pressure_fn
    from src.operators.macroscopic.eos._double_well import _pressure_double_well

    mp = _multiphase_params(eos="double-well", a_eos=None, b_eos=None, r_eos=None, t_eos=None)
    pressure_fn = build_pressure_fn(mp)
    rho = np.linspace(mp.rho_v, mp.rho_l, 20)

    beta = 8.0 * mp.kappa / (float(mp.interface_width) ** 2 * (mp.rho_l - mp.rho_v) ** 2)
    expected = _pressure_double_well(rho, beta, mp.rho_l, mp.rho_v)
    np.testing.assert_allclose(pressure_fn(rho), np.asarray(expected))


def test_bulk_pressure_fn_cs_missing_params_raises():
    from src.operators.macroscopic.eos import build_pressure_fn

    mp = _multiphase_params(a_eos=None)
    with pytest.raises(ValueError, match="required for Carnahan-Starling"):
        build_pressure_fn(mp)


def test_bulk_pressure_fn_unknown_eos_raises():
    from src.operators.macroscopic.eos import build_pressure_fn

    mp = _multiphase_params(eos="not-an-eos")
    with pytest.raises(ValueError, match="Unknown pressure scheme 'not-an-eos'"):
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
        gravity_force={"g": 1e-6},
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
    # Thermodynamic parameters that determine sigma are preserved.
    for name in ("eos", "kappa", "rho_l", "rho_v", "interface_width", "a_eos", "b_eos", "r_eos", "t_eos"):
        assert getattr(calib, name) == getattr(cfg, name), name


class _DensityState(NamedTuple):
    f: np.ndarray
    rho: np.ndarray | None


def test_density_2d_uses_rho_field():
    rho = np.arange(20.0).reshape(4, 5, 1, 1, 1)
    state = _DensityState(f=np.ones((4, 5, 1, 9, 1)), rho=rho)

    result = st._density_2d(state)  # ty: ignore[invalid-argument-type]

    assert result.shape == (4, 5)
    np.testing.assert_array_equal(result, rho[:, :, 0, 0, 0])


def test_density_2d_falls_back_to_population_sum():
    state = _DensityState(f=np.full((4, 5, 1, 9, 1), 0.5), rho=None)

    result = st._density_2d(state)  # ty: ignore[invalid-argument-type]

    assert result.shape == (4, 5)
    np.testing.assert_allclose(result, 4.5)  # 9 populations of 0.5


class _MiniState(NamedTuple):
    t: object


def test_run_to_final_state_advances_nt_steps():
    import jax.numpy as jnp

    def step_fn(_setup, state):
        return _MiniState(t=state.t + 1)

    setup = SimpleNamespace(step_fn=step_fn)
    final = st._run_to_final_state(setup, _MiniState(t=jnp.asarray(0)), nt=7)  # ty: ignore[invalid-argument-type]

    assert int(final.t) == 7


def test_run_to_final_state_requires_step_fn():
    setup = SimpleNamespace(step_fn=None)
    with pytest.raises(TypeError, match="step_fn is required"):
        st._run_to_final_state(setup, _MiniState(t=0), nt=1)  # ty: ignore[invalid-argument-type]


def test_measure_pressure_jumps_missing_params_raises():
    config = _stub_config(interface_width=None)
    with pytest.raises(ValueError, match="required for surface-tension calibration"):
        st._measure_pressure_jumps(config)


def test_measure_pressure_jumps_small_sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "_N_RADII", 2)
    monkeypatch.setattr(st, "_N_ITERATIONS", 2)
    config = _cs_config()
    states_dir = tmp_path / "states"

    radii, delta_p, densities = st._measure_pressure_jumps(config, states_dir=states_dir)

    # min(nx, ny) = 32 → radii span [6.4, 10.67].
    np.testing.assert_allclose(radii, [6.4, 10.666666666666666])
    assert delta_p.shape == (2,)
    assert np.all(np.isfinite(delta_p))
    assert [rho.shape for rho in densities] == [(32, 32), (32, 32)]
    saved = sorted(p.name for p in states_dir.glob("*.npz"))
    assert saved == [
        "radius_10.67_final.npz",
        "radius_10.67_init.npz",
        "radius_6.40_final.npz",
        "radius_6.40_init.npz",
    ]
    snapshot = np.load(states_dir / "radius_6.40_final.npz")
    assert snapshot["f"].shape == (32, 32, 1, 9, 1)
