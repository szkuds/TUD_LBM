"""Tests for numerical surface-tension calibration.

These cover the pure logic — fitting, caching, plot output, and the cache-hit
path — without running the expensive droplet sweep (that is exercised by the
physics integration tests, not here).
"""

from __future__ import annotations
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import NamedTuple
import numpy as np
import pytest
from tud_lbm.io.analysis.surface_tension import surface_tension as st

if TYPE_CHECKING:
    from pathlib import Path


def _stub_config(**overrides):
    """Minimal object exposing the attributes the calibration reads."""
    base = {
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
    return SimpleNamespace(**base)


def test_fit_sigma_recovers_slope():
    radii = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    sigma_true = 0.0123
    delta_p = sigma_true / radii
    assert st._fit_sigma(radii, delta_p) == pytest.approx(sigma_true, rel=1e-9)


def test_pressure_jump_centre_minus_corners():
    # Constant-pressure field => zero jump, independent of margin.
    pressure = np.full((40, 40), 3.0)
    assert st._pressure_jump(pressure, width=4) == pytest.approx(0.0)


def test_cache_round_trip(tmp_path, monkeypatch):
    config = _stub_config()
    path = tmp_path / st._CACHE_FILENAME
    monkeypatch.setattr(st, "_SHARED_CACHE_PATH", path)
    key = st._cache_key(config)
    radii = np.array([1.0, 2.0])
    delta_p = np.array([0.5, 0.25])

    st._store_cache(key, radii, delta_p, sigma=0.5)

    stored = st._load_cache(path)
    assert key in stored
    assert stored[key]["sigma"] == 0.5
    assert json.loads(path.read_text())[key]["radii"] == [1.0, 2.0]


def test_cache_key_changes_with_eos_params(tmp_path):
    base = _stub_config()
    changed = _stub_config(a_eos=base.a_eos + 1.0)
    assert st._cache_key(base) != st._cache_key(changed)


def test_load_cache_drops_malformed_entries(tmp_path):
    path = tmp_path / st._CACHE_FILENAME
    good_entry = {"sigma": 0.5, "radii": [1.0], "delta_p": [0.5]}
    good_key = st._cache_key(_stub_config())
    none_field_key = st._cache_key(_stub_config(a_eos=None))
    bad_entry_key = st._cache_key(_stub_config(kappa=0.02))
    missing_field_key = st._cache_key(_stub_config(kappa=0.03))
    raw = {
        good_key: good_entry,
        none_field_key: good_entry,
        "not-json{": good_entry,
        json.dumps({"unexpected": 1}): good_entry,
        json.dumps(dict(json.loads(good_key), eos="unknown-eos")): good_entry,
        json.dumps(dict(json.loads(good_key), kappa="0.1")): good_entry,
        json.dumps(dict(json.loads(good_key), kappa=True)): good_entry,
        bad_entry_key: "not a dict",
        missing_field_key: {"radii": [1.0], "delta_p": [0.5]},
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
    old_key = st._cache_key(_stub_config(kappa=0.5))
    new_key = st._cache_key(_stub_config())

    st._store_cache(old_key, np.array([1.0]), np.array([0.1]), sigma=0.7)
    st._store_cache(new_key, np.array([2.0]), np.array([0.2]), sigma=0.9)

    stored = st._load_cache(path)
    assert stored[old_key]["sigma"] == 0.7
    assert stored[new_key]["sigma"] == 0.9


def test_calibrate_uses_cache_and_writes_plot(tmp_path, monkeypatch):
    config = _stub_config()

    calls = {"n": 0}
    seen_states_dirs: list[Path | None] = []

    def fake_measure(_config, states_dir=None):
        calls["n"] += 1
        seen_states_dirs.append(states_dir)
        radii = np.array([10.0, 20.0, 30.0])
        return radii, 0.02 / radii

    monkeypatch.setattr(st, "_measure_pressure_jumps", fake_measure)
    monkeypatch.setattr(st, "_SHARED_CACHE_PATH", tmp_path / st._CACHE_FILENAME)

    run_dir_a = tmp_path / "run_a"
    run_dir_b = tmp_path / "run_b"

    sigma_a = st.calibrate_surface_tension(config, run_dir_a)
    sigma_b = st.calibrate_surface_tension(config, run_dir_b)

    assert sigma_a == pytest.approx(0.02, rel=1e-9)
    assert sigma_b == pytest.approx(sigma_a)
    assert calls["n"] == 1  # second call served from cache
    assert seen_states_dirs == [run_dir_a / st._STATES_DIRNAME]
    assert (run_dir_a / st._PLOT_FILENAME).exists()
    assert (run_dir_b / st._PLOT_FILENAME).exists()  # plot written on cache hit too
    data_a = json.loads((run_dir_a / st._DATA_FILENAME).read_text())
    data_b = json.loads((run_dir_b / st._DATA_FILENAME).read_text())  # data written on cache hit too
    assert data_a["sigma"] == pytest.approx(0.02, rel=1e-9)
    assert data_a["radii"] == [10.0, 20.0, 30.0]
    assert data_b == data_a


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
    from tud_lbm.operators.macroscopic import MultiphaseParams

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
    from tud_lbm.operators.macroscopic.eos import carnahan_starling_pressure

    mp = _multiphase_params()
    pressure_fn = st._bulk_pressure_fn(mp)
    rho = np.linspace(mp.rho_v, mp.rho_l, 20)

    expected = carnahan_starling_pressure(rho, mp.a_eos, mp.b_eos, mp.r_eos, mp.t_eos)
    np.testing.assert_allclose(pressure_fn(rho), np.asarray(expected))


def test_bulk_pressure_fn_double_well_matches_reference():
    from tud_lbm.operators.macroscopic.eos import double_well_pressure

    mp = _multiphase_params(eos="double-well", a_eos=None, b_eos=None, r_eos=None, t_eos=None)
    pressure_fn = st._bulk_pressure_fn(mp)
    rho = np.linspace(mp.rho_v, mp.rho_l, 20)

    beta = 8.0 * mp.kappa / (float(mp.interface_width) ** 2 * (mp.rho_l - mp.rho_v) ** 2)
    expected = double_well_pressure(rho, beta, mp.rho_l, mp.rho_v)
    np.testing.assert_allclose(pressure_fn(rho), np.asarray(expected))


def test_bulk_pressure_fn_cs_missing_params_raises():
    mp = _multiphase_params(a_eos=None)
    with pytest.raises(ValueError, match="required for Carnahan-Starling"):
        st._bulk_pressure_fn(mp)


def test_bulk_pressure_fn_unknown_eos_raises():
    mp = _multiphase_params(eos="not-an-eos")
    with pytest.raises(ValueError, match="supports EOS"):
        st._bulk_pressure_fn(mp)


def _cs_config(**overrides):
    """A real, valid Carnahan-Starling multiphase SimulationConfig."""
    from typing import Any
    from tud_lbm.config.simulation_config import SimulationConfig

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

    radii, delta_p = st._measure_pressure_jumps(config, states_dir=states_dir)

    # min(nx, ny) = 32 → radii span [8, 16].
    np.testing.assert_allclose(radii, [8.0, 16.0])
    assert delta_p.shape == (2,)
    assert np.all(np.isfinite(delta_p))
    saved = sorted(p.name for p in states_dir.glob("*.npz"))
    assert saved == [
        "radius_16.00_final.npz",
        "radius_16.00_init.npz",
        "radius_8.00_final.npz",
        "radius_8.00_init.npz",
    ]
    snapshot = np.load(states_dir / "radius_8.00_final.npz")
    assert snapshot["f"].shape == (32, 32, 1, 9, 1)
