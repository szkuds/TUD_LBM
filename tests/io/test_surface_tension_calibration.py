"""Tests for numerical surface-tension calibration.

These cover the pure logic — fitting, caching, plot output, and the cache-hit
path — without running the expensive droplet sweep (that is exercised by the
physics integration tests, not here).
"""

from __future__ import annotations
import json
from types import SimpleNamespace
import numpy as np
import pytest
from tud_lbm.io.analysis.surface_tension import surface_tension as st


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

    def fake_measure(_config):
        calls["n"] += 1
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
    assert (run_dir_a / st._PLOT_FILENAME).exists()
    assert (run_dir_b / st._PLOT_FILENAME).exists()  # plot written on cache hit too
