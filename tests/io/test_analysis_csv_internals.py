"""Coverage for the shared droplet-metric layer and the CSV serializer.

Exercises:
- resolve_r_zero: all three paths (contact-line, init-radii, fallback 27)
- inclination_angle_deg: from gravity_force dict, and no-force fallback
- analytical_sigma_lg: basic computation
- interpolate_interface: left/right interface sub-cell positions
- contact_angles_from_rho, contact_lines_from_rho: from synthetic rho fields
- center_of_mass, avg_x_location: from synthetic rho fields
- backward_diff: with interval>1 and interval=1
- build_simulation_csv: skip when wrong sim_type, skip when no data dir,
  and successful write when valid files exist
- analyse_tree: no-run and single-run paths (mocked CSV write)
- _clean_dir_label: digit prefix stripping
- parse_timestep_from_path: valid and invalid stems
- SimulationCsvExport.compute and .render
"""

from __future__ import annotations
from unittest.mock import patch
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from src.cli.analysis_routing import analyse_tree
from src.config import SimulationConfig
from src.simulation_io.analysis.droplet_metrics import RZero
from src.simulation_io.analysis.droplet_metrics import analytical_sigma_lg
from src.simulation_io.analysis.droplet_metrics import backward_diff
from src.simulation_io.analysis.droplet_metrics import inclination_angle_deg
from src.simulation_io.analysis.droplet_metrics import resolve_r_zero
from src.simulation_io.analysis.droplet_metrics._snapshot import avg_x_location
from src.simulation_io.analysis.droplet_metrics._snapshot import center_of_mass
from src.simulation_io.analysis.droplet_metrics._snapshot import contact_angles_from_rho
from src.simulation_io.analysis.droplet_metrics._snapshot import contact_lines_from_rho
from src.simulation_io.analysis.droplet_metrics._snapshot import interpolate_interface
from src.simulation_io.analysis.droplet_metrics._snapshot import parse_timestep_from_path
from src.simulation_io.plotting.run_comparison import _clean_dir_label
from src.simulation_io.plotting.simulation_csv import SimulationCsvExport
from src.simulation_io.plotting.simulation_csv import build_simulation_csv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RHO_L = 1.0
_RHO_V = 0.2
_RHO_MEAN = 0.5 * (_RHO_L + _RHO_V)


def _droplet_rho_2d(nx: int = 30, ny: int = 10, lo: int = 8, hi: int = 22) -> np.ndarray:
    """Return a (nx, ny) rho_2d array with a liquid column from lo to hi."""
    rho = np.full((nx, ny), _RHO_V)
    rho[lo:hi, :] = _RHO_L
    return rho


def _droplet_rho_5d(nx: int = 30, ny: int = 10) -> np.ndarray:
    return _droplet_rho_2d(nx, ny)[:, :, np.newaxis, np.newaxis, np.newaxis]


def _wetting_config() -> SimulationConfig:
    return SimulationConfig(
        sim_type="multiphase_wetting",
        grid_shape=(30, 10),
        tau=0.8,
        nt=100,
        save_interval=10,
        eos="double-well",
        kappa=0.02,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=2,
        initialisation={"radii": [0.3]},
    )


# ---------------------------------------------------------------------------
# resolve_r_zero
# ---------------------------------------------------------------------------


def test_resolve_r_zero_from_contact_line_length(tmp_path):
    rho = _droplet_rho_5d()
    npz = tmp_path / "init.npz"
    np.savez(npz, rho=rho)
    cfg = SimulationConfig(
        sim_type="multiphase_wetting",
        grid_shape=(30, 10),
        tau=0.8,
        nt=100,
        eos="double-well",
        kappa=0.02,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=2,
        init_type="init_from_file",
        init_dir=str(npz),
    )
    r = resolve_r_zero(cfg)
    assert r[0] > 0.0


def test_resolve_r_zero_from_init_radii():
    cfg = SimulationConfig(
        sim_type="multiphase_wetting",
        grid_shape=(30, 10),
        tau=0.8,
        nt=100,
        eos="double-well",
        kappa=0.02,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=2,
        initialisation={"radii": [0.3]},
    )
    r = resolve_r_zero(cfg)
    # 0.3 * min(30, 10) = 3.0
    assert r == RZero(value=3.0, used_fallback=True)


def test_resolve_r_zero_fallback_27():
    cfg = SimulationConfig(
        sim_type="multiphase_wetting",
        grid_shape=(30, 10),
        tau=0.8,
        nt=100,
        eos="double-well",
        kappa=0.02,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=2,
        initialisation={},
    )
    assert resolve_r_zero(cfg) == RZero(value=27.0, used_fallback=True)


# ---------------------------------------------------------------------------
# inclination_angle_deg
# ---------------------------------------------------------------------------


def test_inclination_angle_from_gravity_force():
    cfg = SimulationConfig(gravity_force={"force_g": 1e-6, "inclination_angle_deg": 45.0})
    assert inclination_angle_deg(cfg) == pytest.approx(45.0)


def test_inclination_angle_defaults_zero_when_no_force():
    cfg = SimulationConfig()
    assert inclination_angle_deg(cfg) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# analytical_sigma_lg
# ---------------------------------------------------------------------------


def test_analytical_sigma_lg_computation():
    cfg = SimulationConfig(
        sim_type="multiphase_wetting",
        grid_shape=(8, 8),
        tau=0.8,
        nt=10,
        eos="double-well",
        kappa=0.02,
        rho_l=1.0,
        rho_v=0.5,
        interface_width=2,
    )
    expected = (2.0 / 3.0) * (0.02 / 2) * (0.5**2)
    assert analytical_sigma_lg(cfg) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# interpolate_interface
# ---------------------------------------------------------------------------


def test_interpolate_interface_returns_pair():
    rho_2d = _droplet_rho_2d()
    xl, xr = interpolate_interface(rho_2d[:, 1], _RHO_MEAN)
    assert xl < xr


# ---------------------------------------------------------------------------
# contact_angles_from_rho and contact_lines_from_rho
# ---------------------------------------------------------------------------


def test_contact_angles_from_rho_returns_two_floats():
    rho_2d = _droplet_rho_2d()
    ca_l, ca_r = contact_angles_from_rho(rho_2d, _RHO_MEAN)
    assert isinstance(ca_l, float)
    assert isinstance(ca_r, float)
    # Both contact angles should be in a physically plausible range
    assert 0 < ca_l < 180
    assert 0 < ca_r < 180


def test_contact_lines_from_rho_returns_left_right():
    rho_2d = _droplet_rho_2d()
    xl, xr = contact_lines_from_rho(rho_2d, _RHO_MEAN)
    assert xl < xr


def _offset_circle_rho_2d(nx: int = 64, ny: int = 32, cx: float = 32.0, cy: float = 10.0, r: float = 18.0):
    """A ``(nx, ny)`` circular droplet whose centre may sit off-grid."""
    x = np.arange(nx, dtype=float)[:, None]
    y = np.arange(ny, dtype=float)[None, :]
    return np.where(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) < r, _RHO_L, _RHO_V)


@pytest.mark.parametrize("wall_edge", ["bottom", "top", "left", "right"])
def test_fallback_matches_live_measurement(wall_edge: str):
    """The numpy fallback must agree with the JAX live path on every wall.

    The live simulation writes ``ca_*`` from ``compute_contact_angle``; the
    numpy ``contact_angles_from_rho`` runs only when those keys are absent. If
    the two disagreed, a top-wall run would report different angles depending
    on whether the snapshot happened to carry the scalars.
    """
    import jax.numpy as jnp
    from src.operators.wetting._contact_angle import compute_contact_angle
    from src.operators.wetting._contact_line import compute_contact_line_location

    # Orient the base droplet so it sits on the requested wall.
    base = _offset_circle_rho_2d()
    if wall_edge == "top":
        rho_2d = base[:, ::-1]
    elif wall_edge == "left":
        rho_2d = base.T
    elif wall_edge == "right":
        rho_2d = base.T[::-1, :]
    else:
        rho_2d = base

    rho_5d = jnp.asarray(rho_2d)[:, :, None, None, None].astype(jnp.float64)

    ca_l_np, ca_r_np = contact_angles_from_rho(rho_2d, _RHO_MEAN, wall_edge)
    ca_l_jx, ca_r_jx = compute_contact_angle(rho_5d, jnp.array(_RHO_MEAN), edge=wall_edge)
    np.testing.assert_allclose([ca_l_np, ca_r_np], [float(ca_l_jx), float(ca_r_jx)], atol=1e-9)

    cll_l_np, cll_r_np = contact_lines_from_rho(rho_2d, _RHO_MEAN, wall_edge)
    cll_l_jx, cll_r_jx = compute_contact_line_location(rho_5d, ca_l_jx, ca_r_jx, jnp.array(_RHO_MEAN), edge=wall_edge)
    # The numpy CLL reads row 1 while the JAX CLL reads row 0 and projects by
    # the contact angle — so compare only their ordering, which both must share.
    assert cll_l_np < cll_r_np
    assert float(cll_l_jx) < float(cll_r_jx)


# ---------------------------------------------------------------------------
# center_of_mass and avg_x_location
# ---------------------------------------------------------------------------


def test_center_of_mass_symmetric_droplet():
    rho_2d = _droplet_rho_2d(nx=30, ny=10, lo=10, hi=20)
    cm_x, cm_y = center_of_mass(rho_2d, _RHO_MEAN)
    # Symmetric droplet centred at x≈14.5
    assert 9 < cm_x < 21
    assert 0 <= cm_y < 10


def test_avg_x_location():
    rho_2d = _droplet_rho_2d(nx=30, ny=10, lo=10, hi=20)
    avg_x = avg_x_location(rho_2d, _RHO_MEAN, offset_x=15.0)
    # Should be close to 0 for a centred droplet with offset=15
    assert abs(avg_x) < 2.0


# ---------------------------------------------------------------------------
# backward_diff
# ---------------------------------------------------------------------------


def test_backward_diff_unit_gaps():
    arr = np.array([0.0, 2.0, 4.0, 8.0])
    diff = backward_diff(arr, np.array([0, 1, 2, 3]), 1)
    np.testing.assert_allclose(diff, [0.0, 2.0, 2.0, 4.0])


def test_backward_diff_uniform_gaps_greater_than_one():
    arr = np.array([0.0, 10.0, 20.0, 30.0])
    diff = backward_diff(arr, np.array([0, 10, 20, 30]), 10)
    # Each step is 10 position / 10 iterations = 1.0
    np.testing.assert_allclose(diff[1:], [1.0, 1.0, 1.0])


def test_backward_diff_uses_actual_gap_not_nominal_interval():
    """A run whose snapshots were pruned must not report inflated velocities."""
    arr = np.array([0.0, 0.5, 1.0])
    iterations = np.array([0, 10, 50])

    diff = backward_diff(arr, iterations, 10)

    # Second gap is 40, not the nominal 10: 0.5/40, not 0.5/10.
    np.testing.assert_allclose(diff, [0.0, 0.05, 0.0125])


def test_backward_diff_repeated_iteration_yields_zero_not_inf():
    arr = np.array([0.0, 3.0, 6.0])

    diff = backward_diff(arr, np.array([0, 5, 5]), 5)

    np.testing.assert_allclose(diff, [0.0, 0.6, 0.0])


def test_backward_diff_leading_element_is_zero():
    diff = backward_diff(np.array([7.0, 9.0]), np.array([5, 10]), 5)

    assert diff[0] == 0.0


# ---------------------------------------------------------------------------
# parse_timestep_from_path
# ---------------------------------------------------------------------------


def test_parse_timestep_from_path_valid(tmp_path):
    p = tmp_path / "timestep_42.npz"
    assert parse_timestep_from_path(p) == 42


def test_parse_timestep_from_path_invalid_returns_minus_one(tmp_path):
    p = tmp_path / "bad_name.npz"
    assert parse_timestep_from_path(p) == -1


# ---------------------------------------------------------------------------
# _clean_dir_label
# ---------------------------------------------------------------------------


def test_clean_dir_label_strips_digit_prefix():
    assert _clean_dir_label("001_my_simulation") == "My simulation"


def test_clean_dir_label_no_digit_prefix():
    label = _clean_dir_label("my_simulation")
    assert "simulation" in label.lower()


# ---------------------------------------------------------------------------
# SimulationCsvExport — compute and render
# ---------------------------------------------------------------------------


def test_simulation_csv_export_compute_returns_empty_payload(tmp_path):
    cfg = _wetting_config()
    op = SimulationCsvExport(config=cfg)
    result = op.compute([])
    assert result["iters"].size == 0
    assert result["values"].size == 0


def test_simulation_csv_export_render_shows_placeholder():
    cfg = _wetting_config()
    op = SimulationCsvExport(config=cfg)
    fig, ax = plt.subplots()
    try:
        op.render(ax, {"iters": np.array([]), "values": np.array([])})
        assert "CSV export operator" in ax.get_title()
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# build_simulation_csv
# ---------------------------------------------------------------------------


def test_build_simulation_csv_skips_wrong_sim_type(tmp_path):
    cfg = SimulationConfig(
        sim_type="single",  # ty: ignore[invalid-argument-type]
        grid_shape=(8, 8),
        tau=0.8,
        nt=10,
        eos="double-well",
        kappa=0.02,
        rho_l=1.0,
        rho_v=0.5,
        interface_width=2,
    )
    result = build_simulation_csv(tmp_path, cfg)
    assert result is None


def test_build_simulation_csv_skips_missing_data_dir(tmp_path):
    cfg = SimulationConfig(
        sim_type="multiphase_wetting",
        grid_shape=(8, 8),
        tau=0.8,
        nt=10,
        eos="double-well",
        kappa=0.02,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=2,
    )
    result = build_simulation_csv(tmp_path, cfg)
    assert result is None


def test_build_simulation_csv_skips_empty_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg = SimulationConfig(
        sim_type="multiphase_wetting",
        grid_shape=(8, 8),
        tau=0.8,
        nt=10,
        eos="double-well",
        kappa=0.02,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=2,
    )
    result = build_simulation_csv(tmp_path, cfg)
    assert result is None


def test_build_simulation_csv_writes_file(tmp_path):
    pd = pytest.importorskip("pandas")

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    rho_2d = _droplet_rho_2d(nx=30, ny=10)
    rho = rho_2d[:, :, np.newaxis, np.newaxis, np.newaxis]
    u = np.zeros((30, 10, 1, 1, 2))

    for step in (10, 20, 30):
        np.savez(data_dir / f"timestep_{step}.npz", rho=rho, u=u)

    cfg = SimulationConfig(
        sim_type="multiphase_wetting",
        grid_shape=(30, 10),
        tau=0.8,
        nt=100,
        save_interval=10,
        eos="double-well",
        kappa=0.02,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=2,
        initialisation={"radii": [0.3]},
    )

    result = build_simulation_csv(tmp_path, cfg)
    assert result is not None
    assert result.exists()
    assert result.suffix == ".csv"

    df = pd.read_csv(result)
    assert "Re" in df.columns
    nu = (0.8 - 0.5) / 3.0
    r_zero = resolve_r_zero(cfg).value
    expected_re = df["avg_u_x"] * (2.0 * r_zero) / nu
    np.testing.assert_allclose(df["Re"].to_numpy(), expected_re.to_numpy())


# ---------------------------------------------------------------------------
# analyse_tree
# ---------------------------------------------------------------------------


def test_analyse_tree_returns_zero_zero_when_no_runs(tmp_path):
    n_runs, n_ok = analyse_tree(tmp_path)
    assert (n_runs, n_ok) == (0, 0)


def test_analyse_tree_skips_init_dirs(tmp_path):
    init_dir = tmp_path / "init" / "run1"
    init_dir.mkdir(parents=True)
    (init_dir / "config.toml").write_text("[simulation_type]\nsim_type='single'\n", encoding="utf-8")
    n_runs, _ = analyse_tree(tmp_path)
    assert n_runs == 0


def test_analyse_tree_counts_valid_run(tmp_path):
    pytest.importorskip("pandas")

    run_dir = tmp_path / "run1"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)

    rho_2d = _droplet_rho_2d()
    rho = rho_2d[:, :, np.newaxis, np.newaxis, np.newaxis]
    u = np.zeros((30, 10, 1, 1, 2))
    np.savez(data_dir / "timestep_10.npz", rho=rho, u=u)

    cfg = SimulationConfig(
        sim_type="multiphase_wetting",
        grid_shape=(30, 10),
        tau=0.8,
        nt=100,
        save_interval=10,
        eos="double-well",
        kappa=0.02,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=2,
        initialisation={"radii": [0.3]},
        simulation_name="test_run",
    )

    # Write a config.toml so the discovery loop finds the run
    (run_dir / "config.toml").write_text("", encoding="utf-8")
    with (
        patch("src.cli.analysis_routing._safe_load_config", return_value=cfg),
        patch("src.cli.analysis_routing.compare_runs"),
    ):
        n_runs, _ = analyse_tree(tmp_path)

    assert n_runs == 1
