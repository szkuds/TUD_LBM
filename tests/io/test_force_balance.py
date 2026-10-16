"""Force-balance analysis: geometry, per-point table, fits and dimensional consistency."""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.simulation_io.analysis.droplet_metrics import DropletSeries
from src.simulation_io.analysis.droplet_metrics import MetricScales
from src.simulation_io.analysis.force_balance.analyses import add_point_friction
from src.simulation_io.analysis.force_balance.config import AnalysisSettings
from src.simulation_io.analysis.force_balance.config import load_force_balance_config
from src.simulation_io.analysis.force_balance.fits import MODEL_BY_KEY
from src.simulation_io.analysis.force_balance.fits import fit_model
from src.simulation_io.analysis.force_balance.geometry import StepClosure
from src.simulation_io.analysis.force_balance.geometry import base_squared_over_area
from src.simulation_io.analysis.force_balance.geometry import cap_angle
from src.simulation_io.analysis.force_balance.points import TRAIL_CROSS_MARGIN_LU
from src.simulation_io.analysis.force_balance.points import build_point_table
from src.simulation_io.analysis.force_balance.points import noise_sigma
from src.simulation_io.analysis.force_balance.points import regime_of
from src.simulation_io.analysis.force_balance.points import split_contact_length
from src.simulation_io.analysis.force_balance.runs import AssumptionLog
from src.simulation_io.analysis.force_balance.runs import RunInfo
from src.simulation_io.analysis.force_balance.runs import Windows
from src.simulation_io.analysis.physical_parameters import DimensionlessInputs
from src.simulation_io.analysis.physical_parameters.physical_parameters import dimensionless_for_inputs

_WINDOWS = Windows(receding_phob=100.0, advancing_phob=120.0, receding_phil=50.0, advancing_phil=70.0)


def _info() -> RunInfo:
    """The reference run's scales: tau 0.99, alpha 60 deg, Omega_D 1016 LU^2."""
    return RunInfo(
        run="test/run",
        path=Path("test/run"),
        name="run",
        groups=["inc"],
        nx=201,
        save_interval=200,
        t_onset=0,
        tau=0.99,
        rho_l=1.0,
        eta=0.49 / 3.0,
        gamma=0.00532267,
        gamma_source="analytical",
        drho=1.0045,
        g=5e-6,
        alpha_deg=60.0,
        omega=1016.0,
        r0=math.sqrt(1016.0),
        step_x=80.4,
        edge_width=1.0,
        windows=_WINDOWS,
        learning_rate=0.005,
        interface_width=5.0,
    )


def _series(info: RunInfo, velocity: float, n: int = 40) -> DropletSeries:
    """A droplet translating rigidly at *velocity* (LU/step), base 40 LU."""
    it = np.arange(1, n + 1) * info.save_interval
    cm = 40.0 + velocity * it
    scales = MetricScales(
        rho_mean=0.5,
        sigma_measured=None,
        sigma_analytical=info.gamma,
        nu=info.eta / info.rho_l,
        r_zero=18.0,
        r_zero_is_fallback=False,
        offset_x=info.step_x,
        incl_deg=info.alpha_deg,
        save_interval=info.save_interval,
        wall_edge="bottom",
    )
    return DropletSeries(
        iteration=it,
        avg_u_x=np.full(n, velocity),
        avg_u_y=np.zeros(n),
        avg_x_location=cm - info.step_x,
        cll_left=cm - 20.0,
        cll_right=cm + 20.0,
        theta_left=np.full(n, 100.0),
        theta_right=np.full(n, 120.0),
        cm_x=cm,
        cm_y=np.full(n, 12.0),
        scales=scales,
    )


class TestGeometry:
    """Circular-cap relations and the M6 closure."""

    def test_semicircle(self):
        """A 90-degree cap is a half disc: A**2/Omega = 4/(pi/2)."""
        assert base_squared_over_area(math.pi / 2) == pytest.approx(8.0 / math.pi)

    @pytest.mark.parametrize("deg", [30.0, 60.0, 90.0, 110.0, 150.0])
    def test_cap_angle_inverts(self, deg):
        assert math.degrees(cap_angle(base_squared_over_area(math.radians(deg)))) == pytest.approx(deg, abs=1e-8)

    def test_closure_is_continuous_through_the_hold(self):
        closure = StepClosure(_WINDOWS)
        eps = 1e-9
        assert closure.base(closure.x_hold - eps) == pytest.approx(closure.base(closure.x_hold + eps), abs=1e-6)
        assert closure.base(closure.x_release - eps) == pytest.approx(closure.base(closure.x_release + eps), abs=1e-6)
        assert closure.hysteresis(closure.x_release + eps) == pytest.approx(_WINDOWS.h3)
        assert closure.hysteresis(closure.x_hold - eps) == pytest.approx(_WINDOWS.h2)

    def test_closure_hysteresis_moves_from_h2_to_h3_while_held(self):
        closure = StepClosure(_WINDOWS)
        x = np.linspace(closure.x_hold, closure.x_release, 50)
        _, h = closure.vectorised(x)
        assert np.all(np.diff(h) >= -1e-12)
        assert h[0] == pytest.approx(_WINDOWS.h2, abs=1e-3)


class TestPointTable:
    """The per-point force-balance table."""

    def test_contact_length_split_at_the_step(self):
        phob, phil = split_contact_length(np.array([70.0, 100.0, 120.0]), np.array([30.0, 60.0, 90.0]), 80.0)
        np.testing.assert_allclose(phob, [40.0, 20.0, 0.0])
        np.testing.assert_allclose(phil, [0.0, 20.0, 30.0])

    def test_regime_needs_the_trailing_line_past_the_margin(self):
        lead = np.array([70.0, 120.0, 120.0, 120.0])
        trail = np.array([30.0, 79.0, 80.0 + 0.5 * TRAIL_CROSS_MARGIN_LU, 80.0 + 2 * TRAIL_CROSS_MARGIN_LU])
        np.testing.assert_array_equal(regime_of(lead, trail, 80.0), [1, 2, 2, 3])

    def test_noise_sigma_of_white_noise(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0.0, 10.0, 20000) ** 2 / 50.0 + rng.normal(0.0, 0.03, 20000)
        assert noise_sigma(x) == pytest.approx(0.03, rel=0.05)

    def test_capillary_number_matches_the_droplet_series(self):
        """Dimensional check: Ca = eta v / gamma, the codebase's Ca_cm with rho_l = 1."""
        info = _info()
        series = _series(info, velocity=1e-3)
        df = build_point_table(info, series, AnalysisSettings(), AssumptionLog())
        interior = slice(5, -5)
        np.testing.assert_allclose(df["Ca_cm"].to_numpy()[interior], series.ca_cm[interior], rtol=1e-9)
        np.testing.assert_allclose(df["Ca_cm"], info.eta * 1e-3 / info.gamma, rtol=1e-9)
        np.testing.assert_allclose(df["inertia_cm"], 0.0, atol=1e-9)
        np.testing.assert_allclose(df["A"], 40.0 / info.r0)
        assert df["T"].iloc[1] - df["T"].iloc[0] == pytest.approx(
            info.save_interval * info.gamma / (info.eta * info.r0)
        )

    def test_hysteresis_columns(self):
        info = _info()
        df = build_point_table(info, _series(info, velocity=1e-3), AnalysisSettings(), AssumptionLog())
        assert df["H_meas"].iloc[0] == pytest.approx(_WINDOWS.h1)
        assert set(df["trail_state"]) == {"at_bound"}
        assert set(df["lead_state"]) == {"at_bound"}


class TestDimensionless:
    """Oh, Bo and Ca against the codebase's own definitions."""

    def test_oh_and_bo_match_the_registered_numbers(self):
        info = _info()
        inputs = DimensionlessInputs(
            gamma=info.gamma,
            gamma_source="analytical",
            drho=info.drho,
            drho_source="measured",
            length=info.r0,
            length_label="init_from_file",
            nu=info.eta / info.rho_l,
            rho_l=info.rho_l,
            g=info.g,
            angle_deg=info.alpha_deg,
        )
        numbers = dimensionless_for_inputs(inputs)
        assert info.oh == pytest.approx(numbers.get("oh"))
        assert info.bo == pytest.approx(numbers.get("bo_parallel"))

    def test_bo_with_rho_l_differs_by_the_density_contrast(self):
        info = _info()
        assert info.bo / info.bo_rho_l == pytest.approx(info.drho / info.rho_l)


def _synthetic_points(c: float, n_runs: int = 4, *, exponent: int = 2) -> pd.DataFrame:
    """Points that obey ``beta = c A**exponent / Omega**(exponent - 1)`` exactly.

    ``exponent = 2`` is M4's ``c A**2/Omega``; ``exponent = 1`` is M7's ``(lambda/eta) A``.
    """
    rng = np.random.default_rng(3)
    frames = []
    for k in range(n_runs):
        n = 50
        ca = rng.uniform(0.02, 0.2, n)
        a_lu = rng.uniform(30.0, 70.0, n)
        omega = 1000.0
        h = rng.uniform(-0.5, 0.3, n)
        bo = 0.8
        inertia = bo - h - c * a_lu**exponent / omega ** (exponent - 1) * ca
        frames.append(
            pd.DataFrame(
                {
                    "run": f"r{k}",
                    "Ca_cm": ca,
                    "A_lu": a_lu,
                    "Omega": omega,
                    "H_meas": h,
                    "Bo": bo,
                    "inertia_cm": inertia,
                    "moving": True,
                    "eta": 0.163,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_fit_recovers_a_known_friction_constant():
    df = _synthetic_points(c=2.7)
    fit = fit_model(df, MODEL_BY_KEY["M4"], "test", n_boot=50, rng=np.random.default_rng(0))

    assert fit.params["c_lub"] == pytest.approx(2.7, rel=1e-10)
    assert fit.ci_low["c_lub"] <= 2.7 <= fit.ci_high["c_lub"]
    assert fit.rms == pytest.approx(0.0, abs=1e-10)


def test_m7_recovers_the_si_friction_coefficient():
    df = _synthetic_points(c=0.08, exponent=1)
    fit = fit_model(df, MODEL_BY_KEY["M7"], "test", n_boot=50, rng=np.random.default_rng(0))

    assert fit.params["lambda_eta"] == pytest.approx(0.08, rel=1e-10)
    assert fit.rms == pytest.approx(0.0, abs=1e-10)


def test_point_lambda_is_beta_over_contact_length():
    df = add_point_friction(_synthetic_points(c=0.08, exponent=1))

    np.testing.assert_allclose(df["lambda_eta_eff"].dropna(), 0.08, rtol=1e-10)
    np.testing.assert_allclose(df["lambda_eff"].dropna(), 0.08 * 0.163, rtol=1e-10)


class TestConfig:
    """The analysis TOML."""

    def test_groups_and_settings(self, tmp_path):
        cfg_path = tmp_path / "fb.toml"
        cfg_path.write_text(
            '[data]\nroot = "/data"\noutput_dir = "/out"\n'
            '[groups]\na = ["x", "y"]\nb = ["y"]\n'
            "[analysis]\nsmoothing_window = 9\n",
            encoding="utf-8",
        )
        cfg = load_force_balance_config(cfg_path)

        assert cfg.settings.smoothing_window == 9
        assert cfg.membership() == {"x": ["a"], "y": ["a", "b"]}
        assert cfg.membership(["b"]) == {"y": ["b"]}

    def test_rejects_unknown_settings(self, tmp_path):
        cfg_path = tmp_path / "fb.toml"
        cfg_path.write_text(
            '[data]\nroot = "/d"\noutput_dir = "/o"\n[groups]\na = ["x"]\n[analysis]\nwindow = 3\n', encoding="utf-8"
        )
        with pytest.raises(ValueError, match="unknown"):
            load_force_balance_config(cfg_path)

    def test_rejects_unknown_group(self, tmp_path):
        cfg_path = tmp_path / "fb.toml"
        cfg_path.write_text('[data]\nroot = "/d"\noutput_dir = "/o"\n[groups]\na = ["x"]\n', encoding="utf-8")
        cfg = load_force_balance_config(cfg_path)
        with pytest.raises(ValueError, match="unknown group"):
            cfg.membership(["zz"])
