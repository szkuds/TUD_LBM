"""Tests for tud_lbm/io/analysis/stability.py (--debug-stability diagnostics)."""

from __future__ import annotations
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import pytest
import tud_lbm.config.config_overview as _flags
from tud_lbm.config.simulation_config import SimulationConfig
from tud_lbm.io.analysis.stability import StabilityAbortError
from tud_lbm.io.analysis.stability import _host_check
from tud_lbm.io.analysis.stability import checkerboard_amplitude
from tud_lbm.io.analysis.stability import compute_stability_metrics
from tud_lbm.io.analysis.stability import wake_mask
from tud_lbm.pipeline.runner import init_state
from tud_lbm.pipeline.runner import run
from tud_lbm.pipeline.setup import build_setup
from tud_lbm.pipeline.state.state import State

# =====================================================================
# checkerboard_amplitude
# =====================================================================


def _field(values: np.ndarray) -> jnp.ndarray:
    """Lift a 2D (nx, ny) array to the 5D field convention (nx, ny, 1, 1, 1)."""
    return jnp.asarray(values)[:, :, None, None, None]


class TestCheckerboardAmplitude:
    """3x3-smoothing residual amplitude on synthetic fields."""

    def test_uniform_field_is_zero(self):
        rho = _field(np.full((8, 8), 0.7))
        mask = jnp.ones_like(rho, dtype=bool)
        assert float(checkerboard_amplitude(rho, mask)) == pytest.approx(0.0, abs=1e-12)

    def test_perfect_checkerboard_matches_analytic_amplitude(self):
        # Even dims keep the (-1)**(i+j) pattern consistent under periodic roll,
        # so the 3x3 mean is rho0 + eps/9 * (-1)**(i+j) everywhere and the
        # residual per cell is (8/9) * eps.
        nx, ny, eps = 8, 8, 1e-3
        i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        rho = _field(1.0 + eps * (-1.0) ** (i + j))
        mask = jnp.ones_like(rho, dtype=bool)
        expected = (8.0 / 9.0) * eps * np.sqrt(nx * ny)
        # rel=1e-3 keeps this valid under float32, where the rho - smooth
        # cancellation against the 1.0 baseline costs ~1e-4 relative accuracy.
        assert float(checkerboard_amplitude(rho, mask)) == pytest.approx(expected, rel=1e-3)

    def test_linear_gradient_interior_is_zero(self):
        # A linear ramp is invariant under the 3-point mean away from the
        # periodic wrap, so an interior-only mask sees zero residual.
        nx, ny = 8, 8
        x = np.arange(nx, dtype=float)
        rho = _field(np.broadcast_to(x[:, None], (nx, ny)).copy())
        interior = np.zeros((nx, ny), dtype=bool)
        interior[1:-1, 1:-1] = True
        mask = _field(interior).astype(bool)
        assert float(checkerboard_amplitude(rho, mask)) == pytest.approx(0.0, abs=1e-10)


# =====================================================================
# wake_mask
# =====================================================================


class TestWakeMask:
    """Vapor-phase + interface-exclusion mask on a tanh droplet profile."""

    def test_tanh_profile_selects_far_vapor_only(self):
        nx, rho_l, rho_v, width, x0 = 64, 1.0, 0.33, 2.0, 32.0
        x = np.arange(nx, dtype=float)
        profile = rho_v + 0.5 * (rho_l - rho_v) * (1.0 + np.tanh((x - x0) / width))
        grad = np.gradient(profile)

        rho = jnp.asarray(profile)[:, None, None, None, None]
        grad_rho = jnp.asarray(grad)[:, None, None, None, None]

        mask = np.asarray(wake_mask(rho, grad_rho, rho_l, rho_v, vapor_frac=0.2, grad_frac=0.05))
        mask = mask[:, 0, 0, 0, 0]

        assert mask[0]  # far vapor: low rho, flat gradient
        assert not mask[int(x0)]  # interface centre: steep gradient
        assert not mask[-1]  # liquid bulk: rho above vapor threshold


# =====================================================================
# compute_stability_metrics
# =====================================================================


def _state(rho, u, force=None, force_ext=None) -> State:
    f = jnp.zeros((*rho.shape[:3], 9, 1))
    return State(f=f, rho=rho, u=u, t=jnp.array(0), force=force, force_ext=force_ext)


class TestComputeStabilityMetrics:
    """Metric vector values and the grad-mu-from-force convention."""

    def test_known_fields_pin_metrics_and_force_convention(self):
        nx, ny = 4, 4
        rho = jnp.full((nx, ny, 1, 1, 1), 1.0).at[0, 0].set(0.5).at[1, 1].set(2.0)
        u = jnp.zeros((nx, ny, 1, 1, 2)).at[2, 2, 0, 0, 0].set(0.3).at[2, 2, 0, 0, 1].set(0.4)

        # Multiphase convention: force = -rho * grad_mu + force_ext.
        grad_mu = jnp.zeros((nx, ny, 1, 1, 2)).at[..., 0].set(0.1)
        force_ext = jnp.full((nx, ny, 1, 1, 2), 1e-3)
        force = -rho * grad_mu + force_ext

        metrics = np.asarray(compute_stability_metrics(_state(rho, u, force, force_ext)))
        max_u, max_grad_mu, rho_min, rho_max, cb_amp, n_wake = metrics

        assert max_u == pytest.approx(0.5)  # |(0.3, 0.4)|
        assert max_grad_mu == pytest.approx(0.1)
        assert rho_min == pytest.approx(0.5)
        assert rho_max == pytest.approx(2.0)
        assert cb_amp > 0.0  # rho has two bumps -> nonzero residual
        assert n_wake == pytest.approx(nx * ny)  # no mp -> whole-domain mask

    def test_single_phase_reports_zero_grad_mu(self):
        rho = jnp.ones((4, 4, 1, 1, 1))
        u = jnp.zeros((4, 4, 1, 1, 2))
        metrics = np.asarray(compute_stability_metrics(_state(rho, u)))
        assert metrics[1] == pytest.approx(0.0)


# =====================================================================
# _host_check (CSV writing + NaN guard, no JIT)
# =====================================================================


class TestHostCheck:
    """Host-side CSV writing and the NaN guard."""

    BENIGN = np.array([0.01, 0.02, 0.33, 1.0, 1e-6, 42.0])

    def test_csv_created_with_header_and_appended(self, tmp_path):
        _host_check(tmp_path, self.BENIGN, 100)
        _host_check(tmp_path, self.BENIGN, 200)

        lines = (tmp_path / "stability_log.csv").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        assert lines[0] == "t,max_u,max_grad_mu,rho_min,rho_max,checkerboard_amp,n_wake_cells"
        assert lines[1].startswith("100,")
        assert lines[2].startswith("200,")

    def test_nan_metric_raises_after_logging(self, tmp_path):
        bad = self.BENIGN.copy()
        bad[0] = np.nan
        with pytest.raises(StabilityAbortError, match="NaN in stability metrics at t=300"):
            _host_check(tmp_path, bad, 300)
        # The row is still written before the abort so the curve ends at the failure.
        assert (tmp_path / "stability_log.csv").exists()


# =====================================================================
# Integration through run()
# =====================================================================


class TestRunIntegration:
    """End-to-end stability logging through run() with the flag set."""

    def _setup(self, tmp_path):
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10, results_dir=str(tmp_path))
        return build_setup(cfg)

    def test_in_memory_mode_writes_csv_rows_at_save_interval(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_flags, "DEBUG_FLAG_STABILITY", True)
        setup = self._setup(tmp_path)
        state = init_state(setup)

        run(setup, state, nt=10, save_interval=2)

        csv_path = tmp_path / "stability_debug" / "stability_log.csv"
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        # t runs 1..10; samples at t % 2 == 0 -> 5 rows + header.
        assert len(lines) == 6
        sampled_t = [int(line.split(",")[0]) for line in lines[1:]]
        assert sampled_t == [2, 4, 6, 8, 10]
        for line in lines[1:]:
            values = [float(v) for v in line.split(",")[1:]]
            assert all(np.isfinite(values))

    def test_streaming_mode_writes_csv_to_run_dir(self, tmp_path, monkeypatch):
        from tud_lbm.io import SimulationIO

        monkeypatch.setattr(_flags, "DEBUG_FLAG_STABILITY", True)
        setup = self._setup(tmp_path)
        state = init_state(setup)
        io = SimulationIO(base_dir=str(tmp_path), output_format="numpy")

        run(setup, state, nt=6, save_interval=2, io_handler=io)

        assert (Path(io.run_dir) / "stability_log.csv").exists()
        snapshots = [p.name for p in Path(io.data_dir).iterdir()]
        assert any(name.endswith(".npz") for name in snapshots)

    def test_flag_off_writes_nothing(self, tmp_path):
        setup = self._setup(tmp_path)
        state = init_state(setup)

        run(setup, state, nt=4, save_interval=2)

        assert not (tmp_path / "stability_debug").exists()
