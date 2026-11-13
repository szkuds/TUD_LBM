"""The wetting wall separates the phases at the run's own phase densities."""

from __future__ import annotations
import numpy as np
import pytest
from src.config import SimulationConfig
from src.config.wetting import wall_rho_mean

NX, NY = 16, 8
MULTIPHASE = {"eos": "double-well", "kappa": 0.02, "rho_l": 1.0, "rho_v": 0.001, "interface_width": 2}


def test_a_restart_separates_the_phases_at_its_init_field_densities(tmp_path):
    """An equilibrated field has relaxed away from ``rho_l``/``rho_v``; the wall follows the field."""
    rho_lo, rho_hi = 0.0083, 0.956
    rho = np.full((NX, NY, 1, 1, 1), rho_lo)
    rho[: NX // 2] = rho_hi
    npz_path = tmp_path / "init_state.npz"
    np.savez(npz_path, rho=rho)
    config = SimulationConfig(
        multiphase=MULTIPHASE,
        grid_shape=(NX, NY),
        init_type="init_from_file",
        init_dir=str(npz_path),
        initialisation={},
    )

    assert wall_rho_mean(config) == pytest.approx(0.5 * (rho_lo + rho_hi))


def test_a_fresh_run_separates_the_phases_at_the_configured_densities():
    config = SimulationConfig(multiphase=MULTIPHASE, grid_shape=(NX, NY))

    assert wall_rho_mean(config) == pytest.approx(0.5005)


def test_a_single_phase_run_has_no_threshold():
    assert wall_rho_mean(SimulationConfig(grid_shape=(NX, NY))) is None
