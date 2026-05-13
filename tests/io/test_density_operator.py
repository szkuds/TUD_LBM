"""Branch tests for density plotting operator."""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from tud_lbm.config import SimulationConfig
from tud_lbm.io.plotting.density import DensityPlotOperator


def test_density_operator_uses_log_scale_for_large_multiphase_ratio():
    cfg = SimulationConfig(
        sim_type="multiphase",
        grid_shape=(8, 8),
        eos="double-well",
        kappa=0.02,
        rho_l=1000.0,
        rho_v=1.0,
        interface_width=2,
    )
    op = DensityPlotOperator(cfg)
    rng = np.random.default_rng(123)
    data = {"rho": np.abs(rng.random((8, 8, 1, 1, 1))) + 1e-3}

    fig, ax = plt.subplots()
    try:
        op(ax, data, timestep=3)
        assert "(log)" in ax.get_title()
    finally:
        plt.close(fig)


def test_density_operator_non_multiphase_uses_linear_scale():
    cfg = SimulationConfig(grid_shape=(8, 8))
    op = DensityPlotOperator(cfg)
    data = {"rho": np.ones((8, 8, 1, 1, 1))}

    fig, ax = plt.subplots()
    try:
        op(ax, data, timestep=4)
        assert "(log)" not in ax.get_title()
    finally:
        plt.close(fig)
