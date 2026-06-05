"""Tests for build_setup with different SimulationConfig variants.

Verifies that the setup factory correctly wires operators, forces, and
multiphase components for the main config categories without running a
full simulation.
"""

from __future__ import annotations
import pytest
from tud_lbm.config.simulation_config import SimulationConfig
from tud_lbm.pipeline.setup import build_setup

_BASE: dict = {"grid_shape": (8, 8), "tau": 0.8, "nt": 5}

# ---------------------------------------------------------------------------
# Parametrized: config → expected setup properties
# ---------------------------------------------------------------------------

_VARIANTS = [
    pytest.param(
        {},
        {"has_forces": False, "has_mp_params": False, "has_wetting_fn": False},
        id="single_phase_default",
    ),
    pytest.param(
        {"collision_scheme": "mrt", "k_diag": (1.0, 1.4, 1.4, 1.0, 1.2, 1.0, 1.2, 1.0, 1.0)},
        {"has_forces": False, "has_mp_params": False, "has_wetting_fn": False},
        id="single_phase_mrt",
    ),
    pytest.param(
        {"gravity_force": {"force_g": 1e-6, "inclination_angle_deg": 0.0}},
        {"has_forces": True, "has_mp_params": False, "has_wetting_fn": False},
        id="single_phase_gravity",
    ),
    pytest.param(
        {
            "sim_type": "multiphase",
            "eos": "double-well",
            "kappa": 0.017,
            "rho_l": 1.0,
            "rho_v": 0.33,
            "interface_width": 4,
        },
        {"has_forces": False, "has_mp_params": True, "has_wetting_fn": False},
        id="multiphase_no_force",
    ),
    pytest.param(
        {
            "sim_type": "multiphase",
            "eos": "double-well",
            "kappa": 0.017,
            "rho_l": 1.0,
            "rho_v": 0.33,
            "interface_width": 4,
            "gravity_force": {"force_g": 2e-6, "inclination_angle_deg": 0.0},
        },
        {"has_forces": True, "has_mp_params": True, "has_wetting_fn": False},
        id="multiphase_with_gravity",
    ),
    pytest.param(
        {
            "sim_type": "multiphase_wetting",
            "eos": "double-well",
            "kappa": 0.017,
            "rho_l": 1.0,
            "rho_v": 0.33,
            "interface_width": 4,
            "wetting_config": {
                "phi_left": 1.0,
                "phi_right": 1.0,
                "d_rho_left": 0.0,
                "d_rho_right": 0.0,
            },
        },
        {"has_forces": False, "has_mp_params": True, "has_wetting_fn": False},
        id="multiphase_wetting",
    ),
]


@pytest.mark.parametrize(("cfg_kwargs", "expected"), _VARIANTS)
def test_build_setup_config_variants(cfg_kwargs: dict, expected: dict) -> None:
    """build_setup correctly wires operators and optional components for each variant."""
    config = SimulationConfig(**{**_BASE, **cfg_kwargs})
    setup = build_setup(config)

    assert setup is not None
    assert setup.lattice is not None
    assert setup.step_fn is not None
    assert setup.collision_fn is not None
    assert setup.streaming_fn is not None
    assert setup.equilibrium_fn is not None

    assert (len(setup.forces.specs) > 0) == expected["has_forces"], (
        f"Force mismatch for {cfg_kwargs}: expected has_forces={expected['has_forces']}"
    )
    assert (setup.multiphase_params is not None) == expected["has_mp_params"], (
        f"Multiphase params mismatch for {cfg_kwargs}"
    )
    assert (setup.wetting_fn is not None) == expected["has_wetting_fn"], f"Wetting fn mismatch for {cfg_kwargs}"


# ---------------------------------------------------------------------------
# Wetting / BC validation
# ---------------------------------------------------------------------------


def test_build_setup_rejects_wetting_without_multiphase() -> None:
    """build_setup raises ValueError when wetting_config is set on single_phase."""
    config = SimulationConfig(
        **_BASE,
        wetting_config={"phi_left": 1.0, "phi_right": 1.0, "d_rho_left": 0.0, "d_rho_right": 0.0},
    )
    with pytest.raises(ValueError, match="Wetting"):
        build_setup(config)


def test_build_setup_grid_shape_stored_correctly() -> None:
    """build_setup stores grid_shape as a 3-tuple (nx, ny, nz)."""
    config = SimulationConfig(grid_shape=(16, 8), tau=0.8, nt=2)
    setup = build_setup(config)
    assert setup.grid_shape == (16, 8, 1)
