"""Tests for SimulationConfig validation error branches.

Covers every raise statement in _validate_common and _validate_multiphase:
- _validate_forces: conflicting gravity forces
- _validate_grid_shape: too few dimensions, zero/negative dimension
- _validate_tau: tau at/below minimum
- _validate_time_steps: nt <= 0
- _validate_collision: invalid scheme, MRT without k_diag
- _validate_init: init_from_file without init_dir
- _validate_save_fields: invalid field name
- _validate_multiphase: each missing required field, rho_l <= rho_v,
                         invalid eos, carnahan-starling missing CS params
"""

from __future__ import annotations
from typing import Any
import pytest
from src.config.simulation_config import SimulationConfig

# ---------------------------------------------------------------------------
# Base configs shared across parametrized tests
# ---------------------------------------------------------------------------

_DW_BASE: dict[str, Any] = {
    "sim_type": "multiphase",
    "grid_shape": (8, 8),
    "tau": 0.99,
    "nt": 3,
    "eos": "double-well",
    "kappa": 0.017,
    "rho_l": 1.0,
    "rho_v": 0.33,
    "interface_width": 4,
}

_CS_BASE: dict[str, Any] = {
    "sim_type": "multiphase",
    "grid_shape": (8, 8),
    "tau": 0.99,
    "nt": 3,
    "eos": "carnahan-starling",
    "kappa": 0.017,
    "rho_l": 1.0,
    "rho_v": 0.33,
    "interface_width": 4,
    "a_eos": 1.0,
    "b_eos": 4.0,
    "r_eos": 1.0,
    "t_eos": 0.9,
}


# ---------------------------------------------------------------------------
# _validate_forces
# ---------------------------------------------------------------------------


class TestValidateForces:
    """Tests for _validate_forces: conflicting gravity force config."""

    def test_both_gravity_forces_raises(self):
        with pytest.raises(ValueError, match="Only one gravity force can be applied"):
            SimulationConfig(
                grid_shape=(8, 8),
                tau=0.99,
                nt=10,
                gravity_force={"force_g": 1e-6, "inclination_angle_deg": 0.0},
                gravity_masked_force={"force_g": 1e-6, "inclination_angle_deg": 0.0},
            )


# ---------------------------------------------------------------------------
# _validate_grid_shape
# ---------------------------------------------------------------------------


class TestValidateGridShape:
    """Tests for _validate_grid_shape: zero and negative dimensions."""

    def test_zero_dimension_raises(self):
        with pytest.raises(ValueError, match="positive"):
            SimulationConfig(grid_shape=(0, 8), tau=0.99, nt=10)

    def test_negative_dimension_raises(self):
        with pytest.raises(ValueError, match="positive"):
            SimulationConfig(grid_shape=(-4, 8), tau=0.99, nt=10)


# ---------------------------------------------------------------------------
# _validate_tau
# ---------------------------------------------------------------------------


class TestValidateTau:
    """Tests for _validate_tau: tau at or below the stability minimum."""

    def test_tau_exactly_minimum_raises(self):
        with pytest.raises(ValueError, match="tau must be"):
            SimulationConfig(grid_shape=(8, 8), tau=0.5, nt=10)

    def test_tau_below_minimum_raises(self):
        with pytest.raises(ValueError, match="tau must be"):
            SimulationConfig(grid_shape=(8, 8), tau=0.3, nt=10)


# ---------------------------------------------------------------------------
# _validate_time_steps
# ---------------------------------------------------------------------------


class TestValidateTimeSteps:
    """Tests for _validate_time_steps: non-positive nt."""

    def test_nt_zero_raises(self):
        with pytest.raises(ValueError, match="nt must be positive"):
            SimulationConfig(grid_shape=(8, 8), tau=0.99, nt=0)

    def test_nt_negative_raises(self):
        with pytest.raises(ValueError, match="nt must be positive"):
            SimulationConfig(grid_shape=(8, 8), tau=0.99, nt=-5)


# ---------------------------------------------------------------------------
# _validate_collision
# ---------------------------------------------------------------------------


class TestValidateCollision:
    """Tests for _validate_collision: bad scheme and MRT without k_diag."""

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="collision_scheme must be one of"):
            SimulationConfig(grid_shape=(8, 8), tau=0.99, nt=10, collision_scheme="unknown")

    def test_mrt_without_k_diag_raises(self):
        with pytest.raises(ValueError, match="k_diag must be provided"):
            SimulationConfig(grid_shape=(8, 8), tau=0.99, nt=10, collision_scheme="mrt")


# ---------------------------------------------------------------------------
# _validate_init
# ---------------------------------------------------------------------------


class TestValidateInit:
    """Tests for _validate_init: init_from_file requires init_dir."""

    def test_init_from_file_without_init_dir_raises(self):
        with pytest.raises(ValueError, match="init_dir must be provided"):
            SimulationConfig(grid_shape=(8, 8), tau=0.99, nt=10, init_type="init_from_file")


# ---------------------------------------------------------------------------
# _validate_save_fields
# ---------------------------------------------------------------------------


class TestValidateSaveFields:
    """Tests for _validate_save_fields: invalid field names."""

    def test_invalid_field_name_raises(self):
        with pytest.raises(ValueError, match="Invalid save_fields"):
            SimulationConfig(grid_shape=(8, 8), tau=0.99, nt=10, save_fields=["rho", "nonexistent"])

    def test_multiple_invalid_fields_raises(self):
        with pytest.raises(ValueError, match="Invalid save_fields"):
            SimulationConfig(grid_shape=(8, 8), tau=0.99, nt=10, save_fields=["foo", "bar"])

    def test_valid_save_fields_accepted(self):
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.99, nt=10, save_fields=["rho", "u", "f"])
        assert cfg.save_fields == ["rho", "u", "f"]


# ---------------------------------------------------------------------------
# _validate_multiphase
# ---------------------------------------------------------------------------


class TestValidateMultiphase:
    """Tests for _validate_multiphase: required fields, density ordering, EOS, CS params."""

    @pytest.mark.parametrize("missing_field", ["kappa", "rho_l", "rho_v", "interface_width", "eos"])
    def test_missing_required_field_raises(self, missing_field: str):
        params: dict[str, Any] = {**_DW_BASE}
        del params[missing_field]
        with pytest.raises(ValueError, match=f"'{missing_field}' is required"):
            SimulationConfig(**params)

    def test_rho_l_not_greater_than_rho_v_raises(self):
        with pytest.raises(ValueError, match="rho_l"):
            SimulationConfig(
                sim_type="multiphase",
                grid_shape=(8, 8),
                tau=0.99,
                nt=3,
                eos="double-well",
                kappa=0.017,
                rho_l=0.2,
                rho_v=0.8,
                interface_width=4,
            )

    def test_rho_l_equal_to_rho_v_raises(self):
        with pytest.raises(ValueError, match="rho_l"):
            SimulationConfig(
                sim_type="multiphase",
                grid_shape=(8, 8),
                tau=0.99,
                nt=3,
                eos="double-well",
                kappa=0.017,
                rho_l=0.5,
                rho_v=0.5,
                interface_width=4,
            )

    def test_invalid_eos_raises(self):
        with pytest.raises(ValueError, match="eos must be one of"):
            SimulationConfig(
                sim_type="multiphase",
                grid_shape=(8, 8),
                tau=0.99,
                nt=3,
                eos="unknown-eos",
                kappa=0.017,
                rho_l=1.0,
                rho_v=0.33,
                interface_width=4,
            )

    @pytest.mark.parametrize("missing_cs_param", ["a_eos", "b_eos", "r_eos", "t_eos"])
    def test_carnahan_starling_missing_param_raises(self, missing_cs_param: str):
        params: dict[str, Any] = {**_CS_BASE}
        del params[missing_cs_param]
        with pytest.raises(ValueError, match=f"'{missing_cs_param}' is required"):
            SimulationConfig(**params)


class TestValidateObstacle:
    """Tests for _validate_obstacle: geometry sanity and BC-edge clearance."""

    def test_none_config_is_valid(self):
        cfg = SimulationConfig(grid_shape=(40, 20, 1), obstacle_config=None)
        assert cfg.obstacle_config is None

    def test_valid_obstacle_round_trips(self):
        cfg = SimulationConfig(
            grid_shape=(40, 20, 1),
            obstacle_config={"center_x": 20, "center_y": 10, "radius": 5},
        )
        assert cfg.obstacle_config == {"center_x": 20, "center_y": 10, "radius": 5}

    def test_nonpositive_radius_raises(self):
        with pytest.raises(ValueError, match="radius must be positive"):
            SimulationConfig(
                grid_shape=(40, 20, 1),
                obstacle_config={"center_x": 20, "center_y": 10, "radius": 0},
            )

    def test_obstacle_touching_top_bottom_wall_raises(self):
        with pytest.raises(ValueError, match="clearance from top/bottom"):
            SimulationConfig(
                grid_shape=(40, 20, 1),
                obstacle_config={"center_x": 20, "center_y": 5, "radius": 5},
            )

    def test_obstacle_outside_x_extent_raises(self):
        with pytest.raises(ValueError, match="x-extent"):
            SimulationConfig(
                grid_shape=(40, 20, 1),
                obstacle_config={"center_x": 2, "center_y": 10, "radius": 5},
            )

    def test_obstacle_3d_grid_raises(self):
        with pytest.raises(ValueError, match="2D"):
            SimulationConfig(
                grid_shape=(40, 20, 4),
                obstacle_config={"center_x": 20, "center_y": 10, "radius": 5},
            )

    def test_obstacle_overlapping_nonperiodic_left_edge_raises(self):
        with pytest.raises(ValueError, match="left edge"):
            SimulationConfig(
                grid_shape=(40, 20, 1),
                bc_config={"left": "bounce-back"},
                obstacle_config={"center_x": 6, "center_y": 10, "radius": 5},
            )

    def test_obstacle_overlapping_nonperiodic_right_edge_raises(self):
        with pytest.raises(ValueError, match="right edge"):
            SimulationConfig(
                grid_shape=(40, 20, 1),
                bc_config={"right": "bounce-back"},
                obstacle_config={"center_x": 34, "center_y": 10, "radius": 5},
            )

    def test_obstacle_far_from_nonperiodic_edges_is_valid(self):
        cfg = SimulationConfig(
            grid_shape=(40, 20, 1),
            bc_config={"left": "bounce-back", "right": "bounce-back"},
            obstacle_config={"center_x": 20, "center_y": 10, "radius": 5},
        )
        assert cfg.obstacle_config is not None
