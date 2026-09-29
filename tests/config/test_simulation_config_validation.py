"""Tests for SimulationConfig validation error branches.

Covers every raise statement in _validate_common and _validate_multiphase:
- _validate_forces: conflicting gravity forces
- _validate_grid_shape: too few dimensions, zero/negative dimension
- _validate_tau: tau at/below minimum
- _validate_time_steps: nt <= 0
- _validate_collision: invalid scheme, MRT without k_diag
- _validate_init: init_from_file without init_dir
- _validate_save_fields: invalid field name
- _validate_hysteresis: unknown [hysteresis] keys
- _validate_multiphase: each missing required field, rho_l <= rho_v,
                         invalid eos, carnahan-starling missing CS params
"""

from __future__ import annotations
from typing import Any
import pytest
import src.operators.initialise  # noqa: F401 - registers the init types parametrised below
from src.config.simulation_config import SimulationConfig
from src.registry import get_operator_names

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

    def test_registered_defaults_are_filled_in(self):
        cfg = SimulationConfig(grid_shape=(8, 8), gravity_masked_force={"force_g": 1e-6})
        assert cfg.gravity_masked_force == {"force_g": 1e-6, "inclination_angle_deg": 0.0, "ramp_start_t": 0.0}

    def test_configured_values_win_over_defaults(self):
        cfg = SimulationConfig(grid_shape=(8, 8), gravity_force={"force_g": 1e-6, "inclination_angle_deg": 30.0})
        assert cfg.gravity_force == {"force_g": 1e-6, "inclination_angle_deg": 30.0}

    def test_missing_required_key_raises(self):
        with pytest.raises(ValueError, match=r"\[gravity_force\] is missing required key\(s\): force_g"):
            SimulationConfig(grid_shape=(8, 8), gravity_force={"inclination_angle_deg": 30.0})

    def test_unknown_key_raises(self):
        """A key no force reads — e.g. ``ramp_steps`` on plain gravity — is rejected, not ignored."""
        with pytest.raises(ValueError, match=r"\[gravity_force\] has unknown key\(s\): ramp_steps"):
            SimulationConfig(grid_shape=(8, 8), gravity_force={"force_g": 1e-6, "ramp_steps": 100})

    def test_non_positive_ramp_steps_raises(self):
        with pytest.raises(ValueError, match="ramp_steps must be positive"):
            SimulationConfig(grid_shape=(8, 8), gravity_masked_force={"force_g": 1e-6, "ramp_steps": 0})

    def test_electric_schema_follows_electric_params(self):
        cfg = SimulationConfig(
            grid_shape=(8, 8),
            electric_force={
                "permittivity_liquid": 80.0,
                "permittivity_vapour": 1.0,
                "conductivity_liquid": 0.01,
                "conductivity_vapour": 0.001,
            },
        )
        assert cfg.electric_force is not None
        assert cfg.electric_force["voltage_top"] == 0.0

    def test_active_forces_lists_only_configured_sections(self):
        cfg = SimulationConfig(grid_shape=(8, 8), gravity_force={"force_g": 1e-6})
        assert list(cfg.active_forces) == ["gravity_force"]
        assert cfg.force_enabled
        assert not SimulationConfig(grid_shape=(8, 8)).force_enabled


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
    """Tests for _validate_init: a registered init_type; init_from_file requires init_dir."""

    def test_init_from_file_without_init_dir_raises(self):
        with pytest.raises(ValueError, match="init_dir must be provided"):
            SimulationConfig(grid_shape=(8, 8), tau=0.99, nt=10, init_type="init_from_file")

    def test_unknown_init_type_raises(self):
        with pytest.raises(ValueError, match="init_type must be one of"):
            SimulationConfig(grid_shape=(8, 8), init_type="not_an_init")

    @pytest.mark.parametrize("init_type", sorted(get_operator_names("initialise") - {"init_from_file"}))
    def test_every_registered_init_type_is_accepted(self, init_type):
        assert SimulationConfig(grid_shape=(8, 8), init_type=init_type).init_type == init_type


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
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.99, nt=10, save_fields=["rho", "u", "f", "pressure"])
        assert cfg.save_fields == ["rho", "u", "f", "pressure"]

    def test_populations_always_saved(self):
        """An explicit list without ``f`` gains it: pressure is computed from the populations."""
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.99, nt=10, save_fields=["rho", "u"])
        assert cfg.save_fields == ["f", "rho", "u"]

    def test_unset_save_fields_saves_everything(self):
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.99, nt=10)
        assert cfg.save_fields is None


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
        assert cfg.obstacle_config == {"center_x": 20, "center_y": 10, "radius": 5, "shape": "circle"}

    def test_unknown_obstacle_shape_raises(self):
        with pytest.raises(ValueError, match="obstacle shape must be one of"):
            SimulationConfig(
                grid_shape=(40, 20, 1),
                obstacle_config={"shape": "square", "center_x": 20, "center_y": 10, "radius": 5},
            )

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


class TestValidateBoundaryConditions:
    """bc_config is completed and checked, so build_bc only looks up and binds."""

    def test_missing_edges_default_to_periodic(self):
        cfg = SimulationConfig(grid_shape=(20, 20, 1), bc_config={"top": "bounce-back"})
        assert cfg.bc_config is not None
        assert cfg.bc_config["bottom"] == "periodic"
        assert cfg.bc_config["back"] == "periodic"

    def test_unknown_bc_name_raises(self):
        with pytest.raises(ValueError, match=r"bc_config\['left'\] must be one of"):
            SimulationConfig(grid_shape=(20, 20, 1), bc_config={"left": "bounceback"})

    def test_boundary_edges_follow_application_order(self):
        cfg = SimulationConfig(grid_shape=(20, 20, 1), bc_config={"top": "symmetry", "bottom": "bounce-back"})
        assert [(e.edge, e.name) for e in cfg.boundary_edges] == [
            ("bottom", "bounce-back"),
            ("top", "symmetry"),
            ("left", "periodic"),
            ("right", "periodic"),
        ]

    def test_edge_without_parameter_section_has_empty_params(self):
        cfg = SimulationConfig(grid_shape=(20, 20, 1), bc_config={"left": "velocity-inlet", "right": "outlet"})
        left = next(e for e in cfg.boundary_edges if e.edge == "left")
        assert left.params == {}

    def test_parameter_section_is_bound_to_its_edge(self):
        cfg = SimulationConfig(
            grid_shape=(20, 20, 1),
            bc_config={"left": "velocity-inlet", "right": "outlet", "left_velocity_inlet": {"u0": 0.02}},
        )
        left = next(e for e in cfg.boundary_edges if e.edge == "left")
        assert left.params == {"u0": 0.02}

    def test_parameter_section_for_another_bc_raises(self):
        with pytest.raises(ValueError, match=r"bc_config\['left_velocity_inlet'\] is not the parameter section"):
            SimulationConfig(
                grid_shape=(20, 20, 1),
                bc_config={"left": "bounce-back", "left_velocity_inlet": {"u0": 0.02}},
            )


# ---------------------------------------------------------------------------
# Wetting-wall resolution (measurement orientation)
# ---------------------------------------------------------------------------


class TestWettingWallConfig:
    """Wetting walls are accepted on any edge, and several at once.

    Measurement orients from the first wetting wall (see
    ``SimulationSetup.wetting_edge``); multiple wetting walls are permitted and
    share one ``WettingState`` parameter pair.
    """

    @staticmethod
    def _wetting_config(**bc: str) -> dict[str, Any]:
        return {
            **_DW_BASE,
            "sim_type": "multiphase_wetting",
            "wetting_config": {"advancing_ca": 100.0},
            "bc_config": bc,
        }

    def test_top_wetting_wall_is_valid(self):
        cfg = SimulationConfig(**self._wetting_config(top="wetting"))
        assert cfg.bc_config is not None
        assert cfg.bc_config["top"] == "wetting"

    def test_two_wetting_walls_are_valid(self):
        cfg = SimulationConfig(**self._wetting_config(bottom="wetting", top="wetting"))
        assert cfg.bc_config is not None
        assert cfg.bc_config["bottom"] == "wetting"
        assert cfg.bc_config["top"] == "wetting"

    def test_zero_wetting_walls_is_permissive(self):
        cfg = SimulationConfig(**self._wetting_config())
        assert cfg.sim_type == "multiphase_wetting"


# ---------------------------------------------------------------------------
# Configuration-derived data (consumed by the operator factories)
# ---------------------------------------------------------------------------


class TestDerivedProperties:
    """Static per-run data the operator packages bind instead of deriving themselves."""

    def test_periodic_axes_need_both_edges_periodic(self):
        cfg = SimulationConfig(grid_shape=(20, 20, 1), bc_config={"top": "bounce-back", "bottom": "bounce-back"})
        assert cfg.periodic_axes == (True, False)

    def test_pad_modes_follow_each_bc_registration(self):
        cfg = SimulationConfig(grid_shape=(20, 20, 1), bc_config={"top": "symmetry", "bottom": "bounce-back"})
        assert cfg.pad_modes == ("edge", "edge", "wrap", "wrap")

    def test_obstacle_mask_is_none_without_obstacle(self):
        assert SimulationConfig(grid_shape=(20, 20, 1)).obstacle_mask is None

    def test_obstacle_mask_marks_the_solid_cells(self):
        cfg = SimulationConfig(
            grid_shape=(40, 20, 1),
            bc_config={"left": "bounce-back", "right": "bounce-back"},
            obstacle_config={"center_x": 20, "center_y": 10, "radius": 5},
        )
        mask = cfg.obstacle_mask
        assert mask is not None
        assert mask.shape == (40, 20, 1, 1, 1)
        assert bool(mask[20, 10, 0, 0, 0])
        assert not bool(mask[0, 0, 0, 0, 0])

    def test_wetting_defaults_are_none_without_wetting(self):
        assert SimulationConfig(grid_shape=(20, 20, 1)).wetting_defaults is None

    def test_wetting_defaults_read_legacy_keys_and_fill_neutral_values(self):
        params: dict[str, Any] = {
            **_DW_BASE,
            "sim_type": "multiphase_wetting",
            "wetting_config": {"phi_l": 1.2},
            "bc_config": {"bottom": "wetting"},
        }
        cfg = SimulationConfig(**params)
        assert cfg.wetting_defaults == {"phi_left": 1.2, "phi_right": 1.0, "d_rho_left": 0.0, "d_rho_right": 0.0}


# ---------------------------------------------------------------------------
# _validate_hysteresis and the hysteresis / chemical-step defaults
# ---------------------------------------------------------------------------


def _hysteresis_config(
    hysteresis: dict[str, Any],
    *,
    sim_type: str = "multiphase_hysteresis",
    chemical_step_config: dict[str, Any] | None = None,
) -> SimulationConfig:
    base = {k: v for k, v in _DW_BASE.items() if k != "sim_type"}
    return SimulationConfig(
        **base,
        sim_type=sim_type,  # ty: ignore[invalid-argument-type]
        bc_config={"bottom": "wetting"},
        hysteresis_config=hysteresis,
        chemical_step_config=chemical_step_config,
    )


class TestValidateHysteresis:
    """Unknown [hysteresis] keys raise; the optimiser settings are defaulted."""

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown \\[hysteresis\\] keys"):
            _hysteresis_config({"ca_advancing": 120.0, "ca_receding": 60.0, "ca_dead_zone": 0.5})

    @pytest.mark.parametrize(
        ("key", "value"), [("carry_inactive_params", True), ("learning_rate_above", 0.1), ("max_iterations_above", 7)]
    )
    def test_optimiser_keys_are_accepted(self, key, value):
        cfg = _hysteresis_config({"ca_advancing": 120.0, "ca_receding": 60.0, key: value})
        assert cfg.hysteresis_config is not None
        assert cfg.hysteresis_config[key] == value

    def test_max_iterations_above_follows_max_iterations(self):
        cfg = _hysteresis_config({"ca_advancing": 120.0, "ca_receding": 60.0, "max_iterations": 12})
        assert cfg.hysteresis_config is not None
        assert cfg.hysteresis_config["max_iterations_above"] == 12

    def test_optimiser_settings_are_defaulted(self):
        cfg = _hysteresis_config({"ca_advancing": 120.0, "ca_receding": 60.0, "learning_rate": 0.05})
        assert cfg.hysteresis_config is not None
        assert cfg.hysteresis_config["learning_rate"] == 0.05
        assert cfg.hysteresis_config["max_iterations"] == 50
        assert cfg.hysteresis_config["loss_tol"] == 1e-4
        assert cfg.hysteresis_config["trial_steps"] == 2
        assert cfg.hysteresis_config["learning_rate_above"] == 0.05
        assert cfg.hysteresis_config["carry_inactive_params"] is False
        assert cfg.hysteresis_config["saturation_gap"] == 1.0

    def test_chemical_step_edge_width_defaults_to_one_cell(self):
        chemical_step = {
            "chemical_step_location": 0.5,
            "ca_advancing_pre_step": 120.0,
            "ca_receding_pre_step": 100.0,
            "ca_advancing_post_step": 70.0,
            "ca_receding_post_step": 50.0,
        }
        cfg = _hysteresis_config({}, sim_type="multiphase_hysteresis_chemical_step", chemical_step_config=chemical_step)
        assert cfg.chemical_step_config is not None
        assert cfg.chemical_step_config["edge_width"] == 1.0


_SAVED_WETTING = {
    "phi_left": 1.0,
    "phi_right": 1.75,
    "d_rho_left": 0.12,
    "d_rho_right": 0.0,
    "cll_left": 48.4,
    "cll_right": 101.6,
}


class TestRestoredWetting:
    """A hysteresis restart resumes the wetting state saved in its init NPZ."""

    def _npz(self, tmp_path, **arrays):
        import numpy as np

        path = tmp_path / "timestep_58000.npz"
        np.savez(path, rho=np.ones((8, 8, 1, 1, 1)), **arrays)
        return str(path)

    def _config(self, init_dir: str, *, hysteresis: bool = True) -> SimulationConfig:
        base = {k: v for k, v in _DW_BASE.items() if k != "sim_type"}
        return SimulationConfig(
            **base,
            sim_type="multiphase_hysteresis" if hysteresis else "multiphase_wetting",
            bc_config={"bottom": "wetting"},
            wetting_config={"phi_left": 1.0},
            hysteresis_config={"ca_advancing": 120.0, "ca_receding": 60.0} if hysteresis else None,
            init_type="init_from_file",
            init_dir=init_dir,
        )

    def test_hysteresis_restart_reads_the_snapshot(self, tmp_path):
        cfg = self._config(self._npz(tmp_path, **_SAVED_WETTING))
        assert cfg.restored_wetting == _SAVED_WETTING

    def test_snapshot_without_wetting_state_is_a_fresh_start(self, tmp_path):
        cfg = self._config(self._npz(tmp_path))
        assert cfg.restored_wetting is None

    def test_fixed_wetting_run_keeps_its_configured_parameters(self, tmp_path):
        cfg = self._config(self._npz(tmp_path, **_SAVED_WETTING), hysteresis=False)
        assert cfg.restored_wetting is None


_STEP = {
    "chemical_step_location": 0.5,
    "ca_advancing_pre_step": 120.0,
    "ca_receding_pre_step": 100.0,
    "ca_advancing_post_step": 50.0,
    "ca_receding_post_step": 30.0,
}


class TestChemicalStepWall:
    """The stepped wall the wetting applicator splits by surface."""

    def test_pre_surface_takes_the_wetting_config_and_post_its_pull_limit(self):
        cfg = _hysteresis_config({}, sim_type="multiphase_hysteresis_chemical_step", chemical_step_config=dict(_STEP))
        wall = cfg.chemical_step_wall
        assert wall is not None
        assert (wall.edge, wall.step_x) == ("bottom", 4.0)  # 0.5 * nx=8
        assert (wall.pre_phi_left, wall.pre_d_rho_left) == (1.0, 0.0)  # neutral [wetting] defaults
        assert (wall.post_phi, wall.post_d_rho) == (1.0 + 5 / 4, 0.0)  # W = 4: phi limit 1 + 5/W

    def test_a_less_wetting_post_surface_pushes_instead(self):
        step = dict(_STEP, ca_advancing_post_step=130.0, ca_receding_post_step=110.0)
        cfg = _hysteresis_config({}, sim_type="multiphase_hysteresis_chemical_step", chemical_step_config=step)
        wall = cfg.chemical_step_wall
        assert wall is not None
        assert (wall.post_phi, wall.post_d_rho) == (1.0, 1.5 / 4)

    def test_no_step_no_wall(self):
        assert _hysteresis_config({}).chemical_step_wall is None
