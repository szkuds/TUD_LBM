"""SimulationConfig validation: rejected configurations, defaults and derived properties."""

from __future__ import annotations
from typing import Any
import pytest
import src.operators.initialise  # noqa: F401 - registers the init types parametrised below
from src.config.boundary import boundary_edges
from src.config.boundary import pad_modes
from src.config.boundary import periodic_axes
from src.config.obstacle import obstacle_mask
from src.config.simulation_config import SimulationConfig
from src.config.wetting import chemical_step_wall
from src.config.wetting import restored_wetting
from src.config.wetting import wetting_scalars
from src.registry import get_operator_names

# ---------------------------------------------------------------------------
# Base configs shared across parametrized tests
# ---------------------------------------------------------------------------

_DW_BASE: dict[str, Any] = {
    "multiphase": {"eos": "double-well", "kappa": 0.017, "rho_l": 1.0, "rho_v": 0.33, "interface_width": 4},
    "grid_shape": (8, 8),
    "tau": 0.99,
    "nt": 3,
}

_CS_BASE: dict[str, Any] = {
    "multiphase": {
        "eos": "carnahan-starling",
        "kappa": 0.017,
        "rho_l": 1.0,
        "rho_v": 0.33,
        "interface_width": 4,
        "a_eos": 1.0,
        "b_eos": 4.0,
        "r_eos": 1.0,
        "t_eos": 0.9,
    },
    "grid_shape": (8, 8),
    "tau": 0.99,
    "nt": 3,
}


# ---------------------------------------------------------------------------
# Invalid configurations: one constructor call, one ValueError
# ---------------------------------------------------------------------------

_INVALID = [
    pytest.param(
        {
            "forces": {
                "gravity_force": {"force_g": 1e-06, "inclination_angle_deg": 0.0},
                "gravity_masked_force": {"force_g": 1e-06, "inclination_angle_deg": 0.0},
            },
            "grid_shape": (8, 8),
            "tau": 0.99,
            "nt": 10,
        },
        "Only one gravity force can be applied",
        id="both_gravity_forces",
    ),
    pytest.param(
        {"forces": {"gravity_force": {"inclination_angle_deg": 30.0}}, "grid_shape": (8, 8)},
        r"\[gravity_force\] is missing required key\(s\): force_g",
        id="missing_required_key",
    ),
    pytest.param(
        {"forces": {"gravity_force": {"force_g": 1e-06, "ramp_steps": 100}}, "grid_shape": (8, 8)},
        r"\[gravity_force\] has unknown key\(s\): ramp_steps",
        id="unknown_key",
    ),
    pytest.param(
        {"forces": {"gravity_masked_force": {"force_g": 1e-06, "ramp_steps": 0}}, "grid_shape": (8, 8)},
        "ramp_steps must be positive",
        id="non_positive_ramp_steps",
    ),
    pytest.param({"grid_shape": (0, 8), "tau": 0.99, "nt": 10}, "positive", id="zero_dimension"),
    pytest.param({"grid_shape": (-4, 8), "tau": 0.99, "nt": 10}, "positive", id="negative_dimension"),
    pytest.param({"grid_shape": (8, 8), "tau": 0.5, "nt": 10}, "tau must be", id="tau_exactly_minimum"),
    pytest.param({"grid_shape": (8, 8), "tau": 0.3, "nt": 10}, "tau must be", id="tau_below_minimum"),
    pytest.param({"grid_shape": (8, 8), "tau": 0.99, "nt": 0}, "nt must be positive", id="nt_zero"),
    pytest.param({"grid_shape": (8, 8), "tau": 0.99, "nt": -5}, "nt must be positive", id="nt_negative"),
    pytest.param(
        {"grid_shape": (8, 8), "tau": 0.99, "nt": 10, "collision_scheme": "unknown"},
        "collision_scheme must be one of",
        id="invalid_scheme",
    ),
    pytest.param(
        {"grid_shape": (8, 8), "tau": 0.99, "nt": 10, "collision_scheme": "mrt"},
        "k_diag must be provided",
        id="mrt_without_k_diag",
    ),
    pytest.param(
        {"grid_shape": (8, 8), "tau": 0.99, "nt": 10, "init_type": "init_from_file"},
        "init_dir must be provided",
        id="init_from_file_without_init_dir",
    ),
    pytest.param(
        {"grid_shape": (8, 8), "init_type": "not_an_init"}, "init_type must be one of", id="unknown_init_type"
    ),
    pytest.param(
        {"grid_shape": (8, 8), "tau": 0.99, "nt": 10, "save_fields": ["rho", "nonexistent"]},
        "Invalid save_fields",
        id="invalid_field_name",
    ),
    pytest.param(
        {"grid_shape": (8, 8), "tau": 0.99, "nt": 10, "save_fields": ["foo", "bar"]},
        "Invalid save_fields",
        id="multiple_invalid_fields",
    ),
    pytest.param(
        {
            "multiphase": {"eos": "double-well", "kappa": 0.017, "rho_l": 0.2, "rho_v": 0.8, "interface_width": 4},
            "grid_shape": (8, 8),
            "tau": 0.99,
            "nt": 3,
        },
        "rho_l",
        id="rho_l_not_greater_than_rho_v",
    ),
    pytest.param(
        {
            "multiphase": {"eos": "double-well", "kappa": 0.017, "rho_l": 0.5, "rho_v": 0.5, "interface_width": 4},
            "grid_shape": (8, 8),
            "tau": 0.99,
            "nt": 3,
        },
        "rho_l",
        id="rho_l_equal_to_rho_v",
    ),
    pytest.param(
        {
            "multiphase": {"eos": "unknown-eos", "kappa": 0.017, "rho_l": 1.0, "rho_v": 0.33, "interface_width": 4},
            "grid_shape": (8, 8),
            "tau": 0.99,
            "nt": 3,
        },
        "eos must be one of",
        id="invalid_eos",
    ),
    pytest.param(
        {
            "grid_shape": (40, 20, 1),
            "obstacle": {"shape": "square", "center_x": 20, "center_y": 10, "radius": 5},
        },
        "obstacle shape must be one of",
        id="unknown_obstacle_shape",
    ),
    pytest.param(
        {"grid_shape": (40, 20, 1), "obstacle": {"center_x": 20, "center_y": 10, "radius": 0}},
        "radius must be positive",
        id="nonpositive_radius",
    ),
    pytest.param(
        {"grid_shape": (40, 20, 1), "obstacle": {"center_x": 20, "center_y": 5, "radius": 5}},
        "clearance from top/bottom",
        id="obstacle_touching_top_bottom_wall",
    ),
    pytest.param(
        {"grid_shape": (40, 20, 1), "obstacle": {"center_x": 2, "center_y": 10, "radius": 5}},
        "x-extent",
        id="obstacle_outside_x_extent",
    ),
    pytest.param(
        {"grid_shape": (40, 20, 4), "obstacle": {"center_x": 20, "center_y": 10, "radius": 5}},
        "2D",
        id="obstacle_3d_grid",
    ),
    pytest.param(
        {
            "grid_shape": (40, 20, 1),
            "boundary_conditions": {"left": "bounce-back"},
            "obstacle": {"center_x": 6, "center_y": 10, "radius": 5},
        },
        "left edge",
        id="obstacle_overlapping_nonperiodic_left_edge",
    ),
    pytest.param(
        {
            "grid_shape": (40, 20, 1),
            "boundary_conditions": {"right": "bounce-back"},
            "obstacle": {"center_x": 34, "center_y": 10, "radius": 5},
        },
        "right edge",
        id="obstacle_overlapping_nonperiodic_right_edge",
    ),
    pytest.param(
        {"grid_shape": (20, 20, 1), "boundary_conditions": {"front": "bounce-back"}},
        r"boundary_conditions\['front'\] has no face on the D2Q9 lattice",
        id="planar_lattice_rejects_a_wall_on_a_face_it_lacks",
    ),
    pytest.param(
        {"grid_shape": (20, 20, 1), "boundary_conditions": {"left": "bounceback"}},
        r"boundary_conditions\['left'\] must be one of",
        id="unknown_bc_name",
    ),
    pytest.param(
        {
            "grid_shape": (20, 20, 1),
            "boundary_conditions": {"left": "bounce-back", "left_velocity_inlet": {"u0": 0.02}},
        },
        r"boundary_conditions\['left_velocity_inlet'\] is not the parameter section",
        id="parameter_section_for_another_bc",
    ),
]


@pytest.mark.parametrize(("kwargs", "match"), _INVALID)
def test_invalid_config_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SimulationConfig(**kwargs)


# ---------------------------------------------------------------------------
# _validate_forces
# ---------------------------------------------------------------------------


class TestValidateEquilibrium:
    """The equilibrium is named in the config; its source term is the pair it registered."""

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            pytest.param({}, "standard_equilibrium", id="single_phase_default"),
            pytest.param(
                {
                    "multiphase": {
                        "eos": "double-well",
                        "kappa": 0.017,
                        "rho_l": 1.0,
                        "rho_v": 0.33,
                        "interface_width": 4,
                    }
                },
                "wb",
                id="multiphase_default",
            ),
            pytest.param({"equilibrium": "wb"}, "wb", id="named"),
        ],
    )
    def test_default_by_phase_or_named(self, kwargs: dict[str, Any], expected: str) -> None:
        assert SimulationConfig(grid_shape=(8, 8), **kwargs).equilibrium == expected

    def test_unregistered_equilibrium_raises(self) -> None:
        with pytest.raises(ValueError, match="equilibrium must be one of"):
            SimulationConfig(grid_shape=(8, 8), equilibrium="nonexistent")

    @pytest.mark.parametrize(("equilibrium", "source"), [("standard_equilibrium", "standard_source"), ("wb", "wb")])
    def test_each_equilibrium_declares_its_source_term(self, equilibrium: str, source: str) -> None:
        from src.config.equilibrium import source_term_for

        assert source_term_for(equilibrium) == source

    def test_an_equilibrium_without_a_source_term_pair_raises(self) -> None:
        from src.registry import register_operator
        from src.registry import unregister_operator

        register_operator("equilibrium", name="zz_unpaired")(lambda rho, u, lattice: rho)
        try:
            with pytest.raises(ValueError, match="registers no source_term pair"):
                SimulationConfig(grid_shape=(8, 8), equilibrium="zz_unpaired")
        finally:
            unregister_operator("equilibrium", "zz_unpaired")


class TestValidateForces:
    """Tests for _validate_forces: conflicting gravity force config."""

    def test_registered_defaults_are_filled_in(self):
        cfg = SimulationConfig(forces={"gravity_masked_force": {"force_g": 1e-6}}, grid_shape=(8, 8))
        assert cfg.forces["gravity_masked_force"] == {
            "force_g": 1e-6,
            "inclination_angle_deg": 0.0,
            "ramp_start_t": 0.0,
        }

    def test_configured_values_win_over_defaults(self):
        cfg = SimulationConfig(
            forces={"gravity_force": {"force_g": 1e-6, "inclination_angle_deg": 30.0}}, grid_shape=(8, 8)
        )
        assert cfg.forces["gravity_force"] == {"force_g": 1e-6, "inclination_angle_deg": 30.0}

    def test_electric_schema_follows_electric_params(self):
        cfg = SimulationConfig(
            forces={
                "electric_force": {
                    "permittivity_liquid": 80.0,
                    "permittivity_vapour": 1.0,
                    "conductivity_liquid": 0.01,
                    "conductivity_vapour": 0.001,
                }
            },
            grid_shape=(8, 8),
        )
        assert cfg.forces["electric_force"] is not None
        assert cfg.forces["electric_force"]["voltage_top"] == 0.0

    def test_active_forces_lists_only_configured_sections(self):
        cfg = SimulationConfig(forces={"gravity_force": {"force_g": 1e-6}}, grid_shape=(8, 8))
        assert list(cfg.forces) == ["gravity_force"]
        assert bool(cfg.forces)
        assert not SimulationConfig(grid_shape=(8, 8)).forces


# ---------------------------------------------------------------------------
# _validate_init
# ---------------------------------------------------------------------------


class TestValidateInit:
    """Tests for _validate_init: a registered init_type; init_from_file requires init_dir."""

    @pytest.mark.parametrize("init_type", sorted(get_operator_names("initialise") - {"init_from_file"}))
    def test_every_registered_init_type_is_accepted(self, init_type):
        assert SimulationConfig(grid_shape=(8, 8), init_type=init_type).init_type == init_type


# ---------------------------------------------------------------------------
# _validate_save_fields
# ---------------------------------------------------------------------------


class TestValidateSaveFields:
    """Tests for _validate_save_fields: invalid field names."""

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
        section = {key: value for key, value in _DW_BASE["multiphase"].items() if key != missing_field}
        params: dict[str, Any] = {**_DW_BASE, "multiphase": section}
        with pytest.raises(ValueError, match=f"'{missing_field}' is required"):
            SimulationConfig(**params)

    @pytest.mark.parametrize("missing_cs_param", ["a_eos", "b_eos", "r_eos", "t_eos"])
    def test_carnahan_starling_missing_param_raises(self, missing_cs_param: str):
        section = {key: value for key, value in _CS_BASE["multiphase"].items() if key != missing_cs_param}
        params: dict[str, Any] = {**_CS_BASE, "multiphase": section}
        with pytest.raises(ValueError, match=f"'{missing_cs_param}' is required"):
            SimulationConfig(**params)


class TestValidateObstacle:
    """Tests for _validate_obstacle: geometry sanity and BC-edge clearance."""

    def test_none_config_is_valid(self):
        cfg = SimulationConfig(grid_shape=(40, 20, 1), obstacle=None)
        assert cfg.obstacle is None

    def test_valid_obstacle_round_trips(self):
        cfg = SimulationConfig(
            grid_shape=(40, 20, 1),
            obstacle={"center_x": 20, "center_y": 10, "radius": 5},
        )
        assert cfg.obstacle == {"center_x": 20, "center_y": 10, "radius": 5, "shape": "circle"}

    def test_obstacle_far_from_nonperiodic_edges_is_valid(self):
        cfg = SimulationConfig(
            grid_shape=(40, 20, 1),
            boundary_conditions={"left": "bounce-back", "right": "bounce-back"},
            obstacle={"center_x": 20, "center_y": 10, "radius": 5},
        )
        assert cfg.obstacle is not None


class TestValidateBoundaryConditions:
    """boundary_conditions is completed and checked, so build_bc only looks up and binds."""

    def test_missing_edges_default_to_periodic(self):
        cfg = SimulationConfig(grid_shape=(20, 20, 1), boundary_conditions={"top": "bounce-back"})
        assert cfg.boundary_conditions is not None
        assert cfg.boundary_conditions["bottom"] == "periodic"

    def test_planar_lattice_has_no_front_or_back_edge(self):
        cfg = SimulationConfig(grid_shape=(20, 20, 1), boundary_conditions={"top": "bounce-back"})
        assert cfg.boundary_conditions is not None
        assert set(cfg.boundary_conditions) == {"top", "bottom", "left", "right"}
        assert set(SimulationConfig(grid_shape=(20, 20, 1)).boundary_conditions or {}) == {
            "top",
            "bottom",
            "left",
            "right",
        }

    def test_planar_lattice_drops_a_periodic_front_and_back(self):
        """A config.toml saved before the edges were lattice-aware still loads."""
        cfg = SimulationConfig(grid_shape=(20, 20, 1), boundary_conditions={"front": "periodic", "back": "periodic"})
        assert cfg.boundary_conditions is not None
        assert "front" not in cfg.boundary_conditions
        assert "back" not in cfg.boundary_conditions

    def test_three_dimensional_lattice_keeps_front_and_back(self):
        cfg = SimulationConfig(grid_shape=(8, 8, 8), lattice_type="D3Q19", boundary_conditions={"top": "bounce-back"})
        assert cfg.boundary_conditions is not None
        assert cfg.boundary_conditions["front"] == "periodic"
        assert cfg.boundary_conditions["back"] == "periodic"

    def test_boundary_edges_follow_application_order(self):
        cfg = SimulationConfig(grid_shape=(20, 20, 1), boundary_conditions={"top": "symmetry", "bottom": "bounce-back"})
        assert [(e.edge, e.name) for e in boundary_edges(cfg)] == [
            ("bottom", "bounce-back"),
            ("top", "symmetry"),
            ("left", "periodic"),
            ("right", "periodic"),
        ]

    def test_edge_without_parameter_section_runs_on_the_defaults(self):
        """The default is filled into the config, so the saved ``config.toml`` shows it."""
        cfg = SimulationConfig(
            grid_shape=(20, 20, 1), boundary_conditions={"left": "velocity-inlet", "right": "outlet"}
        )
        left = next(e for e in boundary_edges(cfg) if e.edge == "left")
        right = next(e for e in boundary_edges(cfg) if e.edge == "right")
        assert left.params == {"u0": 0.05}
        assert (cfg.boundary_conditions or {})["left_velocity_inlet"] == {"u0": 0.05}
        assert right.params == {}

    def test_parameter_section_is_bound_to_its_edge(self):
        cfg = SimulationConfig(
            grid_shape=(20, 20, 1),
            boundary_conditions={"left": "velocity-inlet", "right": "outlet", "left_velocity_inlet": {"u0": 0.02}},
        )
        left = next(e for e in boundary_edges(cfg) if e.edge == "left")
        assert left.params == {"u0": 0.02}


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
        return {**_DW_BASE, "wetting": {"phi_left": 1.0}, "boundary_conditions": bc}

    def test_top_wetting_wall_is_valid(self):
        cfg = SimulationConfig(**self._wetting_config(top="wetting"))
        assert cfg.boundary_conditions is not None
        assert cfg.boundary_conditions["top"] == "wetting"

    def test_two_wetting_walls_are_valid(self):
        cfg = SimulationConfig(**self._wetting_config(bottom="wetting", top="wetting"))
        assert cfg.boundary_conditions is not None
        assert cfg.boundary_conditions["bottom"] == "wetting"
        assert cfg.boundary_conditions["top"] == "wetting"

    def test_zero_wetting_walls_is_permissive(self):
        cfg = SimulationConfig(**self._wetting_config())
        assert cfg.wetting is not None


# ---------------------------------------------------------------------------
# Configuration-derived data (consumed by the operator factories)
# ---------------------------------------------------------------------------


class TestDerivedProperties:
    """Static per-run data the operator packages bind instead of deriving themselves."""

    def test_periodic_axes_need_both_edges_periodic(self):
        cfg = SimulationConfig(
            grid_shape=(20, 20, 1), boundary_conditions={"top": "bounce-back", "bottom": "bounce-back"}
        )
        assert periodic_axes(cfg) == (True, False, True)

    def test_pad_modes_follow_each_bc_registration(self):
        cfg = SimulationConfig(grid_shape=(20, 20, 1), boundary_conditions={"top": "symmetry", "bottom": "bounce-back"})
        assert pad_modes(cfg) == ("edge", "edge", "wrap", "wrap")

    def test_obstacle_mask_is_none_without_obstacle(self):
        assert obstacle_mask(SimulationConfig(grid_shape=(20, 20, 1))) is None

    def test_obstacle_mask_marks_the_solid_cells(self):
        cfg = SimulationConfig(
            grid_shape=(40, 20, 1),
            boundary_conditions={"left": "bounce-back", "right": "bounce-back"},
            obstacle={"center_x": 20, "center_y": 10, "radius": 5},
        )
        mask = obstacle_mask(cfg)
        assert mask is not None
        assert mask.shape == (40, 20, 1, 1, 1)
        assert bool(mask[20, 10, 0, 0, 0])
        assert not bool(mask[0, 0, 0, 0, 0])

    def test_wetting_defaults_are_none_without_wetting(self):
        assert wetting_scalars(SimulationConfig(grid_shape=(20, 20, 1))) is None

    def test_wetting_scalars_fill_neutral_values(self):
        params: dict[str, Any] = {
            **_DW_BASE,
            "wetting": {"phi_left": 1.2},
            "boundary_conditions": {"bottom": "wetting"},
        }
        cfg = SimulationConfig(**params)
        assert wetting_scalars(cfg) == {"phi_left": 1.2, "phi_right": 1.0, "d_rho_left": 0.0, "d_rho_right": 0.0}


# ---------------------------------------------------------------------------
# _validate_hysteresis and the hysteresis / chemical-step defaults
# ---------------------------------------------------------------------------


#: A droplet left of a mid-domain step: the start side and the dispersed phase both derive.
_DROPLET_LEFT_OF_STEP: dict[str, Any] = {"centres": [[0.25, 0.0]], "radii": [0.2], "dispersed": "liquid"}


def _hysteresis_config(
    hysteresis: dict[str, Any],
    *,
    chemical_step: dict[str, Any] | None = None,
    wall: str = "bottom",
    initialisation: dict[str, Any] | None = None,
    init_type: str = "multiphase_bubbles",
) -> SimulationConfig:
    base = dict(_DW_BASE)
    return SimulationConfig(
        **base,
        boundary_conditions={wall: "wetting"},
        hysteresis=hysteresis,
        chemical_step=chemical_step,
        init_type=init_type,
        init_dir="unused.npz" if init_type == "init_from_file" else None,
        initialisation=dict(_DROPLET_LEFT_OF_STEP) if initialisation is None else initialisation,
    )


class TestValidateHysteresis:
    """Unknown [hysteresis] keys raise; the optimiser settings are defaulted."""

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown \\[hysteresis\\] keys"):
            _hysteresis_config({"ca_advancing": 120.0, "ca_receding": 60.0, "ca_dead_zone": 0.5})

    @pytest.mark.parametrize(("key", "value"), [("learning_rate_above", 0.1), ("max_iterations_above", 7)])
    def test_optimiser_keys_are_accepted(self, key, value):
        cfg = _hysteresis_config({"ca_advancing": 120.0, "ca_receding": 60.0, key: value})
        assert cfg.hysteresis is not None
        assert cfg.hysteresis[key] == value

    def test_max_iterations_above_follows_max_iterations(self):
        cfg = _hysteresis_config({"ca_advancing": 120.0, "ca_receding": 60.0, "max_iterations": 12})
        assert cfg.hysteresis is not None
        assert cfg.hysteresis["max_iterations_above"] == 12

    def test_optimiser_settings_are_defaulted(self):
        cfg = _hysteresis_config({"ca_advancing": 120.0, "ca_receding": 60.0, "learning_rate": 0.05})
        assert cfg.hysteresis is not None
        assert cfg.hysteresis["learning_rate"] == 0.05
        assert cfg.hysteresis["max_iterations"] == 50
        assert cfg.hysteresis["loss_tol"] == 1e-4
        assert cfg.hysteresis["trial_steps"] == 2
        assert cfg.hysteresis["learning_rate_above"] == 0.05
        assert cfg.hysteresis["saturation_gap"] == 1.0
        assert cfg.hysteresis["latch_gap"] == 1.0
        assert cfg.hysteresis["stall_steps"] == 200

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("latch_gap", -0.5, "latch_gap must be a number of degrees >= 0"),
            ("latch_gap", "1", "latch_gap must be a number of degrees >= 0"),
            ("stall_steps", 0, "stall_steps must be an integer >= 1"),
            ("stall_steps", 2.5, "stall_steps must be an integer >= 1"),
        ],
    )
    def test_latch_settings_are_validated(self, key, value, message):
        section = {"ca_advancing": 120.0, "ca_receding": 60.0, key: value}

        with pytest.raises(ValueError, match=message):
            _hysteresis_config(section)

    def test_chemical_step_edge_width_defaults_to_one_cell(self):
        chemical_step = {
            "chemical_step_location": 0.5,
            "ca_advancing_pre_step": 120.0,
            "ca_receding_pre_step": 100.0,
            "ca_advancing_post_step": 70.0,
            "ca_receding_post_step": 50.0,
        }
        cfg = _hysteresis_config({}, chemical_step=chemical_step)
        assert cfg.chemical_step is not None
        assert cfg.chemical_step["edge_width"] == 1.0


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
        base = dict(_DW_BASE)
        return SimulationConfig(
            **base,
            boundary_conditions={"bottom": "wetting"},
            wetting={"phi_left": 1.0},
            hysteresis={"ca_advancing": 120.0, "ca_receding": 60.0} if hysteresis else None,
            init_type="init_from_file",
            init_dir=init_dir,
        )

    def test_hysteresis_restart_reads_the_snapshot(self, tmp_path):
        cfg = self._config(self._npz(tmp_path, **_SAVED_WETTING))
        assert restored_wetting(cfg) == _SAVED_WETTING

    def test_hysteresis_restart_also_reads_a_saved_latch(self, tmp_path):
        latch = {"latch_left": 1, "latch_right": 0, "stall_left": 3, "stall_right": 0}
        cfg = self._config(self._npz(tmp_path, **_SAVED_WETTING, **latch))
        assert restored_wetting(cfg) == _SAVED_WETTING | latch

    def test_snapshot_without_wetting_state_is_a_fresh_start(self, tmp_path):
        cfg = self._config(self._npz(tmp_path))
        assert restored_wetting(cfg) is None

    def test_fixed_wetting_run_keeps_its_configured_parameters(self, tmp_path):
        cfg = self._config(self._npz(tmp_path, **_SAVED_WETTING), hysteresis=False)
        assert restored_wetting(cfg) is None


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
        cfg = _hysteresis_config({}, chemical_step=dict(_STEP))
        wall = chemical_step_wall(cfg)
        assert wall is not None
        assert (wall.edge, wall.step_x) == ("bottom", 4.0)  # 0.5 * nx=8
        assert (wall.pre_phi_left, wall.pre_d_rho_left) == (1.0, 0.0)  # neutral [wetting] defaults
        assert (wall.post_phi, wall.post_d_rho) == (1.0 + 5 / 4, 0.0)  # W = 4: phi limit 1 + 5/W

    def test_a_less_wetting_post_surface_pushes_instead(self):
        step = dict(_STEP, ca_advancing_post_step=130.0, ca_receding_post_step=110.0)
        cfg = _hysteresis_config({}, chemical_step=step)
        wall = chemical_step_wall(cfg)
        assert wall is not None
        assert (wall.post_phi, wall.post_d_rho) == (1.0, 1.5 / 4)

    def test_no_step_no_wall(self):
        assert chemical_step_wall(_hysteresis_config({})) is None

    def test_the_step_defaults_to_the_wetting_wall(self):
        cfg = _hysteresis_config({}, chemical_step=dict(_STEP), wall="top")
        wall = chemical_step_wall(cfg)
        assert wall is not None
        assert wall.edge == "top"

    def test_a_step_off_the_wetting_wall_raises(self):
        step = dict(_STEP, chemical_step_edge="bottom")
        with pytest.raises(ValueError, match="chemical_step_edge must be the wetting wall 'top'"):
            _hysteresis_config({}, chemical_step=step, wall="top")

    def test_a_bubble_pulls_its_post_surface_with_d_rho(self):
        """The same dispersed-frame windows: post wets the vapour more, so the wall wets the liquid less."""
        bubble = {"centres": [[0.25, 1.0]], "radii": [0.2], "dispersed": "vapour"}
        cfg = _hysteresis_config({}, chemical_step=dict(_STEP), initialisation=bubble)
        wall = chemical_step_wall(cfg)
        assert wall is not None
        assert (wall.post_phi, wall.post_d_rho) == (1.0, 1.5 / 4)


class TestChemicalStepPreSide:
    """``chemical_step_pre_side``: the side of the step the inclusion starts on."""

    @pytest.mark.parametrize(("x", "side"), [(0.25, "low"), (0.75, "high")])
    def test_it_is_derived_from_the_centres(self, x, side):
        initialisation = {"centres": [[x, 0.0]], "radii": [0.2], "dispersed": "liquid"}
        cfg = _hysteresis_config({}, chemical_step=dict(_STEP), initialisation=initialisation)
        assert cfg.chemical_step is not None
        assert cfg.chemical_step["chemical_step_pre_side"] == side
        wall = chemical_step_wall(cfg)
        assert wall is not None
        assert wall.pre_side == side

    def test_a_configured_side_wins(self):
        cfg = _hysteresis_config({}, chemical_step=dict(_STEP, chemical_step_pre_side="high"))
        assert cfg.chemical_step is not None
        assert cfg.chemical_step["chemical_step_pre_side"] == "high"

    @pytest.mark.parametrize(
        "centres",
        [None, [[0.25, 0.0], [0.75, 0.0]], [[4 / 7, 0.0]]],  # none, both sides, on the step (cell 4 of nx = 8)
        ids=["no-centres", "both-sides", "on-the-step"],
    )
    def test_an_underived_side_is_required(self, centres):
        initialisation = {"dispersed": "liquid"} if centres is None else {"centres": centres, "dispersed": "liquid"}
        step = dict(_STEP)
        with pytest.raises(ValueError, match="chemical_step_pre_side"):
            _hysteresis_config({}, chemical_step=step, initialisation=initialisation, init_type="init_from_file")

    def test_an_unknown_side_raises(self):
        step = dict(_STEP, chemical_step_pre_side="left")
        with pytest.raises(ValueError, match="chemical_step_pre_side must be one of"):
            _hysteresis_config({}, chemical_step=step)

    def test_a_restart_without_a_dispersed_phase_raises(self):
        step = dict(_STEP, chemical_step_pre_side="low")
        with pytest.raises(ValueError, match="dispersed"):
            _hysteresis_config({}, chemical_step=step, initialisation={}, init_type="init_from_file")

    def test_a_bubble_initialiser_defaults_to_vapour(self):
        from src.config.initialisation import DispersedPhase
        from src.config.initialisation import dispersed_phase

        cfg = _hysteresis_config(
            {}, chemical_step=dict(_STEP), initialisation={"centres": [[0.25, 1.0]], "radii": [0.2]}
        )
        assert dispersed_phase(cfg) == DispersedPhase.VAPOUR

    @pytest.mark.parametrize(
        ("side", "expected"), [("low", [False, False, True, True]), ("high", [True, True, False, False])]
    )
    def test_on_post_is_the_destination_side(self, side, expected):
        import numpy as np

        cfg = _hysteresis_config({}, chemical_step=dict(_STEP, chemical_step_pre_side=side))
        wall = chemical_step_wall(cfg)
        assert wall is not None
        assert wall.on_post(np.array([2.0, 3.0, 4.0, 6.0])).tolist() == expected


class TestSectionSchemas:
    """Each section module's schema, as its table declares it."""

    def test_electric_schema_covers_exactly_the_params(self) -> None:
        from src.config.defaults import FORCES
        from src.config.forces import FORCE_SCHEMAS
        from src.operators.force._electric import ElectricParams

        schema = FORCE_SCHEMAS["electric_force"]

        assert {*schema.required, *FORCES["electric_force"]} == set(ElectricParams._fields)

    def test_every_force_with_a_schema_is_registered(self) -> None:
        import src.operators.force  # noqa: F401
        from src.config.forces import FORCE_SCHEMAS
        from src.registry import get_operator_names

        assert set(FORCE_SCHEMAS) == get_operator_names("force")

    def test_an_eos_parameter_of_another_eos_is_rejected(self) -> None:
        section = {**_DW_BASE["multiphase"], "a_eos": 1.0}
        params: dict[str, Any] = {**_DW_BASE, "multiphase": section}

        with pytest.raises(ValueError, match="unknown key"):
            SimulationConfig(**params)

    def test_an_unknown_wetting_key_is_rejected(self) -> None:
        params: dict[str, Any] = {**_DW_BASE, "wetting": {"phi_l": 1.2}}

        with pytest.raises(ValueError, match=r"Unknown \[wetting\] keys"):
            SimulationConfig(**params)

    def test_an_unknown_force_is_rejected(self) -> None:
        forces = {"magnetic_force": {"b": 1.0}}

        with pytest.raises(ValueError, match="Unknown force"):
            SimulationConfig(forces=forces)
