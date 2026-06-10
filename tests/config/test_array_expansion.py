"""Unit tests for array expansion module."""

import pytest
from tud_lbm.config.array_expansion import enumerate_configs
from tud_lbm.config.array_expansion import expand_config


class TestExpandConfig:
    """Tests for expand_config() function."""

    def test_no_arrays_returns_single_config(self):
        """Config without arrays should return single config."""
        config_dict = {
            "sim_type": "single_phase",
            "grid_shape": (100, 100),
            "tau": 0.7,
            "nt": 1000,
            "collision_scheme": "bgk",
            "init_type": "standard",
        }
        configs, metadata = expand_config(config_dict)

        assert len(configs) == 1
        assert metadata is None
        assert configs[0].tau == 0.7

    def test_single_array_parameter(self):
        """Single array parameter should expand correctly."""
        config_dict = {
            "sim_type": "single_phase",
            "grid_shape": (100, 100),
            "tau": [0.6, 0.7, 0.8],
            "nt": 1000,
            "collision_scheme": "bgk",
            "init_type": "standard",
        }
        configs, metadata = expand_config(config_dict)

        assert len(configs) == 3
        assert metadata is not None
        assert metadata.field_names == frozenset({"tau"})
        assert metadata.total_combinations == 3
        assert [c.tau for c in configs] == [0.6, 0.7, 0.8]

    def test_multiple_array_parameters_cartesian_product(self):
        """Multiple arrays should create Cartesian product."""
        config_dict = {
            "sim_type": "single_phase",
            "grid_shape": (100, 100),
            "tau": [0.6, 0.8],
            "nt": [1000, 2000],
            "collision_scheme": "bgk",
            "init_type": "standard",
        }
        configs, metadata = expand_config(config_dict)

        assert len(configs) == 4  # 2 × 2
        assert metadata is not None
        assert metadata.total_combinations == 4
        assert metadata.field_names == frozenset({"tau", "nt"})

        # Check Cartesian product
        taus = [c.tau for c in configs]
        nts = [c.nt for c in configs]
        assert taus == [0.6, 0.6, 0.8, 0.8]
        assert nts == [1000, 2000, 1000, 2000]

    def test_arrays_not_allowed_raises_error(self):
        """allow_arrays=False should raise on array parameters."""
        config_dict = {
            "sim_type": "single_phase",
            "grid_shape": (100, 100),
            "tau": [0.6, 0.7],
            "nt": 1000,
            "collision_scheme": "bgk",
            "init_type": "standard",
        }
        with pytest.raises(ValueError, match="allow_arrays=False"):
            expand_config(config_dict, allow_arrays=False)

    def test_ineligible_field_array_raises_error(self):
        """Arrays in ineligible fields should raise error."""
        config_dict = {
            "sim_type": "single_phase",
            "grid_shape": (100, 100),
            "tau": 0.7,
            "nt": 1000,
            "collision_scheme": ["missing_scheme", "mrt", "bgk"],
            "init_type": "standard",
        }
        with pytest.raises(
            ValueError,
            match=r"collision_scheme must be one of \['bgk', 'mrt'\], got 'missing_scheme'",
        ):
            expand_config(config_dict)

    def test_grid_shape_array(self):
        """grid_shape in ARRAY_ELIGIBLE_FIELDS should work."""
        config_dict = {
            "sim_type": "single_phase",
            "grid_shape": [(64, 64), (128, 128)],
            "tau": 0.7,
            "nt": 1000,
            "collision_scheme": "bgk",
            "init_type": "standard",
        }
        configs, _ = expand_config(config_dict)

        assert len(configs) == 2
        assert configs[0].grid_shape == (64, 64, 1)
        assert configs[1].grid_shape == (128, 128, 1)

    def test_multiphase_arrays(self):
        """Multiphase parameters should support arrays."""
        config_dict = {
            "sim_type": "multiphase",
            "grid_shape": (100, 100),
            "tau": 0.7,
            "nt": 1000,
            "collision_scheme": "bgk",
            "init_type": "standard",
            "eos": "double-well",
            "rho_l": [1.0, 2.0],
            "rho_v": 0.1,
            "kappa": 0.001,
            "interface_width": 4,
            "g": 0.0,
        }
        configs, _ = expand_config(config_dict)

        assert len(configs) == 2
        assert [c.rho_l for c in configs] == [1.0, 2.0]

    def test_tuple_converted_to_list_for_arrays(self):
        """Tuples in ARRAY_ELIGIBLE_FIELDS should work."""
        config_dict = {
            "sim_type": "single_phase",
            "grid_shape": (100, 100),
            "tau": (0.6, 0.7, 0.8),  # Tuple instead of list
            "nt": 1000,
            "collision_scheme": "bgk",
            "init_type": "standard",
        }
        configs, _ = expand_config(config_dict)

        assert len(configs) == 3
        assert [c.tau for c in configs] == [0.6, 0.7, 0.8]


class TestEnumerateConfigs:
    """Tests for enumerate_configs() function."""

    def test_no_arrays_yields_single_config(self):
        """No arrays should yield single (0, {}, config)."""
        config_dict = {
            "sim_type": "single_phase",
            "grid_shape": (100, 100),
            "tau": 0.7,
            "nt": 1000,
            "collision_scheme": "bgk",
            "init_type": "standard",
        }
        results = list(enumerate_configs(config_dict))

        assert len(results) == 1
        idx, params, config = results[0]
        assert idx == 0
        assert params == {}
        assert config.tau == 0.7

    def test_single_array_yields_all_combinations(self):
        """Single array should yield all combinations with parameters."""
        config_dict = {
            "sim_type": "single_phase",
            "grid_shape": (100, 100),
            "tau": [0.6, 0.7, 0.8],
            "nt": 1000,
            "collision_scheme": "bgk",
            "init_type": "standard",
        }
        results = list(enumerate_configs(config_dict))

        assert len(results) == 3
        for idx, (i, params, config) in enumerate(results):
            assert i == idx
            assert "tau" in params
            assert params["tau"] in [0.6, 0.7, 0.8]
            assert config.tau == params["tau"]

    def test_multiple_arrays_yields_cartesian_product(self):
        """Multiple arrays should yield Cartesian product."""
        config_dict = {
            "sim_type": "single_phase",
            "grid_shape": (100, 100),
            "tau": [0.6, 0.8],
            "nt": [1000, 2000],
            "collision_scheme": "bgk",
            "init_type": "standard",
        }
        results = list(enumerate_configs(config_dict))

        assert len(results) == 4
        params_list = [params for _, params, _ in results]
        assert all(p.get("tau") is not None for p in params_list)
        assert all(p.get("nt") is not None for p in params_list)


# ── Base dict helpers ──────────────────────────────────────────────────────


def _base_single_phase(**overrides):
    return {
        "sim_type": "single_phase",
        "grid_shape": (64, 64),
        "tau": 0.8,
        "nt": 500,
        "init_type": "standard",
        **overrides,
    }


def _base_multiphase(**overrides):
    return {
        "sim_type": "multiphase",
        "grid_shape": (64, 64),
        "tau": 0.99,
        "nt": 500,
        "init_type": "standard",
        "eos": "double-well",
        "rho_l": 1.0,
        "rho_v": 0.33,
        "kappa": 0.017,
        "interface_width": 4,
        "g": 0.0,
        **overrides,
    }


# ── get_nested_sweepable_fields ────────────────────────────────────────────


class TestGetNestedSweepableFields:
    """Tests for the get_nested_sweepable_fields() introspection helper."""

    def test_returns_frozenset(self):
        from tud_lbm.config.simulation_config import get_nested_sweepable_fields

        result = get_nested_sweepable_fields()
        assert isinstance(result, frozenset)

    def test_contains_expected_fields(self):
        from tud_lbm.config.simulation_config import get_nested_sweepable_fields

        result = get_nested_sweepable_fields()
        assert "gravity_force" in result
        assert "electric_force" in result
        assert "wetting_config" in result
        assert "hysteresis_config" in result

    def test_does_not_contain_scalar_fields(self):
        from tud_lbm.config.simulation_config import get_nested_sweepable_fields

        result = get_nested_sweepable_fields()
        # Top-level scalar fields must not be included
        assert "tau" not in result
        assert "nt" not in result
        assert "grid_shape" not in result

    def test_nested_sweepable_is_subset_of_array_eligible(self):
        """Every nested-sweepable field must also be array-eligible."""
        from tud_lbm.config.simulation_config import get_array_eligible_fields
        from tud_lbm.config.simulation_config import get_nested_sweepable_fields

        assert get_nested_sweepable_fields().issubset(get_array_eligible_fields())


# ── Nested sweep: gravity_force ────────────────────────────────────────────


class TestNestedSweepGravityForce:
    """expand_config / enumerate_configs with gravity_force sub-key arrays."""

    def test_single_subkey_sweep(self):
        """Sweeping one sub-key of gravity_force expands correctly."""
        cfg = _base_single_phase(
            gravity_force={"force_g": 5e-7, "inclination_angle_deg": [30, 60, 90]},
        )
        configs, meta = expand_config(cfg)

        assert len(configs) == 3
        assert meta is not None
        assert meta.field_names == frozenset({"gravity_force.inclination_angle_deg"})
        angles = [c.gravity_force["inclination_angle_deg"] for c in configs]  # ty: ignore[not-subscriptable]
        assert angles == [30, 60, 90]

    def test_scalar_subkeys_preserved(self):
        """Non-array sub-keys must appear unchanged in every config."""
        cfg = _base_single_phase(
            gravity_force={"force_g": 5e-7, "inclination_angle_deg": [30, 60]},
        )
        configs, _ = expand_config(cfg)
        for c in configs:
            assert c.gravity_force is not None
            assert c.gravity_force["force_g"] == 5e-7

    def test_multiple_subkeys_cartesian_product(self):
        """Two array sub-keys inside gravity_force → Cartesian product."""
        cfg = _base_single_phase(
            gravity_force={"force_g": [1e-7, 5e-7], "inclination_angle_deg": [30, 60]},
        )
        configs, meta = expand_config(cfg)

        assert len(configs) == 4  # 2 × 2
        assert meta is not None
        assert meta.total_combinations == 4
        assert "gravity_force.force_g" in meta.field_names
        assert "gravity_force.inclination_angle_deg" in meta.field_names

    def test_cross_axis_with_top_level_tau(self):
        """gravity_force sub-key sweep × top-level tau → full Cartesian product."""
        cfg = _base_single_phase(
            tau=[0.6, 0.8],
            gravity_force={"force_g": 5e-7, "inclination_angle_deg": [30, 60]},
        )
        configs, meta = expand_config(cfg)

        assert len(configs) == 4  # 2 tau × 2 angles
        assert meta is not None
        assert "tau" in meta.field_names
        assert "gravity_force.inclination_angle_deg" in meta.field_names

    def test_enumerate_dotted_path_params(self):
        """enumerate_configs uses dotted-path keys in the parameters dict."""
        cfg = _base_single_phase(
            gravity_force={"force_g": 5e-7, "inclination_angle_deg": [45, 90]},
        )
        results = list(enumerate_configs(cfg))
        assert len(results) == 2
        for idx, (i, params, config) in enumerate(results):
            assert i == idx
            assert "gravity_force.inclination_angle_deg" in params
            assert config.gravity_force is not None
            assert config.gravity_force["inclination_angle_deg"] == params["gravity_force.inclination_angle_deg"]

    def test_no_array_subkeys_returns_single(self):
        """gravity_force with no list values returns a single config."""
        cfg = _base_single_phase(
            gravity_force={"force_g": 5e-7, "inclination_angle_deg": 45},
        )
        configs, meta = expand_config(cfg)
        assert len(configs) == 1
        assert meta is None

    def test_allow_arrays_false_raises(self):
        """allow_arrays=False should raise when a nested array is found."""
        cfg = _base_single_phase(
            gravity_force={"force_g": 5e-7, "inclination_angle_deg": [30, 60]},
        )
        with pytest.raises(ValueError):
            expand_config(cfg, allow_arrays=False)


# ── Nested sweep: electric_force ──────────────────────────────────────────


class TestNestedSweepElectricForce:
    """expand_config with electric_force sub-key arrays."""

    def test_single_subkey_sweep(self):
        cfg = _base_single_phase(
            electric_force={"charge_density": 1.0, "field_strength": [0.01, 0.05, 0.1]},
        )
        configs, meta = expand_config(cfg)

        assert len(configs) == 3
        assert meta is not None
        assert "electric_force.field_strength" in meta.field_names
        strengths = [c.electric_force["field_strength"] for c in configs]  # ty: ignore[not-subscriptable]
        assert strengths == [0.01, 0.05, 0.1]

    def test_scalar_subkeys_preserved(self):
        cfg = _base_single_phase(
            electric_force={"charge_density": 1.0, "field_strength": [0.01, 0.1]},
        )
        configs, _ = expand_config(cfg)
        for c in configs:
            assert c.electric_force is not None
            assert c.electric_force["charge_density"] == 1.0

    def test_cross_axis_with_gravity_force(self):
        """Sweep electric_force and gravity_force sub-keys simultaneously."""
        cfg = _base_single_phase(
            gravity_force={"force_g": 5e-7, "inclination_angle_deg": [30, 60]},
            electric_force={"charge_density": 1.0, "field_strength": [0.01, 0.1]},
        )
        configs, meta = expand_config(cfg)

        assert len(configs) == 4  # 2 × 2
        assert meta is not None
        assert "gravity_force.inclination_angle_deg" in meta.field_names
        assert "electric_force.field_strength" in meta.field_names


# ── Nested sweep: wetting_config ──────────────────────────────────────────


class TestNestedSweepWettingConfig:
    """expand_config with wetting_config sub-key arrays."""

    def test_single_subkey_sweep(self):
        cfg = _base_multiphase(
            wetting_config={
                "phi_left": [1.0, 1.1, 1.2],
                "phi_right": 1.0,
                "d_rho_left": 0.0,
                "d_rho_right": 0.0,
            },
        )
        configs, meta = expand_config(cfg)

        assert len(configs) == 3
        assert meta is not None
        assert "wetting_config.phi_left" in meta.field_names
        phi_vals = [c.wetting_config["phi_left"] for c in configs]  # ty: ignore[not-subscriptable]
        assert phi_vals == [1.0, 1.1, 1.2]

    def test_scalar_subkeys_preserved(self):
        cfg = _base_multiphase(
            wetting_config={
                "phi_left": [1.0, 1.2],
                "phi_right": 1.0,
                "d_rho_left": 0.0,
                "d_rho_right": 0.0,
            },
        )
        configs, _ = expand_config(cfg)
        for c in configs:
            assert c.wetting_config is not None
            assert c.wetting_config["phi_right"] == 1.0
            assert c.wetting_config["d_rho_left"] == 0.0

    def test_two_subkeys_cartesian(self):
        cfg = _base_multiphase(
            wetting_config={
                "phi_left": [1.0, 1.2],
                "phi_right": [1.0, 1.2],
                "d_rho_left": 0.0,
                "d_rho_right": 0.0,
            },
        )
        configs, meta = expand_config(cfg)
        assert len(configs) == 4
        assert meta is not None
        assert "wetting_config.phi_left" in meta.field_names
        assert "wetting_config.phi_right" in meta.field_names

    def test_cross_axis_with_tau(self):
        cfg = _base_multiphase(
            tau=[0.9, 0.99],
            wetting_config={
                "phi_left": [1.0, 1.1],
                "phi_right": 1.0,
                "d_rho_left": 0.0,
                "d_rho_right": 0.0,
            },
        )
        configs, meta = expand_config(cfg)
        assert len(configs) == 4
        assert meta is not None
        assert "tau" in meta.field_names
        assert "wetting_config.phi_left" in meta.field_names

    def test_enumerate_params_reflect_wetting(self):
        cfg = _base_multiphase(
            wetting_config={
                "phi_left": [1.0, 1.2],
                "phi_right": 1.0,
                "d_rho_left": 0.0,
                "d_rho_right": 0.0,
            },
        )
        for _, params, config in enumerate_configs(cfg):
            assert "wetting_config.phi_left" in params
            assert config.wetting_config is not None
            assert config.wetting_config["phi_left"] == params["wetting_config.phi_left"]


# ── Nested sweep: hysteresis_config ───────────────────────────────────────


class TestNestedSweepHysteresisConfig:
    """expand_config with hysteresis_config sub-key arrays."""

    def test_single_subkey_sweep(self):
        cfg = _base_multiphase(
            wetting_config={
                "phi_left": 1.0,
                "phi_right": 1.0,
                "d_rho_left": 0.0,
                "d_rho_right": 0.0,
            },
            hysteresis_config={
                "ca_advancing": [100.0, 120.0],
                "ca_receding": 60.0,
                "learning_rate": 0.01,
                "max_iterations": 20,
            },
        )
        configs, meta = expand_config(cfg)

        assert len(configs) == 2
        assert meta is not None
        assert "hysteresis_config.ca_advancing" in meta.field_names
        angles = [c.hysteresis_config["ca_advancing"] for c in configs]  # ty: ignore[not-subscriptable]
        assert angles == [100.0, 120.0]

    def test_scalar_subkeys_preserved(self):
        cfg = _base_multiphase(
            hysteresis_config={
                "ca_advancing": [100.0, 120.0],
                "ca_receding": 60.0,
                "learning_rate": 0.01,
                "max_iterations": 20,
            },
        )
        configs, _ = expand_config(cfg)
        for c in configs:
            assert c.hysteresis_config is not None
            assert c.hysteresis_config["ca_receding"] == 60.0
            assert c.hysteresis_config["learning_rate"] == 0.01
            assert c.hysteresis_config["max_iterations"] == 20

    def test_both_ca_angles_swept(self):
        cfg = _base_multiphase(
            hysteresis_config={
                "ca_advancing": [100.0, 120.0],
                "ca_receding": [50.0, 60.0],
                "learning_rate": 0.01,
                "max_iterations": 20,
            },
        )
        configs, meta = expand_config(cfg)
        assert len(configs) == 4
        assert meta is not None
        assert "hysteresis_config.ca_advancing" in meta.field_names
        assert "hysteresis_config.ca_receding" in meta.field_names

    def test_cross_axis_with_wetting_and_tau(self):
        """Three-way sweep: tau × wetting phi_left × hysteresis ca_advancing."""
        cfg = _base_multiphase(
            tau=[0.9, 0.99],
            wetting_config={
                "phi_left": [1.0, 1.1],
                "phi_right": 1.0,
                "d_rho_left": 0.0,
                "d_rho_right": 0.0,
            },
            hysteresis_config={
                "ca_advancing": [100.0, 120.0],
                "ca_receding": 60.0,
                "learning_rate": 0.01,
                "max_iterations": 20,
            },
        )
        configs, meta = expand_config(cfg)
        assert len(configs) == 8  # 2 × 2 × 2
        assert meta is not None
        assert "tau" in meta.field_names
        assert "wetting_config.phi_left" in meta.field_names
        assert "hysteresis_config.ca_advancing" in meta.field_names

    def test_enumerate_params_reflect_hysteresis(self):
        cfg = _base_multiphase(
            hysteresis_config={
                "ca_advancing": [100.0, 120.0],
                "ca_receding": 60.0,
                "learning_rate": 0.01,
                "max_iterations": 20,
            },
        )
        for _, params, config in enumerate_configs(cfg):
            assert "hysteresis_config.ca_advancing" in params
            assert config.hysteresis_config is not None
            assert config.hysteresis_config["ca_advancing"] == params["hysteresis_config.ca_advancing"]

    def test_allow_arrays_false_raises_on_hysteresis(self):
        cfg = _base_multiphase(
            hysteresis_config={
                "ca_advancing": [100.0, 120.0],
                "ca_receding": 60.0,
                "learning_rate": 0.01,
                "max_iterations": 20,
            },
        )
        with pytest.raises(ValueError):
            expand_config(cfg, allow_arrays=False)


# ── Enumerate: index continuity across nested sweeps ──────────────────────


class TestEnumerateConfigsNestedIndex:
    """Verify index, params and config stay consistent for nested sweeps."""

    def test_indices_are_sequential(self):
        cfg = _base_single_phase(
            gravity_force={"force_g": 5e-7, "inclination_angle_deg": [10, 20, 30, 40]},
        )
        indices = [i for i, _, _ in enumerate_configs(cfg)]
        assert indices == list(range(4))

    def test_configs_match_params(self):
        """Config field values must exactly match the parameters dict."""
        cfg = _base_single_phase(
            tau=[0.6, 0.7, 0.8],
            gravity_force={"force_g": [1e-7, 5e-7], "inclination_angle_deg": 45},
        )
        for _, params, config in enumerate_configs(cfg):
            if "tau" in params:
                assert config.tau == params["tau"]
            if "gravity_force.force_g" in params:
                assert config.gravity_force is not None
                assert config.gravity_force["force_g"] == params["gravity_force.force_g"]

    def test_total_count_matches_metadata(self):
        cfg = _base_single_phase(
            tau=[0.6, 0.8],
            gravity_force={"force_g": 5e-7, "inclination_angle_deg": [30, 60, 90]},
        )
        configs, meta = expand_config(cfg)
        results = list(enumerate_configs(cfg))
        assert meta is not None
        assert len(results) == meta.total_combinations == len(configs) == 6
