"""Unit tests for array expansion module."""

import pytest
from config.array_expansion import enumerate_configs
from config.array_expansion import expand_config


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
        with pytest.raises(ValueError, match=r"collision_scheme must be one of \['bgk', 'mrt'\], got 'missing_scheme'"):
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
        assert configs[0].grid_shape == (64, 64)
        assert configs[1].grid_shape == (128, 128)

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
