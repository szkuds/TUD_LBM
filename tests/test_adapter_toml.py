"""Tests for the TOML configuration file adapter.

Tests validate that:
1. Simple single-phase TOML files are correctly parsed into SimulationSetup
2. Complex multiphase TOML files are correctly parsed with forces and BCs
3. Invalid configurations raise appropriate errors
4. The get_adapter factory dispatches correctly
"""

import os
import sys
import textwrap
import pytest

# Ensure src/ is on the path so imports work from the tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config.adapter_base import ConfigAdapter
from config.adapter_base import get_adapter
from config.adapter_toml import TomlAdapter
from config.simulation_config import SimulationConfig

# ── Fixtures ─────────────────────────────────────────────────────────

SIMPLE_TOML = textwrap.dedent("""\
    [simulation_type]
    simulation_name = "Test simple simulation_type"
    type = "single_phase"
    grid_shape = [100, 100]
    lattice_type = "D2Q9"
    tau = 0.6
    nt = 10000
    save_interval = 1000
    init_type = "standard"

    [output]
    results_dir = "~/TUD_LBM_data/results"
    plot_fields = ["density", "velocity"]
""")

MULTIPHASE_TOML = textwrap.dedent("""\
    [simulation_type]
    simulation_name = "Test complex simulation_type"
    type = "multiphase"
    grid_shape = [401, 101]
    lattice_type = "D2Q9"
    tau = 0.99
    nt = 20000
    save_interval = 2000
    init_type = "wetting"

    [multiphase]
    kappa = 0.017
    rho_l = 1.0
    rho_v = 0.33
    interface_width = 4
    eos = "double-well"

    [boundary_conditions]
    left = "periodic"
    right = "periodic"
    top = "symmetry"
    bottom = "wetting"

    [boundary_conditions.wetting_params]
    phi_left = 1.0
    phi_right = 1.0
    d_rho_left = 0.0
    d_rho_right = 0.0

    [boundary_conditions.hysteresis_params]
    ca_advancing = 90.0
    ca_receding = 80.0
    learning_rate = 0.05
    max_iterations = 10

    [output]
    results_dir = "~/TUD_LBM_data/results"
""")

MULTIPHASE_WITH_FORCE_TOML = textwrap.dedent("""\
    [simulation_type]
    type = "multiphase"
    grid_shape = [201, 101]
    tau = 0.99
    nt = 2000
    save_interval = 200
    init_type = "wetting"

    [multiphase]
    kappa = 0.017
    rho_l = 1.0
    rho_v = 0.33
    interface_width = 4
    eos = "double-well"

    [[force]]
    type = "gravity_multiphase"
    force_g = 2e-6
    inclination_angle_deg = 60
""")


@pytest.fixture
def simple_toml_file(tmp_path):
    """Write a simple TOML app_setup to a temp file and return its path."""
    p = tmp_path / "config_simple.toml"
    p.write_text(SIMPLE_TOML)
    return str(p)


@pytest.fixture
def multiphase_toml_file(tmp_path):
    """Write a multiphase TOML app_setup (no forces) to a temp file."""
    p = tmp_path / "config_multiphase.toml"
    p.write_text(MULTIPHASE_TOML)
    return str(p)


@pytest.fixture
def multiphase_force_toml_file(tmp_path):
    """Write a multiphase TOML app_setup with forces to a temp file."""
    p = tmp_path / "config_force.toml"
    p.write_text(MULTIPHASE_WITH_FORCE_TOML)
    return str(p)


# ── get_adapter tests ────────────────────────────────────────────────


class TestGetAdapter:
    """Tests for the get_adapter factory function."""

    def test_toml_extension_returns_toml_adapter(self):
        adapter = get_adapter("some/path/app_setup.toml")
        assert isinstance(adapter, TomlAdapter)

    def test_toml_extension_case_insensitive(self):
        adapter = get_adapter("app_setup.TOML")
        assert isinstance(adapter, TomlAdapter)

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported config file extension"):
            get_adapter("app_setup.yaml")

    def test_no_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported config file extension"):
            get_adapter("app_setup")


# ── TomlAdapter: simple single-phase ─────────────────────────────────


class TestTomlAdapterSimple:
    """Tests for loading simple single-phase configs."""

    def test_load_returns_simulation_bundle(self, simple_toml_file):
        adapter = TomlAdapter()
        bundle = adapter.load(simple_toml_file)
        assert isinstance(bundle, SimulationConfig)

    def test_load_is_single_phase(self, simple_toml_file):
        bundle = TomlAdapter().load(simple_toml_file)
        assert bundle.is_single_phase
        assert not bundle.is_multiphase

    def test_simulation_config_type(self, simple_toml_file):
        bundle = TomlAdapter().load(simple_toml_file)
        assert bundle.is_single_phase

    def test_grid_shape_is_tuple(self, simple_toml_file):
        bundle = TomlAdapter().load(simple_toml_file)
        assert bundle.grid_shape == (100, 100)
        assert isinstance(bundle.grid_shape, tuple)

    def test_physics_parameters(self, simple_toml_file):
        bundle = TomlAdapter().load(simple_toml_file)
        assert bundle.lattice_type == "D2Q9"
        assert bundle.tau == 0.6
        assert bundle.nt == 10000

    def test_runner_config(self, simple_toml_file):
        bundle = TomlAdapter().load(simple_toml_file)
        assert bundle.save_interval == 1000
        assert bundle.init_type == "standard"
        assert bundle.simulation_name == "Test simple simulation_type"

    def test_results_dir_expanded(self, simple_toml_file):
        bundle = TomlAdapter().load(simple_toml_file)
        assert "~" not in bundle.results_dir
        assert "TUD_LBM_data" in bundle.results_dir

    def test_plot_fields_loaded_from_output_table(self, simple_toml_file):
        bundle = TomlAdapter().load(simple_toml_file)
        assert bundle.plot_fields == ["density", "velocity"]

    def test_to_dict_includes_plot_fields(self, simple_toml_file):
        bundle = TomlAdapter().load(simple_toml_file)
        d = bundle.to_dict()
        assert d["plot_fields"] == ["density", "velocity"]

    def test_force_disabled_by_default(self, simple_toml_file):
        bundle = TomlAdapter().load(simple_toml_file)
        assert bundle.force_enabled is False

    def test_to_dict_roundtrip(self, simple_toml_file):
        bundle = TomlAdapter().load(simple_toml_file)
        d = bundle.to_dict()
        assert d["simulation_type"] == "single_phase"
        assert d["grid_shape"] == (100, 100)
        assert d["tau"] == 0.6
        assert d["save_interval"] == 1000


# ── TomlAdapter: multiphase (no forces) ──────────────────────────────


class TestTomlAdapterMultiphase:
    """Tests for loading multiphase configs without forces."""

    def test_is_multiphase(self, multiphase_toml_file):
        bundle = TomlAdapter().load(multiphase_toml_file)
        assert bundle.is_multiphase
        assert not bundle.is_single_phase

    def test_simulation_config_type(self, multiphase_toml_file):
        bundle = TomlAdapter().load(multiphase_toml_file)
        assert bundle.is_multiphase

    def test_multiphase_parameters(self, multiphase_toml_file):
        bundle = TomlAdapter().load(multiphase_toml_file)
        assert bundle.kappa == 0.017
        assert bundle.rho_l == 1.0
        assert bundle.rho_v == 0.33
        assert bundle.interface_width == 4
        assert bundle.eos == "double-well"

    def test_grid_shape(self, multiphase_toml_file):
        bundle = TomlAdapter().load(multiphase_toml_file)
        assert bundle.grid_shape == (401, 101)

    def test_runner_config(self, multiphase_toml_file):
        bundle = TomlAdapter().load(multiphase_toml_file)
        assert bundle.save_interval == 2000
        assert bundle.init_type == "wetting"
        assert bundle.simulation_name == "Test complex simulation_type"

    def test_boundary_conditions_parsed(self, multiphase_toml_file):
        bundle = TomlAdapter().load(multiphase_toml_file)
        bc = bundle.bc_config
        assert bc is not None
        assert bc["left"] == "periodic"
        assert bc["bottom"] == "wetting"
        assert bc["top"] == "symmetry"

    def test_wetting_params_nested(self, multiphase_toml_file):
        bundle = TomlAdapter().load(multiphase_toml_file)
        bc = bundle.bc_config
        assert "wetting_params" in bc
        wp = bc["wetting_params"]
        assert wp["phi_left"] == 1.0
        assert wp["d_rho_left"] == 0.0

    def test_hysteresis_params_nested(self, multiphase_toml_file):
        bundle = TomlAdapter().load(multiphase_toml_file)
        bc = bundle.bc_config
        assert "hysteresis_params" in bc
        hp = bc["hysteresis_params"]
        assert hp["ca_advancing"] == 90.0
        assert hp["ca_receding"] == 80.0


# ── TomlAdapter: multiphase with forces ──────────────────────────────


class TestTomlAdapterForces:
    """Tests for force config loading from TOML [[force]] tables.

    Since the migration to the functional architecture, the adapter
    no longer instantiates force objects.  Instead, ``[[force]]``
    tables are validated and stored as plain dicts in
    ``SimulationConfig.force_config``.  Actual JAX force objects are
    built later in ``build_setup()``.
    """

    def test_force_enabled_when_forces_present(self, multiphase_force_toml_file):
        bundle = TomlAdapter().load(multiphase_force_toml_file)
        assert bundle.force_enabled is True

    def test_force_config_is_list_of_dicts(self, multiphase_force_toml_file):
        bundle = TomlAdapter().load(multiphase_force_toml_file)
        assert isinstance(bundle.force_config, list)
        assert len(bundle.force_config) == 1
        assert isinstance(bundle.force_config[0], dict)

    def test_force_config_contains_correct_type(self, multiphase_force_toml_file):
        bundle = TomlAdapter().load(multiphase_force_toml_file)
        entry = bundle.force_config[0]
        assert entry["type"] == "gravity_multiphase"

    def test_force_config_contains_correct_params(self, multiphase_force_toml_file):
        bundle = TomlAdapter().load(multiphase_force_toml_file)
        entry = bundle.force_config[0]
        assert entry["force_g"] == 2e-6
        assert entry["inclination_angle_deg"] == 60


# ── Error handling ───────────────────────────────────────────────────


class TestTomlAdapterErrors:
    """Tests for error handling in the adapter."""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            TomlAdapter().load("/nonexistent/path/app_setup.toml")

    def test_missing_simulation_table(self, tmp_path):
        p = tmp_path / "empty.toml"
        p.write_text("[output]\nresults_dir = '/tmp'\n")
        with pytest.raises(
            ValueError,
            match="missing the required \\[simulation_type\\] table",
        ):
            TomlAdapter().load(str(p))

    def test_unknown_simulation_type(self, tmp_path):
        content = textwrap.dedent("""\
            [simulation_type]
            type = "unknown_type"
            grid_shape = [10, 10]
            tau = 0.6
        """)
        p = tmp_path / "bad_type.toml"
        p.write_text(content)
        with pytest.raises(ValueError, match="Unknown simulation type"):
            TomlAdapter().load(str(p))

    def test_unknown_force_type_raises_key_error(self, tmp_path):
        content = textwrap.dedent("""\
            [simulation_type]
            type = "multiphase"
            grid_shape = [10, 10]
            tau = 0.6

            [multiphase]
            kappa = 0.1
            rho_l = 1.0
            rho_v = 0.1
            interface_width = 4
            eos = "double-well"

            [[force]]
            type = "nonexistent_force"
        """)
        p = tmp_path / "bad_force.toml"
        p.write_text(content)
        with pytest.raises(KeyError, match="Unknown force type"):
            TomlAdapter().load(str(p))

    def test_invalid_tau_raises_validation_error(self, tmp_path):
        content = textwrap.dedent("""\
            [simulation_type]
            type = "single_phase"
            grid_shape = [10, 10]
            tau = 0.3
        """)
        p = tmp_path / "bad_tau.toml"
        p.write_text(content)
        with pytest.raises(ValueError, match="tau must be > 0.5"):
            TomlAdapter().load(str(p))


# ── ConfigAdapter ABC ────────────────────────────────────────────────


class TestConfigAdapterABC:
    """Tests for the abstract base class."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ConfigAdapter()

    def test_subclass_must_implement_load(self):
        class IncompleteAdapter(ConfigAdapter):
            pass

        with pytest.raises(TypeError):
            IncompleteAdapter()


# ── Integration: loading the actual example_for_test files ────────────────────


class TestExampleFiles:
    """Smoke tests against the actual example_for_test TOML files in the repo."""

    @pytest.fixture
    def example_dir(self):
        return os.path.join(os.path.dirname(__file__), "example_for_test")

    def test_config_simple_loads(self, example_dir):
        path = os.path.join(example_dir, "config_simple.toml")
        if not os.path.exists(path):
            pytest.skip("example_for_test/config_simple.toml not found")
        bundle = TomlAdapter().load(path)
        assert bundle.is_single_phase
        assert bundle.grid_shape == (100, 100)

    def test_config_complex_loads(self, example_dir):
        """Load the complex config — no mocking needed since forces are
        now stored as plain dicts in force_config, not instantiated.
        """
        path = os.path.join(example_dir, "config_complex.toml")
        if not os.path.exists(path):
            pytest.skip("example_for_test/config_complex.toml not found")

        bundle = TomlAdapter().load(path)

        assert bundle.is_multiphase
        assert bundle.grid_shape == (201, 201)
        assert bundle.kappa == 0.017
        assert bundle.save_interval == 400
        assert bundle.force_enabled is True
        assert isinstance(bundle.force_config, list)
        assert bundle.force_config[0]["type"] == "gravity_multiphase"
