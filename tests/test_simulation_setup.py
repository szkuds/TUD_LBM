"""Tests for the SimulationSetup dataclass."""

import pytest

# Trigger operator registration
import simulation_operators  # noqa: F401
import simulation_type  # noqa: F401
import update_timestep  # noqa: F401

from app_setup.simulation_setup import SimulationSetup


class TestSimulationSetupDefaults:
    """Default construction produces a valid D2Q9 setup."""

    def test_default_construction(self):
        setup = SimulationSetup(grid_shape=(64, 64))
        assert setup.sim_type == "single_phase"
        assert setup.lattice_type == "D2Q9"
        assert setup.tau == 1.0
        assert setup.nt == 1000
        assert setup.collision_scheme == "bgk"

    def test_default_bc_config_is_periodic(self):
        setup = SimulationSetup(grid_shape=(8, 8))
        assert setup.bc_config is not None
        for edge in ("top", "bottom", "left", "right"):
            assert setup.bc_config[edge] == "periodic"

    def test_is_single_phase_property(self):
        setup = SimulationSetup(grid_shape=(8, 8))
        assert setup.is_single_phase is True
        assert setup.is_multiphase is False

    def test_default_init_type_is_standard(self):
        setup = SimulationSetup(grid_shape=(8, 8))
        assert setup.init_type == "standard"

    def test_default_save_interval(self):
        setup = SimulationSetup(grid_shape=(8, 8))
        assert setup.save_interval == 100


class TestSimulationSetupValidation:
    """Validation catches bad inputs early."""

    def test_invalid_tau_raises(self):
        with pytest.raises(ValueError, match="tau must be > 0.5"):
            SimulationSetup(grid_shape=(8, 8), tau=0.3)

    def test_invalid_nt_raises(self):
        with pytest.raises(ValueError, match="nt must be positive"):
            SimulationSetup(grid_shape=(8, 8), nt=0)

    def test_invalid_grid_shape_raises(self):
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            SimulationSetup(grid_shape=(8,))

    def test_negative_grid_dimension_raises(self):
        with pytest.raises(ValueError, match="positive"):
            SimulationSetup(grid_shape=(8, -1))

    def test_invalid_lattice_type_raises(self):
        with pytest.raises(ValueError, match="lattice_type"):
            SimulationSetup(grid_shape=(8, 8), lattice_type="D1Q3")

    def test_invalid_collision_scheme_raises(self):
        with pytest.raises(ValueError, match="collision_scheme"):
            SimulationSetup(grid_shape=(8, 8), collision_scheme="invalid")

    def test_mrt_without_k_diag_raises(self):
        with pytest.raises(ValueError, match="k_diag"):
            SimulationSetup(grid_shape=(8, 8), collision_scheme="mrt")

    def test_invalid_save_interval_raises(self):
        with pytest.raises(ValueError, match="save_interval"):
            SimulationSetup(grid_shape=(8, 8), save_interval=0)

    def test_negative_skip_interval_raises(self):
        with pytest.raises(ValueError, match="skip_interval"):
            SimulationSetup(grid_shape=(8, 8), skip_interval=-1)

    def test_init_from_file_without_dir_raises(self):
        with pytest.raises(ValueError, match="init_dir"):
            SimulationSetup(grid_shape=(8, 8), init_type="init_from_file")

    def test_invalid_save_fields_raises(self):
        with pytest.raises(ValueError, match="Invalid save_fields"):
            SimulationSetup(grid_shape=(8, 8), save_fields=["invalid_field"])

    def test_explicit_bc_config_preserved(self):
        bc = {"top": "symmetry", "bottom": "bounce-back", "left": "periodic", "right": "periodic"}
        setup = SimulationSetup(grid_shape=(8, 8), bc_config=bc)
        assert setup.bc_config["top"] == "symmetry"
        assert setup.bc_config["bottom"] == "bounce-back"


class TestSimulationSetupD3Q19:
    """D3Q19 with a 3D grid_shape passes validation."""

    def test_d3q19_construction(self):
        setup = SimulationSetup(
            grid_shape=(8, 8, 8),
            lattice_type="D3Q19",
            tau=0.8,
        )
        assert setup.lattice_type == "D3Q19"
        assert setup.grid_shape == (8, 8, 8)


class TestSimulationSetupMultiphase:
    """Multiphase-specific validation."""

    def test_multiphase_construction(self):
        setup = SimulationSetup(
            sim_type="multiphase",
            grid_shape=(64, 64),
            tau=0.99,
            eos="double-well",
            kappa=0.017,
            rho_l=1.0,
            rho_v=0.33,
            interface_width=4,
        )
        assert setup.is_multiphase is True
        assert setup.is_single_phase is False

    def test_multiphase_missing_kappa_raises(self):
        with pytest.raises(ValueError, match="'kappa' is required"):
            SimulationSetup(
                sim_type="multiphase",
                grid_shape=(8, 8),
                eos="double-well",
                rho_l=1.0,
                rho_v=0.33,
                interface_width=4,
            )

    def test_multiphase_missing_eos_raises(self):
        with pytest.raises(ValueError, match="'eos' is required"):
            SimulationSetup(
                sim_type="multiphase",
                grid_shape=(8, 8),
                kappa=0.1,
                rho_l=1.0,
                rho_v=0.33,
                interface_width=4,
            )

    def test_multiphase_invalid_densities_raises(self):
        with pytest.raises(ValueError, match="rho_l.*must be greater than rho_v"):
            SimulationSetup(
                sim_type="multiphase",
                grid_shape=(8, 8),
                eos="double-well",
                kappa=0.1,
                rho_l=0.1,
                rho_v=1.0,
                interface_width=4,
            )

    def test_multiphase_fields_ignored_for_single_phase(self):
        """Single-phase doesn't validate multiphase fields even when supplied."""
        setup = SimulationSetup(
            sim_type="single_phase",
            grid_shape=(8, 8),
            kappa=0.1,  # ignored — not validated for single_phase
        )
        assert setup.kappa == 0.1


class TestSimulationSetupToDict:
    """to_dict() round-trips correctly."""

    def test_to_dict_contains_sim_type(self):
        setup = SimulationSetup(grid_shape=(8, 8))
        d = setup.to_dict()
        assert d["simulation_type"] == "single_phase"

    def test_to_dict_contains_core_fields(self):
        setup = SimulationSetup(grid_shape=(16, 16), tau=0.7, nt=500)
        d = setup.to_dict()
        assert d["grid_shape"] == (16, 16)
        assert d["tau"] == 0.7
        assert d["nt"] == 500

    def test_to_dict_merges_extra(self):
        setup = SimulationSetup(grid_shape=(8, 8), extra={"custom_key": 42})
        d = setup.to_dict()
        assert d["custom_key"] == 42
        assert "extra" not in d

    def test_to_dict_multiphase(self):
        setup = SimulationSetup(
            sim_type="multiphase",
            grid_shape=(64, 64),
            tau=0.99,
            eos="double-well",
            kappa=0.017,
            rho_l=1.0,
            rho_v=0.33,
            interface_width=4,
        )
        d = setup.to_dict()
        assert d["simulation_type"] == "multiphase"
        assert d["kappa"] == 0.017



class TestSimulationSetupRepr:
    """__repr__ is informative."""

    def test_repr_contains_sim_type(self):
        setup = SimulationSetup(grid_shape=(8, 8))
        r = repr(setup)
        assert "single_phase" in r

    def test_repr_contains_grid_shape(self):
        setup = SimulationSetup(grid_shape=(32, 32))
        r = repr(setup)
        assert "(32, 32)" in r

