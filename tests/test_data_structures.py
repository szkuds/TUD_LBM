"""Phase 1 tests — JAX-compliant data structures.

Tests for:
    - ``setup.lattice.Lattice`` and ``build_lattice``
    - ``state.state.State`` and ``WettingState``
    - ``config.simulation_config.SimulationConfig``
    - ``setup.simulation_setup.SimulationSetup`` and ``build_setup``
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# =====================================================================
# Lattice
# =====================================================================


class TestBuildLattice:
    """``build_lattice`` produces a correct, JAX-friendly D2Q9 lattice."""

    def test_d2q9_dimensions(self):
        from setup.lattice import build_lattice

        lat = build_lattice("D2Q9")
        assert lat.d == 2
        assert lat.q == 9

    def test_d2q9_velocity_shape(self):
        from setup.lattice import build_lattice

        lat = build_lattice("D2Q9")
        assert lat.c.shape == (2, 9)

    def test_d2q9_weights_shape_and_sum(self):
        from setup.lattice import build_lattice

        lat = build_lattice("D2Q9")
        assert lat.w.shape == (9,)
        np.testing.assert_allclose(float(jnp.sum(lat.w)), 1.0)

    def test_d2q9_opposite_indices(self):
        from setup.lattice import build_lattice

        lat = build_lattice("D2Q9")
        # For every velocity i, opp[opp[i]] == i (involution)
        opp = np.array(lat.opp_indices)
        np.testing.assert_array_equal(opp[opp], np.arange(9))

    def test_d2q9_directional_indices_nonempty(self):
        from setup.lattice import build_lattice

        lat = build_lattice("D2Q9")
        assert len(lat.main_indices) > 0
        assert len(lat.right_indices) > 0
        assert len(lat.left_indices) > 0
        assert len(lat.top_indices) > 0
        assert len(lat.bottom_indices) > 0

    def test_d2q9_arrays_are_jax(self):
        from setup.lattice import build_lattice

        lat = build_lattice("D2Q9")
        for arr in (
            lat.c,
            lat.w,
            lat.opp_indices,
            lat.main_indices,
            lat.right_indices,
            lat.left_indices,
            lat.top_indices,
            lat.bottom_indices,
        ):
            assert isinstance(arr, jax.Array), f"{arr} is not a jax.Array"

    def test_case_insensitive(self):
        from setup.lattice import build_lattice

        lat = build_lattice("d2q9")
        assert lat.d == 2

    def test_unsupported_lattice_raises(self):
        from setup.lattice import build_lattice

        with pytest.raises(ValueError, match="Unsupported lattice type"):
            build_lattice("D1Q3")

    def test_lattice_is_pytree(self):
        """Lattice can be flattened and unflattened as a JAX pytree."""
        from setup.lattice import build_lattice

        lat = build_lattice("D2Q9")
        leaves, treedef = jax.tree_util.tree_flatten(lat)
        reconstructed = treedef.unflatten(leaves)
        assert reconstructed.d == lat.d
        assert reconstructed.q == lat.q
        np.testing.assert_array_equal(np.array(reconstructed.c), np.array(lat.c))


# =====================================================================
# State / WettingState
# =====================================================================


class TestState:
    """``State`` NamedTuple is a valid JAX pytree."""

    def _make_state(self):
        from state.state import State

        return State(
            f=jnp.zeros((8, 8, 9, 1)),
            rho=jnp.ones((8, 8, 1, 1)),
            u=jnp.zeros((8, 8, 1, 2)),
            t=jnp.array(0),
        )

    def test_construction(self):
        s = self._make_state()
        assert s.f.shape == (8, 8, 9, 1)
        assert s.rho.shape == (8, 8, 1, 1)
        assert s.u.shape == (8, 8, 1, 2)

    def test_optional_fields_default_none(self):
        s = self._make_state()
        assert s.force is None
        assert s.force_ext is None
        assert s.h is None
        assert s.wetting is None

    def test_pytree_flatten_unflatten(self):
        s = self._make_state()
        leaves, treedef = jax.tree_util.tree_flatten(s)
        s2 = treedef.unflatten(leaves)
        np.testing.assert_array_equal(np.array(s2.f), np.array(s.f))
        np.testing.assert_array_equal(np.array(s2.t), np.array(s.t))

    def test_replace(self):
        s = self._make_state()
        s2 = s._replace(t=jnp.array(42))
        assert int(s2.t) == 42
        assert int(s.t) == 0  # original unchanged


class TestWettingState:
    """``WettingState`` is also a valid pytree."""

    def _make_wetting_state(self):
        from state.state import WettingState

        return WettingState(
            d_rho_left=jnp.array(0.1),
            d_rho_right=jnp.array(0.1),
            phi_left=jnp.array(0.5),
            phi_right=jnp.array(0.5),
            ca_left=jnp.array(90.0),
            ca_right=jnp.array(90.0),
            cll_left=jnp.array(10.0),
            cll_right=jnp.array(10.0),
        )

    def test_construction(self):
        ws = self._make_wetting_state()
        assert float(ws.d_rho_left) == pytest.approx(0.1)

    def test_pytree_round_trip(self):
        ws = self._make_wetting_state()
        leaves, treedef = jax.tree_util.tree_flatten(ws)
        ws2 = treedef.unflatten(leaves)
        np.testing.assert_allclose(float(ws2.ca_left), float(ws.ca_left))

    def test_nested_in_state(self):
        """WettingState can be nested inside State as a pytree leaf."""
        from state.state import State

        ws = self._make_wetting_state()
        s = State(
            f=jnp.zeros((8, 8, 9, 1)),
            rho=jnp.ones((8, 8, 1, 1)),
            u=jnp.zeros((8, 8, 1, 2)),
            t=jnp.array(0),
            wetting=ws,
        )
        leaves, treedef = jax.tree_util.tree_flatten(s)
        s2 = treedef.unflatten(leaves)
        assert s2.wetting is not None
        np.testing.assert_allclose(float(s2.wetting.ca_left), float(ws.ca_left))


# =====================================================================
# SimulationConfig
# =====================================================================


class TestSimulationConfigDefaults:
    """Default construction produces a valid frozen config."""

    def test_default_construction(self):
        from config.simulation_config import SimulationConfig

        cfg = SimulationConfig(grid_shape=(64, 64))
        assert cfg.sim_type == "single_phase"
        assert cfg.lattice_type == "D2Q9"
        assert cfg.tau == 1.0
        assert cfg.nt == 1000
        assert cfg.collision_scheme == "bgk"

    def test_frozen(self):
        from config.simulation_config import SimulationConfig

        cfg = SimulationConfig(grid_shape=(8, 8))
        with pytest.raises(AttributeError):
            cfg.tau = 0.9  # type: ignore[misc]

    def test_default_bc_config_is_periodic(self):
        from config.simulation_config import SimulationConfig

        cfg = SimulationConfig(grid_shape=(8, 8))
        assert cfg.bc_config is not None
        for edge in ("top", "bottom", "left", "right"):
            assert cfg.bc_config[edge] == "periodic"

    def test_is_single_phase_property(self):
        from config.simulation_config import SimulationConfig

        cfg = SimulationConfig(grid_shape=(8, 8))
        assert cfg.is_single_phase is True
        assert cfg.is_multiphase is False


class TestSimulationConfigValidation:
    """Validation matches the legacy SimulationSetup behaviour."""

    def test_invalid_tau_raises(self):
        from config.simulation_config import SimulationConfig

        with pytest.raises(ValueError, match=r"tau must be > 0\.5"):
            SimulationConfig(grid_shape=(8, 8), tau=0.3)

    def test_invalid_nt_raises(self):
        from config.simulation_config import SimulationConfig

        with pytest.raises(ValueError, match="nt must be positive"):
            SimulationConfig(grid_shape=(8, 8), nt=0)

    def test_invalid_grid_shape_raises(self):
        from config.simulation_config import SimulationConfig

        with pytest.raises(ValueError, match="at least 2 dimensions"):
            SimulationConfig(grid_shape=(8,))

    def test_negative_grid_dimension_raises(self):
        from config.simulation_config import SimulationConfig

        with pytest.raises(ValueError, match="positive"):
            SimulationConfig(grid_shape=(8, -1))

    def test_invalid_lattice_type_raises(self):
        from config.simulation_config import SimulationConfig

        with pytest.raises(ValueError, match="lattice_type"):
            SimulationConfig(grid_shape=(8, 8), lattice_type="D1Q3")

    def test_invalid_collision_scheme_raises(self):
        from config.simulation_config import SimulationConfig

        with pytest.raises(ValueError, match="collision_scheme"):
            SimulationConfig(grid_shape=(8, 8), collision_scheme="invalid")

    def test_mrt_without_k_diag_raises(self):
        from config.simulation_config import SimulationConfig

        with pytest.raises(ValueError, match="k_diag"):
            SimulationConfig(grid_shape=(8, 8), collision_scheme="mrt")

    def test_invalid_save_interval_raises(self):
        from config.simulation_config import SimulationConfig

        with pytest.raises(ValueError, match="save_interval"):
            SimulationConfig(grid_shape=(8, 8), save_interval=-1)

    def test_negative_skip_interval_raises(self):
        from config.simulation_config import SimulationConfig

        with pytest.raises(ValueError, match="skip_interval"):
            SimulationConfig(grid_shape=(8, 8), skip_interval=-1)

    def test_init_from_file_without_dir_raises(self):
        from config.simulation_config import SimulationConfig

        with pytest.raises(ValueError, match="init_dir"):
            SimulationConfig(grid_shape=(8, 8), init_type="init_from_file")

    def test_invalid_save_fields_raises(self):
        from config.simulation_config import SimulationConfig

        with pytest.raises(ValueError, match="Invalid save_fields"):
            SimulationConfig(grid_shape=(8, 8), save_fields=["invalid_field"])


class TestSimulationConfigMultiphase:
    """Multiphase-specific validation mirrors legacy behaviour."""

    def test_multiphase_construction(self):
        from config.simulation_config import SimulationConfig

        cfg = SimulationConfig(
            sim_type="multiphase",
            grid_shape=(64, 64),
            tau=0.99,
            eos="double-well",
            kappa=0.017,
            rho_l=1.0,
            rho_v=0.33,
            interface_width=4,
        )
        assert cfg.is_multiphase is True

    def test_multiphase_missing_kappa_raises(self):
        from config.simulation_config import SimulationConfig

        with pytest.raises(ValueError, match="'kappa' is required"):
            SimulationConfig(
                sim_type="multiphase",
                grid_shape=(8, 8),
                eos="double-well",
                rho_l=1.0,
                rho_v=0.33,
                interface_width=4,
            )

    def test_multiphase_missing_eos_raises(self):
        from config.simulation_config import SimulationConfig

        with pytest.raises(ValueError, match="'eos' is required"):
            SimulationConfig(
                sim_type="multiphase",
                grid_shape=(8, 8),
                kappa=0.1,
                rho_l=1.0,
                rho_v=0.33,
                interface_width=4,
            )

    def test_multiphase_invalid_densities_raises(self):
        from config.simulation_config import SimulationConfig

        with pytest.raises(ValueError, match=r"rho_l.*must be greater than rho_v"):
            SimulationConfig(
                sim_type="multiphase",
                grid_shape=(8, 8),
                eos="double-well",
                kappa=0.1,
                rho_l=0.1,
                rho_v=1.0,
                interface_width=4,
            )


class TestSimulationConfigToDict:
    """``to_dict()`` serialisation."""

    def test_to_dict_contains_sim_type(self):
        from config.simulation_config import SimulationConfig

        cfg = SimulationConfig(grid_shape=(8, 8))
        d = cfg.to_dict()
        assert d["simulation_type"] == "single_phase"

    def test_to_dict_contains_core_fields(self):
        from config.simulation_config import SimulationConfig

        cfg = SimulationConfig(grid_shape=(16, 16), tau=0.7, nt=500)
        d = cfg.to_dict()
        assert d["grid_shape"] == (16, 16)
        assert d["tau"] == 0.7
        assert d["nt"] == 500

    def test_to_dict_merges_extra(self):
        from config.simulation_config import SimulationConfig

        cfg = SimulationConfig(grid_shape=(8, 8), extra={"custom_key": 42})
        d = cfg.to_dict()
        assert d["custom_key"] == 42
        assert "extra" not in d


# =====================================================================
# --- SimulationSetup (pytree) and build_setup ---
# =====================================================================


class TestBuildSetup:
    """``build_setup`` produces a correct, immutable SimulationSetup pytree."""

    def test_from_simulation_config_single_phase(self):
        from config.simulation_config import SimulationConfig
        from setup.simulation_setup import build_setup

        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8)
        setup = build_setup(cfg)

        assert setup.grid_shape == (8, 8)
        assert setup.tau == 0.8
        assert setup.lattice.d == 2
        assert setup.lattice.q == 9
        assert setup.multiphase_params is None

    def test_from_simulation_config_multiphase(self):
        from config.simulation_config import SimulationConfig
        from setup.simulation_setup import build_setup

        cfg = SimulationConfig(
            sim_type="multiphase",
            grid_shape=(64, 64),
            tau=0.99,
            eos="double-well",
            kappa=0.017,
            rho_l=1.0,
            rho_v=0.33,
            interface_width=4,
        )
        setup = build_setup(cfg)

        assert setup.multiphase_params is not None
        assert setup.multiphase_params.eos == "double-well"
        assert setup.multiphase_params.kappa == 0.017
        assert setup.multiphase_params.rho_l == 1.0
        assert setup.multiphase_params.rho_v == 0.33
        assert setup.multiphase_params.interface_width == 4

    def test_setup_is_pytree(self):
        """SimulationSetup round-trips through JAX pytree flatten/unflatten."""
        from config.simulation_config import SimulationConfig
        from setup.simulation_setup import build_setup

        cfg = SimulationConfig(grid_shape=(8, 8))
        setup = build_setup(cfg)

        leaves, treedef = jax.tree_util.tree_flatten(setup)
        setup2 = treedef.unflatten(leaves)

        assert setup2.grid_shape == setup.grid_shape
        assert setup2.tau == setup.tau
        np.testing.assert_array_equal(
            np.array(setup2.lattice.c),
            np.array(setup.lattice.c),
        )

    def test_save_fields_tuple(self):
        """save_fields is converted to a tuple (immutable)."""
        from config.simulation_config import SimulationConfig
        from setup.simulation_setup import build_setup

        cfg = SimulationConfig(grid_shape=(8, 8), save_fields=["f", "rho"])
        setup = build_setup(cfg)

        assert isinstance(setup.save_fields, tuple)
        assert setup.save_fields == ("f", "rho")

    def test_bc_config_preserved(self):
        from config.simulation_config import SimulationConfig
        from setup.simulation_setup import build_setup

        bc = {
            "top": "symmetry",
            "bottom": "bounce-back",
            "left": "periodic",
            "right": "periodic",
        }
        cfg = SimulationConfig(grid_shape=(8, 8), bc_config=bc)
        setup = build_setup(cfg)

        assert setup.bc_config["top"] == "symmetry"
        assert setup.bc_config["bottom"] == "bounce-back"

    def test_bc_masks_present(self):
        """build_setup produces BCMasks on the setup."""
        from config.simulation_config import SimulationConfig
        from setup.simulation_setup import build_setup

        cfg = SimulationConfig(grid_shape=(8, 8))
        setup = build_setup(cfg)

        assert setup.bc_masks is not None
        assert setup.bc_masks.top.shape == (8, 8, 1, 1)
        assert setup.bc_masks.bottom.shape == (8, 8, 1, 1)
        assert setup.bc_masks.left.shape == (8, 8, 1, 1)
        assert setup.bc_masks.right.shape == (8, 8, 1, 1)

    def test_bc_masks_correct_edges(self):
        """BCMasks mark the correct boundary rows/columns."""
        from config.simulation_config import SimulationConfig
        from setup.simulation_setup import build_setup

        cfg = SimulationConfig(grid_shape=(8, 8))
        setup = build_setup(cfg)

        # Top mask: y = ny-1 (index 7)
        assert bool(setup.bc_masks.top[0, 7, 0, 0]) is True
        assert bool(setup.bc_masks.top[0, 0, 0, 0]) is False
        # Bottom mask: y = 0
        assert bool(setup.bc_masks.bottom[0, 0, 0, 0]) is True
        assert bool(setup.bc_masks.bottom[0, 7, 0, 0]) is False
        # Left mask: x = 0
        assert bool(setup.bc_masks.left[0, 3, 0, 0]) is True
        assert bool(setup.bc_masks.left[7, 3, 0, 0]) is False
        # Right mask: x = nx-1 (index 7)
        assert bool(setup.bc_masks.right[7, 3, 0, 0]) is True
        assert bool(setup.bc_masks.right[0, 3, 0, 0]) is False

    def test_bc_masks_are_jax_arrays(self):
        from setup.simulation_setup import build_bc_masks

        masks = build_bc_masks((16, 16))
        for arr in (masks.top, masks.bottom, masks.left, masks.right):
            assert isinstance(arr, jax.Array)

    def test_bc_masks_pytree_round_trip(self):
        from setup.simulation_setup import build_bc_masks

        masks = build_bc_masks((8, 8))
        leaves, treedef = jax.tree_util.tree_flatten(masks)
        masks2 = treedef.unflatten(leaves)
        np.testing.assert_array_equal(np.array(masks2.top), np.array(masks.top))


# =====================================================================
# --- build_bc_masks (standalone) ---
# =====================================================================


class TestBuildBCMasks:
    """Standalone ``build_bc_masks`` factory."""

    def test_shapes(self):
        from setup.simulation_setup import build_bc_masks

        masks = build_bc_masks((16, 32))
        assert masks.top.shape == (16, 32, 1, 1)
        assert masks.bottom.shape == (16, 32, 1, 1)
        assert masks.left.shape == (16, 32, 1, 1)
        assert masks.right.shape == (16, 32, 1, 1)

    def test_mask_counts(self):
        """Each edge mask should have exactly one row/column of True."""
        from setup.simulation_setup import build_bc_masks

        masks = build_bc_masks((10, 20))
        # top: entire row y=19 → 10 True cells
        assert int(jnp.sum(masks.top)) == 10
        # bottom: entire row y=0 → 10 True cells
        assert int(jnp.sum(masks.bottom)) == 10
        # left: entire col x=0 → 20 True cells
        assert int(jnp.sum(masks.left)) == 20
        # right: entire col x=9 → 20 True cells
        assert int(jnp.sum(masks.right)) == 20


# =====================================================================
# --- build_multiphase_params (standalone) ---
# =====================================================================


class TestBuildMultiphaseParams:
    """Standalone ``build_multiphase_params`` factory."""

    def test_from_config(self):
        from config.simulation_config import SimulationConfig
        from setup.simulation_setup import build_multiphase_params

        cfg = SimulationConfig(
            sim_type="multiphase",
            grid_shape=(8, 8),
            tau=0.99,
            eos="double-well",
            kappa=0.017,
            rho_l=1.0,
            rho_v=0.33,
            interface_width=4,
        )
        mp = build_multiphase_params(cfg)
        assert mp.eos == "double-well"
        assert mp.kappa == 0.017
        assert mp.rho_l == 1.0
        assert mp.rho_v == 0.33
        assert mp.interface_width == 4
        assert mp.bubble is False

    def test_missing_field_raises(self):
        from dataclasses import dataclass
        from setup.simulation_setup import build_multiphase_params

        @dataclass
        class Incomplete:
            eos: str = "double-well"
            kappa: float = 0.1
            rho_l: float = 1.0
            rho_v: float = None
            interface_width: int = 4

        with pytest.raises(ValueError, match="'rho_v' is required"):
            build_multiphase_params(Incomplete())

    def test_multiphase_params_is_pytree(self):
        from setup.simulation_setup import MultiphaseParams

        mp = MultiphaseParams(
            eos="double-well",
            kappa=0.017,
            rho_l=1.0,
            rho_v=0.33,
            interface_width=4,
        )
        leaves, treedef = jax.tree_util.tree_flatten(mp)
        mp2 = treedef.unflatten(leaves)
        assert mp2.eos == mp.eos
        assert mp2.kappa == mp.kappa


# =====================================================================
# Lattice on 8×8 grid (integration)
# =====================================================================


class TestLatticeOn8x8Grid:
    """Lattice used in the context of an 8×8 grid."""

    def test_build_lattice_and_create_populations(self):
        """Build D2Q9 lattice and create zero-initialised populations on 8×8."""
        from setup.lattice import build_lattice

        lat = build_lattice("D2Q9")
        nx, ny = 8, 8
        f = jnp.zeros((nx, ny, lat.q, 1))
        assert f.shape == (8, 8, 9, 1)

    def test_lattice_weights_broadcast_over_grid(self):
        """Lattice weights can be broadcast across the grid."""
        from setup.lattice import build_lattice

        lat = build_lattice("D2Q9")
        nx, ny = 8, 8
        # Broadcast weights to (nx, ny, q, 1) for equilibrium computation
        w_grid = lat.w[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]
        assert w_grid.shape == (1, 1, 9, 1)
        w_full = jnp.broadcast_to(w_grid, (nx, ny, lat.q, 1))
        assert w_full.shape == (8, 8, 9, 1)
        np.testing.assert_allclose(float(jnp.sum(w_full[0, 0, :, 0])), 1.0)

    def test_streaming_roll_on_8x8(self):
        """Verify that jnp.roll with lattice velocities works on 8×8."""
        from setup.lattice import build_lattice

        lat = build_lattice("D2Q9")
        nx, ny = 8, 8
        f = jnp.zeros((nx, ny, lat.q, 1))
        # Put a pulse in direction 1 (cx=1, cy=0)
        f = f.at[3, 3, 1, 0].set(1.0)
        # Roll direction 1
        i = 1
        shift = (int(lat.c[0, i]), int(lat.c[1, i]))
        f_rolled = jnp.roll(f[:, :, i, :], shift=shift, axis=(0, 1))
        # Should have moved to (4, 3)
        assert float(f_rolled[4, 3, 0]) == 1.0
        assert float(f_rolled[3, 3, 0]) == 0.0
