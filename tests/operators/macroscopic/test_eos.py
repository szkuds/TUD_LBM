"""Tests for EOS operators — carnahan-starling and double-well.

Covers:
    1. Isolated unit tests for ``_eos_carnahan_starling`` pure function
       - Known-value cross-check (numpy vs JAX)
       - Shape / dtype preservation
       - JIT-compatibility
       - Singularity avoidance (never evaluate at bρ = 4)
    2. Registry / factory tests for ``build_eos_fn``
       - Both EOS names resolve
       - Missing required params raise ValueError
       - Unknown name raises ValueError
    3. Pipeline integration
       - ``compute_macroscopic_multiphase`` with CS EOS: finite outputs
       - Uniform density field → zero interaction force
       - ``build_setup`` with full CS config succeeds
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.lattice.lattice import build_lattice

# ---------------------------------------------------------------------------
# Shared CS EOS parameters (from examples/config_cs_simple.toml)
# Subcritical temperature; coexistence at rho_v≈0.015, rho_l≈12.18
# Singularity at bρ = 4  →  ρ_sing ≈ 26.8 — well above rho_l
# ---------------------------------------------------------------------------

_A = 0.00031459670905604266
_B = 0.1490857142857143
_R = 1.0
_T = 0.00039808421247983624
_KAPPA = 0.01
_RHO_L = 12.18
_RHO_V = 0.015
_INTERFACE_WIDTH = 5

NX, NY, NZ = 16, 16, 1


@pytest.fixture(scope="module")
def lattice():
    return build_lattice("D2Q9")


@pytest.fixture(scope="module")
def cs_mp():
    from src.operators.macroscopic import MultiphaseParams

    return MultiphaseParams(
        eos="carnahan-starling",
        kappa=_KAPPA,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=_INTERFACE_WIDTH,
        a_eos=_A,
        b_eos=_B,
        r_eos=_R,
        t_eos=_T,
    )


@pytest.fixture(scope="module")
def dw_mp():
    from src.operators.macroscopic import MultiphaseParams

    return MultiphaseParams(
        eos="double-well",
        kappa=_KAPPA,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=_INTERFACE_WIDTH,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _numpy_cs_mu0(rho: np.ndarray, a: float, b: float, r: float, t: float) -> np.ndarray:
    """Reference CS μ₀ computed in NumPy — used for cross-validation only."""
    return -2.0 * a * rho + r * t * (1.0 + np.log(rho)) + (16.0 * r * t * (b * rho - 12.0)) / (b * rho - 4.0) ** 3


def _build_gradient_and_laplacian(lattice):
    """Return (gradient_standard, laplacian_density) closures for multiphase pipeline."""
    from src.operators.differential import build_differential_fn

    pad_modes = ("wrap", "wrap", "wrap", "wrap")
    _gradient = build_differential_fn("gradient")
    _laplacian = build_differential_fn("laplacian")

    @jax.jit
    def gradient_standard(grid):
        return _gradient(grid, lattice.w, lattice.c, pad_modes)

    @jax.jit
    def laplacian_density(grid):
        return _laplacian(grid, lattice.w, pad_modes)

    return gradient_standard, laplacian_density


# =====================================================================
# 1. Isolated unit tests — pure function
# =====================================================================


class TestCsEosPureFunction:
    """``_eos_carnahan_starling`` matches the reference formula exactly."""

    def test_known_value_numpy_cross_check(self):
        """JAX output matches NumPy reference for a scalar and an array."""
        from src.operators.macroscopic.eos._carnahan_starling import _eos_carnahan_starling

        rho_vals = np.array([_RHO_V, 1.0, 5.0, _RHO_L])
        expected = _numpy_cs_mu0(rho_vals, _A, _B, _R, _T)
        rho_jax = jnp.array(rho_vals)

        result = _eos_carnahan_starling(rho_jax, _A, _B, _R, _T)

        # float32 precision: JAX defaults to float32 without ENABLE_X64
        np.testing.assert_allclose(np.array(result), expected, rtol=1e-5)

    def test_shape_preserved_scalar(self):
        from src.operators.macroscopic.eos._carnahan_starling import _eos_carnahan_starling

        rho = jnp.array([_RHO_L])
        result = _eos_carnahan_starling(rho, _A, _B, _R, _T)
        assert result.shape == rho.shape

    def test_shape_preserved_5d(self):
        """5D array shape (nx, ny, nz, 1, 1) passes through unchanged."""
        from src.operators.macroscopic.eos._carnahan_starling import _eos_carnahan_starling

        rho = jnp.ones((NX, NY, NZ, 1, 1)) * _RHO_L
        result = _eos_carnahan_starling(rho, _A, _B, _R, _T)
        assert result.shape == rho.shape

    def test_output_finite_across_physical_range(self):
        """μ₀ is finite for all densities in [rho_v, rho_l] — no singularity there."""
        from src.operators.macroscopic.eos._carnahan_starling import _eos_carnahan_starling

        rho_range = jnp.linspace(_RHO_V, _RHO_L, 100)
        result = _eos_carnahan_starling(rho_range, _A, _B, _R, _T)

        assert bool(jnp.all(jnp.isfinite(result))), "mu_0 has non-finite values inside [rho_v, rho_l]"

    def test_jittable(self):
        from src.operators.macroscopic.eos._carnahan_starling import _eos_carnahan_starling

        rho = jnp.ones((NX, NY, NZ, 1, 1)) * _RHO_L
        jitted = jax.jit(_eos_carnahan_starling, static_argnums=(1, 2, 3, 4))
        result = jitted(rho, _A, _B, _R, _T)
        assert result.shape == rho.shape

    def test_mu0_negative_at_both_coexistence_phases(self):
        """At rho_l and rho_v, μ₀ is negative for these CS parameters."""
        from src.operators.macroscopic.eos._carnahan_starling import _eos_carnahan_starling

        rho = jnp.array([_RHO_V, _RHO_L])
        result = _eos_carnahan_starling(rho, _A, _B, _R, _T)
        assert float(result[0]) < 0.0, f"Expected mu_0(rho_v) < 0, got {float(result[0])}"
        assert float(result[1]) < 0.0, f"Expected mu_0(rho_l) < 0, got {float(result[1])}"

    def test_mu0_non_monotonic_spinodal_region(self):
        """CS μ₀ has a double-well character: spinodal region is above both phase values.

        The chemical potential is not monotonic between rho_v and rho_l — it rises
        through the unstable spinodal region and then falls back. This non-monotonicity
        is the driving force for phase separation.
        """
        from src.operators.macroscopic.eos._carnahan_starling import _eos_carnahan_starling

        rho_range = jnp.linspace(_RHO_V, _RHO_L, 200)
        mu_range = _eos_carnahan_starling(rho_range, _A, _B, _R, _T)

        mu_at_rho_v = float(_eos_carnahan_starling(jnp.array([_RHO_V]), _A, _B, _R, _T)[0])
        mu_at_rho_l = float(_eos_carnahan_starling(jnp.array([_RHO_L]), _A, _B, _R, _T)[0])
        mu_max_spinodal = float(jnp.max(mu_range))

        assert mu_max_spinodal > mu_at_rho_v, "CS mu_0 has no spinodal rise above rho_v phase value"
        assert mu_max_spinodal > mu_at_rho_l, "CS mu_0 has no spinodal rise above rho_l phase value"


# =====================================================================
# 2. Registry / factory
# =====================================================================


class TestEosFactory:
    """``build_eos_fn`` dispatches to the correct EOS and validates params."""

    def test_cs_resolves(self, cs_mp):
        from src.operators.macroscopic.eos import build_eos_fn

        fn = build_eos_fn("carnahan-starling", cs_mp)
        assert callable(fn)

    def test_cs_callable_returns_correct_shape(self, cs_mp):
        from src.operators.macroscopic.eos import build_eos_fn

        fn = build_eos_fn("carnahan-starling", cs_mp)
        rho = jnp.ones((NX, NY, NZ, 1, 1)) * _RHO_L
        result = fn(rho)
        assert result.shape == rho.shape

    def test_cs_callable_matches_numpy_reference(self, cs_mp):
        """Bound closure matches the numpy reference over the physical range."""
        from src.operators.macroscopic.eos import build_eos_fn

        fn = build_eos_fn("carnahan-starling", cs_mp)
        rho_vals = np.array([_RHO_V, 1.0, 5.0, _RHO_L])
        expected = _numpy_cs_mu0(rho_vals, _A, _B, _R, _T)

        result = fn(jnp.array(rho_vals))
        # float32 precision: JAX defaults to float32 without ENABLE_X64
        np.testing.assert_allclose(np.array(result), expected, rtol=1e-5)

    def test_dw_still_resolves(self, dw_mp):
        """double-well EOS is unaffected by CS addition — regression guard."""
        from src.operators.macroscopic.eos import build_eos_fn

        fn = build_eos_fn("double-well", dw_mp)
        rho = jnp.ones((NX, NY, NZ, 1, 1)) * 0.5
        result = fn(rho)
        assert result.shape == rho.shape

    def test_missing_cs_params_raises(self):
        """Building CS EOS without a/b/r/t raises ValueError."""
        from src.operators.macroscopic import MultiphaseParams
        from src.operators.macroscopic.eos import build_eos_fn

        mp_incomplete = MultiphaseParams(
            eos="carnahan-starling",
            kappa=_KAPPA,
            rho_l=_RHO_L,
            rho_v=_RHO_V,
            interface_width=_INTERFACE_WIDTH,
            # a_eos, b_eos, r_eos, t_eos intentionally absent
        )
        with pytest.raises(ValueError, match="required for Carnahan-Starling"):
            build_eos_fn("carnahan-starling", mp_incomplete)

    def test_unknown_eos_raises(self, cs_mp):
        from src.operators.macroscopic.eos import build_eos_fn

        with pytest.raises(ValueError):
            build_eos_fn("nonexistent-eos", cs_mp)


# =====================================================================
# 3. Pipeline integration
# =====================================================================


class TestCsEosPipeline:
    """CS EOS integrated into the multiphase macroscopic operator."""

    def test_returns_finite_triple(self, lattice, cs_mp):
        """compute_macroscopic_multiphase with CS EOS returns finite (rho, u_eq, force)."""
        from src.operators.macroscopic._multiphase import compute_macroscopic_multiphase

        gradient_standard, laplacian_density = _build_gradient_and_laplacian(lattice)

        # Use rho_l phase (uniform — interface not required for smoke test)
        f = jnp.ones((NX, NY, NZ, 9, 1)) * (_RHO_L / 9.0)

        rho, u_eq, force_total = compute_macroscopic_multiphase(
            f,
            lattice,
            cs_mp,
            gradient_standard=gradient_standard,
            laplacian_density=laplacian_density,
        )

        assert bool(jnp.all(jnp.isfinite(rho))), "rho contains non-finite values"
        assert bool(jnp.all(jnp.isfinite(u_eq))), "u_eq contains non-finite values"
        assert bool(jnp.all(jnp.isfinite(force_total))), "force_total contains non-finite values"

    def test_uniform_field_zero_force(self, lattice, cs_mp):
        """Perfectly uniform density → zero interaction force (no gradient)."""
        from src.operators.macroscopic._multiphase import compute_macroscopic_multiphase

        gradient_standard, laplacian_density = _build_gradient_and_laplacian(lattice)

        f = jnp.ones((NX, NY, NZ, 9, 1)) * (_RHO_L / 9.0)

        _, _, force_total = compute_macroscopic_multiphase(
            f,
            lattice,
            cs_mp,
            gradient_standard=gradient_standard,
            laplacian_density=laplacian_density,
        )

        np.testing.assert_allclose(np.array(force_total), 0.0, atol=1e-8)

    def test_output_shapes(self, lattice, cs_mp):
        from src.operators.macroscopic._multiphase import compute_macroscopic_multiphase

        gradient_standard, laplacian_density = _build_gradient_and_laplacian(lattice)
        f = jnp.ones((NX, NY, NZ, 9, 1)) * (_RHO_L / 9.0)

        rho, u_eq, force_total = compute_macroscopic_multiphase(
            f,
            lattice,
            cs_mp,
            gradient_standard=gradient_standard,
            laplacian_density=laplacian_density,
        )

        assert rho.shape == (NX, NY, NZ, 1, 1)
        assert u_eq.shape == (NX, NY, NZ, 1, 2)
        assert force_total.shape == (NX, NY, NZ, 1, 2)

    def test_jittable(self, lattice, cs_mp):
        from functools import partial
        from src.operators.macroscopic._multiphase import compute_macroscopic_multiphase

        gradient_standard, laplacian_density = _build_gradient_and_laplacian(lattice)
        f = jnp.ones((NX, NY, NZ, 9, 1)) * (_RHO_L / 9.0)

        jitted = jax.jit(
            partial(
                compute_macroscopic_multiphase,
                lattice=lattice,
                mp=cs_mp,
                gradient_standard=gradient_standard,
                laplacian_density=laplacian_density,
            )
        )
        rho, _, _ = jitted(f)
        assert rho.shape == (NX, NY, NZ, 1, 1)

    def test_build_setup_with_cs_eos(self):
        """build_setup resolves the full CS EOS operator chain without error."""
        from src.config.simulation_config import SimulationConfig
        from src.pipeline.setup import build_setup

        cfg = SimulationConfig(
            sim_type="multiphase",
            grid_shape=(8, 8),
            tau=0.99,
            nt=5,
            eos="carnahan-starling",
            kappa=_KAPPA,
            rho_l=_RHO_L,
            rho_v=_RHO_V,
            interface_width=_INTERFACE_WIDTH,
            a_eos=_A,
            b_eos=_B,
            r_eos=_R,
            t_eos=_T,
        )

        setup = build_setup(cfg)

        assert setup.multiphase_params is not None
        assert setup.multiphase_params.eos == "carnahan-starling"
        assert setup.step_fn is not None

    def test_cs_and_dw_produce_different_force(self, lattice, cs_mp, dw_mp):
        """CS and double-well EOS differ on a non-uniform density field (sanity check)."""
        from src.operators.macroscopic._multiphase import compute_macroscopic_multiphase

        gradient_standard, laplacian_density = _build_gradient_and_laplacian(lattice)

        # Build a non-uniform rho field: half liquid, half vapor
        rho_field = jnp.where(
            jnp.arange(NX)[:, None, None, None, None] < NX // 2,
            _RHO_L / 9.0,
            _RHO_V / 9.0,
        ) * jnp.ones((NX, NY, NZ, 9, 1))

        _, _, force_cs = compute_macroscopic_multiphase(
            rho_field,
            lattice,
            cs_mp,
            gradient_standard=gradient_standard,
            laplacian_density=laplacian_density,
        )
        _, _, force_dw = compute_macroscopic_multiphase(
            rho_field,
            lattice,
            dw_mp,
            gradient_standard=gradient_standard,
            laplacian_density=laplacian_density,
        )

        # Forces should be non-zero (interface present) and differ between EOS
        assert float(jnp.max(jnp.abs(force_cs))) > 0.0, "CS force should be non-zero at interface"
        assert float(jnp.max(jnp.abs(force_dw))) > 0.0, "DW force should be non-zero at interface"
        assert not np.allclose(np.array(force_cs), np.array(force_dw)), "CS and DW forces should differ"


# =====================================================================
# 4. Double-well bulk pressure (surface-tension calibration)
# =====================================================================


class TestDoubleWellPressure:
    """``double_well_pressure`` is thermodynamically consistent with ``_eos_double_well``."""

    _BETA = 8.0 * _KAPPA / (_INTERFACE_WIDTH**2 * (_RHO_L - _RHO_V) ** 2)

    def test_zero_at_both_coexistence_densities(self):
        """Flat-interface coexistence: mu_0 = 0 and psi = 0 at rho_l and rho_v, so p_0 = 0."""
        from src.operators.macroscopic.eos import double_well_pressure

        p = double_well_pressure(np.array([_RHO_V, _RHO_L]), self._BETA, _RHO_L, _RHO_V)
        np.testing.assert_allclose(np.asarray(p), 0.0, atol=1e-12)

    def test_gibbs_duhem_consistency(self):
        """dp/drho = rho * dmu_0/drho across the physical density range."""
        from src.operators.macroscopic.eos import double_well_pressure
        from src.operators.macroscopic.eos._double_well import _eos_double_well

        rho = np.linspace(_RHO_V, _RHO_L, 2001)
        p = np.asarray(double_well_pressure(rho, self._BETA, _RHO_L, _RHO_V))
        mu_0 = np.asarray(_eos_double_well(jnp.asarray(rho), self._BETA, _RHO_L, _RHO_V))

        dp = np.gradient(p, rho)
        rho_dmu = rho * np.gradient(mu_0, rho)

        scale = np.max(np.abs(dp))
        np.testing.assert_allclose(dp, rho_dmu, atol=1e-3 * scale)

    def test_accepts_jax_arrays(self):
        from src.operators.macroscopic.eos import double_well_pressure

        rho = jnp.ones((NX, NY, NZ, 1, 1)) * _RHO_L
        p = double_well_pressure(rho, self._BETA, _RHO_L, _RHO_V)
        assert p.shape == rho.shape


# ---------------------------------------------------------------------------
# build_multiphase_params
# ---------------------------------------------------------------------------


class TestBuildMultiphaseParams:
    """build_multiphase_params raises on missing fields and builds correctly."""

    def test_raises_when_required_field_missing(self):
        from types import SimpleNamespace
        from src.operators.macroscopic import build_multiphase_params

        cfg = SimpleNamespace(eos=None, kappa=0.01, rho_l=1.0, rho_v=0.1, interface_width=4)
        with pytest.raises(ValueError, match="'eos' is required"):
            build_multiphase_params(cfg)  # ty: ignore[invalid-argument-type]

    def test_raises_for_each_required_field(self):
        from types import SimpleNamespace
        from src.operators.macroscopic import build_multiphase_params

        base = {"eos": "double-well", "kappa": 0.01, "rho_l": 1.0, "rho_v": 0.1, "interface_width": 4}
        for field in ("kappa", "rho_l", "rho_v", "interface_width"):
            cfg = SimpleNamespace(**{**base, field: None})
            with pytest.raises(ValueError, match=f"'{field}' is required"):
                build_multiphase_params(cfg)  # ty: ignore[invalid-argument-type]

    def test_builds_correctly_with_valid_config(self):
        from types import SimpleNamespace
        from src.operators.macroscopic import MultiphaseParams
        from src.operators.macroscopic import build_multiphase_params

        cfg = SimpleNamespace(
            eos="carnahan-starling",
            kappa=_KAPPA,
            rho_l=_RHO_L,
            rho_v=_RHO_V,
            interface_width=_INTERFACE_WIDTH,
            a_eos=_A,
            b_eos=_B,
            r_eos=_R,
            t_eos=_T,
        )
        mp = build_multiphase_params(cfg)  # ty: ignore[invalid-argument-type]
        assert isinstance(mp, MultiphaseParams)
        assert mp.eos == "carnahan-starling"
        assert mp.kappa == _KAPPA
        assert mp.a_eos == _A

    def test_build_macroscopic_fn_invalid_scheme_raises(self):
        from src.operators.macroscopic import build_macroscopic_fn

        with pytest.raises(ValueError, match="not_a_scheme"):
            build_macroscopic_fn("not_a_scheme")
