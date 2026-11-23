"""Smoke test of the electric potential solver: uniform conductivity between two plates.

For uniform sigma the potential solves Laplace's equation. The plates are the
solid surfaces of the bottom and top walls, half a cell outside the edge rows,
so with ``ny`` rows the exact potential is linear over a plate distance ``ny``:

    U(j) = U_bottom + (U_top - U_bottom) * (j + 1/2) / ny,   E = (0, -(U_top - U_bottom) / ny)

With uniform permittivity the charge ``div(eps*E)`` and the force are zero.
The solver starts from a uniform 0.5 V, not from ``init_state`` (which is
already the answer), and runs only the electric ``post_step``.

Two layers (liquid below, vapour above) make a leaky dielectric. The steady
current is continuous, ``sigma1*E1 = sigma2*E2`` with ``E1*d1 + E2*d2 = V``, and
the force is ``f_y = q*E - E^2/2 * d(eps)/dy = d/dy(eps*E^2/2)``: zero in each
bulk, with a column sum equal to the Maxwell-stress jump ``(eps2*E2^2 - eps1*E1^2)/2``.
The discrete force reaches that sum only as the interface widens (+29% for a
sharp step, +2.3% at W = 4, +0.6% at W = 8, second order in 1/W).
"""

from functools import partial
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.addons._electric import ElectricState
from src.addons._electric import build_electric_addon
from src.config.boundary import pad_modes
from src.config.boundary import periodic_axes
from src.config.simulation_config import SimulationConfig
from src.lattice.lattice import build_lattice
from src.operators.differential import build_differential
from src.operators.force._electric import ElectricForceModule
from src.operators.force._electric import ElectricParams
from src.operators.force._electric import electric_field
from src.operators.force._electric import equilibrium_h
from src.operators.streaming import build_streaming_fn
from src.pipeline.state.state import State

NX, NY = 4, 4
N_STEPS = 2000
_WALLS = {"bottom": "bounce-back", "top": "bounce-back"}


RHO_L, RHO_V = 2.0, 0.5


def _config(
    sigma: float = 0.1,
    voltage_bottom: float = 0.0,
    voltage_top: float = 1.0,
    *,
    ny: int = NY,
    conductivity: tuple[float, float] | None = None,
    permittivity: tuple[float, float] = (2.0, 2.0),
) -> SimulationConfig:
    """An electric config; ``conductivity``/``permittivity`` are ``(liquid, vapour)``."""
    sigma_liquid, sigma_vapour = conductivity or (sigma, sigma)
    section = {
        "permittivity_liquid": permittivity[0],
        "permittivity_vapour": permittivity[1],
        "conductivity_liquid": sigma_liquid,
        "conductivity_vapour": sigma_vapour,
        "voltage_bottom": voltage_bottom,
        "voltage_top": voltage_top,
    }
    return SimulationConfig(forces={"electric_force": section}, grid_shape=(NX, ny), boundary_conditions=_WALLS)


def _relax(config: SimulationConfig, n_steps: int = N_STEPS, rho: np.ndarray | None = None) -> tuple[State, State]:
    """The state after ``n_steps - 1`` and after ``n_steps`` potential steps from a uniform 0.5 V.

    *rho* is the density of each row (uniform 1 without it); it sets sigma and eps.
    """
    lattice = build_lattice("D2Q9")
    hooks = build_electric_addon(config)
    assert hooks is not None
    post_step = hooks.post_step
    assert post_step is not None
    setup = SimpleNamespace(
        grid_shape=config.grid_shape,
        lattice=lattice,
        streaming_fn=build_streaming_fn(periodic_axes(config)),
    )
    ny = config.grid_shape[1]
    column = jnp.ones(ny) if rho is None else jnp.asarray(rho)
    f = jnp.broadcast_to(lattice.w * column[None, :, None, None, None], (NX, ny, 1, lattice.q, 1))
    h = equilibrium_h(0.5 * jnp.ones((NX, ny, 1, 1, 1)), lattice.w)
    state = State(f=f, t=jnp.array(0), ext={"electric": ElectricState(h=h)})
    prev = jax.lax.fori_loop(0, n_steps - 1, lambda _, s: post_step(setup, s, s), state)  # ty: ignore[invalid-argument-type]
    return prev, post_step(setup, prev, prev)  # ty: ignore[invalid-argument-type]


def _potential(state: State) -> np.ndarray:
    return np.asarray(jnp.sum(state.ext["electric"].h, axis=3))[:, :, 0, 0]


def _exact_potential(voltage_bottom: float = 0.0, voltage_top: float = 1.0) -> np.ndarray:
    return voltage_bottom + (voltage_top - voltage_bottom) * (np.arange(NY) + 0.5) / NY


def _gradient(config: SimulationConfig):
    gradient, _ = build_differential(build_lattice("D2Q9"), pad_modes(config))
    return partial(gradient, ext={})


class TestPotentialRelaxes:
    """The potential relaxes to the linear profile between the plate surfaces."""

    @pytest.fixture(scope="class")
    def relaxed(self):
        return _relax(_config())

    def test_it_converges(self, relaxed):
        prev, last = relaxed
        assert np.abs(_potential(last) - _potential(prev)).max() < 1e-13

    def test_every_row_matches_the_exact_profile(self, relaxed):
        _, last = relaxed
        np.testing.assert_allclose(_potential(last), np.broadcast_to(_exact_potential(), (NX, NY)), atol=1e-12)

    @pytest.mark.parametrize("sigma", [0.02, 0.3, 1.0])
    def test_the_plates_do_not_move_with_the_relaxation_time(self, sigma):
        _, last = _relax(_config(sigma=sigma))
        np.testing.assert_allclose(_potential(last), np.broadcast_to(_exact_potential(), (NX, NY)), atol=1e-10)


class TestElectricField:
    """The field and the force of the relaxed potential."""

    @pytest.fixture(scope="class")
    def config(self):
        return _config()

    @pytest.fixture(scope="class")
    def last(self, config):
        return _relax(config)[1]

    def test_the_field_is_uniform_and_normal_to_the_plates(self, config, last):
        params = ElectricParams(**config.forces["electric_force"])
        ex, ey = electric_field(last.ext["electric"].h, _gradient(config), params)
        np.testing.assert_allclose(np.asarray(ex), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(ey), -1.0 / NY, atol=1e-12)

    def test_uniform_permittivity_gives_no_force(self, config, last):
        lattice = build_lattice("D2Q9")
        force = ElectricForceModule.build(
            config.forces["electric_force"], config.grid_shape, config=config, lattice=lattice
        )
        result = force.compute(last, gradient_standard=_gradient(config))
        np.testing.assert_allclose(np.asarray(result), 0.0, atol=1e-12)

    def test_a_common_voltage_shift_changes_neither_field_nor_force(self, config, last):
        shifted_config = _config(voltage_bottom=5.0, voltage_top=6.0)
        shifted = _relax(shifted_config)[1]
        np.testing.assert_allclose(_potential(shifted) - 5.0, _potential(last), atol=1e-11)
        lattice = build_lattice("D2Q9")
        for cfg, state in ((config, last), (shifted_config, shifted)):
            force = ElectricForceModule.build(cfg.forces["electric_force"], cfg.grid_shape, config=cfg, lattice=lattice)
            np.testing.assert_allclose(
                np.asarray(force.compute(state, gradient_standard=_gradient(cfg))), 0.0, atol=1e-11
            )


def _force_y(config: SimulationConfig, state: State) -> np.ndarray:
    """``f_y`` of each row at ``x = 0``."""
    force = ElectricForceModule.build(
        config.forces["electric_force"], config.grid_shape, config=config, lattice=build_lattice("D2Q9")
    )
    return np.asarray(force.compute(state, gradient_standard=_gradient(config)))[0, :, 0, 0, 1]


class TestSharpTwoLayer:
    """Liquid in rows 0..7, vapour in rows 8..15: interface at y = 7.5, layers 8 cells thick."""

    LAYER_NY, K = 16, 8
    SIGMA, EPSILON = (0.3, 0.05), (4.0, 1.0)

    @pytest.fixture(scope="class")
    def config(self):
        return _config(ny=self.LAYER_NY, conductivity=self.SIGMA, permittivity=self.EPSILON)

    @pytest.fixture(scope="class")
    def relaxed(self, config):
        rho = np.where(np.arange(self.LAYER_NY) < self.K, RHO_L, RHO_V)
        return _relax(config, 20000, rho)

    @pytest.fixture(scope="class")
    def fields(self):
        """The exact ``(E1, E2)`` from current continuity and the plate voltage."""
        (sigma1, sigma2), d1, d2 = self.SIGMA, self.K, self.LAYER_NY - self.K
        denominator = sigma2 * d1 + sigma1 * d2
        return sigma2 / denominator, sigma1 / denominator

    def test_it_converges(self, relaxed):
        prev, last = relaxed
        assert np.abs(_potential(last) - _potential(prev)).max() < 1e-13

    def test_the_current_is_continuous_across_the_interface(self, relaxed, fields):
        _, last = relaxed
        e1, e2 = fields
        y = np.arange(self.LAYER_NY) + 0.5
        exact = np.where(y < self.K, e1 * y, e1 * self.K + e2 * (y - self.K))
        np.testing.assert_allclose(_potential(last), np.broadcast_to(exact, (NX, self.LAYER_NY)), atol=1e-12)

    def test_the_bulk_fields_are_exact(self, config, relaxed, fields):
        _, last = relaxed
        e1, e2 = fields
        params = ElectricParams(**config.forces["electric_force"])
        ex, ey = electric_field(last.ext["electric"].h, _gradient(config), params)
        np.testing.assert_allclose(np.asarray(ex), 0.0, atol=1e-12)
        # The central gradient spreads the jump over the two rows each side of the interface.
        np.testing.assert_allclose(np.asarray(ey)[:, : self.K - 1], -e1, atol=1e-12)
        np.testing.assert_allclose(np.asarray(ey)[:, self.K + 1 :], -e2, atol=1e-12)

    def test_the_force_acts_only_at_the_interface(self, config, relaxed):
        _, last = relaxed
        force_y = _force_y(config, last)
        np.testing.assert_allclose(force_y[: self.K - 2], 0.0, atol=1e-12)
        np.testing.assert_allclose(force_y[self.K + 2 :], 0.0, atol=1e-12)
        assert np.abs(force_y[self.K - 2 : self.K + 2]).min() > 1e-5


class TestSmoothTwoLayer:
    """A tanh interface of width W: the force converges to the Maxwell-stress jump."""

    LAYER_NY = 64
    SIGMA, EPSILON = (0.3, 0.05), (4.0, 1.0)

    def _run(self, width: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """``(sum f_y / Maxwell jump, f_y, U, predicted U)`` for an interface of *width*."""
        from src.operators.force._electric import rho_to_phi

        y = np.arange(self.LAYER_NY)
        rho = 0.5 * (RHO_L + RHO_V) - 0.5 * (RHO_L - RHO_V) * np.tanh(2 * (y - (self.LAYER_NY / 2 - 0.5)) / width)
        config = _config(ny=self.LAYER_NY, conductivity=self.SIGMA, permittivity=self.EPSILON)
        last = _relax(config, 60000, rho)[1]
        # Each node is a resistance 1/sigma_j in series; the plates are half a cell out.
        resistance = 1.0 / np.asarray(rho_to_phi(jnp.asarray(rho), *self.SIGMA))
        current = 1.0 / resistance.sum()
        predicted = current * (np.cumsum(resistance) - 0.5 * resistance)
        epsilon = np.asarray(rho_to_phi(jnp.asarray(rho), *self.EPSILON))
        e1, e2 = current * resistance[0], current * resistance[-1]
        jump = 0.5 * (epsilon[-1] * e2**2 - epsilon[0] * e1**2)
        force_y = _force_y(config, last)
        return float(force_y.sum() / jump), force_y, _potential(last)[0], predicted

    @pytest.fixture(scope="class")
    def runs(self):
        return {width: self._run(width) for width in (4.0, 8.0)}

    def test_the_potential_carries_a_continuous_current(self, runs):
        for _, _, potential, predicted in runs.values():
            np.testing.assert_allclose(potential, predicted, atol=1e-8)

    def test_the_force_vanishes_in_the_bulk(self, runs):
        for _, force_y, _, _ in runs.values():
            np.testing.assert_allclose(force_y[:4], 0.0, atol=1e-8)
            np.testing.assert_allclose(force_y[-4:], 0.0, atol=1e-8)

    def test_the_force_sums_to_the_maxwell_stress_jump(self, runs):
        assert runs[4.0][0] == pytest.approx(1.0, abs=0.03)
        assert runs[8.0][0] == pytest.approx(1.0, abs=0.01)

    def test_the_force_error_is_second_order_in_the_width(self, runs):
        assert (runs[4.0][0] - 1.0) / (runs[8.0][0] - 1.0) > 3.0
