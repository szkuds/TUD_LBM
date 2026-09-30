"""End-to-end: van der Waals fluid with the viscous-stress term (Zhang, Guo & Wang 2022).

Short versions of the paper's flat-interface and stationary-droplet cases at
``T = 0.8 T_c``, with ``lambda_v = 1`` and ``tau = 0.75`` (``A = 0.25``,
``nu_l = 0.0833``). The paper runs these to ``~1e-15`` spurious velocity; a CI
run cannot afford that relaxation, so these pin the direction — decaying
velocity, conserved mass, densities converging on the Maxwell pair.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.config import DictAdapter
from src.operators.macroscopic.eos._van_der_waals import coexistence_densities

_A, _B = 9.0 / 392.0, 2.0 / 21.0
_T = 0.8 / 14.0
_RHO_V, _RHO_L = coexistence_densities(_A, _B, 1.0, _T)


def _config(grid_shape: tuple[int, int], **overrides: object):
    base: dict[str, object] = {
        "sim_type": "multiphase",
        "grid_shape": grid_shape,
        "nt": 10,
        "tau": 0.75,
        "lambda_v": 1.0,
        "collision_scheme": "mrt",
        "k_diag": (0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0),
        "eos": "van-der-waals",
        "a_eos": _A,
        "b_eos": _B,
        "r_eos": 1.0,
        "t_eos": _T,
        "kappa": 0.01,
        "rho_l": _RHO_L,
        "rho_v": _RHO_V,
        "interface_width": 4,
    }
    base.update(overrides)
    return DictAdapter().load(base)


def _advance(setup, state, steps: int):
    body = jax.jit(lambda s: jax.lax.fori_loop(0, steps, lambda _, s: setup.step_fn(setup, s), s))
    return body(state)


def _max_speed(state) -> float:
    return float(np.abs(np.asarray(state.u)).max())


def test_flat_interface_relaxes_onto_the_maxwell_densities():
    from src.pipeline.runner import init_state
    from src.pipeline.setup import build_setup

    setup = build_setup(_config((101, 21)))
    x = np.arange(101, dtype=float)[:, None] * np.ones((1, 21))
    rho = _RHO_V + 0.5 * (_RHO_L - _RHO_V) * (np.tanh((x - 25.0) / 2.0) - np.tanh((x - 75.0) / 2.0))
    rho = jnp.asarray(rho)[:, :, None, None, None]
    assert setup.equilibrium_fn is not None
    f = setup.equilibrium_fn(rho, jnp.zeros((*rho.shape[:-1], 2)), setup.lattice)
    state = init_state(setup)._replace(f=f)
    mass = float(jnp.sum(f))

    early = _advance(setup, state, 3000)
    late = _advance(setup, early, 3000)

    assert np.isfinite(np.asarray(late.f)).all()
    assert float(jnp.sum(late.f)) == pytest.approx(mass, rel=1e-10)
    assert _max_speed(late) < _max_speed(early) < 1e-5
    rho_late = np.asarray(late.rho)
    assert rho_late.min() == pytest.approx(_RHO_V, abs=2e-3)
    assert rho_late.max() == pytest.approx(_RHO_L, abs=2e-3)


@pytest.mark.parametrize("tau_gas", [0.75, 0.6, 0.50025])
def test_droplet_is_stable_across_viscosity_ratios(tau_gas):
    """``tau_gas = 0.50025`` is the paper's ``nu_l / nu_v = 1000`` (Fig. 5)."""
    from src.pipeline.runner import init_state
    from src.pipeline.setup import build_setup

    config = _config(
        (64, 64),
        tau_gas=tau_gas,
        init_type="multiphase_bubbles",
        initialisation={"centres": [[0.5, 0.5]], "radii": [0.2], "dispersed": "liquid"},
    )
    setup = build_setup(config)
    state = init_state(setup)
    mass = float(jnp.sum(state.f))

    early = _advance(setup, state, 500)
    late = _advance(setup, early, 1500)

    assert np.isfinite(np.asarray(late.f)).all()
    assert float(jnp.sum(late.f)) == pytest.approx(mass, rel=1e-10)
    assert _max_speed(late) < _max_speed(early)
