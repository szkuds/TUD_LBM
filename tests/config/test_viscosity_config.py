"""``lambda_v`` / ``tau_gas``: the viscosity decoupled from the shear relaxation rate.

``tau`` keeps setting the liquid viscosity every reported number is built from;
``lambda_v`` is the shear relaxation time and ``tau_gas`` the gas viscosity.
"""

from __future__ import annotations
import jax.numpy as jnp
import numpy as np
import pytest
from src.config import DictAdapter
from src.operators.collision._mrt import SHEAR_MOMENTS

_FREE_RATES = (0.0, 1.1, 1.2, 0.0, 1.3, 0.0, 1.4, 1.0, 1.0)


def _config(**overrides: object):
    base: dict[str, object] = {
        "sim_type": "multiphase",
        "grid_shape": (16, 16),
        "tau": 0.75,
        "collision_scheme": "mrt",
        "k_diag": _FREE_RATES,
        "eos": "double-well",
        "kappa": 0.01,
        "rho_l": 2.0,
        "rho_v": 0.5,
        "interface_width": 4,
    }
    base.update(overrides)
    return DictAdapter().load(base)


def test_the_term_is_off_by_default():
    config = _config()

    assert config.viscosity_params is None
    assert config.relaxation_time == pytest.approx(0.75)


def test_parameters_carry_the_phase_viscosities():
    params = _config(lambda_v=1.0, tau_gas=0.6).viscosity_params

    assert params is not None
    assert params.lambda_v == pytest.approx(1.0)
    assert params.nu_l == pytest.approx((0.75 - 0.5) / 3.0)
    assert params.nu_v == pytest.approx((0.6 - 0.5) / 3.0)
    assert (params.rho_l, params.rho_v) == pytest.approx((2.0, 0.5))


def test_tau_gas_alone_keeps_the_shear_rate_at_tau():
    params = _config(tau_gas=0.6).viscosity_params

    assert params is not None
    assert params.lambda_v == pytest.approx(0.75)


def test_mrt_shear_rates_follow_lambda_v():
    config = _config(lambda_v=1.0)

    assert config.k_diag is not None
    for index, rate in enumerate(config.k_diag):
        expected = 1.0 if index in SHEAR_MOMENTS else _FREE_RATES[index]
        assert rate == pytest.approx(expected)


def test_single_phase_is_rejected():
    with pytest.raises(ValueError, match="require a multiphase sim_type"):
        DictAdapter().load({"sim_type": "single_phase", "grid_shape": (16, 16), "lambda_v": 1.0})


@pytest.mark.parametrize(("name", "value"), [("lambda_v", 0.5), ("tau_gas", 0.4)])
def test_relaxation_times_must_exceed_one_half(name, value):
    with pytest.raises(ValueError, match=f"{name} must be > 0.5"):
        _config(**{name: value})


@pytest.mark.parametrize(
    ("overrides", "phase"),
    [({"lambda_v": 0.8, "tau": 1.2}, "tau"), ({"lambda_v": 0.8, "tau_gas": 1.2}, "tau_gas")],
)
def test_a_beyond_the_positivity_bound_is_rejected(overrides, phase):
    """``|A| < lambda_v - 1/2`` (Zhang, Guo & Wang 2022) at both phases.

    ``A = lambda_v - tau_phase`` only reaches the bound from below, for a phase
    whose relaxation time exceeds ``2*lambda_v - 1/2``.
    """
    with pytest.raises(ValueError, match=f"lambda_v - {phase}"):
        _config(**overrides)


def test_setup_relaxes_at_lambda_v_and_carries_the_parameters():
    from src.pipeline.setup import build_setup

    config = _config(lambda_v=1.0, tau_gas=0.6)
    setup = build_setup(config)

    assert setup.tau == pytest.approx(1.0)
    assert setup.viscosity_params == config.viscosity_params


def test_configured_k_diag_reaches_the_collision():
    """The free rates a config sets must be the ones the MRT collision applies."""
    from src.operators.collision._mrt import collide_mrt
    from src.pipeline.setup import build_setup

    config = _config()
    setup = build_setup(config)
    assert setup.collision_fn is not None
    assert config.k_diag is not None
    rng = np.random.default_rng(0)
    f = jnp.asarray(rng.uniform(0.1, 1.0, (4, 3, 1, 9, 1)))
    feq = jnp.asarray(rng.uniform(0.1, 1.0, (4, 3, 1, 9, 1)))

    out = setup.collision_fn(f, feq, setup.tau)

    np.testing.assert_allclose(
        np.asarray(out), np.asarray(collide_mrt(f, feq, setup.tau, k_diag=jnp.asarray(config.k_diag))), atol=1e-12
    )
    assert not np.allclose(np.asarray(out), np.asarray(collide_mrt(f, feq, setup.tau)))
