"""The MRT shear moments carry ``1/tau``, so viscosity cannot decouple from ``tau``.

``nu = cs2*(tau - 0.5)`` is what every reported ``Oh``, ``La``, ``Re`` and ``Ar``
is computed from. A configured ``k_diag`` sets the shear relaxation directly, so
without this coupling a run relaxes at one rate while its overview reports
another — measured on a real config, ``k_diag[7:9] = 1.0`` against ``tau = 0.99``.
"""

from __future__ import annotations
import jax.numpy as jnp
import numpy as np
import pytest
from src.config import DictAdapter
from src.lattice.lattice import build_lattice
from src.operators.collision._mrt import _FREE_RATE_DEFAULTS
from src.operators.collision._mrt import SHEAR_MOMENTS
from src.operators.collision._mrt import M
from src.operators.collision._mrt import collide_mrt
from src.operators.collision._mrt import couple_shear_to_tau

_Q = len(_FREE_RATE_DEFAULTS)
#: A k_diag whose shear entries disagree with tau, as the measured config did.
_DECOUPLED = (0.0, 0.5, 0.6, 0.0, 1.2, 0.0, 1.2, 1.0, 1.0)
_TAU = 0.99


def _config(**overrides: object):
    base: dict[str, object] = {
        "sim_type": "single_phase",
        "grid_shape": (16, 16),
        "tau": _TAU,
        "collision_scheme": "mrt",
        "k_diag": _DECOUPLED,
    }
    base.update(overrides)
    return DictAdapter().load(base)


def test_shear_moments_are_exactly_the_stress_rows_of_the_basis():
    """``SHEAR_MOMENTS`` indexes ``p_xx``/``p_xy``, and no other pair could pass.

    The deviatoric stress components are the projections onto ``cx^2 - cy^2``
    and ``cx*cy``. Exactly two rows of ``M`` have a non-zero projection onto
    either, so asserting over *all nine* rows pins the indices to the basis
    itself — the weaker "zero mass and momentum" property is shared by the four
    free stability rows too, and would pass for the wrong pair.

    The velocity set comes from ``build_lattice``, so a reordering of D2Q9 that
    left ``M`` behind fails here instead of passing against a private copy.
    """
    basis = np.asarray(M)
    c = np.asarray(build_lattice("D2Q9").c)[0, 0, 0]
    c_x, c_y = c[:, 0], c[:, 1]

    # The projections are exactly 0 or exactly +-4, so any threshold well inside
    # that gap separates them; an inequality keeps it off the float-equality rule.
    stress_rows = {
        index
        for index, row in enumerate(basis)
        if abs((row * (c_x**2 - c_y**2)).sum()) > 1e-9 or abs((row * c_x * c_y).sum()) > 1e-9
    }

    assert stress_rows == set(SHEAR_MOMENTS)


def test_couple_shear_to_tau_rewrites_only_the_shear_entries():
    coupled = couple_shear_to_tau(_DECOUPLED, _TAU)

    for index in range(_Q):
        if index in SHEAR_MOMENTS:
            assert coupled[index] == pytest.approx(1.0 / _TAU)
        else:
            assert coupled[index] == pytest.approx(_DECOUPLED[index])


def test_a_too_short_k_diag_is_rejected():
    with pytest.raises(ValueError, match="at least 9 entries"):
        couple_shear_to_tau((0.0, 1.0, 1.0), _TAU)


@pytest.mark.parametrize("tau", [0.6, _TAU, 1.1929632313492542, 2.0])
def test_config_couples_a_decoupled_k_diag(tau):
    """The measured case (k_diag[7:9] = 1.0 against tau = 0.99), across tau.

    Asserted as the whole vector so the free stability knobs are pinned as
    untouched at every tau, not just at one.
    """
    config = _config(tau=tau)

    expected = tuple(1.0 / tau if i in SHEAR_MOMENTS else rate for i, rate in enumerate(_DECOUPLED))
    assert config.k_diag == pytest.approx(expected)


def test_bgk_is_untouched():
    config = _config(collision_scheme="bgk", k_diag=None)

    assert config.k_diag is None


def test_sweeps_couple_each_expanded_config():
    """``tau`` is sweep-eligible, so the coupling must follow each expanded value."""
    from src.config.array_expansion import expand_config

    configs, _ = expand_config(
        {
            "sim_type": "single_phase",
            "grid_shape": (16, 16),
            "tau": [0.8, 1.2],
            "collision_scheme": "mrt",
            "k_diag": _DECOUPLED,
        }
    )

    assert len(configs) == 2
    for config in configs:
        assert config.k_diag is not None
        assert config.k_diag[SHEAR_MOMENTS[0]] == pytest.approx(1.0 / config.tau)


def test_only_the_shear_moments_relax_and_they_relax_at_one_over_tau():
    """Measures, in moment space, which index ``tau`` actually drives.

    With every free rate zeroed, only ``SHEAR_MOMENTS`` may move, and each must
    move by exactly ``(1/tau)*(m_eq - m)``. Had the coupling landed on any other
    index this fails on both counts.
    """
    rng = np.random.default_rng(0)
    tau = 0.85
    shape = (4, 3, 1, _Q, 1)
    f = jnp.asarray(rng.uniform(0.1, 1.0, shape))
    feq = jnp.asarray(rng.uniform(0.1, 1.0, shape))
    shear_only = jnp.asarray(couple_shear_to_tau((0.0,) * _Q, tau))

    out = collide_mrt(f, feq, tau, k_diag=shear_only)

    moments = np.einsum("ij,...j->...i", np.asarray(M), np.asarray(f)[..., 0])
    moments_eq = np.einsum("ij,...j->...i", np.asarray(M), np.asarray(feq)[..., 0])
    moments_out = np.einsum("ij,...j->...i", np.asarray(M), np.asarray(out)[..., 0])

    for index in range(_Q):
        expected = moments[..., index]
        if index in SHEAR_MOMENTS:
            expected = expected + (moments_eq[..., index] - expected) / tau
        np.testing.assert_allclose(moments_out[..., index], expected, atol=1e-10)


def test_the_operator_default_relaxes_the_shear_moments_at_one_over_tau():
    """``k_diag=None`` must carry the coupling too, measured rather than compared.

    Asserting against ``couple_shear_to_tau(_FREE_RATE_DEFAULTS, tau)`` would be
    a tautology now that the operator builds its default with that same call, so
    this reads the relaxation rate back out of the result in moment space.
    """
    rng = np.random.default_rng(1)
    tau = 0.85
    shape = (4, 3, 1, _Q, 1)
    f = jnp.asarray(rng.uniform(0.1, 1.0, shape))
    feq = jnp.asarray(rng.uniform(0.1, 1.0, shape))

    out = collide_mrt(f, feq, tau)

    basis = np.asarray(M)
    moments = np.einsum("ij,...j->...i", basis, np.asarray(f)[..., 0])
    moments_eq = np.einsum("ij,...j->...i", basis, np.asarray(feq)[..., 0])
    moments_out = np.einsum("ij,...j->...i", basis, np.asarray(out)[..., 0])

    for index in SHEAR_MOMENTS:
        expected = moments[..., index] + (moments_eq[..., index] - moments[..., index]) / tau
        np.testing.assert_allclose(moments_out[..., index], expected, atol=1e-10)
