"""The rising bubble at density ratio 1000: what fails, and the one lever that delays it.

Reference: ``~/TUD_LBM_data/bubble_simulation/2026-09-30/11-02-52_rising_bubble``
(128 x 256, double-well ``kappa = 0.04``, ``W = 5``, ``rho_l = 1``, BGK ``tau = 0.99``,
``g = 1e-6``) diverges at t ~ 21000. The study in ``.claude/density_ratio_study/``
traced it to a two-cell ``(-1)**k`` density zig-zag on the vapour side of the
interface, which grows until the vapour density reaches zero and ``u = j / rho``
blows up. It decays at ratio 100 and grows at 300 and 1000. Of the collision and
EOS changes tried, only a lower MRT bulk rate ``s_e`` (more bulk viscosity) delays
it materially: ``s_e = 0.5`` reaches ~80000 steps, ``s_e <= 0.3`` diverges at once.
No configuration tried survives 100000 steps with its bubble intact.

The full grid is required: at 64 x 128 (``R = 12.8`` against ``W = 5``) even ratio
100 diverges, so a reduced grid would not pin the same physics.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax
import numpy as np
import pytest
from src.config import DictAdapter
from src.operators.macroscopic.eos import build_pressure_fn
from src.pipeline.runner import init_state
from src.pipeline.setup import build_setup
from src.simulation_io.analysis.stability import compute_stability_metrics

if TYPE_CHECKING:
    from src.pipeline.state.state import State

pytestmark = [pytest.mark.integration, pytest.mark.slow]

#: Past the reference run's failure (t ~ 21000).
_STEPS = 25_000
_CHUNK = 1000
#: Indices into :func:`compute_stability_metrics`.
_RHO_MIN, _STRIPE = 2, 6


def _config(rho_v: float, **overrides: object):
    base: dict[str, object] = {
        "sim_type": "multiphase",
        "grid_shape": (128, 256),
        "nt": _STEPS,
        "tau": 0.99,
        "collision_scheme": "bgk",
        "init_type": "multiphase_bubbles",
        "eos": "double-well",
        "kappa": 0.04,
        "rho_l": 1.0,
        "rho_v": rho_v,
        "interface_width": 5,
        "bc_config": {"left": "periodic", "right": "periodic", "top": "bounce-back", "bottom": "bounce-back"},
        "initialisation": {"centres": [[0.5, 0.25]], "radii": [0.2], "dispersed": "vapour"},
        "gravity_force": {"inclination_angle_deg": 0.0, "force_g": 1e-6},
    }
    base.update(overrides)
    return DictAdapter().load(base)


def _run(config) -> tuple[State, np.ndarray]:
    """Advance to ``_STEPS`` (or the first non-finite sample); return the state and per-chunk metrics."""
    setup = build_setup(config)
    mp = setup.multiphase_params
    assert mp is not None
    step_fn = setup.step_fn
    assert step_fn is not None
    pressure_fn = build_pressure_fn(mp)
    advance = jax.jit(lambda s: jax.lax.fori_loop(0, _CHUNK, lambda _, s: step_fn(setup, s), s))
    sample = jax.jit(
        lambda s: compute_stability_metrics(s, gradient_density=setup.gradient_density, mp=mp, pressure_fn=pressure_fn)
    )

    state = init_state(setup)
    rows = []
    for _ in range(_STEPS // _CHUNK):
        state = advance(state)
        metrics = np.asarray(sample(state))
        rows.append(metrics)
        if not np.isfinite(metrics).all():
            break
    return state, np.stack(rows)


def _healthy(metrics: np.ndarray, rho_v: float) -> bool:
    """Every sample finite, the vapour above a tenth of its density, the zig-zag below 10%.

    The floor detects collapse toward zero, not an offset: even the stable ratio-100
    bubble holds its vapour at ~0.3 of the configured ``rho_v``, while a diverging run
    reaches ``1e-2 * rho_v`` and below.
    """
    return bool(
        np.isfinite(metrics).all() and metrics[:, _RHO_MIN].min() > 0.1 * rho_v and metrics[:, _STRIPE].max() < 0.1
    )


def test_ratio_100_is_stable_with_bgk():
    """Below the threshold the vapour zig-zag stays at the 1e-3 level (measured 1.4e-3 over 60000 steps)."""
    rho_v = 0.01
    _, metrics = _run(_config(rho_v))

    assert metrics.shape[0] == _STEPS // _CHUNK
    assert _healthy(metrics, rho_v)
    assert metrics[:, _STRIPE].max() < 0.01


@pytest.mark.xfail(
    strict=True,
    reason="ratio 1000 with BGK: the vapour-side zig-zag grows (e-fold ~3000 steps) and diverges at t ~ 21000",
)
def test_ratio_1000_is_stable_with_bgk():
    rho_v = 0.001
    _, metrics = _run(_config(rho_v))

    assert _healthy(metrics, rho_v)


def test_ratio_1000_low_bulk_rate_outlives_bgk_failure():
    """``s_e = 0.5`` (bulk viscosity ``cs2 * 1.5`` against BGK's ``cs2 * 0.49``) is healthy past t = 21000."""
    rho_v = 0.001
    config = _config(rho_v, collision_scheme="mrt", k_diag=(0.0, 0.5, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0))
    _, metrics = _run(config)

    assert metrics.shape[0] == _STEPS // _CHUNK
    assert _healthy(metrics, rho_v)
