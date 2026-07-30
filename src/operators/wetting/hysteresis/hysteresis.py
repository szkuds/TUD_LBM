"""Wetting hysteresis optimisation — pure functions.

Ported from :class:`update_timestep.UpdateMultiphaseHysteresis`.

The legacy class stores mutable wetting parameters on ``self`` and
uses ``@partial(jit, static_argnums=(0,))`` which causes JIT cache
bloat.  This module replaces it with pure functions that operate on
the :class:`~state.state.WettingState` NamedTuple carried through
``jax.lax.scan``.

All inner optimisation loops use ``optax`` + ``jax.lax.while_loop``
with early convergence exit and are fully jittable.

Design
~~~~~~
``update_wetting_state`` is the top-level entry point.  It:

1. Measures contact angles and contact-line locations from ``rho_t_plus1``.
2. Checks whether each side is inside the hysteresis window.
3. Builds per-side objectives (CLL-pin or CA-target, selected via
   ``jnp.where``).
4. Runs **two** sequential ``jax.lax.while_loop`` optimisations —
   one for each side — masking the other side's parameters.
5. Returns an updated :class:`WettingState` — no mutation.

The inner ``_evaluate_with_params`` closure performs a single LBM
step with trial wetting parameters so that ``jax.value_and_grad``
can differentiate through it.

Angle convention
~~~~~~~~~~~~~~~~
``compute_contact_angle`` reports the angle through the **dispersed**
phase — the liquid for a droplet, the vapour for a bubble — so the
``ca_advancing`` / ``ca_receding`` window is in those terms too. For a
droplet that is the usual liquid angle and nothing changes. For a bubble
it is the vapour angle, and because vapour advancing is liquid receding,
a window meant as liquid ``[rec, adv]`` becomes ``[180 - adv, 180 - rec]``
here.

``phi`` and ``d_rho``, by contrast, are **liquid-frame** knobs and are
topology-independent: ``phi`` inflates the ghost-row density so the wall
looks more liquid, ``d_rho`` deflates it. Mapping a dispersed-frame angle
error onto them therefore flips with topology, and :func:`_phi_is_active`
is where that translation happens — the two contact-angle branches invert
for a bubble, the two contact-line-pinning branches do not (see its
docstring).
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Protocol
from typing import cast
import jax
import jax.numpy as jnp
from src.operators.wetting._contact_angle import compute_contact_angle
from src.operators.wetting._contact_line import compute_contact_line_location
from src.operators.wetting._interface_crossings import detect_bubble
from src.operators.wetting._params import WettingParams
from src.registry import wetting_operator
from src.simulation_io.analysis import wetting_debug

if TYPE_CHECKING:
    import types
    from collections.abc import Callable
    from collections.abc import Mapping
    from src.pipeline.setup import SimulationSetup
    from src.pipeline.state.state import WettingState


class _OptaxLike(Protocol):
    """Minimal structural type for optax-compatible optimisers."""

    def init(self, params: object) -> object: ...
    def update(self, updates: object, state: object, params: object | None = ...) -> tuple[object, object]: ...


# ── Helpers ──────────────────────────────────────────────────────────

# Neutral values — the inactive parameter is snapped to these when the
# directional split is applied.
_PHI_NEUTRAL: jnp.ndarray = jnp.array(1.0)
_D_RHO_NEUTRAL: jnp.ndarray = jnp.array(0.0)

# Tolerance for "phi is still sitting on its clamp floor". `_clamp_params`
# pins phi at exactly `_PHI_NEUTRAL`, so a strict `<` comparison is
# unreachable and the d_rho fallback below it would never fire.
_PHI_FLOOR_EPS: float = 1e-6


def _import_optax() -> types.ModuleType:
    """Import optional ``optax`` dependency with a clear install hint."""
    try:
        import optax
    except ImportError as err:
        msg = "The 'optax' package is required for hysteresis wetting.\nInstall it with:  pip install optax"
        raise ImportError(msg) from err
    return optax


def _liquid_is_advancing(
    cll_now: jnp.ndarray,
    cll_stored: jnp.ndarray,
    is_bubble: jnp.ndarray,
    *,
    side: str,
) -> jnp.ndarray:
    """Return True if the liquid is advancing over dry wall at this contact line.

    Contact-line labels are positional, so the dispersed phase expanding is the
    left CL moving in ``−tangential`` and the right CL moving in
    ``+tangential``. For a droplet the dispersed phase *is* the liquid, so that
    expansion is the liquid advancing. For a bubble it is the vapour, so the
    same motion is the liquid receding and the test inverts.

    Args:
        cll_now: Freshly measured contact-line location (scalar).
        cll_stored: Contact-line location carried in ``WettingState``.
        is_bubble: Bool scalar — the dispersed phase at the wall is vapour.
        side: ``"left"`` or ``"right"``.

    Returns:
        Boolean JAX scalar.
    """
    if side == "left":
        dispersed_expanding = cll_now < cll_stored
    elif side == "right":
        dispersed_expanding = cll_now > cll_stored
    else:
        msg = f"side must be 'left' or 'right', got {side!r}"
        raise ValueError(msg)
    return dispersed_expanding ^ is_bubble


def _phi_is_active(
    in_window: jnp.ndarray,
    above_window: jnp.ndarray,
    forward_drift: jnp.ndarray,
    is_bubble: jnp.ndarray,
) -> jnp.ndarray:
    """Return True if phi is the active parameter for this side.

    ``phi`` makes the wall *more* liquid-wetting and ``d_rho`` makes it *less*
    — both liquid-frame statements, true for either topology. The selection
    therefore has to be reasoned in the liquid frame, and the measured angle is
    dispersed-frame (see the module docstring), so the two contact-angle
    branches invert for a bubble:

    ==========================  ==================  ==============  ======
    regime                      theta_liq must      wall becomes    knob
    ==========================  ==================  ==============  ======
    above window, droplet       decrease            more wetting    phi
    above window, bubble        increase            less wetting    d_rho
    below window, droplet       increase            less wetting    d_rho
    below window, bubble        decrease            more wetting    phi
    in window, liquid receding  --                  more wetting    phi
    in window, liquid advancing --                  less wetting    d_rho
    ==========================  ==================  ==============  ======

    The two in-window rows are topology-independent: ``forward_drift`` arrives
    already converted to the liquid frame by :func:`_liquid_is_advancing`, and
    the knobs are liquid-frame too, so no further flip is needed. Pinning the
    contact line means resisting whichever way the liquid is moving.

    Args:
        in_window: bool scalar — CA is between ca_receding and ca_advancing.
        above_window: bool scalar — CA > ca_advancing. Mutually exclusive with
            ``in_window``; both False means below the window.
        forward_drift: bool scalar — the **liquid** is advancing over dry wall
            at this contact line, as returned by :func:`_liquid_is_advancing`.
        is_bubble: bool scalar — the phase dispersed at the wall is vapour, so
            the reported angle is the complement of the liquid angle.

    Returns:
        Boolean JAX scalar; True means phi is active, False means d_rho is active.
    """
    # Out of window: push the reported angle back toward the exceeded bound.
    # Which knob does that depends on the topology, hence the is_bubble flip.
    ca_branch = jnp.where(above_window, ~is_bubble, is_bubble)
    return jnp.where(in_window, ~forward_drift, ca_branch)


def _side_hyperparams(
    hysteresis_config: Mapping[str, object],
    above_window: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(learning_rate, max_iterations)`` for one side.

    A side whose contact angle has run above ``ca_advancing`` is the urgent
    case, and may be given a larger step and a longer budget via
    ``learning_rate_above`` / ``max_iterations_above``.  Both selections are
    traced, so each side is keyed on **its own** flag.

    Note that only the above-window excursion is treated as urgent; a side
    below ``ca_receding`` gets the default budget.

    Args:
        hysteresis_config: The ``hysteresis_config`` mapping.
        above_window: bool scalar — this side's CA exceeds ca_advancing.

    Returns:
        ``(lr, max_iterations)`` as JAX scalars.
    """
    lr_default = float(cast("float", hysteresis_config.get("learning_rate", 0.01)))
    lr_above = float(cast("float", hysteresis_config.get("learning_rate_above", 0.05)))
    max_iter_default = int(cast("int", hysteresis_config.get("max_iterations", 50)))
    max_iter_above = int(cast("int", hysteresis_config.get("max_iterations_above", max_iter_default)))
    return (
        jnp.where(above_window, lr_above, lr_default),
        jnp.where(above_window, max_iter_above, max_iter_default),
    )


def _clamp_params(params: WettingParams, w: jnp.ndarray) -> WettingParams:
    """Clamp wetting parameters to physically reasonable, W-scaled ranges.

    The wetting parameters act on near-wall density profiles whose
    magnitude scales inversely with the interface width ``W``:
    ``phi`` ∈ [1, 1 + 2.5/W] and ``d_rho`` ∈ [0, 1.5/W].  At the base
    resolution (W = 5) these reduce to the previous fixed bounds
    (1.5 and 0.3).

    Note: ``jnp.clip`` has zero gradient at the boundaries, so a
    parameter sitting at a clamp limit receives no further gradient
    signal in that direction.
    """
    return WettingParams(
        phi_left=jnp.clip(params.phi_left, 1.0, 1.0 + 2.5 / w),
        phi_right=jnp.clip(params.phi_right, 1.0, 1.0 + 2.5 / w),
        d_rho_left=jnp.clip(params.d_rho_left, 0.0, 1.5 / w),
        d_rho_right=jnp.clip(params.d_rho_right, 0.0, 1.5 / w),
    )


def _cost_cll(cll_target: jnp.ndarray, cll_current: jnp.ndarray) -> jnp.ndarray:
    """Huber loss for CLL pinning — smooth gradient near zero, linear elsewhere."""
    err = jnp.abs(cll_target - cll_current)
    delta = 0.5
    return jnp.where(err < delta, 0.5 * err**2, delta * (err - 0.5 * delta))


def _cost_ca(ca_target: jnp.ndarray, ca_current: jnp.ndarray) -> jnp.ndarray:
    """Huber loss for CA targeting — smooth gradient near zero, linear elsewhere."""
    err = jnp.abs(ca_target - ca_current)
    delta = 5.0  # Degrees
    return jnp.where(err < delta, 0.5 * err**2, delta * (err - 0.5 * delta))


def _cost_above(ca_adv: jnp.ndarray, ca_current: jnp.ndarray) -> jnp.ndarray:
    """One-sided Huber loss that penalises only CA values above ca_adv."""
    excess = jnp.maximum(ca_current - ca_adv, 0.0)
    delta = 5.0  # Degrees
    return jnp.where(excess < delta, 0.5 * excess**2, delta * (excess - 0.5 * delta))


def _cost_below(ca_rec: jnp.ndarray, ca_current: jnp.ndarray) -> jnp.ndarray:
    """One-sided Huber loss that penalises only CA values below ca_rec."""
    deficit = jnp.maximum(ca_rec - ca_current, 0.0)
    delta = 5.0  # Degrees
    return jnp.where(deficit < delta, 0.5 * deficit**2, delta * (deficit - 0.5 * delta))


def _mask_left_d_rho(g: WettingParams) -> WettingParams:
    z = jnp.zeros_like
    return WettingParams(
        phi_left=z(g.phi_left),
        phi_right=z(g.phi_right),
        d_rho_left=g.d_rho_left,
        d_rho_right=z(g.d_rho_right),
    )


def _mask_left_phi(g: WettingParams) -> WettingParams:
    z = jnp.zeros_like
    return WettingParams(
        phi_left=g.phi_left,
        phi_right=z(g.phi_right),
        d_rho_left=z(g.d_rho_left),
        d_rho_right=z(g.d_rho_right),
    )


def _mask_right_d_rho(g: WettingParams) -> WettingParams:
    z = jnp.zeros_like
    return WettingParams(
        phi_left=z(g.phi_left),
        phi_right=z(g.phi_right),
        d_rho_left=z(g.d_rho_left),
        d_rho_right=g.d_rho_right,
    )


def _mask_right_phi(g: WettingParams) -> WettingParams:
    z = jnp.zeros_like
    return WettingParams(
        phi_left=z(g.phi_left),
        phi_right=g.phi_right,
        d_rho_left=z(g.d_rho_left),
        d_rho_right=z(g.d_rho_right),
    )


# ── Generic optimisation routine ─────────────────────────────────────


def _optimise_single_param(
    objective_fn: Callable[[WettingParams], jnp.ndarray],
    initial_params: WettingParams,
    grad_mask_fn: Callable[[WettingParams], WettingParams],
    optimiser: _OptaxLike,
    max_iterations: int | jnp.ndarray,
    w: jnp.ndarray,
    loss_tol: float = 1e-4,
) -> tuple[WettingParams, jnp.ndarray]:
    """Run an ``optax`` optimisation loop with masked gradients.

    Uses ``jax.lax.while_loop`` with early exit: the loop terminates
    when **either** ``max_iterations`` is reached **or** the loss drops
    below ``loss_tol``, whichever comes first.

    Args:
        objective_fn: ``params → scalar_loss``.
        initial_params: Starting :class:`WettingParams`.
        grad_mask_fn: ``grads → grads`` that zeros out all but the
            target parameter(s).
        optimiser: An ``optax`` optimiser instance.
        max_iterations: Maximum number of inner steps.  May be a traced
            scalar — ``jax.lax.while_loop`` takes its bound from ``cond_fn``,
            so the trip count does not need to be static.
        w: Interface width used to scale the parameter clamp bounds
            (see :func:`_clamp_params`).
        loss_tol: Convergence tolerance; the loop exits once the loss
            drops to or below this value.  The default corresponds to
            ~0.014° CA / ~0.014 l.u. CLL error in the quadratic regime
            of the Huber objectives.

    Returns:
        ``(final_params, final_loss)``.
    """
    import optax  # lazy import — optional dependency

    opt_state = optimiser.init(initial_params)
    initial_loss = objective_fn(initial_params)

    def cond_fn(carry: tuple) -> jnp.ndarray:
        _params, _opt_state, loss, iteration = carry
        return (iteration < max_iterations) & (loss > loss_tol)

    def body_fn(carry: tuple) -> tuple:
        params, opt_state, _loss, iteration = carry
        loss, grads = jax.value_and_grad(objective_fn)(params)
        grads = grad_mask_fn(grads)
        updates, new_opt_state = optimiser.update(grads, opt_state, params)
        new_params = _clamp_params(
            cast("WettingParams", optax.apply_updates(params, cast("WettingParams", updates))), w
        )
        return (new_params, new_opt_state, loss, iteration + 1)

    init_carry = (initial_params, opt_state, initial_loss, jnp.array(0))
    final_params, _opt_state, final_loss, iters = jax.lax.while_loop(
        cond_fn,
        body_fn,
        init_carry,
    )
    wetting_debug.log_optimiser_exit(iters, max_iterations, final_loss)
    return final_params, final_loss


# ── Top-level entry point ────────────────────────────────────────────


def _get_hysteresis_window_chemical_step(setup: SimulationSetup, cll: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (ca_advancing, ca_receding) based on CLL position relative to chemical step."""
    if setup.config.chemical_step_config is None:
        msg = "chemical_step_config is required for chemical step hysteresis"
        raise TypeError(msg)
    csc = setup.config.chemical_step_config
    step_x = csc["chemical_step_location"] * setup.config.grid_shape[0]
    return jax.lax.cond(
        cll < step_x,
        lambda: (
            csc["ca_advancing_pre_step"],
            csc["ca_receding_pre_step"],
        ),
        lambda: (
            csc["ca_advancing_post_step"],
            csc["ca_receding_post_step"],
        ),
    )


def _update_wetting_state_impl(
    wetting: WettingState,
    rho_t_plus1: jnp.ndarray,
    setup: SimulationSetup,
    trial_step_fn: Callable[[WettingParams], tuple[jnp.ndarray, jnp.ndarray]],
    *,
    ca_adv_left: jnp.ndarray,
    ca_rec_left: jnp.ndarray,
    ca_adv_right: jnp.ndarray,
    ca_rec_right: jnp.ndarray,
) -> WettingState:
    """Shared implementation for hysteresis wetting updates."""
    if setup.multiphase_params is None:
        msg = "multiphase_params is required for hysteresis wetting update"
        raise TypeError(msg)
    if setup.config.hysteresis_config is None:
        msg = "hysteresis_config is required for hysteresis wetting update"
        raise TypeError(msg)
    mp = setup.multiphase_params
    rho_mean = 0.5 * (mp.rho_l + mp.rho_v)
    w = jnp.array(float(mp.interface_width))
    if setup.wetting_edge is None:
        msg = "wetting_edge is required for hysteresis wetting update"
        raise TypeError(msg)
    edge = setup.wetting_edge

    ca_left_tplus1, ca_right_tplus1 = compute_contact_angle(rho_t_plus1, jnp.array(rho_mean), edge=edge)
    cll_left_tplus1, cll_right_tplus1 = compute_contact_line_location(
        rho_t_plus1,
        ca_left_tplus1,
        ca_right_tplus1,
        jnp.array(rho_mean),
        edge=edge,
    )

    is_bubble = detect_bubble(rho_t_plus1, jnp.array(rho_mean), edge=edge)
    forward_drift_right = _liquid_is_advancing(cll_right_tplus1, wetting.cll_right, is_bubble, side="right")
    forward_drift_left = _liquid_is_advancing(cll_left_tplus1, wetting.cll_left, is_bubble, side="left")

    in_window_left = (ca_left_tplus1 >= ca_rec_left) & (ca_left_tplus1 <= ca_adv_left)
    in_window_right = (ca_right_tplus1 >= ca_rec_right) & (ca_right_tplus1 <= ca_adv_right)
    above_window_left = ca_left_tplus1 > ca_adv_left
    above_window_right = ca_right_tplus1 > ca_adv_right

    phi_active_right = _phi_is_active(in_window_right, above_window_right, forward_drift_right, is_bubble)
    phi_active_left = _phi_is_active(in_window_left, above_window_left, forward_drift_left, is_bubble)

    hc = setup.config.hysteresis_config
    lr_left, max_iter_left = _side_hyperparams(hc, above_window_left)
    lr_right, max_iter_right = _side_hyperparams(hc, above_window_right)
    loss_tol = hc.get("loss_tol", 1e-4)

    params = WettingParams(
        phi_left=jnp.where(phi_active_left, wetting.phi_left, _PHI_NEUTRAL),
        phi_right=jnp.where(phi_active_right, wetting.phi_right, _PHI_NEUTRAL),
        d_rho_left=jnp.where(phi_active_left, _D_RHO_NEUTRAL, wetting.d_rho_left),
        d_rho_right=jnp.where(phi_active_right, _D_RHO_NEUTRAL, wetting.d_rho_right),
    )

    optax = _import_optax()
    # One optimiser per side: the learning rate is keyed on that side's own
    # above-window flag, so an in-window side is not dragged along by the other.
    optimiser_left = optax.adam(lr_left)
    optimiser_right = optax.adam(lr_right)

    def evaluate_fn(params: WettingParams) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        _, rho_out = trial_step_fn(params)
        ca_l, ca_r = compute_contact_angle(rho_out, jnp.array(rho_mean), edge=edge)
        cll_l, cll_r = compute_contact_line_location(rho_out, ca_l, ca_r, jnp.array(rho_mean), edge=edge)
        return ca_l, ca_r, cll_l, cll_r

    def left_objective(p: WettingParams) -> jnp.ndarray:
        ca_l, _, cll_l, _ = evaluate_fn(p)
        cost_in = _cost_cll(wetting.cll_left, cll_l)
        cost_below = _cost_below(ca_rec_left, ca_l)
        cost_above = _cost_above(ca_adv_left, ca_l)
        return jnp.where(in_window_left, cost_in, jnp.where(above_window_left, cost_above, cost_below))

    def right_objective(p: WettingParams) -> jnp.ndarray:
        _, ca_r, _, cll_r = evaluate_fn(p)
        cost_in = _cost_cll(wetting.cll_right, cll_r)
        cost_below = _cost_below(ca_rec_right, ca_r)
        cost_above = _cost_above(ca_adv_right, ca_r)
        return jnp.where(in_window_right, cost_in, jnp.where(above_window_right, cost_above, cost_below))

    def _opt_left(p: WettingParams) -> tuple[WettingParams, jnp.ndarray]:
        return jax.lax.cond(
            phi_active_left,
            lambda pp: _optimise_single_param(
                left_objective, pp, _mask_left_phi, optimiser_left, max_iter_left, w, loss_tol
            ),
            lambda pp: _optimise_single_param(
                left_objective, pp, _mask_left_d_rho, optimiser_left, max_iter_left, w, loss_tol
            ),
            p,
        )

    def _opt_right(p: WettingParams) -> tuple[WettingParams, jnp.ndarray]:
        return jax.lax.cond(
            phi_active_right,
            lambda pp: _optimise_single_param(
                right_objective, pp, _mask_right_phi, optimiser_right, max_iter_right, w, loss_tol
            ),
            lambda pp: _optimise_single_param(
                right_objective, pp, _mask_right_d_rho, optimiser_right, max_iter_right, w, loss_tol
            ),
            p,
        )

    params_after_left, loss_left = _opt_left(params)
    new_params, loss_right = _opt_right(params_after_left)

    # Fallback: if the phi path was selected but phi saturated back at its
    # clamp floor without converging, phi was the wrong knob for this side —
    # `jnp.clip` has zero gradient there, so it cannot recover. Retry with
    # d_rho, warm-started from the stored accumulated value.
    def _fallback_d_rho_left(p: WettingParams) -> WettingParams:
        fallback = WettingParams(
            phi_left=_PHI_NEUTRAL,
            phi_right=p.phi_right,
            d_rho_left=wetting.d_rho_left,
            d_rho_right=p.d_rho_right,
        )
        return _optimise_single_param(
            left_objective, fallback, _mask_left_d_rho, optimiser_left, max_iter_left, w, loss_tol
        )[0]

    def _fallback_d_rho_right(p: WettingParams) -> WettingParams:
        fallback = WettingParams(
            phi_left=p.phi_left,
            phi_right=_PHI_NEUTRAL,
            d_rho_left=p.d_rho_left,
            d_rho_right=wetting.d_rho_right,
        )
        return _optimise_single_param(
            right_objective, fallback, _mask_right_d_rho, optimiser_right, max_iter_right, w, loss_tol
        )[0]

    # The `loss > loss_tol` conjunct is what keeps this from firing on a side
    # whose phi legitimately converged at ~1.0 — without it every such side
    # would pay a second optimisation. `_optimise_single_param` returns the
    # while_loop carry loss, which lags one iteration behind the returned
    # params; that makes the test conservative rather than wrong, since a
    # parameter pinned at a clamp bound is not moving the loss anyway.
    final_params = jax.lax.cond(
        phi_active_left & (new_params.phi_left <= _PHI_NEUTRAL + _PHI_FLOOR_EPS) & (loss_left > loss_tol),
        _fallback_d_rho_left,
        lambda p: p,
        new_params,
    )
    final_params = jax.lax.cond(
        phi_active_right & (new_params.phi_right <= _PHI_NEUTRAL + _PHI_FLOOR_EPS) & (loss_right > loss_tol),
        _fallback_d_rho_right,
        lambda p: p,
        final_params,
    )

    # Guarded at the call site because the loss terms below each cost a
    # full trial step — `log_sides` re-checks the flag itself.
    if wetting_debug.enabled():
        wetting_debug.log_sides(
            wetting_debug.SideDebugSample(
                ca=ca_left_tplus1,
                ca_adv=ca_adv_left,
                ca_rec=ca_rec_left,
                cll=cll_left_tplus1,
                phi=final_params.phi_left,
                d_rho=final_params.d_rho_left,
                phi_active=phi_active_left,
                loss=left_objective(final_params),
            ),
            wetting_debug.SideDebugSample(
                ca=ca_right_tplus1,
                ca_adv=ca_adv_right,
                ca_rec=ca_rec_right,
                cll=cll_right_tplus1,
                phi=final_params.phi_right,
                d_rho=final_params.d_rho_right,
                phi_active=phi_active_right,
                loss=right_objective(final_params),
            ),
            phi_neutral=_PHI_NEUTRAL,
        )

    return wetting._replace(
        phi_left=final_params.phi_left,
        phi_right=final_params.phi_right,
        d_rho_left=final_params.d_rho_left,
        d_rho_right=final_params.d_rho_right,
        ca_left=ca_left_tplus1,
        ca_right=ca_right_tplus1,
        cll_left=jnp.where(in_window_left, wetting.cll_left, cll_left_tplus1),
        cll_right=jnp.where(in_window_right, wetting.cll_right, cll_right_tplus1),
    )


@wetting_operator(name="hysteresis")
def update_wetting_state(
    wetting: WettingState,
    rho_t_plus1: jnp.ndarray,
    setup: SimulationSetup,
    *,
    trial_step_fn: Callable[[WettingParams], tuple[jnp.ndarray, jnp.ndarray]],
) -> WettingState:
    """Pure JAX update of wetting / hysteresis parameters.

    This replaces the mutable
    :class:`~update_timestep.UpdateMultiphaseHysteresis.__call__`
    method.  It operates entirely on the :class:`WettingState`
    NamedTuple and returns a new instance — no side-effects.

    Each side (left and right) is optimised **independently** in its
    own ``jax.lax.while_loop``, masking out the other side's
    parameters.  This gives each side clean gradients and clean Adam
    state at the cost of two trial-step evaluations per outer
    iteration instead of one.

    Hysteresis regime is selected strictly from the current contact angle:
    pinned when ``ca_receding <= ca <= ca_advancing``; otherwise CA-targeting
    toward the exceeded bound.

    Args:
        wetting: Current :class:`WettingState`.
        rho_t_plus1: Density field, shape ``(nx, ny, nz, 1, 1)``.
        setup: :class:`~setup.simulation_setup.SimulationSetup`
            (closed-over, not traced).
        f_t: Pre-step populations, shape ``(nx, ny, nz, q, 1)``.
        force_tot: Total force from the current step (optional).
        trial_step_fn: Callable ``(WettingParams) → (f_out, rho_out)``
            that evaluates a single multiphase physics pass with trial
            wetting parameters. Required; no default is provided.

    Returns:
        Updated :class:`WettingState`.
    """
    if setup.config.hysteresis_config is None:
        msg = "hysteresis_config is required for wetting state update"
        raise TypeError(msg)
    hc = setup.config.hysteresis_config
    ca_adv = hc["ca_advancing"]
    ca_rec = hc["ca_receding"]
    return _update_wetting_state_impl(
        wetting,
        rho_t_plus1,
        setup,
        trial_step_fn,
        ca_adv_left=ca_adv,
        ca_rec_left=ca_rec,
        ca_adv_right=ca_adv,
        ca_rec_right=ca_rec,
    )


@wetting_operator(name="chemical_step_hysteresis")
def update_wetting_state_chemical_step(
    wetting: WettingState,
    rho_t_plus1: jnp.ndarray,
    setup: SimulationSetup,
    *,
    trial_step_fn: Callable[[WettingParams], tuple[jnp.ndarray, jnp.ndarray]],
) -> WettingState:
    """Hysteresis update where CA targets are determined per-side by chemical step position.

    Identical to update_wetting_state except (ca_advancing, ca_receding) for each
    side is selected by comparing that side's contact-line location against the
    chemical step position, rather than using a single global hysteresis window.

    Args:
        wetting: Current WettingState. wetting.cll_left and wetting.cll_right
                 are used to determine which region each contact line occupies.
        rho_t_plus1: Post-step density field.
        setup: SimulationSetup. Must carry chemical_step_config with keys:
               chemical_step_location, ca_advancing_pre_step, ca_receding_pre_step,
               ca_advancing_post_step, ca_receding_post_step.
        trial_step_fn: Callable (WettingParams) -> (f_out, rho_out).
                       Provided by the step function via partial application.

    Returns:
        Updated WettingState with optimised wetting parameters and measured CA/CLL.
    """
    if setup.multiphase_params is None:
        msg = "multiphase_params is required for chemical step hysteresis"
        raise TypeError(msg)
    mp = setup.multiphase_params
    rho_mean = 0.5 * (mp.rho_l + mp.rho_v)
    if setup.wetting_edge is None:
        msg = "wetting_edge is required for chemical step hysteresis"
        raise TypeError(msg)
    edge = setup.wetting_edge

    # 1. Measure current contact angles and contact-line locations
    ca_left_tplus1, ca_right_tplus1 = compute_contact_angle(rho_t_plus1, jnp.array(rho_mean), edge=edge)
    cll_left_tplus1, cll_right_tplus1 = compute_contact_line_location(
        rho_t_plus1,
        ca_left_tplus1,
        ca_right_tplus1,
        jnp.array(rho_mean),
        edge=edge,
    )

    # Use current measured CLL to select the active pre/post hysteresis window.
    ca_adv_left, ca_rec_left = _get_hysteresis_window_chemical_step(setup, cll_left_tplus1)
    ca_adv_right, ca_rec_right = _get_hysteresis_window_chemical_step(setup, cll_right_tplus1)

    return _update_wetting_state_impl(
        wetting,
        rho_t_plus1,
        setup,
        trial_step_fn,
        ca_adv_left=ca_adv_left,
        ca_rec_left=ca_rec_left,
        ca_adv_right=ca_adv_right,
        ca_rec_right=ca_rec_right,
    )
