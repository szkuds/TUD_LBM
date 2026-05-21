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
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
from tud_lbm.config.config_overview import DEBUG_FLAG
from tud_lbm.operators.wetting._contact_angle import compute_contact_angle
from tud_lbm.operators.wetting._contact_line import compute_contact_line_location
from tud_lbm.operators.wetting._params import WettingParams
from tud_lbm.registry import wetting_operator

if TYPE_CHECKING:
    import types
    from collections.abc import Callable
    from tud_lbm.pipeline.setup import SimulationSetup
    from tud_lbm.pipeline.state.state import WettingState


# ── Helpers ──────────────────────────────────────────────────────────

# Neutral values — the inactive parameter is snapped to these when the
# directional split is applied.
_PHI_NEUTRAL: float = 1.0
_D_RHO_NEUTRAL: float = 0.0


def _import_optax() -> types.ModuleType:
    """Import optional ``optax`` dependency with a clear install hint."""
    try:
        import optax
    except ImportError as err:
        msg = "The 'optax' package is required for hysteresis wetting.\nInstall it with:  pip install optax"
        raise ImportError(msg) from err
    return optax


def _phi_is_active(
    in_window: jnp.ndarray,
    above_window: jnp.ndarray,
    forward_drift: jnp.ndarray,
) -> jnp.ndarray:
    """Return True if phi is the active parameter for this side.

    phi is active (wall needs to become *more* wetting, i.e. lower CA) when:
      - ``above_window`` — CA has exceeded ca_advancing and must be pulled down.
      - ``in_window & ~forward_drift`` — CL is drifting backward (receding);
        resist by increasing wettability.

    d_rho is active (wall needs to become *less* wetting, i.e. raise CA) in
    all other cases:
      - ``in_window & forward_drift`` — CL is advancing; resist by reducing
        wettability.
      - ``~above_window & ~in_window`` (below window) — CA has dropped below
        ca_receding; must be raised.

    Args:
        in_window: bool scalar — CA is between ca_receding and ca_advancing.
        above_window: bool scalar — CA > ca_advancing.
        forward_drift: bool scalar — the trial-step CLL has moved in the
            advancing direction relative to the stored CLL.  Convention:
            right side → forward means ``cll_trial > cll_stored``;
            left side → forward means ``cll_trial < cll_stored``.

    Returns:
        Boolean JAX scalar; True means phi is active, False means d_rho is active.
    """
    return above_window | (in_window & ~forward_drift)


def _clamp_params(params: WettingParams) -> WettingParams:
    """Clamp wetting parameters to physically reasonable ranges.

    Note: ``jnp.clip`` has zero gradient at the boundaries, so a
    parameter sitting at a clamp limit receives no further gradient
    signal in that direction.  The ranges below cover physically
    realistic wetting conditions.
    """
    return WettingParams(
        phi_left=jnp.clip(params.phi_left, 1.0, 1.5),
        phi_right=jnp.clip(params.phi_right, 1.0, 1.5),
        d_rho_left=jnp.clip(params.d_rho_left, 0.0, 0.3),
        d_rho_right=jnp.clip(params.d_rho_right, 0.0, 0.3),
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
    optimiser: object,
    max_iterations: int,
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
        max_iterations: Maximum number of inner steps.

    Returns:
        ``(final_params, final_loss)``.
    """
    import optax  # lazy import — optional dependency

    opt_state = optimiser.init(initial_params)
    initial_loss = objective_fn(initial_params)

    def cond_fn(carry: tuple) -> jnp.ndarray:
        _params, _opt_state, _loss, iteration = carry
        return iteration < max_iterations

    def body_fn(carry: tuple) -> tuple:
        params, opt_state, _loss, iteration = carry
        loss, grads = jax.value_and_grad(objective_fn)(params)
        grads = grad_mask_fn(grads)
        updates, new_opt_state = optimiser.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_params = _clamp_params(new_params)
        return (new_params, new_opt_state, loss, iteration + 1)

    init_carry = (initial_params, opt_state, initial_loss, jnp.array(0))
    final_params, _opt_state, final_loss, _iters = jax.lax.while_loop(
        cond_fn,
        body_fn,
        init_carry,
    )
    return final_params, final_loss


# ── Top-level entry point ────────────────────────────────────────────


def _get_hysteresis_window_chemical_step(setup: SimulationSetup, cll: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (ca_advancing, ca_receding) based on CLL position relative to chemical step."""
    step_x = setup.config.chemical_step_config["chemical_step_location"] * setup.config.grid_shape[0]
    return jax.lax.cond(
        cll < step_x,
        lambda: (
            setup.config.chemical_step_config["ca_advancing_pre_step"],
            setup.config.chemical_step_config["ca_receding_pre_step"],
        ),
        lambda: (
            setup.config.chemical_step_config["ca_advancing_post_step"],
            setup.config.chemical_step_config["ca_receding_post_step"],
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
    mp = setup.multiphase_params
    rho_mean = 0.5 * (mp.rho_l + mp.rho_v)

    ca_left_tplus1, ca_right_tplus1 = compute_contact_angle(rho_t_plus1, jnp.array(rho_mean))
    cll_left_tplus1, cll_right_tplus1 = compute_contact_line_location(
        rho_t_plus1,
        ca_left_tplus1,
        ca_right_tplus1,
        jnp.array(rho_mean),
    )

    forward_drift_right = cll_right_tplus1 > wetting.cll_right
    forward_drift_left = cll_left_tplus1 < wetting.cll_left

    in_window_left = (ca_left_tplus1 >= ca_rec_left) & (ca_left_tplus1 <= ca_adv_left)
    in_window_right = (ca_right_tplus1 >= ca_rec_right) & (ca_right_tplus1 <= ca_adv_right)
    above_window_left = ca_left_tplus1 > ca_adv_left
    above_window_right = ca_right_tplus1 > ca_adv_right

    phi_active_right = _phi_is_active(in_window_right, above_window_right, forward_drift_right)
    phi_active_left = _phi_is_active(in_window_left, above_window_left, forward_drift_left)

    hc = setup.config.hysteresis_config
    lr_default = hc.get("learning_rate", 0.01)
    lr = jnp.where(above_window_right | above_window_left, hc.get("learning_rate_above", 0.05), lr_default)
    max_iter = hc.get("max_iterations_above", 50) if hc.get("max_iterations_above") else hc.get("max_iterations", 50)

    params = WettingParams(
        phi_left=jnp.where(phi_active_left, wetting.phi_left, _PHI_NEUTRAL),
        phi_right=jnp.where(phi_active_right, wetting.phi_right, _PHI_NEUTRAL),
        d_rho_left=jnp.where(phi_active_left, _D_RHO_NEUTRAL, wetting.d_rho_left),
        d_rho_right=jnp.where(phi_active_right, _D_RHO_NEUTRAL, wetting.d_rho_right),
    )

    optax = _import_optax()
    optimiser = optax.adam(lr)

    def evaluate_fn(params: WettingParams) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        _, rho_out = trial_step_fn(params)
        ca_l, ca_r = compute_contact_angle(rho_out, jnp.array(rho_mean))
        cll_l, cll_r = compute_contact_line_location(rho_out, ca_l, ca_r, jnp.array(rho_mean))
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

    def _opt_left(p: WettingParams) -> WettingParams:
        return jax.lax.cond(
            phi_active_left,
            lambda pp: _optimise_single_param(left_objective, pp, _mask_left_phi, optimiser, max_iter)[0],
            lambda pp: _optimise_single_param(left_objective, pp, _mask_left_d_rho, optimiser, max_iter)[0],
            p,
        )

    def _opt_right(p: WettingParams) -> WettingParams:
        return jax.lax.cond(
            phi_active_right,
            lambda pp: _optimise_single_param(right_objective, pp, _mask_right_phi, optimiser, max_iter)[0],
            lambda pp: _optimise_single_param(right_objective, pp, _mask_right_d_rho, optimiser, max_iter)[0],
            p,
        )

    new_params = _opt_right(_opt_left(params))

    # Fallback: if phi path is selected but stays neutral, retry with d_rho
    # warm-started from the stored accumulated value.
    def _fallback_d_rho_left(p: WettingParams) -> WettingParams:
        fallback = WettingParams(
            phi_left=_PHI_NEUTRAL,
            phi_right=p.phi_right,
            d_rho_left=wetting.d_rho_left,
            d_rho_right=p.d_rho_right,
        )
        return _optimise_single_param(left_objective, fallback, _mask_left_d_rho, optimiser, max_iter)[0]

    def _fallback_d_rho_right(p: WettingParams) -> WettingParams:
        fallback = WettingParams(
            phi_left=p.phi_left,
            phi_right=_PHI_NEUTRAL,
            d_rho_left=p.d_rho_left,
            d_rho_right=wetting.d_rho_right,
        )
        return _optimise_single_param(right_objective, fallback, _mask_right_d_rho, optimiser, max_iter)[0]

    final_params = jax.lax.cond(
        phi_active_left & (new_params.phi_left <= _PHI_NEUTRAL),
        _fallback_d_rho_left,
        lambda p: p,
        new_params,
    )
    final_params = jax.lax.cond(
        phi_active_right & (new_params.phi_right <= _PHI_NEUTRAL),
        _fallback_d_rho_right,
        lambda p: p,
        final_params,
    )

    if DEBUG_FLAG:
        phi_engaged_right = phi_active_right & (final_params.phi_right > _PHI_NEUTRAL)
        fallback_right = phi_active_right & ~phi_engaged_right
        phi_engaged_left = phi_active_left & (final_params.phi_left > _PHI_NEUTRAL)
        fallback_left = phi_active_left & ~phi_engaged_left

        # mode: 0=d_rho(normal), 1=phi(engaged), 2=d_rho(fallback)
        mode_right = jnp.where(phi_engaged_right, jnp.array(1), jnp.where(fallback_right, jnp.array(2), jnp.array(0)))
        mode_left = jnp.where(phi_engaged_left, jnp.array(1), jnp.where(fallback_left, jnp.array(2), jnp.array(0)))

        jax.debug.print(
            "\n[R] CA={ca:.3f}° (adv={ca_adv:.1f}° rec={ca_rec:.1f}°) | CLL={cll:.3f} | "
            "mode={mode}(0=d_rho,1=phi,2=fb) | "
            "phi: {phi:.6f} | "
            "d_rho: {d_rho:.6f} | "
            "loss={loss:.3e}",
            ca=ca_right_tplus1,
            ca_adv=ca_adv_right,
            ca_rec=ca_rec_right,
            cll=cll_right_tplus1,
            mode=mode_right,
            phi=final_params.phi_right,
            d_rho=final_params.d_rho_right,
            loss=right_objective(final_params),
        )
        jax.debug.print(
            "[L] CA={ca:.3f}° (adv={ca_adv:.1f}° rec={ca_rec:.1f}°) | CLL={cll:.3f} | "
            "mode={mode}(0=d_rho,1=phi,2=fb) | "
            "phi: {phi:.6f} | "
            "d_rho: {d_rho:.6f} | "
            "loss={loss:.3e}",
            ca=ca_left_tplus1,
            ca_adv=ca_adv_left,
            ca_rec=ca_rec_left,
            cll=cll_left_tplus1,
            mode=mode_left,
            phi=final_params.phi_left,
            d_rho=final_params.d_rho_left,
            loss=left_objective(final_params),
        )

    return wetting._replace(
        phi_left=final_params.phi_left,
        phi_right=final_params.phi_right,
        d_rho_left=final_params.d_rho_left,
        d_rho_right=final_params.d_rho_right,
        ca_left=ca_left_tplus1,
        ca_right=ca_right_tplus1,
        cll_left=cll_left_tplus1,
        cll_right=cll_right_tplus1,
    )


@wetting_operator(name="hysteresis")
def update_wetting_state(
    wetting: WettingState,
    rho_t_plus1: jnp.ndarray,
    setup: SimulationSetup,
    *,
    trial_step_fn: Callable[[WettingParams], tuple[jnp.ndarray, jnp.ndarray]] | None = None,
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
    mp = setup.multiphase_params
    rho_mean = 0.5 * (mp.rho_l + mp.rho_v)

    # 1. Measure current contact angles and contact-line locations
    ca_left_tplus1, ca_right_tplus1 = compute_contact_angle(rho_t_plus1, jnp.array(rho_mean))
    cll_left_tplus1, cll_right_tplus1 = compute_contact_line_location(
        rho_t_plus1,
        ca_left_tplus1,
        ca_right_tplus1,
        jnp.array(rho_mean),
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
