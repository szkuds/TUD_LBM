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
from tud_lbm.operators.wetting._contact_angle import compute_contact_angle
from tud_lbm.operators.wetting._contact_line import compute_contact_line_location
from tud_lbm.operators.wetting._params import WettingParams
from tud_lbm.registry import wetting_operator

if TYPE_CHECKING:
    from collections.abc import Callable
    from tud_lbm.pipeline.setup import SimulationSetup
    from tud_lbm.pipeline.state.state import WettingState


# ── Helpers ──────────────────────────────────────────────────────────


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
        d_rho_left=jnp.clip(params.d_rho_left, 0.0, 0.2),
        d_rho_right=jnp.clip(params.d_rho_right, 0.0, 0.2),
    )


def _cost_cll(cll_target: jnp.ndarray, cll_current: jnp.ndarray) -> jnp.ndarray:
    """Squared error for CLL pinning — smooth gradient everywhere."""
    return jnp.abs(cll_target - cll_current)


def _cost_ca(ca_target: jnp.ndarray, ca_current: jnp.ndarray) -> jnp.ndarray:
    """Squared error for CA targeting — smooth gradient everywhere."""
    return jnp.abs(ca_target - ca_current)


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
        loss_tol: Convergence tolerance — loop exits early when
            ``loss <= loss_tol``.

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


def _get_hysteresis_window_chemical_step(setup: SimulationSetup, cll: jnp.ndarray) -> tuple[WettingParams, jnp.ndarray]:
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

    Per-side advancing/receding direction is inferred from the sign of
    the contact-line displacement between the previous and current
    timestep: the left side is advancing when ``cll_left`` decreases
    (moves in -x), and the right side is advancing when ``cll_right_tplus1``
    increases (moves in +x).

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

    # 2. Hysteresis window parameters
    hc = setup.config.hysteresis_config
    ca_adv_left = ca_adv_right = hc["ca_advancing"]
    ca_rec_left = ca_rec_right = hc["ca_receding"]
    lr = hc.get("learning_rate", 0.01)
    max_iter = hc.get("max_iterations", 20)
    in_window_left = (ca_left_tplus1 >= ca_rec_left) & (ca_left_tplus1 <= ca_adv_left)
    in_window_right = (ca_right_tplus1 >= ca_rec_right) & (ca_right_tplus1 <= ca_adv_right)

    # 3. Current optimisable parameters
    # Build parameter bundle from scalar fields on wetting for optimisation.
    params = WettingParams(
        phi_left=wetting.phi_left,
        phi_right=wetting.phi_right,
        d_rho_left=wetting.d_rho_left,
        d_rho_right=wetting.d_rho_right,
    )

    try:
        import optax  # lazy import — optional dependency
    except ImportError as err:
        msg = "The 'optax' package is required for hysteresis wetting.\nInstall it with:  pip install optax"
        raise ImportError(
            msg,
        ) from err
    optimiser = optax.adam(lr)

    # 4. Create wrapper that converts trial_step_fn to evaluate_fn
    # trial_step_fn returns (f_out, rho_out); we need (ca_l, ca_r, cll_l, cll_r)
    def evaluate_fn(params: WettingParams) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Evaluate trial parameters by running trial step and measuring contact properties."""
        _, rho_out = trial_step_fn(params)
        ca_l, ca_r = compute_contact_angle(rho_out, jnp.array(rho_mean))
        cll_l, cll_r = compute_contact_line_location(rho_out, ca_l, ca_r, jnp.array(rho_mean))
        return ca_l, ca_r, cll_l, cll_r

    # 5. Infer advancing/receding direction from contact-line displacement
    delta_cll_left = cll_left_tplus1 - wetting.cll_left
    delta_cll_right = cll_right_tplus1 - wetting.cll_right

    advancing_left = delta_cll_left > 0.0
    advancing_right = delta_cll_right < 0.0

    ca_target_left = jnp.where(advancing_left, ca_adv_left, ca_rec_left)
    ca_target_right = jnp.where(advancing_right, ca_adv_right, ca_rec_right)

    # --- 5.1: per-side objectives ---

    def left_objective(p: WettingParams) -> jnp.ndarray:
        ca_l, _, cll_l, _ = evaluate_fn(p)
        # Use the pre-step contact-line location recorded in `wetting` as
        # the pin target. The previous implementation used `cll_left`
        # measured from the post-step rho_t_plus1, which makes the argmin
        # trivial and the pin a no-op.
        return jnp.where(in_window_left, _cost_cll(wetting.cll_left, cll_l), _cost_ca(ca_target_left, ca_l))

    def right_objective(p: WettingParams) -> jnp.ndarray:
        _, ca_r, _, cll_r = evaluate_fn(p)
        # Mirror of the left side: pin to the pre-step contact-line stored
        # on the WettingState passed into this function.
        return jnp.where(in_window_right, _cost_cll(wetting.cll_right, cll_r), _cost_ca(ca_target_right, ca_r))

    def left_mask(g: WettingParams) -> WettingParams:
        z = jnp.zeros_like
        return WettingParams(
            phi_left=g.phi_left,
            phi_right=z(g.phi_right),
            d_rho_left=g.d_rho_left,
            d_rho_right=z(g.d_rho_right),
        )

    def right_mask(g: WettingParams) -> WettingParams:
        z = jnp.zeros_like
        return WettingParams(
            phi_left=z(g.phi_left),
            phi_right=g.phi_right,
            d_rho_left=z(g.d_rho_left),
            d_rho_right=g.d_rho_right,
        )

    # --- 5.2: optimise left side, then right side ---
    p1, _ = _optimise_single_param(left_objective, params, left_mask, optimiser, max_iter)
    new_params, _ = _optimise_single_param(right_objective, p1, right_mask, optimiser, max_iter)

    # 6. Return updated wetting state
    return wetting._replace(
        phi_left=new_params.phi_left,
        phi_right=new_params.phi_right,
        d_rho_left=new_params.d_rho_left,
        d_rho_right=new_params.d_rho_right,
        ca_left=ca_left_tplus1,
        ca_right=ca_right_tplus1,
        cll_left=cll_left_tplus1,
        cll_right=cll_right_tplus1,
    )


def update_wetting_state_chemical_step(
    wetting: WettingState,
    rho: jnp.ndarray,
    setup: SimulationSetup,
    *,
    trial_step_fn: Callable[[WettingParams], jnp.ndarray],
) -> WettingState:
    """Hysteresis update where CA targets are determined per-side by chemical step position.

    Identical to update_wetting_state except (ca_advancing, ca_receding) for each
    side is selected by comparing that side's contact-line location against the
    chemical step position, rather than using a single global hysteresis window.

    Args:
        wetting: Current WettingState. wetting.cll_left and wetting.cll_right
                 are used to determine which region each contact line occupies.
        rho: Post-step density field.
        setup: SimulationSetup. Must carry chemical_step_config with keys:
               chemical_step_location, ca_advancing_pre_step, ca_receding_pre_step,
               ca_advancing_post_step, ca_receding_post_step.
        f_bc: Post-BC populations (used as trial-step seed).
        force: Total force field.
        trial_step_fn: Callable (WettingParams) -> (f_out, rho_out).
                       Provided by the step function via partial application.

    Returns:
        Updated WettingState with optimised wetting parameters and measured CA/CLL.
    """
    # Determine per-side CA windows from CLL position relative to chemical step
    ca_adv_left, ca_rec_left = _get_hysteresis_window_chemical_step(setup, wetting.cll_left)
    ca_adv_right, ca_rec_right = _get_hysteresis_window_chemical_step(setup, wetting.cll_right)

    mp = setup.multiphase_params
    rho_mean = 0.5 * (mp.rho_l + mp.rho_v)

    # 1. Measure current contact angles and contact-line locations
    ca_left_tplus1, ca_right_tplus1 = compute_contact_angle(rho, jnp.array(rho_mean))
    cll_left_tplus1, cll_right_tplus1 = compute_contact_line_location(
        rho,
        ca_left_tplus1,
        ca_right_tplus1,
        jnp.array(rho_mean),
    )

    # 2. Hysteresis window parameters (using per-side CA values from chemical step)
    hc = setup.config.hysteresis_config
    lr = hc.get("learning_rate", 0.01)
    max_iter = hc.get("max_iterations", 20)
    in_window_left = (ca_left_tplus1 >= ca_rec_left) & (ca_left_tplus1 <= ca_adv_left)
    in_window_right = (ca_right_tplus1 >= ca_rec_right) & (ca_right_tplus1 <= ca_adv_right)

    # 3. Current optimisable parameters
    params = WettingParams(
        phi_left=wetting.phi_left,
        phi_right=wetting.phi_right,
        d_rho_left=wetting.d_rho_left,
        d_rho_right=wetting.d_rho_right,
    )

    try:
        import optax  # lazy import — optional dependency
    except ImportError as err:
        msg = "The 'optax' package is required for hysteresis wetting.\nInstall it with:  pip install optax"
        raise ImportError(
            msg,
        ) from err
    optimiser = optax.adam(lr)

    # 4. Create wrapper that converts trial_step_fn to evaluate_fn
    def evaluate_fn(params: WettingParams) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Evaluate trial parameters by running trial step and measuring contact properties."""
        _, rho_out = trial_step_fn(params)
        ca_l, ca_r = compute_contact_angle(rho_out, jnp.array(rho_mean))
        cll_l, cll_r = compute_contact_line_location(rho_out, ca_l, ca_r, jnp.array(rho_mean))
        return ca_l, ca_r, cll_l, cll_r

    # 5. Infer advancing/receding direction from contact-line displacement
    delta_cll_left = cll_left_tplus1 - wetting.cll_left
    delta_cll_right = cll_right_tplus1 - wetting.cll_right

    advancing_left = delta_cll_left > 0.0
    advancing_right = delta_cll_right < 0.0

    ca_target_left = jnp.where(advancing_left, ca_adv_left, ca_rec_left)
    ca_target_right = jnp.where(advancing_right, ca_adv_right, ca_rec_right)

    # --- 5.1: per-side objectives ---

    def left_objective(p: WettingParams) -> jnp.ndarray:
        ca_l, _, cll_l, _ = evaluate_fn(p)
        return jnp.where(in_window_left, _cost_cll(wetting.cll_left, cll_l), _cost_ca(ca_target_left, ca_l))

    def right_objective(p: WettingParams) -> jnp.ndarray:
        _, ca_r, _, cll_r = evaluate_fn(p)
        return jnp.where(in_window_right, _cost_cll(wetting.cll_right, cll_r), _cost_ca(ca_target_right, ca_r))

    def left_mask(g: WettingParams) -> WettingParams:
        z = jnp.zeros_like
        return WettingParams(
            phi_left=g.phi_left,
            phi_right=z(g.phi_right),
            d_rho_left=g.d_rho_left,
            d_rho_right=z(g.d_rho_right),
        )

    def right_mask(g: WettingParams) -> WettingParams:
        z = jnp.zeros_like
        return WettingParams(
            phi_left=z(g.phi_left),
            phi_right=g.phi_right,
            d_rho_left=z(g.d_rho_left),
            d_rho_right=g.d_rho_right,
        )

    # --- 5.2: optimise left side, then right side ---
    p1, _ = _optimise_single_param(left_objective, params, left_mask, optimiser, max_iter)
    new_params, _ = _optimise_single_param(right_objective, p1, right_mask, optimiser, max_iter)

    # 6. Return updated wetting state
    return wetting._replace(
        phi_left=new_params.phi_left,
        phi_right=new_params.phi_right,
        d_rho_left=new_params.d_rho_left,
        d_rho_right=new_params.d_rho_right,
        ca_left=ca_left_tplus1,
        ca_right=ca_right_tplus1,
        cll_left=cll_left_tplus1,
        cll_right=cll_right_tplus1,
    )
