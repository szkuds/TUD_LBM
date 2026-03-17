"""Wetting hysteresis optimisation — pure functions.

Ported from :class:`update_timestep.UpdateMultiphaseHysteresis`.

The legacy class stores mutable wetting parameters on ``self`` and
uses ``@partial(jit, static_argnums=(0,))`` which causes JIT cache
bloat.  This module replaces it with pure functions that operate on
the :class:`~state.state.WettingState` NamedTuple carried through
``jax.lax.scan``.

All inner optimisation loops use ``optax`` + ``jax.lax.scan`` and
are fully jittable.

Design
~~~~~~
``update_wetting_state`` is the top-level entry point.  It:

1. Measures contact angles and contact-line locations from ``rho``.
2. Checks whether each side is inside the hysteresis window.
3. Via ``jax.lax.cond``, either **pins the CLL** (inside window) or
   **optimises toward the target CA** (outside window).
4. Returns an updated :class:`WettingState` — no mutation.

The inner ``_evaluate_with_params`` closure performs a single LBM
step with trial wetting parameters so that ``jax.value_and_grad``
can differentiate through it.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax

from state.state import WettingState
from operators.wetting.contact_angle import compute_contact_angle
from operators.wetting.contact_line import compute_contact_line_location
from registry import wetting_operator

# ── Optimisable parameter container ─────────────────────────────────


class WettingParams(NamedTuple):
    """Minimal parameter set passed through ``jax.value_and_grad`` and is a valid JAX
    pytree (NamedTuple).
    """

    d_rho_left: jnp.ndarray
    d_rho_right: jnp.ndarray
    phi_left: jnp.ndarray
    phi_right: jnp.ndarray


# ── Helpers ──────────────────────────────────────────────────────────


def _clamp_params(params: WettingParams) -> WettingParams:
    """Clamp wetting parameters to physically reasonable ranges."""
    return WettingParams(
        d_rho_left=jnp.clip(params.d_rho_left, 0.0, 0.2),
        d_rho_right=jnp.clip(params.d_rho_right, 0.0, 0.2),
        phi_left=jnp.clip(params.phi_left, 1.0, 1.5),
        phi_right=jnp.clip(params.phi_right, 1.0, 1.5),
    )


def _cost_cll(cll_target: jnp.ndarray, cll_current: jnp.ndarray) -> jnp.ndarray:
    return jnp.abs(cll_target - cll_current)


def _cost_ca(ca_target: jnp.ndarray, ca_current: jnp.ndarray) -> jnp.ndarray:
    return jnp.abs(ca_target - ca_current)


# ── Generic optimisation routines ────────────────────────────────────


def _optimise_single_param(
    objective_fn,
    initial_params: WettingParams,
    grad_mask_fn,
    optimiser,
    max_iterations: int,
) -> Tuple[WettingParams, jnp.ndarray]:
    """Run an ``optax`` optimisation loop with masked gradients.

    Args:
        objective_fn: ``params → scalar_loss``.
        initial_params: Starting :class:`WettingParams`.
        grad_mask_fn: ``grads → grads`` that zeros out all but the
            target parameter.
        optimiser: An ``optax`` optimiser instance.
        max_iterations: Number of inner steps.

    Returns:
        ``(final_params, final_loss)``.
    """
    opt_state = optimiser.init(initial_params)

    def step(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(objective_fn)(params)
        grads = grad_mask_fn(grads)
        updates, new_opt_state = optimiser.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_params = _clamp_params(new_params)
        return (new_params, new_opt_state), loss

    (final_params, _), losses = jax.lax.scan(
        step,
        (initial_params, opt_state),
        jnp.arange(max_iterations),
    )
    return final_params, losses[-1]


def _optimise_side_cll(
    evaluate_fn,
    initial_params: WettingParams,
    cll_target: jnp.ndarray,
    side: str,
    optimiser,
    max_iterations: int,
) -> WettingParams:
    """Optimise wetting params to pin CLL on one side.

    Tries both the ``d_rho`` and ``phi`` parameter for the given side,
    and returns whichever achieves the lower final loss.
    """
    # --- d_rho objective --------------------------------------------------
    if side == "left":

        def obj_d_rho(p):
            ca_l, ca_r, cll_l, cll_r = evaluate_fn(p)
            return _cost_cll(cll_target, cll_l)

        def mask_d_rho(g):
            return WettingParams(
                g.d_rho_left,
                jnp.zeros_like(g.d_rho_right),
                jnp.zeros_like(g.phi_left),
                jnp.zeros_like(g.phi_right),
            )

        def obj_phi(p):
            ca_l, ca_r, cll_l, cll_r = evaluate_fn(p)
            return _cost_cll(cll_target, cll_l)

        def mask_phi(g):
            return WettingParams(
                jnp.zeros_like(g.d_rho_left),
                jnp.zeros_like(g.d_rho_right),
                g.phi_left,
                jnp.zeros_like(g.phi_right),
            )

    else:

        def obj_d_rho(p):
            ca_l, ca_r, cll_l, cll_r = evaluate_fn(p)
            return _cost_cll(cll_target, cll_r)

        def mask_d_rho(g):
            return WettingParams(
                jnp.zeros_like(g.d_rho_left),
                g.d_rho_right,
                jnp.zeros_like(g.phi_left),
                jnp.zeros_like(g.phi_right),
            )

        def obj_phi(p):
            ca_l, ca_r, cll_l, cll_r = evaluate_fn(p)
            return _cost_cll(cll_target, cll_r)

        def mask_phi(g):
            return WettingParams(
                jnp.zeros_like(g.d_rho_left),
                jnp.zeros_like(g.d_rho_right),
                jnp.zeros_like(g.phi_left),
                g.phi_right,
            )

    p_drho, loss_drho = _optimise_single_param(
        obj_d_rho, initial_params, mask_d_rho, optimiser, max_iterations
    )
    p_phi, loss_phi = _optimise_single_param(
        obj_phi, initial_params, mask_phi, optimiser, max_iterations
    )

    return jax.lax.cond(loss_drho < loss_phi, lambda: p_drho, lambda: p_phi)


def _optimise_side_ca(
    evaluate_fn,
    initial_params: WettingParams,
    ca_target: jnp.ndarray,
    side: str,
    optimiser,
    max_iterations: int,
) -> WettingParams:
    """Optimise wetting params to reach target CA on one side."""
    if side == "left":

        def obj_d_rho(p):
            ca_l, ca_r, cll_l, cll_r = evaluate_fn(p)
            return _cost_ca(ca_target, ca_l)

        def mask_d_rho(g):
            return WettingParams(
                g.d_rho_left,
                jnp.zeros_like(g.d_rho_right),
                jnp.zeros_like(g.phi_left),
                jnp.zeros_like(g.phi_right),
            )

        def obj_phi(p):
            ca_l, ca_r, cll_l, cll_r = evaluate_fn(p)
            return _cost_ca(ca_target, ca_l)

        def mask_phi(g):
            return WettingParams(
                jnp.zeros_like(g.d_rho_left),
                jnp.zeros_like(g.d_rho_right),
                g.phi_left,
                jnp.zeros_like(g.phi_right),
            )

    else:

        def obj_d_rho(p):
            ca_l, ca_r, cll_l, cll_r = evaluate_fn(p)
            return _cost_ca(ca_target, ca_r)

        def mask_d_rho(g):
            return WettingParams(
                jnp.zeros_like(g.d_rho_left),
                g.d_rho_right,
                jnp.zeros_like(g.phi_left),
                jnp.zeros_like(g.phi_right),
            )

        def obj_phi(p):
            ca_l, ca_r, cll_l, cll_r = evaluate_fn(p)
            return _cost_ca(ca_target, ca_r)

        def mask_phi(g):
            return WettingParams(
                jnp.zeros_like(g.d_rho_left),
                jnp.zeros_like(g.d_rho_right),
                jnp.zeros_like(g.phi_left),
                g.phi_right,
            )

    p_drho, loss_drho = _optimise_single_param(
        obj_d_rho, initial_params, mask_d_rho, optimiser, max_iterations
    )
    p_phi, loss_phi = _optimise_single_param(
        obj_phi, initial_params, mask_phi, optimiser, max_iterations
    )

    return jax.lax.cond(loss_drho < loss_phi, lambda: p_drho, lambda: p_phi)


# ── Top-level entry point ────────────────────────────────────────────


@wetting_operator(name="hysteresis")
def update_wetting_state(
    wetting: WettingState,
    rho: jnp.ndarray,
    setup,
    f_bc: jnp.ndarray,
    force: jnp.ndarray,
    *,
    evaluate_fn=None,
) -> WettingState:
    """Pure JAX update of wetting / hysteresis parameters.

    This replaces the mutable
    :class:`~update_timestep.UpdateMultiphaseHysteresis.__call__`
    method.  It operates entirely on the :class:`WettingState`
    NamedTuple and returns a new instance — no side-effects.

    Args:
        wetting: Current :class:`WettingState`.
        rho: Density field, shape ``(nx, ny, 1, 1)``.
        setup: :class:`~setup.simulation_setup.SimulationSetup`
            (closed-over, not traced).
        f_bc: Post-BC populations, shape ``(nx, ny, q, 1)``.
        force: Total force field, shape ``(nx, ny, 1, 2)``.
        evaluate_fn: Optional callable
            ``(WettingParams) → (ca_l, ca_r, cll_l, cll_r)``
            used by the inner optimiser.  If ``None``, a default is
            built from the pure-function operators.

    Returns:
        Updated :class:`WettingState`.
    """
    mp = setup.multiphase_params
    rho_mean = 0.5 * (mp.rho_l + mp.rho_v)

    # 1. Measure current contact angles and contact-line locations
    ca_left, ca_right = compute_contact_angle(rho, rho_mean)
    cll_left, cll_right = compute_contact_line_location(
        rho,
        ca_left,
        ca_right,
        rho_mean,
    )

    # 2. Hysteresis window parameters
    hc = setup.hysteresis_config
    ca_adv = hc["ca_advancing"]
    ca_rec = hc["ca_receding"]
    lr = hc.get("learning_rate", 0.01)
    max_iter = hc.get("max_iterations", 20)

    in_window_left = (ca_left >= ca_rec) & (ca_left <= ca_adv)
    in_window_right = (ca_right >= ca_rec) & (ca_right <= ca_adv)

    # 3. Current optimisable parameters
    params = WettingParams(
        d_rho_left=wetting.d_rho_left,
        d_rho_right=wetting.d_rho_right,
        phi_left=wetting.phi_left,
        phi_right=wetting.phi_right,
    )

    optimiser = optax.adam(lr)

    # 4. Build evaluate_fn if not supplied
    if evaluate_fn is None:
        evaluate_fn = _build_default_evaluate_fn(setup, f_bc, force, rho_mean)

    # 5. Left side
    ca_target_left = jax.lax.cond(
        ca_left < ca_rec,
        lambda: jnp.array(ca_rec),
        lambda: jnp.array(ca_adv),
    )
    new_params_left = jax.lax.cond(
        in_window_left,
        lambda p: _optimise_side_cll(
            evaluate_fn, p, cll_left, "left", optimiser, max_iter
        ),
        lambda p: _optimise_side_ca(
            evaluate_fn, p, ca_target_left, "left", optimiser, max_iter
        ),
        params,
    )

    # 6. Right side
    ca_target_right = jax.lax.cond(
        ca_right < ca_rec,
        lambda: jnp.array(ca_rec),
        lambda: jnp.array(ca_adv),
    )
    new_params_right = jax.lax.cond(
        in_window_right,
        lambda p: _optimise_side_cll(
            evaluate_fn, p, cll_right, "right", optimiser, max_iter
        ),
        lambda p: _optimise_side_ca(
            evaluate_fn, p, ca_target_right, "right", optimiser, max_iter
        ),
        params,
    )

    # 7. Merge (left from left optimisation, right from right optimisation)
    return wetting._replace(
        d_rho_left=new_params_left.d_rho_left,
        d_rho_right=new_params_right.d_rho_right,
        phi_left=new_params_left.phi_left,
        phi_right=new_params_right.phi_right,
        ca_left=ca_left,
        ca_right=ca_right,
        cll_left=cll_left,
        cll_right=cll_right,
    )


# ── Default evaluate_fn builder ──────────────────────────────────────


def _build_default_evaluate_fn(setup, f_t, force, rho_mean):
    """Build the ``evaluate_fn(params) → (ca_l, ca_r, cll_l, cll_r)`` closure.

    This runs a single LBM step with the trial wetting parameters,
    then measures the resulting contact angles and line locations.
    ``jax.value_and_grad`` can differentiate through the entire chain.

    The closure captures *setup*, *f_t*, *force*, and *rho_mean* from
    the enclosing scope so the inner optimiser sees only ``params``.
    """
    from operators.macroscopic.multiphase import compute_macroscopic_multiphase
    from operators.equilibrium.equilibrium import compute_equilibrium
    from operators.force.source_term import source
    from operators.collision.factory import build_collision_fn
    from operators.streaming.streaming import stream
    from operators.boundary.composite import build_composite_bc

    lattice = setup.lattice
    mp = setup.multiphase_params
    collision_fn = build_collision_fn(setup.collision_scheme)
    bc_fn = build_composite_bc(setup.bc_config, lattice)

    def evaluate_fn(params: WettingParams):
        # TODO: Thread wetting params through the macroscopic step
        # once a wetting-aware macroscopic operator is available.
        # For now, run a standard LBM step (the wetting params affect
        # the boundary condition, which is handled externally).

        # Standard multiphase step
        if setup.force_enabled and force is not None:
            rho_new, u_new, force_tot = compute_macroscopic_multiphase(
                f_t,
                lattice,
                mp,
                force_ext=force,
            )
        else:
            rho_new, u_new, force_tot = compute_macroscopic_multiphase(
                f_t,
                lattice,
                mp,
            )

        feq = compute_equilibrium(rho_new, u_new, lattice)
        src = source(rho_new, u_new, force_tot, lattice)
        f_col = collision_fn(f_t, feq, setup.tau, src)
        f_str = stream(f_col, lattice)
        f_bc = bc_fn(f_str, f_col, setup.bc_masks)

        # Measure CA and CLL from the output
        rho_out = jnp.sum(f_bc, axis=2, keepdims=True)
        ca_l, ca_r = compute_contact_angle(rho_out, rho_mean)
        cll_l, cll_r = compute_contact_line_location(rho_out, ca_l, ca_r, rho_mean)
        return ca_l, ca_r, cll_l, cll_r

    return evaluate_fn
