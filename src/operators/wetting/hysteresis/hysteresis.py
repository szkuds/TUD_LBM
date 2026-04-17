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

1. Measures contact angles and contact-line locations from ``rho``.
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
from typing import NamedTuple
import jax
import jax.numpy as jnp
from operators.wetting._contact_angle import compute_contact_angle
from operators.wetting._contact_line import compute_contact_line_location
from registry import wetting_operator
from state.state import WettingState

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
    """Clamp wetting parameters to physically reasonable ranges.

    Note: ``jnp.clip`` has zero gradient at the boundaries, so a
    parameter sitting at a clamp limit receives no further gradient
    signal in that direction.  The ranges below cover physically
    realistic wetting conditions.
    """
    return WettingParams(
        d_rho_left=jnp.clip(params.d_rho_left, 0.0, 0.2),
        d_rho_right=jnp.clip(params.d_rho_right, 0.0, 0.2),
        phi_left=jnp.clip(params.phi_left, 1.0, 1.5),
        phi_right=jnp.clip(params.phi_right, 1.0, 1.5),
    )


def _cost_cll(cll_target: jnp.ndarray, cll_current: jnp.ndarray) -> jnp.ndarray:
    """Squared error for CLL pinning — smooth gradient everywhere."""
    return (cll_target - cll_current) ** 2


def _cost_ca(ca_target: jnp.ndarray, ca_current: jnp.ndarray) -> jnp.ndarray:
    """Squared error for CA targeting — smooth gradient everywhere."""
    return (ca_target - ca_current) ** 2


# ── Generic optimisation routine ─────────────────────────────────────


def _optimise_single_param(
    objective_fn,
    initial_params: WettingParams,
    grad_mask_fn,
    optimiser,
    max_iterations: int,
    loss_tol: float = 1e-6,
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

    def cond_fn(carry):
        _params, _opt_state, loss, iteration = carry
        return (iteration < max_iterations) & (loss > loss_tol)

    def body_fn(carry):
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


@wetting_operator(name="hysteresis")
def update_wetting_state(
    wetting: WettingState,
    rho: jnp.ndarray,
    setup,
    f_t: jnp.ndarray,
    *,
    evaluate_fn=None,
    force_ext=None,
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

    Args:
        wetting: Current :class:`WettingState`.
        rho: Density field, shape ``(nx, ny, 1, 1)``.
        setup: :class:`~setup.simulation_setup.SimulationSetup`
            (closed-over, not traced).
        f_t: Pre-step populations, shape ``(nx, ny, q, 1)``.
        evaluate_fn: Optional callable
            ``(WettingParams) → (ca_l, ca_r, cll_l, cll_r)``
            used by the inner optimiser.  If ``None``, a default is
            built from the pure-function operators.
        force_ext: External force field, shape ``(nx, ny, 1, 2)``
            or ``None``.  Passed to the multiphase macroscopic function
            inside the default evaluate_fn so that interparticle forces
            are not double-counted.

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
    hc = setup.config.hysteresis_config
    ca_adv = hc["ca_advancing"]
    ca_rec = hc["ca_receding"]
    lr = hc.get("learning_rate", 0.01)
    max_iter = hc.get("max_iterations", 20)
    loss_tol = hc.get("loss_tolerance", 1e-6)

    in_window_left = (ca_left >= ca_rec) & (ca_left <= ca_adv)
    in_window_right = (ca_right >= ca_rec) & (ca_right <= ca_adv)

    # 3. Current optimisable parameters
    params = WettingParams(
        d_rho_left=wetting.d_rho_left,
        d_rho_right=wetting.d_rho_right,
        phi_left=wetting.phi_left,
        phi_right=wetting.phi_right,
    )

    try:
        import optax  # lazy import — optional dependency
    except ImportError as err:
        raise ImportError(
            "The 'optax' package is required for hysteresis wetting.\nInstall it with:  pip install optax"
        ) from err
    optimiser = optax.adam(lr)

    # 4. Build evaluate_fn if not supplied
    if evaluate_fn is None:
        evaluate_fn = _build_default_evaluate_fn(
            setup,
            f_t,
            force_ext,
            rho_mean,
        )

    # 5. Combined objective — both sides in a single loop.
    ca_target_left = jnp.where(ca_left < ca_rec, ca_rec, ca_adv)
    ca_target_right = jnp.where(ca_right < ca_rec, ca_rec, ca_adv)

    # --- 5.1: per-side objectives ---

    def left_objective(p):
        ca_l, _, cll_l, _ = evaluate_fn(p)
        return jnp.where(in_window_left,
                         _cost_cll(cll_left, cll_l),
                         _cost_ca(ca_target_left, ca_l))

    def right_objective(p):
        _, ca_r, _, cll_r = evaluate_fn(p)
        return jnp.where(in_window_right,
                         _cost_cll(cll_right, cll_r),
                         _cost_ca(ca_target_right, ca_r))

    def left_mask(g):
        return g._replace(
            phi_right=jnp.zeros_like(g.phi_right),
            d_rho_right=jnp.zeros_like(g.d_rho_right),
        )

    def right_mask(g):
        return g._replace(
            phi_left=jnp.zeros_like(g.phi_left),
            d_rho_left=jnp.zeros_like(g.d_rho_left),
        )

    # --- 5.2: optimise left side, then right side ---
    p1, _ = _optimise_single_param(left_objective, params, left_mask, optimiser, max_iter, loss_tol)
    new_params, _ = _optimise_single_param(right_objective, p1, right_mask, optimiser, max_iter, loss_tol)

    # 6. Return updated wetting state
    return wetting._replace(
        d_rho_left=new_params.d_rho_left,
        d_rho_right=new_params.d_rho_right,
        phi_left=new_params.phi_left,
        phi_right=new_params.phi_right,
        ca_left=ca_left,
        ca_right=ca_right,
        cll_left=cll_left,
        cll_right=cll_right,
    )


# ── Default evaluate_fn builder ──────────────────────────────────────


def _build_default_evaluate_fn(
    setup,
    f_t,
    force_ext,
    rho_mean,
):
    """Build the ``evaluate_fn(params) → (ca_l, ca_r, cll_l, cll_r)`` closure.

    Runs a single LBM trial step with the candidate wetting parameters
    and measures the resulting contact angles and contact-line locations.
    ``jax.value_and_grad`` can differentiate through the entire chain
    because ``params`` flows into the wetting differential-operator
    shims, creating a differentiable path:

        params → WettingState → shims (gradient / laplacian)
        → macroscopic (chemical-potential force) → collision (+source)
        → streaming → BCs → rho_out → CA / CLL → loss

    The closure captures *setup*, *f_t*, *force_ext*, and *rho_mean*
    from the enclosing scope so the inner optimiser sees only ``params``.

    All operators are the pre-built closures on *setup* — no operator
    rebuilding occurs inside the closure.

    Args:
        setup: :class:`~setup.simulation_setup.SimulationSetup`.
        f_t: Current populations, shape ``(nx, ny, q, 1)``.
        force_ext: External force, shape ``(nx, ny, 1, 2)`` or ``None``.
            Passed to the multiphase macroscopic function so that
            interparticle forces are not double-counted.
        rho_mean: Mean density ``(rho_l + rho_v) / 2``.
    """
    from operators.step._wetting_shims import _make_wetting_differential_ops

    lattice = setup.lattice
    mp = setup.multiphase_params

    def evaluate_fn(params: WettingParams):
        # Build a temporary WettingState from the trial params so that
        # _make_wetting_differential_ops can create shims that close
        # over the *traced* param values — this is the key differentiable
        # connection that was previously missing.
        #
        # Guard: wetting shims are only valid when the setup was built
        # with a wetting_config (i.e. the density differential ops
        # accept (grid, phi_l, phi_r, d_rho_l, d_rho_r)).  Without it,
        # fall back to the plain (grid → result) closures on setup.
        if setup.config.wetting_config is not None:
            temp_wetting = WettingState(
                d_rho_left=params.d_rho_left,
                d_rho_right=params.d_rho_right,
                phi_left=params.phi_left,
                phi_right=params.phi_right,
                ca_left=jnp.zeros(()),
                ca_right=jnp.zeros(()),
                cll_left=jnp.zeros(()),
                cll_right=jnp.zeros(()),
            )
            gradient_density, laplacian_density = _make_wetting_differential_ops(
                setup,
                temp_wetting,
            )
        else:
            gradient_density = setup.gradient_density
            laplacian_density = setup.laplacian_density

        # 1. Multiphase macroscopic (chemical-potential force, wetting-corrected laplacian)
        rho, u, force_tot = setup.macroscopic_fn(
            f_t,
            lattice,
            mp,
            force_ext,
            gradient_standard=setup.gradient_standard,
            laplacian_density=laplacian_density,
        )

        # 2. Equilibrium
        feq = setup.equilibrium_fn(rho, u, lattice)

        # 3. Collision (with source using wetting-corrected gradient)
        if force_tot is not None and setup.forces is not None:
            src = setup.forces.source_term(
                rho,
                u,
                force_tot,
                lattice,
                gradient=gradient_density,
            )
            f_col = setup.collision_fn(f_t, feq, setup.tau, src)
        else:
            f_col = setup.collision_fn(f_t, feq, setup.tau)

        # 4. Streaming
        f_str = setup.streaming_fn(f_col, lattice)

        # 5. Boundary conditions
        f_bc = setup.bc_fn(f_str, f_col, setup.bc_masks)

        # 6. Measure CA and CLL from the output density
        rho_out = jnp.sum(f_bc, axis=2, keepdims=True)
        ca_l, ca_r = compute_contact_angle(rho_out, rho_mean)
        cll_l, cll_r = compute_contact_line_location(rho_out, ca_l, ca_r, rho_mean)
        return ca_l, ca_r, cll_l, cll_r

    return evaluate_fn
