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


# ── Generic optimisation routines ────────────────────────────────────


def _optimise_single_param(
    objective_fn,
    initial_params: WettingParams,
    grad_mask_fn,
    optimiser,
    max_iterations: int,
) -> tuple[WettingParams, jnp.ndarray]:
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
    import optax  # lazy import — optional dependency

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
            _, _, cll_l, _ = evaluate_fn(p)
            return _cost_cll(cll_target, cll_l)

        def mask_d_rho(g):
            return WettingParams(
                g.d_rho_left,
                jnp.zeros_like(g.d_rho_right),
                jnp.zeros_like(g.phi_left),
                jnp.zeros_like(g.phi_right),
            )

        def obj_phi(p):
            _, _, cll_l, _ = evaluate_fn(p)
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
            _, _, _, cll_r = evaluate_fn(p)
            return _cost_cll(cll_target, cll_r)

        def mask_d_rho(g):
            return WettingParams(
                jnp.zeros_like(g.d_rho_left),
                g.d_rho_right,
                jnp.zeros_like(g.phi_left),
                jnp.zeros_like(g.phi_right),
            )

        def obj_phi(p):
            _, _, _, cll_r = evaluate_fn(p)
            return _cost_cll(cll_target, cll_r)

        def mask_phi(g):
            return WettingParams(
                jnp.zeros_like(g.d_rho_left),
                jnp.zeros_like(g.d_rho_right),
                jnp.zeros_like(g.phi_left),
                g.phi_right,
            )

    p_drho, loss_drho = _optimise_single_param(
        obj_d_rho,
        initial_params,
        mask_d_rho,
        optimiser,
        max_iterations,
    )
    p_phi, loss_phi = _optimise_single_param(
        obj_phi,
        initial_params,
        mask_phi,
        optimiser,
        max_iterations,
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
            ca_l, _ca_r, _cll_l, _cll_r = evaluate_fn(p)
            return _cost_ca(ca_target, ca_l)

        def mask_d_rho(g):
            return WettingParams(
                g.d_rho_left,
                jnp.zeros_like(g.d_rho_right),
                jnp.zeros_like(g.phi_left),
                jnp.zeros_like(g.phi_right),
            )

        def obj_phi(p):
            ca_l, _ca_r, _cll_l, _cll_r = evaluate_fn(p)
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
            _ca_l, ca_r, _cll_l, _cll_r = evaluate_fn(p)
            return _cost_ca(ca_target, ca_r)

        def mask_d_rho(g):
            return WettingParams(
                jnp.zeros_like(g.d_rho_left),
                g.d_rho_right,
                jnp.zeros_like(g.phi_left),
                jnp.zeros_like(g.phi_right),
            )

        def obj_phi(p):
            _ca_l, ca_r, _cll_l, _cll_r = evaluate_fn(p)
            return _cost_ca(ca_target, ca_r)

        def mask_phi(g):
            return WettingParams(
                jnp.zeros_like(g.d_rho_left),
                jnp.zeros_like(g.d_rho_right),
                jnp.zeros_like(g.phi_left),
                g.phi_right,
            )

    p_drho, loss_drho = _optimise_single_param(
        obj_d_rho,
        initial_params,
        mask_d_rho,
        optimiser,
        max_iterations,
    )
    p_phi, loss_phi = _optimise_single_param(
        obj_phi,
        initial_params,
        mask_phi,
        optimiser,
        max_iterations,
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
    force_ext=None,
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
            f_bc,
            force_ext,
            rho_mean,
        )

    # 5. Left side
    ca_target_left = jax.lax.cond(
        ca_left < ca_rec,
        lambda: jnp.array(ca_rec),
        lambda: jnp.array(ca_adv),
    )
    new_params_left = jax.lax.cond(
        in_window_left,
        lambda p: _optimise_side_cll(
            evaluate_fn,
            p,
            cll_left,
            "left",
            optimiser,
            max_iter,
        ),
        lambda p: _optimise_side_ca(
            evaluate_fn,
            p,
            ca_target_left,
            "left",
            optimiser,
            max_iter,
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
            evaluate_fn,
            p,
            cll_right,
            "right",
            optimiser,
            max_iter,
        ),
        lambda p: _optimise_side_ca(
            evaluate_fn,
            p,
            ca_target_right,
            "right",
            optimiser,
            max_iter,
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
                setup, temp_wetting,
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
                rho, u, force_tot, lattice, gradient=gradient_density,
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
