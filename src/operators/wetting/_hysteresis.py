"""Wetting hysteresis optimisation — pure functions.

All inner optimisation loops use ``optax`` + ``jax.lax.while_loop``
with early convergence exit and are fully jittable.

Design
~~~~~~
``update_wetting_state`` is the top-level entry point.  It:

1. Measures contact angles and contact-line locations from ``rho_t_plus1``.
2. Classifies each side as pinned, advancing or receding (:func:`_side_regime`).
3. Builds per-side objectives (CLL-pin or CA-bound, selected via
   ``jnp.where``).
4. Runs **two** sequential ``jax.lax.while_loop`` optimisations —
   one for each side — masking the other side's parameters.
5. Returns an updated :class:`WettingState` — no mutation.

The inner ``_evaluate_with_params`` closure performs a single LBM
step with trial wetting parameters so that ``jax.value_and_grad``
can differentiate through it.

Why not one carried signed control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A version replaced the phi/d_rho selection with one signed control per side,
carried across steps and never reset. It oscillated: a freshly initialised
Adam moves by about ``lr * sign(grad)`` per iteration whatever the error, so
the carried control slewed by ``lr * max_iterations`` every step (0.05 at
``lr = 0.005``, capped on 94% of steps), and against the contact angle's lag of
tens of steps that integrator settled into a limit cycle — the control swept
its whole range and the angle swung ±9° inside a 20° window. Snapping the
inactive knob to neutral is what keeps the present optimiser bounded.

Angle convention
~~~~~~~~~~~~~~~~
``compute_contact_angle`` reports the angle through the **dispersed**
phase — the liquid for a droplet, the vapour for a bubble — so the
``ca_advancing`` / ``ca_receding`` window is in those terms too. For a
droplet that is the usual liquid angle and nothing changes. For a bubble
it is the vapour angle, and because vapour advancing is liquid receding,
a window meant as liquid ``[rec, adv]`` becomes ``[180 - adv, 180 - rec]``
here. :func:`_side_regime` works in the same dispersed frame.

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
from typing import NamedTuple
from typing import Protocol
import jax
import jax.numpy as jnp
from src.operators.wetting._contact_angle import compute_contact_angle
from src.operators.wetting._contact_line import compute_contact_line_location
from src.operators.wetting._interface_crossings import detect_bubble
from src.operators.wetting._params import WettingParams
from src.registry import hysteresis_operator
from src.simulation_io.analysis import wetting_debug

if TYPE_CHECKING:
    import types
    from collections.abc import Callable
    from collections.abc import Mapping
    from src.config.simulation_config import SimulationConfig
    from src.pipeline.setup import SimulationSetup
    from src.pipeline.state.state import WettingState


class _OptaxLike(Protocol):
    """Minimal structural type for optax-compatible optimisers."""

    def init(self, params: WettingParams) -> object: ...
    def update(
        self, updates: WettingParams, state: object, params: WettingParams | None = ...
    ) -> tuple[WettingParams, object]: ...


# ── Helpers ──────────────────────────────────────────────────────────

# Neutral values — the inactive parameter is snapped to these when the
# directional split is applied.
_PHI_NEUTRAL: jnp.ndarray = jnp.array(1.0)
_D_RHO_NEUTRAL: jnp.ndarray = jnp.array(0.0)

# Tolerance for "phi is still sitting on its clamp floor". `_clamp_params`
# pins phi at exactly `_PHI_NEUTRAL`, so a strict `<` comparison is
# unreachable and the d_rho fallback below it would never fire.
_PHI_FLOOR_EPS: float = 1e-6

# Regime codes, as reported to the ``--debug-wetting`` trace.
REGIME_PINNED = 0
REGIME_ADVANCING = 1
REGIME_RECEDING = 2
REGIME_SATURATED = 3


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
        cll_stored: Contact-line anchor carried in ``WettingState``.
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


def _side_regime(
    ca: jnp.ndarray,
    cll: jnp.ndarray,
    anchor: jnp.ndarray,
    ca_adv: jnp.ndarray | float,
    ca_rec: jnp.ndarray | float,
    *,
    side: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Classify one contact line: may it move, and in which direction?

    The dispersed phase expanding is the left contact line moving in
    ``-tangential`` and the right one in ``+tangential``, measured against the
    **anchor**, not the previous step. Out of window, a side targets its exceeded
    bound and its anchor follows the contact line unless the line has moved the
    *wrong* way for that bound — contracting above ``ca_adv``, expanding below
    ``ca_rec``. That side, and every in-window side, is pinned.

    That is the difference from the predecessor, which refreshed the anchor on
    every out-of-window step. A contact line receding while its angle rode the
    advancing bound then dragged its own pin along, pinning never resisted the
    recession, and the angle never fell through the window to ``ca_rec``.

    Args:
        ca: Measured (dispersed-frame) contact angle.
        cll: Measured contact-line location.
        anchor: Stored contact-line location the side is pinned to.
        ca_adv: Advancing bound of this side's window.
        ca_rec: Receding bound of this side's window.
        side: ``"left"`` or ``"right"``.

    The two conditions are keyed on the direction of motion, not only on the
    angle, because at a chemical step the window can be *inverted*
    (``ca_rec > ca_adv``, see :func:`_get_hysteresis_window_chemical_step`) and
    an angle can then exceed both bounds at once: the line advances if it is not
    contracting, and recedes otherwise.

    Returns:
        ``(moving, advancing)``: ``moving`` is True when the side targets a
        bound and its anchor follows; ``advancing`` is True when that bound is
        ``ca_adv`` (False with ``moving`` means ``ca_rec``).
    """
    if side == "left":
        expanding, contracting = cll < anchor, cll > anchor
    elif side == "right":
        expanding, contracting = cll > anchor, cll < anchor
    else:
        msg = f"side must be 'left' or 'right', got {side!r}"
        raise ValueError(msg)
    advancing = (ca > ca_adv) & ~contracting
    receding = (ca < ca_rec) & ~expanding
    return advancing | receding, advancing


def _regime_code(moving: jnp.ndarray, advancing: jnp.ndarray, saturated: jnp.ndarray) -> jnp.ndarray:
    """Encode a side's regime for the debug trace."""
    code = jnp.where(moving, jnp.where(advancing, REGIME_ADVANCING, REGIME_RECEDING), REGIME_PINNED)
    return jnp.where(saturated, REGIME_SATURATED, code)


class StepSide(NamedTuple):
    """Where one contact line stands relative to a chemical step.

    Attributes:
        on_post: The line is on the post-step surface.
        held: The line is Gibbs-pinned at the step (see
            :func:`_get_hysteresis_window_chemical_step`).
        pre_phi: ``phi`` of the pre-step surface in this line's band.
        pre_d_rho: ``d_rho`` of the pre-step surface in this line's band.
    """

    on_post: jnp.ndarray
    held: jnp.ndarray
    pre_phi: float
    pre_d_rho: float


def _step_saturation(
    step_side: StepSide,
    *,
    ca: jnp.ndarray,
    ca_adv: jnp.ndarray,
    ca_rec: jnp.ndarray,
    moving: jnp.ndarray,
    advancing: jnp.ndarray,
    is_bubble: jnp.ndarray,
    gap: float,
    w: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``(saturated, phi, d_rho)`` for one side at a chemical step.

    A line on the post surface, or held at the step, whose angle is more than
    *gap* from the bound it is heading for, skips the optimiser and takes the
    wall's full authority toward that bound — the post surface "starts at the
    bound" rather than being optimised up to it from wherever it was:

    * advancing above ``ca_adv``, or held at the step above the post receding
      angle ``ca_rec`` — the angle must fall, so a more liquid-wetting wall;
    * receding below ``ca_rec`` — the angle must rise, so a less wetting one.

    The knob is chosen from that direction (liquid-frame, hence the bubble flip)
    at its clamp limit. A held line standing on the pre-step side of the edge
    instead takes the pre surface's configured values: its own band cells are
    pre-step cells, and the pre surface must keep pushing the liquid away while
    the post-step cells — fixed at their own limit by the applicator — pull.
    Within *gap* of the bound the optimiser takes over.
    """
    receding = moving & ~advancing
    far = (
        (advancing & (ca - ca_adv > gap))
        | (step_side.held & ~moving & (ca - ca_rec > gap))
        | (receding & (ca_rec - ca > gap))
    )
    saturated = (step_side.on_post | step_side.held) & far
    use_phi = (advancing | step_side.held) ^ is_bubble
    phi = jnp.where(use_phi, 1.0 + 5 / w, _PHI_NEUTRAL)
    d_rho = jnp.where(use_phi, _D_RHO_NEUTRAL, 1.5 / w)
    held_on_pre = step_side.held & ~step_side.on_post
    return (
        saturated,
        jnp.where(held_on_pre, step_side.pre_phi, phi),
        jnp.where(held_on_pre, step_side.pre_d_rho, d_rho),
    )


def _saturated_params(params: WettingParams, phi: jnp.ndarray, d_rho: jnp.ndarray, *, side: str) -> WettingParams:
    """*params* with one side's ``phi``/``d_rho`` replaced (see :func:`_step_saturation`)."""
    if side == "left":
        return params._replace(phi_left=phi, d_rho_left=d_rho)
    if side == "right":
        return params._replace(phi_right=phi, d_rho_right=d_rho)
    msg = f"side must be 'left' or 'right', got {side!r}"
    raise ValueError(msg)


def _phi_is_active(
    pinned: jnp.ndarray,
    advancing: jnp.ndarray,
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
    advancing, droplet          decrease            more wetting    phi
    advancing, bubble           increase            less wetting    d_rho
    receding, droplet           increase            less wetting    d_rho
    receding, bubble            decrease            more wetting    phi
    pinned, liquid receding     --                  more wetting    phi
    pinned, liquid advancing    --                  less wetting    d_rho
    ==========================  ==================  ==============  ======

    "Pinned" is every side :func:`_side_regime` does not let move — in window,
    or out of it but moving the wrong way for the exceeded bound. The two
    pinned rows are topology-independent: ``forward_drift`` arrives already
    converted to the liquid frame by :func:`_liquid_is_advancing`, and the knobs
    are liquid-frame too, so no further flip is needed. Pinning the contact line
    means resisting whichever way the liquid is moving.

    Args:
        pinned: bool scalar — the side is held on its anchor.
        advancing: bool scalar — the side targets ``ca_advancing`` (from
            :func:`_side_regime`). Only read when not ``pinned``, where False
            means it targets ``ca_receding``.
        forward_drift: bool scalar — the **liquid** is advancing over dry wall
            at this contact line, as returned by :func:`_liquid_is_advancing`.
        is_bubble: bool scalar — the phase dispersed at the wall is vapour, so
            the reported angle is the complement of the liquid angle.

    Returns:
        Boolean JAX scalar; True means phi is active, False means d_rho is active.
    """
    # Out of window: push the reported angle back toward the exceeded bound.
    # Which knob does that depends on the topology, hence the is_bubble flip.
    ca_branch = jnp.where(advancing, ~is_bubble, is_bubble)
    return jnp.where(pinned, ~forward_drift, ca_branch)


def _side_hyperparams(
    hysteresis_config: Mapping[str, float],
    urgent: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(learning_rate, max_iterations)`` for one side.

    A side advancing above ``ca_advancing`` is the urgent case, and may be given
    a larger step and a longer budget via ``learning_rate_above`` /
    ``max_iterations_above``.  Both selections are traced, so each side is keyed
    on **its own** flag.

    Note that only the above-window advance is treated as urgent; a side
    below ``ca_receding``, or pinned above the window, gets the default budget.

    Args:
        hysteresis_config: The ``hysteresis_config`` mapping, with defaults
            already applied by ``SimulationConfig``.
        urgent: bool scalar — this side is advancing above ca_advancing.

    Returns:
        ``(lr, max_iterations)`` as JAX scalars.
    """
    return (
        jnp.where(urgent, hysteresis_config["learning_rate_above"], hysteresis_config["learning_rate"]),
        jnp.where(urgent, hysteresis_config["max_iterations_above"], hysteresis_config["max_iterations"]),
    )


def _initial_params(
    wetting: WettingState,
    phi_active_left: jnp.ndarray,
    phi_active_right: jnp.ndarray,
    *,
    carry_inactive: bool,
) -> WettingParams:
    """Starting point of this timestep's optimisation.

    By default the knob that is *not* active on a side is snapped to neutral,
    so a side that switches branch discards what it had accumulated. On an edge
    sitting on a window bound that switch happens every few steps, and the wall
    spends much of its time at neutral wettability rather than at the bound it
    reports. ``carry_inactive=True`` (``hysteresis_config.carry_inactive_params``)
    instead keeps both knobs: the gradient mask still updates only the active
    one, the inactive one is simply left where it was. ``carry_inactive`` is a
    Python bool read off the config, so it selects the trace, not a branch in it.
    """
    if carry_inactive:
        return WettingParams(
            phi_left=wetting.phi_left,
            phi_right=wetting.phi_right,
            d_rho_left=wetting.d_rho_left,
            d_rho_right=wetting.d_rho_right,
        )
    return WettingParams(
        phi_left=jnp.where(phi_active_left, wetting.phi_left, _PHI_NEUTRAL),
        phi_right=jnp.where(phi_active_right, wetting.phi_right, _PHI_NEUTRAL),
        d_rho_left=jnp.where(phi_active_left, _D_RHO_NEUTRAL, wetting.d_rho_left),
        d_rho_right=jnp.where(phi_active_right, _D_RHO_NEUTRAL, wetting.d_rho_right),
    )


def _clamp_params(params: WettingParams, w: jnp.ndarray) -> WettingParams:
    """Clamp wetting parameters to physically reasonable, W-scaled ranges.

    The wetting parameters act on near-wall density profiles whose
    magnitude scales inversely with the interface width ``W``:
    ``phi`` ∈ [1, 1 + 5/W] and ``d_rho`` ∈ [0, 1.5/W].  At the base
    resolution (W = 5) these are ``phi <= 2`` and ``d_rho <= 0.3``.

    Note: ``jnp.clip`` has zero gradient at the boundaries, so a
    parameter sitting at a clamp limit receives no further gradient
    signal in that direction.
    """
    return WettingParams(
        phi_left=jnp.clip(params.phi_left, 1.0, 1.0 + 5 / w),
        phi_right=jnp.clip(params.phi_right, 1.0, 1.0 + 5 / w),
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


def _side_cost(
    ca: jnp.ndarray,
    cll: jnp.ndarray,
    *,
    anchor: jnp.ndarray,
    moving: jnp.ndarray,
    advancing: jnp.ndarray,
    ca_adv: jnp.ndarray,
    ca_rec: jnp.ndarray,
) -> jnp.ndarray:
    """Objective of one side: its bound when moving, else its pin inside the window.

    A pinned side holds its contact line *and* keeps its angle within
    ``[ca_rec, ca_adv]``. The window terms are zero in window, so they only act
    on a side pinned because its line moved the wrong way for the bound it has
    exceeded. Without them the pin alone was satisfied while the angle climbed
    past ``ca_adv`` (56° → 65° against 50° in 100 steps): no line can stay
    pinned above its advancing angle. Both terms ask for the same knob there —
    resisting a line that retreats while its angle is too high means a more
    wetting wall, which also lowers the angle — so they never compete.
    """
    window_cost = _cost_above(ca_adv, ca) + _cost_below(ca_rec, ca)
    bound_cost = jnp.where(advancing, _cost_above(ca_adv, ca), _cost_below(ca_rec, ca))
    return jnp.where(moving, bound_cost, _cost_cll(anchor, cll) + window_cost)


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
) -> tuple[WettingParams, jnp.ndarray, jnp.ndarray]:
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
        ``(final_params, final_loss, iterations)``.  ``iterations`` is what
        the ``--debug-wetting`` trace reports against ``max_iterations``:
        equality there means the loop hit the cap rather than the tolerance.
    """
    opt_state = optimiser.init(initial_params)
    initial_loss = objective_fn(initial_params)

    def cond_fn(carry: tuple) -> jnp.ndarray:
        _params, _opt_state, loss, iteration = carry
        return (iteration < max_iterations) & (loss > loss_tol)

    def body_fn(carry: tuple) -> tuple:
        params, opt_state, _loss, iteration = carry
        loss, grads = jax.value_and_grad(objective_fn)(params)
        updates, new_opt_state = optimiser.update(grad_mask_fn(grads), opt_state, params)
        stepped = WettingParams(*(p + u for p, u in zip(params, updates, strict=True)))
        return (_clamp_params(stepped, w), new_opt_state, loss, iteration + 1)

    init_carry = (initial_params, opt_state, initial_loss, jnp.array(0))
    final_params, _opt_state, final_loss, iters = jax.lax.while_loop(
        cond_fn,
        body_fn,
        init_carry,
    )
    return final_params, final_loss, iters


# ── Chemical step ────────────────────────────────────────────────────


def chemical_step_x(config: SimulationConfig) -> float:
    """Tangential position of the chemical step, in lattice units."""
    if config.chemical_step_config is None:
        msg = "chemical_step_config is required for chemical step hysteresis"
        raise TypeError(msg)
    return float(config.chemical_step_config["chemical_step_location"]) * config.grid_shape[0]


def _get_hysteresis_window_chemical_step(
    setup: SimulationSetup, cll: jnp.ndarray, *, side: str
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(ca_advancing, ca_receding, held_at_step)`` for one contact line.

    The window is the surface's own: the post-step window once the line is
    within ``edge_width`` of the step or past it, the pre-step window otherwise.
    It is stateless and has no deadband to latch. The step itself is not
    expressed through the window but through the wall: band cells across the
    step keep their own surface's wettability
    (:class:`~src.config.chemical_step.ChemicalStepWall`), so the pre surface
    pushes while the post surface pulls.

    ``held_at_step`` marks a line at the edge that is receding onto the more
    wetting surface: its probes ``edge_width`` either side straddle the step,
    with the less wetting surface ahead of it (outward: ``-tangential`` for the
    left line, ``+tangential`` for the right). Such a line cannot advance back
    onto the pre surface and has not yet reached the post receding angle. The
    operator pins it to the step position and saturates it toward that angle
    (:func:`_step_saturation`). A line advancing onto the more wetting surface is
    never held; it spreads.
    """
    step_x = chemical_step_x(setup.config)
    csc = setup.config.chemical_step_config
    assert csc is not None  # noqa: S101 - checked by chemical_step_x
    delta = csc["edge_width"]
    if side == "left":
        ahead, behind = cll - delta, cll + delta
    elif side == "right":
        ahead, behind = cll + delta, cll - delta
    else:
        msg = f"side must be 'left' or 'right', got {side!r}"
        raise ValueError(msg)
    post = cll + delta >= step_x
    ca_adv = jnp.where(post, csc["ca_advancing_post_step"], csc["ca_advancing_pre_step"])
    ca_rec = jnp.where(post, csc["ca_receding_post_step"], csc["ca_receding_pre_step"])
    # Held: the surface behind the line (the one it would uncover) is post, the
    # one ahead is pre, and post wets more — its receding angle lies below the
    # pre advancing angle.
    ahead_post, behind_post = ahead >= step_x, behind >= step_x
    post_wets_more = csc["ca_receding_post_step"] < csc["ca_advancing_pre_step"]
    held_at_step = behind_post & ~ahead_post & post_wets_more
    return ca_adv, ca_rec, held_at_step


# ── Top-level entry point ────────────────────────────────────────────


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
    measured: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    pin_left: jnp.ndarray | None = None,
    pin_right: jnp.ndarray | None = None,
    step_sides: tuple[StepSide, StepSide] | None = None,
    t: jnp.ndarray | None = None,
) -> WettingState:
    """Shared implementation for hysteresis wetting updates.

    ``pin_left`` / ``pin_right`` override where a pinned side holds its contact
    line; by default that is its anchor. The chemical-step operator pins a line
    held at the step to the step itself. The anchors still decide which way a
    line is moving and are what the returned state carries.

    ``step_sides`` (chemical-step runs only) lets a side on the post surface, or
    held at the step, skip the optimiser while its angle is more than
    ``hysteresis_config["saturation_gap"]`` from its bound
    (:func:`_step_saturation`). A line that has just met a surface of very
    different wettability cannot be brought to its bound by one step's
    optimiser; the post surface starts at the bound instead.

    ``measured`` is ``(ca_left, ca_right, cll_left, cll_right)`` of
    ``rho_t_plus1`` (:func:`_measure`). ``t`` is the current timestep.  It is
    used only by the ``--debug-wetting`` trace, to stamp each row and to
    rate-limit it; the optimisation itself is timestep-independent.
    """
    if setup.multiphase_params is None:
        msg = "multiphase_params is required for hysteresis wetting update"
        raise TypeError(msg)
    if setup.config.hysteresis_config is None:
        msg = "hysteresis_config is required for hysteresis wetting update"
        raise TypeError(msg)
    mp = setup.multiphase_params
    rho_mean = jnp.array(0.5 * (mp.rho_l + mp.rho_v))
    w = jnp.array(float(mp.interface_width))
    if setup.wetting_edge is None:
        msg = "wetting_edge is required for hysteresis wetting update"
        raise TypeError(msg)
    edge = setup.wetting_edge
    ca_left_tplus1, ca_right_tplus1, cll_left_tplus1, cll_right_tplus1 = measured

    moving_left, advancing_left = _side_regime(
        ca_left_tplus1, cll_left_tplus1, wetting.cll_left, ca_adv_left, ca_rec_left, side="left"
    )
    moving_right, advancing_right = _side_regime(
        ca_right_tplus1, cll_right_tplus1, wetting.cll_right, ca_adv_right, ca_rec_right, side="right"
    )

    is_bubble = detect_bubble(rho_t_plus1, rho_mean, edge=edge)
    forward_drift_right = _liquid_is_advancing(cll_right_tplus1, wetting.cll_right, is_bubble, side="right")
    forward_drift_left = _liquid_is_advancing(cll_left_tplus1, wetting.cll_left, is_bubble, side="left")

    phi_active_right = _phi_is_active(~moving_right, advancing_right, forward_drift_right, is_bubble)
    phi_active_left = _phi_is_active(~moving_left, advancing_left, forward_drift_left, is_bubble)

    hc = setup.config.hysteresis_config
    lr_left, max_iter_left = _side_hyperparams(hc, advancing_left)
    lr_right, max_iter_right = _side_hyperparams(hc, advancing_right)
    loss_tol = float(hc["loss_tol"])
    if step_sides is None:
        no = jnp.array(False)
        saturated_left, sat_phi_left, sat_d_rho_left = no, _PHI_NEUTRAL, _D_RHO_NEUTRAL
        saturated_right, sat_phi_right, sat_d_rho_right = no, _PHI_NEUTRAL, _D_RHO_NEUTRAL
    else:
        gap = float(hc["saturation_gap"])
        saturated_left, sat_phi_left, sat_d_rho_left = _step_saturation(
            step_sides[0],
            ca=ca_left_tplus1,
            ca_adv=ca_adv_left,
            ca_rec=ca_rec_left,
            moving=moving_left,
            advancing=advancing_left,
            is_bubble=is_bubble,
            gap=gap,
            w=w,
        )
        saturated_right, sat_phi_right, sat_d_rho_right = _step_saturation(
            step_sides[1],
            ca=ca_right_tplus1,
            ca_adv=ca_adv_right,
            ca_rec=ca_rec_right,
            moving=moving_right,
            advancing=advancing_right,
            is_bubble=is_bubble,
            gap=gap,
            w=w,
        )
    pin_left = wetting.cll_left if pin_left is None else pin_left
    pin_right = wetting.cll_right if pin_right is None else pin_right

    params = _initial_params(
        wetting,
        phi_active_left,
        phi_active_right,
        carry_inactive=bool(hc["carry_inactive_params"]),
    )

    optax = _import_optax()
    # One optimiser per side: the learning rate is keyed on that side's own
    # urgency flag, so a pinned side is not dragged along by the other.
    optimiser_left = optax.adam(lr_left)
    optimiser_right = optax.adam(lr_right)

    def evaluate_fn(params: WettingParams) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        _, rho_out = trial_step_fn(params)
        ca_l, ca_r = compute_contact_angle(rho_out, rho_mean, edge=edge)
        cll_l, cll_r = compute_contact_line_location(rho_out, ca_l, ca_r, rho_mean, edge=edge)
        return ca_l, ca_r, cll_l, cll_r

    def left_objective(p: WettingParams) -> jnp.ndarray:
        ca_l, _, cll_l, _ = evaluate_fn(p)
        return _side_cost(
            ca_l,
            cll_l,
            anchor=pin_left,
            moving=moving_left,
            advancing=advancing_left,
            ca_adv=ca_adv_left,
            ca_rec=ca_rec_left,
        )

    def right_objective(p: WettingParams) -> jnp.ndarray:
        _, ca_r, _, cll_r = evaluate_fn(p)
        return _side_cost(
            ca_r,
            cll_r,
            anchor=pin_right,
            moving=moving_right,
            advancing=advancing_right,
            ca_adv=ca_adv_right,
            ca_rec=ca_rec_right,
        )

    def _opt_left(p: WettingParams) -> tuple[WettingParams, jnp.ndarray, jnp.ndarray]:
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

    def _opt_right(p: WettingParams) -> tuple[WettingParams, jnp.ndarray, jnp.ndarray]:
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

    def _skip(
        saturate: Callable[[WettingParams], WettingParams],
    ) -> Callable[[WettingParams], tuple[WettingParams, jnp.ndarray, jnp.ndarray]]:
        # Reports zero loss and zero iterations, so the d_rho fallback below —
        # gated on loss > loss_tol — never runs on a saturated side.
        return lambda p: (saturate(p), jnp.zeros(()), jnp.zeros((), dtype=jnp.array(0).dtype))

    params_after_left, loss_left, iters_left = jax.lax.cond(
        saturated_left,
        _skip(lambda p: _saturated_params(p, sat_phi_left, sat_d_rho_left, side="left")),
        _opt_left,
        params,
    )
    new_params, loss_right, iters_right = jax.lax.cond(
        saturated_right,
        _skip(lambda p: _saturated_params(p, sat_phi_right, sat_d_rho_right, side="right")),
        _opt_right,
        params_after_left,
    )

    # Fallback: if the phi path was selected but phi saturated back at its
    # clamp floor without converging, phi was the wrong knob for this side —
    # `jnp.clip` has zero gradient there, so it cannot recover. Retry with
    # d_rho, warm-started from the stored accumulated value.
    # Both branches of the fallback `lax.cond` carry the iteration count so the
    # `--debug-wetting` row can report it; the identity branch reports zero,
    # which is what the trace renders as "no fallback ran".
    def _no_fallback(p: WettingParams) -> tuple[WettingParams, jnp.ndarray]:
        return p, jnp.zeros_like(iters_left)

    def _fallback_d_rho_left(p: WettingParams) -> tuple[WettingParams, jnp.ndarray]:
        fallback = WettingParams(
            phi_left=_PHI_NEUTRAL,
            phi_right=p.phi_right,
            d_rho_left=wetting.d_rho_left,
            d_rho_right=p.d_rho_right,
        )
        params_fb, _loss_fb, iters_fb = _optimise_single_param(
            left_objective, fallback, _mask_left_d_rho, optimiser_left, max_iter_left, w, loss_tol
        )
        return params_fb, iters_fb

    def _fallback_d_rho_right(p: WettingParams) -> tuple[WettingParams, jnp.ndarray]:
        fallback = WettingParams(
            phi_left=p.phi_left,
            phi_right=_PHI_NEUTRAL,
            d_rho_left=p.d_rho_left,
            d_rho_right=wetting.d_rho_right,
        )
        params_fb, _loss_fb, iters_fb = _optimise_single_param(
            right_objective, fallback, _mask_right_d_rho, optimiser_right, max_iter_right, w, loss_tol
        )
        return params_fb, iters_fb

    # The `loss > loss_tol` conjunct is what keeps this from firing on a side
    # whose phi legitimately converged at ~1.0 — without it every such side
    # would pay a second optimisation. `_optimise_single_param` returns the
    # while_loop carry loss, which lags one iteration behind the returned
    # params; that makes the test conservative rather than wrong, since a
    # parameter pinned at a clamp bound is not moving the loss anyway.
    final_params, iters_fb_left = jax.lax.cond(
        phi_active_left & (new_params.phi_left <= _PHI_NEUTRAL + _PHI_FLOOR_EPS) & (loss_left > loss_tol),
        _fallback_d_rho_left,
        _no_fallback,
        new_params,
    )
    final_params, iters_fb_right = jax.lax.cond(
        phi_active_right & (new_params.phi_right <= _PHI_NEUTRAL + _PHI_FLOOR_EPS) & (loss_right > loss_tol),
        _fallback_d_rho_right,
        _no_fallback,
        final_params,
    )

    # Guarded at the call site because building the samples is not free — each
    # objective call below costs a full trial step. `log_sides` re-checks the
    # flag itself, and only calls this thunk on a logged timestep.
    def _debug_sides() -> tuple[wetting_debug.SideDebugSample, wetting_debug.SideDebugSample]:
        return (
            wetting_debug.SideDebugSample(
                ca=ca_left_tplus1,
                ca_adv=jnp.asarray(ca_adv_left),
                ca_rec=jnp.asarray(ca_rec_left),
                cll=cll_left_tplus1,
                regime=_regime_code(moving_left, advancing_left, saturated_left),
                phi=final_params.phi_left,
                d_rho=final_params.d_rho_left,
                phi_active=phi_active_left,
                loss=left_objective(final_params),
                iters=iters_left,
                iters_cap=jnp.asarray(max_iter_left),
                iters_fallback=iters_fb_left,
            ),
            wetting_debug.SideDebugSample(
                ca=ca_right_tplus1,
                ca_adv=jnp.asarray(ca_adv_right),
                ca_rec=jnp.asarray(ca_rec_right),
                cll=cll_right_tplus1,
                regime=_regime_code(moving_right, advancing_right, saturated_right),
                phi=final_params.phi_right,
                d_rho=final_params.d_rho_right,
                phi_active=phi_active_right,
                loss=right_objective(final_params),
                iters=iters_right,
                iters_cap=jnp.asarray(max_iter_right),
                iters_fallback=iters_fb_right,
            ),
        )

    if wetting_debug.enabled():
        wetting_debug.log_sides(_debug_sides, phi_neutral=_PHI_NEUTRAL, t=t)

    return wetting._replace(
        phi_left=final_params.phi_left,
        phi_right=final_params.phi_right,
        d_rho_left=final_params.d_rho_left,
        d_rho_right=final_params.d_rho_right,
        ca_left=ca_left_tplus1,
        ca_right=ca_right_tplus1,
        cll_left=jnp.where(moving_left, cll_left_tplus1, wetting.cll_left),
        cll_right=jnp.where(moving_right, cll_right_tplus1, wetting.cll_right),
    )


def _measure(rho: jnp.ndarray, setup: SimulationSetup) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``(ca_left, ca_right, cll_left, cll_right)`` of *rho* at the wetting wall."""
    if setup.multiphase_params is None:
        msg = "multiphase_params is required for hysteresis wetting update"
        raise TypeError(msg)
    if setup.wetting_edge is None:
        msg = "wetting_edge is required for hysteresis wetting update"
        raise TypeError(msg)
    mp = setup.multiphase_params
    rho_mean = jnp.array(0.5 * (mp.rho_l + mp.rho_v))
    edge = setup.wetting_edge
    ca_left, ca_right = compute_contact_angle(rho, rho_mean, edge=edge)
    cll_left, cll_right = compute_contact_line_location(rho, ca_left, ca_right, rho_mean, edge=edge)
    return ca_left, ca_right, cll_left, cll_right


@hysteresis_operator(name="hysteresis")
def update_wetting_state(
    wetting: WettingState,
    rho_t_plus1: jnp.ndarray,
    setup: SimulationSetup,
    *,
    trial_step_fn: Callable[[WettingParams], tuple[jnp.ndarray, jnp.ndarray]],
    t: jnp.ndarray | None = None,
) -> WettingState:
    """Pure JAX update of wetting / hysteresis parameters.

    Each side (left and right) is optimised **independently** in its
    own ``jax.lax.while_loop``, masking out the other side's
    parameters.  This gives each side clean gradients and clean Adam
    state at the cost of two trial-step evaluations per outer
    iteration instead of one.

    A side is pinned to its anchor unless its angle has left the global
    ``[ca_receding, ca_advancing]`` window *and* its contact line is not moving
    the wrong way for that bound (see :func:`_side_regime`); a moving side
    targets its bound.

    Args:
        wetting: Current :class:`WettingState`.
        rho_t_plus1: Density field, shape ``(nx, ny, nz, 1, 1)``.
        setup: :class:`~setup.simulation_setup.SimulationSetup`
            (closed-over, not traced).
        trial_step_fn: Callable ``(WettingParams) → (f_out, rho_out)``
            that evaluates a single multiphase physics pass with trial
            wetting parameters. Required; no default is provided.
        t: Current timestep, forwarded to the ``--debug-wetting`` trace
            for its ``t`` column and interval gate.  Physics-inert.

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
        measured=_measure(rho_t_plus1, setup),
        t=t,
    )


@hysteresis_operator(name="chemical_step_hysteresis")
def update_wetting_state_chemical_step(
    wetting: WettingState,
    rho_t_plus1: jnp.ndarray,
    setup: SimulationSetup,
    *,
    trial_step_fn: Callable[[WettingParams], tuple[jnp.ndarray, jnp.ndarray]],
    t: jnp.ndarray | None = None,
) -> WettingState:
    """Hysteresis update where CA targets are determined per-side by chemical step position.

    Identical to update_wetting_state except (ca_advancing, ca_receding) for each
    side is read from the surfaces either side of that contact line
    (:func:`_get_hysteresis_window_chemical_step`) rather than from a single
    global hysteresis window.

    Args:
        wetting: Current WettingState.
        rho_t_plus1: Post-step density field.
        setup: SimulationSetup. Must carry chemical_step_config with keys:
               chemical_step_location, edge_width, ca_advancing_pre_step,
               ca_receding_pre_step, ca_advancing_post_step, ca_receding_post_step.
        trial_step_fn: Callable (WettingParams) -> (f_out, rho_out).
                       Provided by the step function via partial application.
        t: Current timestep, forwarded to the ``--debug-wetting`` trace
           for its ``t`` column and interval gate. Physics-inert.

    Returns:
        Updated WettingState with optimised wetting parameters, measured CA
        and anchors.
    """
    measured = _measure(rho_t_plus1, setup)
    _ca_left, _ca_right, cll_left, cll_right = measured
    ca_adv_left, ca_rec_left, held_left = _get_hysteresis_window_chemical_step(setup, cll_left, side="left")
    ca_adv_right, ca_rec_right, held_right = _get_hysteresis_window_chemical_step(setup, cll_right, side="right")
    # A line held at the step is pinned to the step itself, not to wherever its
    # anchor happened to be when it arrived: the trailing line of a droplet
    # pulled over the step must stay on the edge while the droplet stretches,
    # until its angle falls to the post-step receding angle.
    step_x = chemical_step_x(setup.config)
    wall = setup.config.chemical_step_wall
    assert wall is not None  # noqa: S101 - a chemical-step run has a wetting config and a step

    return _update_wetting_state_impl(
        wetting,
        rho_t_plus1,
        setup,
        trial_step_fn,
        ca_adv_left=ca_adv_left,
        ca_rec_left=ca_rec_left,
        ca_adv_right=ca_adv_right,
        ca_rec_right=ca_rec_right,
        measured=measured,
        pin_left=jnp.where(held_left, step_x, wetting.cll_left),
        pin_right=jnp.where(held_right, step_x, wetting.cll_right),
        step_sides=(
            StepSide(cll_left >= step_x, held_left, wall.pre_phi_left, wall.pre_d_rho_left),
            StepSide(cll_right >= step_x, held_right, wall.pre_phi_right, wall.pre_d_rho_right),
        ),
        t=t,
    )
