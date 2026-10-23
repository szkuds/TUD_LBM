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
   left, then right — masking the other side's parameters.
5. Returns an updated :class:`WettingState` — no mutation.

The objective performs a single LBM step with trial wetting parameters so that
``jax.value_and_grad`` can differentiate through it. The knob that is not
active on a side is snapped to neutral each step; that is what keeps the
optimiser bounded.

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
from src.config.chemical_step import D_RHO_SPAN_W
from src.config.chemical_step import PHI_SPAN_W
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

    def init(self, params: WettingParams) -> object: ...
    def update(
        self, updates: WettingParams, state: object, params: WettingParams | None = ...
    ) -> tuple[WettingParams, object]: ...


# ── Helpers ──────────────────────────────────────────────────────────

#: The two contact lines, in optimisation order.
_SIDES = ("left", "right")

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


def _side_regime(
    ca: jnp.ndarray,
    cll: jnp.ndarray,
    anchor: jnp.ndarray,
    ca_adv: jnp.ndarray | float,
    ca_rec: jnp.ndarray | float,
    *,
    side: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Classify one contact line: may it move, and in which direction?

    The dispersed phase expanding is the left contact line moving in
    ``-tangential`` and the right one in ``+tangential``, measured against the
    **anchor**, not the previous step. Out of window, a side targets its exceeded
    bound and its anchor follows the contact line unless the line has moved the
    *wrong* way for that bound — contracting above ``ca_adv``, expanding below
    ``ca_rec``. That side, and every in-window side, is pinned.

    The two conditions are keyed on the direction of motion, not only on the
    angle, because at a chemical step the window can be *inverted*
    (``ca_rec > ca_adv``) and an angle can then exceed both bounds at once: the
    line advances if it is not contracting, and recedes otherwise.

    Returns:
        ``(moving, advancing, expanding)``: ``moving`` is True when the side
        targets a bound and its anchor follows; ``advancing`` is True when that
        bound is ``ca_adv`` (False with ``moving`` means ``ca_rec``);
        ``expanding`` is True when the dispersed phase has spread past the anchor.
    """
    expanding, contracting = (cll < anchor, cll > anchor) if side == "left" else (cll > anchor, cll < anchor)
    advancing = (ca > ca_adv) & ~contracting
    receding = (ca < ca_rec) & ~expanding
    return advancing | receding, advancing, expanding


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
    phi = jnp.where(use_phi, 1.0 + PHI_SPAN_W / w, _PHI_NEUTRAL)
    d_rho = jnp.where(use_phi, _D_RHO_NEUTRAL, D_RHO_SPAN_W / w)
    held_on_pre = step_side.held & ~step_side.on_post
    return (
        saturated,
        jnp.where(held_on_pre, step_side.pre_phi, phi),
        jnp.where(held_on_pre, step_side.pre_d_rho, d_rho),
    )


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

    "Pinned" is every side :func:`_side_regime` does not let move. The two
    pinned rows are topology-independent: ``forward_drift`` arrives already in
    the liquid frame, and the knobs are liquid-frame too. Pinning the contact
    line means resisting whichever way the liquid is moving.

    Args:
        pinned: bool scalar — the side is held on its anchor.
        advancing: bool scalar — the side targets ``ca_advancing``. Only read
            when not ``pinned``, where False means it targets ``ca_receding``.
        forward_drift: bool scalar — the **liquid** is advancing over dry wall
            at this contact line (the dispersed phase expanding, flipped for a bubble).
        is_bubble: bool scalar — the phase dispersed at the wall is vapour.

    Returns:
        Boolean JAX scalar; True means phi is active, False means d_rho is active.
    """
    ca_branch = jnp.where(advancing, ~is_bubble, is_bubble)
    return jnp.where(pinned, ~forward_drift, ca_branch)


def _side_hyperparams(
    hysteresis_config: Mapping[str, float],
    urgent: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(learning_rate, max_iterations)`` for one side, keyed on its own *urgent* flag.

    Only a side advancing above ``ca_advancing`` is urgent, and takes
    ``learning_rate_above`` / ``max_iterations_above``.
    """
    return (
        jnp.where(urgent, hysteresis_config["learning_rate_above"], hysteresis_config["learning_rate"]),
        jnp.where(urgent, hysteresis_config["max_iterations_above"], hysteresis_config["max_iterations"]),
    )


def _clamp_params(params: WettingParams, w: jnp.ndarray) -> WettingParams:
    """Clamp wetting parameters to ``phi`` ∈ [1, 1 + 5/W] and ``d_rho`` ∈ [0, 1.5/W].

    The parameters act on near-wall density profiles whose magnitude scales
    inversely with the interface width ``W``. ``jnp.clip`` has zero gradient at
    the boundaries, so a parameter on a limit gets no further signal that way.
    """
    return WettingParams(
        phi_left=jnp.clip(params.phi_left, 1.0, 1.0 + PHI_SPAN_W / w),
        phi_right=jnp.clip(params.phi_right, 1.0, 1.0 + PHI_SPAN_W / w),
        d_rho_left=jnp.clip(params.d_rho_left, 0.0, D_RHO_SPAN_W / w),
        d_rho_right=jnp.clip(params.d_rho_right, 0.0, D_RHO_SPAN_W / w),
    )


def _huber(err: jnp.ndarray, delta: float) -> jnp.ndarray:
    return jnp.where(err < delta, 0.5 * err**2, delta * (err - 0.5 * delta))


def _cost_cll(cll_target: jnp.ndarray, cll_current: jnp.ndarray) -> jnp.ndarray:
    """Huber loss for CLL pinning — smooth gradient near zero, linear elsewhere."""
    return _huber(jnp.abs(cll_target - cll_current), 0.5)


def _cost_ca(ca_target: jnp.ndarray, ca_current: jnp.ndarray) -> jnp.ndarray:
    """Huber loss (degrees) for CA targeting."""
    return _huber(jnp.abs(ca_target - ca_current), 5.0)


def _cost_above(ca_adv: jnp.ndarray, ca_current: jnp.ndarray) -> jnp.ndarray:
    """One-sided Huber loss that penalises only CA values above ca_adv."""
    return _huber(jnp.maximum(ca_current - ca_adv, 0.0), 5.0)


def _cost_below(ca_rec: jnp.ndarray, ca_current: jnp.ndarray) -> jnp.ndarray:
    """One-sided Huber loss that penalises only CA values below ca_rec."""
    return _huber(jnp.maximum(ca_rec - ca_current, 0.0), 5.0)


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
    ``[ca_rec, ca_adv]``: no line can stay pinned above its advancing angle. The
    window terms are zero in window, so they only act on a side pinned because
    its line moved the wrong way for the bound it has exceeded, and there they
    ask for the same knob as the pin.
    """
    window_cost = _cost_above(ca_adv, ca) + _cost_below(ca_rec, ca)
    bound_cost = jnp.where(advancing, _cost_above(ca_adv, ca), _cost_below(ca_rec, ca))
    return jnp.where(moving, bound_cost, _cost_cll(anchor, cll) + window_cost)


def _grad_mask(field: str) -> Callable[[WettingParams], WettingParams]:
    """``grads -> grads`` keeping only the gradient of *field* (e.g. ``"phi_left"``)."""

    def mask(g: WettingParams) -> WettingParams:
        return WettingParams(**{name: v if name == field else jnp.zeros_like(v) for name, v in g._asdict().items()})

    return mask


def _with_side(params: WettingParams, side: str, phi: jnp.ndarray, d_rho: jnp.ndarray) -> WettingParams:
    """*params* with one side's ``phi``/``d_rho`` replaced."""
    return params._replace(**{f"phi_{side}": phi, f"d_rho_{side}": d_rho})


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
        max_iterations: Maximum number of inner steps; may be a traced scalar.
        w: Interface width used to scale the parameter clamp bounds
            (see :func:`_clamp_params`).
        loss_tol: Convergence tolerance; the loop exits once the loss
            drops to or below this value.

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
    final_params, _opt_state, final_loss, iters = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return final_params, final_loss, iters


# ── Chemical step ────────────────────────────────────────────────────


def _get_hysteresis_window_chemical_step(
    setup: SimulationSetup, cll: jnp.ndarray, *, side: str
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(ca_advancing, ca_receding, held_at_step)`` for one contact line.

    The window is the surface's own: the post-step window once the line is
    within ``edge_width`` of the step or past it, the pre-step window otherwise.
    It is stateless. The step itself is expressed through the wall: band cells
    across the step keep their own surface's wettability
    (:class:`~src.config.chemical_step.ChemicalStepWall`).

    ``held_at_step`` marks a line at the edge that is receding onto the more
    wetting surface: its probes ``edge_width`` either side straddle the step,
    with the less wetting surface ahead of it (outward: ``-tangential`` for the
    left line, ``+tangential`` for the right). The operator pins it to the step
    position and saturates it toward the post receding angle
    (:func:`_step_saturation`). A line advancing onto the more wetting surface is
    never held; it spreads.
    """
    csc = setup.config.chemical_step_config
    wall = setup.config.chemical_step_wall
    assert csc is not None  # noqa: S101 - a chemical-step run has a chemical_step_config
    assert wall is not None  # noqa: S101 - and a wetting config, so a stepped wall
    step_x, delta = wall.step_x, csc["edge_width"]
    ahead, behind = (cll - delta, cll + delta) if side == "left" else (cll + delta, cll - delta)
    post = cll + delta >= step_x
    ca_adv = jnp.where(post, csc["ca_advancing_post_step"], csc["ca_advancing_pre_step"])
    ca_rec = jnp.where(post, csc["ca_receding_post_step"], csc["ca_receding_pre_step"])
    # Held: the surface behind the line (the one it would uncover) is post, the
    # one ahead is pre, and post wets more.
    post_wets_more = csc["ca_receding_post_step"] < csc["ca_advancing_pre_step"]
    held_at_step = (behind >= step_x) & ~(ahead >= step_x) & post_wets_more
    return ca_adv, ca_rec, held_at_step


# ── Top-level entry point ────────────────────────────────────────────


class _Side(NamedTuple):
    """One contact line's classification and optimiser settings for this step."""

    name: str
    index: int
    ca: jnp.ndarray
    cll: jnp.ndarray
    anchor: jnp.ndarray
    pin: jnp.ndarray
    ca_adv: jnp.ndarray
    ca_rec: jnp.ndarray
    moving: jnp.ndarray
    advancing: jnp.ndarray
    phi_active: jnp.ndarray
    max_iter: jnp.ndarray
    optimiser: _OptaxLike
    saturated: jnp.ndarray
    sat_phi: jnp.ndarray
    sat_d_rho: jnp.ndarray


def _update_wetting_state_impl(
    wetting: WettingState,
    rho_t_plus1: jnp.ndarray,
    setup: SimulationSetup,
    trial_step_fn: Callable[[WettingParams], tuple[jnp.ndarray, jnp.ndarray]],
    *,
    windows: tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
    measured: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    pins: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    step_sides: tuple[StepSide, StepSide] | None = None,
    t: jnp.ndarray | None = None,
) -> WettingState:
    """Shared implementation for hysteresis wetting updates.

    ``windows`` is ``((ca_adv, ca_rec) left, (ca_adv, ca_rec) right)``.
    ``pins`` override where a pinned side holds its contact line; by default
    that is its anchor. The chemical-step operator pins a line held at the step
    to the step itself. The anchors still decide which way a line is moving and
    are what the returned state carries.

    ``step_sides`` (chemical-step runs only) lets a side on the post surface, or
    held at the step, skip the optimiser while its angle is more than
    ``hysteresis_config["saturation_gap"]`` from its bound (:func:`_step_saturation`).

    ``measured`` is ``(ca_left, ca_right, cll_left, cll_right)`` of
    ``rho_t_plus1`` (:func:`_measure`). ``t`` is used only by the
    ``--debug-wetting`` trace.
    """
    rho_mean, w, edge = _wall_constants(setup)
    hc = setup.config.hysteresis_config
    if hc is None:
        msg = "hysteresis_config is required for hysteresis wetting update"
        raise TypeError(msg)
    loss_tol = float(hc["loss_tol"])
    is_bubble = detect_bubble(rho_t_plus1, rho_mean, edge=edge)
    optax = _import_optax()

    sides: list[_Side] = []
    for i, name in enumerate(_SIDES):
        ca, cll, anchor = measured[i], measured[2 + i], getattr(wetting, f"cll_{name}")
        ca_adv, ca_rec = windows[i]
        moving, advancing, expanding = _side_regime(ca, cll, anchor, ca_adv, ca_rec, side=name)
        lr, max_iter = _side_hyperparams(hc, advancing)
        if step_sides is None:
            saturated, sat_phi, sat_d_rho = jnp.array(False), _PHI_NEUTRAL, _D_RHO_NEUTRAL
        else:
            saturated, sat_phi, sat_d_rho = _step_saturation(
                step_sides[i],
                ca=ca,
                ca_adv=ca_adv,
                ca_rec=ca_rec,
                moving=moving,
                advancing=advancing,
                is_bubble=is_bubble,
                gap=float(hc["saturation_gap"]),
                w=w,
            )
        sides.append(
            _Side(
                name=name,
                index=i,
                ca=ca,
                cll=cll,
                anchor=anchor,
                pin=anchor if pins is None else pins[i],
                ca_adv=ca_adv,
                ca_rec=ca_rec,
                moving=moving,
                advancing=advancing,
                phi_active=_phi_is_active(~moving, advancing, expanding ^ is_bubble, is_bubble),
                max_iter=max_iter,
                # One optimiser per side: the learning rate is keyed on that
                # side's own urgency flag, so a pinned side is not dragged along.
                optimiser=optax.adam(lr),
                saturated=saturated,
                sat_phi=sat_phi,
                sat_d_rho=sat_d_rho,
            )
        )

    def objective(s: _Side) -> Callable[[WettingParams], jnp.ndarray]:
        def fn(p: WettingParams) -> jnp.ndarray:
            _, rho_out = trial_step_fn(p)
            ca_lr = compute_contact_angle(rho_out, rho_mean, edge=edge)
            cll_lr = compute_contact_line_location(rho_out, *ca_lr, rho_mean, edge=edge)
            return _side_cost(
                ca_lr[s.index],
                cll_lr[s.index],
                anchor=s.pin,
                moving=s.moving,
                advancing=s.advancing,
                ca_adv=s.ca_adv,
                ca_rec=s.ca_rec,
            )

        return fn

    def optimise(s: _Side, knob: str) -> Callable[[WettingParams], tuple[WettingParams, jnp.ndarray, jnp.ndarray]]:
        return lambda p: _optimise_single_param(
            objective(s), p, _grad_mask(f"{knob}_{s.name}"), s.optimiser, s.max_iter, w, loss_tol
        )

    def skip(s: _Side) -> Callable[[WettingParams], tuple[WettingParams, jnp.ndarray, jnp.ndarray]]:
        # Reports zero loss and zero iterations, so the d_rho fallback below —
        # gated on loss > loss_tol — never runs on a saturated side.
        zero_iters = jnp.zeros((), dtype=jnp.array(0).dtype)
        return lambda p: (_with_side(p, s.name, s.sat_phi, s.sat_d_rho), jnp.zeros(()), zero_iters)

    # Start from the stored knobs, the inactive one of each side snapped to neutral.
    params = wetting_params(wetting)
    for s in sides:
        params = _with_side(
            params,
            s.name,
            jnp.where(s.phi_active, getattr(params, f"phi_{s.name}"), _PHI_NEUTRAL),
            jnp.where(s.phi_active, _D_RHO_NEUTRAL, getattr(params, f"d_rho_{s.name}")),
        )

    def optimise_active(s: _Side) -> Callable[[WettingParams], tuple[WettingParams, jnp.ndarray, jnp.ndarray]]:
        return lambda p: jax.lax.cond(s.phi_active, optimise(s, "phi"), optimise(s, "d_rho"), p)

    losses, iters = [], []
    for s in sides:
        params, loss, n = jax.lax.cond(s.saturated, skip(s), optimise_active(s), params)
        losses.append(loss)
        iters.append(n)

    # Fallback: if the phi path was selected but phi saturated back at its
    # clamp floor without converging, phi was the wrong knob for this side —
    # `jnp.clip` has zero gradient there. Retry with d_rho, warm-started from
    # the stored value. The `loss > loss_tol` conjunct keeps it from firing on a
    # side whose phi legitimately converged at ~1.0. Both branches carry the
    # iteration count for the `--debug-wetting` row; zero means none ran.
    iters_fallback = []
    for s in sides:

        def fallback(p: WettingParams, s: _Side = s) -> tuple[WettingParams, jnp.ndarray]:
            start = _with_side(p, s.name, _PHI_NEUTRAL, getattr(wetting, f"d_rho_{s.name}"))
            params_fb, _loss_fb, iters_fb = optimise(s, "d_rho")(start)
            return params_fb, iters_fb

        on_floor = getattr(params, f"phi_{s.name}") <= _PHI_NEUTRAL + _PHI_FLOOR_EPS
        params, n_fb = jax.lax.cond(
            s.phi_active & on_floor & (losses[s.index] > loss_tol),
            fallback,
            lambda p: (p, jnp.zeros_like(iters[0])),
            params,
        )
        iters_fallback.append(n_fb)

    final_params = params

    # Guarded because building the samples is not free — each objective call
    # costs a full trial step. `log_sides` only calls this thunk on a logged step.
    def _debug_sides() -> tuple[wetting_debug.SideDebugSample, wetting_debug.SideDebugSample]:
        left, right = (
            wetting_debug.SideDebugSample(
                ca=s.ca,
                ca_adv=jnp.asarray(s.ca_adv),
                ca_rec=jnp.asarray(s.ca_rec),
                cll=s.cll,
                regime=_regime_code(s.moving, s.advancing, s.saturated),
                phi=getattr(final_params, f"phi_{s.name}"),
                d_rho=getattr(final_params, f"d_rho_{s.name}"),
                phi_active=s.phi_active,
                loss=objective(s)(final_params),
                iters=iters[s.index],
                iters_cap=jnp.asarray(s.max_iter),
                iters_fallback=iters_fallback[s.index],
            )
            for s in sides
        )
        return left, right

    if wetting_debug.enabled():
        wetting_debug.log_sides(_debug_sides, phi_neutral=_PHI_NEUTRAL, t=t)

    return wetting._replace(
        **final_params._asdict(),
        ca_left=sides[0].ca,
        ca_right=sides[1].ca,
        cll_left=jnp.where(sides[0].moving, sides[0].cll, sides[0].anchor),
        cll_right=jnp.where(sides[1].moving, sides[1].cll, sides[1].anchor),
    )


def wetting_params(wetting: WettingState) -> WettingParams:
    """The four wetting knobs carried in *wetting*."""
    return WettingParams(**{name: getattr(wetting, name) for name in WettingParams._fields})


def _wall_constants(setup: SimulationSetup) -> tuple[jnp.ndarray, jnp.ndarray, str]:
    """``(rho_mean, interface_width, wetting_edge)`` of a hysteresis run."""
    if setup.multiphase_params is None:
        msg = "multiphase_params is required for hysteresis wetting update"
        raise TypeError(msg)
    if setup.wetting_edge is None:
        msg = "wetting_edge is required for hysteresis wetting update"
        raise TypeError(msg)
    mp = setup.multiphase_params
    return jnp.array(0.5 * (mp.rho_l + mp.rho_v)), jnp.array(float(mp.interface_width)), setup.wetting_edge


def _measure(rho: jnp.ndarray, setup: SimulationSetup) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``(ca_left, ca_right, cll_left, cll_right)`` of *rho* at the wetting wall."""
    rho_mean, _w, edge = _wall_constants(setup)
    ca_left, ca_right = compute_contact_angle(rho, rho_mean, edge=edge)
    cll_left, cll_right = compute_contact_line_location(rho, ca_left, ca_right, rho_mean, edge=edge)
    return ca_left, ca_right, cll_left, cll_right


@wetting_operator(name="hysteresis")
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
    own ``jax.lax.while_loop``, masking out the other side's parameters.
    A side is pinned to its anchor unless its angle has left the global
    ``[ca_receding, ca_advancing]`` window *and* its contact line is not moving
    the wrong way for that bound (see :func:`_side_regime`); a moving side
    targets its bound.

    Args:
        wetting: Current :class:`WettingState`.
        rho_t_plus1: Density field, shape ``(nx, ny, nz, 1, 1)``.
        setup: :class:`~setup.simulation_setup.SimulationSetup` (closed-over, not traced).
        trial_step_fn: ``(WettingParams) → (f_out, rho_out)``, a single
            multiphase physics pass with trial wetting parameters.
        t: Current timestep, for the ``--debug-wetting`` trace. Physics-inert.

    Returns:
        Updated :class:`WettingState`.
    """
    if setup.config.hysteresis_config is None:
        msg = "hysteresis_config is required for wetting state update"
        raise TypeError(msg)
    hc = setup.config.hysteresis_config
    window = (hc["ca_advancing"], hc["ca_receding"])
    return _update_wetting_state_impl(
        wetting,
        rho_t_plus1,
        setup,
        trial_step_fn,
        windows=(window, window),
        measured=_measure(rho_t_plus1, setup),
        t=t,
    )


@wetting_operator(name="chemical_step_hysteresis")
def update_wetting_state_chemical_step(
    wetting: WettingState,
    rho_t_plus1: jnp.ndarray,
    setup: SimulationSetup,
    *,
    trial_step_fn: Callable[[WettingParams], tuple[jnp.ndarray, jnp.ndarray]],
    t: jnp.ndarray | None = None,
) -> WettingState:
    """Hysteresis update where CA targets are determined per-side by chemical step position.

    Identical to :func:`update_wetting_state` except each side's
    ``(ca_advancing, ca_receding)`` is read from the surfaces either side of
    that contact line (:func:`_get_hysteresis_window_chemical_step`), and a
    line held at the step is pinned to the step itself rather than to its
    anchor: the trailing line of a droplet pulled over the step stays on the
    edge while the droplet stretches, until its angle falls to the post-step
    receding angle.
    """
    measured = _measure(rho_t_plus1, setup)
    wall = setup.config.chemical_step_wall
    assert wall is not None  # noqa: S101 - a chemical-step run has a wetting config and a step
    windows, pins, step_sides = [], [], []
    for i, name in enumerate(_SIDES):
        cll = measured[2 + i]
        ca_adv, ca_rec, held = _get_hysteresis_window_chemical_step(setup, cll, side=name)
        windows.append((ca_adv, ca_rec))
        pins.append(jnp.where(held, wall.step_x, getattr(wetting, f"cll_{name}")))
        step_sides.append(
            StepSide(cll >= wall.step_x, held, getattr(wall, f"pre_phi_{name}"), getattr(wall, f"pre_d_rho_{name}"))
        )
    return _update_wetting_state_impl(
        wetting,
        rho_t_plus1,
        setup,
        trial_step_fn,
        windows=(windows[0], windows[1]),
        measured=measured,
        pins=(pins[0], pins[1]),
        step_sides=(step_sides[0], step_sides[1]),
        t=t,
    )
