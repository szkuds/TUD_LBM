"""Interface-localised wetting density modification.

Provides :func:`_apply_wetting_modification`, which adjusts ghost-cell
densities at the liquid-vapour interface so that the LBM-stencil
gradient "sees" the desired wetting boundary condition.

**The region is anchored on the contact lines, not on an absolute density
window.** The predecessor selected ghost cells by the fixed band
``0.05*rho_l + 0.95*rho_v <= rho < 0.95*rho_l + 0.05*rho_v`` and split it at its
index centroid. That leaves only ``0.05*(rho_l - rho_v)`` of headroom above the
band, and on a gravitating run the liquid at the wetting wall equilibrates
*below* the prescribed ``rho_l`` by more than that: a measured inclined bubble
run (``force_g = 5e-06`` at 60°, ``rho_l = 12.18``) developed a hydrostatic drop
of 0.76 across the domain against a headroom of 0.608, so bulk liquid entered the
band, the modified region grew from 54 to 158 of 200 wall cells, every bulk ghost
cell was clipped to ``0.95*rho_l``, and the run diverged. Mass was exactly
conserved and the domain-mean liquid density flat throughout — that is the steady
state, not a transient, so no absolute threshold can work. Inclination makes it
worse still: gravity stratifies the liquid *along* the wall too, spanning the
full bandwidth, so a *measured* global pair cannot separate interface from bulk
either.

Hence, the four clauses of :func:`wetting_regions`, each of which earns its place:

Measured phase references, shared by both contact lines
    ``hi``/``lo`` are measured over the *union* of the two anchor windows, so the
    hydrostatic offset cancels — variation inside that union is a few per cent of
    the contrast against ~8 % across the wall — while both sides are held to one
    pair of bounds.

    **Per-side references do not work, and the failure is a drift.** They were
    tried, and the two windows generally hold different amounts of liquid, so the
    two contact lines end up clipped to different ceilings. A bubble equilibrating
    on a top wall at ``centres = [[0.7, 1]]`` of a 201-cell domain had its right
    window truncated by the wall: it reached only 9.68 against the left window's
    11.39, on bulk liquid of 11.67, for ceilings of **9.20 against 10.82**. Applied
    to the *converged, mirror-symmetric* field an earlier build had produced
    (contact angles 108.18° / 108.18°), that already injects ``+4.76`` of density on
    the left against ``+4.09`` on the right — **15 % asymmetric** — which is an
    unbalanced tangential force at the wall with no restoring term. It is also a
    feedback loop: a depressed ceiling depresses the adjacent fluid, lowering the
    next step's window max. That run slid 32 cells, merged into the right wall and
    lost its right contact line, where the equivalent run before the change held
    station at 145.2 with 108.18° on both sides. The information needed to make the
    two sides agree is simply not present in one window — a flank with 20 cells of
    liquid cannot see bulk when the recovery length is longer than that — so the
    references must be shared rather than merely measured.

    The union of the two windows, rather than the whole row, keeps them local: a
    long wall's far end cannot set the band.
Window
    Hard-bounds the region at ``2 * R`` cells, so no failure mode can return the
    whole wall.
Contiguous run at the anchor
    Excludes disconnected in-band clusters elsewhere on the wall. In the run
    above a bulk cluster ``[0, 87]`` appeared while the real regions sat at
    ``[104, 133]`` and ``[162, 187]``.

    These two clauses, not the choice of bounds, are what contain the
    hydrostatic failure: replayed over that same inclined run, the *absolute*
    band plus this window and this run gate gives regions of 46/52/55/57/54
    cells where the unguarded absolute band gave 46/136/122/96/56. That is why
    sharing the references costs nothing here.
Contrast floor
    Rejects an anchor sitting in stratified bulk rather than on an interface. A
    spurious crossing in that run had a local contrast of 5.25 against a global
    12.17.

``rho_mean`` stays prescribed: it only has to *separate* the phases, never track
them, so it needs classifier-grade accuracy — the same argument
:mod:`~src.operators.force._gravity_masked` rests its ``_PHASE_BAND`` on.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import NamedTuple
import jax.numpy as jnp
from src.operators.wetting._interface_crossings import interface_crossings

if TYPE_CHECKING:
    from jax.typing import ArrayLike
    from src.config.chemical_step import ChemicalStepWall

#: Inset from each measured phase density, as a fraction of the measured contrast.
_BAND_FRAC = 0.05

#: An anchor whose window spans less than this fraction of the global contrast is
#: not on an interface — it is a crossing of stratified bulk. Its region is empty.
_MIN_CONTRAST_FRAC = 0.5

#: Anchor window half-width: ``max(_MIN_WINDOW, n // _WINDOW_DIVISOR)`` cells.
_MIN_WINDOW = 16
_WINDOW_DIVISOR = 8


class WettingRegion(NamedTuple):
    """One contact line's ghost-row cells and the density bounds they are held to.

    Selection and clipping come from this single object so they can never be
    built from different bounds. The bounds are a property of the *wall*, not of
    the side, so the two regions of one wall always carry the same pair — see the
    module docstring for why a per-side pair drives the inclusion sideways.

    Attributes:
        mask: Boolean, shape ``(n,)`` — the ghost-row cells this side modifies.
        anchor: Sub-cell tangential position of the contact line.
        rho_lower: Lower density bound, ``lo + _BAND_FRAC * span``.
        rho_upper: Upper density bound, ``hi - _BAND_FRAC * span``.
    """

    mask: jnp.ndarray
    anchor: jnp.ndarray
    rho_lower: jnp.ndarray
    rho_upper: jnp.ndarray


def anchor_window_half_width(n: int) -> int:
    """Half-width, in cells, of the window the local phase densities are measured over.

    Derived from the wall's own length rather than from ``interface_width``,
    which is only a tanh-initialisation hint for the Carnahan-Starling EOS and
    not a physical length.
    """
    return max(_MIN_WINDOW, n // _WINDOW_DIVISOR)


def _window_span(row: jnp.ndarray, in_window: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(lo, hi)`` of *row* over *in_window*.

    Fill outside the window with the row's own extrema rather than +/-inf: an
    empty window then yields a negative span, which the contrast floor rejects
    and which leaves every bound finite. jnp.where with a non-finite dead branch
    would poison the gradients the hysteresis optimiser takes through here.
    """
    hi = jnp.max(jnp.where(in_window, row, jnp.min(row)))
    lo = jnp.min(jnp.where(in_window, row, jnp.max(row)))
    return lo, hi


def _anchored_region(
    row: jnp.ndarray,
    anchor: jnp.ndarray,
    rho_lower: jnp.ndarray,
    rho_upper: jnp.ndarray,
    contrast_floor: ArrayLike,
) -> WettingRegion:
    """The in-band run of *row* containing *anchor*, bounded by the anchor window.

    *rho_lower*/*rho_upper* are the wall's shared band bounds; the window's own
    span is used only for the contrast floor, which asks whether *anchor* sits
    on an interface at all — a local question, unlike the bounds.
    """
    n = row.shape[0]
    indices = jnp.arange(n)
    in_window = jnp.abs(indices - anchor) <= anchor_window_half_width(n)

    lo, hi = _window_span(row, in_window)
    span = hi - lo

    band = in_window & (row >= rho_lower) & (row < rho_upper)

    # Contiguous-run label: two in-band cells share a run exactly when no
    # out-of-band cell lies between them, i.e. when this cumulative count agrees.
    centre = jnp.clip(jnp.round(anchor).astype(int), 0, n - 1)
    run_id = jnp.cumsum(~band)
    mask = band & band[centre] & (run_id == run_id[centre]) & (span >= contrast_floor)

    return WettingRegion(mask, anchor, rho_lower, rho_upper)


def wetting_regions(
    edge_slice: jnp.ndarray,
    rho_l: ArrayLike,
    rho_v: ArrayLike,
) -> tuple[WettingRegion, WettingRegion]:
    """Return ``(left, right)`` — the ghost-row cells the wetting BC modifies.

    This is the single definition of the modified region, shared by
    :func:`_apply_wetting_modification` and the contact-angle overlay.

    Both regions carry the **same** ``rho_lower``/``rho_upper``, measured over the
    union of the two anchor windows. With ``phi_left == phi_right`` the wall then
    injects the same density on both contact lines of a mirror-symmetric row,
    which is what keeps an equilibrated inclusion from drifting along the wall.

    **Left/right is positional**, and now by construction: the two anchors are
    the ``rho_mean`` crossings reported by
    :func:`~src.operators.wetting._interface_crossings.interface_crossings` in
    ascending tangential order — the same call, in the same order, that
    :func:`~src.operators.wetting._contact_angle.compute_contact_angle` and
    :func:`~src.operators.wetting._contact_line.compute_contact_line_location`
    measure from. ``phi_l`` addressing the contact line reported as ``cll_left``
    is therefore one computation rather than two conventions kept in lock-step.

    Args:
        edge_slice: Ghost-row densities, shape ``(n,)``.
        rho_l: Liquid density.
        rho_v: Vapour density.

    Note:
        A row with fewer than two crossings collapses both anchors onto index 0
        (``interface_crossings`` pads with ``fill_value=0`` to keep the shape
        static under ``jit``); the contrast floor then empties both regions and
        the modification is a no-op, leaving the row untouched.
    """
    liquid, vapour = jnp.asarray(rho_l), jnp.asarray(rho_v)
    rho_mean = 0.5 * (liquid + vapour)
    contrast_floor = _MIN_CONTRAST_FRAC * (liquid - vapour)
    x_left, x_right, _ = interface_crossings(edge_slice, rho_mean)

    # One pair of bounds for the wall, measured over both anchor windows together.
    indices = jnp.arange(edge_slice.shape[0])
    half_width = anchor_window_half_width(edge_slice.shape[0])
    in_either = (jnp.abs(indices - x_left) <= half_width) | (jnp.abs(indices - x_right) <= half_width)
    lo, hi = _window_span(edge_slice, in_either)
    span = hi - lo
    rho_lower = lo + _BAND_FRAC * span
    rho_upper = hi - _BAND_FRAC * span

    left = _anchored_region(edge_slice, x_left, rho_lower, rho_upper, contrast_floor)
    right = _anchored_region(edge_slice, x_right, rho_lower, rho_upper, contrast_floor)

    # Two contact lines close enough to share one in-band run would otherwise
    # both claim the overlap. Assigning each cell to its nearer anchor keeps the
    # sides disjoint, and is a no-op once the anchors are well separated.
    nearer_left = jnp.abs(indices - x_left) <= jnp.abs(indices - x_right)
    return (
        left._replace(mask=left.mask & nearer_left),
        right._replace(mask=right.mask & ~nearer_left),
    )


def _surface_params(
    region: WettingRegion,
    phi: ArrayLike,
    d_rho: ArrayLike,
    step: ChemicalStepWall,
    *,
    pre_phi: float,
    pre_d_rho: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-cell ``(phi, d_rho)`` for one side's band on a stepped wall.

    Cells on the contact line's own side of the step take the line's live
    parameters; cells across the step keep their own surface's fixed values
    (:class:`~src.config.chemical_step.ChemicalStepWall`), so a line can never
    make the other surface wet like its own.
    """
    post_cell = jnp.arange(region.mask.shape[0]) >= step.step_x
    across = post_cell != (region.anchor >= step.step_x)
    fixed_phi = jnp.where(post_cell, step.post_phi, pre_phi)
    fixed_d_rho = jnp.where(post_cell, step.post_d_rho, pre_d_rho)
    return jnp.where(across, fixed_phi, phi), jnp.where(across, fixed_d_rho, d_rho)


def _apply_wetting_modification(
    edge_slice: jnp.ndarray,
    rho_l: ArrayLike,
    rho_v: ArrayLike,
    phi_l: ArrayLike,
    phi_r: ArrayLike,
    d_rho_l: ArrayLike,
    d_rho_r: ArrayLike,
    step: ChemicalStepWall | None = None,
) -> jnp.ndarray:
    """Apply wetting density modification at the liquid-vapour interface.

    Only modifies ghost cells inside one of the two contact-line regions of
    :func:`wetting_regions`, each receiving its own phi/d_rho and both clipped to
    the wall's shared, measured density bounds.

    Args:
        edge_slice: Ghost-row densities, shape ``(n,)``.
        rho_l: Liquid density.
        rho_v: Vapour density.
        phi_l: Wetting potential for the left contact line.
        phi_r: Wetting potential for the right contact line.
        d_rho_l: Density offset for the left contact line.
        d_rho_r: Density offset for the right contact line.
        step: The stepped wall, or ``None``. With one, band cells across the
            step from their contact line keep their surface's fixed values
            (:func:`_surface_params`).

    Returns:
        Modified edge slice.
    """
    left, right = wetting_regions(edge_slice, rho_l, rho_v)
    if step is not None:
        phi_l, d_rho_l = _surface_params(
            left, phi_l, d_rho_l, step, pre_phi=step.pre_phi_left, pre_d_rho=step.pre_d_rho_left
        )
        phi_r, d_rho_r = _surface_params(
            right, phi_r, d_rho_r, step, pre_phi=step.pre_phi_right, pre_d_rho=step.pre_d_rho_right
        )

    # Wetting modification: phi * rho - d_rho, clamped to the wall's shared bounds.
    modified_left = jnp.clip(phi_l * edge_slice - d_rho_l, left.rho_lower, left.rho_upper)
    modified_right = jnp.clip(phi_r * edge_slice - d_rho_r, right.rho_lower, right.rho_upper)

    result = jnp.where(right.mask, modified_right, edge_slice)
    return jnp.where(left.mask, modified_left, result)
