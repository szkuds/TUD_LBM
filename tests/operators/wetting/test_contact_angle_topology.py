"""Topology-awareness of the contact-angle / contact-line measurement.

A droplet (liquid dispersed in vapour) and a bubble (vapour dispersed in
liquid) have mirror-image density profiles along the wall. These tests pin the
two invariants that make the measurement work for both:

1. Both report the contact angle **through the liquid**.
2. ``left``/``right`` are **positional**, matching how
   :func:`~src.operators.wetting._wetting_modification._apply_wetting_modification`
   splits the interface — the two must agree or the hysteresis optimiser tunes
   one contact line and the applicator applies the result to the other.

Fields are analytic circular caps with a ``tanh`` interface, parameterised by
the liquid-measured contact angle, so the expected answer is known in closed
form for both topologies.
"""

from __future__ import annotations
import math
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.operators.wetting._contact_angle import compute_contact_angle
from src.operators.wetting._contact_line import compute_contact_line_location
from src.operators.wetting._interface_crossings import detect_bubble
from src.operators.wetting._wetting_modification import _apply_wetting_modification

NX, NY = 401, 200
RHO_L, RHO_V = 1.0, 0.33
RHO_MEAN = (RHO_L + RHO_V) / 2.0
WIDTH = 4.0
RADIUS = 60.0
CENTRE_X = 200.0

#: Curvature of a finite cap plus the two-row finite difference bias the
#: measurement by ~2°; the sign of that bias flips between topologies.
ANGLE_TOL = 3.0
#: The half-cell projection lands the contact line within a lattice unit.
CLL_TOL = 1.0

THETAS = [40.0, 60.0, 80.0, 100.0, 120.0, 140.0]


def _cap_rho_2d(theta_deg: float, *, bubble: bool) -> np.ndarray:
    """Circular cap on the bottom wall with liquid-measured contact angle *theta_deg*.

    For a bubble the vapour cap subtends ``180° − theta_deg``, so the same
    liquid angle produces the mirror-image profile.
    """
    x, y = np.meshgrid(np.arange(NX, dtype=float), np.arange(NY, dtype=float), indexing="ij")
    theta_cap = math.radians(180.0 - theta_deg if bubble else theta_deg)
    centre_y = -RADIUS * math.cos(theta_cap)
    dist = np.sqrt((x - CENTRE_X) ** 2 + (y - centre_y) ** 2)
    inside = 0.5 * (1.0 - np.tanh(2.0 * (dist - RADIUS) / WIDTH))
    if bubble:
        return RHO_L - (RHO_L - RHO_V) * inside
    return RHO_V + (RHO_L - RHO_V) * inside


def _cap_rho(theta_deg: float, *, bubble: bool) -> jnp.ndarray:
    """5D ``(nx, ny, 1, 1, 1)`` version of :func:`_cap_rho_2d`."""
    return jnp.asarray(_cap_rho_2d(theta_deg, bubble=bubble)[:, :, None, None, None])


def _analytic_contact_lines(theta_deg: float, *, bubble: bool) -> tuple[float, float]:
    """Exact tangential positions where the cap meets the wall."""
    theta_cap = math.radians(180.0 - theta_deg if bubble else theta_deg)
    half = RADIUS * math.sin(theta_cap)
    return CENTRE_X - half, CENTRE_X + half


def _place_on_edge(rho_2d: np.ndarray, edge: str) -> jnp.ndarray:
    """Move a bottom-wall field onto *edge* (inverse of ``to_canonical``)."""
    if edge == "bottom":
        arr = rho_2d
    elif edge == "top":
        arr = rho_2d[:, ::-1]
    elif edge == "left":
        arr = rho_2d.T
    elif edge == "right":
        arr = rho_2d.T[::-1, :]
    else:
        msg = f"unknown edge {edge!r}"
        raise ValueError(msg)
    return jnp.asarray(arr[:, :, None, None, None])


class TestTopologyDetection:
    """``detect_bubble`` distinguishes the dispersed phase from the wall row."""

    @pytest.mark.parametrize("bubble", [False, True])
    def test_detects_dispersed_phase(self, bubble):
        got = detect_bubble(_cap_rho(90.0, bubble=bubble), RHO_MEAN, edge="bottom")
        assert bool(got) is bubble

    @pytest.mark.parametrize("edge", ["bottom", "top", "left", "right"])
    def test_edge_generic(self, edge):
        rho = _place_on_edge(_cap_rho_2d(90.0, bubble=True), edge)
        assert bool(detect_bubble(rho, RHO_MEAN, edge=edge)) is True


class TestDimensionGuards:
    """Both measurements are 2D-only and say so."""

    def test_contact_angle_rejects_3d(self):
        rho = jnp.zeros((8, 8, 2, 1, 1))
        with pytest.raises(ValueError, match="only implemented in 2D"):
            compute_contact_angle(rho, RHO_MEAN, edge="bottom")

    def test_contact_line_rejects_3d(self):
        rho = jnp.zeros((8, 8, 2, 1, 1))
        zero = jnp.array(90.0)
        with pytest.raises(ValueError, match="only implemented in 2D"):
            compute_contact_line_location(rho, zero, zero, RHO_MEAN, edge="bottom")


class TestContactAngleThroughLiquid:
    """Both topologies report the angle measured through the liquid."""

    @pytest.mark.parametrize("theta", THETAS)
    @pytest.mark.parametrize("bubble", [False, True])
    def test_recovers_the_imposed_angle(self, theta, bubble):
        ca_left, ca_right = compute_contact_angle(_cap_rho(theta, bubble=bubble), RHO_MEAN, edge="bottom")
        assert float(ca_left) == pytest.approx(theta, abs=ANGLE_TOL)
        assert float(ca_right) == pytest.approx(theta, abs=ANGLE_TOL)

    @pytest.mark.parametrize("theta", [60.0, 120.0])
    def test_bubble_is_not_the_supplement_of_the_droplet(self, theta):
        """Guards the sign of the correction: a bubble at 60° is ~60°, not ~120°."""
        ca_bubble, _ = compute_contact_angle(_cap_rho(theta, bubble=True), RHO_MEAN, edge="bottom")
        assert abs(float(ca_bubble) - theta) < abs(float(ca_bubble) - (180.0 - theta))

    @pytest.mark.parametrize("edge", ["bottom", "top", "left", "right"])
    def test_bubble_angle_is_edge_generic(self, edge):
        rho = _place_on_edge(_cap_rho_2d(60.0, bubble=True), edge)
        ca_left, ca_right = compute_contact_angle(rho, RHO_MEAN, edge=edge)
        assert float(ca_left) == pytest.approx(60.0, abs=ANGLE_TOL)
        assert float(ca_right) == pytest.approx(60.0, abs=ANGLE_TOL)

    def test_jittable_for_a_bubble(self):
        fn = jax.jit(lambda r: compute_contact_angle(r, RHO_MEAN, edge="bottom"))
        ca_left, _ = fn(_cap_rho(60.0, bubble=True))
        assert float(ca_left) == pytest.approx(60.0, abs=ANGLE_TOL)


class TestContactLinePositional:
    """``cll_left < cll_right`` holds for a droplet and a bubble alike."""

    @pytest.mark.parametrize("theta", THETAS)
    @pytest.mark.parametrize("bubble", [False, True])
    def test_matches_the_analytic_contact_lines(self, theta, bubble):
        rho = _cap_rho(theta, bubble=bubble)
        ca_left, ca_right = compute_contact_angle(rho, RHO_MEAN, edge="bottom")
        cll_left, cll_right = compute_contact_line_location(rho, ca_left, ca_right, RHO_MEAN, edge="bottom")

        expected_left, expected_right = _analytic_contact_lines(theta, bubble=bubble)
        assert float(cll_left) < float(cll_right)
        assert float(cll_left) == pytest.approx(expected_left, abs=CLL_TOL)
        assert float(cll_right) == pytest.approx(expected_right, abs=CLL_TOL)


class TestMeasurementApplicatorAgreement:
    """The regression that motivated the change.

    ``_apply_wetting_modification`` splits the ghost row positionally. If the
    measurement labelled left/right by transition sign — as the droplet-only
    code did — then for a bubble the optimiser would solve for ``cll_right``
    and the applicator would apply ``phi_r`` at ``cll_left``.
    """

    @staticmethod
    def _responsive_region(ghost_row: jnp.ndarray, *, side: str) -> np.ndarray:
        """Indices whose ghost density changes when only *side*'s phi is perturbed."""
        neutral, offset = jnp.array(1.0), jnp.array(0.0)
        perturbed = jnp.array(1.2)
        args = (ghost_row, jnp.array(RHO_L), jnp.array(RHO_V))
        base = _apply_wetting_modification(*args, neutral, neutral, offset, offset)
        if side == "left":
            probed = _apply_wetting_modification(*args, perturbed, neutral, offset, offset)
        else:
            probed = _apply_wetting_modification(*args, neutral, perturbed, offset, offset)
        return np.nonzero(np.abs(np.asarray(probed - base)) > 1e-9)[0]

    @pytest.mark.parametrize("bubble", [False, True])
    def test_phi_sides_address_the_reported_contact_lines(self, bubble):
        rho_2d = _cap_rho_2d(60.0, bubble=bubble)
        rho = jnp.asarray(rho_2d[:, :, None, None, None])
        ca_left, ca_right = compute_contact_angle(rho, RHO_MEAN, edge="bottom")
        cll_left, cll_right = compute_contact_line_location(rho, ca_left, ca_right, RHO_MEAN, edge="bottom")

        ghost_row = jnp.asarray(rho_2d[:, 0])
        left_region = self._responsive_region(ghost_row, side="left")
        right_region = self._responsive_region(ghost_row, side="right")

        assert left_region.size > 0
        assert right_region.size > 0
        # Each phi acts on the interface band around the contact line the
        # measurement reports under the same label.
        assert left_region.mean() == pytest.approx(float(cll_left), abs=WIDTH)
        assert right_region.mean() == pytest.approx(float(cll_right), abs=WIDTH)
        assert left_region.max() < right_region.min()
