"""Stencil padding: a periodic axis must actually be periodic.

The predecessor padded the two sides of an axis in sequence, so the start-side
``jnp.pad`` ran against an array already extended on the end side. On a fully
periodic axis that made the start ghost a copy of row 0 rather than row ``n-1``,
leaving a zero-gradient seam pinned to the array origin. It broke translation
invariance along the axis — on a 201-cell periodic wall a bubble held its
centroid exactly while its own 40-cell roll drifted 0.85 cells in 2000 steps —
and a periodic domain has no restoring force for translation, so the seam force
integrated without bound.

The existing gradient/Laplacian tests could not see it: they use a constant
field, or deliberately inspect only the interior "to avoid periodic wrap
artefacts". Translation covariance is the invariant that catches it.
"""

from __future__ import annotations
import subprocess
import sys
import jax.numpy as jnp
import numpy as np
import pytest
from src.config.boundary_edges import pad_modes
from src.lattice.lattice import build_lattice
from src.operators.differential._gradient import compute_gradient
from src.operators.differential._laplacian import compute_laplacian
from src.operators.differential._pad_utils import _apply_stencil_padding

_NX, _NY = 13, 11

#: ``(top, bottom, right, left)`` — the order ``pad_modes`` returns.
_ALL_PERIODIC = ("wrap", "wrap", "wrap", "wrap")
_PERIODIC_X = ("edge", "edge", "wrap", "wrap")
_ALL_WALLED = ("edge", "edge", "edge", "edge")


@pytest.fixture(scope="module")
def field():
    return jnp.asarray(np.random.default_rng(20260922).random((_NX, _NY)))


@pytest.fixture(scope="module")
def lattice():
    return build_lattice("D2Q9")


def _as_5d(grid_2d):
    return grid_2d[:, :, None, None, None]


@pytest.mark.parametrize("pad_mode", [_ALL_PERIODIC, _PERIODIC_X], ids=["all-periodic", "periodic-x"])
def test_a_periodic_axis_pads_with_its_true_wrap_partners(field, pad_mode):
    padded = _apply_stencil_padding(field, pad_mode)

    np.testing.assert_array_equal(np.asarray(padded[0, 1:-1]), np.asarray(field[-1]))
    np.testing.assert_array_equal(np.asarray(padded[-1, 1:-1]), np.asarray(field[0]))


def test_a_periodic_y_axis_pads_with_its_true_wrap_partners(field):
    padded = _apply_stencil_padding(field, _ALL_PERIODIC)

    np.testing.assert_array_equal(np.asarray(padded[1:-1, 0]), np.asarray(field[:, -1]))
    np.testing.assert_array_equal(np.asarray(padded[1:-1, -1]), np.asarray(field[:, 0]))


def test_walled_padding_is_the_plain_edge_pad(field):
    """The fix must be confined to wrap axes; ``edge`` behaviour is unchanged."""
    np.testing.assert_array_equal(
        np.asarray(_apply_stencil_padding(field, _ALL_WALLED)),
        np.asarray(jnp.pad(field, ((1, 1), (1, 1)), mode="edge")),
    )


@pytest.mark.parametrize("pad_mode", [_ALL_PERIODIC, _PERIODIC_X], ids=["all-periodic", "periodic-x"])
@pytest.mark.parametrize("shift", [1, 5])
def test_the_padded_interior_follows_a_roll_along_a_periodic_axis(field, pad_mode, shift):
    """Covariance is stated on the interior: a roll moves the ghost layers too."""
    rolled = _apply_stencil_padding(jnp.roll(field, shift, axis=0), pad_mode)

    np.testing.assert_array_equal(
        np.asarray(jnp.roll(_apply_stencil_padding(field, pad_mode)[1:-1], shift, axis=0)),
        np.asarray(rolled[1:-1]),
    )


@pytest.mark.parametrize("pad_mode", [_ALL_PERIODIC, _PERIODIC_X], ids=["all-periodic", "periodic-x"])
def test_gradient_commutes_with_a_roll_along_a_periodic_axis(field, lattice, pad_mode):
    """Translation covariance — the invariant the seam broke.

    A seam pinned to the array origin makes the gradient depend on *where* in
    the array a feature sits, which in a periodic domain is a spurious force.
    """
    shift = 5
    grid = _as_5d(field)
    direct = compute_gradient(_as_5d(jnp.roll(field, shift, axis=0)), lattice.w, lattice.c, pad_mode)

    np.testing.assert_array_equal(
        np.asarray(jnp.roll(compute_gradient(grid, lattice.w, lattice.c, pad_mode), shift, axis=0)),
        np.asarray(direct),
    )


@pytest.mark.parametrize("pad_mode", [_ALL_PERIODIC, _PERIODIC_X], ids=["all-periodic", "periodic-x"])
def test_laplacian_commutes_with_a_roll_along_a_periodic_axis(field, lattice, pad_mode):
    shift = 5
    grid = _as_5d(field)
    direct = compute_laplacian(_as_5d(jnp.roll(field, shift, axis=0)), lattice.w, pad_mode)

    np.testing.assert_array_equal(
        np.asarray(jnp.roll(compute_laplacian(grid, lattice.w, pad_mode), shift, axis=0)),
        np.asarray(direct),
    )


def test_pad_modes_resolve_without_the_caller_importing_the_boundary_package():
    """The per-edge fallback is ``edge``, so an empty registry pads silently wrong.

    ``simulation_io.analysis.wetting_overlay`` reaches ``pad_modes``
    without importing ``src.operators.boundary``, and would otherwise draw a
    contact-angle band built from padding the solver never used. A subprocess is
    the only honest check — in-process another test may already have populated
    the registry.
    """
    source = (
        "from src.config.boundary_edges import pad_modes;"
        "print(pad_modes({'left': 'periodic', 'right': 'periodic',"
        " 'top': 'wetting', 'bottom': 'bounce-back'}))"
    )
    result = subprocess.run(  # noqa: S603 - the argv is this module's own literal, not input
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "('edge', 'edge', 'wrap', 'wrap')"


def test_pad_modes_are_returned_in_the_order_the_padding_expects(field):
    """``(top, bottom, right, left)`` — wrapping x must pad axis 0, not axis 1."""
    modes = pad_modes({"left": "periodic", "right": "periodic", "top": "symmetry", "bottom": "bounce-back"})
    padded = _apply_stencil_padding(field, modes)

    np.testing.assert_array_equal(np.asarray(padded[0, 1:-1]), np.asarray(field[-1]))
    np.testing.assert_array_equal(np.asarray(padded[1:-1, 0]), np.asarray(field[:, 0]))
