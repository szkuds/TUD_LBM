"""Merge rule of the ``ghost_fill`` slot: composed in feature (name) order."""

from __future__ import annotations
import jax.numpy as jnp
from src.features.contribution import Contribution
from src.features.contribution import compose_ghost_fill


def test_no_fill_gives_no_fill():
    assert compose_ghost_fill((("a", Contribution()), ("b", Contribution()))) is None


def test_a_single_fill_is_used_as_is():
    def fill(padded, ext):
        return padded + 1.0

    assert compose_ghost_fill((("a", Contribution(ghost_fill=fill)), ("b", Contribution()))) is fill


def test_fills_are_applied_in_feature_order():
    """Order matters for non-commuting fills: ``a`` then ``b`` is ``(x + 1) * 2``, not ``x * 2 + 1``."""
    features = (
        ("a", Contribution(ghost_fill=lambda padded, ext: padded + ext["shift"])),
        ("b", Contribution(ghost_fill=lambda padded, ext: padded * ext["scale"])),
    )
    composed = compose_ghost_fill(features)
    assert composed is not None

    out = composed(jnp.zeros(3), {"shift": 1.0, "scale": 2.0})

    assert out.tolist() == [2.0, 2.0, 2.0]
