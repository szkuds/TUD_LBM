"""Merge rule of the ``ghost_fill`` slot: composed in addon (name) order, each fill on its own substate."""

from __future__ import annotations
import jax.numpy as jnp
from src.addons.hooks import Hooks
from src.addons.hooks import compose_ghost_fill


def test_no_fill_gives_no_fill():
    assert compose_ghost_fill((("a", Hooks()), ("b", Hooks()))) is None


def test_a_single_fill_receives_its_own_substate():
    def fill(padded, shift):
        return padded + shift

    composed = compose_ghost_fill((("a", Hooks(ghost_fill=fill)), ("b", Hooks())))
    assert composed is not None

    out = composed(jnp.zeros(3), {"a": 1.0, "b": 100.0})

    assert out.tolist() == [1.0, 1.0, 1.0]


def test_fills_are_applied_in_addon_order():
    """Order matters for non-commuting fills: ``a`` then ``b`` is ``(x + 1) * 2``, not ``x * 2 + 1``."""
    addons = (
        ("a", Hooks(ghost_fill=lambda padded, shift: padded + shift)),
        ("b", Hooks(ghost_fill=lambda padded, scale: padded * scale)),
    )
    composed = compose_ghost_fill(addons)
    assert composed is not None

    out = composed(jnp.zeros(3), {"a": 1.0, "b": 2.0})

    assert out.tolist() == [2.0, 2.0, 2.0]
