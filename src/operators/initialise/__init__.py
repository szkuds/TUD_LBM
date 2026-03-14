"""Initialisation operators — pure functions.

Provides JAX-compatible pure functions that produce initial population
distributions ``f`` for various simulation scenarios.

Usage::

    from operators.initialise.factory import get_init_fn
    from setup.lattice import build_lattice

    lattice = build_lattice("D2Q9")
    init_fn = get_init_fn("standard")
    f = init_fn(64, 64, lattice, density=1.0)
"""

from operators.initialise.factory import get_init_fn

__all__ = ["get_init_fn"]
