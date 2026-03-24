"""Collision operators — pure functions."""

# Import modules to trigger registration with the global registry
from operators.collision import bgk as _bgk_mod  # noqa: F401
from operators.collision import mrt as _mrt_mod  # noqa: F401
from operators.collision.bgk import collide_bgk
from operators.collision.factory import build_collision_fn
from operators.collision.mrt import collide_mrt

__all__ = [
    "build_collision_fn",
    "collide_bgk",
    "collide_mrt",
]
