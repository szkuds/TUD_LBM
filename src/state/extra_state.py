"""Structural contract for operation-specific extra simulation state."""

from __future__ import annotations
from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class ExtraState(Protocol):
    """Marker protocol for JAX-pytree-compatible extra state containers.

    Implementations are intentionally unconstrained to support both
    parameter-style containers (e.g. wetting scalars) and distribution-style
    containers (e.g. electric potential populations).
    """
