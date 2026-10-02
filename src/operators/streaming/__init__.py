"""Streaming operators — implementations of StreamingOperator protocol.

Public API: build_streaming_fn()

Implementation modules (_streaming.py) are internal; use the factory to access.

Example:
    from src.operators.streaming import build_streaming_fn

    stream_op = build_streaming_fn(config.periodic_axes)
    f_streamed = stream_op(f, lattice)
"""

from __future__ import annotations
import functools
from typing import TYPE_CHECKING
from src.operators._loader import auto_load_operators
from src.registry import get_operators

if TYPE_CHECKING:
    from src.operators.protocols import StreamingOperator

# Auto-discover and import private operator modules for registry registration
auto_load_operators("src.operators.streaming")


def build_streaming_fn(periodic_axes: tuple[bool, bool], scheme: str = "standard") -> StreamingOperator:
    """Return a streaming operator satisfying StreamingOperator protocol.

    Args:
        periodic_axes: Per-axis periodicity ``(x, y)`` —
            :attr:`SimulationConfig.periodic_axes
            <src.config.simulation_config.SimulationConfig.periodic_axes>`.
        scheme: Streaming model name. Defaults to "standard" (pull-style streaming).

    Returns:
        A callable ``stream(f, lattice) -> f_streamed`` with *periodic_axes* bound.
    """
    return functools.partial(get_operators("stream")[scheme].target, periodic_axes=periodic_axes)


__all__ = [
    "build_streaming_fn",
]
