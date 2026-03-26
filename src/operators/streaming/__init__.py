"""Streaming operators — implementations of StreamingOperator protocol.

Public API: build_streaming_fn()

Implementation modules (_streaming.py) are internal; use the factory to access.

Example:
    from operators.streaming import build_streaming_fn
    
    stream_op = build_streaming_fn("standard")
    f_streamed = stream_op(f, lattice)
"""

from __future__ import annotations

from operators.protocols import StreamingOperator
from operators.factory import build_operator
from operators._loader import auto_load_operators

# Auto-discover and import private operator modules for registry registration
auto_load_operators('operators.streaming')


def build_streaming_fn(scheme: str = "standard") -> StreamingOperator:
    """Return a streaming operator satisfying StreamingOperator protocol.

    Args:
        scheme: Streaming model name ("standard" or others).
                Defaults to "standard" (pull-style streaming).

    Returns:
        A callable satisfying the StreamingOperator protocol.
        Can be called as: operator(f, lattice) → f_streamed
        
        Type-checkers see this as a StreamingOperator, so:
            op: StreamingOperator = build_streaming_fn("standard")
        
        Type-checkers will verify any use of op matches the protocol.

    Raises:
        ValueError: If scheme is not registered.
        
    Examples:
        >>> from operators.streaming import build_streaming_fn
        >>> stream = build_streaming_fn("standard")
        >>> f_streamed = stream(f, lattice)
    """
    return build_operator("stream", scheme)


__all__ = [
    "build_streaming_fn",  # ← Primary API (use this!)
]
