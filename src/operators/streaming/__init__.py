"""Streaming operators — implementations of StreamingOperator protocol.

RECOMMENDED USAGE:
    from operators.streaming import build_streaming_fn
    from operators.protocols import StreamingOperator
    
    stream_op: StreamingOperator = build_streaming_fn("standard")
    f_streamed = stream_op(f, lattice)

The factory function (build_streaming_fn) is the stable public API.
Implementation modules (_streaming.py) are internal details.

For extending with your own streaming operator:
    1. Implement a function matching StreamingOperator protocol
    2. Register it with @streaming_operator(name="your_name")
    3. Access via build_streaming_fn("your_name")
"""

from __future__ import annotations

from operators.protocols import StreamingOperator
from operators.factory import build_operator

# ── Private: Import implementation module to trigger registry registration ──
from operators.streaming import _streaming as _stream_impl  # noqa: F401


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
