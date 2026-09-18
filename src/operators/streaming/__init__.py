"""Streaming operators — implementations of StreamingOperator protocol.

Public API: build_streaming_fn()

Implementation modules (_streaming.py) are internal; use the factory to access.

Example:
    from operators.streaming import build_streaming_fn

    stream_op = build_streaming_fn("standard")
    f_streamed = stream_op(f, lattice)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import cast
from src.operators._loader import auto_load_operators
from src.operators.factory import build_operator

if TYPE_CHECKING:
    from src.operators.protocols import StreamingOperator

# Auto-discover and import private operator modules for registry registration
auto_load_operators("src.operators.streaming")


def build_streaming_fn(
    scheme: str = "standard",
    bc_config: dict | None = None,
) -> StreamingOperator:
    """Return a streaming operator satisfying StreamingOperator protocol.

    Args:
        scheme: Streaming model name ("standard" or others).
                Defaults to "standard" (pull-style streaming).
        bc_config: Boundary-condition config used to determine which axes
            are non-periodic. ``None`` means fully periodic — no zero-fill.

    Returns:
        A callable ``(f, lattice) -> f_streamed`` satisfying StreamingOperator.
        The *bc_config* is bound in a thin closure so all call sites keep
        the two-argument signature ``streaming_fn(f, lattice)``.

        Type-checkers see this as a StreamingOperator, so:
            op: StreamingOperator = build_streaming_fn("standard")

        Type-checkers will verify any use of op matches the protocol.

    Raises:
        ValueError: If scheme is not registered.

    Examples:
        >>> from src.operators.streaming import build_streaming_fn
        >>> stream = build_streaming_fn("standard", bc_config={"top": "bounce-back"})
        >>> f_streamed = stream(f, lattice)
    """
    op = build_operator("stream", scheme)
    _bc = bc_config

    def _stream(
        f: object,
        lattice: object,
        bc_config: dict | None = None,  # noqa: ARG001
    ) -> object:
        return op(f, lattice, _bc)

    return cast("StreamingOperator", _stream)


__all__ = [
    "build_streaming_fn",  # ← Primary API (use this!)
]
