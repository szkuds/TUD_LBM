"""Fixed-width console tables for live debug traces.

Both debug traces — ``--debug-wetting``
(:mod:`src.simulation_io.analysis.wetting_debug`) and ``--debug-stability``
(:mod:`src.simulation_io.analysis.stability`) — print while the simulation is
running, which makes line width load-bearing: a row that wraps loses the vertical
alignment that lets a reader see *which* number changed between samples.

This module renders rows whose width is decided by the layout, not by the data.
Every cell is formatted and then padded **or truncated** to its column width, so a
row is always exactly ``sum(widths) + len(columns) - 1`` characters — an unexpected
magnitude cannot push the line over the terminal edge.

A :class:`DebugTable` carries two column layouts, a wide one and a narrower
fallback, and picks between them from the terminal width each time it prints a
header. Headers repeat every ``header_every`` rows so the column names stay
on-screen during a long run.

Output goes to ``sys.stdout`` as plain ASCII, deliberately not through
:mod:`logging`: the root formatter installed by
:meth:`src.simulation_io.save.SimulationIO._setup_logging` prefixes every record
with a ~30-character timestamp, which would eat the width budget. That same method
tees ``sys.stdout`` into ``<run_dir>/simulation.log``, so these rows are still
captured — and stay readable there because they carry no ANSI styling.
"""

from __future__ import annotations
import shutil
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import TextIO

#: Marks a cell whose rendered text did not fit its column.
_TRUNCATED = "~"

#: Terminal width assumed when the size cannot be determined (not a tty, no
#: ``COLUMNS``). Wide enough for the full layouts, so redirected output — the
#: ``simulation.log`` tee included — keeps every column.
_FALLBACK_SIZE = (120, 24)


@dataclass(frozen=True)
class Column:
    """One fixed-width column of a debug table.

    Attributes:
        key: Key looked up in the value mapping passed to :func:`render_row`.
        header: Column label; truncated like any other cell if too wide.
        width: Exact rendered width, in characters.
        render: Value formatter. Defaults to :func:`str`.
    """

    key: str
    header: str
    width: int
    render: Callable[[Any], str] = str


def fmt(spec: str) -> Callable[[Any], str]:
    """Return a formatter applying *spec*, e.g. ``fmt(".3e")``.

    Values arriving from a JAX host callback are 0-d numpy arrays; ``format``
    handles those the same as Python scalars.
    """

    def _render(value: Any) -> str:  # noqa: ANN401 - any formattable scalar
        return format(value, spec)

    return _render


def cell(text: str, width: int) -> str:
    """Right-align *text* in *width* characters, truncating if it overflows.

    An overlong cell keeps its leading characters and ends with ``~`` so a
    clipped value is never mistaken for a complete one.
    """
    if len(text) > width:
        return text[: width - 1] + _TRUNCATED if width > 1 else _TRUNCATED[:width]
    return text.rjust(width)


def layout_width(columns: Sequence[Column]) -> int:
    """Total rendered width of *columns*, including single-space separators."""
    if not columns:
        return 0
    return sum(column.width for column in columns) + len(columns) - 1


def render_header(columns: Sequence[Column]) -> str:
    """Render the header row for *columns*."""
    return " ".join(cell(column.header, column.width) for column in columns)


def render_row(columns: Sequence[Column], values: Mapping[str, Any]) -> str:
    """Render one data row.

    Args:
        columns: Layout to render into.
        values: Values keyed by :attr:`Column.key`. A missing key renders as
            ``-`` rather than raising — a debug trace should not be able to
            abort the run it is reporting on.

    Returns:
        A string of exactly ``layout_width(columns)`` characters.
    """
    cells = []
    for column in columns:
        if column.key not in values:
            cells.append(cell("-", column.width))
            continue
        cells.append(cell(column.render(values[column.key]), column.width))
    return " ".join(cells)


class DebugTable:
    """Stateful emitter that repeats a header and adapts to the terminal width.

    Args:
        full: Preferred column layout.
        compact: Narrower layout used when *full* does not fit the terminal.
            Defaults to *full* (no adaptation).
        header_every: Rows between header repeats. The header also precedes the
            very first row.
        stream: Output stream. Resolved lazily from ``sys.stdout`` when omitted,
            so the ``simulation.log`` tee installed after import is honoured.
    """

    def __init__(
        self,
        full: Sequence[Column],
        compact: Sequence[Column] | None = None,
        *,
        header_every: int = 20,
        stream: TextIO | None = None,
    ) -> None:
        self._full = tuple(full)
        self._compact = tuple(compact) if compact is not None else self._full
        self._header_every = header_every
        self._stream = stream
        self._rows = 0
        self._columns = self._full

    @property
    def columns(self) -> tuple[Column, ...]:
        """Layout used for the most recent emission."""
        return self._columns

    def reset(self) -> None:
        """Forget the row count, so the next :meth:`emit` reprints the header."""
        self._rows = 0

    def _select_layout(self) -> tuple[Column, ...]:
        available = shutil.get_terminal_size(fallback=_FALLBACK_SIZE).columns
        return self._full if layout_width(self._full) <= available else self._compact

    def _write(self, line: str) -> None:
        stream = self._stream if self._stream is not None else sys.stdout
        stream.write(line + "\n")
        stream.flush()

    def _maybe_header(self) -> None:
        if self._rows % self._header_every == 0:
            self._columns = self._select_layout()
            self._write(render_header(self._columns))

    def emit(self, values: Mapping[str, Any]) -> None:
        """Print one row, preceded by a header every ``header_every`` rows."""
        self._maybe_header()
        self._write(render_row(self._columns, values))
        self._rows += 1

    def emit_block(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Print several rows as one unit.

        The header is checked once for the whole block, so a group of related
        rows — the two contact-line sides of one timestep, say — is never split
        by a header.
        """
        if not rows:
            return
        self._maybe_header()
        for row in rows:
            self._write(render_row(self._columns, row))
        self._rows += len(rows)


__all__ = [
    "Column",
    "DebugTable",
    "cell",
    "fmt",
    "layout_width",
    "render_header",
    "render_row",
]
