"""Tests for the fixed-width debug table renderer.

The point of the module is that a debug row is always exactly one terminal
line, whatever the data does, so these tests pin width, truncation, header
cadence and layout selection.
"""

import io
import pytest
from src.simulation_io.analysis._debug_table import Column
from src.simulation_io.analysis._debug_table import DebugTable
from src.simulation_io.analysis._debug_table import cell
from src.simulation_io.analysis._debug_table import fmt
from src.simulation_io.analysis._debug_table import layout_width
from src.simulation_io.analysis._debug_table import render_header
from src.simulation_io.analysis._debug_table import render_row

FULL = (
    Column("t", "t", 6, fmt("d")),
    Column("x", "x", 8, fmt(".3f")),
    Column("tag", "tag", 5),
)
COMPACT = (FULL[0], FULL[2])


class TestCell:
    """Single-cell padding and truncation."""

    def test_pads_to_width(self):
        assert cell("ab", 5) == "   ab"

    def test_truncates_and_marks_overflow(self):
        assert cell("123456789", 5) == "1234~"

    def test_width_one_is_all_marker(self):
        assert cell("123", 1) == "~"


class TestRender:
    """Header and row rendering at a fixed layout width."""

    def test_row_width_matches_layout(self):
        row = render_row(FULL, {"t": 12, "x": 1.5, "tag": "ok"})
        assert len(row) == layout_width(FULL) == 6 + 8 + 5 + 2

    def test_header_width_matches_layout(self):
        assert len(render_header(FULL)) == layout_width(FULL)

    def test_oversized_value_cannot_widen_the_row(self):
        row = render_row(FULL, {"t": 10**20, "x": -1234567.891, "tag": "overlong"})
        assert len(row) == layout_width(FULL)
        assert "~" in row

    def test_missing_key_renders_as_dash(self):
        row = render_row(FULL, {"t": 1, "tag": "ok"})
        assert len(row) == layout_width(FULL)
        assert "-" in row

    def test_empty_layout_has_zero_width(self):
        assert layout_width(()) == 0


class TestDebugTable:
    """Header cadence, layout selection and stream handling."""

    @staticmethod
    def _table(**kwargs):
        stream = io.StringIO()
        return DebugTable(FULL, COMPACT, stream=stream, **kwargs), stream

    @staticmethod
    def _values(t):
        return {"t": t, "x": float(t), "tag": "ok"}

    def test_header_repeats_on_cadence(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "200")
        table, stream = self._table(header_every=3)
        for t in range(7):
            table.emit(self._values(t))
        lines = stream.getvalue().splitlines()
        header = render_header(FULL)
        assert [i for i, line in enumerate(lines) if line == header] == [0, 4, 8]

    def test_every_line_fits_the_layout(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "200")
        table, stream = self._table()
        table.emit(self._values(1))
        assert {len(line) for line in stream.getvalue().splitlines()} == {layout_width(FULL)}

    def test_narrow_terminal_selects_compact_layout(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", str(layout_width(FULL) - 1))
        table, stream = self._table()
        table.emit(self._values(1))
        assert table.columns == COMPACT
        assert {len(line) for line in stream.getvalue().splitlines()} == {layout_width(COMPACT)}

    def test_wide_terminal_selects_full_layout(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", str(layout_width(FULL)))
        table, _stream = self._table()
        table.emit(self._values(1))
        assert table.columns == FULL

    def test_block_is_not_split_by_a_header(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "200")
        table, stream = self._table(header_every=2)
        table.emit(self._values(0))
        table.emit(self._values(1))
        # The block starts exactly on a header boundary.
        table.emit_block([self._values(2), self._values(3)])
        lines = stream.getvalue().splitlines()
        header = render_header(FULL)
        # Header before the block, never between its two rows.
        assert [i for i, line in enumerate(lines) if line == header] == [0, 3]
        assert len(lines) == 6

    def test_empty_block_emits_nothing(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "200")
        table, stream = self._table()
        table.emit_block([])
        assert stream.getvalue() == ""

    def test_reset_reprints_the_header(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "200")
        table, stream = self._table(header_every=100)
        table.emit(self._values(0))
        table.reset()
        table.emit(self._values(1))
        assert stream.getvalue().count(render_header(FULL)) == 2

    def test_defaults_to_stdout_at_write_time(self, monkeypatch, capsys):
        # Resolved lazily so the simulation.log tee installed after import wins.
        monkeypatch.setenv("COLUMNS", "200")
        DebugTable(FULL).emit(self._values(0))
        assert render_header(FULL) in capsys.readouterr().out


@pytest.mark.parametrize("spec", ["d", ".3f", ".2e"])
def test_fmt_applies_spec(spec):
    assert fmt(spec)(1) == format(1, spec)
