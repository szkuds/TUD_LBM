"""Tests for the ``--debug-wetting`` trace.

Covers what the trace promises: nothing at all when the flag is off, two
aligned single-line rows per logged timestep when it is on, and an interval
gate that skips both the printing and the (expensive) sample construction.
"""

import jax
import jax.numpy as jnp
import pytest
import src.config.config_overview as _flags
from src.simulation_io.analysis import wetting_debug
from src.simulation_io.analysis._debug_table import layout_width

PHI_NEUTRAL = jnp.array(1.0)


def _sample(*, phi=1.4, phi_active=True, d_rho=0.002, iters=37, cap=60, fallback=0, ca=112.345):
    return wetting_debug.SideDebugSample(
        ca=jnp.array(ca),
        ca_adv=jnp.array(110.0),
        ca_rec=jnp.array(70.0),
        cll=jnp.array(23.456),
        phi=jnp.array(phi),
        d_rho=jnp.array(d_rho),
        phi_active=jnp.array(phi_active),
        loss=jnp.array(1.23e-5),
        iters=jnp.array(iters),
        iters_cap=jnp.array(cap),
        iters_fallback=jnp.array(fallback),
    )


@pytest.fixture(autouse=True)
def _fresh_table():
    """Reset the module-level table so each test starts with a header."""
    wetting_debug._TABLE.reset()


@pytest.fixture
def debug_on(monkeypatch):
    """Enable the trace with a wide terminal and per-step logging."""
    monkeypatch.setattr(_flags, "DEBUG_FLAG_WETTING", True)
    monkeypatch.setattr(_flags, "DEBUG_WETTING_INTERVAL", 1)
    monkeypatch.setenv("COLUMNS", "200")


def _log(t, sides=None, calls=None):
    def build():
        if calls is not None:
            calls.append(1)
        return sides if sides is not None else (_sample(), _sample(phi_active=False))

    wetting_debug.log_sides(build, phi_neutral=PHI_NEUTRAL, t=t)


def test_flag_off_emits_nothing_and_never_builds_samples(monkeypatch, capsys):
    monkeypatch.setattr(_flags, "DEBUG_FLAG_WETTING", False)
    calls = []
    _log(jnp.array(0), calls=calls)
    assert capsys.readouterr().out == ""
    assert calls == []


@pytest.mark.usefixtures("debug_on")
def test_emits_one_header_and_two_single_line_rows(capsys):
    _log(jnp.array(1200))
    lines = capsys.readouterr().out.splitlines()

    assert len(lines) == 3  # header + left + right
    assert len({len(line) for line in lines}) == 1
    assert len(lines[0]) == layout_width(wetting_debug._FULL)
    assert lines[1].split()[:2] == ["1200", "L"]
    assert lines[2].split()[:2] == ["1200", "R"]


@pytest.mark.usefixtures("debug_on")
def test_row_reports_mode_window_and_iteration_counts(capsys):
    _log(jnp.array(10), sides=(_sample(iters=37, cap=60), _sample(phi_active=False, fallback=22)))
    left, right = capsys.readouterr().out.splitlines()[1:]

    assert "phi" in left  # phi moved off neutral -> phi drove the wall
    assert "37/60" in left
    assert left.rstrip().endswith("-")  # no fallback ran on this side
    assert "d_rho" in right
    assert right.rstrip().endswith("22")
    assert "[ 70.0,110.0]" in left


@pytest.mark.usefixtures("debug_on")
def test_selected_but_unmoved_phi_reports_the_fallback_mode(capsys):
    _log(jnp.array(0), sides=(_sample(phi=1.0), _sample(phi=1.0)))
    left = capsys.readouterr().out.splitlines()[1]
    assert " fb " in left


@pytest.mark.usefixtures("debug_on")
def test_narrow_terminal_still_fits_one_line(monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "80")
    _log(jnp.array(0))
    lines = capsys.readouterr().out.splitlines()
    assert all(len(line) <= 80 for line in lines)
    assert len(lines[0]) == layout_width(wetting_debug._COMPACT)


class TestInterval:
    """The DEBUG_WETTING_INTERVAL rate limit."""

    def test_off_interval_step_logs_nothing(self, monkeypatch, capsys):
        monkeypatch.setattr(_flags, "DEBUG_FLAG_WETTING", True)
        monkeypatch.setattr(_flags, "DEBUG_WETTING_INTERVAL", 5)
        monkeypatch.setenv("COLUMNS", "200")
        _log(jnp.array(3))
        assert capsys.readouterr().out == ""

    def test_on_interval_step_logs(self, monkeypatch, capsys):
        monkeypatch.setattr(_flags, "DEBUG_FLAG_WETTING", True)
        monkeypatch.setattr(_flags, "DEBUG_WETTING_INTERVAL", 5)
        monkeypatch.setenv("COLUMNS", "200")
        _log(jnp.array(5))
        assert len(capsys.readouterr().out.splitlines()) == 3

    def test_zero_interval_falls_back_to_every_step(self, monkeypatch):
        monkeypatch.setattr(_flags, "DEBUG_WETTING_INTERVAL", 0)
        assert wetting_debug.interval() == 1

    @pytest.mark.usefixtures("debug_on")
    def test_missing_timestep_logs_unconditionally(self, capsys):
        wetting_debug.log_sides(
            lambda: (_sample(), _sample()),
            phi_neutral=PHI_NEUTRAL,
        )
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 3
        assert lines[1].split()[0] == "0"


@pytest.mark.usefixtures("debug_on")
def test_survives_a_jit_trace(capsys):
    """The trace must work from inside the jitted scan body it lives in."""

    @jax.jit
    def step(t):
        _log(t)
        return t + 1

    assert int(step(jnp.array(7))) == 8
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[1] for line in lines[1:]] == ["L", "R"]
