"""Host-callback-based I/O for TUD-LBM.

Provides utilities for saving simulation snapshots during a
``jax.lax.scan`` time loop **without breaking the JIT trace**.

Two strategies are supported:

1.  **Post-hoc slicing** (default in :func:`~runner.run.run`):
    ``lax.scan`` returns the full trajectory; snapshots are selected
    after the scan completes.  Simple but memory-heavy for long runs.

2.  **Debug callback** (:func:`save_snapshot_callback`):
    Uses ``jax.debug.callback`` (ordered) to write data to disk at
    specific steps, keeping device memory constant.  Suitable for
    production runs.

Usage (strategy 2)::

    from runner.io_callbacks import make_save_callback

    # Outside jit:
    io_handler = SimulationIO(...)
    should_save, do_save = make_save_callback(
        io_handler, save_interval=100, skip_interval=0,
    )

    # Inside scan body:
    def scan_body(state, t):
        new_state = step_fn(state)
        do_save(new_state, t)
        return new_state, None
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from tud_lbm.io import SimulationIO
    from tud_lbm.pipeline.state import State


def _state_to_numpy(state: State, fields: tuple | None = None, t: int | None = None) -> dict:
    """Convert a :class:`~state.state.State` pytree to a NumPy dict.

    Only includes non-``None`` array fields suitable for saving.
    Detects and raises on NaN values.

    Args:
        state: The state object to convert.
        fields: Optional tuple of field names to include. If None, includes all.
        t: Optional timestep for error messages.

    Raises:
        FloatingPointError: If NaN values are detected.
    """
    data = {
        k: np.asarray(v)
        for k, v in state._asdict().items()
        if v is not None and hasattr(v, "shape") and (fields is None or k in fields)
    }
    bad = [k for k, v in data.items() if np.isnan(v).any()]
    if bad:
        # Raising here causes the jax.debug.callback to fail
        # and the lax.scan / run(...) to abort at this timestep.
        # TODO: when a NaN is triggered it still needs to plot id that is enabled.
        msg = f"NaNs detected at t={t} in fields: {bad}"
        raise FloatingPointError(msg)
    return data


def make_save_callback(
    io_handler: SimulationIO,
    save_interval: int,
    skip_interval: int = 0,
    save_fields: tuple | None = None,
) -> Callable:
    """Build an I/O callback for use inside a ``lax.scan`` body.

    Returns a callable ``do_save(state, t)`` that can be placed in
    the scan body. Interval checking is performed on-device via
    ``jax.lax.cond``; callback only executes when conditions are met.

    Args:
        io_handler: A :class:`~util.io.SimulationIO` instance.
        save_interval: How often to save.
        skip_interval: Steps to skip before first save.
        save_fields: Optional tuple of field names to save.

    Returns:
        ``do_save(state, t)`` — no return value; writes to disk as a
        side effect.

    Example::

        do_save = make_save_callback(io_handler, save_interval=100)

        def scan_body(state, t):
            new_state = step_fn(state)
            do_save(new_state, t)
            return new_state, None
    """

    def _host_save(state: State, t: int) -> None:
        """Write state to disk; interval already checked on-device."""
        it = int(t)
        data = _state_to_numpy(state, fields=save_fields, t=it)
        io_handler.save_data_step(it, data)

    def do_save(state: State, t: int) -> None:
        jax.lax.cond(
            (t > skip_interval) & (t % save_interval == 0),
            lambda _s, _t: jax.debug.callback(_host_save, state, t, ordered=True),
            lambda _s, _t: None,
            state,
            t,
        )

    return do_save
