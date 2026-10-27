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

    from src.simulation_io.callbacks import make_save_callback

    # Outside jit:
    io_handler = SimulationIO(...)
    do_save = make_save_callback(
        io_handler, partial(observe, setup), save_interval=100, skip_interval=0,
    )

    # Inside scan body:
    def scan_body(state, _t):
        new_state = step_fn(state)
        do_save(new_state, new_state.t)
        return new_state, None
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from src.pipeline.state import State
    from src.pipeline.step import Observables
    from src.simulation_io.save import SimulationIO


def _state_to_numpy(
    state: State, observables: Observables, fields: tuple | None = None, t: int | None = None
) -> dict[str, np.ndarray]:
    """Convert a :class:`~src.pipeline.state.state.State` and its derived fields to a NumPy dict.

    A snapshot carries the carry (``f``, ``t``), the fields
    :func:`~src.pipeline.step.observe` derived from it (``rho``, ``u``,
    ``force``, ``force_ext``, ``pressure``) and the addon substates, under the
    keys they have always had. Only non-``None`` arrays are written, filtered by
    *fields*. Detects and raises on NaN values.

    Args:
        state: The state to convert.
        observables: ``observe(setup, state)``.
        fields: Optional tuple of field names to include. If None, includes all.
        t: Optional timestep for error messages.

    Raises:
        FloatingPointError: If NaN values are detected.
    """
    arrays = {"f": state.f, "t": state.t} | observables._asdict()
    data = {k: np.asarray(v) for k, v in arrays.items() if v is not None and (fields is None or k in fields)}

    # Addon substates are flattened into top-level keys, so a snapshot carries the
    # same names it always has (``ca_left``, ``phi_left``, ``h``, ...). Scalar
    # substate entries (the wetting angles, contact lines and wall parameters) are
    # ALWAYS saved: analysis plots read them, and they cost a few bytes. Field-sized
    # entries (the electric ``h``) follow ``fields`` like the core fields do.
    for substate in state.ext.values():
        for key, value in substate._asdict().items():
            array = np.asarray(value)
            if array.ndim == 0 or fields is None or key in fields:
                data[key] = array

    bad = [k for k, v in data.items() if np.isnan(v).any()]
    if bad:
        msg = f"NaNs detected at t={t} in fields: {bad}; aborting simulation"
        raise FloatingPointError(msg)
    return data


def make_save_callback(
    io_handler: SimulationIO,
    observe_fn: Callable[[State], Observables],
    save_interval: int,
    skip_interval: int = 0,
    save_fields: tuple | None = None,
) -> Callable:
    """Build an I/O callback for use inside a ``lax.scan`` body.

    Returns a callable ``do_save(state, t)`` that can be placed in
    the scan body. Interval checking is performed on-device via
    ``jax.lax.cond``; callback only executes when conditions are met.

    Args:
        io_handler: A :class:`~util.simulation_io.SimulationIO` instance.
        observe_fn: ``partial(observe, setup)``; derives the saved fields from
            the state, on device and only on save steps.
        save_interval: How often to save.
        skip_interval: Steps to skip before first save.
        save_fields: Optional tuple of field names to save.

    Returns:
        ``do_save(state, t)`` — no return value; writes to disk as a
        side effect.

    Example::

        do_save = make_save_callback(io_handler, partial(observe, setup), save_interval=100)

        def scan_body(state, _t):
            new_state = step_fn(state)
            do_save(new_state, new_state.t)
            return new_state, None
    """

    def _host_save(state: State, observables: Observables, t: int) -> None:
        """Write state to disk; interval already checked on-device."""
        it = int(t)
        data = _state_to_numpy(state, observables, fields=save_fields, t=it)
        io_handler.save_data_step(it, data)

    def _save(state: State, t: int) -> None:
        jax.debug.callback(_host_save, state, observe_fn(state), t, ordered=True)

    def do_save(state: State, t: int) -> None:
        jax.lax.cond(
            (t > skip_interval) & (t % save_interval == 0),
            _save,
            lambda _s, _t: None,
            state,
            t,
        )

    return do_save
