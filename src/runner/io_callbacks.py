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
from collections.abc import Callable
import jax
import jax.numpy as jnp
import numpy as np


def _state_to_numpy(state) -> dict:
    """Convert a :class:`~state.state.State` pytree to a NumPy dict.

    Only includes non-``None`` array fields suitable for saving.
    """
    mapping = {}
    if state.f is not None:
        mapping["f"] = np.array(state.f)
    if state.rho is not None:
        mapping["rho"] = np.array(state.rho)
    if state.u is not None:
        mapping["u"] = np.array(state.u)
    if state.force is not None:
        mapping["force"] = np.array(state.force)
    if state.force_ext is not None:
        mapping["force_ext"] = np.array(state.force_ext)
    if state.h is not None:
        mapping["h"] = np.array(state.h)
    return mapping


def save_snapshot_callback(
    io_handler,
    state,
    t: jnp.ndarray,
    save_interval: int,
    skip_interval: int = 0,
    save_fields: tuple | None = None,
) -> None:
    """Write a snapshot to disk (runs on host, not inside XLA).

    This function is meant to be called via ``jax.debug.callback``
    from inside a ``lax.scan`` body.

    Args:
        io_handler: A :class:`~util.io.SimulationIO` instance.
        state: Current :class:`~state.state.State`.
        t: Current timestep (scalar JAX array → Python int inside callback).
        save_interval: How often to save.
        skip_interval: Steps to skip before first save.
        save_fields: Optional tuple of field names to save.
    """
    it = int(t)
    if it <= skip_interval:
        return
    if it % save_interval != 0:
        return

    data = _state_to_numpy(state)

    # Filter fields if requested
    if save_fields is not None:
        data = {k: v for k, v in data.items() if k in save_fields}

    # TODO: when a NaN is triggered it still needs to plot id that is enabled.
    # TODO: The error which this return is not yet clear since it is too long.
    # NaN check before writing
    bad = []
    for name, arr in data.items():
        if np.isnan(arr).any():
            bad.append(name)

    if bad:
        # Raising here causes the jax.debug.callback to fail
        # and the lax.scan / run(...) to abort at this timestep.
        raise FloatingPointError(f"NaNs detected at t={it} in fields: {bad}")

    io_handler.save_data_step(it, data)


def make_save_callback(
    io_handler,
    save_interval: int,
    skip_interval: int = 0,
    save_fields: tuple | None = None,
) -> Callable:
    """Build an I/O callback for use inside a ``lax.scan`` body.

    Returns a callable ``do_save(state, t)`` that can be placed in
    the scan body.  It uses ``jax.debug.callback`` (ordered) so it
    doesn't interfere with XLA compilation.

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

    def _host_save(state, t):
        save_snapshot_callback(
            io_handler,
            state,
            t,
            save_interval=save_interval,
            skip_interval=skip_interval,
            save_fields=save_fields,
        )

    def do_save(state, t):
        jax.debug.callback(_host_save, state, t, ordered=True)

    return do_save
