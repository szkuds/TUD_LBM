"""Stability diagnostics for the ``lax.scan`` time loop.

Enabled via the ``--debug-stability`` CLI flag, which sets
``DEBUG_FLAG_STABILITY`` in :mod:`src.config.config_overview`.
Every ``DEBUG_STABILITY_INTERVAL`` steps (``--debug-stability [INTERVAL]``,
independent of ``save_interval``) the runner samples a small vector of
on-device metrics and ships it to a host callback:

* ``max|u|`` — maximum velocity magnitude
* ``max|grad mu|`` — maximum chemical-potential gradient magnitude,
  derived from the stored interaction force (``grad mu = -F_int / rho``)
* ``min rho`` / ``max rho`` — density range
* checkerboard amplitude — L2 norm of ``rho - 3x3-smoothed rho``
  restricted to the droplet wake (vapor phase, interface excluded)

The host callback appends one row per sample to ``stability_log.csv``
(flushed per write, so the curve survives a crash), prints a fixed-width
status row via
:class:`~src.simulation_io.analysis._debug_table.DebugTable` — one
terminal line, columns aligned across samples — and raises
:class:`StabilityAbortError` when any metric is NaN.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
import numpy as np
from src.simulation_io.analysis._debug_table import Column
from src.simulation_io.analysis._debug_table import DebugTable
from src.simulation_io.analysis._debug_table import fmt

if TYPE_CHECKING:
    from collections.abc import Callable
    from src.config.multiphase_params import MultiphaseParams
    from src.pipeline.state.state import State
    from src.pipeline.step import Observables
    from src.protocols import ExtDifferentialOperator

logger = logging.getLogger(__name__)

_CSV_NAME = "stability_log.csv"
_CSV_HEADER = "t,max_u,max_grad_mu,rho_min,rho_max,checkerboard_amp,n_wake_cells\n"
_METRIC_NAMES = ("max_u", "max_grad_mu", "rho_min", "rho_max", "checkerboard_amp")

#: One line per sample, so consecutive samples stay column-aligned and a
#: drifting metric is visible in place.
_TABLE = DebugTable(
    (
        Column("t", "t", 8, fmt("d")),
        Column("max_u", "max|u|", 8, fmt(".2e")),
        Column("max_grad_mu", "max|dmu|", 8, fmt(".2e")),
        Column("rho_min", "rho_min", 8, fmt(".5f")),
        Column("rho_max", "rho_max", 8, fmt(".5f")),
        Column("checkerboard_amp", "cb", 8, fmt(".2e")),
        Column("n_wake_cells", "n_wake", 8, fmt("d")),
    )
)


class StabilityAbortError(FloatingPointError):
    """Raised by the stability NaN guard; aborts the scan."""


def checkerboard_amplitude(rho: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """L2 norm of ``rho`` minus its 3x3 xy-uniform-smoothed version, on *mask*.

    The smoothing is a separable 3-point mean along the x and y axes
    (z untouched — for 2D simulations nz=1). Rolls wrap at domain
    edges; the wake mask excludes wall/interface layers where that
    matters.

    Args:
        rho: Density field, shape ``(nx, ny, nz, 1, 1)``.
        mask: Boolean mask, broadcastable to ``rho``.

    Returns:
        Scalar checkerboard amplitude.
    """
    sx = (jnp.roll(rho, 1, axis=0) + rho + jnp.roll(rho, -1, axis=0)) / 3.0
    smooth = (jnp.roll(sx, 1, axis=1) + sx + jnp.roll(sx, -1, axis=1)) / 3.0
    return jnp.sqrt(jnp.sum(jnp.where(mask, (rho - smooth) ** 2, 0.0)))


def wake_mask(
    rho: jnp.ndarray,
    grad_rho: jnp.ndarray,
    rho_l: float,
    rho_v: float,
    vapor_frac: float,
    grad_frac: float,
) -> jnp.ndarray:
    """Boolean mask for the droplet wake: vapor phase, interface excluded.

    Recomputed at every sample, so the region follows the droplet.
    A cell is in the wake when its density is close to the vapor
    density AND the local density gradient is flat (excluding the
    diffuse interface, whose steep but smooth variation would dominate
    the checkerboard residual).

    Args:
        rho: Density field, shape ``(nx, ny, nz, 1, 1)``.
        grad_rho: Density gradient, shape ``(nx, ny, nz, 1, d)``.
        rho_l: Liquid coexistence density.
        rho_v: Vapor coexistence density.
        vapor_frac: Vapor threshold — ``rho < rho_v + vapor_frac * (rho_l - rho_v)``.
        grad_frac: Interface exclusion — ``|grad rho| < grad_frac * (rho_l - rho_v)``.

    Returns:
        Boolean mask, shape ``(nx, ny, nz, 1, 1)``.
    """
    delta = rho_l - rho_v
    grad_mag = jnp.sqrt(jnp.sum(grad_rho**2, axis=-1, keepdims=True))
    vapor = rho < rho_v + vapor_frac * delta
    flat = grad_mag < grad_frac * delta
    return vapor & flat


def compute_stability_metrics(
    state: State,
    observables: Observables,
    *,
    gradient_density: ExtDifferentialOperator | None = None,
    mp: MultiphaseParams | None = None,
    vapor_frac: float = 0.2,
    grad_frac: float = 0.05,
) -> jnp.ndarray:
    """Compute the stability-metric vector for one state.

    ``max|grad mu|`` is read off the observed force rather than recomputed:
    for a multiphase run ``force = -rho*grad(mu) + force_ext``, with
    ``force_ext`` observed separately, so ``grad mu = -(force - force_ext) /
    rho``. Single-phase states (``force is None``) report 0.

    Args:
        state: The state, for its ``ext`` (the density gradient's ghost fill).
        observables: ``observe(setup, state)``.
        gradient_density: ``setup.gradient_density``, for the wake mask (``None``: whole domain).
        mp: ``setup.multiphase_params`` (``None`` for single-phase: whole domain).
        vapor_frac: See :func:`wake_mask`.
        grad_frac: See :func:`wake_mask`.

    Without multiphase parameters the checkerboard mask falls back to
    the whole domain.

    Returns:
        Flat vector ``[max_u, max_grad_mu, rho_min, rho_max,
        checkerboard_amp, n_wake_cells]``.
    """
    rho, u, force, force_ext = observables.rho, observables.u, observables.force, observables.force_ext
    max_u = jnp.max(jnp.sqrt(jnp.sum(u**2, axis=-1)))

    if force is not None:
        f_int = force if force_ext is None else force - force_ext
        grad_mu = -f_int / jnp.maximum(rho, 1e-12)
        max_grad_mu = jnp.max(jnp.sqrt(jnp.sum(grad_mu**2, axis=-1)))
    else:
        max_grad_mu = jnp.zeros(())

    rho_min = jnp.min(rho)
    rho_max = jnp.max(rho)

    if mp is not None and gradient_density is not None:
        mask = wake_mask(rho, gradient_density(rho, state.ext), mp.rho_l, mp.rho_v, vapor_frac, grad_frac)
    else:
        mask = jnp.ones_like(rho, dtype=bool)

    cb_amp = checkerboard_amplitude(rho, mask)
    n_wake = jnp.sum(mask).astype(rho.dtype)

    return jnp.stack([max_u, max_grad_mu, rho_min, rho_max, cb_amp, n_wake])


def _host_check(out_dir: Path, metrics: np.ndarray, t: int) -> None:
    """Host-side sample handler: CSV row, console line, NaN guard.

    Raises:
        StabilityAbortError: If any metric is NaN. Note that inside a
            ``jax.debug.callback`` JAX may surface this wrapped in an
            ``XlaRuntimeError``; the message is logged first so the
            diagnosis survives the wrapping.
    """
    it = int(t)
    m = np.asarray(metrics, dtype=np.float64)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / _CSV_NAME
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8") as fh:
        if write_header:
            fh.write(_CSV_HEADER)
        fh.write(f"{it},{m[0]:.8e},{m[1]:.8e},{m[2]:.8e},{m[3]:.8e},{m[4]:.8e},{int(m[5])}\n")

    _TABLE.emit(
        {
            "t": it,
            "max_u": m[0],
            "max_grad_mu": m[1],
            "rho_min": m[2],
            "rho_max": m[3],
            "checkerboard_amp": m[4],
            "n_wake_cells": int(m[5]),
        }
    )

    nan_names = [name for name, value in zip(_METRIC_NAMES, m[:5], strict=True) if np.isnan(value)]
    if nan_names:
        msg = f"NaN in stability metrics at t={it}: {nan_names}; aborting simulation"
        logger.error("[stability] %s", msg)
        raise StabilityAbortError(msg)


def make_stability_callback(
    out_dir: str | Path,
    *,
    observe_fn: Callable[[State], Observables],
    gradient_density: ExtDifferentialOperator | None,
    mp: MultiphaseParams | None,
    log_interval: int,
    vapor_frac: float,
    grad_frac: float,
) -> Callable:
    """Build a stability-diagnostics callback for a ``lax.scan`` body.

    Mirrors :func:`~src.simulation_io.callbacks.make_save_callback`:
    interval checking runs on-device via ``jax.lax.cond`` and the
    metric reductions live inside the true branch, so off-interval
    steps cost one integer compare.

    Args:
        out_dir: Directory for ``stability_log.csv``.
        observe_fn: ``partial(observe, setup)``; evaluated on sample steps only.
        gradient_density: ``setup.gradient_density`` (``None`` for single-phase).
        mp: ``setup.multiphase_params`` (``None`` for single-phase).
        log_interval: Steps between samples (``DEBUG_STABILITY_INTERVAL``).
        vapor_frac: See :func:`wake_mask`.
        grad_frac: See :func:`wake_mask`.

    Returns:
        ``do_check(state, t)`` — no return value; logs as a side effect
        and aborts the scan on NaN.
    """
    out_path = Path(out_dir)
    # A fresh run reprints the column header rather than continuing the
    # cadence of the previous run in the same interpreter (sweeps, tests).
    _TABLE.reset()

    def _host(metrics: np.ndarray, t: int) -> None:
        _host_check(out_path, metrics, t)

    def _fire(state: State, t: jnp.ndarray) -> None:
        metrics = compute_stability_metrics(
            state,
            observe_fn(state),
            gradient_density=gradient_density,
            mp=mp,
            vapor_frac=vapor_frac,
            grad_frac=grad_frac,
        )
        jax.debug.callback(_host, metrics, t, ordered=True)

    def do_check(state: State, t: jnp.ndarray) -> None:
        jax.lax.cond(
            t % log_interval == 0,
            _fire,
            lambda _s, _t: None,
            state,
            t,
        )

    return do_check
