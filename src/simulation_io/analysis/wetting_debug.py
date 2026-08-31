"""Debug logging for the hysteresis wetting optimisation.

Enabled via the ``--debug-wetting`` CLI flag, which sets
``DEBUG_FLAG_WETTING`` in :mod:`src.config.config_overview`.  When on,
:mod:`src.operators.wetting.hysteresis.hysteresis` emits one fixed-width
row per contact line per logged timestep: the measured CA against its
hysteresis window, the CLL, which parameter ended up driving the wall
(``mode``), the wetting parameters themselves, the residual loss, and how
many inner optimiser iterations were spent against the cap — which is how
you tell a converged solve from one that ran out of iterations.

Both rows are emitted through a single ordered host callback, so the two
sides of one timestep always appear together, and through
:class:`~src.simulation_io.analysis._debug_table.DebugTable`, so each row
is exactly one terminal line with columns that line up vertically across
timesteps.

``DEBUG_WETTING_INTERVAL`` (``--debug-wetting-interval``) rate-limits the
trace: the optimiser runs every timestep, but printing every timestep
scrolls faster than it can be read.  The gate is a ``lax.cond`` on the
timestep, and the sample values — two full trial steps' worth of loss
evaluation — are built *inside* it, so skipped steps cost nothing.

Both flags are read at call time, not import time, so the CLI can flip
them after the operator modules are already loaded.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
import src.config.config_overview as _flags
from src.simulation_io.analysis._debug_table import Column
from src.simulation_io.analysis._debug_table import DebugTable
from src.simulation_io.analysis._debug_table import fmt

if TYPE_CHECKING:
    from collections.abc import Callable

# mode codes used in the per-side row
_MODE_D_RHO = 0
_MODE_PHI = 1
_MODE_FALLBACK = 2
_MODE_NAMES = {_MODE_D_RHO: "d_rho", _MODE_PHI: "phi", _MODE_FALLBACK: "fb"}

#: Order in which a side's scalars are packed into the callback vector.
_FIELDS = (
    "ca",
    "ca_adv",
    "ca_rec",
    "cll",
    "phi",
    "d_rho",
    "loss",
    "mode",
    "iters",
    "iters_cap",
    "iters_fallback",
)


def _render_window(bounds: tuple[float, float]) -> str:
    rec, adv = bounds
    return f"[{rec:5.1f},{adv:5.1f}]"


def _render_iters(counts: tuple[int, int]) -> str:
    used, cap = counts
    return f"{used}/{cap}"


def _render_fallback(count: int) -> str:
    return "-" if count == 0 else str(count)


_T = Column("t", "t", 7, fmt("d"))
_SIDE = Column("side", "side", 4)
_CA = Column("ca", "CA", 7, fmt(".2f"))
_WINDOW = Column("window", "rec,adv", 13, _render_window)
_CLL = Column("cll", "CLL", 8, fmt(".3f"))
_MODE = Column("mode", "mode", 5)
_PHI = Column("phi", "phi", 8, fmt(".6f"))
_D_RHO = Column("d_rho", "d_rho", 9, fmt(".6f"))
_LOSS = Column("loss", "loss", 8, fmt(".2e"))
_ITERS = Column("iters", "iters", 6, _render_iters)
_FALLBACK = Column("fallback", "fb", 3, _render_fallback)

#: 88 characters.
_FULL = (_T, _SIDE, _CA, _WINDOW, _CLL, _MODE, _PHI, _D_RHO, _LOSS, _ITERS, _FALLBACK)
#: 70 characters — drops the (constant-per-region) window and the fallback count.
_COMPACT = (_T, _SIDE, _CA, _CLL, _MODE, _PHI, _D_RHO, _LOSS, _ITERS)

_TABLE = DebugTable(_FULL, _COMPACT)


@dataclass(frozen=True)
class SideDebugSample:
    """One side's post-optimisation state, as logged by :func:`log_sides`.

    Attributes:
        ca: Measured contact angle in degrees.
        ca_adv: Advancing bound of the active hysteresis window.
        ca_rec: Receding bound of the active hysteresis window.
        cll: Measured contact-line location.
        phi: Final ``phi`` wetting parameter.
        d_rho: Final ``d_rho`` wetting parameter.
        phi_active: Whether ``phi`` was the parameter selected for this side.
        loss: Objective value at the final parameters.
        iters: Inner optimiser iterations spent on this side.
        iters_cap: Iteration cap that applied to this side.
        iters_fallback: Iterations spent in the ``d_rho`` fallback solve,
            zero when the fallback did not run.
    """

    ca: jnp.ndarray
    ca_adv: jnp.ndarray
    ca_rec: jnp.ndarray
    cll: jnp.ndarray
    phi: jnp.ndarray
    d_rho: jnp.ndarray
    phi_active: jnp.ndarray
    loss: jnp.ndarray
    iters: jnp.ndarray
    iters_cap: jnp.ndarray
    iters_fallback: jnp.ndarray


def enabled() -> bool:
    """Return whether wetting debug logging is currently switched on.

    Read this instead of importing ``DEBUG_FLAG_WETTING`` directly:
    the CLI sets the flag as a module attribute after the operator
    packages have been imported, so a ``from ... import`` binding taken
    at import time would still be ``False``.
    """
    return _flags.DEBUG_FLAG_WETTING


def interval() -> int:
    """Return the number of timesteps between logged samples (at least 1)."""
    return max(1, int(_flags.DEBUG_WETTING_INTERVAL))


def _mode(sample: SideDebugSample, phi_neutral: jnp.ndarray) -> jnp.ndarray:
    """Classify which parameter drove the wall on this side.

    ``phi`` counts as engaged only once it has moved off its neutral
    value; a selected-but-unmoved ``phi`` means the ``d_rho`` fallback
    branch took over.
    """
    phi_engaged = sample.phi_active & (sample.phi > phi_neutral)
    fallback = sample.phi_active & ~phi_engaged
    return jnp.where(
        phi_engaged,
        jnp.array(_MODE_PHI),
        jnp.where(fallback, jnp.array(_MODE_FALLBACK), jnp.array(_MODE_D_RHO)),
    )


def _pack(sample: SideDebugSample, phi_neutral: jnp.ndarray) -> jnp.ndarray:
    """Pack one side into a single float vector for the host callback.

    One array per side keeps the callback to three arguments; the host
    unpacks it by :data:`_FIELDS`.  Integer-valued entries (mode, iteration
    counts) ride along as floats and are cast back host-side.
    """
    dtype = jnp.result_type(float)
    values = {
        "mode": _mode(sample, phi_neutral),
        **{name: getattr(sample, name) for name in _FIELDS if name != "mode"},
    }
    return jnp.stack([jnp.asarray(values[name], dtype=dtype) for name in _FIELDS])


def _row(side: str, t: np.ndarray, packed: np.ndarray) -> dict[str, Any]:
    """Build the table row for one side from the callback's arrays."""
    values = dict(zip(_FIELDS, np.asarray(packed, dtype=np.float64), strict=True))
    return {
        "t": int(t),
        "side": side,
        "ca": values["ca"],
        "window": (values["ca_rec"], values["ca_adv"]),
        "cll": values["cll"],
        "mode": _MODE_NAMES.get(int(values["mode"]), "?"),
        "phi": values["phi"],
        "d_rho": values["d_rho"],
        "loss": values["loss"],
        "iters": (int(values["iters"]), int(values["iters_cap"])),
        "fallback": int(values["iters_fallback"]),
    }


def _host_block(t: np.ndarray, left: np.ndarray, right: np.ndarray) -> None:
    """Emit both sides of one timestep as an unsplittable two-row block."""
    _TABLE.emit_block([_row("L", t, left), _row("R", t, right)])


def log_sides(
    build_sides: Callable[[], tuple[SideDebugSample, SideDebugSample]],
    *,
    phi_neutral: jnp.ndarray,
    t: jnp.ndarray | None = None,
) -> None:
    """Log both contact lines as one aligned two-row block.

    Args:
        build_sides: Thunk returning ``(left, right)`` samples.  It is a
            thunk rather than two values because building the samples costs
            two trial-step evaluations, and on a rate-limited step those
            must not run: the call happens inside the interval gate.
        phi_neutral: The neutral ``phi`` value, used to decide whether
            ``phi`` actually engaged (see :func:`_mode`).
        t: Current timestep, used for the interval gate and the ``t``
            column.  ``None`` logs unconditionally and prints ``t=0``.
    """
    if not enabled():
        return

    def _fire() -> None:
        left, right = build_sides()
        step = jnp.zeros((), dtype=int) if t is None else t
        jax.debug.callback(
            _host_block,
            step,
            _pack(left, phi_neutral),
            _pack(right, phi_neutral),
            ordered=True,
        )

    if t is None:
        _fire()
        return

    jax.lax.cond(t % interval() == 0, _fire, lambda: None)


__all__: list[str] = ["SideDebugSample", "enabled", "interval", "log_sides"]
