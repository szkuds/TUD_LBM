"""Debug logging for the hysteresis wetting optimisation.

Enabled via the ``--debug-wetting`` CLI flag, which sets
``DEBUG_FLAG_WETTING`` in :mod:`src.config.config_overview`.  When on,
:mod:`src.operators.wetting.hysteresis.hysteresis` emits two kinds of
trace through this module:

* an inner-loop line per ``_optimise_single_param`` exit — iteration
  count against the cap and the final loss, which is how you tell a
  converged solve from one that ran out of iterations;
* a per-side summary after the fallback branches have resolved — the
  measured CA against its hysteresis window, the CLL, which parameter
  ended up driving the wall (``mode``), and the residual loss.

All output goes through ``jax.debug.print`` so it survives the
``lax.scan`` trace.  The flag is read at call time, not import time, so
the CLI can flip it after the operator modules are already loaded.
"""

from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import src.config.config_overview as _flags

# mode codes used in the per-side summary line
_MODE_D_RHO = 0
_MODE_PHI = 1
_MODE_FALLBACK = 2

_SIDE_LINE = (
    "[{side}] CA={ca:.3f}° (adv={ca_adv:.1f}° rec={ca_rec:.1f}°) | CLL={cll:.3f} | "
    "mode={mode}(0=d_rho,1=phi,2=fb) | "
    "phi: {phi:.6f} | "
    "d_rho: {d_rho:.6f} | "
    "loss={loss:.3e}"
)


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
    """

    ca: jnp.ndarray
    ca_adv: jnp.ndarray
    ca_rec: jnp.ndarray
    cll: jnp.ndarray
    phi: jnp.ndarray
    d_rho: jnp.ndarray
    phi_active: jnp.ndarray
    loss: jnp.ndarray


def enabled() -> bool:
    """Return whether wetting debug logging is currently switched on.

    Read this instead of importing ``DEBUG_FLAG_WETTING`` directly:
    the CLI sets the flag as a module attribute after the operator
    packages have been imported, so a ``from ... import`` binding taken
    at import time would still be ``False``.
    """
    return _flags.DEBUG_FLAG_WETTING


def log_optimiser_exit(iterations: jnp.ndarray, max_iterations: int, loss: jnp.ndarray) -> None:
    """Log how an inner optimisation loop terminated.

    ``iterations == max_iterations`` means the loop hit the cap rather
    than the loss tolerance.
    """
    if not enabled():
        return
    jax.debug.print(
        "opt exit: iters={i}/{m} loss={l:.3e}",
        i=iterations,
        m=max_iterations,
        l=loss,
    )


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


def _log_side(side: str, sample: SideDebugSample, phi_neutral: jnp.ndarray, prefix: str = "") -> None:
    jax.debug.print(
        prefix + _SIDE_LINE,
        side=side,
        ca=sample.ca,
        ca_adv=sample.ca_adv,
        ca_rec=sample.ca_rec,
        cll=sample.cll,
        mode=_mode(sample, phi_neutral),
        phi=sample.phi,
        d_rho=sample.d_rho,
        loss=sample.loss,
    )


def log_sides(
    left: SideDebugSample,
    right: SideDebugSample,
    *,
    phi_neutral: jnp.ndarray,
) -> None:
    """Log both contact lines as a blank-line-separated two-line block.

    Right is printed first to match the existing trace format.

    Args:
        left: Left contact line's post-optimisation state.
        right: Right contact line's post-optimisation state.
        phi_neutral: The neutral ``phi`` value, used to decide whether
            ``phi`` actually engaged (see :func:`_mode`).
    """
    if not enabled():
        return
    _log_side("R", right, phi_neutral, prefix="\n")
    _log_side("L", left, phi_neutral)


__all__: list[str] = ["SideDebugSample", "enabled", "log_optimiser_exit", "log_sides"]
