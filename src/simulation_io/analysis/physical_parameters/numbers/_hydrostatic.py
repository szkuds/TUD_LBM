"""Hydrostatic load: the gravitational potential a domain demands of its liquid branch.

**The liquid in a diffuse-interface model is not incompressible.** Its stiffness is
whatever the EOS gives: at the shipped Carnahan-Starling parameters
``dp_0/drho = 0.00715`` at ``rho_l``, so the liquid's sound speed is 0.085 lattice
units — 6.8x slower than the lattice's own — and its bulk modulus is
``K = rho dp/drho = 0.087``. Mapping a run at ``Bo = 0.90`` to water/air puts a
2.6 mm bubble in a 15.6 mm domain, where real water stratifies by ``drho/rho ~ 7e-8``;
the simulation stratifies by 16%. That is inherent to the method — the EOS has to
carry both phases on one curve at ``T_r = 0.5`` with an interface resolvable in a
handful of cells — so it cannot be tuned away, only kept small.

The hard limit is the **liquid spinodal**, the density below which ``dp_0/drho < 0``
and the liquid separates spontaneously. Past it the depletion runs away by positive
feedback: thinner liquid is softer, so it thins faster. This number is the ratio of
the gravitational potential the domain *demands* to what the liquid branch *can
carry* before reaching that point::

    demand   = g * (sin(angle)*nx + cos(angle)*ny)
    capacity = integral from rho_spinodal to rho_l of (dp_0/drho)/rho drho
    Hs       = demand / capacity

``Hs >= 1`` means the up-slope, up-height corner reaches the spinodal no matter how
long the run goes. Measured: the double-well regime-map runs sit at **0.51** and
survive; a Carnahan-Starling bubble run at **1.53** depleted its top wall from 11.83
to 5.09 and left its intended regime, having *also* hit the separate contact-line
region bug that the anchored band in ``_wetting_modification`` now fixes.

Deliberately nonlinear. The linearised form ``rho_l g L / K`` uses ``dp/drho`` at
``rho_l`` alone and reads a comfortable 15.7% against a 24% margin for that same
failing run — it misses exactly the softening that causes the runaway.

NumPy at analysis time, never inside a JIT trace, so the sampling grid is generous.
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np
from src.registry import dimensionless_operator

if TYPE_CHECKING:
    from collections.abc import Callable
    from src.simulation_io.analysis.physical_parameters._inputs import DimensionlessInputs

#: Samples across ``(0, rho_l]`` for the spinodal search and the capacity integral.
#: Converged to every digit reported: for the measured Carnahan-Starling case the
#: capacity is 7.3428397e-4 here against 7.3428420e-4 at 2e5 samples, and ``Hs``
#: prints as 1.52918 at both. 2001 samples would shift that last digit by one.
_SAMPLES = 5001

#: Lower end of the sampling range, as a fraction of ``rho_l``. Not zero: the ideal
#: part of every EOS has ``rho log rho``, so ``dp_0/drho`` diverges as ``rho -> 0``.
_FLOOR_FRAC = 1e-4


def _liquid_capacity(pressure: Callable[[np.ndarray], np.ndarray], rho_l: float) -> float | None:
    """Gravitational potential per unit mass the liquid branch carries before the spinodal.

    Returns ``None`` when no spinodal is found below *rho_l* — an EOS with a
    monotone pressure there has no instability to run into, so the question this
    number asks does not apply — or when *rho_l* itself sits at the EOS's own
    divergence, where there is no liquid branch at all.
    """
    rho, d_rho = np.linspace(_FLOOR_FRAC * rho_l, rho_l, _SAMPLES, retstep=True)
    # Carnahan-Starling diverges at ``b rho = 4``. Reaching it at or below *rho_l*
    # means the configured liquid density is at or past the pole, so there is no
    # liquid branch to integrate. Expected, so silenced and handled here rather
    # than left to warn.
    with np.errstate(divide="ignore", invalid="ignore"):
        p_0 = np.asarray(pressure(rho), dtype=float)
    if not np.all(np.isfinite(p_0)):
        return None

    # Uniform grid, so pass the scalar step: handing np.gradient the coordinate
    # array sends it down the non-uniform path for an identical answer.
    dp_drho = np.gradient(p_0, float(d_rho))

    # The liquid spinodal is the *last* sign change below rho_l; the vapour one
    # sits further down the same curve and must not be picked up.
    crossings = np.flatnonzero(np.diff(np.signbit(dp_drho)))
    if crossings.size == 0:
        return None

    # rho ascends, so the stable branch is simply everything from there up.
    start = int(crossings.max())
    capacity = float(np.trapezoid(dp_drho[start:] / rho[start:], rho[start:]))
    return capacity if capacity > 0.0 else None


@dimensionless_operator(
    name="hs",
    label=r"$\mathrm{Hs}$",
    row_label="Hs (hydrostatic load):",
    formula="g*(sinθ*nx+cosθ*ny) / ∫(dp₀/dρ)/ρ dρ",
    order=80,
    needs_gravity=True,
    # Built from the grid and the EOS, not from the inclusion length or the
    # buoyancy contrast, so the row must not annotate either.
    annotates_scale=False,
)
def hydrostatic_load(inputs: DimensionlessInputs) -> float | None:
    """Hs = demanded gravitational potential / what the liquid branch can carry.

    ``None`` without gravity, without a domain, without an EOS registered under
    the ``pressure`` kind, or when the liquid branch has no spinodal to run into.
    """
    if inputs.g is None or inputs.pressure is None or inputs.domain is None or inputs.rho_l <= 0.0:
        return None

    capacity = _liquid_capacity(inputs.pressure, inputs.rho_l)
    if capacity is None:
        return None

    angle = math.radians(inputs.angle_deg or 0.0)
    nx, ny = inputs.domain
    demand = inputs.g * (math.sin(angle) * nx + math.cos(angle) * ny)
    return demand / capacity
