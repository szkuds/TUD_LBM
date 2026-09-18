"""Interface contour levels and geometry for post-processing plots.

Two markers for where the interface is, so they can be compared on one figure:

``config``
    ``(rho_l + rho_v) / 2`` from the run's config — the same threshold every
    droplet metric (contact angles, centre of mass, inclusion mask) uses, so a
    contour at this level coincides with the numbers the analysis reports.

``measured``
    The midpoint of the *equilibrated* bulk densities of one snapshot: split the
    field at a rho_mean threshold and take ``(median(liquid) + median(vapour)) / 2``.
    Medians rather than ``min``/``max`` because the diffuse interface over- and
    undershoots the bulk values, and one such cell would move an extremum. The
    two markers diverge exactly when a run's bulk densities drift away from the
    prescribed coexistence pair.

Lives under ``analysis`` rather than ``plotting`` and imports no matplotlib, for
the same reason as :mod:`src.simulation_io.analysis.run_labels`: the CLI and
tests can use it without paying for the plotting package.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import contourpy
import numpy as np

if TYPE_CHECKING:
    from src.config import SimulationConfig

LEVEL_CONFIG = "config"
LEVEL_MEASURED = "measured"

#: Every interface marker, in display order.
INTERFACE_LEVEL_NAMES: tuple[str, ...] = (LEVEL_CONFIG, LEVEL_MEASURED)

#: Drawn when a config names no ``interface_levels``: both, so the comparison is
#: the default view.
DEFAULT_INTERFACE_LEVELS: tuple[str, ...] = INTERFACE_LEVEL_NAMES


def resolve_interface_levels(config: SimulationConfig) -> tuple[str, ...]:
    """Return the interface markers *config* selects, validated.

    Raises:
        ValueError: If ``config.interface_levels`` names an unknown marker.
    """
    requested = tuple(config.interface_levels or DEFAULT_INTERFACE_LEVELS)
    unknown = [name for name in requested if name not in INTERFACE_LEVEL_NAMES]
    if unknown:
        msg = f"Unknown interface level(s) {unknown}. Available: {list(INTERFACE_LEVEL_NAMES)}"
        raise ValueError(msg)
    return requested


def config_rho_mean(config: SimulationConfig) -> float | None:
    """``(rho_l + rho_v) / 2`` from *config*, or ``None`` when either is unset."""
    if config.rho_l is None or config.rho_v is None:
        return None
    return 0.5 * (float(config.rho_l) + float(config.rho_v))


def measured_phase_densities(rho_2d: np.ndarray, rho_mean: float) -> tuple[float, float] | None:
    """``(dense, light)`` bulk-phase median densities, split at *rho_mean*.

    Medians rather than ``min``/``max`` because the diffuse interface over- and
    undershoots the bulk values. Returns ``None`` when the field does not
    straddle *rho_mean* — a uniform or single-phase field has no two phases.
    """
    rho = np.asarray(rho_2d, dtype=float)
    dense = rho[rho > rho_mean]
    light = rho[rho < rho_mean]
    if dense.size == 0 or light.size == 0:
        return None
    return float(np.median(dense)), float(np.median(light))


def measured_rho_mean(rho_2d: np.ndarray, rho_mean: float) -> float | None:
    """Midpoint of the bulk-phase median densities, split at *rho_mean*.

    Returns ``None`` when the field does not straddle *rho_mean* — a uniform or
    single-phase field has no interface to mark.
    """
    phases = measured_phase_densities(rho_2d, rho_mean)
    return None if phases is None else 0.5 * (phases[0] + phases[1])


def level_densities(name: str, config: SimulationConfig, rho_2d: np.ndarray) -> tuple[float, float] | None:
    """``(dense, light)`` densities behind interface marker *name*, or ``None`` if unavailable.

    ``config`` is the prescribed ``(rho_l, rho_v)``; ``measured`` is this
    snapshot's bulk-phase medians. Every per-marker quantity — the contour level
    here, the wetting band of the contact-angle overlay — is derived from this
    pair, so the two markers mean the same thing wherever they are drawn.

    The ``measured`` split is seeded at the config midpoint when there is one and
    at ``(min + max) / 2`` otherwise, so a standalone snapshot with no multiphase
    config still gets a measured contour.
    """
    if name == LEVEL_CONFIG:
        if config.rho_l is None or config.rho_v is None:
            return None
        return float(config.rho_l), float(config.rho_v)
    if name == LEVEL_MEASURED:
        rho = np.asarray(rho_2d, dtype=float)
        from_config = config_rho_mean(config)
        seed = from_config if from_config is not None else 0.5 * (float(rho.min()) + float(rho.max()))
        return measured_phase_densities(rho, seed)
    msg = f"Unknown interface level {name!r}. Available: {list(INTERFACE_LEVEL_NAMES)}"
    raise ValueError(msg)


def level_value(name: str, config: SimulationConfig, rho_2d: np.ndarray) -> float | None:
    """Density of interface marker *name* for one snapshot, or ``None`` if unavailable.

    The midpoint of :func:`level_densities`.
    """
    phases = level_densities(name, config, rho_2d)
    return None if phases is None else 0.5 * (phases[0] + phases[1])


def interface_lines(rho_2d: np.ndarray, level: float) -> list[np.ndarray]:
    """Polylines of the ``rho == level`` contour, each an ``(k, 2)`` array of ``(x, y)``.

    *rho_2d* is in the ``(nx, ny)`` layout of :func:`extract_rho_2d`. It is
    transposed before contouring so vertex coordinates are the cell-centre
    indices ``imshow(rho.T, origin="lower")`` draws — the convention of every
    field panel — and a contour lands on the panel it overlays. A contour cut by
    a periodic boundary comes back as separate open segments.
    """
    generator = contourpy.contour_generator(z=np.asarray(rho_2d, dtype=float).T, line_type="Separate")
    return [np.asarray(line, dtype=float) for line in generator.lines(level)]
