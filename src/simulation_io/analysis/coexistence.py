"""Maxwell-construction calibration: the coexistence densities an EOS predicts for itself.

For an EOS that registers a coexistence solver (kind ``"coexistence"``,
Carnahan-Starling), the liquid and vapour densities at coexistence follow from
its own parameters. :func:`calibrate_coexistence` computes them and writes them
into the run directory beside the configured densities, so a gap between the two
is visible. Unlike the Young-Laplace calibration of
:mod:`~src.simulation_io.analysis.surface_tension` it needs no sweep and no
cache: it is a scalar root-find that takes milliseconds.

The artefacts sit under ``<run_dir>/coexistence/``, in the ``data/`` + ``plots/``
shape a run directory itself has:

``data/data.json``
    ``eos``, the Maxwell ``rho_v``/``rho_l``, ``p_sat`` and the configured densities.
``plots/maxwell.png``
    The bulk pressure against specific volume, the tie line at ``p_sat`` and the
    two equal areas, with the configured densities marked.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING
import jax.numpy as jnp
import numpy as np
from src.config.multiphase import multiphase_params
from src.config.run_config import DATA_DIRNAME
from src.config.run_config import PLOTS_DIRNAME
from src.operators.macroscopic.eos import build_pressure_fn
from src.operators.macroscopic.eos import coexistence_densities

if TYPE_CHECKING:
    from collections.abc import Callable
    from src.config.simulation_config import SimulationConfig

_OUTPUT_DIRNAME = "coexistence"
_DATA_FILENAME = "data.json"
_PLOT_FILENAME = "maxwell.png"
#: Samples of the isotherm drawn in the figure.
_N_CURVE = 2000


def coexistence_dir(run_dir: str | Path) -> Path:
    """Return the subdirectory of *run_dir* holding the coexistence artefacts."""
    return Path(run_dir) / _OUTPUT_DIRNAME


def calibrate_coexistence(config: SimulationConfig, run_dir: str | Path) -> tuple[float, float] | None:
    """Compute the Maxwell pair of *config*'s EOS and write its artefacts into *run_dir*.

    Returns:
        ``(rho_v, rho_l)``, or ``None`` for a single-phase run or an EOS that
        registers no coexistence solver (its densities are its own parameters).

    Raises:
        ValueError: If the EOS parameters admit no liquid-vapour coexistence.
    """
    params = multiphase_params(config)
    if params is None:
        return None
    pair = coexistence_densities(params.eos, params._asdict())
    if pair is None:
        return None
    rho_v, rho_l = pair
    pressure_fn = build_pressure_fn(params)

    def pressure(rho: np.ndarray) -> np.ndarray:
        return np.asarray(pressure_fn(jnp.asarray(rho)))

    p_sat = float(pressure(np.asarray(rho_v)))
    out = coexistence_dir(run_dir)
    _save_data(
        out / DATA_DIRNAME / _DATA_FILENAME,
        {
            "eos": params.eos,
            "rho_v": rho_v,
            "rho_l": rho_l,
            "p_sat": p_sat,
            "rho_v_configured": float(params.rho_v),
            "rho_l_configured": float(params.rho_l),
        },
    )
    _save_plot(
        out / PLOTS_DIRNAME / _PLOT_FILENAME,
        pressure,
        maxwell=(rho_v, rho_l),
        configured=(float(params.rho_v), float(params.rho_l)),
        p_sat=p_sat,
    )
    return rho_v, rho_l


def _save_data(path: Path, payload: dict[str, str | float]) -> None:
    """Write the coexistence values into the run directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_plot(
    path: Path,
    pressure: Callable[[np.ndarray], np.ndarray],
    *,
    maxwell: tuple[float, float],
    configured: tuple[float, float],
    p_sat: float,
) -> None:
    """Draw ``p_0(1/v)`` with the tie line at *p_sat* and the two equal areas shaded."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    rho_v, rho_l = maxwell
    v_liquid, v_vapour = 1.0 / rho_l, 1.0 / rho_v
    volume = np.geomspace(0.8 * v_liquid, 2.0 * v_vapour, _N_CURVE)
    p = pressure(1.0 / volume)
    inside = (volume >= v_liquid) & (volume <= v_vapour)
    p_inside = p[inside]
    span = float(p_inside.max() - p_inside.min())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(volume, p, color="black", lw=1.5, label="$p_0(1/\\rho)$")
    ax.hlines(p_sat, v_liquid, v_vapour, colors="tab:red", lw=1.5, label=f"$p_{{sat}}$ = {p_sat:.6g}")
    # The isotherm crosses p_sat only at v_l, once between, and at v_v, so each area is one contiguous slice.
    below, above = inside & (p < p_sat), inside & (p > p_sat)
    ax.fill_between(volume[below], p[below], p_sat, color="tab:blue", alpha=0.3, label="liquid-side area")
    ax.fill_between(volume[above], p[above], p_sat, color="tab:orange", alpha=0.3, label="vapour-side area")
    for rho, label, colour in (
        (rho_v, f"Maxwell $\\rho_v$ = {rho_v:.6g}", "tab:red"),
        (rho_l, f"Maxwell $\\rho_l$ = {rho_l:.6g}", "tab:red"),
        (configured[0], f"configured $\\rho_v$ = {configured[0]:.6g}", "tab:green"),
        (configured[1], f"configured $\\rho_l$ = {configured[1]:.6g}", "tab:green"),
    ):
        ax.axvline(1.0 / rho, color=colour, ls=":", lw=1.0, label=label)
    ax.set_xscale("log")
    ax.set_ylim(p_inside.min() - 0.1 * span, p_inside.max() + 0.1 * span)
    ax.set_xlabel("specific volume $1/\\rho$ [lattice units]")
    ax.set_ylabel("$p_0$ [lattice units]")
    ax.set_title("Maxwell construction: areas equal in $v$ (the log axis distorts them)")
    ax.legend(fontsize="small")
    ax.grid(visible=True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
