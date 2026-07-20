"""Per-timestep droplet-metric CSV export."""

from __future__ import annotations
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from typing import NamedTuple
import numpy as np
from tud_lbm.io.analysis.physical_parameters.physical_parameters import _get_setup_contact_line_length
from tud_lbm.io.plotting._analysis_common import _parse_timestep
from tud_lbm.io.plotting._analysis_common import _set_empty_state
from tud_lbm.io.plotting.base import AnalysisPlot
from tud_lbm.registry import comparison_operator

if TYPE_CHECKING:
    import matplotlib.axes
    import pandas as pd
    from tud_lbm.config import SimulationConfig

_WIDTH_EPS = 1e-15

_SUPPORTED_SIM_TYPES = {
    "multiphase_wetting",
    "multiphase_hysteresis",
    "multiphase_hysteresis_chemical_step",
}
_CSV_FILENAME = "simulation_data.csv"


class RZero(NamedTuple):
    """Resolved R₀ plus whether the nominal-radius fallback was used."""

    value: float
    used_fallback: bool


def _resolve_r_zero(config: SimulationConfig) -> RZero:
    """Initial droplet radius in lattice units.

    Derived from half of the setup contact-line length when the init file is
    readable. Otherwise falls back to ``initialisation.radii[0] * min(nx, ny)``
    (or 27.0 when no radii are given) and flags ``used_fallback=True``.
    """
    length = _get_setup_contact_line_length(config)
    if length is not None:
        return RZero(length / 2.0, used_fallback=False)

    init = config.initialisation
    radii = init.get("radii", []) if isinstance(init, dict) else []
    if radii:
        nominal = float(radii[0]) * float(min(config.grid_shape[0], config.grid_shape[1]))
        return RZero(nominal, used_fallback=True)
    return RZero(27.0, used_fallback=True)


def _derive_surface_tension(config: SimulationConfig) -> float | None:
    """Compute lattice surface tension gamma=(2/3)*(kappa/W)*(delta_rho**2)."""
    kappa = getattr(config, "kappa", None)
    width = getattr(config, "interface_width", None)
    rho_l = getattr(config, "rho_l", None)
    rho_v = getattr(config, "rho_v", None)
    if kappa is None or width is None or rho_l is None or rho_v is None:
        return None
    if abs(float(width)) < _WIDTH_EPS:
        return None
    drho = float(rho_l) - float(rho_v)
    return (2.0 / 3.0) * (float(kappa) / float(width)) * drho**2


def _resolve_initial_radius(config: SimulationConfig) -> float | None:
    """Return initial drop radius in lattice units from config.initialisation."""
    init = getattr(config, "initialisation", None)
    if not init or not isinstance(init, dict):
        return None
    try:
        radii = init.get("radii", [])
        if not radii:
            return None
        nx = float(config.grid_shape[0])
        ny = float(config.grid_shape[1])
        return float(radii[0]) * min(nx, ny)
    except (IndexError, ValueError, TypeError):
        return None


def _resolve_step_x(config: SimulationConfig) -> float | None:
    """Return the chemical step x-position in lattice units from config."""
    chem = getattr(config, "chemical_step_config", None)
    if not chem or not isinstance(chem, dict):
        return None
    loc = chem.get("chemical_step_location")
    if loc is None:
        return None
    return float(loc) * float(config.grid_shape[0])


def _inclination_angle_deg(config: SimulationConfig) -> float:
    gf = config.gravity_force
    if gf and isinstance(gf, dict):
        return float(gf.get("inclination_angle_deg", 0.0))
    return 0.0


def _sigma_lg(config: SimulationConfig) -> float:
    if config.kappa is None or config.interface_width is None or config.rho_l is None or config.rho_v is None:
        msg = "kappa, interface_width, rho_l, rho_v are required for surface tension"
        raise ValueError(msg)
    return (
        (2.0 / 3.0)
        * (float(config.kappa) / float(config.interface_width))
        * (float(config.rho_l) - float(config.rho_v)) ** 2
    )


def _interpolate_interface(row: np.ndarray, rho_mean: float) -> tuple[float, float]:
    """Sub-cell left/right interface positions via linear interpolation."""
    mask = (row < rho_mean).astype(int)
    diff = np.diff(mask)
    li = int(np.nonzero(diff == -1)[0][0])
    ri = int(np.nonzero(diff == 1)[0][0]) + 1
    x_left = li + (rho_mean - row[li]) / (row[li + 1] - row[li])
    x_right = ri - (rho_mean - row[ri]) / (row[ri - 1] - row[ri])
    return x_left, x_right


def _ca_from_rho(rho_2d: np.ndarray, rho_mean: float) -> tuple[float, float]:
    xl0, xr0 = _interpolate_interface(rho_2d[:, 1], rho_mean)
    xl1, xr1 = _interpolate_interface(rho_2d[:, 2], rho_mean)
    ca_left = float(np.rad2deg(math.pi / 2.0 + np.arctan(xl0 - xl1)))
    ca_right = float(np.rad2deg(math.pi / 2.0 + np.arctan(xr1 - xr0)))
    return ca_left, ca_right


def _cll_from_rho(rho_2d: np.ndarray, rho_mean: float) -> tuple[float, float]:
    return _interpolate_interface(rho_2d[:, 1], rho_mean)


def _center_of_mass(rho_2d: np.ndarray, rho_mean: float) -> tuple[float, float]:
    mask = rho_2d > rho_mean
    xi, yi = np.indices(rho_2d.shape)
    total = np.sum(mask * rho_2d)
    return float(np.sum(xi * mask * rho_2d) / total), float(np.sum(yi * mask * rho_2d) / total)


def _avg_x_location(rho_2d: np.ndarray, rho_mean: float, offset_x: float) -> float:
    nx = rho_2d.shape[0]
    mask = rho_2d > rho_mean
    x_idx = np.arange(nx, dtype=float) - offset_x
    return float(np.sum(x_idx[:, None] * mask) / np.sum(mask))


def _backward_diff(arr: np.ndarray, interval: int) -> np.ndarray:
    shifted = np.roll(arr, 1)
    shifted[0] = arr[0]
    return (arr - shifted) / max(interval, 1)


def _empty_csv_rows() -> dict[str, list[float | int]]:
    keys = (
        "iteration",
        "avg_u_x",
        "avg_u_y",
        "avg_x_location",
        "cll_left",
        "cll_right",
        "ca_left",
        "ca_right",
        "cm_x",
        "cm_y",
    )
    return {k: [] for k in keys}


def _append_csv_row(rows: dict[str, list[float | int]], row_data: dict[str, float | int]) -> None:
    for key, val in row_data.items():
        rows[key].append(val)


def _extract_optional_contact_metrics(
    raw: np.lib.npyio.NpzFile,
) -> tuple[float | None, float | None, float | None, float | None]:
    ca_l = float(np.asarray(raw["ca_left"]).squeeze()) if "ca_left" in raw else None
    ca_r = float(np.asarray(raw["ca_right"]).squeeze()) if "ca_right" in raw else None
    cll_l = float(np.asarray(raw["cll_left"]).squeeze()) if "cll_left" in raw else None
    cll_r = float(np.asarray(raw["cll_right"]).squeeze()) if "cll_right" in raw else None
    return ca_l, ca_r, cll_l, cll_r


def _collect_csv_rows(files: list[Path], rho_mean: float, offset_x: float) -> dict[str, list[float | int]]:
    rows = _empty_csv_rows()
    for fp in files:
        it = _parse_timestep(fp.stem)
        if it is None:
            continue
        with np.load(fp) as raw:
            rho = np.asarray(raw["rho"])
            u = np.asarray(raw["u"])
            ca_l, ca_r, cll_l, cll_r = _extract_optional_contact_metrics(raw)

        rho_2d = rho[:, :, 0, 0, 0]
        u_x = u[:, :, 0, 0, 0]
        u_y = u[:, :, 0, 0, 1]
        mask = rho_2d > rho_mean
        n_liq = np.sum(mask)
        avg_ux = float(np.sum(u_x * mask) / n_liq) if n_liq > 0 else 0.0
        avg_uy = float(np.sum(u_y * mask) / n_liq) if n_liq > 0 else 0.0

        if cll_l is None or cll_r is None:
            cll_l, cll_r = _cll_from_rho(rho_2d, rho_mean)
        if ca_l is None or ca_r is None:
            ca_l, ca_r = _ca_from_rho(rho_2d, rho_mean)

        cm_x, cm_y = _center_of_mass(rho_2d, rho_mean)
        avg_x = _avg_x_location(rho_2d, rho_mean, offset_x)
        _append_csv_row(
            rows,
            {
                "iteration": it,
                "avg_u_x": avg_ux,
                "avg_u_y": avg_uy,
                "avg_x_location": avg_x,
                "cll_left": cll_l,
                "cll_right": cll_r,
                "ca_left": ca_l,
                "ca_right": ca_r,
                "cm_x": cm_x,
                "cm_y": cm_y,
            },
        )
    return rows


def _finalize_csv_dataframe(
    df: pd.DataFrame,
    *,
    r_zero: float,
    nu: float,
    sigma_lg: float,
    incl_deg: float,
    save_iv: int,
) -> pd.DataFrame:
    it_min = df["iteration"].min()
    it_max = df["iteration"].max()
    span = it_max - it_min
    df["normalised_iteration"] = (df["iteration"] - it_min) / span if span > 0 else 0.0
    df["avg_x_location_norm"] = df["avg_x_location"] / r_zero
    df["v_left"] = _backward_diff(df["cll_left"].to_numpy(), save_iv)
    df["v_right"] = _backward_diff(df["cll_right"].to_numpy(), save_iv)
    df["v_cm"] = _backward_diff(df["cm_x"].to_numpy(), save_iv)
    df["Ca"] = (df["avg_u_x"] * nu) / sigma_lg
    df["Ca_cll_left"] = (df["v_left"] * nu) / sigma_lg
    df["Ca_cll_right"] = (df["v_right"] * nu) / sigma_lg
    df["Ca_cm"] = (df["v_cm"] * nu) / sigma_lg
    df["Ca_norm"] = df["Ca"] / math.sin(math.radians(incl_deg)) if incl_deg > 0 else df["Ca"]
    length = 2.0 * r_zero
    df["Re"] = (df["avg_u_x"] * length) / nu
    return df[
        [
            "iteration",
            "normalised_iteration",
            "avg_x_location",
            "avg_x_location_norm",
            "avg_u_x",
            "avg_u_y",
            "cll_left",
            "cll_right",
            "v_left",
            "v_right",
            "v_cm",
            "ca_left",
            "ca_right",
            "cm_x",
            "cm_y",
            "Ca",
            "Ca_cll_left",
            "Ca_cll_right",
            "Ca_cm",
            "Ca_norm",
            "Re",
        ]
    ]


def build_simulation_csv(run_dir: str | Path, config: SimulationConfig) -> Path | None:
    """Compute per-timestep metrics and write ``simulation_data.csv``.

    Skips silently (returns ``None``) when *config.sim_type* is not a
    supported wetting variant. Saves the CSV directly to *run_dir*.

    Args:
        run_dir: Run directory (contains ``data/timestep_*.npz``).
        config:  :class:`~tud_lbm.config.simulation_config.SimulationConfig`.

    Returns:
        Path to the written CSV, or ``None`` when skipped.
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas is required for CSV export. Install with: pip install pandas")
        return None

    if config.sim_type not in _SUPPORTED_SIM_TYPES:
        return None

    run_dir = Path(run_dir)
    data_dir = run_dir / "data"
    if not data_dir.exists():
        return None

    files = sorted(data_dir.glob("timestep_*.npz"), key=_parse_timestep_from_path)
    if not files:
        return None

    if config.rho_l is None or config.rho_v is None:
        msg = "rho_l and rho_v are required for CSV export"
        raise ValueError(msg)
    rho_mean = (float(config.rho_l) + float(config.rho_v)) / 2.0
    sigma_lg = _sigma_lg(config)
    nu = (float(config.tau) - 0.5) / 3.0
    r_zero = _resolve_r_zero(config)
    if r_zero.used_fallback:
        print(
            f"WARNING: R_0 fell back to nominal radius ({r_zero.value:.4g} lu); "
            f"init file not found (init_dir={config.init_dir!r}). "
            "avg_x_location_norm uses radii*min_dim, not measured L/2.",
            file=sys.stderr,
        )
    incl_deg = _inclination_angle_deg(config)
    save_iv = max(config.save_interval, 1)

    step_x = _resolve_step_x(config)
    offset_x = step_x if step_x is not None else float(config.grid_shape[0] // 2)

    rows = _collect_csv_rows(files, rho_mean, offset_x)

    df = pd.DataFrame(rows)
    if df.empty:
        return None
    df = _finalize_csv_dataframe(
        df,
        r_zero=r_zero.value,
        nu=nu,
        sigma_lg=sigma_lg,
        incl_deg=incl_deg,
        save_iv=save_iv,
    )

    csv_path = run_dir / _CSV_FILENAME
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    return csv_path


def _parse_timestep_from_path(p: Path) -> int:
    try:
        return int(p.stem.rsplit("_", maxsplit=1)[-1])
    except ValueError:
        return -1


@comparison_operator(name="simulation_csv")
class SimulationCsvExport(AnalysisPlot):
    """Export per-timestep droplet metrics to ``simulation_data.csv``."""

    name = "simulation_csv"
    required_keys = ("rho", "u")
    export_only = True

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:  # noqa: ARG002
        """Return an empty payload because this operator is export-only."""
        return {
            "iters": np.asarray([], dtype=int),
            "values": np.asarray([], dtype=float),
        }

    def render(self, ax: matplotlib.axes.Axes, precomputed: dict[str, np.ndarray]) -> None:  # noqa: ARG002
        """Render a placeholder panel when this operator is selected for plotting."""
        _set_empty_state(
            ax,
            title="CSV export operator",
            ylabel="N/A",
            required_keys=self.required_keys,
        )

    def export(self, run_dir: Path) -> Path | None:
        """Write ``simulation_data.csv`` for the configured run directory."""
        if self.config is None:
            return None
        return build_simulation_csv(run_dir, self.config)
