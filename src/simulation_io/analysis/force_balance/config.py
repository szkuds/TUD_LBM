"""The one configuration file of a force-balance analysis.

A TOML file with three tables::

    [data]
    root = "~/TUD_LBM_data"            # run paths below are relative to it
    output_dir = "~/TUD_LBM_data/.../force_balance"

    [groups]                            # group name -> run directories
    inc = ["2026-09-30/00-13-49_var_inc_inc_20", ...]

    [analysis]                          # every knob, all optional
    smoothing_window = 7

    [figures]                           # optional
    representatives = ["2026-09-30/00-13-52_var_inc_inc_60", ...]

The same run may appear in several groups; it is analysed once and keeps the
list of groups it belongs to.
"""

from __future__ import annotations
import tomllib
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from dataclasses import replace
from pathlib import Path


@dataclass(frozen=True)
class AnalysisSettings:
    """Numerical choices of the analysis. Every one is written to the log.

    Attributes:
        smoothing_window: Savitzky-Golay window (samples) for the derivatives.
        smoothing_polyorder: Savitzky-Golay polynomial order.
        moving_sigma: A point enters the fits when ``Ca_cm`` exceeds this many
            standard deviations of its own noise.
        ca_min: ...and exceeds this absolute floor.
        line_pinned_sigma: A contact line is pinned when its speed is below this
            many noise standard deviations.
        angle_tolerance_deg: A measured angle within this of a window bound sits
            *at* the bound.
        bootstrap_samples: Resamples over runs for every confidence interval.
        seed: Seed of the bootstrap generator.
        seam_margin_lu: Data are cut once the leading line comes this close to
            the periodic seam, where the surface turns hydrophobic again.
        moving_dx: Net displacement (units of R0) above which a run moves.
        creep_dx: Net displacement above which a run that does not move creeps.
        stop_ca: Final ``Ca`` below which a run (or model) has stopped.
        integration_dt: Integrator step in T.
    """

    smoothing_window: int = 7
    smoothing_polyorder: int = 3
    moving_sigma: float = 3.0
    ca_min: float = 0.005
    line_pinned_sigma: float = 3.0
    angle_tolerance_deg: float = 2.0
    bootstrap_samples: int = 1000
    seed: int = 20261014
    seam_margin_lu: float = 5.0
    moving_dx: float = 0.5
    creep_dx: float = 0.15
    stop_ca: float = 0.005
    integration_dt: float = 2e-3


@dataclass(frozen=True)
class ForceBalanceConfig:
    """Where the runs are, how they are grouped, and how they are analysed."""

    root: Path
    output_dir: Path
    groups: dict[str, list[str]]
    settings: AnalysisSettings = field(default_factory=AnalysisSettings)
    source: Path | None = None
    #: Runs that get a per-run figure; empty means the first and last run of each group.
    representatives: list[str] = field(default_factory=list)

    def membership(self, selected: list[str] | None = None) -> dict[str, list[str]]:
        """Run path -> groups it belongs to, over *selected* groups (all when ``None``)."""
        names = list(self.groups) if selected is None else selected
        unknown = [g for g in names if g not in self.groups]
        if unknown:
            msg = f"unknown group(s) {unknown}; the config defines {sorted(self.groups)}"
            raise ValueError(msg)
        out: dict[str, list[str]] = {}
        for name in names:
            for run in self.groups[name]:
                out.setdefault(run, [])
                if name not in out[run]:
                    out[run].append(name)
        return out


def load_force_balance_config(path: str | Path, output_dir: str | Path | None = None) -> ForceBalanceConfig:
    """Read *path*; *output_dir* overrides ``[data] output_dir``."""
    path = Path(path)
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    data = raw.get("data", {})
    if "root" not in data:
        msg = f"{path}: [data] root is required"
        raise ValueError(msg)
    out = output_dir if output_dir is not None else data.get("output_dir")
    if out is None:
        msg = f"{path}: give [data] output_dir or --out"
        raise ValueError(msg)
    groups = {str(k): [str(r) for r in v] for k, v in raw.get("groups", {}).items()}
    if not groups:
        msg = f"{path}: [groups] is empty"
        raise ValueError(msg)
    known = {f.name for f in fields(AnalysisSettings)}
    analysis = raw.get("analysis", {})
    unknown = sorted(set(analysis) - known)
    if unknown:
        msg = f"{path}: unknown [analysis] keys {unknown}"
        raise ValueError(msg)
    settings = replace(AnalysisSettings(), **analysis)
    if settings.smoothing_window % 2 == 0 or settings.smoothing_window <= settings.smoothing_polyorder:
        msg = "smoothing_window must be odd and larger than smoothing_polyorder"
        raise ValueError(msg)
    representatives = [str(r) for r in raw.get("figures", {}).get("representatives", [])]
    return ForceBalanceConfig(
        root=Path(data["root"]).expanduser(),
        output_dir=Path(out).expanduser(),
        groups=groups,
        settings=settings,
        source=path,
        representatives=representatives,
    )
