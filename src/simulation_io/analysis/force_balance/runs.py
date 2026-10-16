"""Run discovery and the per-run scales of the force-balance model.

Every number here is read from the run's own ``config.toml`` and init field
through the project's resolvers (``resolve_dimensionless_inputs`` for gamma,
the measured density contrast, gravity and ``R0 = sqrt(Omega_D)``). Nothing is
defaulted: a run missing one of them is reported and skipped.
"""

from __future__ import annotations
import hashlib
import math
import re
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
import numpy as np
from src.config.run_config import CONFIG_FILENAME
from src.simulation_io.analysis.droplet_metrics import droplet_series_for_run
from src.simulation_io.analysis.droplet_metrics import resolve_step_x

if TYPE_CHECKING:
    from pathlib import Path
    from src.config import SimulationConfig
    from src.simulation_io.analysis.droplet_metrics import DropletSeries
    from src.simulation_io.analysis.force_balance.config import ForceBalanceConfig
    from src.simulation_io.analysis.physical_parameters import DimensionlessInputs

_INIT_TIMESTEP = re.compile(r"timestep_(\d+)\.npz$")
_INIT_FROM_FILE_LABEL = "init_from_file"


class AssumptionLog:
    """Every assumption and data problem met during an analysis, in order."""

    def __init__(self) -> None:
        """An empty log."""
        self.entries: list[tuple[str, str]] = []

    def add(self, scope: str, message: str) -> None:
        """Record *message* under *scope* (a run name, or ``"global"``)."""
        self.entries.append((scope, message))

    def lines(self) -> list[str]:
        """The log, one ``[scope] message`` per line."""
        return [f"[{scope}] {message}" for scope, message in self.entries]


def _cos(deg: float) -> float:
    return math.cos(math.radians(deg))


@dataclass(frozen=True)
class Windows:
    """Imposed hysteresis windows (degrees) and the three step-model ``H`` values."""

    receding_phob: float
    advancing_phob: float
    receding_phil: float
    advancing_phil: float

    @property
    def h1(self) -> float:
        """Both lines hydrophobic."""
        return _cos(self.receding_phob) - _cos(self.advancing_phob)

    @property
    def h2(self) -> float:
        """Trailing line hydrophobic, leading line hydrophilic."""
        return _cos(self.receding_phob) - _cos(self.advancing_phil)

    @property
    def h3(self) -> float:
        """Both lines hydrophilic."""
        return _cos(self.receding_phil) - _cos(self.advancing_phil)


@dataclass(frozen=True)
class RunInfo:
    """One run with every scale the model needs, in lattice units unless noted.

    ``eta = rho_l * nu`` is the liquid's dynamic viscosity, ``nu = (tau - 1/2)/3``
    (BGK, ``cs**2 = 1/3``). ``bo`` is built with the *measured* density
    contrast, which is the buoyancy the masked gravity force injects;
    ``bo_rho_l`` uses ``rho_l`` instead, as the SI writes it.
    """

    run: str
    path: Path
    name: str
    groups: list[str]
    nx: int
    save_interval: int
    t_onset: int
    tau: float
    rho_l: float
    eta: float
    gamma: float
    gamma_source: str
    drho: float
    g: float
    alpha_deg: float
    omega: float
    r0: float
    step_x: float
    edge_width: float
    windows: Windows
    learning_rate: float | None
    interface_width: float
    duplicates: list[str] = field(default_factory=list)

    @property
    def oh(self) -> float:
        """``Oh = eta / sqrt(rho_l * gamma * R0)``."""
        return self.eta / math.sqrt(self.rho_l * self.gamma * self.r0)

    @property
    def bo(self) -> float:
        """``Bo_par = drho * g * R0**2 * sin(alpha) / gamma``, the injected force."""
        return self.drho * self.g * self.r0**2 * math.sin(math.radians(self.alpha_deg)) / self.gamma

    @property
    def bo_rho_l(self) -> float:
        """``Bo_par`` with ``rho_l`` in place of the measured contrast."""
        return self.rho_l * self.g * self.r0**2 * math.sin(math.radians(self.alpha_deg)) / self.gamma

    @property
    def time_scale(self) -> float:
        """``eta*R0/gamma``: lattice steps per unit ``T``."""
        return self.eta * self.r0 / self.gamma


def _onset(config: SimulationConfig, first_iteration: int, log: AssumptionLog, scope: str) -> int:
    """Timestep at which gravity switched on: the init snapshot's timestep."""
    match = _INIT_TIMESTEP.search(str(config.init_dir or ""))
    if match:
        return int(match.group(1))
    onset = first_iteration - config.save_interval
    log.add(scope, f"init timestep not in init_dir; gravity onset taken as first sample - save_interval = {onset}")
    return onset


class _Resolved(NamedTuple):
    """The config values a run needs, each checked present."""

    inputs: DimensionlessInputs
    g: float
    angle_deg: float
    step: dict[str, Any]
    step_x: float
    rho_l: float
    interface_width: float


def _resolve(config: SimulationConfig) -> _Resolved | str:
    """Everything :func:`load_run` needs from *config*, or why the run is unusable."""
    from src.simulation_io.analysis.physical_parameters import resolve_dimensionless_inputs

    inputs = resolve_dimensionless_inputs(config)
    if inputs is None:
        return "gamma/drho/R0 unresolvable"
    if inputs.g is None or inputs.angle_deg is None:
        return "no gravity configured"
    if _INIT_FROM_FILE_LABEL not in inputs.length_label:
        return f"R0 not measured from the init field ({inputs.length_label})"
    step = config.chemical_step_config
    step_x = resolve_step_x(config)
    if not step or step_x is None:
        return "no chemical step"
    if config.rho_l is None or config.interface_width is None:
        return "rho_l or interface_width missing"
    return _Resolved(inputs, inputs.g, inputs.angle_deg, step, step_x, config.rho_l, float(config.interface_width))


def load_run(
    rel: str, groups: list[str], cfg: ForceBalanceConfig, log: AssumptionLog
) -> tuple[RunInfo, DropletSeries] | None:
    """Resolve *rel* into its :class:`RunInfo` and droplet series, or ``None`` (logged)."""
    from src.simulation_io.readers import TomlAdapter

    path = cfg.root / rel
    if not (path / CONFIG_FILENAME).exists():
        log.add(rel, "SKIPPED: path or config.toml does not exist")
        return None
    config = TomlAdapter().load(str(path / CONFIG_FILENAME))
    resolved = _resolve(config)
    if isinstance(resolved, str):
        log.add(rel, f"SKIPPED: {resolved}")
        return None
    series = droplet_series_for_run(path, config)
    if series is None or len(series) < cfg.settings.smoothing_window:
        log.add(rel, "SKIPPED: no usable snapshots")
        return None

    inputs, step = resolved.inputs, resolved.step
    nu = (float(config.tau) - 0.5) / 3.0
    if not math.isclose(nu, inputs.nu, rel_tol=1e-12):
        log.add(rel, f"nu from tau ({nu}) differs from the resolver's ({inputs.nu})")
    if inputs.gamma_source != "measured":
        log.add(rel, f"gamma = {inputs.gamma:.6g} is the {inputs.gamma_source} closed form (no calibration)")
    info = RunInfo(
        run=rel,
        path=path,
        name=str(config.simulation_name),
        groups=list(groups),
        nx=int(config.grid_shape[0]),
        save_interval=int(config.save_interval),
        t_onset=_onset(config, int(series.iteration[0]), log, rel),
        tau=float(config.tau),
        rho_l=float(resolved.rho_l),
        eta=float(resolved.rho_l) * nu,
        gamma=float(inputs.gamma),
        gamma_source=inputs.gamma_source,
        drho=float(inputs.drho),
        g=float(resolved.g),
        alpha_deg=float(resolved.angle_deg),
        omega=float(inputs.length) ** 2,
        r0=float(inputs.length),
        step_x=float(resolved.step_x),
        edge_width=float(step.get("edge_width", 0.0)),
        windows=Windows(
            receding_phob=float(step["ca_receding_pre_step"]),
            advancing_phob=float(step["ca_advancing_pre_step"]),
            receding_phil=float(step["ca_receding_post_step"]),
            advancing_phil=float(step["ca_advancing_post_step"]),
        ),
        learning_rate=(config.hysteresis_config or {}).get("learning_rate"),
        interface_width=resolved.interface_width,
    )
    return info, series


def trajectory_digest(series: DropletSeries) -> str:
    """Content hash of a run's measured trajectory, for finding duplicate runs."""
    stacked = np.stack([series.iteration.astype(float), series.cm_x, series.cll_left, series.cll_right])
    return hashlib.sha256(np.round(stacked, 10).tobytes()).hexdigest()


def load_runs(
    cfg: ForceBalanceConfig,
    selected: list[str] | None,
    log: AssumptionLog,
) -> list[tuple[RunInfo, DropletSeries]]:
    """Every selected run, with bit-identical trajectories collapsed into one.

    A duplicate keeps its groups: they are merged into the first run with the
    same trajectory, which is the one analysed.
    """
    runs: list[tuple[RunInfo, DropletSeries]] = []
    by_digest: dict[str, int] = {}
    for rel, groups in cfg.membership(selected).items():
        loaded = load_run(rel, groups, cfg, log)
        if loaded is None:
            continue
        info, series = loaded
        digest = trajectory_digest(series)
        if digest in by_digest:
            keep = runs[by_digest[digest]][0]
            keep.duplicates.append(rel)
            for g in groups:
                if g not in keep.groups:
                    keep.groups.append(g)
            log.add(rel, f"identical trajectory to {keep.run}; counted once, groups merged")
            continue
        by_digest[digest] = len(runs)
        runs.append((info, series))
    return runs
