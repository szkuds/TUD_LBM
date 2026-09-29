"""Two-phase wetting initialisation driven from the ``run`` command."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
import numpy as np
from rich.panel import Panel
from rich.prompt import Prompt
from src.cli._console import console
from src.cli.config_loading import _expand_single_phase
from src.cli.config_loading import _latest_snapshot_in
from src.cli.display import _display_config_summary
from src.cli.display import _display_full_overview
from src.cli.display import _print_dry_run_message
from src.cli.overrides import _apply_overrides
from src.cli.overrides import confirm_or_override

if TYPE_CHECKING:
    from src.config import SimulationConfig

#: Default length of the Phase 1 equilibration run (``--init-wetting NT``).
_WETTING_INIT_NT = 50_000

#: Phase 1 saves this many snapshots so max|u| can be plotted against time.
#: One final snapshot would show the end state but not whether it settled.
_WETTING_INIT_SNAPSHOTS = 20

#: Analysis operator rendered after Phase 1 as the equilibration check.
_CONVERGENCE_FIELD = "max_velocity"

#: Relative max|u| change across the last saved interval above which Phase 1
#: is reported as still evolving.
_EQUILIBRIUM_REL_TOL = 0.01

#: Stands in for Phase 1's final snapshot when ``--dry-run`` previews Phase 2:
#: that file is only written by the run the preview is declining to start.
#: ``src.config.init_field.resolve_npz_path`` returns ``None`` for a path that
#: does not exist, so ``--overview`` degrades rather than raising.
_DRY_RUN_SNAPSHOT = "<Phase 1 final snapshot>"

_WETTING_PARAM_DEFAULTS: dict[str, float] = {
    "phi_left": 1.0,
    "phi_right": 1.0,
    "d_rho_left": 0.0,
    "d_rho_right": 0.0,
}


def _prompt_wetting_params(base_raw: dict[str, Any], *, no_prompt: bool) -> dict[str, float]:
    """Return wetting params, reading from config and only prompting for missing ones."""
    from_config: dict[str, Any] = base_raw.get("wetting_config") or {}
    missing = [k for k in _WETTING_PARAM_DEFAULTS if k not in from_config]

    if missing and not no_prompt:
        console.print("[cyan]Wetting init - enter missing wetting boundary parameters:[/cyan]")

    params: dict[str, float] = {}
    for key, default in _WETTING_PARAM_DEFAULTS.items():
        if key in from_config:
            params[key] = float(from_config[key])
        elif no_prompt:
            params[key] = default
        else:
            params[key] = float(Prompt.ask(f"  {key}", default=str(default)))

    if missing and not no_prompt:
        console.print()

    return params


def _wetting_init_save_interval(nt: int) -> int:
    """Snapshot spacing giving ``_WETTING_INIT_SNAPSHOTS`` samples over *nt* steps."""
    return max(1, nt // _WETTING_INIT_SNAPSHOTS)


def _build_wetting_init_raw(
    base_raw: dict[str, Any],
    wetting_params: dict[str, float],
    nt: int = _WETTING_INIT_NT,
) -> dict[str, Any]:
    init_raw = deepcopy(base_raw)
    init_raw["sim_type"] = "multiphase_wetting"
    init_raw["init_type"] = "multiphase_bubbles"
    init_raw.pop("hysteresis_config", None)
    init_raw.pop("chemical_step_config", None)
    init_raw.pop("gravity_force", None)
    init_raw.pop("gravity_masked_force", None)
    if "bc_config" in init_raw:
        init_raw["bc_config"] = dict(init_raw["bc_config"])
    init_raw["nt"] = nt
    # Equilibration history, not just the end state: the convergence plot needs
    # intermediate snapshots, and skip_interval would drop the early ones.
    init_raw["save_interval"] = _wetting_init_save_interval(nt)
    init_raw["skip_interval"] = 0
    init_raw["output_format"] = "numpy"
    base_name = base_raw.get("simulation_name")
    init_raw["simulation_name"] = f"wetting_init_{base_name}" if base_name else "wetting_init"
    init_raw.setdefault("wetting_config", {}).update(wetting_params)
    return init_raw


def _build_wetting_gravity_raw(
    base_raw: dict[str, Any],
    wetting_params: dict[str, float],
    init_snapshot: str,
) -> dict[str, Any]:
    gravity_raw = deepcopy(base_raw)
    gravity_raw["init_type"] = "init_from_file"
    gravity_raw["init_dir"] = init_snapshot
    gravity_raw.setdefault("wetting_config", {}).update(wetting_params)
    return gravity_raw


def _report_equilibrium(iters: np.ndarray, values: np.ndarray) -> None:
    """Print the max|u| trend so the operator can judge Phase 1 convergence."""
    final = float(values[-1])
    peak_index = int(values.argmax())
    console.print("  Equilibration     : max|u|")
    console.print(f"    peak            : {float(values[peak_index]):.3e} (t={int(iters[peak_index])})")
    console.print(f"    final           : {final:.3e} (t={int(iters[-1])})")

    if len(values) < 2:  # noqa: PLR2004
        return
    previous = float(values[-2])
    if np.isclose(previous, 0.0, rtol=1e-09, atol=1e-09):
        return
    change = (final - previous) / previous
    console.print(f"    last interval   : {change:+.1%}")
    if abs(change) > _EQUILIBRIUM_REL_TOL:
        console.print("[yellow]    max|u| is still changing; consider a longer --init-wetting NT.[/yellow]")


def _plot_wetting_init_convergence(init_config: SimulationConfig, data_dir: str) -> None:
    """Render max|u| vs timestep for Phase 1 and summarise the trend.

    The equilibration run is only useful if it actually settled, and a single
    end-state snapshot cannot show that — this reads the saved history instead.
    """
    from src.simulation_io.plotting import MaxVelocityPlot
    from src.simulation_io.plotting.figure_builder import FigureBuilder

    run_dir = Path(data_dir).parent
    builder = FigureBuilder(init_config, run_dir, fields=[_CONVERGENCE_FIELD])
    files = [fp for _, fp in builder.sorted_timed_files()]
    if not files:
        console.print("[yellow]  No snapshots saved; skipping the equilibration plot.[/yellow]")
        return

    saved = builder.build_analysis(files)
    for path in saved:
        console.print(f"  Convergence plot  : {path}")

    series = MaxVelocityPlot(config=init_config).compute(files)
    if len(series["values"]):
        _report_equilibrium(series["iters"], series["values"])


def _warn_if_phase1_saves_nothing(init_raw: dict[str, Any]) -> None:
    """Warn when Phase 1's spacing is wider than its length, so it writes no snapshot.

    ``_build_wetting_init_raw`` derives ``save_interval`` from ``--init-wetting NT``,
    so an ``--override-phase1 nt=...`` shortens the run without rescaling the
    spacing. Phase 2 seeds from Phase 1's last snapshot, so the failure is a
    ``FileNotFoundError`` an hour into the equilibration rather than at setup.
    """
    nt = init_raw.get("nt")
    save_interval = init_raw.get("save_interval")
    if not isinstance(nt, int) or not isinstance(save_interval, int) or save_interval <= nt:
        return
    console.print(
        f"[yellow]Phase 1 saves every {save_interval} steps but only runs {nt}: "
        f"no snapshot would be written and Phase 2 could not seed from it. "
        f"Use --init-wetting {nt}, or add --override-phase1 "
        f"save_interval={_wetting_init_save_interval(nt)}.[/yellow]",
    )
    console.print()


def _present_phase(
    raw: dict[str, Any],
    phase_name: str,
    title: str,
    *,
    no_prompt: bool,
    overview: bool,
    confirm: bool,
) -> SimulationConfig | None:
    """Expand and display one phase, then confirm it or override it in place.

    *raw* is that phase's own raw dict, so an override typed at the prompt is
    scoped to this phase alone — the interactive counterpart of
    ``--override-phase1`` / ``--override-phase2``. Returns ``None`` when the
    operator declines to start it.
    """

    def rebuild(candidate: dict[str, Any]) -> SimulationConfig:
        config = _expand_single_phase(candidate, phase_name)
        _display_config_summary(config)
        if overview:
            _display_full_overview(config)
        return config

    console.print(Panel.fit(title))
    config = rebuild(raw)
    if no_prompt or not confirm:
        return config
    return confirm_or_override(
        raw,
        config,
        prompt=lambda _: f"[bold]Start {phase_name}?[/bold]",
        rebuild=rebuild,
    )


def _run_two_phase_wetting_init(
    config_path: str,
    overrides: tuple[str, ...],
    *,
    no_prompt: bool,
    overview: bool,
    init_nt: int = _WETTING_INIT_NT,
    override_phase1: tuple[str, ...] = (),
    override_phase2: tuple[str, ...] = (),
    dry_run: bool = False,
) -> None:
    """Two-phase wetting initialisation.

    Phase 1 equilibrates the droplet without gravity for *init_nt* steps,
    saving ``_WETTING_INIT_SNAPSHOTS`` snapshots so max|u| can be plotted
    against time as an equilibrium check. Phase 2 then runs with gravity,
    initialised from Phase 1's last snapshot.

    *overrides* apply to both phases, being folded into the base config before
    either is built. *override_phase1* / *override_phase2* apply to one phase
    each and are folded in **after** that phase's raw dict is built, so they
    beat the values this module sets itself — ``_build_wetting_init_raw``
    rewrites ``sim_type``, ``nt``, ``save_interval`` and the rest, which would
    otherwise silently discard a matching ``--override``.
    """
    # Imported here rather than at module scope: execution.py imports this
    # module for _run_impl, so a top-level import would close a cycle.
    from src.cli.execution import _run_simulation
    from src.config.adapter_toml import TomlAdapter

    console.print(f"[cyan]Loading configuration from:[/cyan] {config_path}")
    base_raw = TomlAdapter().load_raw(config_path)
    _apply_overrides(base_raw, overrides)

    wetting_params = _prompt_wetting_params(base_raw, no_prompt=no_prompt)
    init_raw = _build_wetting_init_raw(base_raw, wetting_params, init_nt)
    _apply_overrides(init_raw, override_phase1)
    _warn_if_phase1_saves_nothing(init_raw)

    init_config = _present_phase(
        init_raw,
        "Phase 1",
        "[bold cyan]Phase 1 - wetting equilibration (no gravity)[/bold cyan]",
        no_prompt=no_prompt,
        overview=overview,
        confirm=not dry_run,
    )
    if init_config is None:
        console.print("[yellow]Cancelled.[/yellow]")
        return

    if dry_run:
        init_snapshot = _DRY_RUN_SNAPSHOT
    else:
        init_data_dir = _run_simulation(init_config)
        _plot_wetting_init_convergence(init_config, init_data_dir)
        # The last snapshot on disk, not ``timestep_{init_nt}``: an nt that is not a
        # multiple of the save interval never writes a snapshot at nt itself.
        init_snapshot = str(_latest_snapshot_in(Path(init_data_dir), context="--init-wetting Phase 2"))
    console.print(f"  Init snapshot     : {init_snapshot}")
    console.print()

    gravity_raw = _build_wetting_gravity_raw(base_raw, wetting_params, init_snapshot)
    _apply_overrides(gravity_raw, override_phase2)

    gravity_config = _present_phase(
        gravity_raw,
        "Phase 2",
        "[bold cyan]Phase 2 - full simulation with gravity[/bold cyan]",
        no_prompt=no_prompt,
        overview=overview,
        confirm=not dry_run,
    )
    if gravity_config is None:
        console.print("[yellow]Phase 2 cancelled.[/yellow]")
        return

    if dry_run:
        _print_dry_run_message(None)
        return

    _run_simulation(gravity_config)
