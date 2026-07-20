"""Two-phase wetting initialisation driven from the ``run`` command."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any
from rich.panel import Panel
from rich.prompt import Confirm
from rich.prompt import Prompt
from src.cli._console import console
from src.cli.config_loading import _expand_single_phase
from src.cli.display import _display_config_summary
from src.cli.display import _display_full_overview
from src.cli.overrides import _apply_overrides

_WETTING_INIT_NT = 50_000

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


def _build_wetting_init_raw(base_raw: dict[str, Any], wetting_params: dict[str, float]) -> dict[str, Any]:
    init_raw = deepcopy(base_raw)
    init_raw["sim_type"] = "multiphase_wetting"
    init_raw["init_type"] = "multiphase_bubbles"
    init_raw.pop("hysteresis_config", None)
    init_raw.pop("chemical_step_config", None)
    init_raw.pop("gravity_force", None)
    init_raw.pop("gravity_masked_force", None)
    if "bc_config" in init_raw:
        init_raw["bc_config"] = dict(init_raw["bc_config"])
    init_raw["nt"] = _WETTING_INIT_NT
    init_raw["save_interval"] = _WETTING_INIT_NT
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


def _run_two_phase_wetting_init(
    config_path: str,
    overrides: tuple[str, ...],
    *,
    no_prompt: bool,
    overview: bool,
) -> None:
    """Two-phase wetting initialisation.

    Phase 1 equilibrates the droplet without gravity for ``_WETTING_INIT_NT``
    steps and writes one final snapshot. Phase 2 then runs with gravity,
    initialised from the Phase 1 snapshot.
    """
    # Imported here rather than at module scope: execution.py imports this
    # module for _run_impl, so a top-level import would close a cycle.
    from src.cli.execution import _run_simulation
    from src.config.adapter_toml import TomlAdapter

    console.print(f"[cyan]Loading configuration from:[/cyan] {config_path}")
    base_raw = TomlAdapter().load_raw(config_path)
    _apply_overrides(base_raw, overrides)

    wetting_params = _prompt_wetting_params(base_raw, no_prompt=no_prompt)
    init_raw = _build_wetting_init_raw(base_raw, wetting_params)
    init_config = _expand_single_phase(init_raw, "Phase 1")

    console.print(Panel.fit("[bold cyan]Phase 1 - wetting equilibration (no gravity)[/bold cyan]"))
    _display_config_summary(init_config)
    if overview:
        _display_full_overview(init_config)

    if not no_prompt and not Confirm.ask("[bold]Start Phase 1?[/bold]", default=True):
        console.print("[yellow]Cancelled.[/yellow]")
        return

    init_data_dir = _run_simulation(init_config)
    init_snapshot = str(Path(init_data_dir) / f"timestep_{_WETTING_INIT_NT}.npz")
    console.print(f"  Init snapshot     : {init_snapshot}")
    console.print()

    gravity_raw = _build_wetting_gravity_raw(base_raw, wetting_params, init_snapshot)
    gravity_config = _expand_single_phase(gravity_raw, "Phase 2")

    console.print(Panel.fit("[bold cyan]Phase 2 - full simulation with gravity[/bold cyan]"))
    _display_config_summary(gravity_config)
    if overview:
        _display_full_overview(gravity_config)

    if not no_prompt and not Confirm.ask("[bold]Start Phase 2?[/bold]", default=True):
        console.print("[yellow]Phase 2 cancelled.[/yellow]")
        return

    _run_simulation(gravity_config)
