"""Loading, expanding and validating simulation configs for the CLI."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
import click
from rich.prompt import Prompt
from tud_lbm.cli._console import console
from tud_lbm.cli.overrides import _apply_overrides
from tud_lbm.config import SimulationConfig

if TYPE_CHECKING:
    from tud_lbm.config.array_expansion import ArrayParameterSet


def _validate_cli_args(
    overrides: tuple[str, ...],
    config_path: str | None,
    *,
    init_wetting: bool = False,
    init_dir: str | None = None,
    continue_run: bool = False,
) -> None:
    """Reject option combinations that need a CONFIG_PATH but were given none.

    Raising here rather than inside ``run``'s body keeps this a click
    ``UsageError`` (exit code 2) rather than a generic command failure.
    """
    if continue_run and init_dir is not None:
        msg = "--continue cannot be used with --init-dir"
        raise click.UsageError(msg)
    if config_path:
        return
    requires_config = (
        ("--override", overrides),
        ("--init-wetting", init_wetting),
        ("--init-dir", init_dir),
        ("--continue", continue_run),
    )
    for option, given in requires_config:
        if given:
            msg = f"{option} requires CONFIG_PATH"
            raise click.UsageError(msg)


def _find_latest_snapshot(config_path: str) -> Path:
    """Return the highest-numbered NumPy snapshot saved beside *config_path*."""
    data_dir = Path(config_path).expanduser().resolve().parent / "data"
    snapshots = [
        (int(step), path)
        for path in data_dir.glob("timestep_*.npz")
        if path.is_file() and (step := path.stem.removeprefix("timestep_")).isdigit()
    ]
    if not snapshots:
        msg = f"No saved snapshots found in {data_dir}; --continue requires timestep_<N>.npz files."
        raise FileNotFoundError(msg)
    return max(snapshots, key=lambda snapshot: snapshot[0])[1]


def _load_raw_config(
    config_path: str,
    overrides: tuple[str, ...],
    *,
    init_dir: str | None = None,
    continue_run: bool = False,
) -> dict[str, Any]:
    """Load a TOML file, apply resume defaults and CLI overrides; return the raw dict."""
    from tud_lbm.config.adapter_toml import TomlAdapter

    console.print(f"[cyan]Loading configuration from:[/cyan] {config_path}")
    raw_config = TomlAdapter().load_raw(config_path) or {}
    if continue_run:
        init_dir = str(_find_latest_snapshot(config_path))
        console.print(f"[cyan]Continuing from latest snapshot:[/cyan] {init_dir}")
    if init_dir is not None:
        raw_config["init_dir"] = str(init_dir)
        raw_config["init_type"] = "init_from_file"
    _apply_overrides(raw_config, overrides)
    return raw_config


def _expand_raw_config(
    raw_config: dict[str, Any],
) -> tuple[list[SimulationConfig], SimulationConfig | None, ArrayParameterSet | None, list[dict[str, Any]] | None]:
    """Expand a raw config dict; return (configs, config, sweep_metadata, parameters_list)."""
    from tud_lbm.config.array_expansion import enumerate_configs
    from tud_lbm.config.array_expansion import expand_config

    configs, sweep_metadata = expand_config(raw_config)
    if sweep_metadata is None:
        return configs, configs[0], None, None
    parameters_list = [params for _, params, _ in enumerate_configs(raw_config)]
    return configs, None, sweep_metadata, parameters_list


def _load_config_interactive() -> tuple[list[SimulationConfig], SimulationConfig, None, None]:
    """Collect simulation parameters interactively; return the same 4-tuple as _load_config_from_file."""
    console.print("[cyan]Interactive mode - creating default simulation config[/cyan]")

    grid_x = int(Prompt.ask("Grid size X", default="100"))
    grid_y = int(Prompt.ask("Grid size Y", default="100"))
    tau = float(Prompt.ask("Relaxation time (tau)", default="0.6"))
    nt = int(Prompt.ask("Number of timesteps", default="1000"))
    save_interval = int(Prompt.ask("Save interval", default=str(nt // 10)))

    config = SimulationConfig(
        grid_shape=(grid_x, grid_y),
        tau=tau,
        nt=nt,
        save_interval=save_interval,
    )
    return [config], config, None, None


def _expand_single_phase(raw_config: dict[str, Any], phase_name: str) -> SimulationConfig:
    from tud_lbm.config.array_expansion import expand_config

    configs, _ = expand_config(raw_config)
    if len(configs) != 1:
        msg = f"--init-wetting does not support parameter sweeps ({phase_name} expansion must yield exactly 1 config)."
        raise click.UsageError(msg)
    return configs[0]


def _validate_run_dir_has_config(run_dir: str) -> Path:
    """TRY301: raise lives here, outside the try-block in animate()."""
    config_path = Path(run_dir) / "config.toml"
    if not config_path.exists():
        msg = f"No config.toml found in {run_dir}. Is this a valid run directory?"
        raise FileNotFoundError(msg)
    return config_path


def _load_single_config(config_toml: str) -> SimulationConfig:
    """Load CONFIG_TOML and expand it to exactly one config; sweeps are rejected."""
    raw_config = _load_raw_config(config_toml, ())
    _configs, config, sweep_metadata, _params = _expand_raw_config(raw_config)
    if sweep_metadata is not None or config is None:
        msg = "analyse does not support parameter sweeps; remove list-valued fields from the config"
        raise click.UsageError(msg)
    return config
