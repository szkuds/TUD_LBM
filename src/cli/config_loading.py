"""Loading and validating simulation configs for the CLI."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import click
from rich.prompt import Prompt
from src.cli._console import console
from src.cli.overrides import _apply_overrides
from src.config import SimulationConfig
from src.config.run_config import CONFIG_FILENAME
from src.config.run_config import DATA_DIRNAME
from src.config.run_config import SNAPSHOT_GLOB
from src.config.run_config import SNAPSHOT_PREFIX


def _validate_cli_args(
    overrides: tuple[str, ...],
    config_path: str | None,
    *,
    init_wetting: bool = False,
    init_dir: str | None = None,
    continue_run: bool = False,
    override_phase1: tuple[str, ...] = (),
    override_phase2: tuple[str, ...] = (),
) -> None:
    """Reject option combinations that need a CONFIG_PATH but were given none.

    Raising here rather than inside ``run``'s body keeps this a click
    ``UsageError`` (exit code 2) rather than a generic command failure.
    """
    if continue_run and init_dir is not None:
        msg = "--continue cannot be used with --init-dir"
        raise click.UsageError(msg)
    # The phase overrides name the two --init-wetting phases, so without it
    # there is no phase to apply them to; silently dropping them would look
    # like an override that had no effect.
    if not init_wetting:
        for option, given in (("--override-phase1", override_phase1), ("--override-phase2", override_phase2)):
            if given:
                msg = f"{option} requires --init-wetting; use --override for a single-phase run"
                raise click.UsageError(msg)
    if config_path:
        return
    requires_config = (
        ("--override", overrides),
        ("--init-wetting", init_wetting),
        ("--init-dir", init_dir),
        ("--continue", continue_run),
    )
    # The phase overrides need no entry here: they already require
    # --init-wetting, which requires CONFIG_PATH just above.
    for option, given in requires_config:
        if given:
            msg = f"{option} requires CONFIG_PATH"
            raise click.UsageError(msg)


def _latest_snapshot_in(data_dir: Path, *, context: str) -> Path:
    """Return the highest-numbered NumPy snapshot in *data_dir*.

    *context* names the caller in the error message raised when the directory
    holds no ``timestep_<N>.npz`` files.
    """
    snapshots = [
        (int(step), path)
        for path in data_dir.glob(SNAPSHOT_GLOB)
        if path.is_file() and (step := path.stem.removeprefix(SNAPSHOT_PREFIX)).isdigit()
    ]
    if not snapshots:
        msg = f"No saved snapshots found in {data_dir}; {context} requires timestep_<N>.npz files."
        raise FileNotFoundError(msg)
    return max(snapshots, key=lambda snapshot: snapshot[0])[1]


def _find_latest_snapshot(config_path: str) -> Path:
    """Return the highest-numbered NumPy snapshot saved beside *config_path*."""
    data_dir = Path(config_path).expanduser().resolve().parent / DATA_DIRNAME
    return _latest_snapshot_in(data_dir, context="--continue")


def _load_raw_config(
    config_path: str,
    overrides: tuple[str, ...],
    *,
    init_dir: str | None = None,
    continue_run: bool = False,
) -> dict[str, Any]:
    """Load a TOML file, apply resume defaults and CLI overrides; return the raw dict."""
    from src.config.adapter_toml import TomlAdapter

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


def _build_config(raw_config: dict[str, Any]) -> SimulationConfig:
    """The validated config of a raw dict. A list where a sweep axis used to be raises ``ValueError``."""
    return SimulationConfig(**raw_config)


def _load_config_interactive() -> SimulationConfig:
    """Collect simulation parameters interactively."""
    console.print("[cyan]Interactive mode - creating default simulation config[/cyan]")

    grid_x = int(Prompt.ask("Grid size X", default="100"))
    grid_y = int(Prompt.ask("Grid size Y", default="100"))
    tau = float(Prompt.ask("Relaxation time (tau)", default="0.6"))
    nt = int(Prompt.ask("Number of timesteps", default="1000"))
    save_interval = int(Prompt.ask("Save interval", default=str(nt // 10)))

    return SimulationConfig(
        grid_shape=(grid_x, grid_y),
        tau=tau,
        nt=nt,
        save_interval=save_interval,
    )


def _validate_run_dir_has_config(run_dir: str) -> Path:
    """TRY301: raise lives here, outside the try-block in animate()."""
    config_path = Path(run_dir) / CONFIG_FILENAME
    if not config_path.exists():
        msg = f"No config.toml found in {run_dir}. Is this a valid run directory?"
        raise FileNotFoundError(msg)
    return config_path


def _load_single_config(config_toml: str, overrides: tuple[str, ...] = ()) -> SimulationConfig:
    """Load CONFIG_TOML, apply *overrides* and return its validated config."""
    return _build_config(_load_raw_config(config_toml, overrides))
