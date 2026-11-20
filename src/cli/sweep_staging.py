"""Staging a parameter sweep as one completed TOML per combination.

A sweep is never run in process. Each combination of the ``--set`` axes is
built and validated as an ordinary :class:`SimulationConfig` and written with
:meth:`TomlAdapter.save`, the writer of a run directory's own ``config.toml``,
so every staged file shows the values its run will use. The files then go
through ``tud-lbm run`` or ``scripts/db_pipeline.sh`` like any other config.
"""

from __future__ import annotations
import string
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING
from typing import Any
import click
from src.cli.overrides import _normalise_override_path
from src.cli.overrides import _parse_override_argument
from src.cli.overrides import _reject_unknown_field
from src.cli.overrides import _set_nested_override
from src.config.run_config import CONFIG_FILENAME
from src.config.sections import config_from_flat

if TYPE_CHECKING:
    from pathlib import Path
    from src.config import SimulationConfig


@dataclass(frozen=True)
class Axis:
    """One swept parameter: its ``--set`` path, its template field and its values."""

    path: str
    field: str
    values: tuple[Any, ...]


@dataclass(frozen=True)
class StagedConfig:
    """One combination: the values it takes per axis path, and its validated config."""

    values: dict[str, Any]
    config: SimulationConfig


def parse_axes(sets: tuple[str, ...]) -> tuple[Axis, ...]:
    """The axes of ``--set PATH=[v1, v2, ...]`` expressions, in the order given."""
    axes: list[Axis] = []
    for expression in sets:
        try:
            path, values = _parse_override_argument(expression)
            parts = _normalise_override_path(path)
            _reject_unknown_field(path, parts[0])
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        if not isinstance(values, list) or not values:
            msg = f"--set {path} needs a non-empty TOML array, e.g. {path}=[0.8, 0.9]"
            raise click.UsageError(msg)
        if parts == ["grid_shape"]:
            values = [tuple(value) if isinstance(value, list) else value for value in values]
        axes.append(Axis(path=path, field=parts[-1], values=tuple(values)))

    fields = [axis.field for axis in axes]
    if duplicated := sorted({name for name in fields if fields.count(name) > 1}):
        msg = f"two --set axes share the name field {', '.join(duplicated)}; give each axis a distinct last key"
        raise click.UsageError(msg)
    return tuple(axes)


def _name_text(value: object) -> str:
    """*value* as it reads in a run name: ``0.85``, ``5e-07``, ``201x101``."""
    if isinstance(value, (list, tuple)):
        return "x".join(str(item) for item in value)
    return str(value)


def _default_template(base: str, axes: tuple[Axis, ...]) -> str:
    """``<base>_<field>_{<field>}`` for each axis."""
    return base + "".join(f"_{axis.field}_{{{axis.field}}}" for axis in axes)


def _check_template(template: str, axes: tuple[Axis, ...]) -> None:
    """Reject a ``--name`` template field that names no axis."""
    known = {axis.field for axis in axes}
    used = {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}
    if unknown := sorted(used - known):
        msg = f"--name field(s) {', '.join(unknown)} name no --set axis; use {', '.join(sorted(known))}"
        raise click.UsageError(msg)


def stage_configs(raw: dict[str, Any], axes: tuple[Axis, ...], template: str | None) -> list[StagedConfig]:
    """Build and validate every combination of *axes* over the flat *raw* config.

    *template* is a ``str.format`` pattern over the axis fields; ``None`` takes
    the base ``simulation_name`` with ``_<field>_<value>`` per axis.
    """
    if template is None:
        base = raw.get("simulation_name")
        if not base:
            msg = "the config has no simulation_name to build run names from; set one or pass --name"
            raise click.UsageError(msg)
        template = _default_template(str(base), axes)
    _check_template(template, axes)

    staged: list[StagedConfig] = []
    for combination in product(*(axis.values for axis in axes)):
        candidate = deepcopy(raw)
        for axis, value in zip(axes, combination, strict=True):
            _set_nested_override(candidate, axis.path, value)
        name = template.format(**{axis.field: _name_text(value) for axis, value in zip(axes, combination, strict=True)})
        candidate["simulation_name"] = name
        values = {axis.path: value for axis, value in zip(axes, combination, strict=True)}
        try:
            config = config_from_flat(candidate)
        except (TypeError, ValueError) as exc:
            msg = f"{name}: {exc}"
            raise ValueError(msg) from exc
        staged.append(StagedConfig(values=values, config=config))

    names = [entry.config.simulation_name for entry in staged]
    if duplicated := sorted({name for name in names if name is not None and names.count(name) > 1}):
        msg = f"the --name template gives the same run name more than once: {', '.join(duplicated)}"
        raise click.UsageError(msg)
    return staged


def staged_path(out_dir: Path, config: SimulationConfig) -> Path:
    """Where *config* is written: ``<out_dir>/<simulation_name>.toml``.

    Never ``config.toml``, which a load would treat as a run directory's own copy.
    """
    path = out_dir / f"{config.simulation_name}.toml"
    if path.name == CONFIG_FILENAME:
        msg = f"a staged run may not be named {path.stem!r}"
        raise click.UsageError(msg)
    return path


def write_staged(staged: list[StagedConfig], out_dir: Path, *, force: bool) -> list[Path]:
    """Write every staged config; refuse to overwrite a file unless *force*.

    Nothing is written when any target already exists without *force*.
    """
    from src.config.adapter_toml import TomlAdapter

    paths = [staged_path(out_dir, entry.config) for entry in staged]
    if not force and (existing := [str(path) for path in paths if path.exists()]):
        msg = f"already staged (pass --force to overwrite): {', '.join(existing)}"
        raise click.UsageError(msg)
    adapter = TomlAdapter()
    for entry, path in zip(staged, paths, strict=True):
        adapter.save(entry.config, str(path))
    return paths
