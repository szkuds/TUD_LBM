"""The mapping between the nested config layout and the :class:`SimulationConfig` fields.

Every adapter reads its format into nested sections (``[simulation_type]``,
``[multiphase]``, ``[*_force]``, ...) and hands them to :func:`load_sections`;
:func:`build_sections` is the reverse, for ``save``. The layout is config
knowledge, so it lives here and not on the adapter base class
(:class:`~src.protocols.ConfigAdapter`).
"""

from __future__ import annotations
import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import Any
from src.config.forces import FORCE_SCHEMAS
from src.config.legacy import upgrade
from src.config.simulation_config import CONFIG_SECTION
from src.config.simulation_config import SimulationConfig

#: Sections stored whole on the :class:`SimulationConfig` field of the same name.
_DICT_SECTIONS: tuple[str, ...] = (
    "multiphase",
    "boundary_conditions",
    "wetting",
    "hysteresis",
    "chemical_step",
    "obstacle",
    "initialisation",
)
#: Field names of the old layout. ``config_from_flat`` keeps an unknown key in
#: ``extra`` without a word, which would turn a flat ``kappa`` into a single-phase run.
_RETIRED_FIELDS: frozenset[str] = frozenset(
    {
        "sim_type",
        "type",
        "bc_config",
        "obstacle_config",
        "wetting_config",
        "hysteresis_config",
        "chemical_step_config",
        "eos",
        "kappa",
        "rho_l",
        "rho_v",
        "interface_width",
        "g",
        "a_eos",
        "b_eos",
        "r_eos",
        "t_eos",
        *FORCE_SCHEMAS,
    }
)


def _process_forces(raw: dict[str, Any], sim_table: dict[str, Any]) -> None:
    """Route every ``[*_force]`` section of *raw* into ``sim_table["forces"]``."""
    forces: dict[str, Any] = {}
    for key, value in raw.items():
        if not key.endswith("_force"):
            continue
        if key not in FORCE_SCHEMAS:
            msg = f"Unknown force type '{key}'"
            raise KeyError(msg)
        if not isinstance(value, dict):
            msg = f"Force section '[{key}]' must be a table, got {type(value).__name__}."
            raise TypeError(msg)
        forces[key] = dict(value)
    if forces:
        sim_table["forces"] = forces


def _apply_output(sim_table: dict[str, Any], output: dict[str, Any]) -> None:
    """Merge ``[output]`` overrides into *sim_table* in-place."""
    for key, value in output.items():
        if key == "results_dir":
            sim_table[key] = str(Path(value).expanduser())
        else:
            sim_table[key] = value


def merge_sections(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested section dict into a SimulationConfig kwarg dict."""
    sim_table = dict(raw.get("simulation_type", {}))
    if not sim_table:
        msg = "Config is missing the required 'simulation_type' section."
        raise ValueError(msg)

    if "grid_shape" in sim_table and not isinstance(sim_table["grid_shape"], list):
        sim_table["grid_shape"] = tuple(sim_table["grid_shape"])

    if "k_diag" in sim_table and isinstance(sim_table["k_diag"], list):
        sim_table["k_diag"] = tuple(sim_table["k_diag"])

    for section in _DICT_SECTIONS:
        if section in raw:
            sim_table[section] = dict(raw[section])

    _process_forces(raw, sim_table)
    _apply_output(sim_table, raw.get("output", {}))
    return sim_table


def load_sections(raw: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """The flat kwarg dict of nested *raw*, and whether *raw* was in the old layout."""
    upgraded, changed = upgrade(raw)
    return merge_sections(upgraded), changed


def serialise_safe(value: Any) -> Any:  # noqa: ANN401
    """Convert tuples to lists and recursively process nested structures."""
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, dict):
        return {k: serialise_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialise_safe(v) for v in value]
    return value


def build_sections(config: SimulationConfig) -> dict[str, Any]:
    """Build a format-agnostic nested dict from *config*, routed by CONFIG_SECTION metadata."""
    sections = {f.name: f.metadata.get(CONFIG_SECTION, "simulation_type") for f in dataclasses.fields(SimulationConfig)}

    buckets: dict[str, dict[str, Any]] = defaultdict(dict)
    for key, value in dataclasses.asdict(config).items():
        section = sections.get(key, "simulation_type")
        if value is None or section == "extra":
            continue
        if key == "forces":
            buckets.update({name: serialise_safe(params) for name, params in value.items()})
        elif isinstance(value, dict):
            buckets[section].update(serialise_safe(value))
        else:
            buckets[section][key] = serialise_safe(value)

    for ek, ev in (config.extra or {}).items():
        buckets["simulation_type"][ek] = serialise_safe(ev)

    return {
        "simulation_type": buckets.pop("simulation_type", {}),
        **{k: buckets[k] for k in sorted(buckets) if buckets[k]},
    }


def config_from_flat(flat: dict[str, Any]) -> SimulationConfig:
    """A validated :class:`SimulationConfig` from a flat dict, unknown keys kept in ``extra``."""
    flat.pop("simulation_type", None)

    if "grid_shape" in flat and not isinstance(flat["grid_shape"], tuple):
        flat["grid_shape"] = tuple(flat["grid_shape"])

    retired = sorted(flat.keys() & _RETIRED_FIELDS)
    if retired:
        msg = f"Old-layout field(s) {', '.join(retired)}: use the [multiphase], [wetting], ... sections and `forces`."
        raise ValueError(msg)

    known_fields = {f.name for f in dataclasses.fields(SimulationConfig)}
    config_kwargs: dict[str, Any] = {}
    extra: dict[str, Any] = dict(flat.get("extra", {}))
    for k, v in flat.items():
        if k == "extra":
            continue
        if k in known_fields:
            config_kwargs[k] = v
        else:
            extra[k] = v
    config_kwargs["extra"] = extra
    return SimulationConfig(**config_kwargs)
