"""Old-layout configs, upgraded on load.

Removing this at the next version bump is deleting this file and its one call in
:func:`src.config.sections.load_sections`. Nothing else imports it.

The old layout selected the phase with ``[simulation_type] type``, and a
single-phase run ignored any ``[multiphase]`` section. It also let flat keys in
``[simulation_type]`` stand in for whole sections, let ``[multiphase]`` carry any
field, and accepted short wall-parameter names in ``[wetting]``. A config without
any of those is already in the new layout, including one with a ``[multiphase]``
section and no ``type``, which the old loader would have run as single-phase.
"""

from __future__ import annotations
from copy import deepcopy
from typing import Any
from src.config.multiphase import EOS_PARAMETERS
from src.config.multiphase import OPTIONAL
from src.config.multiphase import REQUIRED

_SIM_TYPES: tuple[str, ...] = (
    "single_phase",
    "multiphase",
    "multiphase_wetting",
    "multiphase_hysteresis",
    "multiphase_hysteresis_chemical_step",
)
_TYPE_KEYS: tuple[str, ...] = ("type", "sim_type")
_SECTION_KEYS: dict[str, str] = {
    "bc_config": "boundary_conditions",
    "wetting_config": "wetting",
    "hysteresis_config": "hysteresis",
    "chemical_step_config": "chemical_step",
    "obstacle_config": "obstacle",
}
_MULTIPHASE_KEYS: frozenset[str] = frozenset(
    {*REQUIRED, *OPTIONAL, *(k for row in EOS_PARAMETERS.values() for k in row)}
)
_WALL_KEYS: dict[str, str] = {
    "phi_l": "phi_left",
    "phi_r": "phi_right",
    "d_rho_l": "d_rho_left",
    "d_rho_r": "d_rho_right",
}


def _is_old_layout(table: dict[str, Any], raw: dict[str, Any]) -> bool:
    return (
        any(key in table for key in (*_TYPE_KEYS, *_SECTION_KEYS, *_MULTIPHASE_KEYS))
        or any(key.endswith("_force") for key in table)
        or any(key in _WALL_KEYS for key in raw.get("wetting", {}))
    )


def _sim_type(table: dict[str, Any]) -> str | None:
    """Remove and return the old phase selector, ``None`` when the table has none."""
    found = [table.pop(key) for key in _TYPE_KEYS if key in table]
    sim_type = found[0] if found else None
    if sim_type is not None and sim_type not in _SIM_TYPES:
        msg = f"Unknown simulation type '{sim_type}'. Expected one of: {', '.join(_SIM_TYPES)}."
        raise ValueError(msg)
    return sim_type


def _lift_sections(table: dict[str, Any], raw: dict[str, Any]) -> None:
    """Move flat stand-ins for whole sections out of ``[simulation_type]``."""
    for key, section in _SECTION_KEYS.items():
        if key in table:
            raw[section] = table.pop(key)
    for key in [key for key in table if key.endswith("_force")]:
        raw[key] = table.pop(key)


def _upgrade_multiphase(table: dict[str, Any], raw: dict[str, Any], sim_type: str | None) -> None:
    """Rebuild ``[multiphase]`` from what the old merge of the two tables meant."""
    section = {key: table.pop(key) for key in _MULTIPHASE_KEYS & table.keys()}
    merged = {**section, **raw.pop("multiphase", {})}
    if sim_type == "single_phase" or (sim_type is None and not merged):
        return
    for key in merged.keys() - _MULTIPHASE_KEYS:
        table[key] = merged.pop(key)
    raw["multiphase"] = merged


def upgrade(raw: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """The nested sections of *raw* in the new layout, and whether anything changed.

    Idempotent, and *raw* is left untouched.

    Raises:
        ValueError: On an unknown ``type``.
    """
    table = raw.get("simulation_type")
    if not isinstance(table, dict) or not _is_old_layout(table, raw):
        return raw, False

    upgraded = deepcopy(raw)
    table = upgraded["simulation_type"]
    sim_type = _sim_type(table)
    _lift_sections(table, upgraded)
    _upgrade_multiphase(table, upgraded, sim_type)
    if "wetting" in upgraded:
        upgraded["wetting"] = {_WALL_KEYS.get(key, key): value for key, value in upgraded["wetting"].items()}
    return upgraded, True
