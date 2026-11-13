"""Old-layout configs, upgraded on load.

Removing this at the next version bump is deleting this file and its one call in
:func:`src.config.sections.load_sections`. Nothing else imports it.

The old layout selected the phase with ``[simulation_type] type``, and a
single-phase run ignored any ``[multiphase]`` section. It also let flat keys in
``[simulation_type]`` stand in for whole sections, let ``[multiphase]`` carry any
field, and accepted short wall-parameter names in ``[wetting]``. A config without
any of those is already in the new layout, including one with a ``[multiphase]``
section and no ``type``, which the old loader would have run as single-phase.

Two later changes are upgraded here too, whatever the layout:

* ``[hysteresis] carry_inactive_params`` was removed with the option. ``false`` (the
  default every saved config carries) is dropped; ``true`` moves to
  ``[simulation_type]``, so the run keeps the record of how it was made.
* A run directory's own ``config.toml`` from before ``chemical_step_pre_side``
  existed labelled its windows by position ("post" at higher ``x``). When its
  inclusion starts on the high side, the pre and post windows are swapped and the
  side is written out, so "pre" is the start surface as the current code reads it
  and each window stays on the surface it was on. Only a saved config is relabelled:
  it always carries the key once written by the current code, while an input TOML
  without the key is simply a new config.
"""

from __future__ import annotations
from copy import deepcopy
from typing import Any
from src.config.multiphase import EOS_PARAMETERS
from src.config.multiphase import OPTIONAL
from src.config.multiphase import REQUIRED
from src.config.wetting import TangentialSide
from src.config.wetting import start_side

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
_REMOVED_HYSTERESIS_KEY = "carry_inactive_params"
_WINDOW_KEYS: tuple[str, ...] = ("ca_advancing", "ca_receding")
_PRE_SIDE_KEY = "chemical_step_pre_side"


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


def _upgrade_layout(upgraded: dict[str, Any]) -> bool:
    """Move an old-layout ``[simulation_type]`` into the current sections, in place."""
    table = upgraded.get("simulation_type")
    if not isinstance(table, dict) or not _is_old_layout(table, upgraded):
        return False
    sim_type = _sim_type(table)
    _lift_sections(table, upgraded)
    _upgrade_multiphase(table, upgraded, sim_type)
    if "wetting" in upgraded:
        upgraded["wetting"] = {_WALL_KEYS.get(key, key): value for key, value in upgraded["wetting"].items()}
    return True


def _drop_removed_keys(upgraded: dict[str, Any]) -> bool:
    """Take the removed ``carry_inactive_params`` out of ``[hysteresis]``, in place."""
    section = upgraded.get("hysteresis")
    if not isinstance(section, dict) or _REMOVED_HYSTERESIS_KEY not in section:
        return False
    if section.pop(_REMOVED_HYSTERESIS_KEY):
        upgraded.setdefault("simulation_type", {})[_REMOVED_HYSTERESIS_KEY] = True
    return True


def _relabel_chemical_step(upgraded: dict[str, Any]) -> bool:
    """Swap the windows of a saved pre-``chemical_step_pre_side`` config that starts high, in place."""
    section = upgraded.get("chemical_step")
    table = upgraded.get("simulation_type")
    if not isinstance(section, dict) or not isinstance(table, dict) or _PRE_SIDE_KEY in section:
        return False
    grid_shape = table.get("grid_shape")
    location = section.get("chemical_step_location")
    if not isinstance(grid_shape, list | tuple) or not grid_shape or not isinstance(location, int | float):
        return False
    centres = upgraded.get("initialisation", {}).get("centres")
    if start_side(location, centres, int(grid_shape[0])) != TangentialSide.HIGH:
        return False
    for key in _WINDOW_KEYS:
        pre, post = f"{key}_pre_step", f"{key}_post_step"
        if pre in section and post in section:
            section[pre], section[post] = section[post], section[pre]
    section[_PRE_SIDE_KEY] = TangentialSide.HIGH.value
    return True


def upgrade(raw: dict[str, Any], *, saved: bool = False) -> tuple[dict[str, Any], bool]:
    """The nested sections of *raw* in the current layout, and whether anything changed.

    Idempotent, and *raw* is left untouched. *saved* marks a run directory's own
    ``config.toml``, the only file whose chemical-step windows are relabelled.

    Raises:
        ValueError: On an unknown ``type``.
    """
    upgraded = deepcopy(raw)
    changed = _upgrade_layout(upgraded)
    changed = _drop_removed_keys(upgraded) or changed
    if saved:
        changed = _relabel_chemical_step(upgraded) or changed
    return (upgraded, True) if changed else (raw, False)
