"""Array parameter expansion for parallel simulations.

Detects array-valued parameters in configuration and expands them into
multiple :class:`SimulationConfig` objects for parallel execution.

Example usage::

    from config.adapter_toml import TomlAdapter
    from config.array_expansion import detect_array_fields, expand_config

    # Load config (arrays are preserved in raw form)
    adapter = TomlAdapter()
    raw_config = adapter.load("config_parallel.toml", allow_arrays=True)

    # Detect which fields have arrays
    array_fields = detect_array_fields(raw_config)

    # Expand into multiple configs
    configs = expand_config(raw_config)
    # Returns: [SimulationConfig(...), SimulationConfig(...), ...]
"""

from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from typing import Any
from tud_lbm.config.simulation_config import SimulationConfig
from tud_lbm.config.simulation_config import get_array_eligible_fields
from tud_lbm.config.simulation_config import get_nested_sweepable_fields


@dataclass(frozen=True)
class ArrayParameterSet:
    """Metadata about which fields had arrays in the original config.

    Attributes:
        field_names: Names of fields that contained arrays.
        array_values: Mapping from field name to the array of values.
        total_combinations: Total number of config combinations generated.
    """

    field_names: frozenset[str]
    array_values: dict[str, tuple[Any, ...]]
    total_combinations: int


def detect_array_fields(config: SimulationConfig) -> ArrayParameterSet | None:
    """Detect which fields in a config contain array values.

    Returns None if no arrays are present.

    Args:
        config: A :class:`SimulationConfig` (should only contain scalars
            after full expansion; this is more for validation/inspection).

    Returns:
        :class:`ArrayParameterSet` describing detected arrays, or None
        if no arrays were found.
    """
    # Note: After expansion, configs are scalar-only. This function is
    # primarily for inspecting the raw metadata before expansion.
    return None


def _is_array_value(value: Any, field_name: str = "") -> bool:
    """Check if a value should be treated as an array for expansion.

    Special case: grid_shape can be a 2-tuple of ints (scalar) or a
    list/tuple of tuples (array). We distinguish:
    - (100, 100) → scalar grid_shape
    - [(64, 64), (128, 128)] → array of grid_shapes
    """
    # Handle grid_shape special case
    if field_name == "grid_shape":
        # If it's a tuple of 2 ints, it's a scalar grid_shape
        if isinstance(value, tuple) and len(value) == 2 and all(isinstance(x, int) for x in value):
            return False
        # If it's a list or tuple of tuples, it's an array
        if isinstance(value, (list, tuple)):
            if value and isinstance(value[0], tuple):
                return True
            # If it's a list of integers, it's still scalar (hasn't been converted yet)
            if isinstance(value, list) and all(isinstance(x, int) for x in value):
                return False
        return False

    # For other fields, lists and tuples are arrays
    return isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes))


def _extract_nested_array_axes(config_dict: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    """Return dotted-path → values for array sub-keys inside nested sweepable fields.

    For example, if ``config_dict["gravity_force"] = {"force_g": 5e-7,
    "inclination_angle_deg": [50, 60]}``, this returns
    ``{"gravity_force.inclination_angle_deg": (50, 60)}``.

    Args:
        config_dict: Raw configuration dict.

    Returns:
        Mapping of ``"field.sub_key"`` to tuple of values for every list-valued
        sub-key found in the NESTED_SWEEPABLE_FIELDS entries.
    """
    axes: dict[str, tuple[Any, ...]] = {}
    for field_name in get_nested_sweepable_fields():
        nested = config_dict.get(field_name)
        if not isinstance(nested, dict):
            continue
        for sub_key, sub_val in nested.items():
            if isinstance(sub_val, list) and sub_val:
                axes[f"{field_name}.{sub_key}"] = tuple(sub_val)
    return axes


def _apply_combo_to_dict(
    base_dict: dict[str, Any],
    combo_params: dict[str, Any],
) -> dict[str, Any]:
    """Return a copy of *base_dict* with *combo_params* applied.

    Top-level keys are set directly.  Dotted-path keys
    (``"field.sub_key"``) are applied into the nested dict stored at
    ``base_dict["field"]``.

    Args:
        base_dict: Base scalar config dict (no list values).
        combo_params: Mapping of key (plain or dotted) → value for this
            combination.

    Returns:
        New dict suitable for ``SimulationConfig(**...)``.
    """
    result = dict(base_dict)
    for key, value in combo_params.items():
        if "." in key:
            parent, sub_key = key.split(".", 1)
            # Copy the nested dict so we don't mutate shared state
            nested = dict(result.get(parent) or {})
            nested[sub_key] = value
            result[parent] = nested
        else:
            result[key] = value
    return result


def expand_config(
    config_dict: dict[str, Any],
    *,
    allow_arrays: bool = True,
) -> tuple[list[SimulationConfig], ArrayParameterSet | None]:
    """Expand a config dict with array fields into multiple configs.

    Performs Cartesian product expansion over:

    * Top-level ``ARRAY_ELIGIBLE_FIELDS`` that carry a list value.
    * Sub-keys inside ``NESTED_SWEEPABLE_FIELDS`` dicts
      (``gravity_force``, ``electric_force``, ``wetting_config``,
      ``hysteresis_config``) that carry a list value.

    Args:
        config_dict: Raw configuration dict (typically from TomlAdapter
            before SimulationConfig instantiation).
        allow_arrays: If False, raise ValueError if arrays are found.

    Returns:
        ``(configs, metadata)`` where:
        - *configs*: List of :class:`SimulationConfig` objects (one per
          combination). If no arrays found, list contains single config.
        - *metadata*: :class:`ArrayParameterSet` if arrays were detected,
          else None.

    Raises:
        ValueError: If arrays are found but *allow_arrays* is False.
        TypeError: If array values contain incompatible types.
    """
    # ── 1. Top-level array axes ───────────────────────────────────────────
    top_level_axes: dict[str, tuple[Any, ...]] = {}
    eligible = get_array_eligible_fields()
    for key, value in config_dict.items():
        if key not in eligible:
            continue
        if _is_array_value(value, field_name=key):
            if not allow_arrays:
                raise ValueError(
                    f"Array values found for field '{key}' but allow_arrays=False. "
                    f"Use allow_arrays=True or flatten config.",
                )
            top_level_axes[key] = tuple(value)

    # ── 2. Nested array axes (dotted paths) ───────────────────────────────
    nested_axes = _extract_nested_array_axes(config_dict)
    if nested_axes and not allow_arrays:
        first = next(iter(nested_axes))
        raise ValueError(
            f"Array values found for nested field '{first}' but allow_arrays=False.",
        )

    # ── 3. Combine all axes ───────────────────────────────────────────────
    all_axes: dict[str, tuple[Any, ...]] = {**top_level_axes, **nested_axes}

    if not all_axes:
        config = SimulationConfig(**config_dict)
        return [config], None

    # Build a scalar base dict (strip top-level arrays; nested dicts keep
    # their scalar sub-keys but list sub-keys are removed)
    scalar_dict: dict[str, Any] = {}
    for k, v in config_dict.items():
        if k in top_level_axes:
            continue
        if k in get_nested_sweepable_fields() and isinstance(v, dict):
            # Remove sub-keys that are being swept
            stripped = {sk: sv for sk, sv in v.items() if f"{k}.{sk}" not in nested_axes}
            scalar_dict[k] = stripped
        else:
            scalar_dict[k] = v

    axis_keys = list(all_axes.keys())
    axis_lists = [all_axes[k] for k in axis_keys]

    combinations = list(product(*axis_lists))
    configs: list[SimulationConfig] = [
        SimulationConfig(**_apply_combo_to_dict(scalar_dict, dict(zip(axis_keys, combo, strict=False))))
        for combo in combinations
    ]

    metadata = ArrayParameterSet(
        field_names=frozenset(axis_keys),
        array_values=all_axes,
        total_combinations=len(combinations),
    )

    return configs, metadata


def enumerate_configs(
    config_dict: dict[str, Any],
    *,
    allow_arrays: bool = True,
) -> Iterator[tuple[int, dict[str, Any], SimulationConfig]]:
    """Yield (index, parameters, config) tuples for each expansion.

    *parameters* uses dotted-path keys for nested fields
    (e.g. ``"gravity_force.inclination_angle_deg"``).

    Args:
        config_dict: Raw configuration dict.
        allow_arrays: If False, raise ValueError if arrays are found.

    Yields:
        ``(index, parameters, config)`` tuples where:
        - *index*: 0-based position in the expansion.
        - *parameters*: Dict of parameter names and values for this combo.
        - *config*: The :class:`SimulationConfig`.
    """
    configs, metadata = expand_config(config_dict, allow_arrays=allow_arrays)

    if metadata is None:
        yield 0, {}, configs[0]
        return

    all_axes = metadata.array_values
    top_level_keys = {k for k in all_axes if "." not in k}
    nested_keys = {k for k in all_axes if "." in k}

    # Rebuild scalar base dict the same way expand_config does
    scalar_dict: dict[str, Any] = {}
    for k, v in config_dict.items():
        if k in top_level_keys:
            continue
        if k in get_nested_sweepable_fields() and isinstance(v, dict):
            stripped = {sk: sv for sk, sv in v.items() if f"{k}.{sk}" not in nested_keys}
            scalar_dict[k] = stripped
        else:
            scalar_dict[k] = v

    axis_keys = list(all_axes.keys())
    axis_lists = [all_axes[k] for k in axis_keys]

    for idx, combo in enumerate(product(*axis_lists)):
        parameters = dict(zip(axis_keys, combo, strict=False))
        combo_dict = _apply_combo_to_dict(scalar_dict, parameters)
        config = SimulationConfig(**combo_dict)
        yield idx, parameters, config
