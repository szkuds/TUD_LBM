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
from config.simulation_config import SimulationConfig
from config.simulation_config import get_array_eligible_fields


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


def expand_config(
    config_dict: dict[str, Any],
    *,
    allow_arrays: bool = True,
) -> tuple[list[SimulationConfig], ArrayParameterSet | None]:
    """Expand a config dict with array fields into multiple configs.

    Performs Cartesian product expansion over all array-valued fields
    that are in :const:`ARRAY_ELIGIBLE_FIELDS`.

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
        ValueError: If arrays are found but *allow_arrays* is False,
            or if array field is not in :const:`ARRAY_ELIGIBLE_FIELDS`.
        TypeError: If array values contain incompatible types.
    """
    # Identify array fields
    array_fields: dict[str, tuple[Any, ...]] = {}

    eligible = get_array_eligible_fields()
    for key, value in list(config_dict.items()):
        if key not in eligible:
            continue
        if _is_array_value(value, field_name=key):
            if not allow_arrays:
                raise ValueError(
                    f"Array values found for field '{key}' but allow_arrays=False. "
                    f"Use allow_arrays=True or flatten config.",
                )

            # Convert to tuple for consistent iteration
            array_fields[key] = tuple(value)

    # If no arrays, return single config
    if not array_fields:
        config = SimulationConfig(**config_dict)
        return [config], None

    # Extract scalar values and array fields
    scalar_dict = {k: v for k, v in config_dict.items() if k not in array_fields}
    array_keys = list(array_fields.keys())
    array_lists = [array_fields[k] for k in array_keys]

    # Generate all combinations
    combinations = list(product(*array_lists))
    configs: list[SimulationConfig] = [
        SimulationConfig(**{**scalar_dict, **dict(zip(array_keys, combo, strict=False))}) for combo in combinations
    ]

    metadata = ArrayParameterSet(
        field_names=frozenset(array_keys),
        array_values=array_fields,
        total_combinations=len(combinations),
    )

    return configs, metadata


def enumerate_configs(
    config_dict: dict[str, Any],
    *,
    allow_arrays: bool = True,
) -> Iterator[tuple[int, dict[str, Any], SimulationConfig]]:
    """Yield (index, parameters, config) tuples for each expansion.

    Useful for logging which parameter combination corresponds to each
    simulation.

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
        # No arrays; single config
        yield 0, {}, configs[0]
        return

    # Generate combinations again for traceability
    array_fields = metadata.array_values
    scalar_dict = {k: v for k, v in config_dict.items() if k not in array_fields}
    array_keys = list(array_fields.keys())
    array_lists = [array_fields[k] for k in array_keys]

    for idx, combo in enumerate(product(*array_lists)):
        parameters = dict(zip(array_keys, combo, strict=False))
        combo_dict = dict(scalar_dict)
        combo_dict.update(parameters)
        config = SimulationConfig(**combo_dict)
        yield idx, parameters, config
