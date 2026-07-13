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
from dataclasses import dataclass
from dataclasses import field
from itertools import product
from typing import TYPE_CHECKING
from typing import Any
from tud_lbm.config.simulation_config import SimulationConfig
from tud_lbm.config.simulation_config import get_array_eligible_fields
from tud_lbm.config.simulation_config import get_nested_sweepable_fields

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator

# Constants for grid_shape validation
GRID_SHAPE_TUPLE_LEN = 2  # grid_shape is a (nx, ny) tuple


@dataclass(frozen=True)
class ArrayParameterSet:
    """Metadata about which fields had arrays in the original config.

    Attributes:
        field_names: Names of fields that contained arrays.
        array_values: Mapping from field name to the array of values.
        total_combinations: Total number of config combinations generated.
    """

    field_names: frozenset[str] | Iterable[str] = field(default_factory=frozenset)
    array_values: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    total_combinations: int = 0

    def __post_init__(self) -> None:  # noqa: D105
        if not isinstance(self.field_names, frozenset):
            object.__setattr__(self, "field_names", frozenset(self.field_names))


def detect_array_fields(_config: SimulationConfig) -> ArrayParameterSet | None:
    """Detect which fields in a config contain array values.

    Returns None if no arrays are present.

    Args:
        _config: A :class:`SimulationConfig` (should only contain scalars
            after full expansion; this is more for validation/inspection).

    Returns:
        :class:`ArrayParameterSet` describing detected arrays, or None
        if no arrays were found.
    """
    # Note: After expansion, configs are scalar-only. This function is
    # primarily for inspecting the raw metadata before expansion.
    return None


# ── grid_shape helpers ────────────────────────────────────────────────────────


def _is_scalar_grid_shape(value: Any) -> bool:  # noqa: ANN401
    """Return True if *value* is a single (nx, ny) integer tuple."""
    return isinstance(value, tuple) and len(value) == GRID_SHAPE_TUPLE_LEN and all(isinstance(x, int) for x in value)


def _is_scalar_grid_shape_list(value: Any) -> bool:  # noqa: ANN401
    """Return True if *value* is a flat list of ints (not yet converted)."""
    return isinstance(value, list) and all(isinstance(x, int) for x in value)


def _is_array_grid_shape(value: Any) -> bool:  # noqa: ANN401
    """Return True if *value* is a list/tuple of (nx, ny) tuples."""
    return isinstance(value, (list, tuple)) and bool(value) and isinstance(value[0], tuple)


def _is_array_value(value: Any, field_name: str = "") -> bool:  # noqa: ANN401
    """Check if a value should be treated as an array for expansion.

    Special case: grid_shape can be a 2-tuple of ints (scalar) or a
    list/tuple of tuples (array). We distinguish:
    - (100, 100)              → scalar grid_shape
    - [(64, 64), (128, 128)]  → array of grid_shapes
    """
    if field_name == "grid_shape":
        return _is_array_grid_shape(value)

    return isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes))


# ── nested-axes helpers ───────────────────────────────────────────────────────


def _collect_list_subkeys(field_name: str, nested: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    """Return dotted-path → values for every list-valued sub-key in *nested*.

    Args:
        field_name: Parent field name (used to build dotted keys).
        nested: Sub-dict of a sweepable nested field.

    Returns:
        Mapping of ``"field_name.sub_key"`` to tuple of values.
    """
    return {f"{field_name}.{sk}": tuple(sv) for sk, sv in nested.items() if isinstance(sv, list) and sv}


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
        if isinstance(nested, dict):
            axes.update(_collect_list_subkeys(field_name, nested))
    return axes


def _apply_dotted_key(result: dict[str, Any], key: str, value: Any) -> None:  # noqa: ANN401
    """Apply a single dotted-path key into *result* in-place.

    Splits ``"parent.sub_key"`` and merges *value* into the nested dict
    stored at ``result["parent"]``, copying it first to avoid mutating
    shared state.

    Args:
        result: Config dict being built.
        key: Dotted key of the form ``"parent.sub_key"``.
        value: Value to assign at the sub-key.
    """
    parent, sub_key = key.split(".", 1)
    nested = dict(result.get(parent) or {})
    nested[sub_key] = value
    result[parent] = nested


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
            _apply_dotted_key(result, key, value)
        else:
            result[key] = value
    return result


# ── expansion internals ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class _ExpandResult:
    """Internal expansion artefacts shared by expand_config and enumerate_configs.

    Attributes:
        configs: Expanded list of :class:`SimulationConfig` objects.
        metadata: :class:`ArrayParameterSet` when arrays were found, else None.
        scalar_dict: Base scalar config dict (list values stripped).
        axis_keys: Ordered list of swept parameter names (plain or dotted).
        all_axes: Mapping from axis key to its tuple of values.
    """

    configs: list[SimulationConfig]
    metadata: ArrayParameterSet | None
    scalar_dict: dict[str, Any]
    axis_keys: list[str]
    all_axes: dict[str, tuple[Any, ...]]


def _collect_top_level_axes(
    config_dict: dict[str, Any],
    allow_arrays: bool,
) -> dict[str, tuple[Any, ...]]:
    """Collect and validate top-level array axes from *config_dict*.

    Iterates over array-eligible fields and returns those carrying list
    values.  Raises :exc:`ValueError` for any array found when
    *allow_arrays* is False.

    Args:
        config_dict: Raw configuration dict.
        allow_arrays: If False, raise on any detected array.

    Returns:
        Mapping of eligible field name → tuple of values.

    Raises:
        ValueError: If an array is found and *allow_arrays* is False.
    """
    eligible = get_array_eligible_fields()
    top_level_axes: dict[str, tuple[Any, ...]] = {}

    for key, value in config_dict.items():
        if key not in eligible:
            continue
        if not _is_array_value(value, field_name=key):
            continue
        if not allow_arrays:
            msg = (
                f"Array values found for field '{key}' but allow_arrays=False. Use allow_arrays=True or flatten config."
            )
            raise ValueError(msg)
        top_level_axes[key] = tuple(value)

    return top_level_axes


def _validate_nested_axes(
    nested_axes: dict[str, tuple[Any, ...]],
    allow_arrays: bool,
) -> None:
    """Raise :exc:`ValueError` if nested arrays are present but not allowed.

    Args:
        nested_axes: Mapping of dotted-path → values produced by
            :func:`_extract_nested_array_axes`.
        allow_arrays: If False and *nested_axes* is non-empty, raise.

    Raises:
        ValueError: If *nested_axes* is non-empty and *allow_arrays* is False.
    """
    if nested_axes and not allow_arrays:
        first = next(iter(nested_axes))
        msg = f"Array values found for nested field '{first}' but allow_arrays=False."
        raise ValueError(msg)


def _strip_nested_field(
    field_name: str,
    nested_dict: dict[str, Any],
    nested_axes: dict[str, tuple[Any, ...]],
) -> dict[str, Any]:
    """Return *nested_dict* with list sub-keys that are swept axes removed.

    Args:
        field_name: Parent field name (used to build dotted keys).
        nested_dict: Sub-dict of a sweepable nested field.
        nested_axes: Set of dotted-path keys that are being swept.

    Returns:
        Copy of *nested_dict* without the swept sub-keys.
    """
    return {sk: sv for sk, sv in nested_dict.items() if f"{field_name}.{sk}" not in nested_axes}


def _build_scalar_dict(
    config_dict: dict[str, Any],
    top_level_axes: dict[str, tuple[Any, ...]],
    nested_axes: dict[str, tuple[Any, ...]],
) -> dict[str, Any]:
    """Build a scalar base dict by stripping swept array values.

    Top-level keys that are axes are omitted entirely.  Nested sweepable
    dicts have their list sub-keys removed via :func:`_strip_nested_field`.

    Args:
        config_dict: Raw configuration dict.
        top_level_axes: Top-level fields identified as sweep axes.
        nested_axes: Dotted-path fields identified as nested sweep axes.

    Returns:
        Dict suitable for passing to ``SimulationConfig(**...)`` after a
        combo dict has been merged in.
    """
    scalar_dict: dict[str, Any] = {}
    for k, v in config_dict.items():
        if k in top_level_axes:
            continue
        if k in get_nested_sweepable_fields() and isinstance(v, dict):
            scalar_dict[k] = _strip_nested_field(k, v, nested_axes)
        else:
            scalar_dict[k] = v
    return scalar_dict


def _build_configs_and_metadata(
    scalar_dict: dict[str, Any],
    axis_keys: list[str],
    all_axes: dict[str, tuple[Any, ...]],
) -> tuple[list[SimulationConfig], ArrayParameterSet]:
    """Build the Cartesian-product configs and accompanying metadata.

    Args:
        scalar_dict: Base scalar config dict (no swept list values).
        axis_keys: Ordered list of axis names (plain or dotted).
        all_axes: Mapping from axis key to its tuple of values.

    Returns:
        Tuple of ``(configs, metadata)`` covering all parameter combinations.
    """
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


def _build_expand_result(
    config_dict: dict[str, Any],
    *,
    allow_arrays: bool,
) -> _ExpandResult:
    """Core expansion logic shared by :func:`expand_config` and :func:`enumerate_configs`.

    Args:
        config_dict: Raw configuration dict.
        allow_arrays: If False, raise ValueError if arrays are found.

    Returns:
        :class:`_ExpandResult` with all expansion artefacts.

    Raises:
        ValueError: If arrays are found but *allow_arrays* is False.
    """
    top_level_axes = _collect_top_level_axes(config_dict, allow_arrays)
    nested_axes = _extract_nested_array_axes(config_dict)
    _validate_nested_axes(nested_axes, allow_arrays)

    all_axes: dict[str, tuple[Any, ...]] = {**top_level_axes, **nested_axes}

    if not all_axes:
        config = SimulationConfig(**config_dict)
        return _ExpandResult(
            configs=[config],
            metadata=None,
            scalar_dict=dict(config_dict),
            axis_keys=[],
            all_axes={},
        )

    scalar_dict = _build_scalar_dict(config_dict, top_level_axes, nested_axes)
    axis_keys = list(all_axes.keys())
    configs, metadata = _build_configs_and_metadata(scalar_dict, axis_keys, all_axes)

    return _ExpandResult(
        configs=configs,
        metadata=metadata,
        scalar_dict=scalar_dict,
        axis_keys=axis_keys,
        all_axes=all_axes,
    )


# ── public API ────────────────────────────────────────────────────────────────


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
    result = _build_expand_result(config_dict, allow_arrays=allow_arrays)
    return result.configs, result.metadata


def _iter_combinations(
    result: _ExpandResult,
) -> Iterator[tuple[int, dict[str, Any], SimulationConfig]]:
    """Yield ``(index, parameters, config)`` for every axis combination in *result*.

    Args:
        result: Fully populated :class:`_ExpandResult` with at least one axis.

    Yields:
        ``(index, parameters, config)`` tuples.
    """
    axis_lists = [result.all_axes[k] for k in result.axis_keys]
    for idx, combo in enumerate(product(*axis_lists)):
        parameters = dict(zip(result.axis_keys, combo, strict=False))
        combo_dict = _apply_combo_to_dict(result.scalar_dict, parameters)
        yield idx, parameters, SimulationConfig(**combo_dict)


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
    result = _build_expand_result(config_dict, allow_arrays=allow_arrays)

    if result.metadata is None:
        yield 0, {}, result.configs[0]
        return

    yield from _iter_combinations(result)
