"""Base class for configuration file adapters."""

from __future__ import annotations
import dataclasses
import importlib
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any
from tud_lbm.config.simulation_config import CONFIG_SECTION
from tud_lbm.config.simulation_config import SimulationConfig


class ConfigAdapter(ABC):
    """Converts a config source into a SimulationConfig and back."""

    @abstractmethod
    def load_raw(self, source: Any) -> dict[str, Any]:  # noqa: ANN401
        """Parse *source* and return a flat config dict without instantiating SimulationConfig."""
        ...

    @abstractmethod
    def save(self, config: SimulationConfig, path: str) -> None:
        """Save a :class:`SimulationConfig` to *path*."""
        ...

    # ── shared section-merging helpers ──────────────────────────────────

    @staticmethod
    def _process_forces(raw: dict[str, Any], sim_table: dict[str, Any]) -> None:
        """Extract and validate force sections from *raw* into *sim_table*."""
        known_force_fields = {f.name for f in dataclasses.fields(SimulationConfig) if f.name.endswith("_force")}
        for key, value in raw.items():
            if not key.endswith("_force"):
                continue
            if key not in known_force_fields:
                msg = f"Unknown force type '{key}'"
                raise KeyError(msg)
            if not isinstance(value, dict):
                msg = f"Force section '[{key}]' must be a table, got {type(value).__name__}."
                raise TypeError(msg)
            sim_table[key] = dict(value)

    @staticmethod
    def _apply_output(sim_table: dict[str, Any], output: dict[str, Any]) -> None:
        """Merge ``[output]`` overrides into *sim_table* in-place."""
        for key, value in output.items():
            if key == "results_dir":
                sim_table[key] = str(Path(value).expanduser())
            else:
                sim_table[key] = value

    @staticmethod
    def _merge_sections(raw: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
        """Flatten a nested section dict into a SimulationConfig kwarg dict."""
        sim_table = dict(raw.get("simulation_type", {}))
        if not sim_table:
            msg = "Config is missing the required 'simulation_type' section."
            raise ValueError(msg)

        sim_type: str = sim_table.pop("type", "single_phase")

        if "grid_shape" in sim_table and not isinstance(sim_table["grid_shape"], list):
            sim_table["grid_shape"] = tuple(sim_table["grid_shape"])  # type: ignore[arg-type]

        if "k_diag" in sim_table and isinstance(sim_table["k_diag"], list):
            sim_table["k_diag"] = tuple(sim_table["k_diag"])  # type: ignore[arg-type]

        valid_types = (
            "single_phase",
            "multiphase",
            "multiphase_wetting",
            "multiphase_hysteresis",
            "multiphase_hysteresis_chemical_step",
        )
        if sim_type not in valid_types:
            msg = f"Unknown simulation type '{sim_type}'. Expected one of: {', '.join(valid_types)}."
            raise ValueError(msg)

        if "multiphase" in sim_type:
            sim_table.update(raw.get("multiphase", {}))

        bc_config = raw.get("boundary_conditions")
        if bc_config is not None:
            sim_table["bc_config"] = dict(bc_config)

        if "wetting" in raw:
            sim_table["wetting_config"] = dict(raw["wetting"])
        if "hysteresis" in raw:
            sim_table["hysteresis_config"] = dict(raw["hysteresis"])
        if "chemical_step" in raw:
            sim_table["chemical_step_config"] = dict(raw["chemical_step"])

        if "initialisation" in raw:
            sim_table["initialisation"] = dict(raw["initialisation"])

        ConfigAdapter._process_forces(raw, sim_table)
        ConfigAdapter._apply_output(sim_table, raw.get("output", {}))

        sim_table["sim_type"] = sim_type
        return sim_table

    def load(self, source: Any) -> SimulationConfig:  # noqa: ANN401
        """Parse *source* and return a validated :class:`SimulationConfig`."""
        flat = self.load_raw(source)
        flat.pop("simulation_type", None)

        if "grid_shape" in flat and not isinstance(flat["grid_shape"], tuple):
            flat["grid_shape"] = tuple(flat["grid_shape"])  # type: ignore[arg-type]

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

    @staticmethod
    def _serialize_safe(value: Any) -> Any:  # noqa: ANN401
        """Convert tuples to lists and recursively process nested structures."""
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, dict):
            return {k: ConfigAdapter._serialize_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [ConfigAdapter._serialize_safe(v) for v in value]
        return value

    @classmethod
    def build_sections(cls, config: SimulationConfig) -> dict[str, Any]:
        """Build a format-agnostic nested dict from *config*, routed by CONFIG_SECTION metadata."""
        sections = {
            f.name: f.metadata.get(CONFIG_SECTION, "simulation_type") for f in dataclasses.fields(SimulationConfig)
        }
        sim_type = config.sim_type
        skip = {"identity", "extra"}

        buckets: dict[str, dict[str, Any]] = defaultdict(dict)
        for key, value in dataclasses.asdict(config).items():
            section = sections.get(key, "simulation_type")
            if value is None or section in skip:
                continue
            if section == "multiphase" and "multiphase" not in sim_type:
                continue
            if isinstance(value, dict):
                buckets[section].update(cls._serialize_safe(value))
            else:
                buckets[section][key] = cls._serialize_safe(value)

        buckets["simulation_type"]["type"] = sim_type
        for ek, ev in (config.extra or {}).items():
            buckets["simulation_type"][ek] = cls._serialize_safe(ev)

        return {
            "simulation_type": buckets.pop("simulation_type", {}),
            **{k: buckets[k] for k in sorted(buckets) if buckets[k]},
        }


_ADAPTER_MAP: dict[str, str] = {
    ".toml": "tud_lbm.config.adapter_toml.TomlAdapter",
}


def get_adapter(path: str) -> ConfigAdapter:
    """Return the appropriate adapter for *path* based on file extension."""
    ext = Path(path).suffix.lower()
    fqn = _ADAPTER_MAP.get(ext)
    if not fqn:
        msg = f"Unsupported extension '{ext}'. Supported: {', '.join(sorted(_ADAPTER_MAP))}"
        raise ValueError(msg)
    module_path, class_name = fqn.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)()
