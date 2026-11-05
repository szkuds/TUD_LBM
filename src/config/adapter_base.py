"""Adapter dispatch by file extension."""

from __future__ import annotations
import importlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.protocols import ConfigAdapter

_ADAPTER_MAP: dict[str, str] = {
    ".toml": "src.config.adapter_toml.TomlAdapter",
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
