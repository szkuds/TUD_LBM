"""TOML configuration file adapter.

Reads and writes ``.toml`` config files.

Requires Python ≥ 3.11 (``tomllib`` in stdlib) **or** the ``tomli``
back-port on Python 3.10::

    pip install tomli

Example usage::

    from config.adapter_toml import TomlAdapter

    adapter = TomlAdapter()
    config = adapter.load("example_for_test/config_simple.toml")
    adapter.save(config, "output/config.toml")
"""

from __future__ import annotations
import contextlib
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Any as _Any
from tud_lbm.config.adapter_base import ConfigAdapter

if TYPE_CHECKING:
    from tud_lbm.config.simulation_config import SimulationConfig

tomli_w: _Any = None
with contextlib.suppress(ImportError):
    import tomli_w


class TomlAdapter(ConfigAdapter):
    """Adapter that reads and writes TOML configuration files."""

    def load_raw(self, source: str) -> dict[str, Any]:
        """Parse the TOML file at *source* and return a flat config dict."""
        path_obj: Path = Path(source).expanduser()
        if not path_obj.is_file():
            msg = f"Config file not found: {path_obj}"
            raise FileNotFoundError(msg)
        with path_obj.open("rb") as fh:
            raw = tomllib.load(fh)
        return self._merge_sections(raw)

    def save(self, config: SimulationConfig, path: str) -> None:
        """Serialise *config* to a ``.toml`` file at *path*.

        Delegates the field → section bucketing to
        :meth:`~ConfigAdapter.build_sections` (shared by all adapters)
        and writes the result with ``tomli_w``.

        Args:
            config: A validated :class:`SimulationConfig`.
            path: Destination file path.

        Raises:
            OSError: If the file cannot be written.
            ImportError: If tomli_w is not installed.
        """
        if tomli_w is None:
            msg = "tomli_w is required for saving TOML files. Install with: pip install tomli-w"
            raise ImportError(msg)

        path_obj: Path = Path(path).expanduser()
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        doc = self.build_sections(config)

        with path_obj.open("wb") as fh:
            tomli_w.dump(doc, fh)
