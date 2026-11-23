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
import logging
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Any as _Any
from src.config.run_config import CONFIG_FILENAME
from src.config.sections import build_sections
from src.config.sections import config_from_flat
from src.config.sections import load_sections
from src.protocols import ConfigAdapter

if TYPE_CHECKING:
    from src.config.simulation_config import SimulationConfig

logger = logging.getLogger(__name__)

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
        saved = path_obj.name == CONFIG_FILENAME
        flat, upgraded = load_sections(raw, saved=saved)
        if upgraded and saved:
            self._write_back(dict(flat), path_obj)
        return flat

    def _write_back(self, flat: dict[str, Any], path: Path) -> None:
        """Convert an old-layout run directory's ``config.toml`` in place, once.

        Only a run directory's file is rewritten, so an input TOML never changes
        behind the user's back. The file is replaced whole, because concurrent
        analyses may be reading it. Failing to write only leaves the conversion
        to the next load: an archived run on a read-only disk must still load.
        A file that is no single valid run is left for the caller to report.
        """
        try:
            config = config_from_flat(flat)
        except (TypeError, ValueError):
            return
        converted = path.with_name(f"{path.name}.upgraded")
        try:
            self.save(config, str(converted))
            converted.replace(path)
        except (ImportError, OSError) as exc:
            logger.warning("Could not write the upgraded config back to %s: %s", path, exc)

    def save(self, config: SimulationConfig, path: str) -> None:
        """Serialise *config* to a ``.toml`` file at *path*.

        Delegates the field → section bucketing to
        :func:`~src.config.sections.build_sections` (shared by all adapters)
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

        doc = build_sections(config)

        with path_obj.open("wb") as fh:
            tomli_w.dump(doc, fh)
