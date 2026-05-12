"""Output data writers for simulation results."""

import importlib
import pathlib
import pkgutil
from .base import OutputWriter

# --- Automatic module discovery ---
_package_dir = pathlib.Path(__file__).parent

for module_info in pkgutil.iter_modules([str(_package_dir)]):
    if module_info.name != "base":
        importlib.import_module(f"{__name__}.{module_info.name}")


class _OutputWriterRegistry:
    """Registry for available output writer implementations."""

    def __getitem__(self, name: str) -> type[OutputWriter]:
        """Get an output writer class by name.

        Args:
            name: The name of the output writer.

        Returns:
            The OutputWriter subclass.

        Raises:
            KeyError: If the output writer is not found.
        """
        if name not in OutputWriter.registry:
            msg = f"Unknown output writer '{name}'. Available: {list(OutputWriter.registry.keys())}"
            raise KeyError(
                msg,
            )

        return OutputWriter.registry[name]

    def available(self) -> list[str]:
        """Get list of available output writers.

        Returns:
            List of available output writer names.
        """
        return list(OutputWriter.registry.keys())

    def __repr__(self):
        return f"<Output writers: {self.available()}>"


output_writers = _OutputWriterRegistry()


__all__ = ["output_writers"]
