import importlib
import pkgutil
import pathlib

from .base import OutputWriter

# --- Automatic module discovery ---
_package_dir = pathlib.Path(__file__).parent

for module_info in pkgutil.iter_modules([str(_package_dir)]):
    if module_info.name != "base":
        importlib.import_module(f"{__name__}.{module_info.name}")


class _OutputWriterRegistry:

    def __getitem__(self, name):
        if name not in OutputWriter.registry:
            raise KeyError(
                f"Unknown output writer '{name}'. "
                f"Available: {list(OutputWriter.registry.keys())}"
            )

        return OutputWriter.registry[name]

    def available(self):
        return list(OutputWriter.registry.keys())

    def __repr__(self):
        return f"<Output writers: {self.available()}>"


output_writers = _OutputWriterRegistry()


__all__ = ["output_writers"]
