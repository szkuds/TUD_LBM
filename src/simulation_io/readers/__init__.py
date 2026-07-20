"""Configuration file readers.

Internal module for loading simulation configurations.
Use src.simulation_io.readers for the public API.

Modules:
    - DictAdapter       : Load from Python dictionaries
    - TomlAdapter       : Load from TOML files
"""

from src.config.adapter_dict import DictAdapter
from src.config.adapter_toml import TomlAdapter

__all__ = ["DictAdapter", "TomlAdapter"]
