"""Configuration file readers.

Internal module for loading simulation configurations.
Use tud_lbm.readers for the public API.

Modules:
    - DictAdapter       : Load from Python dictionaries
    - TomlAdapter       : Load from TOML files
"""

from tud_lbm.config.adapter_dict import DictAdapter
from tud_lbm.config.adapter_toml import TomlAdapter

__all__ = ["DictAdapter", "TomlAdapter"]

