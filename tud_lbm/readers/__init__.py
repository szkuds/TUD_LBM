"""Configuration file readers (public API).

Provides adapter classes for loading simulation parameters from various formats.
All adapters implement the ConfigAdapter interface and return SimulationConfig objects.

Classes
-------

DictAdapter
    Load configuration from Python dictionaries.

    Example::

        from tud_lbm.readers import DictAdapter
        config = DictAdapter().load({
            'grid_shape': (64, 64),
            'tau': 0.8,
            'nt': 1000,
        })

TomlAdapter
    Load configuration from TOML files.

    Example::

        from tud_lbm.readers import TomlAdapter
        config = TomlAdapter().load("config.toml")

Configuration Format
--------------------

Both adapters accept parameters matching SimulationConfig fields:
- grid_shape (tuple[int, ...]): Grid dimensions
- tau (float): Relaxation time
- nt (int): Number of timesteps
- sim_type (str): 'single_phase' or 'multiphase'
- (and other optional parameters)

See SimulationConfig for complete parameter reference.
"""

from tud_lbm.io.readers import DictAdapter
from tud_lbm.io.readers import TomlAdapter

__all__ = ["DictAdapter", "TomlAdapter"]
