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

from tud_lbm.io.readers import DictAdapter, TomlAdapter

__all__ = ["DictAdapter", "TomlAdapter"]
"""Configuration file readers (public API).

Load simulation configurations from various file formats.

Modules:
    - DictAdapter       : Load from Python dictionaries
    - TomlAdapter       : Load from TOML files

Example usage::

    from tud_lbm.readers import DictAdapter, TomlAdapter
    from tud_lbm import build_setup, run
    from tud_lbm.pipeline.runner import init_state

    # Load from dict
    dict_config = DictAdapter().load({"grid_shape": (64, 64), "tau": 0.8})

    # Load from TOML
    toml_config = TomlAdapter().load("config.toml")

    setup = build_setup(dict_config)
    state = init_state(setup)
    final_state, _ = run(setup, state, nt=100)
"""

from tud_lbm.config.adapter_dict import DictAdapter
from tud_lbm.config.adapter_toml import TomlAdapter

__all__ = ["DictAdapter", "TomlAdapter"]
