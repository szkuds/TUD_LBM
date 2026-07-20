"""TUD LBM — Physics-first Lattice Boltzmann Method simulation framework.

A JAX-accelerated LBM package for PhD students and researchers, emphasizing
readability and extensibility.

Quick Start
-----------

The main workflow: Config → Setup → State → Run::

    from tud_lbm import SimulationConfig, build_setup, run, init_state

    # 1. Create configuration with sensible defaults
    config = SimulationConfig(grid_shape=(64, 64), tau=0.8, nt=5000)

    # 2. Build simulation setup from config
    setup = build_setup(config)

    # 3. Initialize state (single-phase or multiphase)
    state = init_state(setup)

    # 4. Run simulation
    final_state, trajectory = run(setup, state, nt=config.nt)

Key Classes & Functions
-----------------------

**Configuration & Setup:**
- SimulationConfig     : Immutable configuration container (sensible defaults)
- build_lattice()      : Create velocity model (D2Q9, D3Q19, etc.)
- build_setup()        : Build complete simulation setup
- State                : Single-phase simulation state (rho, u, f)
- WettingState         : Multiphase state with contact angle and interface tracking

**Execution:**
- init_state()         : Initialize state from setup
- run()                : Execute simulation, returns (final_state, trajectory)

**Input/Output:**
- readers              : Config loaders (DictAdapter, TomlAdapter)
- plotting             : Visualization (FigureBuilder, PlotOperator, visualise)
- io                   : Output writers (NumPy, VTK) and managers (SimulationIO)

Full API Reference
-------------------

See https://github.com/tudelft-ceg/tud-lbm for documentation.
"""

__version__ = "0.3.0"

# Apply JAX settings (x64 precision) before any submodule creates arrays;
# module-level constants bake in the dtype active at import time.
from tud_lbm.config.jax_config import configure_jax as _configure_jax

_configure_jax()


# Lazy imports to avoid circular dependencies
def __getattr__(name):  # noqa: PLR0911, ANN001, ANN202
    """Lazy load main API to avoid circular imports.

    Exports:
    --------
    SimulationConfig : Immutable frozen dataclass with physics & grid parameters
    Lattice          : Velocity model (D2Q9, D3Q19, etc.)
    build_lattice()  : Factory to create lattice from config
    build_setup()    : Factory to create complete Setup from config
    init_state()     : Initialize State or WettingState from setup
    run()            : Execute simulation for N timesteps
    State            : Single-phase simulation state (rho, u, f, t)
    WettingState     : Multiphase state for contact angle tracking
    """
    if name == "SimulationConfig":
        from tud_lbm.config.simulation_config import SimulationConfig

        return SimulationConfig
    if name == "Lattice":
        from tud_lbm.lattice.lattice import Lattice

        return Lattice
    if name == "build_lattice":
        from tud_lbm.lattice.lattice import build_lattice

        return build_lattice
    if name == "build_setup":
        from tud_lbm.pipeline.setup import build_setup

        return build_setup
    if name == "init_state":
        from tud_lbm.pipeline.runner import init_state

        return init_state
    if name == "run":
        from tud_lbm.pipeline.runner import run

        return run
    if name == "State":
        from tud_lbm.pipeline.state.state import State

        return State
    if name == "WettingState":
        from tud_lbm.pipeline.state.state import WettingState

        return WettingState
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__():  # noqa: ANN202
    """Expose public API for IDE autocompletion and help()."""
    return [
        "SimulationConfig",
        "Lattice",
        "build_lattice",
        "build_setup",
        "init_state",
        "run",
        "State",
        "WettingState",
    ]
