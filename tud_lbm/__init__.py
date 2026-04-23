"""TUD LBM — Physics-first Lattice Boltzmann Method simulation framework.

A JAX-accelerated LBM package for PhD students and researchers, emphasizing
readability and extensibility. Structured as:

- lattice/      : Velocity models (D2Q9, D3Q19, etc.)
- operators/    : Physics operators (collision, streaming, etc.)
- pipeline/     : Composition and execution
- config/       : Configuration (pure data)
- io/           : Output adapters (plotting, saving)
- cli/          : Command-line interface
- readers/      : Input adapters (TOML, YAML, etc.)

Example usage::

    from tud_lbm import SimulationConfig, build_setup, run, State
    from tud_lbm.pipeline.runner import init_state

    # Create configuration with sensible defaults
    config = SimulationConfig(grid_shape=(64, 64), tau=0.8, nt=5000)

    # Build simulation setup from config
    setup = build_setup(config)

    # Initialize state
    state = init_state(setup)

    # Run simulation
    final_state, trajectory = run(setup, state, nt=config.nt)
"""

__version__ = "0.2.0"


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy load main API to avoid circular imports."""
    if name == "SimulationConfig":
        from tud_lbm.config.simulation_config import SimulationConfig

        return SimulationConfig
    elif name == "Lattice":
        from tud_lbm.lattice.lattice import Lattice

        return Lattice
    elif name == "build_lattice":
        from tud_lbm.lattice.lattice import build_lattice

        return build_lattice
    elif name == "build_setup":
        from tud_lbm.pipeline.setup import build_setup

        return build_setup
    elif name == "run":
        from tud_lbm.pipeline.runner import run

        return run
    elif name == "State":
        from tud_lbm.pipeline.state.state import State

        return State
    elif name == "WettingState":
        from tud_lbm.pipeline.state.state import WettingState

        return WettingState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Expose public API."""
    return [
        "SimulationConfig",
        "Lattice",
        "build_lattice",
        "build_setup",
        "run",
        "State",
        "WettingState",
    ]
