"""Quick-start guide for TUD-LBM public API.

This document explains how to use the TUD-LBM package for running lattice
Boltzmann simulations. All examples can run with sensible defaults.
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# QUICK START: 5-MINUTE TUTORIAL

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1. Basic simulation with 5 lines of code

from tud_lbm import SimulationConfig, build_setup, run
from tud_lbm.pipeline.runner import init_state

config = SimulationConfig(grid_shape=(64, 64), tau=0.8, nt=1000)
setup = build_setup(config)
state = init_state(setup)
final_state, trajectory = run(setup, state, nt=config.nt)

# That's it! You now have a complete D2Q9 simulation.

# Defaults provide:

# - Lattice: D2Q9 (2D, 9 velocities)

# - Collision: BGK

# - Boundaries: Periodic on all sides

# - Initialization: Rest equilibrium

# - Relaxation: tau=0.8 (nu = 0.1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# LOADING FROM CONFIG FILES

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 2a. From TOML file (reproducibility)

from tud_lbm.readers import toml

config = toml.load*simulation_config("examples/config_simple.toml")
setup = build_setup(config)
state = init_state(setup)
final_state, * = run(setup, state, nt=config.nt)

# 2b. From Python dict (Jupyter-friendly, parameter sweeps)

from tud_lbm.readers import dict as dict_reader

config = dict*reader.load_simulation_config({
"grid_shape": (128, 128),
"tau": 0.8,
"nt": 5000,
"gravity_force": {"force_g": 1e-6, "inclination_angle_deg": 90.0}
})
setup = build_setup(config)
state = init_state(setup)
final_state, * = run(setup, state, nt=config.nt)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# SAVING RESULTS

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 3. Export results to standard formats

from tud_lbm.io.output_data import write_vtk, write_numpy

final*state, * = run(setup, state, nt=config.nt)

# ParaView compatible

write_vtk(final_state, "output.vtk")

# NumPy compatible (for post-processing with matplotlib, etc.)

write_numpy(final_state, "output.npz")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CONFIGURATION OPTIONS (ALL HAVE DEFAULTS)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Complete configuration with explanations:

config = SimulationConfig( # Simulation type
sim_type="single_phase", # or "multiphase"
simulation_name="My Simulation", # optional

    # Grid and lattice
    grid_shape=(64, 64),               # spatial dimensions
    lattice_type="D2Q9",               # or "D3Q19" for 3D

    # Time stepping
    nt=1000,                           # number of timesteps
    tau=0.8,                           # relaxation time (must be > 0.5)

    # Collision and equilibrium
    collision_scheme="bgk",            # or "mrt"
    k_diag=None,                       # only needed for MRT

    # Boundary conditions
    bc_config={
        "left": "periodic",
        "right": "periodic",
        "top": "periodic",
        "bottom": "periodic",
    },

    # Optional physics: gravity
    gravity_force={
        "force_g": 1e-6,
        "inclination_angle_deg": 0.0,
    },

    # Optional physics: electric field
    electric_force=None,

    # Initialization
    init_type="standard",              # rest equilibrium
    initialisation={},                 # extra init params

    # Output
    results_dir="~/tud_lbm_results",
    save_interval=100,                 # save every 100 steps (auto-calculated if 0)
    save_fields=["rho", "u"],          # which fields to save
    plot_fields=None,
    output_format="numpy",             # numpy or vtk

)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# COMMON PATTERNS

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Parameter sweep (e.g., investigate effect of relaxation time)

for tau in [0.6, 0.8, 1.0, 1.2]:
config = SimulationConfig(grid*shape=(64, 64), tau=tau, nt=1000)
setup = build_setup(config)
state = init_state(setup)
final_state, * = run(setup, state, nt=config.nt) # Analyze or save results

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# API REFERENCE

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
Public API Functions:

1. SimulationConfig
   - What: Immutable configuration dataclass
   - How: config = SimulationConfig(grid_shape=(64, 64), tau=0.8)
   - Returns: Configuration object with full physics parameters

2. build_setup(config)
   - What: Build simulation setup from config
   - How: setup = build_setup(config)
   - Returns: SimulationSetup (immutable, JAX-compatible)

3. init_state(setup)
   - What: Initialize simulation state
   - How: state = init_state(setup)
   - Returns: State NamedTuple (f, rho, u, forces, etc.)

4. run(setup, state, nt)
   - What: Run simulation for nt timesteps
   - How: final_state, trajectory = run(setup, state, nt=1000)
   - Returns: (final_state, trajectory) or (final_state, None) if streaming I/O

Input Adapters (readers/):

- toml.load_simulation_config(path) → SimulationConfig
- dict_reader.load_simulation_config(data_dict) → SimulationConfig

Output Adapters (io/output_data/):

- write_vtk(state, path) → VTK file (ParaView)
- write_numpy(state, path) → NPZ file (NumPy)

Data Types:

- State: NamedTuple with fields f, rho, u, t, force, force_ext, ...
- Lattice: Velocity model (D2Q9, D3Q19)
- SimulationSetup: Container of operators and parameters
  """

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# WHERE TO FIND MORE

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
Documentation:

- docs/architecture.rst — Hexagonal architecture & design decisions
- docs/operators.rst — Operator categories and physics models
- docs/adapters.rst — Input/output adapter pattern
- docs/lattice.rst — Lattice models (D2Q9, D3Q19)

Examples:

- examples/minimal_example.py — Bare minimum (5 lines)
- examples/from_toml_example.py — Load from config file
- examples/dict_config_example.py — Jupyter-style parameter sweeps
- tests/integration/test_poiseuille.py — Validated physics test

GitHub Issues:

- See CONTRIBUTING.md for development guidelines
- Report bugs or request features via GitHub Issues
  """
