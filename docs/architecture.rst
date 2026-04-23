Architecture Overview
=====================

The tud_lbm package follows a **physics-first** folder structure combined with **hexagonal architecture** principles to create a maintainable, testable, and scalable lattice Boltzmann method library.

Core Design Principles
----------------------

1. **Physics First**: Folders are organized by lattice physics concepts such as "lattice", "collision operators", "streaming operators".
2. **Separation of Concerns**: Input adapters, simulation logic, and output handling are separated.
3. **Immutability**: State is immutable (JAX NamedTuples), enabling transparent JAX compilation
4. **Registry Pattern**: Operators self-register to avoid hardcoded factory dependencies
5. **Protocol Contracts**: Operators satisfy protocols, not inheritance hierarchies

Folder Structure Rationale
--------------------------

::

    tud_lbm/
    ├── lattice/           ← Velocity model (d, q, c, w)
    ├── operators/         ← Physics transformations (collision, streaming, etc)
    │   └── 9 subcategories
    ├── pipeline/          ← Execution harness (setup, runner, state)
    ├── config/            ← Pure immutable configuration
    ├── readers/           ← Input adapters (TOML, dict)
    ├── io/                ← Output adapters (plotting, saving)
    ├── cli/               ← Command-line interface
    └── registry.py        ← Global operator registry


Brief Intro to Hexagonal Architecture
----------------------

What is Hexagonal Architecture?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hexagonal Architecture (also called "Ports and Adapters") is a design pattern that isolates your core business logic from external dependencies like file formats, databases, user interfaces, and configuration sources. The core physics simulation sits at the center, surrounded by ports (abstract contracts) and adapters (concrete implementations). This separation makes the code highly flexible and extensible: you can add new input formats (JSON, YAML, HDF5), new output formats (HDF5, Parquet, NetCDF), or new execution contexts (CLI, Jupyter, API, batch processing) without modifying a single line of core simulation code. The hexagon represents your irreplaceable business logic, and the ports/adapters represent the interchangeable interfaces to the outside world.

Terminology: Core, Ports, and Adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. The Core (Center of the Hexagon)**

The core is your pure business logic — the irreplaceable physics simulation:

.. code-block:: python

    # Core: pure physics, no I/O
    def build_setup(config: SimulationConfig) -> SimulationSetup:
        """Build simulation from config - doesn't care where config came from."""
        operators = registry.lookup_operators(config)
        return SimulationSetup(operators=operators, lattice=lattice)
    
    def run(state: State, setup: SimulationSetup, n_steps: int) -> State:
        """Run simulation - doesn't know or care where output goes."""
        for _ in range(n_steps):
            state = apply_collision(state, setup)
            state = apply_streaming(state, setup)
            state = apply_boundary(state, setup)
        return state

Core characteristics:
- **Format-agnostic**: Doesn't know about TOML, JSON, YAML.
- **Pure**: No side effects, no I/O, no database calls
- **Immutable**: Works with frozen dataclasses and JAX arrays
- **Testable**: No mocking needed, just pass data in, check data out

**2. Ports (Abstract Interfaces)**

Ports are the contracts your core logic depends on. They define WHAT your core needs without defining HOW it gets it:

.. code-block:: python

    # Port 1: Configuration Input
    # "I need some kind of SimulationConfig"
    def build_setup(config: SimulationConfig) -> SimulationSetup:
        pass  # Don't care if config came from TOML, JSON, dict, API, etc.
    
    # Port 2: Operator Registry
    # "I need to lookup operators somehow"
    collision_fn = registry.get_operator("collision_models", "bgk")
    # Don't care if registry is in-memory, database, or REST API
    
    # Port 3: State Output
    # "I return computed state, someone else handles it"
    final_state = run(state, setup, n_steps=1000)
    # Don't care if it goes to file, console, database, or memory

Three Input Ports:

1. **Configuration Input Port** — Accepts ``SimulationConfig`` dataclass
2. **Operator Lookup Port** — Uses ``registry`` to find operator implementations
3. **Dependency Port** — Receives ``Lattice`` and other dependencies

Three Output Ports:

1. **State Output Port** — Returns final ``State`` and ``diagnostics``
2. **Side Effects Port** — Invokes optional callbacks during execution
3. **Extensibility Port** — Supports custom operator additions via registry

**3. Adapters (Concrete Implementations)**

Adapters are the concrete implementations that translate between external formats and the abstract port interfaces. They're the "drivers" that connect external tools to your core:

**Input Adapters** (in ``readers/`` directory):

Currently implemented adapters:

.. code-block:: python

    # TOML Adapter: Converts TOML file → SimulationConfig
    class TOMLAdapter:
        def load_simulation_config(self, path: str) -> SimulationConfig:
            data = toml.parse(path)  # External tool
            return SimulationConfig(**data)  # Internal format
    
    # Dictionary Adapter: Converts dict → SimulationConfig
    class DictAdapter:
        def load_simulation_config(self, data: dict) -> SimulationConfig:
            # Already compatible, just validate
            return SimulationConfig(**data)

All adapters implement the same interface, so core doesn't care which one is used:

.. code-block:: python

    from tud_lbm.readers import toml, dict as dict_reader
    
    # Load from TOML file
    config = toml.load_simulation_config("config.toml")
    
    # Or load from Python dict
    config = dict_reader.load_simulation_config({"lattice_type": "D2Q9", ...})
    
    # Both return SimulationConfig - core is identical
    setup = build_setup(config)  # ← No changes needed

**Output Adapters** (in ``io/`` directory):

Currently implemented adapters handle data export and visualization:

.. code-block:: python

    # VTK Adapter: Converts State → VTK file (for ParaView visualization)
    class VTKAdapter:
        def write(self, state: State, path: str) -> None:
            vtk_data = convert_state_to_vtk(state)  # Internal → External
            write_vtk_file(vtk_data, path)
    
    # NumPy Adapter: Exports State → NumPy arrays for post-processing
    class NumpyAdapter:
        def write(self, state: State, path: str) -> None:
            arrays = {
                'f': numpy.array(state.f),
                'u': numpy.array(state.u),
                'rho': numpy.array(state.rho)
            }
            numpy.savez_compressed(path, **arrays)

Future adapters could provide additional analysis and documentation capabilities:

.. code-block:: python

    # Future: Plotting Adapter (visualization in Jupyter/matplotlib)
    # generate_plots(state, config) → matplotlib figures
    
    # Future: Jupyter Notebook Generator (auto-generate reproducible notebooks)
    # generate_notebook(config, final_state) → .ipynb file with:
    #   - Simulation parameters
    #   - Plots of velocity/density/pressure fields
    #   - Physics analysis (kinetic energy, momentum conservation, etc.)
    #   - Executable code cells for reproducibility
    
    # Future: HDF5 Adapter (large-scale data storage with metadata)
    # write_hdf5(state_trajectory, "results.h5")

Core doesn't care which adapter is used. Here's how to save results in different formats:

.. code-block:: python

    final_state = run(state, setup, n_steps=1000)
    
    # Currently implemented output formats
    from tud_lbm.io.output_data import write_vtk, write_numpy
    write_vtk(final_state, "output.vtk")
    write_numpy(final_state, "output.npz")
    
    # Future: Visualization and reproducible notebooks
    # from tud_lbm.io import generate_plots, generate_notebook
    # generate_plots(final_state, config)
    # generate_notebook(config, final_state, "report.ipynb")

Design Illustration
~~~~~~~~~~~~~~~~~~~

The diagram shows how data flows through the architecture:

::

    ┌──────────────────────────────────────────────────────────┐
    │  EXTERNAL WORLD                                           │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
    │  │   TOML   │  │  Dict    │  │  Jupyter │  │   VTK    │  │
    │  │(currently)│  │(currently)│  │  input   │  │(currently)│ │
    │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
    │  │   JSON   │  │   YAML   │  │  NumPy   │  │ Plots &  │  │
    │  │ (future) │  │ (future) │  │(currently)│ │ Notebooks│  │
    │  │          │  │          │  │          │  │ (future) │  │
    │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
    └───────┼─────────────┼─────────────┼─────────────┼─────────┘
            │             │             │             │
    ┌───────▼─────────────▼─────────────▼─────────────▼─────────┐
    │  ADAPTER LAYER                                            │
    │  ┌────────────────┐  ┌────────────────┐  ┌─────────────┐  │
    │  │ Input Adapters │  │Output Adapters │  │ CLI/Jupyter │  │
    │  │ (readers/)     │  │ (io/)          │  │ (cli/)      │  │
    │  └────────────────┘  └────────────────┘  └─────────────┘  │
    └───────┬─────────────────────────────┬──────────────────────┘
            │                             │
            │  SimulationConfig           │  State
            │  (port interface)           │  (port interface)
            │                             │
    ┌───────▼─────────────────────────────▼──────────────────────┐
    │  CORE LOGIC (Pure Physics)                                │
    │                                                             │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │ build_setup(config) → SimulationSetup               │  │
    │  │ run(state, setup, n_steps) → final_state            │  │
    │  │ Operators: collision, streaming, boundary, etc.     │  │
    │  └──────────────────────────────────────────────────────┘  │
    │                                                             │
    │  Dependencies (Immutable):                                │
    │  • Lattice (D2Q9, D3Q19, ...)                             │
    │  • Configuration                                           │
    │  • Registry (operator lookup)                              │
    │  • State & WettingState (JAX pytrees)                      │
    └─────────────────────────────────────────────────────────────┘

Data Flow Through Hexagon
~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario: Run simulation from Jupyter, output to NumPy arrays**

::

    Step 1: Input Adapter (Dict → Config)
    ─────────────────────────────────
    config_dict = {
        "lattice_type": "D2Q9",
        "tau": 0.8,
        "grid_shape": [64, 64]
    }
    ↓
    [Dict Adapter converts to SimulationConfig]
    ↓
    config = SimulationConfig(
        lattice_type="D2Q9",
        tau=0.8,
        grid_shape=(64, 64)
    )
    
    
    Step 2: Core Logic (Setup)
    ─────────────────────────
    setup = build_setup(config)
    ↓
    [Core looks up operators from registry]
    [Core builds immutable SimulationSetup]
    [No knowledge of where config came from!]
    
    
    Step 3: Core Logic (Simulation)
    ───────────────────────────────
    state = init_state(config, setup)
    final_state = run(state, setup, n_steps=1000)
    ↓
    [Pure functional computation]
    [No I/O, no formatting, no file writing]
    [No knowledge of where output will go!]
    
    
    Step 4: Output Adapter (State → NumPy)
    ────────────────────────────────────
    [NumPy Adapter converts JAX arrays to NumPy]
    ↓
    write_numpy(final_state, "results.npz")
    ↓
    File written, simulation code unchanged

Why This Architecture is Powerful
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Flexibility Example 1: Change Input Format**

.. code-block:: python

    from tud_lbm.readers import toml, dict as dict_reader
    from tud_lbm.pipeline import build_setup
    
    # Load from TOML
    config = toml.load_simulation_config("config.toml")
    
    # Or load from dict (Jupyter notebook)
    config = dict_reader.load_simulation_config({
        "lattice_type": "D2Q9",
        "tau": 0.8
    })
    
    # Rest of code is identical!
    setup = build_setup(config)
    final_state = run(state, setup, n_steps=1000)

**Flexibility Example 2: Change Output Format**

.. code-block:: python

    final_state = run(state, setup, n_steps=1000)
    
    # Multiple outputs simultaneously, no core changes needed!
    from tud_lbm.io.output_data import write_vtk, write_numpy
    write_vtk(final_state, "results.vtk")
    write_numpy(final_state, "arrays.npz")
    
    # Future: Simulation data handling and visualization
    # from tud_lbm.io import generate_plots, generate_notebook
    # generate_plots(final_state, config, "plots/")
    # generate_notebook(config, final_state, "report.ipynb")
    # 
    # The generated notebook would automatically include:
    # - Simulation parameters and configuration
    # - Velocity/density/pressure field visualizations
    # - Physics diagnostics (conservation laws, stability checks)
    # - Executable code cells for reproducibility

**Flexibility Example 3: Add New Context**

.. code-block:: python

    # Core is reusable everywhere with same code:
    
    # 1. CLI (click framework)
    @click.command()
    def run_cli(config_path):
        config = toml_adapter.load_simulation_config(config_path)
        setup = build_setup(config)
        # ... same core code ...
    
    # 2. Jupyter notebook
    config = dict_adapter.load_simulation_config({...})
    setup = build_setup(config)
    # ... same core code ...
    
    # 3. Batch processing with parameter sweep
    for params in sweep_parameters():
        config = dict_adapter.load_simulation_config(params)
        setup = build_setup(config)
        # ... same core code ...

**Testability Benefit**:

.. code-block:: python

    def test_poiseuille_flow():
        """Test physics without any I/O"""
        # Create config in memory
        config = dict_adapter.load_simulation_config({
            "lattice_type": "D2Q9",
            "tau": 0.8,
            "grid_shape": [64, 64]
        })
        
        # Run core logic
        setup = build_setup(config)
        state = init_state(config, setup)
        final_state = run(state, setup, n_steps=1000)
        
        # Assert physics properties
        assert velocity_profile_is_parabolic(final_state.u)
        
        # No file system, no mocking, no adapter complexity!

Summary Table
~~~~~~~~~~~~~

+---------------------+------------------------------------------+-----------------------------------------+
| Core (Hexagon)      | ``pipeline/``, ``operators/``            | Pure physics simulation                  |
+=====================+==========================================+=========================================+
| Configuration Port  | Accepts ``SimulationConfig`` type        | Abstract input interface                |
+---------------------+------------------------------------------+-----------------------------------------+
| Input Adapters      | ``readers/toml.py``, ``readers/dict.py`` | Convert external formats to Config      |
| (currently)         |                                          |                                         |
+---------------------+------------------------------------------+-----------------------------------------+
| Input Adapters      | (Future) JSON, YAML, HDF5                | Extensible for new formats              |
| (planned)           |                                          |                                         |
+---------------------+------------------------------------------+-----------------------------------------+
| Operator Port       | Uses ``registry`` lookup                 | Abstract operator interface             |
+---------------------+------------------------------------------+-----------------------------------------+
| State Port          | Returns ``State`` from ``run()``         | Abstract output interface               |
+---------------------+------------------------------------------+-----------------------------------------+
| Output Adapters     | ``io/output_data/write_vtk.py``          | Visualization for ParaView               |
| (currently)         | ``io/output_data/write_numpy.py``        | Data export for post-processing          |
+---------------------+------------------------------------------+-----------------------------------------+
| Output Adapters     | (Future) Plotting, Jupyter notebook      | Simulation data handling and             |
| (planned)           | generation, HDF5, Parquet, NetCDF        | reproducible science notebooks           |
+---------------------+------------------------------------------+-----------------------------------------+
| Extensibility       | ``readers/`` and ``io/`` are open-ended | Easy to add new adapters without        |
|                     |                                          | modifying core code                     |
+---------------------+------------------------------------------+-----------------------------------------+

Operator Registration System
-----------------------------

Operators self-register via decorators:

.. code-block:: python

    from tud_lbm import registry

    @registry.collision_operator(name="bgk")
    def bgk_collision(f, feq, tau):
        return f - (f - feq) / tau

This avoids the **factory anti-pattern** where a single factory file must import everything. Instead:

1. Operators register themselves on import
2. Registry is a lookup table (key → implementation)
3. Factory simply does: ``registry.get_operators(kind)[name]``

Benefits:
- New operators added without modifying factory
- Circular imports prevented (registry is independent)
- Operator location reflects physics category

Configuration as Pure Data
--------------------------

Configuration uses immutable dataclasses:

.. code-block:: python

    @dataclass(frozen=True)
    class SimulationConfig:
        grid_shape: tuple[int, ...]
        lattice_type: str
        tau: float
        run_type: str
        # ... other physics parameters

**NOT** (❌):

.. code-block:: python

    class Simulation:
        def __init__(self, ...):
            self.collision_fn = load_collision_operator(...)
            self.streaming_fn = load_streaming_operator(...)
            # Live objects embedded in config

**Why immutable**:
- JAX pytree-compatible
- No hidden state mutations
- Thread-safe
- Testable: same config → reproducible results
- Config can be serialized/deserialized easily

State Management
----------------

Simulation state flows through JAX's functional paradigm:

.. code-block:: python

    class State(NamedTuple):
        f: jax.Array           # Populations (d×q array)
        u: jax.Array           # Velocity field
        rho: jax.Array         # Density field

    # Pure function: same state → same output
    def step(state: State, setup: SimulationSetup) -> State:
        return apply_operators(state, setup.operators)

    # Jitted timestep loop
    final_state, diagnostics = jax.lax.scan(step, initial_state, n_steps)

Benefits:
- Transparent JAX compilation (no side effects)
- Reproducible results
- Easy checkpointing/restart
- Functional composition

Pipeline Execution
------------------

Three layers:

1. **build_setup()** (composition root)
   - Loads config
   - Instantiates operators from registry
   - Creates SimulationSetup (immutable, jittable)

2. **init_state()** (initialization)
   - Uses selected initializer operator
   - Returns initial State (NamedTuple)

3. **run()** (jitted timestep loop)
   - Uses JAX scan for efficient compilation
   - Applies collision, streaming, boundary operators
   - Returns final state + diagnostics

Input/Output Adapters
---------------------

**Readers** (input adapters in ``readers/``)

Convert external formats to SimulationConfig:

::

    TOML file
      ↓
    toml.py (parser)
      ↓
    SimulationConfig (pure dataclass)

Benefits:
- Decouples file format from core logic
- Multiple input formats supported
- Easy to add CSV, HDF5, JSON, etc.

**IO Handlers** (output adapters in ``io/``)

Convert State to external formats:

::

    State (JAX arrays)
      ↓
    output_data.py (handler)
      ↓
    VTK, NumPy, matplotlib plots

Benefits:
- Core logic doesn't know about file formats
- Post-processing loosely coupled
- Multiple simultaneous outputs
- Testable independently

Why Not Everything in Operators?
--------------------------------

Some libraries put everything (collision, streaming, boundaries, output) in a monolithic "operators" directory.

Our approach separates:

- ``operators/`` — Physics transformations (pure functions)
- ``pipeline/`` — Execution (composition, timing, callbacks)
- ``io/`` — Output formatting (presentation layer)
- ``readers/`` — Input parsing (presentation layer)

**Rationale**:

- **Operators** are stable (physics doesn't change)
- **Pipeline** evolves (new execution strategies)
- **I/O** is volatile (new format requests)
- **Readers** are configurable (many input sources)

Separation enables:

- Changing output format without touching collision code
- Adding new input format without recompiling operators
- Testing operators independently of file I/O
- Reusing operators in different execution contexts (JAX, NumPy, GPU)

Test Organization
-----------------

Tests mirror module structure:

::

    tests/
    ├── test_tud_lbm_imports.py        ← Package structure
    ├── integration/
    │   └── test_poiseuille.py         ← Physics validation
    ├── operators/
    │   ├── test_collision.py          ← Unit tests
    │   ├── test_streaming.py
    │   └── ...
    ├── test_registry.py               ← Registry behavior
    └── test_config.py                 ← Configuration

This structure:
- Easy to find relevant tests
- Enables parallel test execution
- Clear separation of unit vs integration

Summary
-------

The tud_lbm architecture achieves:

✓ **Clarity** — Physics-first naming, clear responsibilities
✓ **Testability** — Operators decoupled from I/O and config
✓ **Extensibility** — New operators/readers/writers without modifying core
✓ **Performance** — JAX jit-compatible immutable state
✓ **Maintainability** — Separation of concerns, minimal duplication
