Architecture Overview
=====================

TUD-LBM combines a **physics-first** folder structure with **ports and
adapters** boundaries.  Directories are named after lattice-Boltzmann
concepts, and everything that touches the outside world — file formats,
plotting, the CLI — is kept at the edge, so the simulation core stays a set
of pure functions over JAX arrays.

Core design principles
----------------------

1. **Physics first.**  Folders are named after concepts (``lattice``,
   ``collision``, ``streaming``), not layers.
2. **Separation of concerns.**  Input adapters, simulation logic, and output
   handling never import each other.
3. **Immutability.**  Configuration is a frozen dataclass; state is a
   ``NamedTuple``.  Nothing mutates in place.
4. **Registry over factories.**  Operators self-register at import time, so
   no module has to import every implementation.
5. **Protocols, not inheritance.**  Operators satisfy structural contracts
   from :mod:`src.operators.protocols`.
6. **scipy/numpy at setup time, JAX inside the loop.**  One-time computation
   happens before the JIT boundary; everything inside ``lax.scan`` is
   traceable.

Repository layout
-----------------

The import root is ``src``.  The distribution is still named ``tud_lbm``;
only the import package was renamed.

.. note::

   The I/O subpackage is ``src.simulation_io``, **not** ``src.io`` — the
   latter shadowed the stdlib ``io`` module once ``src`` became the import
   root.

::

    src/
    ├── registry.py          ← central operator registry
    ├── config/              ← frozen SimulationConfig + input adapters
    ├── lattice/             ← velocity models (D2Q9, D3Q19)
    ├── operators/           ← physics transformations
    │   ├── collision/  equilibrium/  macroscopic/  eos/
    │   ├── boundary/   streaming/    differential/
    │   ├── force/      initialise/   obstacle/
    │   ├── wetting/    step/
    │   ├── protocols.py     ← structural contracts
    │   ├── factory.py       ← generic build_operator()
    │   └── _loader.py       ← auto_load_operators()
    ├── pipeline/            ← setup, runner, state
    ├── simulation_io/       ← writers, plotting, analysis, callbacks
    └── cli/                 ← click command-line interface

Where the boundaries fall
-------------------------

::

    ┌──────────────────────────────────────────────────────────────┐
    │  EXTERNAL WORLD                                              │
    │   TOML file    Python dict    .npz / VTK    PNG / MP4        │
    └──────┬─────────────┬───────────────▲────────────▲────────────┘
           │             │               │            │
    ┌──────▼─────────────▼───────────────┴────────────┴────────────┐
    │  ADAPTER LAYER                                               │
    │   config/adapter_toml.py     simulation_io/output_data/      │
    │   config/adapter_dict.py     simulation_io/plotting/         │
    │   cli/                       simulation_io/callbacks.py      │
    └──────┬───────────────────────────────────▲──────────────────-┘
           │  SimulationConfig                 │  State
    ┌──────▼───────────────────────────────────┴──────────────────-┐
    │  CORE (pure physics, JIT-compatible)                         │
    │                                                              │
    │   build_setup(config)      → SimulationSetup                 │
    │   init_state(setup)        → State                           │
    │   run(setup, state, nt)    → (State, State | None)           │
    │                                                              │
    │   Dependencies: Lattice · registry · operators               │
    └──────────────────────────────────────────────────────────────┘

The core never learns where its configuration came from or where its results
go.  That is what lets the same three calls back the CLI, a Jupyter session,
and a parallel sweep worker without changing.

The execution pipeline
----------------------

Three stages, each with a single entry point.

1. ``build_setup(config)`` — composition root
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~src.pipeline.setup.build_setup` resolves every operator the run will
need and returns an immutable
:class:`~src.pipeline.setup.SimulationSetup`.  All registry lookups, mask
construction, and closure building happen here — once, outside the JIT
boundary.

``SimulationSetup`` holds *built callables*, not raw configuration:
``step_fn``, ``collision_fn``, ``equilibrium_fn``, ``macroscopic_fn``,
``streaming_fn``, ``bc_fn``, ``initial_f_fn``, the differential-operator
closures, ``bc_masks``, ``forces``, ``multiphase_params``,
``extra_state_plugins``, and a reference back to ``config`` for anything that
must not enter the trace.

Many fields are typed ``| None`` because they do not apply to every
``sim_type``.  For a given simulation type the relevant ones are always
populated, and step code narrows them with ``assert x is not None``.

2. ``init_state(setup)`` — initialisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~src.pipeline.runner.init_state` builds the starting
:class:`~src.pipeline.state.state.State`.  Populations come from
``setup.initial_f_fn`` (or a caller-supplied ``f``); density follows from the
moments; velocity starts at zero.

.. important::

   Every optional field that will *ever* be written must be initialised to
   zeros rather than left as ``None``.  ``lax.scan`` requires the carry
   pytree to have identical structure on every iteration, so a field that
   appears mid-run would change the structure and fail the trace.

3. ``run(setup, state, nt)`` — the scanned loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~src.pipeline.runner.run` drives ``jax.lax.scan`` over a JIT-compiled
step body, in one of two modes:

**In-memory** — the full trajectory is returned as a stacked ``State``
pytree:

.. code-block:: python

    final_state, trajectory = run(setup, state, nt=1000)

**Streaming I/O** — snapshots are written from inside the loop through
``jax.debug.callback`` and ``trajectory`` is ``None``:

.. code-block:: python

    final_state, _ = run(setup, state, nt=1000, io_handler=io, save_interval=100)

Configuration as pure data
--------------------------

:class:`~src.config.simulation_config.SimulationConfig` is a frozen
dataclass and the sole configuration container.  It never enters a JIT
boundary.  Validation runs in ``__post_init__``, so an invalid configuration
cannot exist as an object.

.. code-block:: python

    @dataclass(frozen=True)
    class SimulationConfig:
        sim_type: Literal["single_phase", "multiphase", ...] = "single_phase"
        grid_shape: tuple[int, ...] = (64, 64)
        tau: float = 1.0
        ...

What it deliberately is **not** is a container of live objects:

.. code-block:: python

    # ✗ not this
    class Simulation:
        def __init__(self, ...):
            self.collision_fn = load_collision_operator(...)

Keeping resolved callables out of the config is what makes it serialisable,
hashable across processes for sweeps, and safe to write back out as the
``config.toml`` in a run directory.

See :doc:`adapters` for the section mapping, the defaults applied on
construction, and parameter-sweep expansion.

State management
----------------

:class:`~src.pipeline.state.state.State` is the ``lax.scan`` carry:

.. code-block:: python

    class State(NamedTuple):
        f: jnp.ndarray                 # (nx, ny, nz, q, 1)
        rho: jnp.ndarray               # (nx, ny, nz, 1, 1)
        u: jnp.ndarray                 # (nx, ny, nz, 1, d)
        t: jnp.ndarray                 # scalar — current timestep
        force: jnp.ndarray | None      # multiphase interaction force
        force_ext: jnp.ndarray | None  # external force
        h: jnp.ndarray | None          # electric potential distributions
        wetting: WettingState | None

:class:`~src.pipeline.state.state.WettingState` is a nested ``NamedTuple``
carrying the hysteresis parameters — ``phi_left``/``phi_right``,
``d_rho_left``/``d_rho_right``, the measured angles ``ca_left``/``ca_right``,
and the contact-line locations ``cll_left``/``cll_right``.  Being part of the
carry is what lets the optimiser's result persist from step to step.

Because ``NamedTuple`` is a pytree, the whole state maps cleanly through
``jit``, ``scan``, and ``vmap`` with no registration boilerplate.

Multiphase physics
------------------

The multiphase step (``src/operators/step/_multiphase.py``) runs:

1. ``ρ`` and ``u`` from the moments of ``f``
2. bulk chemical potential ``μ₀(ρ)`` from the configured EOS
3. total chemical potential ``μ = μ₀ − κ∇²ρ``
4. interparticle force ``F_int = −ρ∇μ``
5. force-corrected velocity ``u_eq = u + F_total/(2ρ)``

then equilibrium → collision → streaming → boundary conditions through the
shared ``_multiphase_pipeline`` helper in ``_common.py``.

.. warning::

   Lattice symmetry is load-bearing.  In a periodic domain there is no
   viscous sink for a systematic net force, so it accumulates.  Droplet
   centres must be integer-aligned to preserve the D4 discrete symmetry.

The wetting and hysteresis layers build on top: fixed wetting applies a
contact-angle condition at the wall, while the hysteresis operators solve for
the wetting parameters each step with a ``lax.while_loop`` around Adam,
targeting an advancing/receding angle window.  See :doc:`operators` for the
per-operator detail.

Why the split
-------------

``operators/`` is stable — the physics does not change.  ``pipeline/``
evolves as execution strategies change.  ``simulation_io/`` is volatile;
every new output request lands there.  ``config/`` absorbs new input sources.
Separating them means a new output format cannot break a collision operator,
and operators can be tested with no file system at all:

.. code-block:: python

    config = DictAdapter().load({"grid_shape": (32, 32), "tau": 0.8, "nt": 10})
    setup = build_setup(config)
    state = init_state(setup)
    final_state, _ = run(setup, state, nt=10)

    assert jnp.isfinite(final_state.rho).all()

Extension points
----------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - To add …
     - Do this
   * - a physics operator
     - drop a ``_*.py`` file in the right ``operators/`` subpackage,
       decorated with that kind's decorator
   * - a velocity model
     - register a ``Lattice`` builder with ``@lattice_operator``
   * - an input format
     - subclass ``ConfigAdapter``, add the extension to ``_ADAPTER_MAP``
   * - an output format
     - subclass ``OutputWriter`` in ``simulation_io/output_data/``
   * - a per-timestep figure
     - subclass ``PlotOperator``, register with ``@plotting_operator``
   * - a run-history figure
     - subclass ``AnalysisPlot``, register with ``@analysis_operator``
   * - a CLI command
     - add a module under ``cli/commands/``, decorated with ``cli_command``

None of these require editing a dispatch table or an import list.

Command-line interface
----------------------

The entry point is ``tud-lbm = "src.cli.commands:cli"``.  Importing
``src.cli.commands`` is what registers the commands onto the group defined in
``cli/app.py``; importing ``cli.app`` alone yields an empty group.

Every command carries the ``cli_command`` decorator from ``cli/_console.py``,
which fixes the error contract: ``KeyboardInterrupt`` exits 130,
``click.UsageError`` uses click's own 2, ``SystemExit`` passes through, and
anything else prints a red ``Error:`` line and exits 1 — or re-raises when
``TUD_LBM_DEBUG`` is set.

``visualise`` is a click *group* with ``invoke_without_command=True`` and a
required ``RUN_DIR`` on the group itself.  Two details there are load-bearing:
the group sets ``allow_interspersed_args`` (groups disable it by default,
which would reject ``visualise DIR --no-prompt``), and ``cli_command`` must
**not** decorate the group callback, since click invokes subcommands only
after that callback returns — an exception raised in a subcommand would
escape uncaught.

Testing
-------

Tests mirror the module structure under ``tests/`` and validate public
behaviour and invariants of the final design, not internal structure.
Markers declared in ``pyproject.toml``:

=================  =========================================================
Marker             Scope
=================  =========================================================
``unit``           individual operators
``integration``    end-to-end physics pipelines
``conformance``    operators satisfy their protocol contracts
``slow``           full-pipeline smoke tests, excluded from fast CI
=================  =========================================================

.. code-block:: console

    pytest                          # fast unit tests
    pytest -m slow                  # includes end-to-end example runs
    uv run pytest --cov --cov-report xml   # refresh coverage.xml for SonarCloud
