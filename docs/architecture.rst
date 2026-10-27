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
``collision_fn``, ``equilibrium_fn``, ``source_fn``, ``macroscopic_fn``,
``streaming_fn``, ``bc_fn``, ``initial_f_fn``, the three differential
operators, ``forces``, ``addons``, ``multiphase_params``, and a reference
back to ``config`` for anything that must not enter the trace.  The operators
are always built, so they are not optional; only ``multiphase_params``, the
obstacle and ``k_diag`` are ``None`` when the run has none.

``build_setup`` is a fixed recipe — lattice, addon merge, differential
operators, forces, plain operators — and distinguishes two kinds of
component:

* **Plain operators** (collision, equilibrium, source term, streaming, BC,
  obstacle, initialiser, force, macroscopic) are selectable and homogeneous:
  each is a registry lookup of its configured name.
* **Addons** (registry kind ``addon``: ``wetting``, ``electric``) touch
  several parts of a step.  Each returns a
  :class:`~src.addons.hooks.Hooks` — or ``None`` when its
  config section is absent — that fills up to three fixed slots:

  ============== ================================ ==========================
  Slot           Shape                            Merge rule
  ============== ================================ ==========================
  ``ghost_fill`` ``(padded, substate) -> padded`` composed into the density
                                                  gradient and Laplacian
  ``post_step``  ``(setup, prev, new) -> new``    run after the kernel
  ``init_state`` ``(setup) -> substate``          ``State.ext[name]``
  ============== ================================ ==========================

  Addons merge in name order and must be independent: each reads and writes
  only its own ``State.ext`` entry.  Adding one is adding a
  ``src/addons/_*.py`` file; ``build_setup`` changes only for a new kind of
  slot.  ``sim_type`` selects the phase (macroscopic operator, equilibrium /
  source pair), never an addon: the ``[wetting]``, ``[hysteresis]`` and
  ``[chemical_step]`` sections switch those on.

.. note::

   There is deliberately no slot from a force into the equilibrium.  A
   reference pressure (as an improved well-balanced scheme needs) is resolved
   from ``rho`` inside the equilibrium and the source term, never supplied by
   a force, so the equilibrium/source pair stays a function of the moments.

2. ``init_state(setup)`` — initialisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~src.pipeline.runner.init_state` builds the starting
:class:`~src.pipeline.state.state.State`.  Populations come from
``setup.initial_f_fn`` (or a caller-supplied ``f``), the time from the init
snapshot (or 0), and ``ext`` from the addons' ``init_state`` slots.

.. important::

   ``lax.scan`` requires the carry pytree to have identical structure on every
   iteration.  The carry holds only ``f``, ``t`` and ``ext``, and ``ext``'s
   keys are fixed here, so nothing can appear mid-run.

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
        t: jnp.ndarray                 # scalar — current timestep
        ext: dict[str, Any]            # addon substates, keyed by addon name

The carry holds only what a next step cannot recompute: the physics reads
nothing else (checked by overwriting every other field with NaN before every
step, which left ``f`` bit-identical).  Density, velocity, forces and pressure
are derived on demand by :func:`~src.pipeline.step.observe`, which returns them
all at the state's own time level — on save steps only in streaming I/O, and
by mapping it over the trajectory in in-memory mode.  Snapshots keep the keys
``rho``, ``u``, ``force``, ``force_ext`` and ``pressure``.

``ext`` holds one entry per active addon, built by its ``init_state`` slot,
so its keys are fixed for the whole scan.  ``ext["wetting"]`` is a
:class:`~src.operators.wetting._params.WettingState` — the wall parameters
``phi_left``/``phi_right``, ``d_rho_left``/``d_rho_right``, the measured
angles ``ca_left``/``ca_right`` and the contact-line anchors
``cll_left``/``cll_right`` — present for a fixed-wetting run as well as a
hysteresis one; ``ext["electric"]`` is ``ElectricState(h)``, the electric-potential
distributions.  Being part of the carry is what lets the optimiser's result
persist from step to step.  Snapshots flatten each substate into top-level
keys, so a saved run still carries ``ca_left``, ``phi_left`` and ``h``.

Because ``NamedTuple`` is a pytree, the whole state maps cleanly through
``jit``, ``scan``, and ``vmap`` with no registration boilerplate.

Multiphase physics
------------------

There is one step, :func:`~src.pipeline.step.step`: forces, then
:func:`~src.pipeline.step.kernel`, then each addon's ``post_step``.  For a
multiphase run the kernel runs:

1. ``ρ`` and ``u`` from the moments of ``f``
2. bulk chemical potential ``μ₀(ρ)`` from the configured EOS
3. total chemical potential ``μ = μ₀ − κ∇²ρ``
4. interparticle force ``F_int = −ρ∇μ``
5. force-corrected velocity ``u_eq = u + F_total/(2ρ)``

then equilibrium → collision (+ source) → streaming → obstacle → boundary
conditions, and the moments of the result.  The kernel binds ``State.ext``
into the differential operators and names no addon.

.. warning::

   Lattice symmetry is load-bearing.  In a periodic domain there is no
   viscous sink for a systematic net force, so it accumulates.  Droplet
   centres must be integer-aligned to preserve the D4 discrete symmetry.

The wetting addon builds on top: its ``ghost_fill`` rewrites the wall ghost
cells of the density stencils from ``ext["wetting"]``, and its ``post_step``
either re-measures the angles (fixed wetting) or runs the hysteresis
optimiser, which solves for the wall parameters each step with a
``lax.while_loop`` around Adam on trial passes of the kernel under candidate
parameters, targeting an advancing/receding angle window.  See :doc:`operators` for the
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
