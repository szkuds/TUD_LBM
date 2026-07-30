Quickstart
==========

This page takes you from a clean checkout to a finished simulation with
figures on disk.  It assumes no prior knowledge of the package internals —
for those, continue with :doc:`architecture`.

Installation
------------

TUD-LBM requires **Python 3.11 or newer**.  We recommend
`uv <https://docs.astral.sh/uv/getting-started/installation/>`_ for
reproducible environments::

    git clone https://github.com/szkuds/tud_lbm.git
    cd tud_lbm
    uv sync
    uv run tud-lbm --help

Optional extras are declared in ``pyproject.toml``:

===============  ============================================================
Extra            Contents
===============  ============================================================
``dev``          pytest, pytest-cov, coverage, ruff, ty
``docs``         sphinx, sphinx-rtd-theme, myst-parser, sphinx-autoapi
``animation``    moviepy — required by ``tud-lbm animate``
``publishing``   build, twine
===============  ============================================================

Install several at once with ``uv sync --extra dev --extra docs``.  Syncing a
single extra removes the packages belonging to the others, so pass every
extra you need in one command.

Your first simulation
---------------------

The fastest route is the command-line interface with one of the bundled
example configurations::

    uv run tud-lbm run examples/config_simple.toml

That runs a 100×100 single-phase D2Q9 domain for 1000 steps, saving a
snapshot every 100, and writes everything to a timestamped run directory.
Before committing to a long run, inspect what the configuration resolves to::

    uv run tud-lbm run examples/config_simple.toml --dry-run

Values can be changed without editing the file.  Override keys may be a bare
field name or a dotted section path (``simulation_type.simulation_name``,
``hysteresis.ca_advancing``), and values are parsed as TOML literals — so
strings need quotes::

    uv run tud-lbm run examples/config_simple.toml \
        --override tau=0.8 \
        --override nt=5000 \
        --override simulation_name="my_first_run"

The same run from Python
------------------------

The CLI is a thin wrapper over a four-step functional API — configuration
in, final state out:

.. code-block:: python

    from src import SimulationConfig, build_setup, init_state, run

    config = SimulationConfig(grid_shape=(64, 64), tau=0.8, nt=1000)
    setup = build_setup(config)
    state = init_state(setup)
    final_state, trajectory = run(setup, state, nt=config.nt)

    print(final_state.rho.shape)   # (64, 64, 1, 1, 1)

A bare :class:`~src.config.simulation_config.SimulationConfig` defaults to a
D2Q9 lattice, BGK collision, periodic boundaries on all six faces, and
rest-equilibrium initialisation.  ``trajectory`` is the stacked history of
every step; for long runs prefer the streaming mode described in
:ref:`quickstart-streaming`.

To start from a file instead, load it through an adapter:

.. code-block:: python

    from src.config.adapter_toml import TomlAdapter

    config = TomlAdapter().load("examples/config_simple.toml")

Anatomy of a configuration file
-------------------------------

TOML files are organised into sections that the adapter flattens onto
``SimulationConfig`` fields (see :doc:`adapters` for the full mapping).  Only
``[simulation_type]`` is mandatory:

.. code-block:: toml

    [simulation_type]
    simulation_name = "test_multiphase"
    type = "multiphase"          # → SimulationConfig.sim_type
    grid_shape = [128, 128]
    lattice_type = "D2Q9"
    tau = 0.99
    nt = 500
    init_type = "multiphase_bubbles"

    [initialisation]             # keyword arguments for the initialiser
    centres = [[0.5, 0.5]]
    radii = [0.2]
    dispersed = "liquid"

    [multiphase]                 # required when type contains "multiphase"
    kappa = 0.04
    rho_l = 1.0
    rho_v = 0.001
    interface_width = 5
    eos = "double-well"

    [boundary_conditions]        # per face; omitted faces default to periodic
    left = "periodic"
    right = "periodic"
    top = "periodic"
    bottom = "periodic"

    [output]
    results_dir = "~/TUD_LBM_data/test_runs"
    save_fields = ["rho", "u"]
    plot_fields = ["density"]

The recognised sections are ``[simulation_type]``, ``[initialisation]``,
``[multiphase]``, ``[boundary_conditions]``, ``[obstacle]``, ``[wetting]``,
``[hysteresis]``, ``[chemical_step]``, ``[output]``, and any
``[<name>_force]`` table matching a force field on the config
(``[gravity_force]``, ``[gravity_masked_force]``, ``[electric_force]``).
Unknown keys are collected into ``config.extra`` rather than rejected.

Choosing a simulation type
~~~~~~~~~~~~~~~~~~~~~~~~~~

``type`` selects the step operator that drives each timestep:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ``type``
     - Adds on top of the previous row
   * - ``single_phase``
     - BGK/MRT collide–stream–boundary loop
   * - ``multiphase``
     - chemical potential, interparticle force
   * - ``multiphase_wetting``
     - fixed contact-angle wall condition
   * - ``multiphase_hysteresis``
     - per-step optimisation of the wetting parameters against an
       advancing/receding contact-angle window
   * - ``multiphase_hysteresis_chemical_step``
     - per-side angle targets, chosen from the contact line's position
       relative to a chemical step

Any type containing ``multiphase`` requires ``[multiphase]`` with ``eos``,
``kappa``, ``rho_l``, ``rho_v``, and ``interface_width``.  Choosing
``eos = "carnahan-starling"`` additionally requires ``a_eos``, ``b_eos``,
``r_eos``, and ``t_eos``.

What a run produces
-------------------

Results land in a timestamped directory under ``results_dir``
(default ``~/TUD_LBM_data``, overridable with the ``TUD_LBM_DATA_DIR``
environment variable)::

    <results_dir>/2026-07-30/14-22-05_my_first_run/
    ├── config.toml               # the exact configuration used
    ├── physical_parameters.txt   # Bond/Ohnesorge numbers, derived scales
    ├── simulation.log            # stdout and stderr, tee'd
    ├── data/                     # one .npz per saved timestep
    └── plots/                    # figures, when plot_fields is set
        └── snapshots/            # one figure per saved timestep

``save_interval`` controls how often a snapshot is written and defaults to
``nt // 10``.  ``save_fields`` selects which arrays go into each ``.npz``;
``plot_fields`` and ``animate_fields`` select which figure operators run
afterwards.

.. _quickstart-streaming:

Streaming snapshots to disk
---------------------------

Passing an ``io_handler`` to :func:`~src.pipeline.runner.run` writes
snapshots from inside the ``lax.scan`` loop via ``jax.debug.callback``, so
the full trajectory never has to fit in memory.  This is what the CLI does,
and it is the right mode for anything longer than a few hundred steps:

.. code-block:: python

    from src.config.adapter_toml import TomlAdapter
    from src.pipeline.runner import init_state, run
    from src.pipeline.setup import build_setup
    from src.simulation_io.plotting.figure_builder import FigureBuilder
    from src.simulation_io.save import SimulationIO

    config = TomlAdapter().load("examples/config_simple.toml")
    setup = build_setup(config)
    state = init_state(setup)

    io = SimulationIO(
        base_dir=config.results_dir,
        config=config,
        simulation_name=config.simulation_name,
        output_format=config.output_format,
    )

    final_state, _ = run(
        setup,
        state,
        nt=config.nt,
        save_interval=config.save_interval,
        io_handler=io,
        save_fields=tuple(config.save_fields) if config.save_fields else None,
    )

    FigureBuilder(config=config, run_dir=io.run_dir).build_all()

In this mode ``run()`` returns ``(final_state, None)`` — the second element is
``None`` because no trajectory is accumulated.

Parameter sweeps
----------------

Any sweep-eligible field given a **list** value expands into a Cartesian
product of runs.  This works for top-level fields and for keys nested inside
``[wetting]``, ``[hysteresis]``, ``[chemical_step]``, and the force tables:

.. code-block:: toml

    [simulation_type]
    type = "single_phase"
    tau = [0.6, 0.7, 0.8]     # three runs

    [gravity_force]
    force_g = [1e-6, 5e-6]    # × two → six runs total

Run them in parallel and build cross-run comparison plots afterwards::

    uv run tud-lbm run examples/config_parallel.toml --max-workers 4 --compare

The same expansion is available from Python through
:func:`~src.config.array_expansion.expand_config`.

Looking at the results
----------------------

Post-processing commands operate on a finished run directory::

    uv run tud-lbm visualise <run_dir>            # field snapshots + analysis figures
    uv run tud-lbm visualise <run_dir> fields     # only per-timestep field maps
    uv run tud-lbm visualise <run_dir> analysis   # only snapshot-history plots
    uv run tud-lbm animate <run_dir> --fields density
    uv run tud-lbm compare <parent_dir>           # cross-run comparison for a sweep

Add ``--no-prompt`` to take the configured fields instead of being asked
interactively.  To see what is available before choosing::

    uv run tud-lbm run --list-simulation-operators
    uv run tud-lbm run --list-simulation-analysis

Where to go next
----------------

* :doc:`architecture` — how config, setup, state, and the step loop fit together.
* :doc:`operators` — the registry, every registered operator, and how to add one.
* :doc:`lattice` — velocity models and the 5-D array convention.
* :doc:`adapters` — configuration input and simulation output formats.
