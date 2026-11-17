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

That runs a 100×100 single-phase D2Q9 domain for 101 steps, saving a
snapshot every 100, and writes everything to a timestamped run directory.
Before committing to a long run, inspect what the configuration resolves to::

    uv run tud-lbm run examples/config_simple.toml --dry-run

Values can be changed without editing the file.  Override keys may be a bare
field name or a dotted section path (``simulation_type.simulation_name``,
``multiphase.kappa``, ``hysteresis.ca_advancing``,
``forces.gravity_force.force_g``), and values are parsed as TOML literals —
so strings need quotes::

    uv run tud-lbm run examples/config_simple.toml \
        --override tau=0.8 \
        --override nt=5000 \
        --override simulation_name="my_first_run"

The same run from Python
------------------------

The CLI is a thin wrapper over a four-step functional API — configuration
in, final state out:

.. code-block:: python

    from src import SimulationConfig, build_setup, init_state, observe, run

    config = SimulationConfig(grid_shape=(64, 64), tau=0.8, nt=1000)
    setup = build_setup(config)
    state = init_state(setup)
    final_state, trajectory = run(setup, state, nt=config.nt)

    print(observe(setup, final_state).rho.shape)   # (64, 64, 1, 1, 1)

A bare :class:`~src.config.simulation_config.SimulationConfig` defaults to a
D2Q9 lattice, BGK collision, the default periodic boundary condition on all four edges, and
rest-equilibrium initialisation.  The state carries only the populations
``f``, the time ``t`` and the addon substates; :func:`~src.pipeline.step.observe`
derives density, velocity, forces and pressure from it.

``trajectory`` is the stacked history of every step, so its memory grows with
``nt`` times the grid — a 48³ D3Q19 run over 6000 steps needs about 100 GB.
For anything long use the streaming mode described in
:ref:`quickstart-streaming`, or advance without keeping the history:

.. code-block:: python

    import jax
    from src.pipeline.step import step

    final_state = jax.lax.scan(lambda s, _: (step(setup, s), None), state, None, length=1000)[0]

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
    grid_shape = [128, 128]
    lattice_type = "D2Q9"
    tau = 0.99
    nt = 500
    init_type = "multiphase_bubbles"

    [initialisation]             # keyword arguments for the initialiser
    centres = [[0.5, 0.5]]
    radii = [0.2]
    dispersed = "liquid"

    [multiphase]                 # its presence makes the run multiphase
    kappa = 0.04
    rho_l = 1.0
    rho_v = 0.001
    interface_width = 5
    eos = "double-well"

    [boundary_conditions]        # per edge; an omitted edge is periodic (the default)
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
``[<name>_force]`` table with a schema in ``config/forces.py``
(``[gravity_force]``, ``[gravity_masked_force]``, ``[electric_force]``),
which loads into ``config.forces``.  Unknown keys in ``[simulation_type]``
are collected into ``config.extra`` rather than rejected.  A file written in
the old layout, with ``type = ...`` in ``[simulation_type]``, still loads; see
:doc:`adapters`.

Choosing the physics
~~~~~~~~~~~~~~~~~~~~

There is no simulation type.  Each section switches on what it describes,
and they combine:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Section
     - Adds
   * - (none)
     - single-phase BGK/MRT collide–stream loop with boundary conditions
   * - ``[multiphase]``
     - chemical-potential multiphase simulation with the well-balanced LBM (WB-LBM) and a specified EOS. Set by ``eos``, ``kappa``, ``rho_l``, ``rho_v``, ``interface_width``.
   * - ``[boundary_conditions]``
     - a boundary condition per edge; the default boundary condition is periodic. Set by e.g. ``bottom = "bounce-back"``.
   * - ``[obstacle]``
     - a circular obstacle with a bounce-back boundary condition. Set by ``center_x``, ``center_y``, ``radius``.
   * - ``[wetting]`` + a ``"wetting"`` edge
     - wetting boundary condition with a fixed contact angle at that edge. Set by ``phi_left``, ``phi_right``, ``d_rho_left``, ``d_rho_right``.
   * - ``[hysteresis]``
     - wetting boundary condition with contact angle hysteresis on that edge. Set by ``ca_advancing``, ``ca_receding``.
   * - ``[chemical_step]``
     - an edge with two regions, each with its own wetting boundary condition, with or without contact angle hysteresis. Set by ``ca_advancing_pre_step``, ``ca_receding_pre_step``, ``ca_advancing_post_step``, ``ca_receding_post_step``; ``chemical_step_location`` separates the regions.
   * - ``[<name>_force]``
     - a body force (``gravity_force``, ``gravity_masked_force``, ``electric_force``; at most one gravity force). Set by e.g. ``force_g``, ``inclination_angle_deg``.

Without ``[hysteresis]``, the pre-step region of a chemical step follows the ``[wetting]`` values and the post-step region takes the clamp limit toward the post-step contact angles. Every wetting section needs ``[multiphase]`` and a ``"wetting"`` edge.

``[multiphase]`` needs ``eos``, ``kappa``, ``rho_l``, ``rho_v`` and
``interface_width``; ``eos = "carnahan-starling"`` additionally needs
``a_eos``, ``b_eos``, ``r_eos`` and ``t_eos``, and may omit ``rho_l``/``rho_v``:
they are then the Maxwell coexistence pair of those four (omit them to sweep
``t_eos``).  A 3D run (``lattice_type =
"D3Q19"``) supports the bulk physics only: see :doc:`operators`.

.. _terminology:

Terminology
~~~~~~~~~~~

The documentation, the CLI and the error messages use one term for one thing:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Term
     - Meaning
   * - edge
     - A side of the domain: ``bottom``, ``top``, ``left``, ``right``, and
       ``front``/``back`` on a 3D lattice.
   * - boundary condition
     - What an edge has (``periodic``, ``bounce-back``, ``wetting``, ...).
       The default boundary condition is ``periodic``.
   * - wetting boundary condition
     - A ``"wetting"`` edge with ``[wetting]``: a fixed contact angle.
   * - contact angle hysteresis
     - ``[hysteresis]``: the wetting boundary condition holds the contact angle
       between the receding contact angle (``ca_receding``) and the advancing
       contact angle (``ca_advancing``).
   * - chemical step, chemical step location
     - ``[chemical_step]``: the wetting edge is split at
       ``chemical_step_location`` into two regions.
   * - pre-step region, post-step region
     - The two regions of a chemical step. The inclusion starts on the
       pre-step region and moves toward the post-step region.
   * - wetting band
     - The ghost cells round a contact line that the wetting boundary
       condition modifies.
   * - inclusion
     - The dispersed droplet or bubble.
   * - body force
     - A force from a ``[<name>_force]`` section.
   * - WB-LBM
     - The well-balanced lattice Boltzmann method that the multiphase model uses.

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

A sweep is staged, not run: ``tud-lbm sweep stage`` writes one completed TOML
per combination of its ``--set`` axes (the Cartesian product).  The paths use
the ``--override`` syntax::

    uv run tud-lbm sweep stage cfg.toml --set 'tau=[0.6, 0.7, 0.8]' \
        --set 'gravity_force.force_g=[1e-6, 5e-6]' --out configs/var_tau_g/   # six files

Each file is ``<simulation_name>.toml``, validated and with every default filled
in.  Name the runs with ``--name 'var_{tau}_{force_g}'`` (fields are the last
key of each axis).  Run the files one by one with ``tud-lbm run``, or pass the
directory to ``scripts/db_pipeline.sh``, then compare them with
``tud-lbm compare``.  A list value in a config is an error, not a sweep.

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
