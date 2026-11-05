Adapters: Configuration In, Results Out
=======================================

Adapters are the boundary between the simulation core and the outside world.
Input adapters turn an external representation into a
:class:`~src.config.simulation_config.SimulationConfig`; output writers turn
a :class:`~src.pipeline.state.state.State` into files.  Neither the setup
code nor the operators know which adapter was used.

Configuration input
-------------------

All input adapters subclass
:class:`~src.protocols.ConfigAdapter` and implement two methods:

``load_raw(source) -> dict``
    Read the source into the nested section layout and hand it to
    :func:`~src.config.sections.load_sections`, which upgrades an old-layout
    config and flattens the sections onto field names.

``save(config, path) -> None``
    Serialise a config back out, from :func:`~src.config.sections.build_sections`.

The shared ``load()`` on the base class handles the rest
(:func:`~src.config.sections.config_from_flat`): normalising ``grid_shape``
to a tuple, keeping unknown keys in ``extra``, rejecting the old flat field
names, and constructing the frozen ``SimulationConfig`` (which validates
itself).  The layout is config knowledge, so it lives in
:mod:`src.config.sections`, not on the adapter base class.

.. code-block:: python

    from src.config.adapter_toml import TomlAdapter
    from src.config.adapter_dict import DictAdapter

    config = TomlAdapter().load("examples/config_simple.toml")

    config = DictAdapter().load({
        "grid_shape": (64, 64),
        "tau": 0.8,
        "nt": 1000,
    })

Both are also re-exported from :mod:`src.simulation_io.readers`.

:func:`~src.config.adapter_base.get_adapter` picks an adapter from a file
extension.  Only ``.toml`` is mapped today; adding a format means writing one
subclass and adding a line to ``_ADAPTER_MAP``.

``TomlAdapter``
~~~~~~~~~~~~~~~

Reads with the stdlib ``tomllib``.  Writing requires the optional
``tomli_w`` package — ``save()`` raises ``ImportError`` with an install hint
if it is missing.  Every run directory gets a ``config.toml`` written this
way, so a run is always reproducible from its own output.

``DictAdapter``
~~~~~~~~~~~~~~~

Accepts two shapes.  **Flat**, using ``SimulationConfig`` field names
directly::

    {"multiphase": {"eos": "double-well", "kappa": 0.04, ...},
     "forces": {"gravity_force": {"force_g": 1e-6}}}

or **nested**, mirroring the TOML section structure::

    {"simulation_type": {"nt": 1000, ...}, "multiphase": {...},
     "gravity_force": {"force_g": 1e-6}}

The nested form is detected by the presence of a ``simulation_type`` dict.
``DictAdapter.save()`` raises ``NotImplementedError`` — use ``TomlAdapter``
to write.

Section mapping
~~~~~~~~~~~~~~~

:func:`~src.config.sections.merge_sections` flattens the nested representation onto config fields.
The mapping is:

=========================  ===============================================
TOML section               ``SimulationConfig`` destination
=========================  ===============================================
``[simulation_type]``      top-level fields (``nt``, ``tau``, ``grid_shape``, …)
``[multiphase]``           ``multiphase``; its presence makes the run multiphase
``[boundary_conditions]``  ``boundary_conditions``
``[initialisation]``       ``initialisation``
``[wetting]``              ``wetting``
``[hysteresis]``           ``hysteresis``
``[chemical_step]``        ``chemical_step``
``[obstacle]``             ``obstacle``
``[<name>_force]``         ``forces["<name>_force"]``
``[output]``               ``results_dir``, ``save_fields``, ``plot_fields``,
                           ``animate_fields``, ``output_format``, …
=========================  ===============================================

Rules worth knowing:

* ``[simulation_type]`` is required; its absence raises ``ValueError``.
* Each section is stored whole on the field of the same name, so the section
  names *are* the field names, and ``--override multiphase.kappa=0.02`` needs
  no alias table.
* A ``[*_force]`` section with no schema in ``config/forces.py`` raises
  ``KeyError``; one that is not a table raises ``TypeError``.
* A flat old field name (``sim_type``, ``kappa``, ``wetting_config``, …)
  passed to ``config_from_flat`` raises ``ValueError`` instead of landing in
  ``extra``, which would silently have made the run single-phase.
* ``results_dir`` is passed through ``Path.expanduser()``, so ``~`` works.
* Keys that match no field are collected into ``config.extra`` rather than
  rejected.  This is how calibration results such as ``surface_tension`` are
  carried alongside a run.

The reverse direction is driven by the same metadata.  Every field carries a
``CONFIG_SECTION`` entry, and :func:`~src.config.sections.build_sections` buckets fields back into
sections so that a saved config round-trips to the shape it was loaded from.

Defaults applied on construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every default lives once, in :mod:`src.config.defaults`.
``SimulationConfig.__post_init__`` fills them in, so the object you get back —
and the ``config.toml`` written to the run directory — is not always what you
wrote, but always what the run used:

* ``grid_shape`` is promoted to 3-D — ``(64, 64)`` becomes ``(64, 64, 1)``.
* ``save_interval`` defaults to ``nt // 10`` when left at ``0``.
* every edge of the lattice gets a boundary condition, ``periodic`` when
  omitted (four edges on a 2D lattice, six on a 3D one), and an edge whose
  condition takes parameters gets them (``u0 = 0.05`` for a
  ``velocity-inlet``).
* ``equilibrium`` defaults by phase: ``standard_equilibrium`` single-phase,
  ``wb`` multiphase; the source term is the one it registered as its pair.
* the ``[hysteresis]``, ``[chemical_step]`` and ``[*_force]`` sections get
  their optimiser, step and force defaults, and a ``[hysteresis]`` section
  without ``[wetting]`` gets neutral wetting parameters (``phi = 1.0``,
  ``d_rho = 0.0``).
* an MRT ``k_diag`` has its shear entries rewritten to ``1/tau``.
* ``output_format`` is lower-cased, and a list collapses to its first entry.

Validation covers grid shape, lattice type, ``tau > 0.5``, ``nt > 0``,
collision scheme (with ``k_diag`` required for MRT), the equilibrium,
``init_type`` against the registry and ``init_dir`` for ``init_from_file``,
``save_fields`` membership, every boundary-condition name and parameter key,
obstacle geometry, each force against its schema, the wetting, hysteresis
and chemical-step sections, and — for multiphase — the common keys, that
``rho_l > rho_v``, and the further parameters of the selected EOS (the four
Carnahan–Starling ones).  Only one gravity force may be configured.  A 3D
lattice refuses what is still two-dimensional: wetting, hysteresis, chemical
step, the electric force, MRT and an obstacle.

Old-layout configs
~~~~~~~~~~~~~~~~~~

:mod:`src.config.legacy` upgrades a config written before the section
layout — ``[simulation_type] type = ...``, flat multiphase keys, per-force
fields, the ``*_config`` names.  ``TomlAdapter`` rewrites a run directory's
``config.toml`` in the new layout on its first load, through its own
``save``; any other file is upgraded in memory and left as it is.  The upgrade
is idempotent, and nothing outside ``legacy.py`` and its one call knows the
old layout.

Parameter sweeps
----------------

Fields declared with ``array_field()`` are sweep-eligible: giving one a list
value expands the configuration into the Cartesian product of all such
lists.

.. code-block:: python

    from src.config.array_expansion import expand_config

    configs, params = expand_config(raw_dict)
    len(configs)          # one per combination
    params.field_names    # frozenset of the fields that were swept

``expand_config`` takes the *raw dict* — the output of ``load_raw()``, before
``SimulationConfig`` instantiation — and returns
``(list[SimulationConfig], ArrayParameterSet | None)``.  The metadata is
``None`` when no arrays were found.  Pass ``allow_arrays=False`` to make list
values an error instead, which is what single-run code paths do.

Expansion applies at three levels: top-level sweep-eligible fields;
sub-keys inside the *nested sweepable* sections ``multiphase``, ``wetting``,
``hysteresis`` and ``chemical_step``; and, one level deeper, the keys of
each force under ``forces``.  That is what makes this work:

.. code-block:: toml

    [simulation_type]
    tau = [0.6, 0.7]

    [hysteresis]
    ca_advancing = [110.0, 120.0]

    [gravity_masked_force]
    force_g = [1e-6, 2e-6]           # 2 × 2 × 2 = 8 runs

On the command line the same sweep is
``--override forces.gravity_masked_force.force_g=[1e-6,2e-6]``.

Simulation output
-----------------

:class:`~src.simulation_io.save.SimulationIO` owns the run directory.
Constructing it creates a timestamped directory under ``base_dir``, writes
``config.toml`` and ``physical_parameters.txt``, and tees stdout/stderr into
a log file:

.. code-block:: python

    from src.simulation_io.save import SimulationIO

    io = SimulationIO(
        base_dir=config.results_dir,
        config=config,
        simulation_name=config.simulation_name,
        output_format=config.output_format,
    )
    io.run_dir     # <base_dir>/YYYY-MM-DD/HH-MM-SS_<simulation_name>
    io.data_dir    # <run_dir>/data

The directory name uses UTC.  ``output_format`` selects a writer from the
``output_writers`` registry in :mod:`src.simulation_io.output_data`:

``numpy``
    ``Numpy`` — one ``.npz`` per saved timestep, written with ``np.savez``.
    This is the format every analysis and plotting path reads.

``vtk``
    ``Vtk`` — VTK files for ParaView.

Writers are discovered the same way operators are.  Every module in the
``output_data`` package is imported at package load, and ``OutputWriter``
registers each subclass in ``__init_subclass__`` under its **lower-cased
class name** — which is why the class is ``Numpy`` and the config value is
``"numpy"``.

Streaming writes
~~~~~~~~~~~~~~~~

:mod:`src.simulation_io.callbacks` bridges the JIT boundary.
``make_save_callback(io_handler, save_interval, skip_interval, save_fields)``
returns a ``do_save(state, t)`` that is gated on-device with
``jax.lax.cond`` and dispatched through ``jax.debug.callback(ordered=True)``,
so saving happens from inside the ``lax.scan`` body without breaking the
trace.  ``_state_to_numpy`` converts the pytree to a numpy dict, always
including the wetting scalars, and raises ``FloatingPointError`` if the field
has gone NaN — a blown-up run fails loudly at the first bad snapshot instead
of writing a directory full of garbage.

:func:`~src.pipeline.runner.run` installs this only when an ``io_handler`` is
supplied.

Reading results back
--------------------

One rule governs the analysis layer:
:func:`~src.simulation_io.analysis.droplet_metrics.compute_droplet_series` is
the only code that reads ``.npz`` snapshots for droplet metrics, and
``build_simulation_csv`` is the only code that decides whether a run gets a
``simulation_data.csv``.

``DropletSeries`` is a frozen dataclass of per-snapshot arrays with
``cached_property`` derived quantities.  Every consumer — the CSV export and
the contact-angle, contact-line-speed, and regime-map plots — reads from it,
so *N* analysis operators over one run cost one pass over the snapshots
rather than *N* + 1.  ``series_for_files`` memoises on the resolved paths
plus a configuration fingerprint, bounded at eight runs.

.. warning::

   ``ca_left`` is overloaded.  In ``.npz`` files and CSV columns it is a
   contact **angle** in degrees.  On ``DropletSeries`` the angles are
   ``theta_left``/``theta_right`` and every ``ca*`` attribute is a capillary
   **number**.  ``to_dataframe()`` renames back to the historical column
   names on the way out.

Two further details matter for correctness:

* ``backward_diff(values, iterations, fallback_interval)`` divides by the
  *actual* iteration gap rather than the nominal ``save_interval``.  These
  differ on resumed runs, ``skip_interval`` runs, and pruned snapshots, where
  a fixed-interval difference over-reports contact-line speeds by the ratio
  of the gaps.
* ``MetricScales`` carries both ``sigma_measured`` (from the calibration
  cache, through ``cached_surface_tension(config)``) and ``sigma_analytical``
  (``(2/3)(κ/W)·Δρ²``).  ``sigma_primary`` prefers the measured value,
  because the closed form is not valid for ``carnahan-starling``.  The CSV
  exports ``sigma_lg``, ``sigma_lg_source``, and ``Ca_analytical`` so both
  are available.

Adding a format
---------------

**Input**: subclass ``ConfigAdapter``, implement ``load_raw`` and ``save``,
and register the extension in ``_ADAPTER_MAP``.  Reuse :func:`~src.config.sections.merge_sections`
if your format has the same section structure.

**Output**: add a module to ``src/simulation_io/output_data/`` with an
``OutputWriter`` subclass implementing ``save_data_step``.  The class name,
lower-cased, becomes the ``output_format`` value.  ``SimulationIO`` binds
``save_data_step`` as a bound method, so it receives ``self`` and can use
``self.data_dir``.

Neither change touches ``build_setup``, ``run``, or any operator.
