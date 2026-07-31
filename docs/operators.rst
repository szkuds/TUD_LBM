Operators and the Registry
==========================

Every piece of physics in TUD-LBM is an *operator*: a function or class that
self-registers under a *kind* and is resolved by name at setup time.  Adding
new physics means adding a file, never editing a dispatch table.

The registry
------------

:mod:`src.registry` is the single source of truth.  It maintains two indexes,
``OPERATOR_REGISTRY`` keyed by ``"{kind}:{name}"`` and a secondary
``kind → {name → entry}`` index that keeps lookups O(1).  Each entry is a
frozen :class:`~src.registry.OperatorEntry` holding the name, kind, target,
and any metadata passed to the decorator.

Registration happens at import time:

.. code-block:: python

    from src.registry import collision_model

    @collision_model(name="bgk")
    def collide_bgk(f, feq, tau, source=None):
        ...

The name may also be taken from a ``name`` attribute on the decorated object
or, failing that, its ``__name__``.  Registering the same ``kind:name`` twice
raises ``ValueError`` — there is no silent override.

Query helpers
~~~~~~~~~~~~~

.. code-block:: python

    from src.registry import get_operators, get_operator_names, get_operator_category

    get_operator_category()              # {"collision_models", "eos", ...}
    get_operator_names("collision_models")   # {"bgk", "mrt"}
    get_operators("eos")["double-well"].target

``unregister_operator(kind, name)`` exists for tests only.

Operator kinds
--------------

Each kind has a convenience decorator in :mod:`src.registry`:

==========================  ==============================  ===========================================
Kind                        Decorator                       Resolved by
==========================  ==============================  ===========================================
``collision_models``        ``@collision_model``            ``config.collision_scheme``
``equilibrium``             ``@equilibrium_operator``       fixed (``"wb"``)
``macroscopic``             ``@macroscopic_operator``       single-phase vs. multiphase
``eos``                     ``@eos_operator``               ``config.eos``
``boundary_condition``      ``@boundary_condition``         ``config.bc_config`` per face
``force``                   ``@force_model``                ``*_force`` config sections
``stream``                  ``@stream_operator``            fixed (``"standard"``)
``differential``            ``@differential_operator``      built by ``build_diff_ops()``
``initialise``              ``@initialise_operator``        ``config.init_type``
``wetting``                 ``@wetting_operator``           ``config.hysteresis_config``
``obstacle``                ``@obstacle_operator``          ``config.obstacle_config``
``lattice``                 ``@lattice_operator``           ``config.lattice_type``
``update_timestep``         ``@update_timestep_operator``   ``config.sim_type``
``extra_state``             ``@extra_state_plugin``         active physics
``plotting``                ``@plotting_operator``          ``config.plot_fields``
``analysis``                ``@analysis_operator``          ``config.plot_fields``
``simulation_type``         ``@simulation_type_operator``   —
==========================  ==============================  ===========================================

.. note::

   The per-timestep step functions live in ``src/operators/step/`` but
   register under the kind ``update_timestep``, not ``step``.  The factory
   :func:`~src.operators.step.build_step_fn` hides this.

Auto-discovery
--------------

Each operator subpackage calls
:func:`~src.operators._loader.auto_load_operators` in its ``__init__.py``:

.. code-block:: python

    # src/operators/collision/__init__.py
    from src.operators._loader import auto_load_operators
    from src.operators.factory import build_operator

    auto_load_operators("src.operators.collision")

    def build_collision_fn(scheme: str) -> CollisionOperator:
        return cast("CollisionOperator", build_operator("collision_models", scheme))

The loader imports every ``_*.py`` module in the package directory, which
fires the decorators.  Sub-packages and public modules are skipped, so
implementation files must be named with a leading underscore to be
discovered.

All per-kind factories delegate to one generic
:func:`~src.operators.factory.build_operator`, which raises ``ValueError``
listing the valid names when a lookup fails.  Because
``build_operator`` returns ``Callable[..., object] | type``, callers cast to
the relevant protocol from :mod:`src.operators.protocols`.

Registered operators
--------------------

Collision — ``collision_models``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``bgk``
    Single-relaxation-time BGK,
    ``f_col = (1 − ω)·f + ω·feq + (1 − ω/2)·source`` with ``ω = 1/τ``.

``mrt``
    D2Q9 moment-space collision.  Requires ``k_diag`` (the diagonal
    relaxation rates) in the configuration; ``SimulationConfig`` rejects
    ``collision_scheme = "mrt"`` without it.

Equilibrium — ``equilibrium``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``wb``
    Well-balanced second-order Maxwell–Boltzmann equilibrium.  The rest
    population is recovered from mass conservation,
    ``f₀ = ρ − Σ_{i>0} feq_i``, which keeps a static interface at rest
    exactly.  ``build_setup()`` always selects this one; there is no
    configuration key for it.

``standard_equilibrium``
    Textbook second-order equilibrium, including the rest direction.
    Registered and available through
    :func:`~src.operators.equilibrium.build_equilibrium_fn`, but not wired
    into any simulation type.

Macroscopic — ``macroscopic``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``standard``
    ``ρ = Σᵢ fᵢ``, ``u = Σᵢ fᵢcᵢ / ρ``, with optional force correction
    ``u_eq = u + F/(2ρ)``.

``multiphase``
    Additionally computes the chemical potential and the interparticle force
    using the configured EOS.

Equation of state — ``eos``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``double-well``
    ``μ₀(ρ) = 2β(ρ − ρ_l)(ρ − ρ_v)(2ρ − ρ_l − ρ_v)`` with
    ``β = 8κ / (W²(ρ_l − ρ_v)²)``.  Here the interface width ``W`` is a real
    physics parameter.

``carnahan-starling``
    ``μ₀(ρ) = −2aρ + RT(1 + ln ρ) + 16RT(bρ − 12)/(bρ − 4)³``, parameterised
    by ``a_eos``, ``b_eos``, ``r_eos``, ``t_eos``.  ``interface_width`` is
    only a hint for the tanh initialiser, **not** physics.  The expression is
    singular at ``bρ = 4``; the physical range ``[ρ_v, ρ_l]`` sits well below
    it.  For typical LBM parameters ``μ₀`` is negative at both coexistence
    densities and positive in the unstable region between them — the
    double-well character is in the shape, not the sign.

Boundary conditions — ``boundary_condition``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configured per face in ``[boundary_conditions]`` and applied after
streaming.  Faces you omit default to ``periodic``.

``periodic``
    Wraps via ``jnp.roll``; carries pad-edge mode ``"wrap"`` for the
    differential operators.

``bounce-back``
    Solid wall — reverses populations against the wall.

``symmetry``
    Mirror plane (pad-edge mode ``"edge"``), including the diagonal roll
    correction at the top edge.

``velocity-inlet``
    Resets the inlet edge to the equilibrium distribution for a prescribed
    parabolic profile (``u0``).  Deliberately simple — not a Zou–He
    pressure-coupled inlet.

``outlet``
    Zero-gradient (Neumann) outflow on the ``"right"`` edge: the
    second-to-last column is copied onto the last one after streaming.

The value ``"wetting"`` may also appear on a face.  It is not a
boundary-condition operator: it marks which wall the wetting and hysteresis
machinery acts on, and the first such face becomes
``SimulationSetup.wetting_edge``, the wall against which contact angles are
measured.

Streaming — ``stream``
~~~~~~~~~~~~~~~~~~~~~~

``standard``
    ``jnp.roll`` per velocity direction, followed by zeroing of the layers
    that wrapped across **non-periodic** axes — without this, mass leaks
    through bounce-back walls.  :func:`build_streaming_fn` closes over
    ``bc_config`` so the periodic-axis mask is bound once at setup time.

Forces — ``force``
~~~~~~~~~~~~~~~~~~

Forces use a two-phase API.  ``build(params, grid_shape, **kwargs)`` runs at
setup time and may use numpy/scipy freely; ``compute(state, precomputed)``
runs inside the JIT trace and must be JAX-compatible.  ``build_forces()``
returns a :class:`~src.operators.force.ForceSetup` bundling the per-force
specs with the resolved source term.

``gravity_force``
    Uniform body force over the whole domain.

``gravity_masked_force``
    Body force restricted to the **dispersed** phase — the liquid in a
    droplet run, the vapour in a bubble run — via a density mask between
    ``rho_l`` and ``rho_v``.  The topology is resolved once at build time:
    ``[gravity_masked_force].dispersed`` wins, then
    ``[initialisation].dispersed``, then — for ``init_type =
    "init_from_file"`` only — the minority phase measured from the snapshot,
    falling back to ``"vapour"``.  A droplet run must therefore say
    ``dispersed = "liquid"``, including with the ``wetting``,
    ``wetting_drop_top`` and ``wetting_chem_step`` initialisers, which
    hard-code a liquid droplet and take no ``dispersed`` key of their own.
    Runtime wall-row detection is deliberately not used, because it collapses
    to "droplet" as soon as the dispersed phase leaves the wall, silently
    moving the momentum injection onto the whole continuous phase.

``source_term_wb``
    The well-balanced forcing source term used by the collision operators.

``electric_force``
    Electric potential carried by a second distribution ``h`` on the state,
    supplied through an ``extra_state`` plugin.

Setting ``gravity_force`` and ``gravity_masked_force`` together is a
configuration error.

Differential — ``differential``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Raw primitives ``gradient``, ``laplacian``, ``gradient_wetting``, and
``laplacian_wetting`` are rarely used directly.  Instead
:func:`~src.operators.differential.build_diff_ops` returns five closures with
the boundary pad modes, lattice weights, and wetting parameters already
bound:

.. code-block:: python

    (gradient_standard,
     gradient_density,
     laplacian_density,
     gradient_density_wetting,
     laplacian_density_wetting) = build_diff_ops(config, mp_params, lattice)

* ``gradient_standard(grid)`` — ∇μ, never wetting-corrected.
* ``gradient_density(grid)`` — ∇ρ for the source term, wetting-corrected
  when a wetting configuration is present.
* ``laplacian_density(grid)`` — ∇²ρ for the chemical potential, likewise.
* ``gradient_density_wetting(grid, phi_l, phi_r, d_rho_l, d_rho_r)`` and
  ``laplacian_density_wetting(...)`` — the *parametric* forms the hysteresis
  optimiser uses to build trial steps.  Both are ``None`` unless hysteresis
  is configured.

This is why :class:`~src.operators.protocols.DifferentialOperator` accepts
``*args, **kwargs``: it has to cover both the one-argument and the
five-argument call signature.

Initialisation — ``initialise``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selected by ``init_type``; keyword arguments come from the
``[initialisation]`` section.

``standard``
    Rest equilibrium, ``fᵢ = wᵢ`` — the single-phase default.

``multiphase_bubbles``
    One or more circular inclusions with a tanh interface profile.  Requires
    ``centres`` (``[fx, fy]`` fractions of the domain) and ``radii``
    (fractions of ``min(nx, ny)``); ``dispersed`` selects ``"vapour"``
    inclusions in liquid (the default) or ``"liquid"`` inclusions in vapour.

``multiphase_bubble_top``
    Inclusion attached to the top wall.

``wetting``, ``wetting_drop_top``, ``wetting_chem_step``
    Sessile configurations against the wetting wall; the chemical-step
    variant places the drop relative to the step.

``init_from_file``
    Resumes from a saved ``.npz`` snapshot.  Requires ``init_dir``.

.. note::

   Droplet centres should be integer-aligned on the grid.  In a periodic
   domain any systematic net force accumulates without viscous dissipation,
   and breaking the D4 discrete symmetry is enough to produce spurious drift.

Wetting — ``wetting``
~~~~~~~~~~~~~~~~~~~~~

``contact_angle``
    Measures left and right contact angles from the density field.

``contact_line_location``
    Locates the contact lines along the wetting wall.

``applicator``
    A *builder*: given ``rho_l``, ``rho_v``, and ``bc_config`` it returns a
    closure that applies the wetting ghost-cell correction to a padded
    density array, with the edge resolution baked in.

``hysteresis``
    Per-step solve for the wetting parameters ``phi``/``d_rho`` against a
    global advancing/receding angle window, using ``jax.lax.while_loop``
    around an Adam optimiser.  Configured by ``[hysteresis]``
    (``ca_advancing``, ``ca_receding``, ``learning_rate``,
    ``max_iterations``).

``chemical_step_hysteresis``
    Same machinery, but each side's target angles are chosen from the
    contact line's position relative to a chemical step defined in
    ``[chemical_step]``.

Contact-angle bookkeeping lives on
:class:`~src.pipeline.state.state.WettingState`, which is carried through
``lax.scan`` alongside the fields.

Step operators — ``update_timestep``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One per ``sim_type``: ``single_phase``, ``multiphase``,
``multiphase_wetting``, ``multiphase_hysteresis``, and
``multiphase_hysteresis_chemical_step``.  Each has signature
``(setup, state) → state`` and all multiphase variants route through the
shared ``_multiphase_pipeline`` helper, which sequences
equilibrium → collision → streaming → boundary conditions.

Obstacle — ``obstacle``
~~~~~~~~~~~~~~~~~~~~~~~

``circle``
    Circular interior obstacle, from ``[obstacle]``.  2-D only (``nz == 1``),
    and the configuration is validated to fit inside the grid.

Plotting and analysis
~~~~~~~~~~~~~~~~~~~~~

``plotting`` operators draw one panel per saved timestep: ``density``,
``velocity``, ``force``, ``force_ext``.

``analysis`` operators plot a whole run's snapshot history: ``avg_density``,
``total_mass``, ``max_velocity``, ``density_ratio``, ``contact_angle_left``,
``contact_angle_right``, ``contact_angles_pair``, ``contact_line_speed_left``,
``contact_line_speed_right``, ``contact_line_speeds_pair``,
``ca_theta_vs_time``, ``ca_theta_vs_x``, ``snapshot_fig``, and
``simulation_csv``.

Both kinds are named in ``plot_fields``.  Cross-run comparison is
deliberately **not** an operator — it lives in
:mod:`src.simulation_io.plotting.run_comparison` and reads the
``simulation_data.csv`` files produced by ``simulation_csv``.

Adding a new operator
---------------------

1. Create ``src/operators/<category>/_my_operator.py`` — the leading
   underscore is what makes auto-discovery pick it up.
2. Decorate the implementation with the kind's decorator and give it a name.
3. Match the corresponding protocol in :mod:`src.operators.protocols`.

.. code-block:: python

    # src/operators/collision/_trt.py
    from __future__ import annotations
    import jax.numpy as jnp
    from src.registry import collision_model

    @collision_model(name="trt")
    def collide_trt(f, feq, tau, source=None):
        """Two-relaxation-time collision."""
        ...

No factory, ``__init__.py``, or import list needs editing.  The operator is
immediately selectable from configuration (``collision_scheme = "trt"``) and
appears in ``tud-lbm run --list-simulation-operators``.

Protocols
---------

:mod:`src.operators.protocols` defines the structural contracts — operators
satisfy them by shape, not by inheritance.  The main ones are
``CollisionOperator``, ``StreamingOperator``, ``EquilibriumOperator``,
``MacroscopicOperator``, ``BoundaryOperator``, ``ObstacleOperator``,
``InitialiserOperator``, ``InitialPopulationOperator``, ``StepOperator``,
``MultiphaseStepOperator``, ``HysteresisOperator``, ``ForceOperator``,
``DifferentialOperator``, ``EOSFunction``, ``ExtraState``,
``ExtraStatePlugin``, ``PlotOperator``, ``ConfigReader``, and
``SimulationRepository``.

Conformance is exercised by the tests marked ``conformance`` in the test
suite.
