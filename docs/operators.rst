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
``equilibrium``             ``@equilibrium_operator``       ``config.equilibrium``
``source_term``             ``@source_term_operator``       the equilibrium's registered pair
``macroscopic``             ``@macroscopic_operator``       whether ``[multiphase]`` is present
``eos``                     ``@eos_operator``               ``multiphase.eos``
``pressure``                ``@pressure_operator``          ``multiphase.eos`` (if it has one)
``surface_tension``         ``@surface_tension_operator``   ``multiphase.eos`` (if it has one)
``coexistence``             ``@coexistence_operator``       ``multiphase.eos`` (if it has one)
``boundary_condition``      ``@boundary_condition``         ``config.boundary_conditions`` per face
``force``                   ``@force_model``                the keys of ``config.forces``
``stream``                  ``@stream_operator``            fixed (``"standard"``)
``initialise``              ``@initialise_operator``        ``config.init_type``
``obstacle``                ``@obstacle_operator``          ``obstacle.shape``
``lattice``                 ``@lattice_operator``           ``config.lattice_type``
``addon``                   ``@addon_operator``             every one; active by config section
``plotting``                ``@plotting_operator``          ``config.plot_fields``
``analysis``                ``@analysis_operator``          ``config.plot_fields``
``dimensionless``           ``@dimensionless_operator``     every one, in ``physical_parameters.txt``
==========================  ==============================  ===========================================

.. note::

   There is no step kind.  One step, :func:`~src.pipeline.step.step`, serves
   every run; what differs between runs is which *addons* are active
   (``src/addons/``, kind ``addon``) and the slots they fill — see
   :doc:`architecture`.

Auto-discovery
--------------

Each operator subpackage calls
:func:`~src.operators._loader.auto_load_operators` in its ``__init__.py``:

.. code-block:: python

    # src/operators/collision/__init__.py
    from src.operators._loader import auto_load_operators
    from src.registry import get_operators

    auto_load_operators("src.operators.collision")

    def build_collision_fn(scheme: str) -> CollisionOperator:
        return get_operators("collision_models")[scheme].target

The loader imports every ``_*.py`` module in the package directory, which
fires the decorators.  Sub-packages and public modules are skipped, so
implementation files must be named with a leading underscore to be
discovered.

Every per-kind factory looks up ``get_operators(kind)[name].target`` and is
annotated with the relevant protocol from :mod:`src.protocols`; no
cast is needed. Names are validated by :class:`~src.config.simulation_config.SimulationConfig`
before a factory sees them, so an unknown name reaching a factory is a bare
``KeyError``.

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

Named by ``equilibrium`` in ``[simulation_type]``; without it a run takes
``standard_equilibrium`` single-phase and ``wb`` multiphase
(:data:`~src.config.equilibrium.DEFAULT_EQUILIBRIUM`).  The source term is not
configured separately: each equilibrium declares its pair at registration
(``@equilibrium_operator(name=..., source_term=...)``), because the pairing is
about where the pressure lives and mixing the pairs loses it or counts it
twice.  :mod:`src.config.equilibrium` reads and checks the pairing.

``wb``
    Well-balanced second-order Maxwell–Boltzmann equilibrium.  The rest
    population is recovered from mass conservation,
    ``f₀ = ρ − Σ_{i>0} feq_i``, which keeps a static interface at rest
    exactly.  It carries no pressure; a multiphase run supplies it as the
    force ``−ρ∇μ``.  Pairs with the ``wb`` source term.

``wb_hydro_inc``
    The ``wb`` equilibrium with a pressure ``p`` in its second moment
    (``w_i p / cs²`` on the moving directions, the rest direction again from
    mass conservation), after Zhang et al., Int. J. Multiphase Flow 198, 105668
    (2026).  Multiphase only.  The macroscopic operator supplies
    ``p = ρ₀(μ − μ_b)`` and reduces the interparticle force to
    ``−(ρ − ρ₀)∇μ``, so the momentum equation is unchanged.  ``ρ₀`` is
    ``[multiphase] equilibrium_reference_density`` (default: the vapour density
    at coexistence) and ``μ_b`` the chemical potential there.  Pairs with the
    ``wb`` source term.

``standard_equilibrium``
    Textbook second-order equilibrium, including the rest direction, so it
    carries the ideal-gas pressure ``cs²ρ``.  Pairs with ``standard_source``
    (Guo forcing).

Macroscopic — ``macroscopic``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both share one signature, ``(f, lattice, force, *, gradient_standard,
laplacian_density)``; the step calls whichever the run has without a phase
branch.

``standard``
    ``ρ = Σᵢ fᵢ``, ``u = Σᵢ fᵢcᵢ / ρ``, with optional force correction
    ``u_eq = u + F/(2ρ)``.  Takes the differential operators and ignores them.

``multiphase``
    Additionally computes the chemical potential and the interparticle force
    using the configured EOS.  The multiphase parameters are bound at build
    time (:func:`~src.operators.macroscopic.build_multiphase_macroscopic_fn`).

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

Coexistence densities — ``coexistence``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An EOS whose liquid and vapour densities follow from its own parameters
registers a solver here under its own name; membership is the capability
check, and :func:`~src.operators.macroscopic.eos.coexistence_densities`
returns ``(ρ_v, ρ_l)``, or ``None`` for an EOS outside the kind.

``carnahan-starling``
    The Maxwell construction: equal bulk pressure and equal chemical
    potential, i.e. equal areas under ``p₀(1/ρ)`` about the tie line
    ``p_sat``.  Solved by nested bisection under ``jax.jit`` from the two
    spinodals of ``dp₀/dρ``.  A ``t_eos`` at or above the critical
    temperature raises ``ValueError``.  ``rho_l``/``rho_v`` may be omitted
    from ``[multiphase]``; the config then fills them with this pair, rounded
    to six significant figures.  A configured value is never overwritten.

``double-well`` registers no solver: ``rho_l`` and ``rho_v`` are its
parameters, so both are required.

A multiphase run with a solver writes ``coexistence/data/data.json`` (the
pair, ``p_sat`` and the configured densities) and
``coexistence/plots/maxwell.png`` into its run directory at start, and
``physical_parameters.txt`` gets a *Maxwell coexistence* row.
``tud-lbm analyse config.toml --coexistence`` writes the same without a run.

Boundary conditions — ``boundary_condition``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configured per face in ``[boundary_conditions]`` and applied after
streaming, in the order bottom, top, left, right, front, back (a 2D lattice
has only the first four).  Faces you omit default to ``periodic``.  An
obstacle's bounce-back, when there is one, runs first.  A condition's
parameters go in the face's ``{edge}_{name}`` key (``left_velocity_inlet``),
completed from ``BOUNDARY_PARAMETERS`` in :mod:`src.config.defaults`; a key
that is not the parameter section of its face's condition is rejected.  Every
condition is written over the face's normal axis, so the same code serves the
z faces of a 3D run.

``periodic``
    Wraps via ``jnp.roll``; carries pad-edge mode ``"wrap"`` for the
    differential operators.

``bounce-back``
    Solid wall — reverses populations against the wall.

``symmetry``
    Specular (free-slip) mirror plane on any face, pad-edge mode ``"edge"``.
    Each population entering through the face is the one leaving with the
    normal component of its velocity flipped, taken from the neighbouring
    cell along its tangential components, so tangential momentum is kept.
    (The earlier 2D version paired the top and bottom diagonals the wrong way
    and reversed the tangential flow.)

``velocity-inlet``
    Resets the ``"left"`` edge to the equilibrium distribution for a
    prescribed parabolic profile of peak ``u0`` (default 0.05).  Deliberately
    simple — not a Zou–He pressure-coupled inlet.

``outlet``
    Zero-gradient (Neumann) outflow on the ``"right"`` edge: the
    second-to-last column is copied onto the last one after streaming.

``wetting``
    ``bounce-back`` registered a second time under this name.  On the
    populations it is a solid wall; the name marks the wall the wetting addon
    acts on, through the density stencils rather than the populations.  The
    first such face is the wall the contact angles are measured at.

Streaming — ``stream``
~~~~~~~~~~~~~~~~~~~~~~

``standard``
    ``jnp.roll`` per velocity direction, followed by zeroing of the layers
    that wrapped across **non-periodic** axes — without this, mass leaks
    through bounce-back walls.  :func:`build_streaming_fn` binds the run's
    periodic axes (an axis is periodic when both its faces are) once at setup
    time, so the operator is ``stream(f, lattice)``.

Forces — ``force``
~~~~~~~~~~~~~~~~~~

A force is a class bound to its payload (:class:`~src.protocols.ForceOperator`).
``build(params, grid_shape, *, config, lattice)`` runs at setup time, may use
numpy/scipy freely, and returns the bound instance; ``compute(state, *,
gradient_standard, gradient_density, laplacian_density)`` runs inside the JIT
trace and reads what ``build`` stored.  ``build_forces(config, grid_shape,
lattice)`` returns one bound force per entry of ``config.forces``.  What each
force's section must hold is its schema in :mod:`src.config.forces`, enforced
by the configuration, so ``build`` reads ``params[key]`` and checks nothing.

``gravity_force``
    Uniform body force over the whole domain.

``gravity_masked_force``
    Body force weighted by a band-limited phase indicator: exactly ``0``
    throughout the light phase, exactly ``Δρ·g`` throughout the dense phase,
    and linear between, with the ramp inset from each phase density by a
    tenth of the contrast.  The reference densities are the run's phase
    references — measured off the init field for ``init_from_file``, else
    ``rho_l``/``rho_v`` — because an equilibrated field relaxes away from the
    prescribed ones, and a plain ``(ρ − ρ_v)·g`` then drives the ambient
    vapour.

    There is no topology branch.  A liquid droplet is driven at ``Δρ·g`` with
    its vapour ambient at zero; a vapour bubble is itself force-free and the
    *liquid* around it carries ``Δρ·g``, so the bubble rises on the resulting
    pressure gradient with no sign flag anywhere.  Referencing ``ρ_v`` rather
    than ``ρ_l`` is the point: a body force enters as ``Δu = F/ρ``, so masking
    the net buoyancy onto the vapour instead accelerates it by a factor
    ``Δρ/ρ_v`` — around 800 at a realistic density ratio — and balancing it
    hydrostatically would need a pressure difference across the inclusion many
    times the vapour's absolute pressure, so no equilibrium exists and the gas
    evacuates.  The dense phase has no such limit.  Buoyancy on the inclusion
    is then not injected at all: it is the reaction to the liquid's
    hydrostatic gradient, which is also what supplies the added mass bounding
    the rise speed.  The net force over an inclusion is still ``Δρ·g·A``, the
    same ``Δρ`` the Bond, Archimedes and Reynolds numbers in
    ``physical_parameters.txt`` are built from, because a tanh interface is
    antisymmetric about ``rho_mean``.

    **Boundary conditions are load-bearing.**  The liquid can only stay at
    rest if it can hold a hydrostatic ramp along every direction carrying a
    gravity component, and a periodic axis cannot support a linear pressure
    ramp.  Under an inclined gravity the tangential component is then balanced
    by wall shear instead and the domain fills with Poiseuille flow, easily
    orders of magnitude above the inclusion's own drift.  Close that axis with
    ``left = right = "bounce-back"`` — not ``"symmetry"``, whose correction
    assumes a periodic x.

    Optional ``ramp_steps`` and ``ramp_start_t`` bring the force up linearly
    instead of instantaneously.  Switching gravity on abruptly leaves the fluid
    one acoustic crossing behind the force and rings for tens of thousands of
    steps; a ramp long compared to that crossing keeps the fluid
    quasi-statically balanced and excites no wave.  ``ramp_start_t`` is an
    *absolute* timestep, because ``state.t`` survives restarts — a resumed run
    must name the step at which gravity was first applied so every chunk agrees
    on the same ramp.

``electric_force``
    Leaky-dielectric force from the electric potential, a second distribution
    ``h`` in ``State.ext["electric"]`` that the ``electric`` addon
    initialises and advances.

Setting ``gravity_force`` and ``gravity_masked_force`` together is a
configuration error.

Source term — ``source_term``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not a force: it couples the total force into the populations, so it has its
own package, ``operators/source_term/``.  ``wb`` (with the ``grad(ρ)``
corrections that stand in for the pressure the ``wb`` equilibrium lacks) and
``standard_source`` (Guo forcing).  It is never configured; it is the pair the
equilibrium registered, and the kernel applies it whenever a force acts.

Differential operators
~~~~~~~~~~~~~~~~~~~~~~

There is no registry kind: nothing is selected by configuration.
:func:`~src.operators.differential.build_differential` returns the gradient
and the Laplacian, ``op(grid, ext)``, with the lattice, the pad modes and an
optional *ghost fill* bound.  The fill — the addons' composed
``ghost_fill`` slot — rewrites the padded field's ghost layer from the addon
substates in ``ext`` between padding and the stencil; without one the stencil
is plain and ignores ``ext``.  ``build_setup`` builds three:

* ``gradient_standard`` — ∇μ, never with a fill.
* ``gradient_density`` — ∇ρ for the source term, with the fill.
* ``laplacian_density`` — ∇²ρ for the chemical potential, with the fill.

The stencil core is chosen by the lattice dimension at build time: the D2Q9
cores in 2D, and the D3Q19 isotropic stencils ``∇f = 3Σᵢwᵢcᵢf(x+cᵢ)`` and
``∇²f = 6Σᵢwᵢ(f(x+cᵢ) − f(x))`` in 3D.  A 2D run is bit-identical to what it
was before the 3D cores were added.

The kernel binds ``State.ext`` into all three, so the macroscopic operator,
the source term and the forces call them as ``op(grid)``
(:class:`~src.protocols.DifferentialOperator`); the unbound form is
:class:`~src.protocols.ExtDifferentialOperator`.

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
    A centre on a wall (``fy`` of 0 or 1) gives a sessile inclusion.

``init_from_file``
    Resumes from a saved ``.npz`` snapshot.  Requires ``init_dir``.

.. note::

   Droplet centres should be integer-aligned on the grid.  In a periodic
   domain any systematic net force accumulates without viscous dissipation,
   and breaking the D4 discrete symmetry is enough to produce spurious drift.

Wetting — the ``wetting`` addon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wetting is an addon (``src/addons/_wetting.py``), active with a
``[wetting]`` or ``[hysteresis]`` section, not an operator kind.  Its wall
parameters always live in ``State.ext["wetting"]``, a fixed-wetting run
included, so one kernel path serves both kinds of run and the optimiser's
trial steps.  It fills three slots:

* ``ghost_fill`` — ``build_wetting_applicator`` (``operators/wetting/_applicator``)
  rewrites the wall ghost cells of the density stencils from the four wall
  parameters in ``ext``;
* ``init_state`` — the configured (or, on a hysteresis restart, the saved)
  parameters and the measured initial angles;
* ``post_step`` — without ``[hysteresis]``, re-measures the contact angles
  and lines (``compute_contact_angle``, ``compute_contact_line_location``);
  with it, runs one of the two optimisers in ``operators/wetting/_hysteresis``
  below, on trial passes of the kernel under candidate parameters.
  ``[chemical_step]`` selects the chemical-step variant.

The wall separates the phases at one fixed density,
:func:`~src.config.wetting.wall_rho_mean`: the midpoint of the run's phase
references, so the init field's own densities on an ``init_from_file`` run.
The applicator's contact-line anchors and contrast gate, the measured angles
and contact lines, the optimiser and the contact-angle overlay all use it.
Measuring it every step was tried and rejected: it changed angles by under
0.6° in the test cases and ran 6.5 times slower on a 201×101 bubble.

``update_wetting_state`` (``[hysteresis]``)
    Per-step solve for the wetting parameters ``phi``/``d_rho`` against a
    global advancing/receding angle window, using ``jax.lax.while_loop``
    around an Adam optimiser.  A side is pinned to its anchor
    (``cll_left``/``cll_right``) unless its angle is outside the window and its
    contact line is not moving the wrong way for the exceeded bound; only then
    does it target that bound and its anchor follow the line.  Configured by
    ``[hysteresis]`` (``ca_advancing``, ``ca_receding``, ``learning_rate``,
    ``learning_rate_above``, ``max_iterations``, ``max_iterations_above``,
    ``loss_tol``, ``trial_steps``, ``saturation_gap``); any other key is
    rejected.

``update_wetting_state_chemical_step`` (``[hysteresis]`` + ``[chemical_step]``)
    Same machinery, with each contact line's window taken from the surface it
    is on: the post-step window once the line is within ``edge_width``
    (default 1 cell) of the step or past it.  The step itself is a property
    of the wall: a line's band cells across the step keep their own surface's
    wettability (pre-step at the ``[wetting]`` values, post-step at the clamp
    limit toward the post surface).  A line receding onto the more wetting
    surface is held at the step, pinned to its position, and the post surface
    starts at the bound: until the angle is within ``saturation_gap`` (1°) of
    it, the optimiser is skipped and the wall takes its clamp limit.
    Configured by ``[chemical_step]``.

``trial_steps`` sets the horizon the optimiser sees, and the default of 2 is
too short to resolve the contact angle's response to the wall.  Replayed on a
chemical-step droplet (double-well, post-step window [30, 50]), varying ``phi``
from 1 to 2 moved the angle by 0.2° over 2 steps, with a gradient that changed
sign above ``phi`` ≈ 1.8, against 6° over 20 steps and 32° over 200 steps.
Over 2 steps the optimiser random-walks ``phi``.  Use tens of steps where the
angle must actually reach its bound; the cost is linear in ``trial_steps``.

Contact-angle bookkeeping lives on
:class:`~src.operators.wetting._params.WettingState`, the addon's
``State.ext["wetting"]``, which is carried through ``lax.scan`` alongside the
fields.

Step
~~~~

One step, :func:`~src.pipeline.step.step` ``(setup, state) → state``, for
every run: the forces, then :func:`~src.pipeline.step.kernel` (macroscopic,
equilibrium, collision + source, streaming, the boundary chain with an
obstacle's bounce-back first, macroscopic of the result), then the addons'
``post_step`` hooks in name order.

Three-dimensional runs
~~~~~~~~~~~~~~~~~~~~~~

``lattice_type = "D3Q19"`` with ``nz > 1`` runs the bulk physics in 3D:
streaming, every boundary condition on all six faces, the differential
stencils, the multiphase macroscopic operator, the well-balanced source term
and ``multiphase_bubbles`` (three-coordinate ``centres``).  Verified against
Poiseuille flow (0.13% from the parabola) and the Laplace law of a droplet
(within 10% at 32³; the measured ratio is 0.94).
What is still two-dimensional is refused by the configuration with an error
naming it: ``[wetting]``, ``[hysteresis]``, ``[chemical_step]``, the electric
force, MRT and ``[obstacle]``.  D2Q9 with ``nz > 1`` is refused too.

Obstacle — ``obstacle``
~~~~~~~~~~~~~~~~~~~~~~~

``circle``
    Circular interior obstacle, from ``[obstacle]``.  2-D only (``nz == 1``),
    and the configuration is validated to fit inside the grid.

The registered shapes only build the solid-cell mask.  The bounce-back on it
is the first link of the boundary chain: :func:`~src.operators.boundary.build_bc`
puts it ahead of the edge BCs, with the same ``(f_stream, f_col)`` shape.

Plotting and analysis
~~~~~~~~~~~~~~~~~~~~~

``plotting`` operators draw one panel per saved timestep: ``density``,
``velocity``, ``force``, ``force_ext``, ``pressure``, ``pressure_total``.

The two pressure operators are **opt-in** (``PlotOperator.opt_in = True``):
they are skipped by the "every registered operator" default and only render
when named explicitly in ``plot_fields`` / ``animate_fields``.  ``pressure``
shows the bulk EOS pressure ``p_0(rho)`` — the same function the Young-Laplace
surface-tension calibration uses — while ``pressure_total`` adds the
interfacial terms, ``p = p_0 - kappa (rho grad^2 rho + |grad rho|^2 / 2)``.
Both require a multiphase run whose EOS has a pressure implementation
(``double-well`` or ``carnahan-starling``); otherwise the panel is dropped.

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
3. Match the corresponding protocol in :mod:`src.protocols`.
4. Add a conformance test for it in
   ``tests/operators/utilities/test_protocols.py``: the type checker verifies
   everything downstream of a factory against the protocol, but not that a
   registered implementation conforms to it.

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

:mod:`src.protocols` holds every interface in the codebase.  The operator
protocols are structural contracts — operators satisfy them by shape, not by
inheritance: ``CollisionOperator``, ``StreamingOperator``,
``EquilibriumOperator``, ``MacroscopicOperator``, ``BoundaryCondition``,
``BoundaryOperator``, ``InitialiserOperator``, ``InitialPopulationOperator``,
``ForceOperator``, ``SourceTermOperator``, ``DifferentialOperator``,
``ExtDifferentialOperator`` and ``EosOperator``.  It also holds the addon hook
types (``GhostFill``, ``PostStepHook``, ``InitStateHook``), the analysis
``DimensionlessNumberOperator``, the wetting ``OptaxLike``, and the base classes
``PlotOperator``, ``AnalysisPlot``, ``ConfigAdapter`` and ``OutputWriter``.
At runtime it imports only the standard library, ``jax`` and ``numpy``
(``tests/test_import_constraints.py``).

Conformance is exercised by the tests marked ``conformance`` in the test
suite.
