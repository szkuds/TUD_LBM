"""JAX-friendly SimulationSetup and ``build_setup`` factory.

:class:`SimulationSetup` is an immutable :class:`typing.NamedTuple` that holds
everything the jitted step needs. It is built from a validated
:class:`~src.config.simulation_config.SimulationConfig` by :func:`build_setup`.

The pattern ``build_setup`` follows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Plain operators** (collision, equilibrium, source term, streaming, BC,
  obstacle, initialiser, force, macroscopic) are homogeneous and selectable:
  each is a registry lookup of the configured name. They need no coupling.
* **Addons** (registry kind ``addon``; wetting, electric potential) add
  behaviour that touches several parts of a step. Every registered addon is
  built, and those the config uses return a
  :class:`~src.addons.hooks.Hooks` filling up to three fixed slots:

  - ``ghost_fill`` — composed into the density gradient and Laplacian;
  - ``post_step`` — run by :func:`~src.pipeline.step.step` after the kernel;
  - ``init_state`` — the addon's ``State.ext`` entry.

  Addons are merged in **name order** and must be independent of each other.
  Adding an addon is adding a ``src/addons/_*.py`` file; ``build_setup`` is
  edited only to add a new *kind* of slot.

Usage::

    from src.config.simulation_config import SimulationConfig
    from src.pipeline.setup import build_setup

    cfg = SimulationConfig(grid_shape=(64, 64), tau=0.8)
    setup = build_setup(cfg)
"""

from __future__ import annotations
import functools
from typing import TYPE_CHECKING
from typing import NamedTuple
import jax.numpy as jnp
from src.addons.hooks import compose_ghost_fill
from src.lattice.lattice import Lattice
from src.lattice.lattice import build_lattice

if TYPE_CHECKING:
    from src.addons.hooks import Hooks
    from src.config.multiphase_params import MultiphaseParams
    from src.config.simulation_config import SimulationConfig
    from src.protocols import BoundaryOperator
    from src.protocols import CollisionOperator
    from src.protocols import EquilibriumOperator
    from src.protocols import ExtDifferentialOperator
    from src.protocols import ForceOperator
    from src.protocols import InitialPopulationOperator
    from src.protocols import MacroscopicOperator
    from src.protocols import SourceTermOperator
    from src.protocols import StreamingOperator


class SimulationSetup(NamedTuple):
    """Immutable operator container — only what the jitted step needs.

    IO and initialisation metadata live on the original
    :class:`~src.config.simulation_config.SimulationConfig`, reachable via
    :attr:`config`.

    Attributes:
        config: The original :class:`SimulationConfig`.
        lattice: The :class:`~src.lattice.lattice.Lattice` pytree.
        grid_shape: Spatial dimensions ``(nx, ny, nz)``.
        tau: Relaxation time (> 0.5).
        collision_scheme: Name of the collision model (``"bgk"`` / ``"mrt"``).
        addons: The active addons as ``(name, hooks)`` pairs, in name
            order; the name is the addon's ``State.ext`` key.
        forces: The configured forces.
        collision_fn / equilibrium_fn / source_fn / macroscopic_fn /
            streaming_fn / bc_fn: The plain operators, resolved at setup time.
        gradient_standard: Gradient of the chemical potential, ``∇μ``. Never
            carries a ghost fill.
        gradient_density: Density gradient ``∇ρ`` (source term), with the
            addons' ghost fill.
        laplacian_density: Density Laplacian ``∇²ρ`` (chemical potential), with
            the addons' ghost fill.
        initial_f_fn: Initial-population closure.
        k_diag: MRT relaxation rates (``None`` for BGK); already bound into
            ``collision_fn``.
        multiphase_params: ``None`` for single-phase runs.
        obstacle_mask: ``None`` without an obstacle; its bounce-back heads ``bc_fn``.
    """

    config: SimulationConfig
    lattice: Lattice
    grid_shape: tuple[int, ...]
    tau: float
    collision_scheme: str
    addons: tuple[tuple[str, Hooks], ...]
    forces: tuple[ForceOperator, ...]
    collision_fn: CollisionOperator
    equilibrium_fn: EquilibriumOperator
    source_fn: SourceTermOperator
    macroscopic_fn: MacroscopicOperator
    streaming_fn: StreamingOperator
    bc_fn: BoundaryOperator
    gradient_standard: ExtDifferentialOperator
    gradient_density: ExtDifferentialOperator
    laplacian_density: ExtDifferentialOperator
    initial_f_fn: InitialPopulationOperator
    k_diag: tuple[float, ...] | None = None
    multiphase_params: MultiphaseParams | None = None
    obstacle_mask: jnp.ndarray | None = None


def _build_initial_f_fn(
    config: SimulationConfig,
    mp_params: MultiphaseParams | None,
    lattice: Lattice,
) -> InitialPopulationOperator:
    """The initial-population closure: the configured initialiser bound to its keywords."""
    from src.operators.initialise import build_initialise_fn

    nx, ny, nz = config.grid_shape

    def _initial_f_fn(init_kwargs: dict | None = None) -> jnp.ndarray:
        kw: dict = {}
        if mp_params is not None:
            kw.update(rho_l=mp_params.rho_l, rho_v=mp_params.rho_v, interface_width=mp_params.interface_width)
        kw.update(config.initialisation)
        if init_kwargs:
            kw.update(init_kwargs)
        if config.init_type == "init_from_file" and "npz_path" not in kw and config.init_dir is not None:
            kw["npz_path"] = config.init_dir
        return build_initialise_fn(config.init_type)((nx, ny, nz), lattice, **kw)

    return _initial_f_fn


def build_setup(config: SimulationConfig) -> SimulationSetup:
    """Construct a :class:`SimulationSetup` from a validated config.

    The recipe runs in a fixed order; each step names the slots it consumes
    (see the module docstring for the pattern).

    Args:
        config: A validated :class:`SimulationConfig`.

    Returns:
        An immutable :class:`SimulationSetup` ready for the jitted step.
    """
    # Function-local: importing the operator and addon packages at module level
    # closes an import cycle back into src.pipeline.
    from src.addons import build_addons
    from src.config.equilibrium import source_term_for
    from src.operators.boundary import build_bc
    from src.operators.collision import build_collision_fn
    from src.operators.differential import build_differential
    from src.operators.equilibrium import build_equilibrium_fn
    from src.operators.force import build_forces
    from src.operators.macroscopic import build_macroscopic_fn
    from src.operators.macroscopic import build_multiphase_macroscopic_fn
    from src.operators.source_term import build_source_fn
    from src.operators.streaming import build_streaming_fn

    # 1. Lattice.
    lattice = build_lattice(config.lattice_type)
    mp_params = config.multiphase_params

    # 2. Addon merge: every registered addon the config uses, in name order.
    #    Consumed below: ghost_fill (3); post_step by step(), init_state by init_state().
    addons = build_addons(config)

    # 3. Differential: ∇μ is always plain; the density operators carry the composed ghost fill.
    gradient_standard, _ = build_differential(lattice, config.pad_modes)
    gradient_density, laplacian_density = build_differential(lattice, config.pad_modes, compose_ghost_fill(addons))

    # 4. Forces: the configured *_force sections.
    forces = build_forces(config, tuple(config.grid_shape), lattice)

    # 5. Plain operators. The equilibrium is the configured one and the source term
    #    the pair it registered (src.config.equilibrium); the macroscopic operator
    #    is the multiphase one exactly when the run has multiphase parameters.
    collision_fn = build_collision_fn(config.collision_scheme)
    if config.k_diag is not None:
        # A keyword the protocol lacks is bound, not typed: the configured rates
        # (shear entries already coupled to the relaxation time) reach collide_mrt.
        collision_fn = functools.partial(collision_fn, k_diag=jnp.asarray(config.k_diag))
    assert config.equilibrium is not None  # noqa: S101 - SimulationConfig._apply_defaults fills it in
    equilibrium_fn = build_equilibrium_fn(config.equilibrium)
    source_fn = build_source_fn(source_term_for(config.equilibrium))
    macroscopic_fn = build_macroscopic_fn() if mp_params is None else build_multiphase_macroscopic_fn(mp_params)
    obstacle_mask = config.obstacle_mask

    # 6. Step: the setup the kernel and step() close over.
    return SimulationSetup(
        config=config,
        lattice=lattice,
        grid_shape=tuple(config.grid_shape),
        tau=config.tau,
        collision_scheme=config.collision_scheme,
        addons=addons,
        forces=forces,
        collision_fn=collision_fn,
        equilibrium_fn=equilibrium_fn,
        source_fn=source_fn,
        macroscopic_fn=macroscopic_fn,
        streaming_fn=build_streaming_fn(config.periodic_axes),
        bc_fn=build_bc(config.boundary_edges, lattice, obstacle_mask),
        gradient_standard=gradient_standard,
        gradient_density=gradient_density,
        laplacian_density=laplacian_density,
        initial_f_fn=_build_initial_f_fn(config, mp_params, lattice),
        k_diag=config.k_diag,
        multiphase_params=mp_params,
        obstacle_mask=obstacle_mask,
    )
