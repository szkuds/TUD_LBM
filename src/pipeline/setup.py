"""JAX-friendly SimulationSetup and ``build_setup`` factory.

:class:`SimulationSetup` is an immutable :class:`typing.NamedTuple` that holds
everything the jitted step needs. It is built from a validated
:class:`~src.config.simulation_config.SimulationConfig` by :func:`build_setup`.

The pattern ``build_setup`` follows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Plain operators** (collision, equilibrium, source term, streaming, BC,
  obstacle, initialiser, force, macroscopic) are homogeneous and selectable:
  each is a registry lookup of the configured name. They need no coupling.
* **Features** (registry kind ``feature``; wetting, electric potential) add
  behaviour that touches several parts of a step. Every registered feature is
  built, and those the config uses return a
  :class:`~src.features.contribution.Contribution` filling up to four fixed slots:

  - ``ghost_fill`` — composed into the density gradient and Laplacian;
  - ``forces`` — appended to the configured forces;
  - ``post_step`` — run by :func:`~src.pipeline.step.step` after the kernel;
  - ``init_state`` — the feature's ``State.ext`` entry.

  Features are merged in **name order** and must be independent of each other.
  Adding a feature is adding a ``src/features/_*.py`` file; ``build_setup`` is
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
from src.features.contribution import compose_ghost_fill
from src.lattice.lattice import Lattice
from src.lattice.lattice import build_lattice

if TYPE_CHECKING:
    from src.config.multiphase_params import MultiphaseParams
    from src.config.simulation_config import SimulationConfig
    from src.features.contribution import Contribution
    from src.operators.protocols import BoundaryOperator
    from src.operators.protocols import CollisionOperator
    from src.operators.protocols import EquilibriumOperator
    from src.operators.protocols import ExtDifferentialOperator
    from src.operators.protocols import ForceOperator
    from src.operators.protocols import InitialPopulationOperator
    from src.operators.protocols import MacroscopicOperator
    from src.operators.protocols import ObstacleOperator
    from src.operators.protocols import SourceTermOperator
    from src.operators.protocols import StreamingOperator


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
        features: The active features as ``(name, contribution)`` pairs, in name
            order; the name is the feature's ``State.ext`` key.
        forces: Configured forces plus the features' ``forces`` slot.
        collision_fn / equilibrium_fn / source_fn / macroscopic_fn /
            streaming_fn / bc_fn: The plain operators, resolved at setup time.
        gradient_standard: Gradient of the chemical potential, ``∇μ``. Never
            carries a ghost fill.
        gradient_density: Density gradient ``∇ρ`` (source term), with the
            features' ghost fill.
        laplacian_density: Density Laplacian ``∇²ρ`` (chemical potential), with
            the features' ghost fill.
        initial_f_fn: Initial-population closure.
        k_diag: MRT relaxation rates (``None`` for BGK); already bound into
            ``collision_fn``.
        multiphase_params: ``None`` for single-phase runs.
        obstacle_mask / obstacle_fn: ``None`` without an obstacle.
    """

    config: SimulationConfig
    lattice: Lattice
    grid_shape: tuple[int, ...]
    tau: float
    collision_scheme: str
    features: tuple[tuple[str, Contribution], ...]
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
    obstacle_fn: ObstacleOperator | None = None


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
    # Function-local: importing the operator and feature packages at module level
    # closes an import cycle back into src.pipeline.
    from src.features import build_features
    from src.operators.boundary import build_bc
    from src.operators.collision import build_collision_fn
    from src.operators.differential import build_differential
    from src.operators.equilibrium import build_equilibrium_fn
    from src.operators.force import build_forces
    from src.operators.macroscopic import build_macroscopic_fn
    from src.operators.obstacle import build_obstacle_fn
    from src.operators.source_term import build_source_fn
    from src.operators.streaming import build_streaming_fn

    # 1. Lattice.
    lattice = build_lattice(config.lattice_type)
    mp_params = config.multiphase_params

    # 2. Feature merge: every registered feature the config uses, in name order.
    #    Consumed below: ghost_fill (3), forces (4); post_step by step(), init_state by init_state().
    features = build_features(config, lattice)

    # 3. Differential: ∇μ is always plain; the density operators carry the composed ghost fill.
    gradient_standard, _ = build_differential(lattice, config.pad_modes)
    gradient_density, laplacian_density = build_differential(lattice, config.pad_modes, compose_ghost_fill(features))

    # 4. Forces: the configured *_force sections, then the features' forces slot.
    forces = build_forces(config, tuple(config.grid_shape), lattice) + tuple(
        force for _, contribution in features for force in contribution.forces
    )

    # 5. Plain operators. The equilibrium and source term are a pair chosen by phase:
    #    single-phase keeps the pressure cs^2*rho in its equilibrium, multiphase ("wb")
    #    takes it from the force -rho*grad(mu).
    collision_fn = build_collision_fn(config.collision_scheme)
    if config.k_diag is not None:
        # A keyword the protocol lacks is bound, not typed: the configured rates
        # (shear entries already coupled to the relaxation time) reach collide_mrt.
        collision_fn = functools.partial(collision_fn, k_diag=jnp.asarray(config.k_diag))
    if config.is_multiphase:
        equilibrium_fn, source_fn = build_equilibrium_fn("wb"), build_source_fn("wb")
        macroscopic_fn = build_macroscopic_fn("multiphase")
    else:
        equilibrium_fn, source_fn = build_equilibrium_fn("standard_equilibrium"), build_source_fn("standard_source")
        macroscopic_fn = build_macroscopic_fn("standard")
    obstacle_mask = config.obstacle_mask

    # 6. Step: the setup the kernel and step() close over.
    return SimulationSetup(
        config=config,
        lattice=lattice,
        grid_shape=tuple(config.grid_shape),
        tau=config.tau,
        collision_scheme=config.collision_scheme,
        features=features,
        forces=forces,
        collision_fn=collision_fn,
        equilibrium_fn=equilibrium_fn,
        source_fn=source_fn,
        macroscopic_fn=macroscopic_fn,
        streaming_fn=build_streaming_fn(config.periodic_axes),
        bc_fn=build_bc(config.boundary_edges, lattice),
        gradient_standard=gradient_standard,
        gradient_density=gradient_density,
        laplacian_density=laplacian_density,
        initial_f_fn=_build_initial_f_fn(config, mp_params, lattice),
        k_diag=config.k_diag,
        multiphase_params=mp_params,
        obstacle_mask=obstacle_mask,
        obstacle_fn=build_obstacle_fn(obstacle_mask, lattice),
    )
