"""JAX-friendly SimulationSetup and ``build_setup`` factory.

:class:`SimulationSetup` is an immutable :class:`typing.NamedTuple`
(i.e. a valid JAX pytree) that holds everything needed by the jitted
step function.  It is built from a validated
:class:`~config.simulation_config.SimulationConfig` via
:func:`build_setup`.

Design rationale
~~~~~~~~~~~~~~~~
* **Closed over, not ``static_argnums``:**  ``SimulationSetup`` is
  captured by ``functools.partial`` (or a closure) when constructing
  the scan body.  JAX treats it as a compile-time constant and caches
  the compilation.  If the setup changes, a new compilation is
  triggered — the correct behaviour.
* **No mutable class instances:**  All fields are Python scalars,
  tuples, or ``jax.Array`` values.  No operator *objects* live here.

Usage::

    from config.simulation_config import SimulationConfig
    from setup.simulation_setup import build_setup

    cfg = SimulationConfig(grid_shape=(64, 64), tau=0.8)
    setup = build_setup(cfg)
"""

from __future__ import annotations
from collections.abc import Callable
from typing import NamedTuple
import jax.numpy as jnp
from config.simulation_config import SimulationConfig
from operators.boundary import BCMasks
from operators.differential import build_diff_ops
from operators.force import ForceSetup
from operators.force import build_forces
from operators.macroscopic import MultiphaseParams
from operators.protocols import BoundaryOperator
from operators.protocols import CollisionOperator
from operators.protocols import DifferentialOperator
from operators.protocols import EquilibriumOperator
from operators.protocols import HysteresisOperator
from operators.protocols import StepOperator
from operators.protocols import StreamingOperator
from setup.lattice import Lattice
from setup.lattice import build_lattice


class SimulationSetup(NamedTuple):
    """Immutable operator container — only what the jitted step needs.

    ``SimulationSetup`` stores built artifacts (operators, masks, closures)
    and physics scalars that the step function reads at JIT time.  IO and
    initialisation metadata live on the original
    :class:`~config.simulation_config.SimulationConfig`, accessible via
    the :attr:`config` reference.

    Attributes:
        config: The original :class:`SimulationConfig` — for IO, init,
            and any metadata that does not enter the JIT boundary.
        lattice: The :class:`~setup.lattice.Lattice` pytree.
        grid_shape: Spatial dimensions, e.g. ``(64, 64)``.
        tau: Relaxation time (> 0.5).
        collision_scheme: Name of the collision model (``"bgk"`` / ``"mrt"``).
        k_diag: MRT relaxation rates (``None`` for BGK).
        bc_masks: Pre-computed boundary-condition masks (:class:`BCMasks`).
        forces: Pre-built force setup (:class:`~operators.force.ForceSetup`) containing
            specs and source-term callable, or ``None`` if no forces are active.
        multiphase_params: ``None`` for single-phase runs.
        gradient_standard: Standard gradient ``∇μ`` (chemical potential).
            Always used for chemical-potential gradient. Never wetting-corrected.
        gradient_density: Density gradient ``∇ρ`` used in source term.
            Wetting-corrected when applicable.
        laplacian_density: Laplacian of density ``∇²ρ`` in chemical-potential computation.
            Wetting-corrected when applicable.
        step_fn: The unbound step operator resolved from the registry,
            implementing :class:`~operators.protocols.StepOperator`.
            Signature: ``(setup, state) → state_next``.
        wetting_fn: The hysteresis operator for updating wetting state,
            implementing :class:`~operators.protocols.HysteresisOperator`.
            Built only when both ``wetting_config`` and ``hysteresis_config`` are present;
            ``None`` otherwise.
        collision_fn: Pre-built collision operator, resolved at setup time.
        equilibrium_fn: Pre-built equilibrium operator, resolved at setup time.
        macroscopic_fn: Pre-built macroscopic operator, resolved at setup time.
        streaming_fn: Pre-built streaming operator, resolved at setup time.
        bc_fn: Pre-built boundary-condition operator, resolved at setup time.
    """

    # ── Core references ──
    config: SimulationConfig
    lattice: Lattice

    # ── Physics scalars (read at runtime inside JIT) ──
    grid_shape: tuple[int, ...]
    tau: float
    collision_scheme: str
    k_diag: tuple[float, ...] | None = None

    # ── Pre-built operators ──
    bc_masks: BCMasks | None = None
    forces: ForceSetup | None = None
    multiphase_params: MultiphaseParams | None = None

    # ── Differential operator closures (pre-built) ──
    gradient_standard: DifferentialOperator | None = None
    gradient_density: DifferentialOperator | None = None
    laplacian_density: DifferentialOperator | None = None

    # ── Step function (unbound: (setup, state) -> State) ──
    step_fn: StepOperator | None = None
    wetting_fn: HysteresisOperator | None = None

    # ── Pre-built operator closures (resolved at setup time) ──
    collision_fn: CollisionOperator | None = None
    equilibrium_fn: EquilibriumOperator | None = None
    macroscopic_fn: Callable[..., tuple[jnp.ndarray, ...]] | None = None
    streaming_fn: StreamingOperator | None = None
    bc_fn: BoundaryOperator | None = None


# ── Main factory ─────────────────────────────────────────────────────


def build_setup(config: SimulationConfig) -> SimulationSetup:
    """Construct a JAX-friendly :class:`SimulationSetup` from a config.

    Args:
        config: A validated :class:`SimulationConfig`.

    Returns:
        An immutable :class:`SimulationSetup` NamedTuple ready for the
        jitted step function.

    Raises:
        ValueError: If wetting configuration is present but sim_type is not "multiphase".
    """
    # ── Validation: wetting config requires multiphase sim_type ──
    if config.wetting_config is not None and config.sim_type != "multiphase":
        raise ValueError(
            f"Wetting configuration present but sim_type is '{config.sim_type}'. "
            "Wetting requires sim_type = 'multiphase'. "
            "Wetting is an addon to multiphase simulations, detected by the presence of [wetting] config."
        )

    # Import here to avoid circular import issues at module level
    from operators.boundary import build_bc
    from operators.boundary import build_bc_masks
    from operators.collision import build_collision_fn
    from operators.equilibrium import build_equilibrium_fn
    from operators.factory import build_operator
    from operators.macroscopic import build_macroscopic_fn
    from operators.macroscopic import build_multiphase_params
    from operators.streaming import build_streaming_fn
    from operators.wetting import build_wetting_fn

    lattice = build_lattice(config.lattice_type)
    bc_masks = build_bc_masks(tuple(config.grid_shape))

    # Build multiphase params if applicable (multiphase runs with optional wetting)
    mp_params = build_multiphase_params(config) if config.sim_type == "multiphase" else None

    # Build force specs
    force_setup = build_forces(config, tuple(config.grid_shape), lattice)
    # Convert to None if no forces are present
    forces = force_setup if force_setup.specs else None

    # Build differential operators
    gradient_standard, gradient_density, laplacian_density = build_diff_ops(config, mp_params, lattice)

    # Resolve step operator from registry
    step_fn = build_operator("update_timestep", config.sim_type)

    # Build operator closures (pre-resolved at setup time)
    collision_fn = build_collision_fn(config.collision_scheme)
    equilibrium_fn = build_equilibrium_fn("wb")
    streaming_fn = build_streaming_fn("standard")
    macroscopic_fn = (
        build_macroscopic_fn(mp_params.eos)  # EOS-aware for multiphase
        if config.sim_type == "multiphase"
        else build_macroscopic_fn("standard")  # single-phase
    )
    bc_fn = build_bc(config.bc_config, lattice)

    # Build wetting function if both wetting and hysteresis configs are present
    wetting_fn = None
    if config.wetting_config is not None and config.hysteresis_config is not None:
        wetting_fn = build_wetting_fn("hysteresis")

    return SimulationSetup(
        config=config,
        lattice=lattice,
        grid_shape=tuple(config.grid_shape),
        tau=config.tau,
        collision_scheme=config.collision_scheme,
        k_diag=config.k_diag,
        bc_masks=bc_masks,
        forces=forces,
        multiphase_params=mp_params,
        gradient_standard=gradient_standard,
        gradient_density=gradient_density,
        laplacian_density=laplacian_density,
        step_fn=step_fn,
        wetting_fn=wetting_fn,
        collision_fn=collision_fn,
        equilibrium_fn=equilibrium_fn,
        macroscopic_fn=macroscopic_fn,
        streaming_fn=streaming_fn,
        bc_fn=bc_fn,
    )
