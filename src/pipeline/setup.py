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
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast
from src.lattice.lattice import Lattice
from src.lattice.lattice import build_lattice
from src.operators.differential import build_diff_ops
from src.operators.force import ForceSetup
from src.operators.force import build_forces

if TYPE_CHECKING:
    import jax.numpy as jnp
    from src.config.simulation_config import SimulationConfig
    from src.operators.boundary import BCMasks
    from src.operators.macroscopic import MultiphaseParams
    from src.operators.protocols import BoundaryOperator
    from src.operators.protocols import CollisionOperator
    from src.operators.protocols import DifferentialOperator
    from src.operators.protocols import EquilibriumOperator
    from src.operators.protocols import ExtraStatePlugin
    from src.operators.protocols import HysteresisOperator
    from src.operators.protocols import InitialPopulationOperator
    from src.operators.protocols import MacroscopicOperator
    from src.operators.protocols import ObstacleOperator
    from src.operators.protocols import StepOperator
    from src.operators.protocols import StreamingOperator
    from src.operators.protocols import WettingDifferentialOperator


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
        gradient_density_wetting: Parametric density gradient for hysteresis optimisation.
            Populated for hysteresis runs.
            Signature: ``(grid, phi_l, phi_r, d_rho_l, d_rho_r) -> result``.
            ``None`` for non-hysteresis cases.
        laplacian_density_wetting: Parametric Laplacian of density for hysteresis optimisation.
            Populated for hysteresis runs.
            Signature: ``(grid, phi_l, phi_r, d_rho_l, d_rho_r) -> result``.
            ``None`` for non-hysteresis cases.
        step_fn: The unbound step operator resolved from the registry,
            implementing :class:`~operators.protocols.StepOperator`.
            Signature: ``(setup, state) → state_next``.
        wetting_fn: The hysteresis operator for updating wetting state,
            implementing :class:`~operators.protocols.HysteresisOperator`.
            Built when ``hysteresis_config`` is present;
            ``None`` otherwise.
        wetting_edge: The wall used to orient contact-angle measurement
            (``"bottom"`` / ``"top"`` / ``"left"`` / ``"right"``), taken as the
            first ``"wetting"`` edge in ``bc_config``. Wetting is applied to
            every ``"wetting"`` edge, but measurement reads only this one, and a
            single ``WettingState`` parameter pair is shared across them.
            ``None`` for non-wetting runs.
        extra_state_plugins: Active plugin tuple used to initialise and update
            operation-specific extra state (e.g. electric potential, wetting state).
        collision_fn: Pre-built collision operator, resolved at setup time.
        equilibrium_fn: Pre-built equilibrium operator, resolved at setup time.
        macroscopic_fn: Pre-built macroscopic operator, resolved at setup time.
        streaming_fn: Pre-built streaming operator, resolved at setup time.
        bc_fn: Pre-built boundary-condition operator, resolved at setup time.
        initial_f_fn: Pre-built initial population function, resolved at setup time.
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
    obstacle_mask: jnp.ndarray | None = None
    obstacle_fn: ObstacleOperator | None = None

    # ── Differential operator closures (pre-built) ──
    gradient_standard: DifferentialOperator | None = None
    gradient_density: DifferentialOperator | None = None
    laplacian_density: DifferentialOperator | None = None
    gradient_density_wetting: WettingDifferentialOperator | None = None
    laplacian_density_wetting: WettingDifferentialOperator | None = None

    # ── Step function (unbound: (setup, state) -> State) ──
    step_fn: StepOperator | None = None
    wetting_fn: HysteresisOperator | None = None
    wetting_edge: str | None = None
    extra_state_plugins: tuple[ExtraStatePlugin, ...] = ()

    # ── Pre-built operator closures (resolved at setup time) ──
    collision_fn: CollisionOperator | None = None
    equilibrium_fn: EquilibriumOperator | None = None
    macroscopic_fn: MacroscopicOperator[..., tuple[jnp.ndarray, ...]] | None = None
    streaming_fn: StreamingOperator | None = None
    bc_fn: BoundaryOperator | None = None
    initial_f_fn: InitialPopulationOperator[..., jnp.ndarray] | None = None


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
    if config.wetting_config is not None and "multiphase" not in config.sim_type:
        msg = (
            f"Wetting configuration present but sim_type is '{config.sim_type}'. "
            "Wetting requires sim_type = 'multiphase'. "
            "Wetting is an addon to multiphase simulations, detected by the presence of [wetting] config."
        )
        raise ValueError(
            msg,
        )

    # ── Validation: hysteresis config requires a hysteresis sim_type ──
    # Otherwise build_setup would still build a hysteresis wetting_fn, but the
    # selected step operator would never supply a trial_step_fn, crashing the
    # optimiser with a cryptic 'NoneType is not callable' deep inside the trace.
    if config.hysteresis_config is not None and "hysteresis" not in config.sim_type:
        msg = (
            f"Hysteresis configuration present but sim_type is '{config.sim_type}'. "
            "The hysteresis optimiser only runs under sim_type = 'multiphase_hysteresis' "
            "(or 'multiphase_hysteresis_chemical_step'). "
            "Set one of those sim_types, or remove the [hysteresis] config for a static wetting run."
        )
        raise ValueError(
            msg,
        )

    # Import here to avoid circular import issues at module level
    from src.operators.boundary import build_bc
    from src.operators.boundary import build_bc_masks
    from src.operators.collision import build_collision_fn
    from src.operators.equilibrium import build_equilibrium_fn
    from src.operators.initialise import build_initialise_fn
    from src.operators.macroscopic import build_macroscopic_fn
    from src.operators.macroscopic import build_multiphase_params
    from src.operators.obstacle import build_obstacle_fn
    from src.operators.obstacle import build_obstacle_mask
    from src.operators.step import build_step_fn
    from src.operators.streaming import build_streaming_fn
    from src.operators.wetting import build_wetting_fn
    from src.registry import get_operators

    lattice = build_lattice(config.lattice_type)
    bc_masks = build_bc_masks(tuple(config.grid_shape))

    # Build multiphase params if applicable (multiphase runs with optional wetting)
    mp_params = build_multiphase_params(config) if "multiphase" in config.sim_type else None

    # Build force specs
    forces = build_forces(config, tuple(config.grid_shape), lattice)

    # Build differential operators (returns 5-tuple: standard, density, laplacian, raw_density, raw_laplacian)
    gradient_standard, gradient_density, laplacian_density, gradient_density_wetting, laplacian_density_wetting = (
        build_diff_ops(config, mp_params, lattice)
    )

    # Resolve step operator from registry
    step_fn = build_step_fn(config.sim_type)

    # Build operator closures (pre-resolved at setup time)
    collision_fn = build_collision_fn(config.collision_scheme)
    equilibrium_fn = build_equilibrium_fn("wb")
    streaming_fn = build_streaming_fn("standard", config.bc_config)
    macroscopic_fn = (
        build_macroscopic_fn("multiphase")  # unified multiphase op; EOS selected from mp.eos
        if "multiphase" in config.sim_type
        else build_macroscopic_fn("standard")  # single-phase
    )
    bc_fn = build_bc(config.bc_config, lattice)

    obstacle_mask = build_obstacle_mask(config.obstacle_config, cast("tuple[int, int, int]", tuple(config.grid_shape)))
    obstacle_fn = build_obstacle_fn(obstacle_mask, lattice)

    # Build wetting function for hysteresis-capable runs.
    wetting_fn = None
    if config.hysteresis_config is not None:
        wetting_scheme = (
            "chemical_step_hysteresis" if config.sim_type == "multiphase_hysteresis_chemical_step" else "hysteresis"
        )
        wetting_fn = build_wetting_fn(wetting_scheme)

    # Orient measurement from the first wetting wall. Wetting BCs are applied
    # to every "wetting" edge, but contact angles are read only at this one.
    from src.operators.wetting._edge_config import _resolve_wetting_edges

    wetting_edges = _resolve_wetting_edges(config.bc_config) if config.bc_config else []
    wetting_edge = wetting_edges[0][0] if wetting_edges else None

    extra_state_plugins = tuple(
        cast("ExtraStatePlugin", entry.target)
        for entry in sorted(get_operators("extra_state").values(), key=lambda e: e.name)
        if cast("ExtraStatePlugin", entry.target).is_active(config)
    )

    def _initial_f_fn(init_kwargs: dict | None = None) -> jnp.ndarray:
        kw: dict = {}
        if mp_params is not None:
            kw.update(rho_l=mp_params.rho_l, rho_v=mp_params.rho_v, interface_width=mp_params.interface_width)
        kw.update(config.initialisation)
        if init_kwargs:
            kw.update(init_kwargs)
        if config.init_type == "init_from_file" and "npz_path" not in kw and config.init_dir is not None:
            kw["npz_path"] = config.init_dir
        return build_initialise_fn(config.init_type)(cast("tuple[int, int, int]", config.grid_shape), lattice, **kw)

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
        obstacle_mask=obstacle_mask,
        obstacle_fn=obstacle_fn,
        gradient_standard=gradient_standard,
        gradient_density=gradient_density,
        laplacian_density=laplacian_density,
        gradient_density_wetting=gradient_density_wetting,
        laplacian_density_wetting=laplacian_density_wetting,
        step_fn=step_fn,
        wetting_fn=wetting_fn,
        wetting_edge=wetting_edge,
        extra_state_plugins=extra_state_plugins,
        collision_fn=collision_fn,
        equilibrium_fn=equilibrium_fn,
        macroscopic_fn=macroscopic_fn,
        streaming_fn=streaming_fn,
        bc_fn=bc_fn,
        initial_f_fn=_initial_f_fn,
    )
