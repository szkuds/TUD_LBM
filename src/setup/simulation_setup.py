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
from typing import Any
from typing import NamedTuple
import jax.numpy as jnp
from operators.differential.config import DifferentialConfig
from operators.differential.factory import build_differential_operators
from operators.differential.operators import DifferentialOperators
from operators.differential.pad_modes import determine_pad_modes
from setup.lattice import Lattice
from setup.lattice import build_lattice


class BCMasks(NamedTuple):
    """Pre-computed boundary-condition masks — valid JAX pytree.

    Each mask is a boolean ``jax.Array`` of shape ``(nx, ny, 1, 1)``
    that is ``True`` on the corresponding edge row/column.

    Attributes:
        top: Mask for the top boundary (y = ny-1).
        bottom: Mask for the bottom boundary (y = 0).
        left: Mask for the left boundary (x = 0).
        right: Mask for the right boundary (x = nx-1).
    """

    top: jnp.ndarray
    bottom: jnp.ndarray
    left: jnp.ndarray
    right: jnp.ndarray


class MultiphaseParams(NamedTuple):
    """Equation-of-state and surface-tension parameters.

    All fields are Python scalars (compile-time constants).
    """

    eos: str
    kappa: float
    rho_l: float
    rho_v: float
    interface_width: int
    bubble: bool = False
    rho_ref: float | None = None
    g: float | None = None


class SimulationSetup(NamedTuple):
    """Immutable, JAX-friendly simulation setup.

    Passed as a static / closed-over argument to jitted functions.
    All fields are either JAX arrays, Python scalars, tuples, or
    nested :class:`NamedTuple` values.

    Attributes:
        lattice: The :class:`~setup.lattice.Lattice` pytree.
        grid_shape: Spatial dimensions, e.g. ``(64, 64)``.
        tau: Relaxation time (> 0.5).
        nt: Number of time steps.
        collision_scheme: Name of the collision model (``"bgk"`` / ``"mrt"``).
        k_diag: MRT relaxation rates (``None`` for BGK).
        bc_config: Boundary-condition mapping (edge → type string).
        bc_masks: Pre-computed boundary-condition masks (:class:`BCMasks`).
        force_enabled: Whether external forcing is active.
        force_config: Optional dict with force parameters.
        init_type: Initialisation strategy identifier.
        init_dir: Directory for ``init_from_file`` runs.
        results_dir: Where to write output.
        save_interval: How often to save snapshots.
        skip_interval: Steps to skip before first save.
        save_fields: Which fields to save (``None`` = all defaults).
        multiphase_params: ``None`` for single-phase runs.
        wetting_config: Optional wetting configuration dict.
        hysteresis_config: Optional hysteresis configuration dict.
        extra: Catch-all for additional parameters.
    """

    # Lattice and grid
    lattice: Lattice
    grid_shape: tuple[int, ...]

    # Physics
    tau: float
    nt: int
    collision_scheme: str
    k_diag: tuple[float, ...] | None = None

    # Boundary
    bc_config: dict[str, Any] | None = None
    bc_masks: BCMasks | None = None

    # Force
    force_enabled: bool = False
    force_config: dict[str, Any] | None = None

    # Initialisation
    init_type: str = "standard"
    init_dir: str | None = None

    # IO
    results_dir: str = ""
    save_interval: int = 100
    skip_interval: int = 0
    save_fields: tuple | None = None  # tuple (immutable) instead of list

    # Multiphase (None for single-phase)
    multiphase_params: MultiphaseParams | None = None

    # Wetting / hysteresis
    wetting_config: dict[str, Any] | None = None
    hysteresis_config: dict[str, Any] | None = None

    # Gravity force (pre-computed template, or None)
    gravity_template: jnp.ndarray | None = None

    # Electric force params (or None)
    electric_params: Any | None = None  # ElectricParams NamedTuple

    # Differential operators (pre-built closures, or None)
    diff_ops: DifferentialOperators | None = None

    # Extra
    extra: dict[str, Any] | None = None


# ── Helper factories ─────────────────────────────────────────────────


def build_bc_masks(
    grid_shape: tuple[int, ...],
    bc_config: dict[str, Any] | None = None,
) -> BCMasks:
    """Construct pre-computed boundary-condition masks.

    Each mask is a boolean array of shape ``(nx, ny, 1, 1)`` that is
    ``True`` on the corresponding edge row/column.

    Args:
        grid_shape: Spatial dimensions ``(nx, ny, ...)``.
        bc_config: Boundary-condition mapping (unused for now; reserved
            for future edge-type encoding).

    Returns:
        A :class:`BCMasks` NamedTuple.
    """
    nx, ny = grid_shape[:2]
    top = jnp.zeros((nx, ny, 1, 1), dtype=bool).at[:, -1].set(True)
    bottom = jnp.zeros((nx, ny, 1, 1), dtype=bool).at[:, 0].set(True)
    left = jnp.zeros((nx, ny, 1, 1), dtype=bool).at[0, :].set(True)
    right = jnp.zeros((nx, ny, 1, 1), dtype=bool).at[-1, :].set(True)
    return BCMasks(top=top, bottom=bottom, left=left, right=right)


def build_multiphase_params(config) -> MultiphaseParams:
    """Construct :class:`MultiphaseParams` from a configuration object.

    Args:
        config: An object with multiphase attributes (``eos``, ``kappa``,
            ``rho_l``, ``rho_v``, ``interface_width``, …).

    Returns:
        A :class:`MultiphaseParams` NamedTuple.

    Raises:
        ValueError: If required multiphase fields are missing.
    """
    for name in ("eos", "kappa", "rho_l", "rho_v", "interface_width"):
        if getattr(config, name, None) is None:
            raise ValueError(f"'{name}' is required for multiphase simulations")

    return MultiphaseParams(
        eos=config.eos,
        kappa=config.kappa,
        rho_l=config.rho_l,
        rho_v=config.rho_v,
        interface_width=config.interface_width,
        bubble=getattr(config, "bubble", False),
        rho_ref=getattr(config, "rho_ref", None),
        g=getattr(config, "g", None),
    )


# ── Main factory ─────────────────────────────────────────────────────


def build_setup(config) -> SimulationSetup:
    """Construct a JAX-friendly :class:`SimulationSetup` from a config.

    *config* can be either a
    :class:`~config.simulation_config.SimulationConfig` or the legacy
    :class:`~app_setup.simulation_setup.SimulationSetup` dataclass — any
    object whose attributes match the expected field names.

    Args:
        config: A validated configuration object.

    Returns:
        An immutable :class:`SimulationSetup` NamedTuple ready for the
        jitted step function.
    """
    lattice = build_lattice(config.lattice_type)

    # Build pre-computed boundary masks
    bc_config = getattr(config, "bc_config", None)
    bc_masks = build_bc_masks(tuple(config.grid_shape), bc_config)

    # Build multiphase params if applicable
    mp_params: MultiphaseParams | None = None
    sim_type = getattr(config, "sim_type", "single_phase")
    if sim_type == "multiphase":
        mp_params = build_multiphase_params(config)

    # Normalise save_fields to tuple (immutable) or None
    sf = getattr(config, "save_fields", None)
    save_fields_tuple = tuple(sf) if sf is not None else None

    # Build force objects from force_config
    gravity_template = None
    electric_params = None
    force_config = getattr(config, "force_config", None)
    if force_config:
        import inspect
        from registry import get_operators

        force_ops = get_operators("force")

        # force_config can be a single dict or a list of dicts
        force_entries = force_config if isinstance(force_config, list) else [force_config]
        for entry in force_entries:
            if not isinstance(entry, dict):
                continue
            ftype = entry.get("type", "")
            if not ftype:
                continue

            if ftype not in force_ops:
                raise ValueError(f"Unknown force type '{ftype}'. Available: {sorted(force_ops)}")

            op_entry = force_ops[ftype]
            builder = op_entry.target

            # Build available kwargs: entry params (minus "type") + grid_shape
            available = {k: v for k, v in entry.items() if k != "type"}
            available["grid_shape"] = tuple(config.grid_shape)

            # Only pass params the builder actually accepts
            sig = inspect.signature(builder)
            kwargs = {k: v for k, v in available.items() if k in sig.parameters}
            result = builder(**kwargs)

            # Store result in the field indicated by registry metadata
            result_field = op_entry.metadata.get("result_field") if op_entry.metadata else None
            if result_field == "gravity_template":
                gravity_template = result
            elif result_field == "electric_params":
                electric_params = result

    # ── Build differential operators ──────────────────────────────────
    # Ensure BC modules are imported so their @boundary_condition decorators
    # have fired before we query pad_edge_mode metadata.
    from operators.boundary import _bounce_back as _bb  # noqa: F401
    from operators.boundary import _periodic as _per  # noqa: F401
    from operators.boundary import _symmetry as _sym  # noqa: F401

    pad_modes = determine_pad_modes(bc_config)
    diff_cfg = DifferentialConfig(
        w=lattice.w,
        c=lattice.c,
        pad_modes=pad_modes,
        wetting_params=getattr(config, "wetting_config", None),
        chemical_step=getattr(config, "chemical_step", None),
        bc_config=bc_config,
    )
    diff_ops = build_differential_operators(diff_cfg)
    # ──────────────────────────────────────────────────────────────────

    return SimulationSetup(
        lattice=lattice,
        grid_shape=tuple(config.grid_shape),
        tau=config.tau,
        nt=config.nt,
        collision_scheme=config.collision_scheme,
        k_diag=config.k_diag,
        bc_config=bc_config,
        bc_masks=bc_masks,
        force_enabled=getattr(config, "force_enabled", False),
        force_config=getattr(config, "force_config", None),
        init_type=getattr(config, "init_type", "standard"),
        init_dir=getattr(config, "init_dir", None),
        results_dir=getattr(config, "results_dir", ""),
        save_interval=getattr(config, "save_interval", 100),
        skip_interval=getattr(config, "skip_interval", 0),
        save_fields=save_fields_tuple,
        multiphase_params=mp_params,
        wetting_config=getattr(config, "wetting_config", None),
        hysteresis_config=getattr(config, "hysteresis_config", None),
        gravity_template=gravity_template,
        electric_params=electric_params,
        diff_ops=diff_ops,
        extra=getattr(config, "extra", None),
    )
