"""Functional ``lax.scan``-based simulation runner for TUD-LBM.

Provides :func:`run` — a thin wrapper around ``jax.lax.scan`` that
replaces the Python ``for``-loop in the legacy time stepper.

Two execution modes are supported:

**In-memory trajectory** (default)
    ``lax.scan`` returns the full trajectory; snapshots are selected
    after the scan completes via *save_interval*.  Simple but
    memory-heavy for long runs.

**Streaming I/O** (when *io_handler* is supplied)
    Uses ``jax.debug.callback`` to write snapshots to disk at the
    requested *save_interval*.  Device memory stays constant because
    no trajectory is accumulated.  The returned *trajectory* is
    ``None``.

.. note::

   Host callbacks are non-differentiable.  The streaming I/O path
   is intended for forward simulations only.  If you need gradients
   through the time loop, use the in-memory trajectory mode.

Usage::

    from setup.simulation_setup import build_setup
    from runner.run import run, init_state

    # In-memory (short runs / debugging)
    setup = build_setup(config)
    state = init_state(setup)
    final, trajectory = run(setup, state, nt=1000)

    # Streaming to disk (production)
    from util.io import SimulationIO
    io = SimulationIO(base_dir=config.results_dir,
                      config=config.to_dict(),
                      simulation_name=config.simulation_name)
    final, _ = run(setup, state, nt=config.nt,
                   save_interval=config.save_interval,
                   io_handler=io)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
from state.state import State

if TYPE_CHECKING:
    from util.io import SimulationIO


# ── State initialisation ─────────────────────────────────────────────


def init_state(
    setup,
    *,
    f: jnp.ndarray | None = None,
    init_kwargs: dict | None = None,
) -> State:
    """Create an initial :class:`State` for the given setup.

    If *f* is not supplied, the population distribution is initialised
    using the ``init_type`` specified in ``setup`` (via the
    :mod:`operators.initialise` factory).  For ``"standard"`` this is
    the rest equilibrium (``f_i = w_i``); for multiphase types it
    produces a tanh density profile at equilibrium.

    For multiphase simulations the ``force`` field is pre-populated
    with zeros so that the pytree structure stays constant across
    ``lax.scan`` iterations (JAX requires identical carry structure).

    Args:
        setup: :class:`~setup.simulation_setup.SimulationSetup`.
        f: Optional pre-computed initial distribution function,
           shape ``(nx, ny, q, 1)``.
        init_kwargs: Extra keyword arguments forwarded to the
           initialisation function (e.g. ``density``, ``rho_l``,
           ``rho_v``, ``interface_width``, ``npz_path``).

    Returns:
        A :class:`State` ready to be passed to :func:`run`.
    """
    from operators.initialise.factory import get_init_fn

    lattice = setup.lattice
    nx, ny = setup.grid_shape[0], setup.grid_shape[1]
    d = lattice.d

    if f is None:
        init_type = getattr(setup, "init_type", "standard")
        init_fn = get_init_fn(init_type)

        # Build kwargs from setup for multiphase initialisers
        kw: dict = {}
        mp = setup.multiphase_params
        if mp is not None:
            kw.update(
                rho_l=mp.rho_l,
                rho_v=mp.rho_v,
                interface_width=mp.interface_width,
            )
        # Allow caller overrides
        if init_kwargs:
            kw.update(init_kwargs)

        # For init_from_file, inject npz_path from init_dir
        if init_type == "init_from_file" and "npz_path" not in kw:
            init_dir = getattr(setup, "init_dir", None)
            if init_dir is not None:
                kw["npz_path"] = init_dir

        f = init_fn(nx, ny, lattice, **kw)

    rho = jnp.sum(f, axis=2, keepdims=True)  # (nx, ny, 1, 1)
    u = jnp.zeros((nx, ny, 1, d))
    t = jnp.array(0)

    # Pre-populate optional fields that the step function will write.
    # lax.scan requires carry pytree structure to be constant, so any
    # field that transitions from None → array must start as zeros.
    is_multiphase = setup.multiphase_params is not None
    force = jnp.zeros((nx, ny, 1, d)) if is_multiphase else None
    force_ext = jnp.zeros((nx, ny, 1, d)) if setup.force_enabled else None

    # Electric potential distribution (if electric force is enabled)
    h = None
    if getattr(setup, "electric_params", None) is not None:
        from operators.force.electric import init_hi

        ep = setup.electric_params
        h = init_hi(
            nx,
            ny,
            lattice,
            voltage_top=ep.voltage_top,
            voltage_bottom=ep.voltage_bottom,
        )

    return State(
        f=f,
        rho=rho,
        u=u,
        t=t,
        force=force,
        force_ext=force_ext,
        h=h,
    )


# ── Functional run ───────────────────────────────────────────────────


def run(
    setup,
    initial_state: State,
    nt: int | None = None,
    save_interval: int = 1,
    io_handler: SimulationIO | None = None,
    skip_interval: int = 0,
    save_fields: tuple[str, ...] | None = None,
) -> tuple[State, State | None]:
    """Run *nt* steps via ``jax.lax.scan``.

    Args:
        setup: :class:`~setup.simulation_setup.SimulationSetup`.
        initial_state: Starting :class:`~state.state.State`.
        nt: Number of time steps.  Defaults to ``setup.nt``.
        save_interval: Snapshot frequency (default: 1 = every step).
        io_handler: Optional :class:`~util.io.SimulationIO`.  When
            supplied, snapshots are streamed to disk via host callbacks
            and the returned *trajectory* is ``None``.
        skip_interval: Number of initial steps to skip before saving
            (only used with *io_handler*).
        save_fields: Subset of field names to write, e.g.
            ``("rho", "u")``.  ``None`` means all fields.
            Only used with *io_handler*.

    Returns:
        ``(final_state, trajectory)``

        * **No io_handler** — *trajectory* is a :class:`State` pytree
          with a leading axis of length *nt* (or *nt / save_interval*).
        * **With io_handler** — *trajectory* is ``None``; data has been
          written to ``io_handler.data_dir``.
    """
    from runner.step import get_step_fn

    if nt is None:
        nt = setup.nt

    step_fn = get_step_fn(setup)

    # ── Streaming I/O mode ───────────────────────────────────────
    if io_handler is not None:
        from runner.io_callbacks import make_save_callback

        do_save = make_save_callback(
            io_handler,
            save_interval=save_interval,
            skip_interval=skip_interval,
            save_fields=save_fields,
        )

        @jax.jit
        def scan_body_io(state, t):
            new_state = step_fn(state)
            do_save(new_state, t)
            return new_state, None

        final_state, _ = jax.lax.scan(
            scan_body_io,
            initial_state,
            jnp.arange(nt),
        )
        return final_state, None

    # ── In-memory trajectory mode ────────────────────────────────
    @jax.jit
    def scan_body(state, t):
        new_state = step_fn(state)
        return new_state, new_state

    final_state, trajectory = jax.lax.scan(
        scan_body,
        initial_state,
        jnp.arange(nt),
    )

    if save_interval > 1:
        idx = jnp.arange(0, nt, save_interval)
        trajectory = jax.tree.map(lambda x: x[idx], trajectory)

    return final_state, trajectory
