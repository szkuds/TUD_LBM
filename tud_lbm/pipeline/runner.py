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
from pathlib import Path
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
import tud_lbm.config.config_overview as _flags
from tud_lbm.pipeline.state.state import State

if TYPE_CHECKING:
    from tud_lbm.config import SimulationConfig
    from tud_lbm.io import SimulationIO
    from tud_lbm.pipeline.setup import SimulationSetup


# ── State initialisation ─────────────────────────────────────────────


def _t_from_snapshot(config: SimulationConfig) -> jnp.ndarray:
    """Infer the starting timestep from an ``init_from_file`` snapshot name.

    Expected filename format is ``timestep_{N}.npz``. Any non-conforming
    name, parse failure, or non-file-based init type falls back to ``0``.
    """
    if config.init_type != "init_from_file" or config.init_dir is None:
        return jnp.array(0)

    stem = Path(config.init_dir).stem
    if not stem.startswith("timestep_"):
        return jnp.array(0)

    step_str = stem.removeprefix("timestep_")
    if not step_str.isdigit():
        return jnp.array(0)

    return jnp.array(int(step_str))


def init_state(
    setup: SimulationSetup,
    *,
    f: jnp.ndarray | None = None,
    init_kwargs: dict | None = None,
) -> State:
    """Create an initial :class:`State` for the given setup.

    If *f* is not supplied, the population distribution is initialised
    using ``setup.initial_f_fn``, which is built at setup time from
    ``init_type``.  For ``"standard"`` this is the rest equilibrium
    (``f_i = w_i``); for multiphase types it produces a tanh density
    profile at equilibrium.  Remaining state arrays are initialised
    with zeros so that the pytree structure stays constant across
    ``lax.scan`` iterations (JAX requires identical carry structure).

    Args:
        setup: :class:`~setup.simulation_setup.SimulationSetup`.
        f: Optional pre-computed initial distribution function,
           shape ``(nx, ny, nz, q, 1)``.
        init_kwargs: Extra keyword arguments forwarded to the
           initialisation function (e.g. ``density``, ``rho_l``,
           ``rho_v``, ``interface_width``, ``npz_path``).

    Returns:
        A :class:`State` ready to be passed to :func:`run`.
    """
    from tud_lbm.pipeline.state import build_extra_state
    from tud_lbm.pipeline.state import build_optional_fields

    lattice = setup.lattice
    nx, ny, nz = setup.grid_shape[0], setup.grid_shape[1], setup.grid_shape[2]

    if f is None:
        if setup.initial_f_fn is None:
            msg = "initial_f_fn is required in SimulationSetup to initialize state"
            raise TypeError(msg)
        f = setup.initial_f_fn(init_kwargs)

    rho = jnp.sum(f, axis=-2, keepdims=True)
    u = jnp.zeros((nx, ny, nz, 1, lattice.d))
    t = _t_from_snapshot(setup.config)

    force, force_ext = build_optional_fields(setup, nx, ny, nz, lattice.d)
    extra_state = build_extra_state(setup)

    return State(
        f=f,
        rho=rho,
        u=u,
        t=t,
        force=force,
        force_ext=force_ext,
        **extra_state,
    )


# ── Functional run ───────────────────────────────────────────────────


def run(
    setup: SimulationSetup,
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
        nt: Number of time steps.  Defaults to ``setup.config.nt``.
        save_interval: Snapshot frequency (default: 1 = every step).
        io_handler: Optional :class:`~util.io.SimulationIO`.  When
            supplied, snapshots are streamed to disk via host callbacks
            and the returned *trajectory* is ``None``.
        skip_interval: Absolute simulation-time threshold; steps with
            ``state.t <= skip_interval`` are not saved.  For a fresh
            run this equals the number of initial steps to skip.  For
            a resumed ``init_from_file`` run, compare against the
            *absolute* timestep, not the number of new steps
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
    if nt is None:
        nt = setup.config.nt

    if setup.step_fn is None:
        msg = "step_fn is required in SimulationSetup to run simulation"
        raise TypeError(msg)
    _step_fn = setup.step_fn  # capture narrowed value for closures

    # ── Stability diagnostics (--debug-stability) ────────────────
    # Flag read as a module attribute at call time; a from-import would
    # freeze the value at import, before the CLI can set it.
    do_stab = None
    if _flags.DEBUG_FLAG_STABILITY:
        from tud_lbm.io.analysis.stability import make_stability_callback

        stab_dir = (
            Path(io_handler.run_dir)
            if io_handler is not None
            else Path(setup.config.results_dir).expanduser() / "stability_debug"
        )
        do_stab = make_stability_callback(
            stab_dir,
            gradient_density=setup.gradient_density,
            mp=setup.multiphase_params,
            log_interval=save_interval if save_interval >= 1 else max(1, nt // 10),
            vapor_frac=_flags.STABILITY_VAPOR_FRACTION,
            grad_frac=_flags.STABILITY_GRAD_RHO_FRACTION,
        )

    # ── Streaming I/O mode ───────────────────────────────────────
    if io_handler is not None:
        from tud_lbm.pipeline.io_callbacks import _state_to_numpy
        from tud_lbm.pipeline.io_callbacks import make_save_callback

        do_save = make_save_callback(
            io_handler,
            save_interval=save_interval,
            skip_interval=skip_interval,
            save_fields=save_fields,
        )

        @jax.jit
        def scan_body_io(state: State, _t: int) -> tuple[State, None]:
            new_state = _step_fn(setup, state)
            if do_stab is not None:
                do_stab(new_state, new_state.t)
            do_save(new_state, new_state.t)
            return new_state, None

        final_state, _ = jax.lax.scan(
            scan_body_io,
            initial_state,
            jnp.arange(nt),
        )

        final_t = int(final_state.t)
        io_handler.save_data_step(
            final_t,
            _state_to_numpy(final_state, fields=save_fields, t=final_t),
        )
        return final_state, None

    # ── In-memory trajectory mode ────────────────────────────────
    @jax.jit
    def scan_body(state: State, _t: int) -> tuple[State, State]:
        new_state = _step_fn(setup, state)
        if do_stab is not None:
            do_stab(new_state, new_state.t)
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
