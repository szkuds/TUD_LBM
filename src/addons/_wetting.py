"""Wetting addon — the wetting wall, contact-angle measurement and hysteresis.

The wall parameters always live in ``State.ext["wetting"]``
(:class:`~src.operators.wetting._params.WettingState`), for a fixed-wetting run
as well as a hysteresis one, so a single kernel path serves both and the
optimiser's trial steps.

* ``ghost_fill``: the wetting applicator rewrites the wall ghost cells of the
  density gradient and Laplacian from the four wall parameters of its substate.
* ``init_state``: the configured (or restored) parameters and the measured
  initial angles.
* ``post_step``: with ``[hysteresis]``, the optimiser (per-side windows from the
  chemical step when ``[chemical_step]`` is present); otherwise the
  contact-angle and contact-line measurement alone.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
from src.addons.hooks import Hooks
from src.operators.wetting._applicator import build_wetting_applicator
from src.operators.wetting._contact_angle import compute_contact_angle
from src.operators.wetting._contact_line import compute_contact_line_location
from src.operators.wetting._edge_config import first_wetting_edge
from src.operators.wetting._hysteresis import update_wetting_state
from src.operators.wetting._hysteresis import update_wetting_state_chemical_step
from src.operators.wetting._params import WettingParams
from src.operators.wetting._params import WettingState
from src.registry import addon_operator

if TYPE_CHECKING:
    from src.addons.hooks import GhostFill
    from src.addons.hooks import InitStateHook
    from src.addons.hooks import PostStepHook
    from src.config.simulation_config import SimulationConfig
    from src.pipeline.setup import SimulationSetup
    from src.pipeline.state.state import State

#: This addon's ``State.ext`` key, and its registered name.
NAME = "wetting"

#: What the ghost fill reads: the live state, or a candidate of the optimiser's trial steps.
WallParams = WettingParams | WettingState


def wetting_ghost_fill(config: SimulationConfig) -> GhostFill[WallParams]:
    """The wall ghost-cell rewrite, from the four wall parameters of the substate.

    The substate may be the live :class:`WettingState` or a candidate
    :class:`WettingParams`; both carry the four parameter names.
    """
    mp = config.multiphase_params
    assert mp is not None  # noqa: S101 - build_wetting_addon checks the phase first (_wall)
    apply = build_wetting_applicator(mp.rho_l, mp.rho_v, config.bc_config, config.chemical_step_wall)

    def ghost_fill(padded: jnp.ndarray, wall: WallParams) -> jnp.ndarray:
        return apply(padded, wall.phi_left, wall.phi_right, wall.d_rho_left, wall.d_rho_right)

    return ghost_fill


def _wall(config: SimulationConfig) -> tuple[float, str]:
    """``(rho_mean, edge)``: the phase threshold and the wall the angles are measured at.

    Raises:
        ValueError: On a single-phase run, or without a ``"wetting"`` edge.
            ``SimulationConfig`` accepts both (analysis reads such configs), but
            a run has no multiphase stencils to put the wall on, or no wall to
            apply it to and measure at.
    """
    mp = config.multiphase_params
    if mp is None:
        msg = (
            f"Wetting configuration present but sim_type is '{config.sim_type}'. "
            "Wetting is an addon to multiphase simulations: it modifies the multiphase density stencils."
        )
        raise ValueError(msg)
    edge = first_wetting_edge(config.bc_config)
    if edge is None:
        msg = 'A wetting run needs a "wetting" boundary edge to apply its wall on and measure its angles at.'
        raise ValueError(msg)
    return 0.5 * (mp.rho_l + mp.rho_v), edge


def _density(f: jnp.ndarray) -> jnp.ndarray:
    """``rho = sum_i f_i``, shape ``(nx, ny, nz, 1, 1)``; the only field the wetting hooks need."""
    return jnp.sum(f, axis=-2, keepdims=True)


def _measure(rho: jnp.ndarray, rho_mean: float, edge: str) -> dict[str, jnp.ndarray]:
    """``ca_*`` and ``cll_*`` of *rho* at the measurement wall."""
    ca_left, ca_right = compute_contact_angle(rho, jnp.array(rho_mean), edge=edge)
    cll_left, cll_right = compute_contact_line_location(rho, ca_left, ca_right, jnp.array(rho_mean), edge=edge)
    return {"ca_left": ca_left, "ca_right": ca_right, "cll_left": cll_left, "cll_right": cll_right}


def _init_state(config: SimulationConfig, rho_mean: float, edge: str) -> InitStateHook[WettingState]:
    """The configured wall parameters, the measured initial angles and, on a restart, the saved state."""
    defaults = config.wetting_defaults
    assert defaults is not None  # noqa: S101 - the addon is built only with wetting defaults
    params = WettingParams.from_defaults(defaults)

    def init_state(setup: SimulationSetup) -> WettingState:
        rho = _density(setup.initial_f_fn())
        wetting = WettingState(**params._asdict(), **_measure(rho, rho_mean, edge))
        # A hysteresis restart resumes the wall parameters and anchors it saved;
        # the angles above are always measured off the field.
        restored = setup.config.restored_wetting
        if restored is not None:
            wetting = wetting._replace(**{key: jnp.array(value) for key, value in restored.items()})
        return wetting

    return init_state


def build_measurement_hook(config: SimulationConfig) -> PostStepHook:
    """Fixed wetting: re-measure the angles; the wall parameters stay as configured.

    Also the ``no_optimiser`` rung of ``tud-lbm benchmark``'s ablation ladder.
    """
    rho_mean, edge = _wall(config)

    def post_step(setup: SimulationSetup, prev: State, new: State) -> State:  # noqa: ARG001 - PostStepHook shape
        wetting = new.ext[NAME]._replace(**_measure(_density(new.f), rho_mean, edge))
        return new._replace(ext=new.ext | {NAME: wetting})

    return post_step


def _optimiser(config: SimulationConfig, edge: str) -> PostStepHook:
    """Hysteresis: optimise the wall parameters for the next step on trial steps of the kernel."""
    hc = config.hysteresis_config
    assert hc is not None  # noqa: S101 - built only for a hysteresis run
    update = update_wetting_state if config.chemical_step_config is None else update_wetting_state_chemical_step
    trial_steps: int = hc["trial_steps"]

    def post_step(setup: SimulationSetup, prev: State, new: State) -> State:
        # Function-local: the pipeline imports the addons package through build_setup.
        from src.operators.force._force_aggregator import compute_total_force_ext
        from src.pipeline.step import kernel

        # The trial steps continue the step just taken, under the force it used.
        force_ext = compute_total_force_ext(setup, prev)

        def trial_step(candidate: WettingParams) -> tuple[jnp.ndarray, jnp.ndarray]:
            ext = new.ext | {NAME: candidate}

            def body(f: jnp.ndarray, _: None) -> tuple[jnp.ndarray, None]:
                return kernel(setup, f, force_ext, ext), None

            f_out, _ = jax.lax.scan(body, new.f, None, length=trial_steps)
            return f_out, _density(f_out)

        wetting = update(prev.ext[NAME], _density(new.f), setup, trial_step_fn=trial_step, edge=edge, t=new.t)
        return new._replace(ext=new.ext | {NAME: wetting})

    return post_step


@addon_operator(name=NAME)
def build_wetting_addon(config: SimulationConfig) -> Hooks[WallParams] | None:
    """The wetting hooks, or ``None`` without a ``[wetting]`` / ``[hysteresis]`` section."""
    if config.wetting_defaults is None:
        return None
    rho_mean, edge = _wall(config)
    post_step = build_measurement_hook(config) if config.hysteresis_config is None else _optimiser(config, edge)
    return Hooks(
        ghost_fill=wetting_ghost_fill(config),
        post_step=post_step,
        init_state=_init_state(config, rho_mean, edge),
    )
