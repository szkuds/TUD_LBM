"""Invariants of the addon-slot step: one kernel, one step, addons through fixed slots.

* ``State.ext`` holds exactly the active addons' substates, and the scan carry
  keeps its structure across steps;
* the ghost fill reaches the density operators, never the chemical-potential gradient;
* the kernel is a pure function of its arguments and names no addon;
* a registered addon is picked up, merged in name order, and its hook sees the
  state before and after the step.
"""

from __future__ import annotations
import inspect
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.addons.hooks import Hooks
from src.config.simulation_config import SimulationConfig
from src.pipeline.runner import init_state
from src.pipeline.setup import build_setup
from src.pipeline.step import kernel
from src.pipeline.step import observe
from src.pipeline.step import step
from src.registry import register_operator
from src.registry import unregister_operator

_MULTIPHASE: dict[str, Any] = {
    "multiphase": {"eos": "double-well", "kappa": 0.017, "rho_l": 1.0, "rho_v": 0.33, "interface_width": 4},
    "grid_shape": (32, 24),
    "tau": 0.99,
    "nt": 3,
}
_WALL: dict[str, Any] = {"bottom": "wetting", "top": "bounce-back", "left": "periodic", "right": "periodic"}
_WETTING: dict[str, Any] = {"phi_left": 1.2, "phi_right": 1.2, "d_rho_left": 0.05, "d_rho_right": 0.05}
_HYSTERESIS: dict[str, Any] = {"ca_advancing": 120.0, "ca_receding": 60.0, "learning_rate": 0.01, "max_iterations": 2}
_ELECTRIC: dict[str, Any] = {
    "permittivity_liquid": 80.0,
    "permittivity_vapour": 1.0,
    "conductivity_liquid": 0.01,
    "conductivity_vapour": 0.001,
    "voltage_top": 1.0e-3,
    "voltage_bottom": 0.0,
}
_DROPLET: dict[str, Any] = {"init_type": "wetting", "initialisation": {"r": 8}}

_CASES: dict[str, tuple[dict[str, Any], set[str]]] = {
    "single_phase": ({"grid_shape": (16, 16), "tau": 0.8, "nt": 3}, set()),
    "multiphase": (_MULTIPHASE, set()),
    "wetting": (_MULTIPHASE | _DROPLET | {"boundary_conditions": _WALL, "wetting": _WETTING}, {"wetting"}),
    "hysteresis": (
        _MULTIPHASE | _DROPLET | {"boundary_conditions": _WALL, "wetting": _WETTING, "hysteresis": _HYSTERESIS},
        {"wetting"},
    ),
    "electric": (_MULTIPHASE | {"forces": {"electric_force": _ELECTRIC}}, {"electric"}),
    "wetting_and_electric": (
        _MULTIPHASE
        | _DROPLET
        | {"forces": {"electric_force": _ELECTRIC}, "boundary_conditions": _WALL, "wetting": _WETTING},
        {"electric", "wetting"},
    ),
}


@pytest.fixture(scope="module", params=sorted(_CASES))
def case(request):
    """``(setup, initial state, expected ext keys)`` for one addon combination."""
    kwargs, expected = _CASES[request.param]
    setup = build_setup(SimulationConfig(**kwargs))
    return setup, init_state(setup), expected


@pytest.fixture(scope="module")
def wetting_setup():
    kwargs, _ = _CASES["wetting"]
    setup = build_setup(SimulationConfig(**kwargs))
    return setup, init_state(setup)


# ── ext structure ────────────────────────────────────────────────────


def test_ext_holds_exactly_the_active_addons(case):
    setup, state, expected = case
    assert set(state.ext) == expected
    assert [name for name, _ in setup.addons] == sorted(expected)


def test_the_carry_keeps_its_structure_across_a_step(case):
    setup, state, _ = case
    new_state = step(setup, state)

    assert jax.tree_util.tree_structure(new_state) == jax.tree_util.tree_structure(state)
    assert int(new_state.t) == int(state.t) + 1
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(new_state))


# ── slot wiring ──────────────────────────────────────────────────────


def test_the_ghost_fill_reaches_the_density_operators_only(wetting_setup):
    """Changing the wall in ``ext`` moves the density gradient and Laplacian, never grad(mu)."""
    setup, state = wetting_setup
    rho = observe(setup, state).rho
    other = state.ext | {"wetting": state.ext["wetting"]._replace(phi_left=jnp.array(1.5), phi_right=jnp.array(1.5))}

    for op in (setup.gradient_density, setup.laplacian_density):
        assert not np.allclose(np.asarray(op(rho, state.ext)), np.asarray(op(rho, other)))
    np.testing.assert_array_equal(
        np.asarray(setup.gradient_standard(rho, state.ext)), np.asarray(setup.gradient_standard(rho, other))
    )


def test_without_a_ghost_fill_the_density_operators_ignore_ext():
    kwargs, _ = _CASES["multiphase"]
    setup = build_setup(SimulationConfig(**kwargs))
    rho = observe(setup, init_state(setup)).rho
    np.testing.assert_array_equal(
        np.asarray(setup.laplacian_density(rho, {})), np.asarray(setup.laplacian_density(rho, {"anything": 1.0}))
    )


# ── kernel purity ────────────────────────────────────────────────────


def test_the_kernel_is_a_pure_function_of_its_arguments(wetting_setup):
    setup, state = wetting_setup
    first = kernel(setup, state.f, None, state.ext)
    second = kernel(setup, state.f, None, state.ext)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_the_kernel_reads_the_wall_from_its_ext_argument(wetting_setup):
    """The hysteresis trial steps rely on this: a candidate wall in ``ext`` changes the pass."""
    setup, state = wetting_setup
    candidate = state.ext | {"wetting": state.ext["wetting"]._replace(phi_left=jnp.array(1.5))}

    f_live = kernel(setup, state.f, None, state.ext)
    f_candidate = kernel(setup, state.f, None, candidate)
    assert not np.allclose(np.asarray(f_live), np.asarray(f_candidate))


def test_the_step_module_names_no_addon():
    import src.addons
    import src.pipeline.step
    from src.registry import get_operator_names

    source = inspect.getsource(src.pipeline.step)
    assert src.addons is not None
    for name in get_operator_names("addon"):
        assert f'"{name}"' not in source


# ── registration and hooks ───────────────────────────────────────────


def test_a_registered_addon_is_merged_in_name_order_and_sees_both_states():
    """A new addon is one registration: no edit to build_setup, the step or the runner."""
    seen: list[tuple[int, int]] = []

    def post_step(setup, prev, new):
        seen.append((int(prev.t), int(new.t)))
        return new._replace(ext=new.ext | {"zz_probe": new.ext["zz_probe"] + 1.0})

    def build(config):
        return Hooks(post_step=post_step, init_state=lambda setup: jnp.array(0.0))

    register_operator("addon", name="zz_probe")(build)
    try:
        kwargs, _ = _CASES["wetting"]
        setup = build_setup(SimulationConfig(**kwargs))
        state = init_state(setup)
        new_state = step(setup, state)
    finally:
        unregister_operator("addon", "zz_probe")

    assert [name for name, _ in setup.addons] == ["wetting", "zz_probe"]
    assert float(new_state.ext["zz_probe"]) == 1.0
    assert seen == [(0, 1)]


# ── observe: the derived fields ──────────────────────────────────────


def test_observed_fields_belong_to_the_state_they_were_observed_on():
    """``observe`` reads one state: its density is ``sum(f)``, its force the force at its own ``t``.

    A ramped (masked) gravity makes the force depend on ``t``, so a field lagging one step
    behind ``f`` — as the stored fields once did — would show here.
    """
    from src.operators.force._force_aggregator import compute_total_force_ext
    from src.pipeline.runner import run

    config = SimulationConfig(forces={"gravity_masked_force": {"force_g": 1e-5, "ramp_steps": 10}}, **_MULTIPHASE)
    setup = build_setup(config)
    state, _ = run(setup, init_state(setup), nt=4)
    observed = observe(setup, state)

    np.testing.assert_array_equal(np.asarray(observed.rho), np.asarray(jnp.sum(state.f, axis=-2, keepdims=True)))
    assert observed.force_ext is not None
    np.testing.assert_array_equal(np.asarray(observed.force_ext), np.asarray(compute_total_force_ext(setup, state)))
    previous = state._replace(t=state.t - 1)
    assert not np.allclose(np.asarray(observed.force_ext), np.asarray(compute_total_force_ext(setup, previous)))


def test_the_trajectory_carries_only_the_state():
    """In-memory mode stacks ``(f, t, ext)``; derived fields are observed from it on demand."""
    from src.pipeline.runner import run

    kwargs, _ = _CASES["wetting"]
    setup = build_setup(SimulationConfig(**kwargs))
    _, trajectory = run(setup, init_state(setup), nt=3)

    assert trajectory is not None
    assert trajectory._fields == ("f", "t", "ext")
    observed = jax.vmap(lambda s: observe(setup, s))(trajectory)
    assert observed.rho.shape[0] == 3
