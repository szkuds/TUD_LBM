"""Tests for plugin-driven extra-state orchestration."""

import warnings
from types import SimpleNamespace
import jax.numpy as jnp
import pytest
from state._extra_state import _build_extra_state
from state._extra_state import _update_extra_state
from state.state import State


class _PluginA:
    name = "a"

    @staticmethod
    def init_state(setup):
        return {"dup": jnp.array(1.0)}

    @staticmethod
    def update_state(setup, prev_state, new_state, **context):
        return new_state


class _PluginB:
    name = "b"

    @staticmethod
    def init_state(setup):
        return {"dup": jnp.array(2.0)}

    @staticmethod
    def update_state(setup, prev_state, new_state, **context):
        return new_state


class _TickPlugin:
    name = "tick"

    @staticmethod
    def init_state(setup):
        return {}

    @staticmethod
    def update_state(setup, prev_state, new_state, **context):
        return new_state._replace(t=new_state.t + 1)


def _state_template(t: int = 0) -> State:
    return State(
        f=jnp.zeros((2, 2, 9, 1)),
        rho=jnp.ones((2, 2, 1, 1)),
        u=jnp.zeros((2, 2, 1, 2)),
        t=jnp.array(t),
    )


def test_duplicate_key_warns_once_per_process():
    setup = SimpleNamespace(extra_state_plugins=(_PluginA(), _PluginB()))

    with pytest.warns(RuntimeWarning, match="multiple plugins"):
        extra = _build_extra_state(setup)
    assert float(extra["dup"]) == 1.0

    # Warn-once policy: second call should not emit the same warning.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _build_extra_state(setup)
    assert len(record) == 0


def test_update_pipeline_applies_plugins_in_order():
    setup = SimpleNamespace(extra_state_plugins=(_TickPlugin(),))
    prev_state = _state_template(t=0)
    new_state = _state_template(t=1)

    updated = _update_extra_state(setup, prev_state, new_state)
    assert int(updated.t) == 2
