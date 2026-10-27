"""Extra branch coverage for callbacks helpers."""

from __future__ import annotations
from typing import Any
import jax.numpy as jnp
import numpy as np
import pytest
from src.pipeline.state import State
from src.pipeline.state import WettingState
from src.pipeline.step import Observables


def _state_with_wetting() -> State:
    wet = WettingState(*(jnp.array(v, dtype=jnp.float64) for v in (1, 1, 0, 0, 90, 90, 3, 9)))
    return State(f=jnp.ones((2, 2, 1, 9, 1)), t=jnp.array(2), ext={"wetting": wet})


def _observables(**fields: jnp.ndarray | None) -> Observables:
    """Derived fields as ``observe`` returns them; single-phase without forces by default."""
    base: dict[str, Any] = {
        "rho": jnp.ones((2, 2, 1, 1, 1)),
        "u": jnp.zeros((2, 2, 1, 1, 2)),
        "force": None,
        "force_ext": None,
        "pressure": jnp.full((2, 2, 1, 1, 1), 1.0 / 3.0),
    }
    merged: dict[str, Any] = base | fields
    return Observables(**merged)


def test_state_to_numpy_writes_the_derived_fields_under_their_snapshot_keys():
    """A snapshot carries the carry plus ``observe``'s fields, as it always has; ``None`` is omitted."""
    from src.simulation_io import callbacks

    data = callbacks._state_to_numpy(_state_with_wetting(), _observables(), t=7)

    assert {"f", "t", "rho", "u", "pressure"} <= set(data)
    assert "force" not in data
    assert "force_ext" not in data


def test_state_to_numpy_persists_wetting_even_with_field_filter():
    from src.simulation_io import callbacks

    data = callbacks._state_to_numpy(_state_with_wetting(), _observables(), fields=("rho",), t=7)

    assert "rho" in data
    assert "ca_left" in data
    assert "cll_right" in data


def test_state_to_numpy_flattens_field_substates_under_their_own_keys():
    """A field-sized substate entry (the electric ``h``) keeps its snapshot key and follows ``fields``."""
    from src.simulation_io import callbacks

    h = jnp.full((2, 2, 1, 9, 1), 0.5)
    state = _state_with_wetting()._replace(ext={"electric": {"h": h}})
    observables = _observables()

    np.testing.assert_array_equal(callbacks._state_to_numpy(state, observables, t=7)["h"], np.asarray(h))
    assert "h" in callbacks._state_to_numpy(state, observables, fields=("h",), t=7)
    assert "h" not in callbacks._state_to_numpy(state, observables, fields=("rho",), t=7)
    assert "electric" not in callbacks._state_to_numpy(state, observables, t=7)


def test_state_to_numpy_saves_pressure_unless_filtered_out():
    from src.simulation_io import callbacks

    observables = _observables()
    state = _state_with_wetting()

    data = callbacks._state_to_numpy(state, observables, t=7)
    np.testing.assert_array_equal(data["pressure"], np.asarray(observables.pressure))
    assert "pressure" not in callbacks._state_to_numpy(state, observables, fields=("rho",), t=7)


def test_state_to_numpy_raises_on_nan():
    from src.simulation_io import callbacks

    state = _state_with_wetting()
    observables = _observables(rho=jnp.array([[[[[np.nan]]]]]))

    with pytest.raises(FloatingPointError, match=r"NaNs detected at t=7.*rho"):
        callbacks._state_to_numpy(state, observables, fields=("rho",), t=7)


def test_make_save_callback_runs_only_on_matching_interval(monkeypatch):
    from src.simulation_io import callbacks

    saved: list[tuple[int, dict]] = []
    observed: list[int] = []

    class _IO:
        def save_data_step(self, step: int, data: dict) -> None:
            saved.append((step, data))

    def observe_fn(state: State) -> Observables:
        observed.append(int(state.t))
        return _observables()

    monkeypatch.setattr(
        callbacks.jax.lax,
        "cond",
        lambda pred, on_true, on_false, state, t: on_true(state, t) if pred else on_false(state, t),
    )
    monkeypatch.setattr(callbacks.jax.debug, "callback", lambda fn, *args, **_kwargs: fn(*args))

    do_save = callbacks.make_save_callback(_IO(), observe_fn, save_interval=2, skip_interval=1, save_fields=("rho",))  # ty: ignore[invalid-argument-type]
    state = _state_with_wetting()

    do_save(state, 1)
    do_save(state, 2)
    do_save(state, 4)

    assert [step for step, _ in saved] == [2, 4]
    assert all("rho" in payload for _, payload in saved)
    assert len(observed) == 2  # the fields are derived on save steps only
