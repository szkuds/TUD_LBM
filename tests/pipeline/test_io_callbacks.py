"""Extra branch coverage for io_callbacks helpers."""

from __future__ import annotations
import jax.numpy as jnp
import numpy as np
import pytest
from tud_lbm.pipeline.state.state import State
from tud_lbm.pipeline.state.state import WettingState


def _state_with_wetting() -> State:
    wet = WettingState(*(jnp.array(v, dtype=jnp.float32) for v in (1, 1, 0, 0, 90, 90, 3, 9)))
    return State(
        f=jnp.ones((2, 2, 1, 9, 1)),
        rho=jnp.ones((2, 2, 1, 1, 1)),
        u=jnp.zeros((2, 2, 1, 1, 2)),
        t=jnp.array(2),
        wetting=wet,
    )


def test_state_to_numpy_persists_wetting_even_with_field_filter():
    from tud_lbm.pipeline import io_callbacks

    data = io_callbacks._state_to_numpy(_state_with_wetting(), fields=("rho",), t=7)

    assert "rho" in data
    assert "ca_left" in data
    assert "cll_right" in data


def test_state_to_numpy_raises_on_nan():
    from tud_lbm.pipeline import io_callbacks

    state = _state_with_wetting()._replace(rho=jnp.array([[[[[np.nan]]]]]))

    with pytest.raises(FloatingPointError, match=r"NaNs detected at t=7.*rho"):
        io_callbacks._state_to_numpy(state, fields=("rho",), t=7)


def test_make_save_callback_runs_only_on_matching_interval(monkeypatch):
    from tud_lbm.pipeline import io_callbacks

    saved: list[tuple[int, dict]] = []

    class _IO:
        def save_data_step(self, step: int, data: dict) -> None:
            saved.append((step, data))

    monkeypatch.setattr(
        io_callbacks.jax.lax,
        "cond",
        lambda pred, on_true, on_false, state, t: on_true(state, t) if pred else on_false(state, t),
    )
    monkeypatch.setattr(io_callbacks.jax.debug, "callback", lambda fn, state, t, ordered=True: fn(state, t))

    do_save = io_callbacks.make_save_callback(_IO(), save_interval=2, skip_interval=1, save_fields=("rho",))  # ty: ignore[invalid-argument-type]
    state = _state_with_wetting()

    do_save(state, 1)
    do_save(state, 2)
    do_save(state, 4)

    assert [step for step, _ in saved] == [2, 4]
    assert all("rho" in payload for _, payload in saved)
