from __future__ import annotations
import jax.numpy as jnp
import pytest
from tud_lbm.operators.wetting._params import WettingParams
from tud_lbm.operators.wetting.hysteresis.hysteresis import _clamp_params
from tud_lbm.operators.wetting.hysteresis.hysteresis import _import_optax
from tud_lbm.operators.wetting.hysteresis.hysteresis import _phi_is_active


def test_phi_is_active_truth_table():
    # Above-window CA always activates phi.
    assert bool(_phi_is_active(jnp.array(False), jnp.array(True), jnp.array(True)))

    # In-window + backward drift activates phi.
    assert bool(_phi_is_active(jnp.array(True), jnp.array(False), jnp.array(False)))

    # In-window + forward drift keeps d_rho active.
    assert not bool(_phi_is_active(jnp.array(True), jnp.array(False), jnp.array(True)))


def test_import_optax_raises_helpful_message_when_missing(mock_optax_missing):
    with pytest.raises(ImportError, match="required for hysteresis wetting"):
        _import_optax()


def test_clamp_params_limits_ranges():
    raw = WettingParams(
        phi_left=jnp.array(0.1),
        phi_right=jnp.array(9.0),
        d_rho_left=jnp.array(-1.0),
        d_rho_right=jnp.array(1.0),
    )

    out = _clamp_params(raw)

    assert float(out.phi_left) == pytest.approx(1.0)
    assert float(out.phi_right) == pytest.approx(1.5)
    assert float(out.d_rho_left) == pytest.approx(0.0)
    assert float(out.d_rho_right) == pytest.approx(0.25)
