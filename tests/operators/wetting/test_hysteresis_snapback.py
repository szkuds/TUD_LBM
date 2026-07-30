from __future__ import annotations
import jax.numpy as jnp
import pytest
from src.operators.wetting._params import WettingParams
from src.operators.wetting.hysteresis.hysteresis import _clamp_params
from src.operators.wetting.hysteresis.hysteresis import _import_optax
from src.operators.wetting.hysteresis.hysteresis import _phi_is_active

_DROPLET = jnp.array(False)


def test_phi_is_active_truth_table():
    """Droplet-topology spot checks; the exhaustive table with both topologies
    lives in ``test_hysteresis_internals.py``.
    """
    # Above-window CA activates phi — for a droplet, where the reported angle
    # is the liquid angle and phi lowers it.
    assert bool(_phi_is_active(jnp.array(False), jnp.array(True), jnp.array(True), _DROPLET))

    # In-window + backward drift activates phi.
    assert bool(_phi_is_active(jnp.array(True), jnp.array(False), jnp.array(False), _DROPLET))

    # In-window + forward drift keeps d_rho active.
    assert not bool(_phi_is_active(jnp.array(True), jnp.array(False), jnp.array(True), _DROPLET))


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

    out = _clamp_params(raw, jnp.array(5.0))

    assert float(out.phi_left) == pytest.approx(1.0)
    assert float(out.phi_right) == pytest.approx(1.5)
    assert float(out.d_rho_left) == pytest.approx(0.0)
    assert float(out.d_rho_right) == pytest.approx(0.3)
