"""A chemical-step run and its x-mirror evolve as mirror images.

"Pre" and "post" are the inclusion's start and destination surfaces, so a droplet
sliding toward -x from the high side of a step is the mirror image of one sliding
toward +x from the low side: step, centre and gravity mirrored, and the start side
derived as ``high``. Every rule that once assumed travel toward +x (the window, the
wall split, saturation, the crossing pair) is exercised here, with the leading line
crossing onto the post surface.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax
import numpy as np
import pytest
from src.config.simulation_config import SimulationConfig
from src.pipeline.runner import init_state
from src.pipeline.setup import build_setup
from src.pipeline.step import observe
from src.pipeline.step import step

if TYPE_CHECKING:
    from src.operators.wetting._params import WettingState

_NX, _NY = 64, 24
_CENTRE = 25  # cell index; the leading line starts just short of the step at 32
_STEPS = 2000


def _config(*, mirrored: bool) -> SimulationConfig:
    centre = (_NX - 1 - _CENTRE) if mirrored else _CENTRE
    return SimulationConfig(
        multiphase={"eos": "double-well", "kappa": 0.02, "rho_l": 1.0, "rho_v": 0.1, "interface_width": 3},
        grid_shape=(_NX, _NY),
        tau=0.9,
        nt=_STEPS,
        init_type="multiphase_bubbles",
        initialisation={"centres": [[centre / (_NX - 1), 0.0]], "radii": [0.3], "dispersed": "liquid"},
        boundary_conditions={"bottom": "wetting", "top": "symmetry", "left": "bounce-back", "right": "bounce-back"},
        wetting={"phi_left": 1.0, "phi_right": 1.0, "d_rho_left": 0.0, "d_rho_right": 0.0},
        hysteresis={"max_iterations": 5, "learning_rate": 0.005},
        chemical_step={
            "chemical_step_location": 0.5,
            "ca_advancing_pre_step": 100.0,
            "ca_receding_pre_step": 80.0,
            "ca_advancing_post_step": 70.0,
            "ca_receding_post_step": 50.0,
        },
        # force_x = -g sin(angle): a negative angle drives the droplet toward +x.
        forces={"gravity_force": {"force_g": 1e-4, "inclination_angle_deg": 30.0 if mirrored else -30.0}},
    )


def _final(config: SimulationConfig) -> tuple[np.ndarray, WettingState]:
    setup = build_setup(config)
    state = jax.lax.scan(lambda s, _: (step(setup, s), None), init_state(setup), None, length=_STEPS)[0]
    return np.asarray(observe(setup, state).rho)[:, :, 0, 0, 0], state.ext["wetting"]


@pytest.mark.slow
def test_a_mirrored_run_is_the_mirror_image():
    low, high = _config(mirrored=False), _config(mirrored=True)
    assert low.chemical_step is not None
    assert high.chemical_step is not None
    assert (low.chemical_step["chemical_step_pre_side"], high.chemical_step["chemical_step_pre_side"]) == (
        "low",
        "high",
    )

    rho, wet = _final(low)
    rho_m, wet_m = _final(high)

    assert float(wet.cll_right) > _NX / 2  # the leading line is on the post surface
    np.testing.assert_allclose(rho_m[::-1], rho, atol=1e-8)
    np.testing.assert_allclose(
        [
            float(v)
            for v in (
                wet_m.ca_right,
                wet_m.ca_left,
                wet_m.phi_right,
                wet_m.phi_left,
                wet_m.d_rho_right,
                wet_m.d_rho_left,
            )
        ],
        [float(v) for v in (wet.ca_left, wet.ca_right, wet.phi_left, wet.phi_right, wet.d_rho_left, wet.d_rho_right)],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        [_NX - 1 - float(wet_m.cll_right), _NX - 1 - float(wet_m.cll_left)],
        [float(wet.cll_left), float(wet.cll_right)],
        atol=1e-6,
    )
