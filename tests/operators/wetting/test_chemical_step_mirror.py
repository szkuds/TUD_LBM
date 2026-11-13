"""A chemical-step run and its x-mirror evolve as mirror images.

"Pre" and "post" are the inclusion's start and destination surfaces, so a droplet
sliding toward -x from the high side of a step is the mirror image of one sliding
toward +x from the low side: step, centre and gravity mirrored, and the start side
derived as ``high``. Every rule that once assumed travel toward +x (the window, the
wall split, saturation, the crossing pair) is exercised here, with the leading line
crossing onto the post surface.

The image is exact while the lines are pinned or far from a bound (150 steps: the
leading line is on the post surface by then). A line held at its bound is iterated
only when its loss is above ``loss_tol``, and that on/off choice turns the ~1e-9
difference between optimising the left line first and the right line first into a
bounded jitter of the wall parameters. After 2000 steps the two runs agree to 2e-3
in density, 0.04° and 3e-4 cells, so the long run is held to the physical image, not
to round-off. An orientation error is not subtle on that scale: the one this test
found lost a contact line.
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
_EXACT_STEPS = 150


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


def _final(config: SimulationConfig, steps: int) -> tuple[np.ndarray, WettingState]:
    setup = build_setup(config)
    state = jax.lax.scan(lambda s, _: (step(setup, s), None), init_state(setup), None, length=steps)[0]
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

    rho, wet = _final(low, _EXACT_STEPS)
    rho_m, wet_m = _final(high, _EXACT_STEPS)

    assert float(wet.cll_right) > _NX / 2  # the leading line is on the post surface
    np.testing.assert_allclose(rho_m[::-1], rho, atol=1e-8)
    np.testing.assert_allclose(_mirrored(wet_m, ("ca", "phi", "d_rho")), _sides(wet, ("ca", "phi", "d_rho")), atol=1e-6)
    np.testing.assert_allclose(_mirrored_lines(wet_m), _sides(wet, ("cll",)), atol=1e-6)

    rho, wet = _final(low, _STEPS)
    rho_m, wet_m = _final(high, _STEPS)

    np.testing.assert_allclose(rho_m[::-1], rho, atol=1e-2)
    np.testing.assert_allclose(_mirrored(wet_m, ("ca",)), _sides(wet, ("ca",)), atol=0.5)
    np.testing.assert_allclose(_mirrored_lines(wet_m), _sides(wet, ("cll",)), atol=0.05)
    assert (int(wet_m.latch_right), int(wet_m.latch_left)) == (int(wet.latch_left), int(wet.latch_right))


def _sides(wetting: WettingState, names: tuple[str, ...]) -> list[float]:
    return [float(getattr(wetting, f"{name}_{side}")) for name in names for side in ("left", "right")]


def _mirrored(wetting: WettingState, names: tuple[str, ...]) -> list[float]:
    """The per-line values of a mirrored run, in the order of the run it mirrors."""
    return [float(getattr(wetting, f"{name}_{side}")) for name in names for side in ("right", "left")]


def _mirrored_lines(wetting: WettingState) -> list[float]:
    return [_NX - 1 - float(wetting.cll_right), _NX - 1 - float(wetting.cll_left)]
