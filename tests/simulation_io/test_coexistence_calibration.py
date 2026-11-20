"""``calibrate_coexistence`` writes the Maxwell pair into a run directory, for a CS fluid only."""

from __future__ import annotations
import json
import pytest
from src.config.run_config import DATA_DIRNAME
from src.config.run_config import PLOTS_DIRNAME
from src.simulation_io.analysis.coexistence import calibrate_coexistence
from src.simulation_io.analysis.coexistence import coexistence_dir
from src.simulation_io.readers import DictAdapter

_CS = {
    "eos": "carnahan-starling",
    "kappa": 0.015,
    "rho_l": 12.18,
    "rho_v": 0.015,
    "interface_width": 4,
    "a_eos": 0.00031459670905604266,
    "b_eos": 0.1490857142857143,
    "r_eos": 1.0,
    "t_eos": 0.00039808421247983624,
}


def _config(multiphase: dict | None):
    raw: dict = {"grid_shape": (16, 16), "nt": 10}
    if multiphase is not None:
        raw["multiphase"] = multiphase
    return DictAdapter().load(raw)


def test_carnahan_starling_writes_the_pair_and_the_figure(tmp_path):
    pair = calibrate_coexistence(_config(_CS), tmp_path)

    assert pair == pytest.approx((0.01680445791, 12.18327187), rel=1e-6)
    out = coexistence_dir(tmp_path)
    data = json.loads((out / DATA_DIRNAME / "data.json").read_text(encoding="utf-8"))
    assert data["eos"] == "carnahan-starling"
    assert (data["rho_v"], data["rho_l"]) == pair
    assert data["p_sat"] == pytest.approx(6.61753622e-6, rel=1e-6)
    assert (data["rho_v_configured"], data["rho_l_configured"]) == (0.015, 12.18)
    assert (out / PLOTS_DIRNAME / "maxwell.png").stat().st_size > 0


@pytest.mark.parametrize(
    "multiphase",
    [None, {"eos": "double-well", "kappa": 0.015, "rho_l": 1.0, "rho_v": 0.1, "interface_width": 4}],
    ids=["single-phase", "double-well"],
)
def test_no_solver_returns_none_and_writes_nothing(tmp_path, multiphase):
    assert calibrate_coexistence(_config(multiphase), tmp_path) is None
    assert not coexistence_dir(tmp_path).exists()
