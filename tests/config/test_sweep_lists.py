"""A list where a parameter sweep used to read its axis is rejected, and names the staging command."""

from __future__ import annotations
import pytest
from src.config import DictAdapter

_MULTIPHASE = {"eos": "double-well", "kappa": 0.01, "rho_l": 1.0, "rho_v": 0.1, "interface_width": 4}


@pytest.mark.parametrize(
    ("fields", "path"),
    [
        ({"tau": [0.7, 0.9]}, "tau"),
        ({"nt": [10, 20]}, "nt"),
        ({"collision_scheme": ["bgk", "mrt"]}, "collision_scheme"),
        ({"grid_shape": [[8, 8], [16, 8]]}, "grid_shape"),
        ({"multiphase": {**_MULTIPHASE, "kappa": [0.01, 0.02]}}, "multiphase.kappa"),
        (
            {"forces": {"gravity_force": {"force_g": 1e-6, "inclination_angle_deg": [20, 40]}}},
            "gravity_force.inclination_angle_deg",
        ),
    ],
    ids=["tau", "nt", "collision", "grid-shape", "section", "force"],
)
def test_a_sweep_list_is_rejected(fields, path):
    raw = {"grid_shape": (8, 8), **fields}
    adapter = DictAdapter()

    with pytest.raises(ValueError, match=f"{path}: a list is not a value.*tud-lbm sweep stage"):
        adapter.load(raw)


def test_a_grid_shape_list_is_a_value():
    config = DictAdapter().load({"grid_shape": [8, 8]})

    assert config.grid_shape == (8, 8, 1)
