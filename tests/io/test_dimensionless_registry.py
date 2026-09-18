"""The ``dimensionless`` registry kind: contract every registered number must meet.

These tests are what make "adding a number is adding a file" true. They are
written against the registry rather than a list of names, so a new module under
``analysis/physical_parameters/numbers/`` is covered the moment it exists.
"""

from __future__ import annotations
import math
import pytest
from src.config import SimulationConfig
from src.registry import get_operators
from src.simulation_io.analysis.physical_parameters import compute_dimensionless_numbers
from src.simulation_io.analysis.physical_parameters import dimensionless_keys
from src.simulation_io.analysis.physical_parameters import dimensionless_label
from src.simulation_io.analysis.physical_parameters import resolve_dimensionless_inputs
from src.simulation_io.analysis.run_labels import LABEL_PARAM_CHOICES
from src.simulation_io.analysis.run_labels import NAME_KEY
from src.simulation_io.analysis.run_labels import resolve_label_keys

_REQUIRED_META = ("label", "row_label", "formula", "order", "needs_gravity")


def _config(**kwargs) -> SimulationConfig:
    base = {
        "sim_type": "multiphase",
        "grid_shape": (40, 20),
        "eos": "double-well",
        "kappa": 0.02,
        "rho_l": 1.0,
        "rho_v": 0.5,
        "interface_width": 2,
        "tau": 0.8,
        "gravity_force": {"force_g": 1e-6, "inclination_angle_deg": 30.0},
        "initialisation": {"radii": [0.25], "centres": [[0.5, 0.5]]},
    }
    base.update(kwargs)
    return SimulationConfig(**base)  # ty: ignore[invalid-argument-type]


def _entries():
    return get_operators("dimensionless").values()


def _meta(entry, key: str) -> object:
    """One metadata value, asserting the entry carries metadata at all."""
    assert entry.metadata is not None, entry.name
    return entry.metadata[key]


def _order(entry) -> int:
    """The entry's display order, as the int the registry contract requires."""
    value = _meta(entry, "order")
    assert isinstance(value, int), entry.name
    return value


# --- registration contract ---------------------------------------------------


@pytest.mark.parametrize("key", _REQUIRED_META)
def test_every_number_declares_its_presentation(key: str):
    """Missing metadata would leave a number unlabelled on an axis or in a row."""
    for entry in _entries():
        assert entry.metadata is not None, entry.name
        assert key in entry.metadata, f"{entry.name} is missing {key!r}"


def test_display_order_is_unique():
    """Ties would make the overview row order depend on module import order."""
    orders = [_order(entry) for entry in _entries()]
    assert len(set(orders)) == len(orders)


def test_keys_are_returned_in_display_order():
    by_order = sorted(_entries(), key=_order)
    assert dimensionless_keys() == tuple(e.name for e in by_order)


def test_every_label_is_mathtext():
    for key in dimensionless_keys():
        label = dimensionless_label(key)
        assert label.startswith("$"), (key, label)
        assert label.endswith("$"), (key, label)


def test_dimensionless_label_rejects_an_unknown_key():
    with pytest.raises(KeyError):
        dimensionless_label("not_a_number")


# --- coupling to the consumers ----------------------------------------------


def test_every_number_is_a_valid_label_param():
    """The CLI's --label-param choices are the registry plus the run name."""
    assert list(LABEL_PARAM_CHOICES) == [*dimensionless_keys(), NAME_KEY]


def test_every_choice_resolves_through_the_label_selector():
    numbers = [compute_dimensionless_numbers(_config())]

    assert resolve_label_keys(numbers, LABEL_PARAM_CHOICES) == list(LABEL_PARAM_CHOICES)


def test_a_resolvable_config_reports_every_registered_number():
    dn = compute_dimensionless_numbers(_config())

    assert set(dn.values) == set(dimensionless_keys())
    assert all(dn.get(key) is not None for key in dimensionless_keys())


# --- the numbers themselves --------------------------------------------------


def test_laplace_is_the_reciprocal_square_of_ohnesorge():
    dn = compute_dimensionless_numbers(_config())

    la, oh = dn.get("la"), dn.get("oh")
    assert la is not None
    assert oh is not None
    assert math.isclose(la, 1.0 / oh**2, rel_tol=1e-12)


_CS_EOS_PARAMS = {"a_eos": 1.0, "b_eos": 4.0, "r_eos": 1.0, "t_eos": 0.07}


def test_no_inputs_without_a_surface_tension():
    """A calibration-only EOS with no measurement resolves nothing at all."""
    cfg = _config(eos="carnahan-starling", **_CS_EOS_PARAMS)

    assert resolve_dimensionless_inputs(cfg) is None
    assert compute_dimensionless_numbers(cfg).values == {}


def test_gravity_is_carried_as_none_rather_than_failing_resolution():
    """Oh and La need no gravity, so a gravity-free run must still resolve inputs."""
    inputs = resolve_dimensionless_inputs(_config(gravity_force=None))

    assert inputs is not None
    assert inputs.g is None
    assert inputs.angle_deg is None


def test_gravity_driven_numbers_are_exactly_those_marked_needs_gravity():
    """The metadata flag and the operator's own gate must agree."""
    dn = compute_dimensionless_numbers(_config(gravity_force=None))

    unresolved = {key for key in dimensionless_keys() if dn.get(key) is None}
    marked = {entry.name for entry in _entries() if _meta(entry, "needs_gravity")}
    assert unresolved == marked
