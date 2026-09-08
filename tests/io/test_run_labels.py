"""Legend-label selection for cross-run comparison plots."""

from __future__ import annotations
from src.simulation_io.analysis.physical_parameters import DimensionlessNumbers
from src.simulation_io.plotting.run_labels import LABEL_PARAM_CHOICES
from src.simulation_io.plotting.run_labels import build_run_labels
from src.simulation_io.plotting.run_labels import label_sort_key
from src.simulation_io.plotting.run_labels import resolve_label_keys


def _dn(**kwargs) -> DimensionlessNumbers:
    base = {
        "oh": 0.3,
        "bo": 1.0,
        "bo_perp": 0.87,
        "bo_parallel": 0.5,
        "ar": 4.0,
        "re": 2.0,
        "inclination_deg": 0.0,
    }
    base.update(kwargs)
    return DimensionlessNumbers(**base)


def _labels(numbers, names=None, requested=None) -> list[str]:
    names = names or [f"run {i}" for i in range(len(numbers))]
    return build_run_labels(names, numbers, resolve_label_keys(numbers, requested))


# --- automatic selection -----------------------------------------------------


def test_constant_quantity_is_not_selected():
    numbers = [_dn(), _dn()]

    assert resolve_label_keys(numbers) == []


def test_varying_quantity_is_selected_and_rendered():
    numbers = [_dn(bo=1.0), _dn(bo=2.0)]

    assert resolve_label_keys(numbers) == ["bo"]
    assert _labels(numbers) == [r"$\mathrm{Bo} = 1.00$", r"$\mathrm{Bo} = 2.00$"]


def test_bo_parallel_replaces_bo_when_runs_are_inclined():
    numbers = [
        _dn(bo=1.0, bo_parallel=0.5, inclination_deg=30.0),
        _dn(bo=2.0, bo_parallel=1.0, inclination_deg=30.0),
    ]

    assert resolve_label_keys(numbers) == ["bo_parallel"]


def test_bo_is_used_when_no_run_is_inclined():
    numbers = [_dn(bo=1.0, bo_parallel=0.0), _dn(bo=2.0, bo_parallel=0.0)]

    assert resolve_label_keys(numbers) == ["bo"]


def test_bo_perp_and_ar_are_never_selected_automatically():
    numbers = [_dn(bo_perp=0.1, ar=1.0, re=2.0), _dn(bo_perp=0.2, ar=2.0, re=2.0)]

    assert resolve_label_keys(numbers) == []


def test_measurement_noise_is_not_a_difference():
    """Values a few tenths of a percent apart come from the init field, not the sweep."""
    numbers = [_dn(bo=0.263), _dn(bo=0.264)]

    assert resolve_label_keys(numbers) == []


def test_a_real_sweep_axis_clears_the_spread_threshold():
    numbers = [_dn(bo=0.10), _dn(bo=0.30)]

    assert resolve_label_keys(numbers) == ["bo"]


def test_all_zero_quantity_is_not_selected():
    numbers = [_dn(bo=0.0), _dn(bo=0.0)]

    assert resolve_label_keys(numbers) == []


def test_unresolvable_run_counts_as_differing():
    numbers = [_dn(bo=1.0), _dn(bo=None)]

    assert resolve_label_keys(numbers) == ["bo"]
    assert _labels(numbers) == [r"$\mathrm{Bo} = 1.00$", "run 1"]


def test_several_differing_representatives_are_ordered_by_precedence():
    numbers = [_dn(bo=1.0, oh=0.3, re=2.0), _dn(bo=2.0, oh=0.6, re=4.0)]

    assert resolve_label_keys(numbers) == ["bo", "oh", "re"]


# --- explicit override -------------------------------------------------------


def test_explicit_keys_are_rendered_even_when_constant():
    numbers = [_dn(), _dn()]

    # Constant across the runs, so every label collides and is disambiguated.
    assert _labels(numbers, names=["a", "b"], requested=["ar"]) == [
        r"$\mathrm{Ar} = 4.00$ (a)",
        r"$\mathrm{Ar} = 4.00$ (b)",
    ]


def test_explicit_keys_keep_the_requested_order():
    numbers = [_dn(bo=1.0), _dn(bo=2.0)]

    assert resolve_label_keys(numbers, ["oh", "bo"]) == ["oh", "bo"]


def test_name_key_restores_the_run_name():
    numbers = [_dn(bo=1.0), _dn(bo=2.0)]

    assert _labels(numbers, names=["left", "right"], requested=["name"]) == ["left", "right"]


def test_unknown_key_is_ignored():
    assert resolve_label_keys([_dn()], ["nonsense"]) == []


def test_every_choice_is_resolvable():
    numbers = [_dn()]

    assert resolve_label_keys(numbers, LABEL_PARAM_CHOICES) == list(LABEL_PARAM_CHOICES)


# --- fallbacks and uniqueness ------------------------------------------------


def test_all_none_numbers_fall_back_to_names():
    numbers = [DimensionlessNumbers(oh=None, bo=None, bo_perp=None, bo_parallel=None)] * 2

    assert resolve_label_keys(numbers) == []
    assert _labels(numbers, names=["a", "b"]) == ["a", "b"]


def test_duplicate_labels_are_disambiguated_by_name():
    numbers = [_dn(bo=1.0), _dn(bo=1.0)]

    labels = _labels(numbers, names=["a", "b"], requested=["bo"])

    assert labels == [r"$\mathrm{Bo} = 1.00$ (a)", r"$\mathrm{Bo} = 1.00$ (b)"]


def test_name_fallback_labels_are_not_suffixed_with_themselves():
    numbers = [_dn(), _dn()]

    assert _labels(numbers, names=["a", "b"]) == ["a", "b"]


# --- ordering ----------------------------------------------------------------


def test_label_sort_key_orders_by_the_labelled_quantities():
    numbers = [_dn(bo=2.0), _dn(bo=1.0)]

    ordered = sorted(numbers, key=lambda dn: label_sort_key(dn, ["bo"]))

    assert [dn.bo for dn in ordered] == [1.0, 2.0]


def test_label_sort_key_sorts_unresolvable_runs_last():
    numbers = [_dn(bo=None), _dn(bo=1.0)]

    ordered = sorted(numbers, key=lambda dn: label_sort_key(dn, ["bo"]))

    assert [dn.bo for dn in ordered] == [1.0, None]


def test_label_sort_key_ignores_the_name_key():
    assert label_sort_key(_dn(bo=1.0), ["name", "bo"]) == (1.0,)
