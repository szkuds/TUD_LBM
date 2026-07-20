"""Guard the ``analysis`` operator kind registration.

The kind was renamed from ``comparison``: every registered operator is a
per-run snapshot-history plot, whereas true cross-run comparison lives in
:mod:`src.simulation_io.plotting.run_comparison` and is not an operator at all.

A missed string literal during that rename would silently drop an operator
from the registry rather than raising, so these tests pin the full set.
"""

from __future__ import annotations
import src.simulation_io.plotting  # noqa: F401  (import registers the operators)
from src.registry import get_operator_category
from src.registry import get_operator_names

_EXPECTED_ANALYSIS_OPERATORS = frozenset(
    {
        "avg_density",
        "ca_theta_vs_time",
        "ca_theta_vs_x",
        "contact_angle_left",
        "contact_angle_right",
        "contact_angles_pair",
        "contact_line_speed_left",
        "contact_line_speed_right",
        "contact_line_speeds_pair",
        "density_ratio",
        "max_velocity",
        "simulation_csv",
        "snapshot_fig",
        "total_mass",
    }
)

_EXPECTED_PLOTTING_OPERATORS = frozenset({"density", "velocity", "force", "force_ext"})


def test_analysis_kind_holds_every_expected_operator() -> None:
    """All history-plot operators register under the ``analysis`` kind."""
    assert set(get_operator_names("analysis")) == _EXPECTED_ANALYSIS_OPERATORS


def test_plotting_kind_holds_every_expected_operator() -> None:
    """Field-plot operators stay under the ``plotting`` kind."""
    assert set(get_operator_names("plotting")) == _EXPECTED_PLOTTING_OPERATORS


def test_comparison_kind_no_longer_exists() -> None:
    """The old kind name is fully removed, with no compatibility alias."""
    assert "comparison" not in get_operator_category()


def test_analysis_and_plotting_kinds_are_disjoint() -> None:
    """No operator registers under both visual kinds."""
    assert not _EXPECTED_ANALYSIS_OPERATORS & _EXPECTED_PLOTTING_OPERATORS
