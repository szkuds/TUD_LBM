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
        "interface_evolution_config",
        "interface_evolution_measured",
        "max_velocity",
        "simulation_csv",
        "snapshot_fig",
        "total_mass",
    }
)

_EXPECTED_PLOTTING_OPERATORS = frozenset(
    {"contact_angle", "density", "velocity", "force", "force_ext", "interface", "pressure", "pressure_total"}
)

#: Registered, but excluded from the default figure — they only render when
#: named in ``plot_fields``. See ``PlotOperator.opt_in``.
_EXPECTED_OPT_IN_OPERATORS = frozenset({"contact_angle", "interface", "pressure", "pressure_total"})

#: May be named in ``overlay_fields``. See ``PlotOperator.supports_overlay``.
_EXPECTED_OVERLAY_OPERATORS = frozenset({"contact_angle", "interface"})


def test_analysis_kind_holds_every_expected_operator() -> None:
    """All history-plot operators register under the ``analysis`` kind."""
    assert set(get_operator_names("analysis")) == _EXPECTED_ANALYSIS_OPERATORS


def test_plotting_kind_holds_every_expected_operator() -> None:
    """Field-plot operators stay under the ``plotting`` kind."""
    assert set(get_operator_names("plotting")) == _EXPECTED_PLOTTING_OPERATORS


def test_opt_in_flag_matches_the_expected_plotting_operators() -> None:
    """Exactly the pressure, interface and contact-angle operators are opt-in; every other field plot is default-on."""
    from src.registry import get_operators

    opt_in = {name for name, entry in get_operators("plotting").items() if getattr(entry.target, "opt_in", False)}
    assert opt_in == _EXPECTED_OPT_IN_OPERATORS


def test_overlay_capability_matches_the_expected_plotting_operators() -> None:
    """Overlays are a flag on plotting operators, not a registry kind of their own."""
    from src.registry import get_operators

    capable = {
        name for name, entry in get_operators("plotting").items() if getattr(entry.target, "supports_overlay", False)
    }
    assert capable == _EXPECTED_OVERLAY_OPERATORS


def test_comparison_kind_no_longer_exists() -> None:
    """The old kind name is fully removed, with no compatibility alias."""
    assert "comparison" not in get_operator_category()


def test_analysis_and_plotting_kinds_are_disjoint() -> None:
    """No operator registers under both visual kinds."""
    assert not _EXPECTED_ANALYSIS_OPERATORS & _EXPECTED_PLOTTING_OPERATORS
