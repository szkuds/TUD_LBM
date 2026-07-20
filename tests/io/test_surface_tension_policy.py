"""Which surface tension the droplet metrics normalise with.

A calibrated (measured) value is preferred over the closed-form one, and both
are carried through to the CSV so the two can be compared.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import pandas as pd
import pytest
from src.simulation_io.analysis.droplet_metrics import analytical_sigma_lg
from src.simulation_io.analysis.droplet_metrics import droplet_series_for_run
from src.simulation_io.analysis.droplet_metrics import measured_sigma_lg
from src.simulation_io.analysis.droplet_metrics import resolve_scales
from src.simulation_io.plotting.simulation_csv import build_simulation_csv
from tests.support.run_dirs import build_run_dir
from tests.support.run_dirs import wetting_config

if TYPE_CHECKING:
    from pathlib import Path

_MEASURED = 0.011


def _calibrated_config():
    return wetting_config(extra={"surface_tension": _MEASURED})


def test_measured_sigma_read_from_config_extra():
    assert measured_sigma_lg(_calibrated_config()) == pytest.approx(_MEASURED)


def test_measured_sigma_absent_without_calibration():
    assert measured_sigma_lg(wetting_config()) is None


def test_primary_prefers_measured_over_analytical():
    scales = resolve_scales(_calibrated_config())

    assert scales is not None
    assert scales.sigma_primary == pytest.approx(_MEASURED)
    assert scales.sigma_source == "measured"
    # The closed-form value is still carried, for comparison.
    assert scales.sigma_analytical == pytest.approx(analytical_sigma_lg(_calibrated_config()))


def test_primary_falls_back_to_analytical_without_calibration():
    scales = resolve_scales(wetting_config())

    assert scales is not None
    assert scales.sigma_source == "analytical"
    assert scales.sigma_measured is None
    assert scales.sigma_primary == pytest.approx(scales.sigma_analytical)


def test_calibrated_run_normalises_capillary_numbers_by_measured_sigma(tmp_path: Path):
    """This is the deliberate numerical change: Ca shifts onto measured sigma."""
    config = _calibrated_config()
    run_dir = build_run_dir(tmp_path, config=config)

    series = droplet_series_for_run(run_dir, config)

    assert series is not None
    assert series.scales.sigma_analytical is not None
    ratio = series.scales.sigma_analytical / _MEASURED
    # Ca and Ca_analytical differ by exactly the ratio of the two sigmas.
    assert series.ca == pytest.approx(series.ca_analytical * ratio)


def test_csv_records_which_sigma_was_used(tmp_path: Path):
    config = _calibrated_config()
    run_dir = build_run_dir(tmp_path, config=config)

    out = build_simulation_csv(run_dir, config)

    assert out is not None
    df = pd.read_csv(out)
    assert (df["sigma_lg_source"] == "measured").all()
    assert df["sigma_lg"].eq(_MEASURED).all()


def test_uncalibrated_csv_reports_analytical_source(tmp_path: Path):
    config = wetting_config()
    run_dir = build_run_dir(tmp_path, config=config)

    out = build_simulation_csv(run_dir, config)

    assert out is not None
    df = pd.read_csv(out)
    assert (df["sigma_lg_source"] == "analytical").all()
    # With no measured value the two Ca columns coincide.
    assert df["Ca"].to_numpy() == pytest.approx(df["Ca_analytical"].to_numpy())
