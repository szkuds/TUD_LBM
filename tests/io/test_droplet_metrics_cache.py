"""The shared metric layer must read each snapshot once, not once per consumer.

Before the shared layer existed, a run with N selected analysis operators read
every ``.npz`` file N+1 times: once for the CSV export and once inside each
operator's own ``compute``. These tests pin the caching that removed that.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import pytest
from tests.support.run_dirs import build_run_dir
from tests.support.run_dirs import wetting_config
from tud_lbm.io.analysis.droplet_metrics import clear_series_cache
from tud_lbm.io.analysis.droplet_metrics import droplet_series_for_run
from tud_lbm.io.analysis.droplet_metrics import series_for_files

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clear_cache():
    """Never let a cached series leak between tests."""
    clear_series_cache()
    yield
    clear_series_cache()


@pytest.fixture
def count_loads(monkeypatch) -> dict[str, int]:
    """Count how many times ``np.load`` is called by the metric layer."""
    calls = {"n": 0}
    real_load = np.load

    def counting_load(*args, **kwargs):
        calls["n"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr("tud_lbm.io.analysis.droplet_metrics.series.np.load", counting_load)
    return calls


def test_repeated_calls_read_snapshots_only_once(tmp_path: Path, count_loads) -> None:
    """A second request for the same run is served from cache."""
    run_dir = build_run_dir(tmp_path)
    config = wetting_config()

    first = droplet_series_for_run(run_dir, config)
    loads_after_first = count_loads["n"]
    second = droplet_series_for_run(run_dir, config)

    assert first is not None
    assert second is first
    assert loads_after_first > 0
    assert count_loads["n"] == loads_after_first


def test_each_snapshot_is_read_exactly_once(tmp_path: Path, count_loads) -> None:
    """One ``np.load`` per snapshot file, no more."""
    iterations = (5, 10, 15, 20)
    run_dir = build_run_dir(tmp_path, iterations=iterations)

    droplet_series_for_run(run_dir, wetting_config())

    assert count_loads["n"] == len(iterations)


def test_two_analysis_operators_share_one_read(tmp_path: Path, count_loads) -> None:
    """Both Ca/theta operators over the same files read each snapshot once total.

    This is the duplication the shared layer exists to remove: previously each
    operator ran its own loop over every ``.npz``.
    """
    from tud_lbm.io.plotting.ca_theta_plot import CaThetaVsTimePlot
    from tud_lbm.io.plotting.ca_theta_plot import CaThetaVsXPlot

    iterations = (5, 10, 15, 20)
    run_dir = build_run_dir(tmp_path, iterations=iterations)
    files = sorted((run_dir / "data").glob("timestep_*.npz"))
    config = wetting_config()

    CaThetaVsTimePlot(config=config).compute(files)
    CaThetaVsXPlot(config=config).compute(files)

    assert count_loads["n"] == len(iterations)


def test_ca_theta_operators_agree_on_shared_columns(tmp_path: Path) -> None:
    """The two operators differ only in which x-axis they render."""
    from tud_lbm.io.plotting.ca_theta_plot import CaThetaVsTimePlot
    from tud_lbm.io.plotting.ca_theta_plot import CaThetaVsXPlot

    run_dir = build_run_dir(tmp_path)
    files = sorted((run_dir / "data").glob("timestep_*.npz"))
    config = wetting_config()

    by_time = CaThetaVsTimePlot(config=config).compute(files)
    by_x = CaThetaVsXPlot(config=config).compute(files)

    for key in ("theta_trailing", "theta_leading", "ca_trailing", "ca_leading", "timesteps"):
        np.testing.assert_allclose(by_time[key], by_x[key])


def test_ca_theta_arrays_match_the_csv_columns(tmp_path: Path) -> None:
    """The plot adapter and the CSV serialiser cannot drift apart."""
    import pandas as pd
    from tud_lbm.io.plotting.ca_theta_plot import CaThetaVsXPlot
    from tud_lbm.io.plotting.simulation_csv import build_simulation_csv

    config = wetting_config()
    run_dir = build_run_dir(tmp_path, config=config)
    files = sorted((run_dir / "data").glob("timestep_*.npz"))

    arrays = CaThetaVsXPlot(config=config).compute(files)
    csv_path = build_simulation_csv(run_dir, config)

    assert csv_path is not None
    df = pd.read_csv(csv_path)
    # ca_left/ca_right are ANGLES in the CSV; Ca_cll_* are the capillary numbers.
    np.testing.assert_allclose(arrays["theta_trailing"], df["ca_left"].to_numpy())
    np.testing.assert_allclose(arrays["theta_leading"], df["ca_right"].to_numpy())
    np.testing.assert_allclose(arrays["ca_trailing"], df["Ca_cll_left"].to_numpy())
    np.testing.assert_allclose(arrays["ca_leading"], df["Ca_cll_right"].to_numpy())
    np.testing.assert_allclose(arrays["x_pos"], df["avg_x_location_norm"].to_numpy())


def test_changed_config_is_not_served_from_cache(tmp_path: Path) -> None:
    """A config change that affects scaling produces a distinct series."""
    run_dir = build_run_dir(tmp_path)

    base = droplet_series_for_run(run_dir, wetting_config())
    altered = droplet_series_for_run(run_dir, wetting_config(rho_l=1.5))

    assert base is not None
    assert altered is not None
    assert altered is not base
    assert altered.scales.rho_mean != base.scales.rho_mean


def test_cache_is_bounded(tmp_path: Path) -> None:
    """The cache evicts rather than growing without limit."""
    from tud_lbm.io.analysis.droplet_metrics import series as series_mod

    config = wetting_config()
    for i in range(series_mod._MAX_CACHED_RUNS + 4):
        run = build_run_dir(tmp_path / f"r{i}", iterations=(5, 10))
        droplet_series_for_run(run, config)

    assert len(series_mod._CACHE) <= series_mod._MAX_CACHED_RUNS


def test_series_for_files_returns_none_without_usable_snapshots(tmp_path: Path) -> None:
    """No parseable snapshots is a capability failure, not an error."""
    assert series_for_files([], wetting_config()) is None


def test_series_is_none_when_config_lacks_interface(tmp_path: Path) -> None:
    """A config without rho_l/rho_v cannot define droplet metrics."""
    from tud_lbm.config import SimulationConfig

    run_dir = build_run_dir(tmp_path)
    bare = SimulationConfig(grid_shape=(16, 12), tau=0.9, nt=20)

    assert droplet_series_for_run(run_dir, bare) is None
