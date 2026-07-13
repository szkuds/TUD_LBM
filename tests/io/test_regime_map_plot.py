"""End-to-end tests for tud_lbm.io.plotting.regime_map_plot."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from tud_lbm.config import SimulationConfig
from tud_lbm.io.plotting.regime_map_plot import _REGIME_COLORS
from tud_lbm.io.plotting.regime_map_plot import _REGIME_MARKERS
from tud_lbm.io.plotting.regime_map_plot import RunRegimeEntry
from tud_lbm.io.plotting.regime_map_plot import build_regime_map
from tud_lbm.io.plotting.regime_map_plot import parse_run_dir_list
from tud_lbm.io.plotting.regime_map_plot import plot_regime_map
from tud_lbm.io.plotting.regime_map_plot import process_run_dir
from tud_lbm.io.readers import TomlAdapter

if TYPE_CHECKING:
    from pathlib import Path

_NX, _NY = 60, 12


def _run_config(**kwargs) -> SimulationConfig:
    base = {
        "sim_type": "multiphase_hysteresis_chemical_step",
        "grid_shape": (_NX, _NY),
        "tau": 0.9,
        "nt": 30,
        "save_interval": 10,
        "eos": "double-well",
        "kappa": 0.02,
        "interface_width": 2,
        "rho_l": 1.0,
        "rho_v": 0.2,
        "gravity_masked_force": {"force_g": 1e-6, "inclination_angle_deg": 30.0},
        "initialisation": {"radii": [0.25], "centres": [[0.5, 0.5]]},
    }
    base.update(kwargs)
    return SimulationConfig(**base)  # ty: ignore[invalid-argument-type]


def _write_snapshot(data_dir: Path, step: int, x_start: int, ux: float) -> None:
    rho = np.full((_NX, _NY, 1, 1, 1), 0.2)
    rho[x_start : x_start + 8, 1:8, 0, 0, 0] = 1.0
    u = np.zeros((_NX, _NY, 1, 1, 2))
    u[:, :, 0, 0, 0] = ux

    np.savez(
        data_dir / f"timestep_{step}.npz",
        rho=rho,
        u=u,
        ca_left=np.array(80.0),
        ca_right=np.array(95.0),
        cll_left=np.array(3.0 + step * 0.05),
        cll_right=np.array(10.0 + step * 0.05),
    )


def _build_run_dir(run_dir: Path, x_starts: list[int], config: SimulationConfig) -> None:
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)
    for step, x_start in zip(range(0, len(x_starts) * 10, 10), x_starts, strict=True):
        _write_snapshot(data_dir, step, x_start, ux=0.01 * (step + 1))
    TomlAdapter().save(config, str(run_dir / "config.toml"))


def test_parse_run_dir_list_skips_blank_and_comment_lines(tmp_path: Path):
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_b").mkdir()
    txt_path = tmp_path / "dirs.txt"
    txt_path.write_text("# header comment\n\nrun_a\nrun_b\n", encoding="utf-8")

    dirs = parse_run_dir_list(txt_path)

    assert dirs == [tmp_path / "run_a", tmp_path / "run_b"]


def test_parse_run_dir_list_unescapes_single_quoted_special_chars(tmp_path: Path):
    run_name = r"22-12-23_$Bo_\parallel = 0.60; Oh = 0.47$"
    (tmp_path / run_name).mkdir()
    txt_path = tmp_path / "dirs.txt"
    quoted_line = r"'" + tmp_path.as_posix() + r"/22-12-23_\$Bo_\\parallel\ \=\ 0.60\;\ Oh\ \=\ 0.47\$'"
    txt_path.write_text(quoted_line + "\n", encoding="utf-8")

    dirs = parse_run_dir_list(txt_path)

    assert dirs == [tmp_path / run_name]


def test_parse_run_dir_list_unescapes_unquoted_special_chars(tmp_path: Path):
    run_name = r"08-13-44_$Bo_\parallel = 0.80; Oh = 0.45$"
    (tmp_path / run_name).mkdir()
    txt_path = tmp_path / "dirs.txt"
    escaped_line = tmp_path.as_posix() + r"/08-13-44_\$Bo_\\parallel\ \=\ 0.80\;\ Oh\ \=\ 0.45\$"
    txt_path.write_text(escaped_line + "\n", encoding="utf-8")

    dirs = parse_run_dir_list(txt_path)

    assert dirs == [tmp_path / run_name]


def test_parse_run_dir_list_keeps_unquoted_spaces_in_one_line(tmp_path: Path):
    run_name = r"08-13-44_$Bo_\parallel = 0.60; Oh = 0.30$"
    (tmp_path / run_name).mkdir()
    other_name = r"08-13-44_$Bo_\parallel = 0.80; Oh = 0.30$"
    (tmp_path / other_name).mkdir()
    txt_path = tmp_path / "dirs.txt"
    txt_path.write_text(f"{tmp_path.as_posix()}/{run_name}\n{tmp_path.as_posix()}/{other_name}\n", encoding="utf-8")

    dirs = parse_run_dir_list(txt_path)

    assert dirs == [tmp_path / run_name, tmp_path / other_name]


def test_parse_run_dir_list_strips_double_quotes(tmp_path: Path):
    (tmp_path / "run_a").mkdir()
    txt_path = tmp_path / "dirs.txt"
    txt_path.write_text(f'"{tmp_path.as_posix()}/run_a"\n', encoding="utf-8")

    dirs = parse_run_dir_list(txt_path)

    assert dirs == [tmp_path / "run_a"]


def test_process_run_dir_classifies_pinned_run(tmp_path: Path):
    run_dir = tmp_path / "pinned_run"
    _build_run_dir(run_dir, [4, 4, 4, 4], _run_config())

    entry = process_run_dir(run_dir)

    assert entry is not None
    assert entry.regime == "Pinning"
    assert (run_dir / "plots" / "analysis" / "acceleration_analysis.png").exists()


def test_process_run_dir_classifies_mobile_run(tmp_path: Path):
    run_dir = tmp_path / "mobile_run"
    _build_run_dir(run_dir, [4, 10, 30, 48], _run_config())

    entry = process_run_dir(run_dir)

    assert entry is not None
    assert entry.regime != "pinning"
    assert (run_dir / "plots" / "analysis" / "acceleration_analysis.png").exists()


def test_process_run_dir_none_for_missing_config(tmp_path: Path):
    run_dir = tmp_path / "no_config"
    run_dir.mkdir()

    assert process_run_dir(run_dir) is None


def test_process_run_dir_none_for_calibration_only_eos_without_surface_tension(tmp_path: Path):
    run_dir = tmp_path / "no_dimensionless"
    cfg = _run_config(eos="carnahan-starling", a_eos=1.0, b_eos=4.0, r_eos=1.0, t_eos=0.07)
    _build_run_dir(run_dir, [4, 4, 4, 4], cfg)

    assert process_run_dir(run_dir) is None


def test_plot_regime_map_writes_file_with_all_regime_markers(tmp_path: Path):
    entries = [
        RunRegimeEntry(run_dir=tmp_path / "a", label="a", bo_parallel=1.0, oh=0.1, regime="Pinning"),
        RunRegimeEntry(run_dir=tmp_path / "b", label="b", bo_parallel=2.0, oh=0.2, regime="Dissipative"),
        RunRegimeEntry(run_dir=tmp_path / "c", label="c", bo_parallel=3.0, oh=0.3, regime="Inertial"),
        RunRegimeEntry(run_dir=tmp_path / "d", label="d", bo_parallel=4.0, oh=0.4, regime="unknown"),
    ]
    assert set(_REGIME_MARKERS) == {e.regime for e in entries}
    assert set(_REGIME_COLORS) == {e.regime for e in entries}

    out_path = plot_regime_map(entries, tmp_path / "regime_map.png")

    assert out_path.exists()


def test_build_regime_map_end_to_end(tmp_path: Path):
    _build_run_dir(tmp_path / "pinned_run", [4, 4, 4, 4], _run_config())
    _build_run_dir(tmp_path / "mobile_run", [4, 10, 30, 48], _run_config())
    txt_path = tmp_path / "dirs.txt"
    txt_path.write_text(
        "# comparison run list\n\npinned_run\nmobile_run\n",
        encoding="utf-8",
    )

    out_path = build_regime_map(txt_path)

    assert out_path is not None
    assert out_path.exists()
    assert out_path.parent == tmp_path / "regime_map_analysis"


def test_build_regime_map_none_when_no_runs_usable(tmp_path: Path):
    txt_path = tmp_path / "dirs.txt"
    txt_path.write_text("missing_run\n", encoding="utf-8")

    assert build_regime_map(txt_path) is None
