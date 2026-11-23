"""``tud-lbm sweep stage``: one completed, validated TOML per combination."""

from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
from click.testing import CliRunner
from src.cli.commands import cli
from src.config import TomlAdapter
from src.operators.collision._mrt import SHEAR_MOMENTS

if TYPE_CHECKING:
    from pathlib import Path
    from src.config import SimulationConfig

_EXIT_USAGE = 2

_BASE = """\
[simulation_type]
simulation_name = "base"
grid_shape = [16, 16]
tau = 0.8
nt = 10
"""

_DECOUPLED = "[0.0, 0.5, 0.6, 0.0, 1.2, 0.0, 1.2, 1.0, 1.0]"


def _config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "cfg.toml"
    path.write_text(_BASE + extra, encoding="utf-8")
    return path


def _stage(config: Path, *args: str):
    return CliRunner().invoke(cli, ["sweep", "stage", str(config), *args])


def _staged(out: Path) -> dict[str, SimulationConfig]:
    """The staged configs in *out*, by file name."""
    return {path.name: TomlAdapter().load(str(path)) for path in sorted(out.glob("*.toml"))}


def test_every_combination_is_written_and_reloads_to_its_values(tmp_path):
    out = tmp_path / "out"

    result = _stage(
        _config(tmp_path, "\n[gravity_force]\nforce_g = 1e-6\n"),
        "--set",
        "tau=[0.7, 0.9]",
        "--set",
        "gravity_force.inclination_angle_deg=[20, 40, 60]",
        "--out",
        str(out),
    )

    assert result.exit_code == 0, result.output
    staged = _staged(out)
    assert len(staged) == 6
    config = staged["base_tau_0.9_inclination_angle_deg_40.toml"]
    assert config.simulation_name == "base_tau_0.9_inclination_angle_deg_40"
    assert config.tau == pytest.approx(0.9)
    assert config.forces["gravity_force"] == {"force_g": 1e-6, "inclination_angle_deg": 40}


def test_template_names_the_runs(tmp_path):
    out = tmp_path / "out"

    result = _stage(_config(tmp_path), "--set", "tau=[0.7, 0.9]", "--name", "var_tau_{tau}", "--out", str(out))

    assert result.exit_code == 0, result.output
    assert sorted(_staged(out)) == ["var_tau_0.7.toml", "var_tau_0.9.toml"]


def test_grid_shape_axis_takes_arrays_of_arrays(tmp_path):
    out = tmp_path / "out"

    result = _stage(_config(tmp_path), "--set", "grid_shape=[[16, 16], [32, 16]]", "--out", str(out))

    assert result.exit_code == 0, result.output
    config = _staged(out)["base_grid_shape_32x16.toml"]
    assert config.grid_shape == (32, 16, 1)


def test_default_out_dir_sits_beside_the_config(tmp_path):
    result = _stage(_config(tmp_path), "--set", "tau=[0.7]")

    assert result.exit_code == 0, result.output
    assert (tmp_path / "cfg_sweep" / "base_tau_0.7.toml").is_file()


def test_each_staged_mrt_config_couples_its_own_tau(tmp_path):
    """The saved ``k_diag`` carries the shear rates of that run's ``tau``, not the base's."""
    out = tmp_path / "out"
    config = _config(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8") + f'collision_scheme = "mrt"\nk_diag = {_DECOUPLED}\n', encoding="utf-8"
    )

    result = _stage(config, "--set", "tau=[0.8, 1.2]", "--out", str(out))

    assert result.exit_code == 0, result.output
    for staged in _staged(out).values():
        assert staged.k_diag is not None
        assert staged.k_diag[SHEAR_MOMENTS[0]] == pytest.approx(1.0 / staged.tau)


def test_override_applies_before_the_axes(tmp_path):
    out = tmp_path / "out"

    result = _stage(_config(tmp_path), "--set", "tau=[0.7]", "--override", "nt=20", "--out", str(out))

    assert result.exit_code == 0, result.output
    assert _staged(out)["base_tau_0.7.toml"].nt == 20


@pytest.mark.parametrize(
    "args",
    [
        ("--set", "tau=0.7"),
        ("--set", "tau=[]"),
        ("--set", "taau=[0.7]"),
        ("--set", "tau=[0.7]", "--set", "simulation_type.tau=[0.9]"),
        ("--set", "tau=[0.7]", "--name", "run_{nt}"),
        ("--set", "tau=[0.7, 0.9]", "--name", "same"),
    ],
    ids=["not-array", "empty", "unknown-field", "duplicate-field", "unknown-template-field", "duplicate-names"],
)
def test_bad_axes_are_usage_errors_and_write_nothing(tmp_path, args):
    out = tmp_path / "out"

    result = _stage(_config(tmp_path), *args, "--out", str(out))

    assert result.exit_code == _EXIT_USAGE
    assert not out.exists()


def test_missing_base_name_is_a_usage_error(tmp_path):
    config = tmp_path / "cfg.toml"
    config.write_text("[simulation_type]\ngrid_shape = [16, 16]\n", encoding="utf-8")

    result = _stage(config, "--set", "tau=[0.7]", "--out", str(tmp_path / "out"))

    assert result.exit_code == _EXIT_USAGE
    assert "simulation_name" in result.output


def test_an_invalid_combination_writes_no_file(tmp_path):
    out = tmp_path / "out"

    result = _stage(_config(tmp_path), "--set", "tau=[0.8, 0.4]", "--out", str(out))

    assert result.exit_code == 1
    assert "base_tau_0.4" in result.output
    assert not out.exists()


def test_existing_files_need_force(tmp_path):
    out = tmp_path / "out"
    config = _config(tmp_path)
    assert _stage(config, "--set", "tau=[0.7]", "--out", str(out)).exit_code == 0

    refused = _stage(config, "--set", "tau=[0.7]", "--override", "nt=30", "--out", str(out))
    forced = _stage(config, "--set", "tau=[0.7]", "--override", "nt=30", "--out", str(out), "--force")

    assert refused.exit_code == _EXIT_USAGE
    assert forced.exit_code == 0
    assert _staged(out)["base_tau_0.7.toml"].nt == 30


def test_dry_run_writes_nothing(tmp_path):
    out = tmp_path / "out"

    result = _stage(_config(tmp_path), "--set", "tau=[0.7, 0.9]", "--out", str(out), "--dry-run")

    assert result.exit_code == 0, result.output
    assert "base_tau_0.9" in result.output
    assert not out.exists()
