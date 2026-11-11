"""Old-layout configs: upgraded on load, and written back only for a run directory's file."""

from __future__ import annotations
import shutil
import tomllib
from pathlib import Path
import pytest
from src.config import DictAdapter
from src.config import SimulationConfig
from src.config import TomlAdapter
from src.config.array_expansion import expand_config
from src.config.legacy import upgrade
from src.config.run_config import CONFIG_FILENAME
from src.config.sections import build_sections

_FIXTURES = Path(__file__).parent / "fixtures"
_RUNS = ("legacy_run_wetting_init", "legacy_run_chemical_step")
_MULTIPHASE = {"eos": "double-well", "kappa": 0.02, "rho_l": 1.0, "rho_v": 0.5, "interface_width": 2}


def _run_dir_copy(name: str, tmp_path: Path) -> Path:
    path = tmp_path / CONFIG_FILENAME
    shutil.copy(_FIXTURES / name / CONFIG_FILENAME, path)
    return path


@pytest.mark.parametrize("name", _RUNS)
def test_an_old_run_directory_is_converted_in_place(name: str, tmp_path: Path) -> None:
    path = _run_dir_copy(name, tmp_path)

    config = TomlAdapter().load(str(path))

    converted = tomllib.loads(path.read_text(encoding="utf-8"))
    assert config.multiphase is not None
    assert "type" not in converted["simulation_type"]
    assert converted == build_sections(config)


@pytest.mark.parametrize("name", _RUNS)
def test_a_second_load_of_a_converted_file_changes_nothing(name: str, tmp_path: Path) -> None:
    path = _run_dir_copy(name, tmp_path)
    first = TomlAdapter().load(str(path))
    converted = path.read_text(encoding="utf-8")
    written = path.stat().st_mtime_ns

    second = TomlAdapter().load(str(path))

    assert second == first
    assert path.read_text(encoding="utf-8") == converted
    assert path.stat().st_mtime_ns == written


def test_an_input_toml_is_upgraded_but_not_rewritten(tmp_path: Path) -> None:
    path = tmp_path / "my_input.toml"
    shutil.copy(_FIXTURES / _RUNS[0] / CONFIG_FILENAME, path)
    original = path.read_text(encoding="utf-8")

    assert TomlAdapter().load(str(path)).multiphase is not None
    assert path.read_text(encoding="utf-8") == original


def test_a_read_only_run_directory_still_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _run_dir_copy(_RUNS[0], tmp_path)
    original = path.read_text(encoding="utf-8")

    def refuse(*_args: object) -> None:
        raise OSError

    monkeypatch.setattr(TomlAdapter, "save", refuse)

    assert TomlAdapter().load(str(path)).multiphase is not None
    assert path.read_text(encoding="utf-8") == original


def test_a_sweep_named_config_toml_is_left_alone(tmp_path: Path) -> None:
    path = tmp_path / CONFIG_FILENAME
    path.write_text('[simulation_type]\ntype = "single_phase"\ntau = [0.6, 0.7]\n', encoding="utf-8")
    original = path.read_text(encoding="utf-8")

    raw = TomlAdapter().load_raw(str(path))

    assert raw["tau"] == [0.6, 0.7]
    assert path.read_text(encoding="utf-8") == original


def test_a_single_phase_type_drops_the_section_the_old_loader_ignored() -> None:
    raw = {"simulation_type": {"type": "single_phase", "tau": 0.8}, "multiphase": _MULTIPHASE}

    upgraded, changed = upgrade(raw)

    assert changed
    assert "multiphase" not in upgraded
    assert raw["simulation_type"]["type"] == "single_phase"


def test_a_multiphase_type_without_its_section_still_fails_validation() -> None:
    raw = {"simulation_type": {"type": "multiphase", "tau": 0.8}}

    with pytest.raises(ValueError, match="'kappa' is required"):
        DictAdapter().load(raw)


def test_an_unknown_type_is_rejected() -> None:
    raw = {"simulation_type": {"type": "unknown_type"}}

    with pytest.raises(ValueError, match="Unknown simulation type"):
        upgrade(raw)


def test_flat_stand_ins_move_to_their_sections() -> None:
    raw = {
        "simulation_type": {
            "sim_type": "multiphase_wetting",
            "wetting_config": {"phi_l": 1.2},
            "gravity_force": {"force_g": 1e-6},
            **_MULTIPHASE,
        },
        "multiphase": {"lambda_v": 3.0},
    }

    upgraded, _ = upgrade(raw)

    assert upgraded["multiphase"] == _MULTIPHASE
    assert upgraded["wetting"] == {"phi_left": 1.2}
    assert upgraded["gravity_force"] == {"force_g": 1e-6}
    assert upgraded["simulation_type"] == {"lambda_v": 3.0}


def test_upgrade_is_idempotent() -> None:
    upgraded, _ = upgrade({"simulation_type": {"type": "multiphase", "tau": 0.8}, "multiphase": _MULTIPHASE})

    again, changed = upgrade(upgraded)

    assert not changed
    assert again == upgraded


def test_a_new_layout_config_is_not_touched() -> None:
    raw = {"simulation_type": {"tau": 0.8}, "multiphase": _MULTIPHASE, "gravity_force": {"force_g": 1e-6}}

    upgraded, changed = upgrade(raw)

    assert not changed
    assert upgraded is raw


def test_a_stray_flat_key_keeps_the_multiphase_section_without_a_type() -> None:
    raw = {"simulation_type": {"g": 1e-6}, "multiphase": _MULTIPHASE}

    upgraded, _ = upgrade(raw)

    assert upgraded["multiphase"] == {**_MULTIPHASE, "g": 1e-6}


def test_a_flat_old_field_is_rejected_rather_than_kept_in_extra() -> None:
    flat = {"grid_shape": (8, 8), "kappa": 0.02}

    with pytest.raises(ValueError, match="Old-layout field"):
        DictAdapter().load(flat)


def test_two_level_sweep_expands() -> None:
    raw = {
        "grid_shape": (8, 8),
        "forces": {"gravity_force": {"force_g": [1e-6, 2e-6], "inclination_angle_deg": 30.0}},
    }

    configs, metadata = expand_config(raw)

    assert [c.forces["gravity_force"]["force_g"] for c in configs] == [1e-6, 2e-6]
    assert all(c.forces["gravity_force"]["inclination_angle_deg"] == 30.0 for c in configs)
    assert metadata is not None
    assert metadata.field_names == frozenset({"forces.gravity_force.force_g"})


def test_force_defaults_come_from_one_table() -> None:
    config = SimulationConfig(grid_shape=(8, 8), forces={"gravity_force": {"force_g": 1e-6}})

    assert config.forces["gravity_force"] == {"force_g": 1e-6, "inclination_angle_deg": 0.0}


def _stepped(centre_x: float, **chemical_step: object) -> dict[str, dict[str, object]]:
    """A current-layout bubble config with windows labelled by position, as saved before 2c."""
    return {
        "simulation_type": {"grid_shape": [201, 101], "tau": 0.8},
        "initialisation": {"centres": [[centre_x, 1]], "radii": [0.3], "dispersed": "vapour"},
        "chemical_step": {
            "chemical_step_location": 0.5,
            "ca_advancing_pre_step": 70.0,
            "ca_receding_pre_step": 50.0,
            "ca_advancing_post_step": 120.0,
            "ca_receding_post_step": 100.0,
            **chemical_step,
        },
    }


def test_a_saved_config_starting_on_the_high_side_is_relabelled() -> None:
    raw = _stepped(0.7)

    upgraded, changed = upgrade(raw, saved=True)

    assert changed
    assert upgraded["chemical_step"] == {
        "chemical_step_location": 0.5,
        "ca_advancing_pre_step": 120.0,
        "ca_receding_pre_step": 100.0,
        "ca_advancing_post_step": 70.0,
        "ca_receding_post_step": 50.0,
        "chemical_step_pre_side": "high",
    }
    assert raw["chemical_step"]["ca_advancing_pre_step"] == 70.0
    assert upgrade(upgraded, saved=True) == (upgraded, False)


@pytest.mark.parametrize(
    ("raw", "saved"),
    [
        pytest.param(_stepped(0.7), False, id="an input TOML"),
        pytest.param(_stepped(0.3), True, id="a low start"),
        pytest.param(_stepped(0.7, chemical_step_pre_side="low"), True, id="a configured side"),
    ],
)
def test_other_chemical_step_configs_keep_their_labels(raw: dict[str, dict[str, object]], saved: bool) -> None:
    upgraded, changed = upgrade(raw, saved=saved)

    assert not changed
    assert upgraded is raw


def test_the_removed_carry_option_is_dropped_when_it_was_off() -> None:
    raw = {"simulation_type": {"tau": 0.8}, "hysteresis": {"ca_advancing": 90.0, "carry_inactive_params": False}}

    upgraded, changed = upgrade(raw)

    assert changed
    assert upgraded == {"simulation_type": {"tau": 0.8}, "hysteresis": {"ca_advancing": 90.0}}


def test_the_removed_carry_option_stays_on_record_when_it_was_on() -> None:
    raw = {"simulation_type": {"tau": 0.8}, "hysteresis": {"ca_advancing": 90.0, "carry_inactive_params": True}}

    upgraded, _ = upgrade(raw)

    assert upgraded["hysteresis"] == {"ca_advancing": 90.0}
    assert upgraded["simulation_type"] == {"tau": 0.8, "carry_inactive_params": True}
    assert upgrade(upgraded) == (upgraded, False)


def test_a_saved_bubble_run_loads_relabelled_and_is_rewritten(tmp_path: Path) -> None:
    raw = _stepped(0.7, chemical_step_edge="top", edge_width=1.0)
    raw["simulation_type"] |= {"nt": 10}
    raw["multiphase"] = dict(_MULTIPHASE)
    raw["boundary_conditions"] = {"top": "wetting", "bottom": "bounce-back"}
    raw["hysteresis"] = {"carry_inactive_params": False}
    path = tmp_path / CONFIG_FILENAME
    TomlAdapter().save(DictAdapter().load(_stepped_current(raw)), str(path))
    text = path.read_text(encoding="utf-8").replace('chemical_step_pre_side = "low"\n', "")
    path.write_text(text.replace("[hysteresis]\n", "[hysteresis]\ncarry_inactive_params = false\n"), encoding="utf-8")

    config = TomlAdapter().load(str(path))

    assert config.chemical_step is not None
    assert config.chemical_step["chemical_step_pre_side"] == "high"
    assert config.chemical_step["ca_advancing_pre_step"] == 120.0
    assert tomllib.loads(path.read_text(encoding="utf-8")) == build_sections(config)


def _stepped_current(raw: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    """*raw* as the pre-2c code read it: post at higher x, whatever the start side."""
    current = {name: dict(section) for name, section in raw.items()}
    current["chemical_step"]["chemical_step_pre_side"] = "low"
    current["hysteresis"] = {}
    return current
