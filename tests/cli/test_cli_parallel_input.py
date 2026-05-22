from __future__ import annotations
from click.testing import CliRunner
from tud_lbm.config import SimulationConfig
from tud_lbm.config.array_expansion import ArrayParameterSet
from tud_lbm.pipeline.parallel_runner import SimulationResult


def _make_config(results_dir: str, tau: float = 0.8) -> SimulationConfig:
    return SimulationConfig(
        grid_shape=(8, 8),
        tau=tau,
        nt=10,
        simulation_name="test",
        results_dir=results_dir,
    )


def test_cli_single_config_uses_single_run(monkeypatch, tmp_path):
    from tud_lbm.cli.cli import main

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[simulation_type]\ntype = 'single_phase'\n", encoding="utf-8")

    config = _make_config(str(tmp_path))

    monkeypatch.setattr("tud_lbm.config.adapter_toml.TomlAdapter.load_raw", lambda self, path: {"stub": "raw"})
    monkeypatch.setattr("tud_lbm.config.array_expansion.expand_config", lambda raw: ([config], None))

    called = {"single": False}

    def _fake_single_run(cfg):
        called["single"] = True
        assert cfg == config
        return str(tmp_path / "single")

    monkeypatch.setattr("tud_lbm.cli.cli._run_simulation", _fake_single_run)

    result = CliRunner().invoke(main, [str(cfg_path), "--no-prompt"])

    assert result.exit_code == 0
    assert called["single"] is True


def test_cli_list_operators_includes_plotting_and_analysis():
    from tud_lbm.cli.cli import main

    result = CliRunner().invoke(main, ["--list-simulation-operators"])

    assert result.exit_code == 0
    assert "plotting" in result.output
    assert "analysis" in result.output
    assert "density" in result.output
    assert "max_velocity" in result.output
    assert "simulation_csv" in result.output


def test_cli_array_config_uses_parallel_sweep(monkeypatch, tmp_path):
    from tud_lbm.cli.cli import main

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[simulation_type]\ntype = 'single_phase'\n", encoding="utf-8")

    config_a = _make_config(str(tmp_path), tau=0.6)
    config_b = _make_config(str(tmp_path), tau=0.8)

    metadata = ArrayParameterSet(
        field_names=frozenset({"tau"}),
        array_values={"tau": (0.6, 0.8)},
        total_combinations=2,
    )

    monkeypatch.setattr("tud_lbm.config.adapter_toml.TomlAdapter.load_raw", lambda self, path: {"stub": "raw"})
    monkeypatch.setattr(
        "tud_lbm.config.array_expansion.expand_config",
        lambda raw: ([config_a, config_b], metadata),
    )
    monkeypatch.setattr(
        "tud_lbm.config.array_expansion.enumerate_configs",
        lambda raw: iter(
            [
                (0, {"tau": 0.6}, config_a),
                (1, {"tau": 0.8}, config_b),
            ],
        ),
    )

    captured = {"called": False, "params": None}

    def _fake_parallel_sweep(configs, parameters_list, **kwargs):
        captured["called"] = True
        captured["params"] = parameters_list
        assert len(configs) == 2
        return [
            SimulationResult(index=0, config=config_a, status="success"),
            SimulationResult(index=1, config=config_b, status="success"),
        ]

    monkeypatch.setattr("tud_lbm.cli.cli._run_parallel_sweep", _fake_parallel_sweep)

    result = CliRunner().invoke(main, [str(cfg_path), "--no-prompt"])

    assert result.exit_code == 0
    assert captured["called"] is True
    assert captured["params"] == [{"tau": 0.6}, {"tau": 0.8}]


def test_cli_array_config_dry_run_skips_parallel_execution(monkeypatch, tmp_path):
    from tud_lbm.cli.cli import main

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[simulation_type]\ntype = 'single_phase'\n", encoding="utf-8")

    config_a = _make_config(str(tmp_path), tau=0.6)
    config_b = _make_config(str(tmp_path), tau=0.8)

    metadata = ArrayParameterSet(
        field_names=frozenset({"tau"}),
        array_values={"tau": (0.6, 0.8)},
        total_combinations=2,
    )

    monkeypatch.setattr("tud_lbm.config.adapter_toml.TomlAdapter.load_raw", lambda self, path: {"stub": "raw"})
    monkeypatch.setattr(
        "tud_lbm.config.array_expansion.expand_config",
        lambda raw: ([config_a, config_b], metadata),
    )
    monkeypatch.setattr(
        "tud_lbm.config.array_expansion.enumerate_configs",
        lambda raw: iter(
            [
                (0, {"tau": 0.6}, config_a),
                (1, {"tau": 0.8}, config_b),
            ],
        ),
    )

    called = {"parallel": False}

    def _fake_parallel_sweep(*args, **kwargs):
        called["parallel"] = True
        return []

    monkeypatch.setattr("tud_lbm.cli.cli._run_parallel_sweep", _fake_parallel_sweep)

    result = CliRunner().invoke(main, [str(cfg_path), "--no-prompt", "--dry-run"])

    assert result.exit_code == 0
    assert called["parallel"] is False


def test_cli_override_updates_scalar_field_before_single_run(monkeypatch, tmp_path):
    from tud_lbm.cli.cli import main

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[simulation_type]\ntype = 'single_phase'\n", encoding="utf-8")

    captured = {"raw": None}

    def _fake_load_raw(self, path):
        return {
            "sim_type": "single_phase",
            "simulation_name": "old_name",
            "grid_shape": (8, 8),
            "tau": 0.8,
            "nt": 10,
            "results_dir": str(tmp_path),
        }

    def _fake_expand(raw):
        captured["raw"] = dict(raw)
        return [_make_config(str(tmp_path), tau=0.8)], None

    monkeypatch.setattr("tud_lbm.config.adapter_toml.TomlAdapter.load_raw", _fake_load_raw)
    monkeypatch.setattr("tud_lbm.config.array_expansion.expand_config", _fake_expand)
    monkeypatch.setattr("tud_lbm.cli.cli._run_simulation", lambda cfg: str(tmp_path / "single"))

    result = CliRunner().invoke(
        main,
        [
            str(cfg_path),
            "--no-prompt",
            "--override",
            'simulation_type.simulation_name="new name"',
        ],
    )

    assert result.exit_code == 0
    assert captured["raw"]["simulation_name"] == "new name"


def test_cli_override_updates_nested_sweep_field(monkeypatch, tmp_path):
    from tud_lbm.cli.cli import main

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[simulation_type]\ntype = 'single_phase'\n", encoding="utf-8")

    config_a = _make_config(str(tmp_path), tau=0.6)
    metadata = ArrayParameterSet(
        field_names=frozenset({"gravity_force.inclination_angle_deg"}),
        array_values={"gravity_force.inclination_angle_deg": (50, 60)},
        total_combinations=2,
    )

    captured = {"raw": None}

    monkeypatch.setattr(
        "tud_lbm.config.adapter_toml.TomlAdapter.load_raw",
        lambda self, path: {"gravity_force": {"force_g": 5e-7, "inclination_angle_deg": 50}},
    )

    def _fake_expand(raw):
        captured["raw"] = raw
        return [config_a, config_a], metadata

    monkeypatch.setattr("tud_lbm.config.array_expansion.expand_config", _fake_expand)
    monkeypatch.setattr(
        "tud_lbm.config.array_expansion.enumerate_configs",
        lambda raw: iter([(0, {"gravity_force.inclination_angle_deg": 50}, config_a)] * 2),
    )
    monkeypatch.setattr(
        "tud_lbm.cli.cli._run_parallel_sweep",
        lambda configs, parameters_list, **kwargs: [SimulationResult(index=0, config=config_a, status="success")],
    )

    result = CliRunner().invoke(
        main,
        [
            str(cfg_path),
            "--no-prompt",
            "--override",
            "gravity_force.inclination_angle_deg=[50, 60]",
        ],
    )

    assert result.exit_code == 0
    assert captured["raw"]["gravity_force"]["inclination_angle_deg"] == [50, 60]


def test_cli_override_rejects_invalid_value(monkeypatch, tmp_path):
    from tud_lbm.cli.cli import main

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[simulation_type]\ntype = 'single_phase'\n", encoding="utf-8")

    monkeypatch.setattr("tud_lbm.config.adapter_toml.TomlAdapter.load_raw", lambda self, path: {"tau": 0.8})

    result = CliRunner().invoke(
        main,
        [
            str(cfg_path),
            "--no-prompt",
            "--override",
            "tau=not_a_toml_literal",
        ],
    )

    assert result.exit_code == 1
    assert "invalid override value" in result.output


def test_cli_init_dir_sets_init_type_to_init_from_file(monkeypatch, tmp_path):
    from tud_lbm.cli.cli import main

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[simulation_type]\ntype = 'single_phase'\n", encoding="utf-8")

    init_snapshot = tmp_path / "timestep_100.npz"
    init_snapshot.write_text("stub", encoding="utf-8")

    captured = {"raw": None}

    def _fake_load_raw(self, path):
        return {
            "sim_type": "single_phase",
            "init_type": "multiphase_bubbles",
            "grid_shape": (8, 8),
            "tau": 0.8,
            "nt": 10,
            "results_dir": str(tmp_path),
        }

    def _fake_expand(raw):
        captured["raw"] = dict(raw)
        return [_make_config(str(tmp_path))], None

    monkeypatch.setattr("tud_lbm.config.adapter_toml.TomlAdapter.load_raw", _fake_load_raw)
    monkeypatch.setattr("tud_lbm.config.array_expansion.expand_config", _fake_expand)
    monkeypatch.setattr("tud_lbm.cli.cli._run_simulation", lambda cfg: str(tmp_path / "single"))

    result = CliRunner().invoke(
        main,
        [
            str(cfg_path),
            "--no-prompt",
            "--init-dir",
            str(init_snapshot),
        ],
    )

    assert result.exit_code == 0
    assert captured["raw"]["init_dir"] == str(init_snapshot)
    assert captured["raw"]["init_type"] == "init_from_file"


def test_cli_init_dir_default_init_type_is_overrideable(monkeypatch, tmp_path):
    from tud_lbm.cli.cli import main

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[simulation_type]\ntype = 'single_phase'\n", encoding="utf-8")

    init_snapshot = tmp_path / "timestep_100.npz"
    init_snapshot.write_text("stub", encoding="utf-8")

    captured = {"raw": None}

    monkeypatch.setattr(
        "tud_lbm.config.adapter_toml.TomlAdapter.load_raw",
        lambda self, path: {"sim_type": "single_phase", "results_dir": str(tmp_path)},
    )

    def _fake_expand(raw):
        captured["raw"] = dict(raw)
        return [_make_config(str(tmp_path))], None

    monkeypatch.setattr("tud_lbm.config.array_expansion.expand_config", _fake_expand)
    monkeypatch.setattr("tud_lbm.cli.cli._run_simulation", lambda cfg: str(tmp_path / "single"))

    result = CliRunner().invoke(
        main,
        [
            str(cfg_path),
            "--no-prompt",
            "--init-dir",
            str(init_snapshot),
            "--override",
            'init_type="multiphase_bubbles"',
        ],
    )

    assert result.exit_code == 0
    assert captured["raw"]["init_dir"] == str(init_snapshot)
    assert captured["raw"]["init_type"] == "multiphase_bubbles"


def test_cli_init_wetting_requires_config_path():
    from tud_lbm.cli.cli import main

    result = CliRunner().invoke(main, ["--init-wetting"])

    assert result.exit_code == 1
    assert "--init-wetting requires CONFIG_PATH" in result.output


def test_cli_debug_wetting_sets_runtime_flag(monkeypatch, tmp_path):
    from tud_lbm.cli.cli import main
    from tud_lbm.config import config_overview

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[simulation_type]\ntype = 'single_phase'\n", encoding="utf-8")

    config = _make_config(str(tmp_path))

    monkeypatch.setattr("tud_lbm.config.adapter_toml.TomlAdapter.load_raw", lambda self, path: {"stub": "raw"})
    monkeypatch.setattr("tud_lbm.config.array_expansion.expand_config", lambda raw: ([config], None))
    monkeypatch.setattr("tud_lbm.cli.cli._run_simulation", lambda cfg: str(tmp_path / "single"))
    monkeypatch.setattr(config_overview, "DEBUG_FLAG", False)

    result = CliRunner().invoke(main, [str(cfg_path), "--no-prompt", "--debug-wetting"])

    assert result.exit_code == 0
    assert config_overview.DEBUG_FLAG is True


def _assert_phase1_raw(phase1_raw: dict[str, object]) -> None:
    assert phase1_raw["sim_type"] == "multiphase_wetting"
    assert phase1_raw["init_type"] == "multiphase_bubbles"
    assert "hysteresis_config" not in phase1_raw
    assert "chemical_step_config" not in phase1_raw
    assert "gravity_force" not in phase1_raw
    assert "gravity_masked_force" not in phase1_raw
    assert phase1_raw["bc_config"]["top"] == "bounce-back"
    assert phase1_raw["bc_config"]["bottom"] == "bounce-back"
    assert phase1_raw["nt"] == 50_000
    assert phase1_raw["save_interval"] == 50_000
    assert phase1_raw["output_format"] == "numpy"
    assert phase1_raw["simulation_name"] == "wetting_init"
    assert phase1_raw["wetting_config"]["contact_angle"] == 90.0
    assert phase1_raw["wetting_config"]["phi_left"] == 1.2
    assert phase1_raw["wetting_config"]["phi_right"] == 1.2
    assert phase1_raw["wetting_config"]["d_rho_left"] == 0.3
    assert phase1_raw["wetting_config"]["d_rho_right"] == 0.3


def _assert_phase2_raw(phase2_raw: dict[str, object], phase1_data_dir: str) -> None:
    assert phase2_raw["gravity_force"]["force_g"] == 5e-7
    assert phase2_raw["gravity_masked_force"]["force_g"] == 7e-7
    assert phase2_raw["sim_type"] == "multiphase_hysteresis_chemical_step"
    assert phase2_raw["init_type"] == "init_from_file"
    assert phase2_raw["init_dir"] == f"{phase1_data_dir}/timestep_50000.npz"
    assert phase2_raw["wetting_config"]["contact_angle"] == 90.0
    assert phase2_raw["wetting_config"]["phi_left"] == 1.2
    assert phase2_raw["wetting_config"]["phi_right"] == 1.2
    assert phase2_raw["wetting_config"]["d_rho_left"] == 0.3
    assert phase2_raw["wetting_config"]["d_rho_right"] == 0.3


def test_cli_init_wetting_runs_two_phase_flow(monkeypatch, tmp_path):
    from tud_lbm.cli.cli import main

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[simulation_type]\ntype = 'single_phase'\n", encoding="utf-8")

    phase1_cfg = _make_config(str(tmp_path / "phase1"))
    phase2_cfg = _make_config(str(tmp_path / "phase2"))

    base_raw = {
        "sim_type": "multiphase_hysteresis_chemical_step",
        "init_type": "init_from_file",
        "gravity_force": {"force_g": 5e-7},
        "gravity_masked_force": {"force_g": 7e-7},
        "hysteresis_config": {"foo": 1},
        "chemical_step_config": {"bar": 2},
        "bc_config": {"top": "bounce-back", "bottom": "bounce-back"},
        "wetting_config": {"contact_angle": 90.0},
    }
    expanded_raws: list[dict[str, object]] = []
    run_calls: list[SimulationConfig] = []

    monkeypatch.setattr(
        "tud_lbm.config.adapter_toml.TomlAdapter.load_raw",
        lambda self, path: dict(base_raw),
    )

    def _fake_expand(raw):
        expanded_raws.append(dict(raw))
        if len(expanded_raws) == 1:
            return [phase1_cfg], None
        return [phase2_cfg], None

    monkeypatch.setattr("tud_lbm.config.array_expansion.expand_config", _fake_expand)
    monkeypatch.setattr(
        "tud_lbm.cli.cli.Prompt.ask",
        lambda _text, default=None: {
            "1.0": "1.2",
            "0.0": "0.3",
        }[default],
    )

    def _fake_run_simulation(cfg):
        run_calls.append(cfg)
        if len(run_calls) == 1:
            return str(tmp_path / "phase1_data")
        return str(tmp_path / "phase2_data")

    monkeypatch.setattr("tud_lbm.cli.cli._run_simulation", _fake_run_simulation)
    monkeypatch.setattr("tud_lbm.cli.cli._display_config_summary", lambda cfg: None)
    monkeypatch.setattr("tud_lbm.cli.cli.Confirm.ask", lambda *args, **kwargs: True)

    result = CliRunner().invoke(main, [str(cfg_path), "--init-wetting"])

    assert result.exit_code == 0
    assert len(expanded_raws) == 2
    phase1_raw, phase2_raw = expanded_raws
    _assert_phase1_raw(phase1_raw)
    _assert_phase2_raw(phase2_raw, str(tmp_path / "phase1_data"))
    assert run_calls == [phase1_cfg, phase2_cfg]


def test_cli_init_wetting_rejects_phase1_sweep(monkeypatch, tmp_path):
    from tud_lbm.cli.cli import main

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[simulation_type]\ntype = 'single_phase'\n", encoding="utf-8")

    cfg_a = _make_config(str(tmp_path / "a"), tau=0.6)
    cfg_b = _make_config(str(tmp_path / "b"), tau=0.8)

    monkeypatch.setattr("tud_lbm.config.adapter_toml.TomlAdapter.load_raw", lambda self, path: {})
    monkeypatch.setattr("tud_lbm.config.array_expansion.expand_config", lambda raw: ([cfg_a, cfg_b], None))
    monkeypatch.setattr("tud_lbm.cli.cli.Prompt.ask", lambda _text, default=None: default)

    result = CliRunner().invoke(main, [str(cfg_path), "--no-prompt", "--init-wetting"])

    assert result.exit_code == 1
    assert "--init-wetting does not support parameter sweeps" in result.output
    assert "Phase 1" in result.output


def test_cli_init_wetting_rejects_phase2_sweep(monkeypatch, tmp_path):
    from tud_lbm.cli.cli import main

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[simulation_type]\ntype = 'single_phase'\n", encoding="utf-8")

    cfg_phase1 = _make_config(str(tmp_path / "phase1"))
    cfg_a = _make_config(str(tmp_path / "a"), tau=0.6)
    cfg_b = _make_config(str(tmp_path / "b"), tau=0.8)

    calls = {"expand": 0}

    monkeypatch.setattr("tud_lbm.config.adapter_toml.TomlAdapter.load_raw", lambda self, path: {})

    def _fake_expand(raw):
        calls["expand"] += 1
        if calls["expand"] == 1:
            return [cfg_phase1], None
        return [cfg_a, cfg_b], None

    monkeypatch.setattr("tud_lbm.config.array_expansion.expand_config", _fake_expand)
    monkeypatch.setattr("tud_lbm.cli.cli.Prompt.ask", lambda _text, default=None: default)
    monkeypatch.setattr("tud_lbm.cli.cli._display_config_summary", lambda cfg: None)
    monkeypatch.setattr("tud_lbm.cli.cli._run_simulation", lambda cfg: str(tmp_path / "phase1_data"))

    result = CliRunner().invoke(main, [str(cfg_path), "--no-prompt", "--init-wetting"])

    assert result.exit_code == 1
    assert "--init-wetting does not support parameter sweeps" in result.output
    assert "Phase 2" in result.output
