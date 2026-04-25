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
        return cfg

    monkeypatch.setattr("tud_lbm.cli.cli._run_simulation", _fake_single_run)

    result = CliRunner().invoke(main, [str(cfg_path), "--no-prompt"])

    assert result.exit_code == 0
    assert called["single"] is True


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
    monkeypatch.setattr("tud_lbm.cli.cli._run_simulation", lambda cfg: cfg)

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
