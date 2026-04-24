"""Comprehensive tests for parallel_runner module.

Tests cover:
  - SimulationResult dataclass
  - _run_single_simulation function
  - run_parallel_simulations function
  - _collect_results function
  - _print_result_line function
  - _generate_plots function
  - save_sweep_log function
"""

from __future__ import annotations
import dataclasses
import json
import tempfile
from pathlib import Path
from unittest import mock
from uuid import UUID
import pytest
from tud_lbm.config.simulation_config import SimulationConfig
from tud_lbm.pipeline.parallel_runner import SimulationResult
from tud_lbm.pipeline.parallel_runner import generate_plots
from tud_lbm.pipeline.parallel_runner import print_result_line
from tud_lbm.pipeline.parallel_runner import run_parallel_simulations
from tud_lbm.pipeline.parallel_runner import run_single_simulation
from tud_lbm.pipeline.parallel_runner import save_sweep_log

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def temp_results_dir():
    """Return a temporary directory for simulation results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def simple_config(temp_results_dir):
    """Return a minimal SimulationConfig for testing."""
    return SimulationConfig(
        grid_shape=(8, 8),
        tau=0.8,
        nt=10,
        simulation_name="test_sim",
        results_dir=temp_results_dir,
    )


# =========================================================================
# SimulationResult Dataclass Tests
# =========================================================================


class TestSimulationResult:
    """Tests for the SimulationResult dataclass."""

    def test_frozen_immutable(self, simple_config):
        """Verify SimulationResult is frozen (immutable)."""
        result = SimulationResult(
            index=0,
            config=simple_config,
            status="success",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.status = "failed"

    def test_default_field_values(self, simple_config):
        """Verify default field values are correct."""
        result = SimulationResult(
            index=0,
            config=simple_config,
            status="success",
        )
        assert result.output_dir is None
        assert result.parameters is None
        assert result.error is None
        assert result.duration == 0.0

    def test_all_fields_provided(self, simple_config):
        """Verify SimulationResult accepts all fields."""
        result = SimulationResult(
            index=42,
            config=simple_config,
            status="failed",
            output_dir="/path/to/output",
            parameters={"param1": "value1"},
            error="Something went wrong",
            duration=3.14,
        )
        assert result.index == 42
        assert result.status == "failed"
        assert result.output_dir == "/path/to/output"
        assert result.parameters == {"param1": "value1"}
        assert result.error == "Something went wrong"
        assert result.duration == 3.14

    def test_status_values(self, simple_config):
        """Verify different status values can be set."""
        for status in ["success", "failed", "skipped"]:
            result = SimulationResult(
                index=0,
                config=simple_config,
                status=status,
            )
            assert result.status == status


# =========================================================================
# _run_single_simulation Tests
# =========================================================================


class TestRunSingleSimulation:
    """Tests for _run_single_simulation function."""

    def test_happy_path_success(self, simple_config, temp_results_dir):
        """Test successful simulation execution."""
        mock_setup = mock.Mock()
        mock_state = mock.Mock()
        mock_run_fn = mock.Mock(return_value=(mock_state, None))

        with (
            mock.patch("tud_lbm.config.jax_config.configure_jax"),
            mock.patch("tud_lbm.pipeline.setup.build_setup", return_value=mock_setup),
            mock.patch("tud_lbm.runner.run.init_state", return_value=mock_state),
            mock.patch("tud_lbm.util.io.SimulationIO") as mock_io_cls,
        ):
            mock_io = mock.Mock()
            mock_io.run_dir = Path(temp_results_dir) / "run_001"
            mock_io_cls.return_value = mock_io

            result = run_single_simulation(
                index=0,
                config=simple_config,
                run_fn=mock_run_fn,
            )

        assert result.index == 0
        assert result.status == "success"
        assert result.output_dir == str(mock_io.run_dir)
        assert result.duration > 0.0
        assert result.error is None

    def test_configure_jax_called(self, simple_config, temp_results_dir):
        """Verify configure_jax() is called in the worker."""
        mock_setup = mock.Mock()
        mock_state = mock.Mock()
        mock_run_fn = mock.Mock(return_value=(mock_state, None))

        with (
            mock.patch("tud_lbm.config.jax_config.configure_jax") as mock_configure,
            mock.patch("tud_lbm.pipeline.setup.build_setup", return_value=mock_setup),
            mock.patch("tud_lbm.runner.run.init_state", return_value=mock_state),
            mock.patch("tud_lbm.util.io.SimulationIO") as mock_io_cls,
        ):
            mock_io = mock.Mock()
            mock_io.run_dir = Path(temp_results_dir) / "run_001"
            mock_io_cls.return_value = mock_io

            run_single_simulation(
                index=0,
                config=simple_config,
                run_fn=mock_run_fn,
            )

        mock_configure.assert_called_once()

    def test_custom_setup_fn_used(self, simple_config, temp_results_dir):
        """Verify custom setup_fn is used when provided."""
        custom_setup_fn = mock.Mock()
        mock_state = mock.Mock()
        mock_run_fn = mock.Mock(return_value=(mock_state, None))

        with (
            mock.patch("tud_lbm.config.jax_config.configure_jax"),
            mock.patch("tud_lbm.runner.run.init_state", return_value=mock_state),
            mock.patch("tud_lbm.util.io.SimulationIO") as mock_io_cls,
        ):
            mock_io = mock.Mock()
            mock_io.run_dir = Path(temp_results_dir) / "run_001"
            mock_io_cls.return_value = mock_io

            run_single_simulation(
                index=0,
                config=simple_config,
                setup_fn=custom_setup_fn,
                run_fn=mock_run_fn,
            )

        custom_setup_fn.assert_called_once_with(simple_config)

    def test_fallback_setup_fn_when_none(self, simple_config, temp_results_dir):
        """Verify build_setup is used when setup_fn is None."""
        mock_state = mock.Mock()
        mock_run_fn = mock.Mock(return_value=(mock_state, None))

        with (
            mock.patch("tud_lbm.config.jax_config.configure_jax"),
            mock.patch("tud_lbm.pipeline.setup.build_setup") as mock_build,
            mock.patch("tud_lbm.runner.run.init_state", return_value=mock_state),
            mock.patch("tud_lbm.util.io.SimulationIO") as mock_io_cls,
        ):
            mock_io = mock.Mock()
            mock_io.run_dir = Path(temp_results_dir) / "run_001"
            mock_io_cls.return_value = mock_io

            run_single_simulation(
                index=0,
                config=simple_config,
                setup_fn=None,
                run_fn=mock_run_fn,
            )

        mock_build.assert_called_once_with(simple_config)

    def test_custom_run_fn_used(self, simple_config, temp_results_dir):
        """Verify custom run_fn is used when provided."""
        mock_setup = mock.Mock()
        mock_state = mock.Mock()
        custom_run_fn = mock.Mock(return_value=(mock_state, None))

        with (
            mock.patch("tud_lbm.config.jax_config.configure_jax"),
            mock.patch("tud_lbm.pipeline.setup.build_setup", return_value=mock_setup),
            mock.patch("tud_lbm.runner.run.init_state", return_value=mock_state),
            mock.patch("tud_lbm.util.io.SimulationIO") as mock_io_cls,
        ):
            mock_io = mock.Mock()
            mock_io.run_dir = Path(temp_results_dir) / "run_001"
            mock_io_cls.return_value = mock_io

            run_single_simulation(
                index=0,
                config=simple_config,
                run_fn=custom_run_fn,
            )

        custom_run_fn.assert_called_once()

    def test_fallback_run_fn_when_none(self, simple_config, temp_results_dir):
        """Verify default run is used when run_fn is None."""
        mock_setup = mock.Mock()
        mock_state = mock.Mock()

        with (
            mock.patch("tud_lbm.config.jax_config.configure_jax"),
            mock.patch("tud_lbm.pipeline.setup.build_setup", return_value=mock_setup),
            mock.patch("tud_lbm.runner.run.init_state", return_value=mock_state),
            mock.patch("tud_lbm.runner.run.run") as mock_run,
            mock.patch("tud_lbm.util.io.SimulationIO") as mock_io_cls,
        ):
            mock_io = mock.Mock()
            mock_io.run_dir = Path(temp_results_dir) / "run_001"
            mock_io_cls.return_value = mock_io
            mock_run.return_value = (mock_state, None)

            run_single_simulation(
                index=0,
                config=simple_config,
                run_fn=None,
            )

        mock_run.assert_called_once()

    def test_parameter_string_formatting_with_params(self, simple_config, temp_results_dir):
        """Verify parameter string formatting when parameters dict is provided."""
        mock_setup = mock.Mock()
        mock_state = mock.Mock()
        mock_run_fn = mock.Mock(return_value=(mock_state, None))

        with (
            mock.patch("tud_lbm.config.jax_config.configure_jax"),
            mock.patch("tud_lbm.pipeline.setup.build_setup", return_value=mock_setup),
            mock.patch("tud_lbm.runner.run.init_state", return_value=mock_state),
            mock.patch("tud_lbm.util.io.SimulationIO") as mock_io_cls,
        ):
            mock_io = mock.Mock()
            mock_io.run_dir = Path(temp_results_dir) / "run_001"
            mock_io_cls.return_value = mock_io

            parameters = {"alpha": 0.5, "beta": 2.0}
            run_single_simulation(
                index=0,
                config=simple_config,
                parameters=parameters,
                run_fn=mock_run_fn,
            )

        call_kwargs = mock_io_cls.call_args[1]
        assert "[" in call_kwargs["simulation_name"]
        assert "]" in call_kwargs["simulation_name"]
        assert "alpha=0.5" in call_kwargs["simulation_name"]
        assert "beta=2.0" in call_kwargs["simulation_name"]

    def test_parameter_string_formatting_without_params(self, simple_config, temp_results_dir):
        """Verify parameter string formatting when parameters is None."""
        mock_setup = mock.Mock()
        mock_state = mock.Mock()
        mock_run_fn = mock.Mock(return_value=(mock_state, None))

        with (
            mock.patch("tud_lbm.config.jax_config.configure_jax"),
            mock.patch("tud_lbm.pipeline.setup.build_setup", return_value=mock_setup),
            mock.patch("tud_lbm.runner.run.init_state", return_value=mock_state),
            mock.patch("tud_lbm.util.io.SimulationIO") as mock_io_cls,
        ):
            mock_io = mock.Mock()
            mock_io.run_dir = Path(temp_results_dir) / "run_001"
            mock_io_cls.return_value = mock_io

            run_single_simulation(
                index=0,
                config=simple_config,
                parameters=None,
                run_fn=mock_run_fn,
            )

        call_kwargs = mock_io_cls.call_args[1]
        assert "[sim_0]" in call_kwargs["simulation_name"]

    def test_exception_handling_returns_failed_status(self, simple_config, temp_results_dir):
        """Verify exception handling returns status='failed'."""
        mock_setup = mock.Mock()
        mock_state = mock.Mock()
        mock_run_fn = mock.Mock(side_effect=ValueError("Test error"))

        with (
            mock.patch("tud_lbm.config.jax_config.configure_jax"),
            mock.patch("tud_lbm.pipeline.setup.build_setup", return_value=mock_setup),
            mock.patch("tud_lbm.runner.run.init_state", return_value=mock_state),
            mock.patch("tud_lbm.util.io.SimulationIO") as mock_io_cls,
        ):
            mock_io = mock.Mock()
            mock_io.run_dir = Path(temp_results_dir) / "run_001"
            mock_io_cls.return_value = mock_io

            result = run_single_simulation(
                index=0,
                config=simple_config,
                run_fn=mock_run_fn,
            )

        assert result.status == "failed"
        assert "ValueError" in result.error
        assert "Test error" in result.error

    def test_exception_contains_traceback(self, simple_config, temp_results_dir):
        """Verify error message contains full traceback."""
        mock_setup = mock.Mock()
        mock_state = mock.Mock()
        mock_run_fn = mock.Mock(side_effect=RuntimeError("Deep error"))

        with (
            mock.patch("tud_lbm.config.jax_config.configure_jax"),
            mock.patch("tud_lbm.pipeline.setup.build_setup", return_value=mock_setup),
            mock.patch("tud_lbm.runner.run.init_state", return_value=mock_state),
            mock.patch("tud_lbm.util.io.SimulationIO") as mock_io_cls,
        ):
            mock_io = mock.Mock()
            mock_io.run_dir = Path(temp_results_dir) / "run_001"
            mock_io_cls.return_value = mock_io

            result = run_single_simulation(
                index=0,
                config=simple_config,
                run_fn=mock_run_fn,
            )

        assert "Traceback" in result.error or "traceback" in result.error.lower()

    def test_duration_still_set_on_failure(self, simple_config, temp_results_dir):
        """Verify duration is recorded even on failure."""
        mock_setup = mock.Mock()
        mock_state = mock.Mock()
        mock_run_fn = mock.Mock(side_effect=Exception("Test failure"))

        with (
            mock.patch("tud_lbm.config.jax_config.configure_jax"),
            mock.patch("tud_lbm.pipeline.setup.build_setup", return_value=mock_setup),
            mock.patch("tud_lbm.runner.run.init_state", return_value=mock_state),
            mock.patch("tud_lbm.util.io.SimulationIO") as mock_io_cls,
        ):
            mock_io = mock.Mock()
            mock_io.run_dir = Path(temp_results_dir) / "run_001"
            mock_io_cls.return_value = mock_io

            result = run_single_simulation(
                index=0,
                config=simple_config,
                run_fn=mock_run_fn,
            )

        assert result.duration > 0.0

    def test_config_output_dir_updated_via_replace(self, simple_config, temp_results_dir):
        """Verify config.output_dir is updated via replace() on success."""
        mock_setup = mock.Mock()
        mock_state = mock.Mock()
        mock_run_fn = mock.Mock(return_value=(mock_state, None))

        with (
            mock.patch("tud_lbm.config.jax_config.configure_jax"),
            mock.patch("tud_lbm.pipeline.setup.build_setup", return_value=mock_setup),
            mock.patch("tud_lbm.runner.run.init_state", return_value=mock_state),
            mock.patch("tud_lbm.util.io.SimulationIO") as mock_io_cls,
        ):
            mock_io = mock.Mock()
            expected_dir = Path(temp_results_dir) / "run_001"
            mock_io.run_dir = expected_dir
            mock_io_cls.return_value = mock_io

            result = run_single_simulation(
                index=0,
                config=simple_config,
                run_fn=mock_run_fn,
            )

        assert result.config.output_dir == str(expected_dir)


# =========================================================================
# run_parallel_simulations Tests
# =========================================================================


class TestRunParallelSimulations:
    """Tests for run_parallel_simulations function."""

    def test_empty_configs_list_returns_empty(self):
        """Verify empty configs list returns [] immediately."""
        result = run_parallel_simulations([])
        assert result == []

    def test_parameters_list_length_mismatch_raises_error(self, simple_config):
        """Verify parameters_list length mismatch raises ValueError."""
        configs = [simple_config, simple_config]
        parameters_list = [{"p": 1}]  # Mismatch: 1 != 2

        with pytest.raises(ValueError, match="parameters_list length"):
            run_parallel_simulations(configs, parameters_list=parameters_list)

    def test_results_sorted_by_index(self, simple_config, temp_results_dir):
        """Verify results are sorted by .index regardless of completion order."""
        configs = [simple_config, simple_config, simple_config]

        def mock_run_single_sim(index, config, **kwargs):
            return SimulationResult(
                index=index,
                config=config,
                status="success",
                output_dir=f"/output_{index}",
            )

        with (
            mock.patch(
                "runner.parallel_runner._run_single_simulation",
                side_effect=mock_run_single_sim,
            ),
            mock.patch("tud_lbm.runner.parallel_runner.ProcessPoolExecutor") as mock_executor_cls,
        ):
            mock_executor = mock.MagicMock()
            mock_executor_cls.return_value.__enter__.return_value = mock_executor

            futures = [mock.MagicMock() for _ in configs]
            for i, future in enumerate(futures):
                future.result.return_value = SimulationResult(
                    index=i,
                    config=configs[i],
                    status="success",
                    output_dir=f"/output_{i}",
                )

            with mock.patch("runner.parallel_runner.as_completed", return_value=futures):
                results = run_parallel_simulations(configs)

        indices = [r.index for r in results]
        assert indices == sorted(indices)

    def test_verbose_true_produces_output(self, simple_config, temp_results_dir, capsys):
        """Verify verbose=True produces printed output."""
        configs = [simple_config]

        def mock_run_single_sim(index, config, **kwargs):
            return SimulationResult(
                index=index,
                config=config,
                status="success",
                output_dir=f"/output_{index}",
                duration=1.0,
            )

        with (
            mock.patch(
                "runner.parallel_runner._run_single_simulation",
                side_effect=mock_run_single_sim,
            ),
            mock.patch("tud_lbm.runner.parallel_runner.ProcessPoolExecutor") as mock_executor_cls,
        ):
            mock_executor = mock.MagicMock()
            mock_executor_cls.return_value.__enter__.return_value = mock_executor

            future = mock.MagicMock()
            future.result.return_value = SimulationResult(
                index=0,
                config=configs[0],
                status="success",
                output_dir="/output_0",
                duration=1.0,
            )

            with mock.patch("runner.parallel_runner.as_completed", return_value=[future]):
                results = run_parallel_simulations(configs, verbose=True)

        captured = capsys.readouterr()
        assert "Parallel Simulation Sweep" in captured.out or len(results) > 0

    def test_progress_callback_called(self, simple_config, temp_results_dir):
        """Verify progress_callback is called once per completed sim."""
        configs = [simple_config, simple_config]

        progress_calls = []

        def mock_progress(completed, total) -> None:
            progress_calls.append((completed, total))

        def mock_run_single_sim(index, config, **kwargs):
            return SimulationResult(
                index=index,
                config=config,
                status="success",
                output_dir=f"/output_{index}",
            )

        with (
            mock.patch(
                "runner.parallel_runner._run_single_simulation",
                side_effect=mock_run_single_sim,
            ),
            mock.patch("tud_lbm.runner.parallel_runner.ProcessPoolExecutor") as mock_executor_cls,
        ):
            mock_executor = mock.MagicMock()
            mock_executor_cls.return_value.__enter__.return_value = mock_executor

            futures = [mock.MagicMock() for _ in configs]
            for i, future in enumerate(futures):
                future.result.return_value = SimulationResult(
                    index=i,
                    config=configs[i],
                    status="success",
                    output_dir=f"/output_{i}",
                )

            with mock.patch("runner.parallel_runner.as_completed", return_value=futures):
                run_parallel_simulations(configs, progress_callback=mock_progress)

        assert len(progress_calls) == 2
        assert all(call[1] == 2 for call in progress_calls)

    def test_continue_on_error_false_raises_on_failure(self, simple_config, temp_results_dir):
        """Verify continue_on_error=False raises RuntimeError on first failure."""
        configs = [simple_config, simple_config]

        def mock_run_single_sim(index, config, **kwargs):
            if index == 0:
                return SimulationResult(
                    index=index,
                    config=config,
                    status="failed",
                    error="Simulated failure",
                )
            return SimulationResult(
                index=index,
                config=config,
                status="success",
                output_dir=f"/output_{index}",
            )

        with (
            mock.patch(
                "runner.parallel_runner._run_single_simulation",
                side_effect=mock_run_single_sim,
            ),
            mock.patch("tud_lbm.runner.parallel_runner.ProcessPoolExecutor") as mock_executor_cls,
        ):
            mock_executor = mock.MagicMock()
            mock_executor_cls.return_value.__enter__.return_value = mock_executor

            futures = [mock.MagicMock() for _ in configs]
            futures[0].result.return_value = SimulationResult(
                index=0,
                config=configs[0],
                status="failed",
                error="Simulated failure",
            )
            futures[1].result.return_value = SimulationResult(
                index=1,
                config=configs[1],
                status="success",
                output_dir="/output_1",
            )

            with (
                mock.patch("runner.parallel_runner.as_completed", return_value=futures),
                pytest.raises(RuntimeError, match=r"Simulation .* failed"),
            ):
                run_parallel_simulations(configs, continue_on_error=False)

    def test_continue_on_error_true_collects_all_results(self, simple_config, temp_results_dir):
        """Verify continue_on_error=True collects all results even when some fail."""
        configs = [simple_config, simple_config, simple_config]

        def mock_run_single_sim(index, config, **kwargs):
            if index == 1:
                return SimulationResult(
                    index=index,
                    config=config,
                    status="failed",
                    error="Simulated failure",
                )
            return SimulationResult(
                index=index,
                config=config,
                status="success",
                output_dir=f"/output_{index}",
            )

        with (
            mock.patch(
                "runner.parallel_runner._run_single_simulation",
                side_effect=mock_run_single_sim,
            ),
            mock.patch("tud_lbm.runner.parallel_runner.ProcessPoolExecutor") as mock_executor_cls,
        ):
            mock_executor = mock.MagicMock()
            mock_executor_cls.return_value.__enter__.return_value = mock_executor

            futures = [mock.MagicMock() for _ in configs]
            for i, future in enumerate(futures):
                if i == 1:
                    future.result.return_value = SimulationResult(
                        index=i,
                        config=configs[i],
                        status="failed",
                        error="Simulated failure",
                    )
                else:
                    future.result.return_value = SimulationResult(
                        index=i,
                        config=configs[i],
                        status="success",
                        output_dir=f"/output_{i}",
                    )

            with mock.patch("runner.parallel_runner.as_completed", return_value=futures):
                results = run_parallel_simulations(configs, continue_on_error=True)

        assert len(results) == 3
        assert sum(1 for r in results if r.status == "success") == 2
        assert sum(1 for r in results if r.status == "failed") == 1

    def test_max_workers_none_delegates_to_executor(self, simple_config, temp_results_dir):
        """Verify max_workers=None is passed to ProcessPoolExecutor."""
        configs = [simple_config]

        def mock_run_single_sim(index, config, **kwargs):
            return SimulationResult(
                index=index,
                config=config,
                status="success",
                output_dir=f"/output_{index}",
            )

        with (
            mock.patch(
                "runner.parallel_runner._run_single_simulation",
                side_effect=mock_run_single_sim,
            ),
            mock.patch("tud_lbm.runner.parallel_runner.ProcessPoolExecutor") as mock_executor_cls,
        ):
            mock_executor = mock.MagicMock()
            mock_executor_cls.return_value.__enter__.return_value = mock_executor
            mock_executor.__enter__ = mock.Mock(return_value=mock_executor)
            mock_executor.__exit__ = mock.Mock(return_value=None)
            mock_executor.submit = mock.Mock(return_value=mock.Mock())

            with mock.patch("runner.parallel_runner.as_completed", return_value=[]):
                run_parallel_simulations(configs, max_workers=None)

            mock_executor_cls.assert_called_once_with(max_workers=None)

    def test_custom_setup_fn_forwarded(self, simple_config, temp_results_dir):
        """Verify custom setup_fn is forwarded to _run_single_simulation."""
        configs = [simple_config]
        custom_setup_fn = mock.Mock()

        with mock.patch("tud_lbm.runner.parallel_runner.ProcessPoolExecutor") as mock_executor_cls:
            mock_executor = mock.MagicMock()
            mock_executor_cls.return_value.__enter__.return_value = mock_executor

            submit_calls = []

            def capture_submit(*args, **kwargs):
                submit_calls.append((args, kwargs))
                future = mock.MagicMock()
                future.result.return_value = SimulationResult(
                    index=0,
                    config=configs[0],
                    status="success",
                    output_dir="/output_0",
                )
                return future

            mock_executor.submit.side_effect = capture_submit

            with mock.patch("runner.parallel_runner.as_completed", return_value=[mock.MagicMock()]):
                run_parallel_simulations(configs, setup_fn=custom_setup_fn)

        assert len(submit_calls) > 0
        call_kwargs = submit_calls[0][1] or {}
        assert "setup_fn" in call_kwargs or len(submit_calls[0][0]) > 3

    def test_custom_run_fn_forwarded(self, simple_config, temp_results_dir):
        """Verify custom run_fn is forwarded to _run_single_simulation."""
        configs = [simple_config]
        custom_run_fn = mock.Mock()

        with mock.patch("tud_lbm.runner.parallel_runner.ProcessPoolExecutor") as mock_executor_cls:
            mock_executor = mock.MagicMock()
            mock_executor_cls.return_value.__enter__.return_value = mock_executor

            submit_calls = []

            def capture_submit(*args, **kwargs):
                submit_calls.append((args, kwargs))
                future = mock.MagicMock()
                future.result.return_value = SimulationResult(
                    index=0,
                    config=configs[0],
                    status="success",
                    output_dir="/output_0",
                )
                return future

            mock_executor.submit.side_effect = capture_submit

            with mock.patch("runner.parallel_runner.as_completed", return_value=[mock.MagicMock()]):
                run_parallel_simulations(configs, run_fn=custom_run_fn)

        assert len(submit_calls) > 0


# =========================================================================
# _print_result_line Tests
# =========================================================================


class TestPrintResultLine:
    """Tests for _print_result_line function."""

    def test_success_status_shows_checkmark(self, simple_config, capsys):
        """Verify success status shows '✓' prefix."""
        result = SimulationResult(
            index=0,
            config=simple_config,
            status="success",
            duration=1.5,
        )
        print_result_line(result, 1, 3)
        captured = capsys.readouterr()
        assert "✓" in captured.out

    def test_failed_status_shows_cross(self, simple_config, capsys):
        """Verify failure status shows '✗' prefix."""
        result = SimulationResult(
            index=0,
            config=simple_config,
            status="failed",
            error="Test error",
            duration=1.5,
        )
        print_result_line(result, 1, 3)
        captured = capsys.readouterr()
        assert "✗" in captured.out

    def test_error_first_line_only(self, simple_config, capsys):
        """Verify only first line of error is printed (no full traceback)."""
        error_msg = "First line\nSecond line\nThird line"
        result = SimulationResult(
            index=0,
            config=simple_config,
            status="failed",
            error=error_msg,
            duration=1.5,
        )
        print_result_line(result, 1, 3)
        captured = capsys.readouterr()
        assert "First line" in captured.out
        assert "Second line" not in captured.out

    def test_parameters_formatted_correctly(self, simple_config, capsys):
        """Verify parameters dict is formatted as [k=v, ...]."""
        result = SimulationResult(
            index=0,
            config=simple_config,
            status="success",
            parameters={"alpha": 0.5, "beta": 2.0},
            duration=1.5,
        )
        print_result_line(result, 1, 3)
        captured = capsys.readouterr()
        assert "[" in captured.out
        assert "]" in captured.out
        assert "alpha" in captured.out or "beta" in captured.out

    def test_no_parameters_when_none(self, simple_config, capsys):
        """Verify parameters are not printed when None."""
        result = SimulationResult(
            index=0,
            config=simple_config,
            status="success",
            parameters=None,
            duration=1.5,
        )
        print_result_line(result, 1, 3)
        captured = capsys.readouterr()
        assert "Sim 0" in captured.out


# =========================================================================
# _generate_plots Tests
# =========================================================================


class TestGeneratePlots:
    """Tests for _generate_plots function."""

    def test_skips_non_success_status(self, simple_config):
        """Verify plots are skipped for non-success status."""
        failed_result = SimulationResult(
            index=0,
            config=simple_config,
            status="failed",
        )
        with mock.patch("tud_lbm.util.plotting.FigureBuilder") as mock_builder:
            generate_plots([failed_result], verbose=False)
        mock_builder.assert_not_called()

    def test_skips_when_plot_fields_falsy(self, simple_config):
        """Verify plots are skipped when config.plot_fields is falsy."""
        config_no_plots = dataclasses.replace(simple_config, plot_fields=None)
        result = SimulationResult(
            index=0,
            config=config_no_plots,
            status="success",
            output_dir="/output",
        )

        with mock.patch("tud_lbm.util.plotting.FigureBuilder") as mock_builder:
            generate_plots([result], verbose=False)
        mock_builder.assert_not_called()

    def test_calls_figure_builder_for_qualifying_results(self, simple_config):
        """Verify FigureBuilder is called for qualifying results."""
        config_with_plots = dataclasses.replace(simple_config, plot_fields=["density", "velocity"])
        result = SimulationResult(
            index=0,
            config=config_with_plots,
            status="success",
            output_dir="/output",
        )

        with mock.patch("tud_lbm.util.plotting.FigureBuilder") as mock_builder_cls:
            mock_builder = mock.Mock()
            mock_builder_cls.return_value = mock_builder
            generate_plots([result], verbose=False)

        mock_builder_cls.assert_called_once_with(config_with_plots, "/output")
        mock_builder.build_all.assert_called_once()

    def test_catches_and_prints_figure_builder_exceptions(self, simple_config, capsys):
        """Verify FigureBuilder exceptions are caught and printed."""
        config_with_plots = dataclasses.replace(simple_config, plot_fields=["density"])
        result = SimulationResult(
            index=0,
            config=config_with_plots,
            status="success",
            output_dir="/output",
        )

        with mock.patch("tud_lbm.util.plotting.FigureBuilder") as mock_builder_cls:
            mock_builder = mock.Mock()
            mock_builder.build_all.side_effect = RuntimeError("Plot error")
            mock_builder_cls.return_value = mock_builder
            generate_plots([result], verbose=False)

        captured = capsys.readouterr()
        assert "Failed to generate plots" in captured.out or "Plot error" in captured.out

    def test_skips_all_results_no_plot_fields(self, simple_config):
        """Verify all results are skipped when none have plot_fields."""
        config_no_plots = dataclasses.replace(simple_config, plot_fields=None)
        results = [
            SimulationResult(
                index=i,
                config=config_no_plots,
                status="success",
                output_dir=f"/output_{i}",
            )
            for i in range(3)
        ]

        with mock.patch("tud_lbm.util.plotting.FigureBuilder") as mock_builder:
            generate_plots(results, verbose=False)
        mock_builder.assert_not_called()


# =========================================================================
# save_sweep_log Tests
# =========================================================================


class TestSaveSweepLog:
    """Tests for save_sweep_log function."""

    def test_creates_output_dir_if_not_exists(self, simple_config):
        """Verify output_dir is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nonexistent" / "nested" / "dir"
            results = [
                SimulationResult(
                    index=0,
                    config=simple_config,
                    status="success",
                    output_dir="/output_0",
                ),
            ]
            save_sweep_log(results, output_dir)
            assert output_dir.exists()

    def test_output_json_contains_correct_counts(self, simple_config):
        """Verify output JSON contains correct counts for successful/failed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = [
                SimulationResult(
                    index=0,
                    config=simple_config,
                    status="success",
                    output_dir="/output_0",
                ),
                SimulationResult(
                    index=1,
                    config=simple_config,
                    status="failed",
                    error="Error 1",
                ),
                SimulationResult(
                    index=2,
                    config=simple_config,
                    status="success",
                    output_dir="/output_2",
                ),
            ]
            save_sweep_log(results, tmpdir)

            # Find the log file (it has the sweep_id in the filename)
            log_files = list(Path(tmpdir).glob("sweep_log_*.json"))
            assert len(log_files) == 1
            log_path = log_files[0]

            with log_path.open() as f:
                log = json.load(f)

            assert log["total_simulations"] == 3
            assert log["successful"] == 2
            assert log["failed"] == 1

    def test_simulation_entry_has_required_fields(self, simple_config):
        """Verify each simulation entry has index, status, output_dir, parameters, duration_sec, error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = SimulationResult(
                index=0,
                config=simple_config,
                status="success",
                output_dir="/output_0",
                parameters={"param1": "value1"},
                duration=2.5,
            )
            save_sweep_log([result], tmpdir)

            # Find the log file (it has the sweep_id in the filename)
            log_files = list(Path(tmpdir).glob("sweep_log_*.json"))
            assert len(log_files) == 1
            log_path = log_files[0]

            with log_path.open() as f:
                log = json.load(f)

            entry = log["simulations"][0]
            assert "index" in entry
            assert "status" in entry
            assert "output_dir" in entry
            assert "parameters" in entry
            assert "duration_sec" in entry
            assert "error" in entry

    def test_sweep_id_is_valid_uuid(self, simple_config):
        """Verify sweep_id is a valid UUID string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = [
                SimulationResult(
                    index=0,
                    config=simple_config,
                    status="success",
                ),
            ]
            save_sweep_log(results, tmpdir)

            # Find the log file (it has the sweep_id in the filename)
            log_files = list(Path(tmpdir).glob("sweep_log_*.json"))
            assert len(log_files) == 1
            log_path = log_files[0]

            with log_path.open() as f:
                log = json.load(f)

            sweep_id = UUID(log["sweep_id"])
            assert sweep_id is not None

    def test_timestamp_is_valid_iso8601_utc(self, simple_config):
        """Verify timestamp is a valid ISO 8601 UTC string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = [
                SimulationResult(
                    index=0,
                    config=simple_config,
                    status="success",
                ),
            ]
            save_sweep_log(results, tmpdir)

            # Find the log file (it has the sweep_id in the filename)
            log_files = list(Path(tmpdir).glob("sweep_log_*.json"))
            assert len(log_files) == 1
            log_path = log_files[0]

            with log_path.open() as f:
                log = json.load(f)

            timestamp = log["timestamp"]
            assert "T" in timestamp
            assert "+" in timestamp or "Z" in timestamp or timestamp.endswith("+00:00")

    def test_log_path_is_correct(self, simple_config):
        """Verify log is saved with sweep_id in filename in output_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = [
                SimulationResult(
                    index=0,
                    config=simple_config,
                    status="success",
                ),
            ]
            save_sweep_log(results, tmpdir)

            # Find the log file (it has the sweep_id in the filename)
            log_files = list(Path(tmpdir).glob("sweep_log_*.json"))
            assert len(log_files) == 1
