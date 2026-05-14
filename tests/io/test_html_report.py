"""Unit tests for HtmlReport and _inject_data."""

from __future__ import annotations
import json
from typing import TYPE_CHECKING
import pytest
from tud_lbm.config import SimulationConfig
from tud_lbm.io.report.html_report import HtmlReport
from tud_lbm.io.report.html_report import _inject_data

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_config() -> SimulationConfig:
    """Minimal single-phase SimulationConfig for testing."""
    return SimulationConfig(
        simulation_name="Test Run",
        grid_shape=(32, 32),
        tau=0.8,
        nt=100,
        save_interval=10,
        plot_fields=["density", "velocity"],
    )


@pytest.fixture
def run_dir_with_plots(simple_config: SimulationConfig, tmp_path: Path) -> Path:
    """A temporary run directory with two stub PNG frames in plots/."""
    plots = tmp_path / "plots"
    plots.mkdir()
    # Minimal 1-pixel red PNG (valid PNG bytes)
    _minimal_png = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
        b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    (plots / "timestep_000010.png").write_bytes(_minimal_png)
    (plots / "timestep_000020.png").write_bytes(_minimal_png)
    return tmp_path


# ---------------------------------------------------------------------------
# _inject_data
# ---------------------------------------------------------------------------


class TestInjectData:
    """Tests for the module-level _inject_data helper."""

    def _make_template(self, body: str = "") -> str:
        return (
            "<html><head></head><body>"
            '<script id="report-data" type="application/json">\n'
            '{"sim_name":"old","frames":[],"params":[],"comments":[]}\n'
            "</script>" + body + "</body></html>"
        )

    def test_replaces_data_block(self):
        payload = {"sim_name": "New Run", "frames": [], "params": [], "comments": []}
        result = _inject_data(self._make_template(), payload)
        data = _parse_injected_json(result)
        assert data["sim_name"] == "New Run"

    def test_preserves_html_outside_script_block(self):
        template = self._make_template(body="<p id='sentinel'>keep me</p>")
        result = _inject_data(template, {"sim_name": "x", "frames": [], "params": [], "comments": []})
        assert "sentinel" in result

    def test_raises_on_missing_injection_point(self):
        with pytest.raises(ValueError, match="report-data"):
            _inject_data("<html><body>no tag here</body></html>", {})

    def test_payload_is_valid_json_in_output(self):
        payload = {
            "sim_name": "Run A",
            "generated_at": "2026-01-01 00:00 UTC",
            "params": [["τ", "0.8"], ["nt", "100"]],
            "frames": [],
            "comments": [],
        }
        result = _inject_data(self._make_template(), payload)
        parsed = _parse_injected_json(result)
        assert parsed["params"][0] == ["τ", "0.8"]


class TestHtmlReportBuild:
    """Integration-style tests for HtmlReport.build()."""

    def test_creates_report_html(self, simple_config: SimulationConfig, tmp_path: Path):
        report = HtmlReport(config=simple_config, run_dir=tmp_path)
        path = report.build()
        assert path == tmp_path / "report.html"
        assert path.exists()

    def test_html_is_not_empty(self, simple_config: SimulationConfig, tmp_path: Path):
        path = HtmlReport(config=simple_config, run_dir=tmp_path).build()
        content = path.read_text(encoding="utf-8")
        assert len(content) > 500

    def test_sim_name_in_output(self, simple_config: SimulationConfig, tmp_path: Path):
        path = HtmlReport(config=simple_config, run_dir=tmp_path).build()
        data = _parse_injected_json(path.read_text(encoding="utf-8"))
        assert data["sim_name"] == "Test Run"

    def test_params_contain_grid_shape(self, simple_config: SimulationConfig, tmp_path: Path):
        path = HtmlReport(config=simple_config, run_dir=tmp_path).build()
        data = _parse_injected_json(path.read_text(encoding="utf-8"))
        labels = [row[0] for row in data["params"]]
        assert "Grid Shape" in labels

    def test_params_do_not_contain_results_directory(self, simple_config: SimulationConfig, tmp_path: Path):
        path = HtmlReport(config=simple_config, run_dir=tmp_path).build()
        data = _parse_injected_json(path.read_text(encoding="utf-8"))
        labels = [row[0] for row in data["params"]]
        assert "Results Directory" not in labels

    def test_frames_embedded_for_png_files(self, simple_config: SimulationConfig, run_dir_with_plots: Path):
        path = HtmlReport(config=simple_config, run_dir=run_dir_with_plots).build()
        data = _parse_injected_json(path.read_text(encoding="utf-8"))
        assert len(data["frames"]) == 2
        assert data["frames"][0].startswith("data:image/png;base64,")

    def test_no_frames_when_plots_dir_absent(self, simple_config: SimulationConfig, tmp_path: Path):
        path = HtmlReport(config=simple_config, run_dir=tmp_path).build()
        data = _parse_injected_json(path.read_text(encoding="utf-8"))
        assert data["frames"] == []

    def test_comments_initialised_empty(self, simple_config: SimulationConfig, tmp_path: Path):
        path = HtmlReport(config=simple_config, run_dir=tmp_path).build()
        data = _parse_injected_json(path.read_text(encoding="utf-8"))
        assert data["comments"] == []

    def test_generated_at_field_present(self, simple_config: SimulationConfig, tmp_path: Path):
        path = HtmlReport(config=simple_config, run_dir=tmp_path).build()
        data = _parse_injected_json(path.read_text(encoding="utf-8"))
        assert "generated_at" in data
        assert "UTC" in data["generated_at"]

    def test_frames_sorted_by_filename(self, simple_config: SimulationConfig, run_dir_with_plots: Path):
        path = HtmlReport(config=simple_config, run_dir=run_dir_with_plots).build()
        data = _parse_injected_json(path.read_text(encoding="utf-8"))
        # Both frames should be present and in filename order (10 before 20)
        assert len(data["frames"]) == 2


# ---------------------------------------------------------------------------
# HtmlReport._extract_params
# ---------------------------------------------------------------------------


class TestExtractParams:
    """Unit tests for param extraction logic."""

    def test_no_results_directory_in_params(self, simple_config: SimulationConfig, tmp_path: Path):
        report = HtmlReport(config=simple_config, run_dir=tmp_path)
        labels = [row[0] for row in report._extract_params()]
        assert "Results Directory" not in labels

    def test_plot_fields_included_when_set(self, simple_config: SimulationConfig, tmp_path: Path):
        report = HtmlReport(config=simple_config, run_dir=tmp_path)
        labels = [row[0] for row in report._extract_params()]
        assert "Plot Fields" in labels

    def test_plot_fields_absent_when_none(self, tmp_path: Path):
        cfg = SimulationConfig(grid_shape=(32, 32), tau=0.8, nt=100)
        report = HtmlReport(config=cfg, run_dir=tmp_path)
        labels = [row[0] for row in report._extract_params()]
        assert "Plot Fields" not in labels

    def test_wetting_fields_included_for_multiphase_wetting(self, tmp_path: Path):
        cfg = SimulationConfig(
            sim_type="multiphase_wetting",
            grid_shape=(32, 32),
            tau=0.9,
            nt=100,
            eos="double-well",
            kappa=0.04,
            rho_l=1.0,
            rho_v=0.001,
            interface_width=5,
            wetting_config={"phi_left": 1.2, "phi_right": 1.2},
        )
        report = HtmlReport(config=cfg, run_dir=tmp_path)
        labels = [row[0] for row in report._extract_params()]
        assert "wetting.phi_left" in labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_injected_json(html: str) -> dict:
    """Extract and parse the JSON from the <script id="report-data"> block."""
    import re

    match = re.search(
        r'<script\s+id="report-data"[^>]*>([\s\S]*?)</script>',
        html,
        re.IGNORECASE,
    )
    assert match, "Could not find report-data script tag in HTML"
    return json.loads(match.group(1))
