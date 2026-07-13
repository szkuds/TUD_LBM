"""HTML report adapter for completed simulation runs.

Generates ``report.html`` inside the run directory by injecting simulation
data (params, base64 frames, comments) into a standalone HTML/CSS/JS
template that lives in ``assets/template.html``.

The template is a real, browser-openable file — edit it independently to
change layout, colours, or behaviour without touching Python.

Usage::

    from tud_lbm.io.report import HtmlReport

    report = HtmlReport(config=config, run_dir=io.run_dir)
    path = report.build()   # -> run_dir/report.html
"""

from __future__ import annotations
import base64
import json
import re
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tud_lbm.config import SimulationConfig

# Path to the HTML/CSS/JS template shipped with this package.
_TEMPLATE_PATH = Path(__file__).parent / "assets" / "template.html"

# Regex that matches the <script id="report-data"> … </script> injection point.
_DATA_TAG_RE = re.compile(
    r'<script\s+id="report-data"[^>]*>[\s\S]*?</script>',
    re.IGNORECASE,
)


class HtmlReport:
    """Generate a self-contained HTML report for a completed simulation run.

    Parameters
    ----------
    config:
        The :class:`~tud_lbm.config.SimulationConfig` used for the run.
    run_dir:
        Root directory of the run (contains ``data/``, ``plots/``, …).
    author:
        Default author label pre-filled in the comment box.
    """

    def __init__(
        self,
        config: SimulationConfig,
        run_dir: str | Path,
        author: str = "Researcher",
    ) -> None:
        """Initialise the report generator."""
        self.config = config
        self.run_dir = Path(run_dir)
        self.author = author

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> Path:
        """Build the report and write it to ``<run_dir>/report.html``.

        Returns:
        -------
        Path
            Absolute path to the written HTML file.
        """
        payload = self._build_payload()
        html = _inject_data(_TEMPLATE_PATH.read_text(encoding="utf-8"), payload)
        out = self.run_dir / "report.html"
        out.write_text(html, encoding="utf-8")
        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_payload(self) -> dict:
        """Assemble the JSON data payload for the template."""
        return {
            "sim_name": self.config.simulation_name or "Simulation Report",
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "params": self._extract_params(),
            "frames": self._load_frames(),
            "comments": [],
        }

    def _extract_params(self) -> list[tuple[str, str]]:
        """Return relevant config fields as [label, value] pairs."""
        cfg = self.config
        rows: list[tuple[str, str]] = [
            ("Simulation Name", cfg.simulation_name or "—"),
            ("Type", cfg.sim_type),
            ("Grid Shape", str(cfg.grid_shape)),
            ("Lattice", cfg.lattice_type),
            ("Relaxation Time (τ)", str(cfg.tau)),
            ("Time Steps (nt)", str(cfg.nt)),
            ("Save Interval", str(cfg.save_interval)),
            ("Collision Scheme", cfg.collision_scheme),
            ("Init Type", cfg.init_type),
        ]
        if cfg.plot_fields:
            rows.append(("Plot Fields", ", ".join(cfg.plot_fields)))
        if getattr(cfg, "is_multiphase", False):
            rows += [
                ("EOS", str(cfg.eos)),
                ("κ (kappa)", str(cfg.kappa)),
                ("ρ_liquid", str(cfg.rho_l)),
                ("ρ_vapour", str(cfg.rho_v)),
                ("Interface Width", str(cfg.interface_width)),
            ]
        for force_field in ("gravity_force", "gravity_masked_force", "electric_force"):
            force_cfg = getattr(cfg, force_field, None)
            if force_cfg:
                for k, v in force_cfg.items():
                    rows.append((f"{force_field}.{k}", str(v)))
        if cfg.wetting_config:
            for k, v in cfg.wetting_config.items():
                rows.append((f"wetting.{k}", str(v)))
        if cfg.hysteresis_config:
            for k, v in cfg.hysteresis_config.items():
                rows.append((f"hysteresis.{k}", str(v)))
        return rows

    def _load_frames(self) -> list[str]:
        """Return base64 data-URIs for every PNG in ``plots/``."""
        plots_dir = self.run_dir / "plots"
        if not plots_dir.exists():
            return []
        pngs = sorted(plots_dir.glob("*.png"), key=lambda p: p.stem)
        return ["data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii") for p in pngs]


# ---------------------------------------------------------------------------
# Module-level helper — injectable and independently testable
# ---------------------------------------------------------------------------


def _inject_data(template: str, payload: dict) -> str:
    """Replace the ``<script id="report-data">`` block in *template* with *payload*.

    Parameters
    ----------
    template:
        Raw HTML string (contents of ``assets/template.html``).
    payload:
        Dict with keys ``sim_name``, ``generated_at``, ``params``,
        ``frames``, ``comments``.

    Returns:
    -------
    str
        HTML string with the data block replaced.

    Raises:
    ------
    ValueError
        If the injection point is not found in *template*.
    """
    replacement = (
        '<script id="report-data" type="application/json">\n'
        + json.dumps(payload, indent=2, ensure_ascii=False)
        + "\n</script>"
    )
    result, n = _DATA_TAG_RE.subn(replacement, template, count=1)
    if n == 0:
        msg = f'Could not find the <script id="report-data"> injection point in template: {_TEMPLATE_PATH}'
        raise ValueError(msg)
    return result
