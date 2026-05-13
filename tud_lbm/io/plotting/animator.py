"""Animation builder for field and analysis plotting."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from tud_lbm.io.plotting.figure_builder import FigureBuilder

if TYPE_CHECKING:
    from tud_lbm.config import SimulationConfig


class Animator:
    """Create frame-by-frame animations from saved snapshots."""

    def __init__(
        self,
        config: SimulationConfig,
        run_dir: str | Path,
        fps: int = 10,
        dpi: int = 150,
    ) -> None:
        """Initialise animation settings for a simulation run directory."""
        fields = config.animate_fields or config.plot_fields
        self.fps = fps
        self.dpi = dpi
        self.builder = FigureBuilder(config=config, run_dir=run_dir, dpi=dpi, fields=fields)
        self._frames_dir = self.builder.plot_dir / "frames"

    def build_frames(self) -> list[Path]:
        """Build one composite frame per timestep file."""
        timed_files = self.builder.sorted_timed_files()
        if not timed_files:
            return []
        files = [fp for _, fp in timed_files]

        self._frames_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: list[Path] = []

        for idx, (timestep, fp) in enumerate(timed_files):
            with np.load(fp) as raw:
                data = {key: raw[key] for key in raw.files}

            history = files[: idx + 1]
            fig = self.builder.render_figure(data, timestep, history_files=history)
            if fig is None:
                continue

            out_path = self._frames_dir / f"frame_{idx:06d}.png"
            fig.savefig(out_path, dpi=self.dpi)
            plt.close(fig)
            frame_paths.append(out_path)

        return frame_paths

    def create(self, output_path: str | Path | None = None) -> Path:
        """Create an mp4/gif animation from generated frames."""
        frame_paths = self.build_frames()
        if not frame_paths:
            msg = f"No snapshot files found under {self.builder.data_dir}"
            raise FileNotFoundError(msg)

        output = Path(output_path) if output_path is not None else self.builder.plot_dir / "animation.mp4"

        try:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        except ImportError as exc:
            msg = "moviepy is required to encode animation files. Install it with: pip install moviepy"
            raise ImportError(msg) from exc

        clip = ImageSequenceClip([str(p) for p in frame_paths], fps=self.fps)
        if output.suffix.lower() == ".gif":
            clip.write_gif(str(output), fps=self.fps)
        else:
            clip.write_videofile(str(output), fps=self.fps, audio=False, logger=None)

        if hasattr(clip, "close"):
            clip.close()
        return output
