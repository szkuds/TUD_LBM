"""Tests for Animator.create edge paths."""

from __future__ import annotations
import builtins
import sys
import types
import pytest
from PIL import Image
from src.config import SimulationConfig
from src.simulation_io.plotting.animator import Animator


def _animator(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "data").mkdir(parents=True)
    cfg = SimulationConfig(plot_fields=[])
    return Animator(config=cfg, run_dir=run_dir)


def test_create_raises_when_no_frames(tmp_path, monkeypatch):
    animator = _animator(tmp_path)
    monkeypatch.setattr(animator, "build_frames", list)

    with pytest.raises(FileNotFoundError, match="No snapshot files"):
        animator.create()


def test_create_wraps_moviepy_import_error(tmp_path, monkeypatch):
    animator = _animator(tmp_path)
    frame = tmp_path / "f.png"
    frame.write_bytes(b"x")
    monkeypatch.setattr(animator, "build_frames", lambda: [frame])

    orig_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("moviepy"):
            msg = "missing moviepy"
            raise ImportError(msg)
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ImportError, match="moviepy is required"):
        animator.create()


def test_create_writes_looping_gif_with_every_frame(tmp_path, monkeypatch):
    animator = _animator(tmp_path)
    frames = []
    for idx, colour in enumerate(("red", "blue", "green")):
        frame = tmp_path / f"f{idx}.png"
        Image.new("RGB", (16, 8), colour).save(frame)
        frames.append(frame)
    monkeypatch.setattr(animator, "build_frames", lambda: frames)

    out = animator.create(tmp_path / "anim.gif")

    with Image.open(out) as gif:
        assert gif.n_frames == len(frames)
        assert gif.info["loop"] == 0
        assert gif.info["duration"] == round(1000 / animator.fps)


def test_create_writes_mp4_with_fake_moviepy(tmp_path, monkeypatch):
    animator = _animator(tmp_path)
    frame = tmp_path / "f.png"
    frame.write_bytes(b"x")
    monkeypatch.setattr(animator, "build_frames", lambda: [frame])

    calls = {"mp4": 0, "close": 0}

    class _FakeClip:
        def __init__(self, seq, fps):
            self.seq = seq
            self.fps = fps

        def write_videofile(self, output, fps, audio, logger):
            calls["mp4"] += 1

        def close(self):
            calls["close"] += 1

    module = types.ModuleType("moviepy.video.io.ImageSequenceClip")
    module.ImageSequenceClip = _FakeClip  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "moviepy", types.ModuleType("moviepy"))
    monkeypatch.setitem(sys.modules, "moviepy.video", types.ModuleType("moviepy.video"))
    monkeypatch.setitem(sys.modules, "moviepy.video.io", types.ModuleType("moviepy.video.io"))
    monkeypatch.setitem(sys.modules, "moviepy.video.io.ImageSequenceClip", module)

    out = animator.create(tmp_path / "anim.mp4")

    assert out.suffix == ".mp4"
    assert calls["mp4"] == 1
    assert calls["close"] == 1
