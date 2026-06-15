"""Tests for Animator.create edge paths."""

from __future__ import annotations
import builtins
import sys
import types
import pytest
from tud_lbm.config import SimulationConfig
from tud_lbm.io.plotting.animator import Animator


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


def test_create_writes_gif_with_fake_moviepy(tmp_path, monkeypatch):
    animator = _animator(tmp_path)
    frame = tmp_path / "f.png"
    frame.write_bytes(b"x")
    monkeypatch.setattr(animator, "build_frames", lambda: [frame])

    calls = {"gif": 0, "close": 0}

    class _FakeClip:
        def __init__(self, seq, fps):
            self.seq = seq
            self.fps = fps

        def write_gif(self, output, fps):
            calls["gif"] += 1

        def write_videofile(self, output, fps, audio, logger):
            msg = "mp4 path should not be used"
            raise AssertionError(msg)

        def close(self):
            calls["close"] += 1

    module = types.ModuleType("moviepy.video.io.ImageSequenceClip")
    module.ImageSequenceClip = _FakeClip  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "moviepy", types.ModuleType("moviepy"))
    monkeypatch.setitem(sys.modules, "moviepy.video", types.ModuleType("moviepy.video"))
    monkeypatch.setitem(sys.modules, "moviepy.video.io", types.ModuleType("moviepy.video.io"))
    monkeypatch.setitem(sys.modules, "moviepy.video.io.ImageSequenceClip", module)

    out = animator.create(tmp_path / "anim.gif")

    assert out.suffix == ".gif"
    assert calls["gif"] == 1
    assert calls["close"] == 1
