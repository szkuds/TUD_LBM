"""How a multi-run command names its runs: a directory or a ``.txt`` list."""

from __future__ import annotations
import sys
from typing import TYPE_CHECKING
import click
import pytest
from src.cli.analysis_routing import parse_run_dir_list
from src.cli.analysis_routing import resolve_run_set

if TYPE_CHECKING:
    from pathlib import Path

_no_literal_backslash_dirs = pytest.mark.skipif(
    sys.platform == "win32",
    reason="directory names containing a literal backslash are not representable "
    "as a single path component on Windows (backslash is always the path separator there)",
)


def _make_run(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.toml").write_text("[simulation_type]\n", encoding="utf-8")
    return path


def test_a_directory_gives_every_run_beneath_it(tmp_path: Path):
    run_a = _make_run(tmp_path / "2026-11-01" / "run_a")
    run_b = _make_run(tmp_path / "2026-11-02" / "run_b")
    _make_run(tmp_path / "init" / "seed")
    _make_run(tmp_path / "comparison_analysis" / "stale")

    run_set = resolve_run_set(tmp_path)

    assert run_set.run_dirs == (run_a, run_b)
    assert run_set.out_root == tmp_path


def test_a_txt_list_gives_its_runs_and_outputs_beside_the_list(tmp_path: Path):
    run_b = _make_run(tmp_path / "runs" / "run_b")
    run_a = _make_run(tmp_path / "runs" / "run_a")
    lists = tmp_path / "lists"
    lists.mkdir()
    txt_path = lists / "set.txt"
    txt_path.write_text(f"{run_b.as_posix()}\n{run_a.as_posix()}\n", encoding="utf-8")

    run_set = resolve_run_set(txt_path, allowed_roots=[tmp_path])

    # The list's order is kept: it is the order the user chose.
    assert run_set.run_dirs == (run_b.resolve(), run_a.resolve())
    assert run_set.out_root == lists.resolve()


def test_a_directory_and_its_txt_list_give_the_same_runs(tmp_path: Path):
    run_a = _make_run(tmp_path / "run_a")
    run_b = _make_run(tmp_path / "run_b")
    txt_path = tmp_path / "set.txt"
    txt_path.write_text("run_a\nrun_b\n", encoding="utf-8")

    from_dir = resolve_run_set(tmp_path)
    from_txt = resolve_run_set(txt_path, allowed_roots=[tmp_path])

    assert [p.resolve() for p in from_dir.run_dirs] == list(from_txt.run_dirs) == [run_a.resolve(), run_b.resolve()]


def test_any_other_file_is_a_usage_error(tmp_path: Path):
    other = tmp_path / "config.toml"
    other.write_text("", encoding="utf-8")

    with pytest.raises(click.UsageError, match=r"directory of runs or a \.txt list"):
        resolve_run_set(other)


def test_parse_run_dir_list_skips_blank_and_comment_lines(tmp_path: Path):
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_b").mkdir()
    txt_path = tmp_path / "dirs.txt"
    txt_path.write_text("# header comment\n\nrun_a\nrun_b\n", encoding="utf-8")

    dirs = parse_run_dir_list(txt_path, allowed_roots=[tmp_path])

    assert dirs == [tmp_path / "run_a", tmp_path / "run_b"]


@_no_literal_backslash_dirs
def test_parse_run_dir_list_unescapes_single_quoted_special_chars(tmp_path: Path):
    run_name = r"22-12-23_$Bo_\parallel = 0.60; Oh = 0.47$"
    (tmp_path / run_name).mkdir()
    txt_path = tmp_path / "dirs.txt"
    quoted_line = r"'" + tmp_path.as_posix() + r"/22-12-23_\$Bo_\\parallel\ \=\ 0.60\;\ Oh\ \=\ 0.47\$'"
    txt_path.write_text(quoted_line + "\n", encoding="utf-8")

    dirs = parse_run_dir_list(txt_path, allowed_roots=[tmp_path])

    assert dirs == [tmp_path / run_name]


@_no_literal_backslash_dirs
def test_parse_run_dir_list_unescapes_unquoted_special_chars(tmp_path: Path):
    run_name = r"08-13-44_$Bo_\parallel = 0.80; Oh = 0.45$"
    (tmp_path / run_name).mkdir()
    txt_path = tmp_path / "dirs.txt"
    escaped_line = tmp_path.as_posix() + r"/08-13-44_\$Bo_\\parallel\ \=\ 0.80\;\ Oh\ \=\ 0.45\$"
    txt_path.write_text(escaped_line + "\n", encoding="utf-8")

    dirs = parse_run_dir_list(txt_path, allowed_roots=[tmp_path])

    assert dirs == [tmp_path / run_name]


@_no_literal_backslash_dirs
def test_parse_run_dir_list_keeps_unquoted_spaces_in_one_line(tmp_path: Path):
    run_name = r"08-13-44_$Bo_\parallel = 0.60; Oh = 0.30$"
    (tmp_path / run_name).mkdir()
    other_name = r"08-13-44_$Bo_\parallel = 0.80; Oh = 0.30$"
    (tmp_path / other_name).mkdir()
    txt_path = tmp_path / "dirs.txt"
    txt_path.write_text(f"{tmp_path.as_posix()}/{run_name}\n{tmp_path.as_posix()}/{other_name}\n", encoding="utf-8")

    dirs = parse_run_dir_list(txt_path, allowed_roots=[tmp_path])

    assert dirs == [tmp_path / run_name, tmp_path / other_name]


def test_parse_run_dir_list_strips_double_quotes(tmp_path: Path):
    (tmp_path / "run_a").mkdir()
    txt_path = tmp_path / "dirs.txt"
    txt_path.write_text(f'"{tmp_path.as_posix()}/run_a"\n', encoding="utf-8")

    dirs = parse_run_dir_list(txt_path, allowed_roots=[tmp_path])

    assert dirs == [tmp_path / "run_a"]


def test_parse_run_dir_list_rejects_relative_traversal_outside_parent(tmp_path: Path):
    outside = tmp_path.parent / "outside_secret"
    outside.mkdir(exist_ok=True)
    list_dir = tmp_path / "lists"
    list_dir.mkdir()
    txt_path = list_dir / "dirs.txt"
    txt_path.write_text("../../outside_secret\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside every allowed root"):
        parse_run_dir_list(txt_path, allowed_roots=[list_dir])


def test_parse_run_dir_list_allows_absolute_path_within_allowed_root(tmp_path: Path):
    outside = tmp_path.parent / "outside_run"
    outside.mkdir(exist_ok=True)
    list_dir = tmp_path / "lists"
    list_dir.mkdir()
    txt_path = list_dir / "dirs.txt"
    txt_path.write_text(f"{outside.as_posix()}\n", encoding="utf-8")

    dirs = parse_run_dir_list(txt_path, allowed_roots=[outside.parent])

    assert dirs == [outside]


def test_parse_run_dir_list_rejects_absolute_path_outside_allowed_roots(tmp_path: Path):
    outside = tmp_path.parent / "outside_run"
    outside.mkdir(exist_ok=True)
    list_dir = tmp_path / "lists"
    list_dir.mkdir()
    txt_path = list_dir / "dirs.txt"
    txt_path.write_text(f"{outside.as_posix()}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside every allowed root"):
        parse_run_dir_list(txt_path, allowed_roots=[list_dir])


def test_parse_run_dir_list_accepts_default_results_root_with_no_allowed_roots(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.cli.analysis_routing.BASE_RESULTS_DIR", str(tmp_path))
    (tmp_path / "run_a").mkdir()
    txt_path = tmp_path / "dirs.txt"
    txt_path.write_text("run_a\n", encoding="utf-8")

    dirs = parse_run_dir_list(txt_path)

    assert dirs == [tmp_path / "run_a"]
