"""Routing between the CLI's analysis entry points and the metric layer.

Every command that analyses saved snapshots — ``visualise analysis``,
``compare runs``, ``compare regime-map`` and ``run --compare`` — goes through
:func:`analyse_run` or :func:`analyse_runs`. Both share one
:class:`~src.simulation_io.analysis.droplet_metrics.DropletSeries` per run, so a
run's ``.npz`` files are read once regardless of how many analysis figures are
requested.

A multi-run command names its runs with one argument, a directory or a ``.txt``
list of run directories; :func:`resolve_run_set` turns either into a
:class:`RunSet`.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
import click
from src.config.config_overview import BASE_RESULTS_DIR
from src.config.run_config import COMPARISON_DIRNAME
from src.config.run_config import CONFIG_FILENAME
from src.simulation_io.plotting._simulation_csv import build_simulation_csv
from src.simulation_io.plotting.run_comparison import _safe_load_config
from src.simulation_io.plotting.run_comparison import compare_runs

if TYPE_CHECKING:
    from collections.abc import Sequence
    from src.config import SimulationConfig

#: Path fragments marking directories that are not simulation runs.
_SKIP_DIRS = ("init", COMPARISON_DIRNAME)

#: Suffix of a run-directory list file.
RUN_LIST_SUFFIX = ".txt"

# A quoted line needs its two quote characters.
_MIN_QUOTED_LINE_LEN = 2


@dataclass(frozen=True)
class RunSet:
    """The runs one multi-run command works on, and where its output goes by default.

    Attributes:
        run_dirs: The run directories, in a stable order.
        out_root: The directory a default output folder is made in: the
            directory itself, or the directory holding the list file.
    """

    run_dirs: tuple[Path, ...]
    out_root: Path


def resolve_run_set(target: str | Path, allowed_roots: Sequence[str | Path] = ()) -> RunSet:
    """Resolve a directory or a ``.txt`` run list into a :class:`RunSet`.

    A directory gives every run directory beneath it (:func:`find_run_dirs`). A
    ``.txt`` file gives the run directories it lists (:func:`parse_run_dir_list`,
    which checks each entry against *allowed_roots*).

    Raises:
        click.UsageError: *target* is neither a directory nor a ``.txt`` file.
    """
    path = Path(target)
    if path.is_dir():
        return RunSet(run_dirs=tuple(find_run_dirs(path)), out_root=path)
    if path.is_file() and path.suffix.lower() == RUN_LIST_SUFFIX:
        return RunSet(run_dirs=tuple(parse_run_dir_list(path, allowed_roots)), out_root=path.resolve().parent)
    msg = f"{path}: expected a directory of runs or a {RUN_LIST_SUFFIX} list of run directories"
    raise click.UsageError(msg)


def analyse_run(
    run_dir: str | Path,
    config: SimulationConfig,
    *,
    fields: list[str] | None = None,
) -> Path | None:
    """Export ``simulation_data.csv`` and per-run analysis figures for one run.

    Args:
        run_dir: A single run directory containing ``data/timestep_*.npz``.
        config: The run's configuration, passed in memory rather than re-read
            from ``config.toml`` so expanded sweep fields survive.
        fields: Analysis operator names to render. ``None`` exports the CSV only.

    Returns:
        The CSV path, or ``None`` when the run has no analysable data.
    """
    from src.simulation_io.plotting.figure_builder import FigureBuilder

    run_dir = Path(run_dir)
    csv_path = build_simulation_csv(run_dir, config)
    if fields:
        # Reads from the same cached series the CSV export just populated.
        FigureBuilder(config=config, run_dir=run_dir, fields=fields).build_analysis()
    return csv_path


def find_run_dirs(parent_dir: str | Path) -> list[Path]:
    """Run directories under *parent_dir*, identified by a ``config.toml``.

    Directories whose path mentions ``init`` or the comparison output folder
    are not runs and are skipped.
    """
    parent = Path(parent_dir)
    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for toml in sorted(parent.rglob(CONFIG_FILENAME)):
        candidate = toml.parent
        if candidate in seen or any(skip in str(candidate).lower() for skip in _SKIP_DIRS):
            continue
        seen.add(candidate)
        run_dirs.append(candidate)
    return run_dirs


def analyse_runs(run_set: RunSet, out_dir: Path, *, label_keys: Sequence[str] | None = None) -> int:
    """Export every run's ``simulation_data.csv``, then overlay them in *out_dir*.

    The comparison plots are made only when at least one run produced a CSV.

    Args:
        run_set: The runs to compare.
        out_dir: Directory for the comparison plots.
        label_keys: Legend-label keys for the comparison plots; ``None`` labels
            each run with whichever dimensionless numbers differ across the set.

    Returns:
        The number of runs that produced a CSV.
    """
    print(f"Found {len(run_set.run_dirs)} simulation(s) to process.")

    n_ok = 0
    for run_dir in run_set.run_dirs:
        config = _safe_load_config(run_dir / CONFIG_FILENAME)
        if config is None:
            continue
        if analyse_run(run_dir, config) is not None:
            n_ok += 1

    if n_ok > 0:
        print("\nGenerating comparison plots...")
        compare_runs(run_set.run_dirs, out_dir, label_keys)
        print(f"Done. Comparison plots in {out_dir}")

    return n_ok


_SHELL_ESCAPE_RE = re.compile(r"\\[ $;='\"\\]")


def parse_run_dir_list(txt_path: str | Path, allowed_roots: Sequence[str | Path] = ()) -> list[Path]:
    r"""Parse one run-dir path per non-blank, non-``#`` line.

    Lines may be shell-quoted/escaped (e.g. copied from a terminal listing of
    directory names containing spaces or special characters); such quoting is
    stripped and backslash escapes are resolved before building the path.
    This applies whether or not the line is wrapped in quotes — shell
    tab-completion commonly emits backslash-escaped, *unquoted* lines (e.g.
    ``\$Bo_\\parallel\ \=\ 0.60\;\ Oh\ \=\ 0.30\$``). A line is only treated
    as escaped if it contains a backslash before a shell metacharacter
    (space, ``$``, ``;``, ``=``, a quote, or another backslash) — a lone
    literal backslash that is simply part of the directory name (e.g.
    ``\parallel``) is left untouched.

    Each line is treated as exactly one path, even if it contains
    (escaped or unescaped) whitespace — word-splitting would silently
    truncate paths like ``$Bo_\\parallel = 0.60; Oh = 0.30$`` at the first
    space.

    Every entry (relative or absolute) must resolve inside at least one
    trusted root: ``BASE_RESULTS_DIR`` (this project's default results
    location) plus any extra directories passed via ``allowed_roots``. An
    entry outside all of them (e.g. a relative ``..`` escape, or an absolute
    path pointing somewhere untrusted) is rejected with :class:`ValueError`.
    Unlike the list file's own directory, ``allowed_roots`` are supplied
    directly by the caller rather than parsed from the list file's content,
    so they form a real trust boundary rather than one derived from the
    same untrusted text being validated (see SonarQube rule
    ``pythonsecurity:S6549`` — a canonicalized path is only "sanitized" if
    checked against a fixed, independently-supplied root).

    A directory name containing a literal backslash (as in the ``\parallel``
    example above) can only be created on a POSIX filesystem — Windows always
    treats ``\`` as a path separator, so such a name can never exist as a
    single path component there. On a Windows host such an entry is instead
    silently split into nested path components like any other Windows path;
    this function does not probe the filesystem to disambiguate that case,
    since doing so on unvalidated, user-controlled path text would make file
    existence itself an oracle. Callers that hit an unexpectedly missing run
    directory on Windows should check the list file for a backslash that was
    meant as a literal character rather than a path separator.

    ``txt_path`` itself (which may originate from an untrusted CLI argument,
    e.g. when this tool is driven by an external or automated caller) is
    resolved to a canonical, absolute path and confirmed to be an existing
    regular file *before* anything is read from it, so a directory, device
    file, or nonexistent path is rejected with :class:`ValueError` up front
    rather than reaching the filesystem read in an unvalidated state.
    """
    txt_path = Path(txt_path).resolve()
    if not txt_path.is_file():
        msg = f"{txt_path}: run-dir list must be an existing regular file"
        raise ValueError(msg)
    parent = txt_path.parent
    roots = [Path(BASE_RESULTS_DIR), *(Path(root) for root in allowed_roots)]
    resolved_roots = [root.resolve() for root in roots]
    # The lexical check also takes each root as written, so an entry through a
    # symlinked prefix of a root (macOS /var -> /private/var) still matches it.
    lexical_roots = [*resolved_roots, *(Path(os.path.normpath(root.absolute())) for root in roots)]
    dirs: list[Path] = []
    for raw_line in txt_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = _unquote_and_unescape(line)
        if not line:
            continue
        dirs.append(_resolve_run_dir_entry(line, txt_path, parent, lexical_roots, resolved_roots))
    return dirs


def _unquote_and_unescape(line: str) -> str:
    """Strip a single layer of shell quoting and resolve backslash escapes."""
    quoted = len(line) >= _MIN_QUOTED_LINE_LEN and line[0] == line[-1] and line[0] in ("'", '"')
    if quoted:
        line = line[1:-1]
    if quoted or _SHELL_ESCAPE_RE.search(line):
        line = re.sub(r"\\(.)", r"\1", line)
    return line


def _resolve_run_dir_entry(
    line: str, txt_path: Path, parent: Path, lexical_roots: list[Path], resolved_roots: list[Path]
) -> Path:
    """Resolve one parsed line to a run directory, rejecting entries outside every allowed root.

    The untrusted ``line`` touches the filesystem only after it is known to lie inside
    an allowed root (SonarQube ``pythonsecurity:S6549``). It is first normalised
    lexically (``os.path.normpath``, no filesystem access), checked against the roots,
    and only then resolved; the resolved path is checked again, so a symlink inside a
    root cannot lead out of it. The error names the entry, never its resolved target,
    so a rejected line reveals nothing about where a path points.
    """
    candidate = Path(os.path.normpath(Path(line) if Path(line).is_absolute() else parent / line))
    if _inside_any(candidate, lexical_roots):
        resolved = candidate.resolve()
        if _inside_any(resolved, resolved_roots):
            return resolved
    roots_desc = ", ".join(str(root) for root in resolved_roots)
    msg = (
        f"{txt_path}: run directory {line!r} is outside every allowed root "
        f"({roots_desc}). Pass additional trusted locations via --allowed-root."
    )
    raise ValueError(msg)


def _inside_any(path: Path, roots: list[Path]) -> bool:
    """Whether *path* lies inside one of *roots* (a lexical test)."""
    return any(path.is_relative_to(root) for root in roots)
