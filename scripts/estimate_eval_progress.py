#!/usr/bin/env python3
"""Estimate GPSBench evaluation progress with the runner's resume rule.

Progress is measured the same way as ``run_benchmark.py`` resumes work: a
sample is counted as completed/reusable only when its checkpoint record has no
``error``, contains a ``correct`` field, and has a non-empty ``response``.
Missing samples and retryable/API-failed samples stay pending; valid but
incorrect model answers are counted as completed so reruns do not bias accuracy.

Examples:
    uv run python scripts/estimate_eval_progress.py
    uv run python scripts/estimate_eval_progress.py --run qwen3-4b-thinking-2507_10pct_test
    uv run python scripts/estimate_eval_progress.py --track pure_gps --task distance_calculation
    uv run python scripts/estimate_eval_progress.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
TRACK_DIRS = {
    "pure_gps": "track_pure_gps",
    "applied": "track_applied",
}
TRACK_CHOICES = tuple(TRACK_DIRS) + ("both", "all")


@dataclass(frozen=True)
class EvalTarget:
    track: str
    task: str
    task_file: str
    split_path: str
    data_total: int

    @property
    def result_filename(self) -> str:
        return f"{self.track}_{self.task}.json"

    @property
    def display_task(self) -> str:
        label = self.task
        parts = label.split("_", 1)
        if parts and parts[0].startswith("task") and len(parts) > 1:
            label = parts[1]
        return label.replace("_", " ")


@dataclass(frozen=True)
class TaskProgress:
    track: str
    task: str
    task_file: str
    completed: int
    correct: int
    retryable: int
    total: int
    data_total: int
    progress: float
    accuracy_done: float | None
    accuracy_lower_bound: float
    result_path: str | None
    result_records: int
    observed_records: int
    duplicate_records: int
    invalid_records: int
    missing_result: bool
    unreadable_result: bool
    result_total: int | None
    total_source: str
    status: str
    warning: str | None


@dataclass(frozen=True)
class RunProgress:
    run: str
    output_dir: str
    model: str
    provider: str | None
    summary_path: str | None
    max_samples_per_task: int | None
    completed: int
    correct: int
    retryable: int
    total: int
    progress: float
    accuracy_done: float | None
    accuracy_lower_bound: float
    completed_tasks: int
    started_tasks: int
    total_tasks: int
    missing_tasks: int
    unreadable_tasks: int
    tasks: list[TaskProgress]


def split_values(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in value.split(",") if part.strip())
    return items


def normalize_filter(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def selected_tracks(raw_tracks: Sequence[str] | None) -> list[str]:
    values = split_values(raw_tracks)
    if not values:
        return list(TRACK_DIRS)
    tracks: list[str] = []
    for value in values:
        normalized = value.lower()
        if normalized in ("both", "all"):
            tracks.extend(TRACK_DIRS)
        elif normalized in TRACK_DIRS:
            tracks.append(normalized)
        else:
            expected = ", ".join(TRACK_CHOICES)
            raise ValueError(f"unknown track '{value}', expected one of: {expected}")
    return list(dict.fromkeys(tracks))


def task_matches(target: EvalTarget, task_filters: Sequence[str]) -> bool:
    if not task_filters:
        return True
    haystacks = {
        normalize_filter(target.task),
        normalize_filter(target.task_file),
        normalize_filter(target.display_task),
    }
    return any(any(needle in haystack for haystack in haystacks) for needle in task_filters)


def discover_targets(data_dir: Path, tracks: Sequence[str], task_filters: Sequence[str]) -> list[EvalTarget]:
    if not data_dir.exists():
        raise FileNotFoundError(f"data directory not found: {data_dir}; run scripts/prepare_data.sh first")

    normalized_filters = [normalize_filter(value) for value in task_filters]
    targets: list[EvalTarget] = []
    for track in tracks:
        split_dir = data_dir / TRACK_DIRS[track] / "splits"
        if not split_dir.exists():
            continue
        for split_path in sorted(split_dir.glob("*_test.json")):
            task = split_path.name[: -len("_test.json")]
            target = EvalTarget(
                track=track,
                task=task,
                task_file=f"{task}.json",
                split_path=relpath(split_path),
                data_total=load_split_total(split_path),
            )
            if task_matches(target, normalized_filters):
                targets.append(target)

    if not targets:
        track_text = ",".join(tracks)
        filter_text = f" matching tasks {', '.join(task_filters)}" if task_filters else ""
        raise FileNotFoundError(f"no *_test.json targets found under {data_dir} for tracks {track_text}{filter_text}")
    return targets


def load_split_total(path: Path) -> int:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"expected JSON list in split file: {path}")
    return len(data)


def read_summary(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        summary = load_json(summary_path)
    except (json.JSONDecodeError, OSError):
        return {}
    return summary if isinstance(summary, dict) else {}


def discover_runs(results_dir: Path) -> list[Path]:
    if (results_dir / "task_results").is_dir():
        return [results_dir]
    if not results_dir.exists():
        return []
    return sorted(
        path
        for path in results_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".") and (path / "task_results").is_dir()
    )


def resolve_run_paths(raw_runs: Sequence[str], results_dir: Path) -> list[Path]:
    runs: list[Path] = []
    for raw in raw_runs:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            direct = (Path.cwd() / candidate).resolve()
            under_results = (results_dir / candidate).resolve()
            candidate = direct if direct.exists() else under_results
        if not candidate.exists():
            raise FileNotFoundError(f"result run does not exist: {raw}")
        if not (candidate / "task_results").is_dir():
            raise FileNotFoundError(f"result run has no task_results/ directory: {candidate}")
        runs.append(candidate)
    return list(dict.fromkeys(runs))


def run_label(output_dir: Path, results_dir: Path) -> str:
    try:
        return str(output_dir.resolve().relative_to(results_dir.resolve()))
    except ValueError:
        return output_dir.name


def result_needs_retry(record: Any) -> bool:
    """Mirror GPSBenchRunner._result_needs_retry without importing runtime deps."""
    if not isinstance(record, dict):
        return True
    if record.get("error"):
        return True
    if "correct" not in record:
        return True
    response = record.get("response")
    if response is None:
        return True
    if isinstance(response, str) and not response.strip():
        return True
    return False


def read_task_result(path: Path) -> tuple[dict[str, Any] | None, list[Any], str | None]:
    try:
        payload = load_json(path)
    except (json.JSONDecodeError, OSError) as exc:
        return None, [], str(exc)
    if not isinstance(payload, dict):
        return None, [], "result file is not a JSON object"
    records = payload.get("results", [])
    if records is None:
        records = []
    if not isinstance(records, list):
        return payload, [], "result file field 'results' is not a list"
    return payload, records, None


def expected_total_for_task(
    target: EvalTarget,
    result_payload: dict[str, Any] | None,
    *,
    cli_max_samples: int | None,
    summary_max_samples: int | None,
) -> tuple[int, str, int | None]:
    if cli_max_samples is not None:
        return min(target.data_total, cli_max_samples), "cli --max-samples", None
    if summary_max_samples is not None:
        return min(target.data_total, summary_max_samples), "summary max_samples_per_task", None
    result_total = None
    if result_payload is not None:
        result_total = parse_positive_int(result_payload.get("total"))
        if result_total is not None:
            return result_total, "checkpoint total", result_total
    return target.data_total, "data split", result_total


def summarize_task(
    output_dir: Path,
    target: EvalTarget,
    *,
    cli_max_samples: int | None,
    summary_max_samples: int | None,
) -> TaskProgress:
    result_path = output_dir / "task_results" / target.result_filename
    missing_result = not result_path.exists()
    unreadable_result = False
    result_payload: dict[str, Any] | None = None
    records: list[Any] = []
    read_error: str | None = None

    if not missing_result:
        result_payload, records, read_error = read_task_result(result_path)
        unreadable_result = read_error is not None

    total, total_source, result_total = expected_total_for_task(
        target,
        result_payload,
        cli_max_samples=cli_max_samples,
        summary_max_samples=summary_max_samples,
    )

    latest_by_id: dict[int, Any] = {}
    duplicate_records = 0
    invalid_records = 0
    for record in records:
        if not isinstance(record, dict):
            invalid_records += 1
            continue
        try:
            example_id = int(record.get("example_id"))
        except (TypeError, ValueError):
            invalid_records += 1
            continue
        if example_id < 0 or example_id >= total:
            invalid_records += 1
            continue
        if example_id in latest_by_id:
            duplicate_records += 1
        latest_by_id[example_id] = record

    completed = 0
    correct = 0
    for record in latest_by_id.values():
        if result_needs_retry(record):
            continue
        completed += 1
        if record.get("correct") is True:
            correct += 1

    retryable = max(total - completed, 0)
    progress = completed / total if total else 0.0
    accuracy_done = correct / completed if completed else None
    accuracy_lower_bound = correct / total if total else 0.0

    warning_parts: list[str] = []
    if read_error:
        warning_parts.append(read_error)
    if result_total is not None and cli_max_samples is None and summary_max_samples is None and result_total != target.data_total:
        warning_parts.append(f"checkpoint total {result_total} differs from data total {target.data_total}")
    if duplicate_records:
        warning_parts.append(f"duplicates={duplicate_records}")
    if invalid_records:
        warning_parts.append(f"invalid_records={invalid_records}")

    if unreadable_result:
        status = "unreadable"
    elif missing_result:
        status = "missing"
    elif completed >= total:
        status = "done"
    elif latest_by_id:
        status = "partial"
    elif records:
        status = "invalid"
    else:
        status = "empty"

    return TaskProgress(
        track=target.track,
        task=target.task,
        task_file=target.task_file,
        completed=completed,
        correct=correct,
        retryable=retryable,
        total=total,
        data_total=target.data_total,
        progress=progress,
        accuracy_done=accuracy_done,
        accuracy_lower_bound=accuracy_lower_bound,
        result_path=relpath(result_path) if not missing_result else None,
        result_records=len(records),
        observed_records=len(latest_by_id),
        duplicate_records=duplicate_records,
        invalid_records=invalid_records,
        missing_result=missing_result,
        unreadable_result=unreadable_result,
        result_total=result_total,
        total_source=total_source,
        status=status,
        warning="; ".join(warning_parts) if warning_parts else None,
    )


def summarize_run(
    output_dir: Path,
    results_dir: Path,
    targets: list[EvalTarget],
    *,
    cli_max_samples: int | None,
) -> RunProgress:
    summary = read_summary(output_dir)
    summary_max_samples = parse_positive_int(summary.get("max_samples_per_task"))
    model = str(summary.get("model") or output_dir.name)
    provider = summary.get("provider")
    provider_text = str(provider) if provider is not None else None
    tasks = [
        summarize_task(
            output_dir,
            target,
            cli_max_samples=cli_max_samples,
            summary_max_samples=summary_max_samples,
        )
        for target in targets
    ]

    completed = sum(task.completed for task in tasks)
    correct = sum(task.correct for task in tasks)
    total = sum(task.total for task in tasks)
    retryable = max(total - completed, 0)
    progress = completed / total if total else 0.0
    accuracy_done = correct / completed if completed else None
    accuracy_lower_bound = correct / total if total else 0.0
    started_tasks = sum(1 for task in tasks if not task.missing_result and task.status not in {"unreadable", "empty"})
    completed_tasks = sum(1 for task in tasks if task.completed >= task.total and task.total > 0)
    missing_tasks = sum(1 for task in tasks if task.missing_result)
    unreadable_tasks = sum(1 for task in tasks if task.unreadable_result)
    summary_path = output_dir / "summary.json"

    return RunProgress(
        run=run_label(output_dir, results_dir),
        output_dir=relpath(output_dir),
        model=model,
        provider=provider_text,
        summary_path=relpath(summary_path) if summary_path.exists() else None,
        max_samples_per_task=summary_max_samples,
        completed=completed,
        correct=correct,
        retryable=retryable,
        total=total,
        progress=progress,
        accuracy_done=accuracy_done,
        accuracy_lower_bound=accuracy_lower_bound,
        completed_tasks=completed_tasks,
        started_tasks=started_tasks,
        total_tasks=len(tasks),
        missing_tasks=missing_tasks,
        unreadable_tasks=unreadable_tasks,
        tasks=tasks,
    )


def model_matches(item: RunProgress, filters: Sequence[str]) -> bool:
    if not filters:
        return True
    haystacks = [normalize_filter(item.model), normalize_filter(item.run), normalize_filter(Path(item.output_dir).name)]
    needles = [normalize_filter(value) for value in filters]
    return any(any(needle in haystack for haystack in haystacks) for needle in needles)


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_optional_pct(value: float | None) -> str:
    return "n/a" if value is None else format_pct(value)


def ratio(completed: int, total: int) -> str:
    return f"{completed:,}/{total:,}"


def progress_bar(progress: float, width: int = 18) -> str:
    bounded = max(0.0, min(1.0, progress))
    filled = round(bounded * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    return value[: width - 1] + "…"


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]], *, max_widths: dict[int, int] | None = None) -> str:
    max_widths = max_widths or {}
    normalized_rows = [[str(cell) for cell in row] for row in rows]
    widths: list[int] = []
    for idx, header in enumerate(headers):
        width = len(header)
        for row in normalized_rows:
            width = max(width, len(row[idx]))
        if idx in max_widths:
            width = min(width, max_widths[idx])
        widths.append(width)

    def format_row(row: Sequence[str]) -> str:
        cells = [truncate(str(cell), widths[idx]).ljust(widths[idx]) for idx, cell in enumerate(row)]
        return "  ".join(cells).rstrip()

    divider = "  ".join("-" * width for width in widths)
    lines = [format_row(headers), divider]
    lines.extend(format_row(row) for row in normalized_rows)
    return "\n".join(lines)


def render_report(items: list[RunProgress], *, targets: list[EvalTarget], data_dir: Path, results_dir: Path, show_tasks: bool) -> None:
    total_cases = sum(target.data_total for target in targets)
    print("GPSBench evaluation progress")
    print(f"Data:    {data_dir}")
    print(f"Results: {results_dir}")
    print(f"Targets: {len(targets)} tasks, {total_cases:,} data samples")
    print("Rule: completed = no error + has correct + non-empty response; wrong-but-valid answers are reusable")
    print()

    if not items:
        print("No result runs found.")
        return

    summary_rows = []
    for item in items:
        issues = []
        if item.retryable:
            issues.append(f"retry {item.retryable:,}")
        if item.missing_tasks:
            issues.append(f"missing {item.missing_tasks}")
        if item.unreadable_tasks:
            issues.append(f"unreadable {item.unreadable_tasks}")
        summary_rows.append(
            [
                item.run,
                item.model,
                f"{progress_bar(item.progress)} {format_pct(item.progress)}",
                ratio(item.completed, item.total),
                format_optional_pct(item.accuracy_done),
                format_pct(item.accuracy_lower_bound),
                ratio(item.correct, item.completed) if item.completed else "0/0",
                f"done {ratio(item.completed_tasks, item.total_tasks)}; started {ratio(item.started_tasks, item.total_tasks)}",
                ", ".join(issues) if issues else "done",
            ]
        )
    print(render_table(
        ["Run", "Model", "Progress", "Samples", "Acc(done)", "Acc(lb)", "Correct", "Tasks", "Issues"],
        summary_rows,
        max_widths={0: 28, 1: 32, 7: 28, 8: 30},
    ))

    if not show_tasks:
        return

    for item in items:
        print(f"\n{item.run} task details")
        task_rows = []
        for task in item.tasks:
            status = task.status
            if task.warning:
                status = f"{status}; {task.warning}"
            task_rows.append(
                [
                    task.track,
                    task.task,
                    f"{progress_bar(task.progress, width=12)} {format_pct(task.progress)}",
                    ratio(task.completed, task.total),
                    format_optional_pct(task.accuracy_done),
                    ratio(task.correct, task.completed) if task.completed else "0/0",
                    f"{task.retryable:,}",
                    f"{task.observed_records:,}/{task.result_records:,}",
                    status,
                ]
            )
        print(render_table(
            ["Track", "Task", "Progress", "Samples", "Acc(done)", "Correct", "Retry", "Seen/Records", "Status"],
            task_rows,
            max_widths={1: 34, 8: 48},
        ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="GPSBench data directory. Default: ./data")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Evaluation results directory. Default: ./results")
    parser.add_argument("--run", "--output", dest="runs", action="append", help="Result run/output directory to inspect; can repeat or use comma lists.")
    parser.add_argument("--model", dest="models", action="append", help="Filter by summary model name or run directory; can repeat or use comma lists.")
    parser.add_argument("--track", action="append", help=f"Track to include ({', '.join(TRACK_CHOICES)}). Can repeat or use comma lists. Default: both tracks.")
    parser.add_argument("--task", action="append", help="Task substring/filter, e.g. distance_calculation. Can repeat or use comma lists.")
    parser.add_argument("--max-samples", type=int, default=None, help="Override expected samples per task, matching run_benchmark.py --max-samples.")
    parser.add_argument("--no-tasks", action="store_true", help="Only print the per-run summary table.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of tables.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")

    try:
        tracks = selected_tracks(args.track)
        task_filters = split_values(args.task)
        targets = discover_targets(args.data_dir, tracks, task_filters)
        raw_runs = split_values(args.runs)
        run_paths = resolve_run_paths(raw_runs, args.results_dir) if raw_runs else discover_runs(args.results_dir)
        items = [summarize_run(path, args.results_dir, targets, cli_max_samples=args.max_samples) for path in run_paths]
        model_filters = split_values(args.models)
        items = [item for item in items if model_matches(item, model_filters)]
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    if args.json:
        payload = {
            "data_dir": str(args.data_dir),
            "results_dir": str(args.results_dir),
            "tracks": tracks,
            "task_filters": task_filters,
            "num_targets": len(targets),
            "total_data_samples": sum(target.data_total for target in targets),
            "targets": [asdict(target) for target in targets],
            "runs": [asdict(item) for item in items],
        }
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    render_report(items, targets=targets, data_dir=args.data_dir, results_dir=args.results_dir, show_tasks=not args.no_tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
