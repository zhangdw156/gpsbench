import json
import tempfile
import unittest
from pathlib import Path

from scripts.estimate_eval_progress import (
    discover_targets,
    selected_tracks,
    summarize_run,
)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class EstimateEvalProgressTests(unittest.TestCase):
    def test_counts_only_resume_reusable_records_as_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split = root / "data" / "track_pure_gps" / "splits" / "task1_format_conversion_test.json"
            write_json(split, [{"q": 0}, {"q": 1}, {"q": 2}])

            output_dir = root / "results" / "fake_run"
            write_json(
                output_dir / "summary.json",
                {"model": "fake-model", "provider": "openai", "max_samples_per_task": None},
            )
            write_json(
                output_dir / "task_results" / "pure_gps_task1_format_conversion.json",
                {
                    "task_file": "task1_format_conversion.json",
                    "total": 3,
                    "results": [
                        {"example_id": 0, "response": "FINAL ANSWER: A", "correct": True},
                        {"example_id": 1, "response": "FINAL ANSWER: Z", "correct": False},
                        {"example_id": 2, "response": "", "correct": False, "error": "transport failed"},
                    ],
                },
            )

            targets = discover_targets(root / "data", ["pure_gps"], [])
            progress = summarize_run(output_dir, root / "results", targets, cli_max_samples=None)

            self.assertEqual(progress.model, "fake-model")
            self.assertEqual(progress.completed, 2)
            self.assertEqual(progress.correct, 1)
            self.assertEqual(progress.retryable, 1)
            self.assertAlmostEqual(progress.progress, 2 / 3)
            self.assertAlmostEqual(progress.accuracy_done, 1 / 2)
            self.assertAlmostEqual(progress.accuracy_lower_bound, 1 / 3)
            self.assertEqual(progress.tasks[0].status, "partial")

    def test_uses_checkpoint_total_for_max_sample_runs_without_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split = root / "data" / "track_applied" / "splits" / "task1_place_association_test.json"
            write_json(split, [{"q": idx} for idx in range(10)])

            output_dir = root / "results" / "smoke"
            write_json(
                output_dir / "task_results" / "applied_task1_place_association.json",
                {
                    "task_file": "task1_place_association.json",
                    "total": 2,
                    "results": [
                        {"example_id": 0, "response": "ok", "correct": True},
                        {"example_id": 1, "response": "ok", "correct": False},
                    ],
                },
            )

            targets = discover_targets(root / "data", ["applied"], [])
            progress = summarize_run(output_dir, root / "results", targets, cli_max_samples=None)

            self.assertEqual(progress.total, 2)
            self.assertEqual(progress.completed, 2)
            self.assertEqual(progress.completed_tasks, 1)
            self.assertEqual(progress.tasks[0].total_source, "checkpoint total")

    def test_selected_tracks_expands_both_and_comma_lists(self):
        self.assertEqual(selected_tracks(["pure_gps,applied"]), ["pure_gps", "applied"])
        self.assertEqual(selected_tracks(["both"]), ["pure_gps", "applied"])


if __name__ == "__main__":
    unittest.main()
