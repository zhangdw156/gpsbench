import json
import tempfile
import unittest
from pathlib import Path

from run_benchmark import GPSBenchRunner
from llm_client import LLMResponse


TASK_INFO = {"file": "task1_format_conversion.json", "name": "Format Conversion"}


class FakeLLM:
    model = "fake-model"
    provider = "fake-provider"

    def __init__(self, answers=None, interrupt_after=None):
        self.answers = answers or {}
        self.calls = []
        self.interrupt_after = interrupt_after

    def generate(self, prompt, system_prompt=None):
        self.calls.append(prompt)
        if self.interrupt_after is not None and len(self.calls) > self.interrupt_after:
            raise KeyboardInterrupt()
        for key, answer in self.answers.items():
            if key in prompt:
                return LLMResponse(text=f"FINAL ANSWER: {answer}", model=self.model)
        return LLMResponse(text="FINAL ANSWER: A", model=self.model)


def write_task_data(root: Path):
    split_dir = root / "data" / "track_pure_gps" / "splits"
    split_dir.mkdir(parents=True)
    data = [
        {"question": "Question q0", "ground_truth": {"answer": "A"}},
        {"question": "Question q1", "ground_truth": {"answer": "B"}},
        {"question": "Question q2", "ground_truth": {"answer": "C"}},
    ]
    (split_dir / "task1_format_conversion_test.json").write_text(json.dumps(data), encoding="utf-8")


class ResumeEvaluationTests(unittest.TestCase):
    def test_resume_only_evaluates_error_and_missing_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_task_data(root)
            output_dir = root / "results" / "fake_run"
            task_dir = output_dir / "task_results"
            task_dir.mkdir(parents=True)
            existing = {
                "task_name": "Format Conversion",
                "task_file": "task1_format_conversion.json",
                "total": 3,
                "correct": 1,
                "accuracy": 33.3333333333,
                "results": [
                    {
                        "example_id": 0,
                        "prompt": "Question q0",
                        "system_prompt": "system",
                        "response": "FINAL ANSWER: A",
                        "ground_truth": {"answer": "A"},
                        "correct": True,
                    },
                    {
                        "example_id": 1,
                        "prompt": "Question q1",
                        "system_prompt": "system",
                        "response": "",
                        "ground_truth": {"answer": "B"},
                        "error": "transport failed",
                        "correct": False,
                    },
                ],
            }
            (task_dir / "pure_gps_task1_format_conversion.json").write_text(
                json.dumps(existing), encoding="utf-8"
            )

            llm = FakeLLM({"q1": "B", "q2": "C"})
            runner = GPSBenchRunner(llm, data_dir=str(root / "data"), results_dir=str(root / "results"))

            result = runner.evaluate_task("pure_gps", TASK_INFO, output_dir=output_dir, use_concurrent=True, max_workers=2)

            self.assertEqual(len(llm.calls), 2)
            self.assertFalse(any("q0" in prompt for prompt in llm.calls))
            self.assertTrue(any("q1" in prompt for prompt in llm.calls))
            self.assertTrue(any("q2" in prompt for prompt in llm.calls))
            self.assertEqual([r["example_id"] for r in result["results"]], [0, 1, 2])
            self.assertEqual(result["correct"], 3)
            self.assertEqual(result["resume"]["reused"], 1)
            self.assertEqual(result["resume"]["evaluated"], 2)

    def test_checkpoint_is_written_after_each_completed_sample_before_interrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_task_data(root)
            output_dir = root / "results" / "fake_run"
            output_dir.mkdir(parents=True)

            llm = FakeLLM({"q0": "A"}, interrupt_after=1)
            runner = GPSBenchRunner(llm, data_dir=str(root / "data"), results_dir=str(root / "results"))

            with self.assertRaises(KeyboardInterrupt):
                runner.evaluate_task("pure_gps", TASK_INFO, output_dir=output_dir)

            checkpoint_path = output_dir / "task_results" / "pure_gps_task1_format_conversion.json"
            self.assertTrue(checkpoint_path.exists())
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["status"], "in_progress")
            self.assertEqual([r["example_id"] for r in checkpoint["results"]], [0])
            self.assertEqual(checkpoint["resume"]["pending"], 2)

    def test_resume_does_not_retry_valid_incorrect_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_task_data(root)
            output_dir = root / "results" / "fake_run"
            task_dir = output_dir / "task_results"
            task_dir.mkdir(parents=True)
            existing = {
                "task_name": "Format Conversion",
                "task_file": "task1_format_conversion.json",
                "total": 3,
                "correct": 2,
                "accuracy": 66.6666666667,
                "results": [
                    {
                        "example_id": 0,
                        "prompt": "Question q0",
                        "system_prompt": "system",
                        "response": "FINAL ANSWER: Z",
                        "ground_truth": {"answer": "A"},
                        "correct": False,
                    },
                    {
                        "example_id": 1,
                        "prompt": "Question q1",
                        "system_prompt": "system",
                        "response": "FINAL ANSWER: B",
                        "ground_truth": {"answer": "B"},
                        "correct": True,
                    },
                    {
                        "example_id": 2,
                        "prompt": "Question q2",
                        "system_prompt": "system",
                        "response": "",
                        "ground_truth": {"answer": "C"},
                        "error": "transport failed",
                        "correct": False,
                    },
                ],
            }
            (task_dir / "pure_gps_task1_format_conversion.json").write_text(
                json.dumps(existing), encoding="utf-8"
            )

            llm = FakeLLM({"q2": "C"})
            runner = GPSBenchRunner(llm, data_dir=str(root / "data"), results_dir=str(root / "results"))

            result = runner.evaluate_task("pure_gps", TASK_INFO, output_dir=output_dir)

            self.assertEqual(len(llm.calls), 1)
            self.assertFalse(any("q0" in prompt for prompt in llm.calls))
            self.assertTrue(any("q2" in prompt for prompt in llm.calls))
            self.assertEqual(result["correct"], 2)
            self.assertFalse(result["results"][0]["correct"])
            self.assertEqual(result["resume"]["reused"], 2)
            self.assertEqual(result["resume"]["evaluated"], 1)


if __name__ == "__main__":
    unittest.main()
