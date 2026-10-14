import json
import tempfile
import unittest
from pathlib import Path

from training.data import DatasetError, load_records, normalize_records, split_records
from training.benchmark import compare_results
from training.trajectory import curate_trajectories


class TrainingDataTests(unittest.TestCase):
    def test_jsonl_normalization_deduplicates_and_redacts(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.jsonl"
            source.write_text("\n".join(json.dumps(row) for row in [
                {"instruction": "Use api_key=abcdefgh12345678", "output": "safe answer", "metadata": {"license": "MIT"}},
                {"prompt": "Use api_key=abcdefgh12345678", "response": "safe answer"},
                {"prompt": "", "response": ""},
            ]), encoding="utf-8")
            records, report = normalize_records(load_records(source))
        self.assertEqual(report["input"], 3)
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(report["duplicates"], 1)
        self.assertEqual(report["rejected"], 1)
        self.assertIn("[REDACTED_SECRET]", records[0]["messages"][0]["content"])
        self.assertEqual(records[0]["metadata"]["license"], "MIT")

    def test_invalid_jsonl_fails_with_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.jsonl"
            source.write_text("{bad}\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetError, "line 1"):
                load_records(source)

    def test_splits_are_reproducible_and_duplicate_hashes_stay_together(self):
        records = [{"id": str(i), "content_sha256": str(i // 2)} for i in range(20)]
        split = split_records(records, seed=17)
        self.assertEqual(split, split_records(records, seed=17))
        locations = {}
        for name, rows in split.items():
            for row in rows:
                locations.setdefault(row["content_sha256"], set()).add(name)
        self.assertTrue(all(len(names) == 1 for names in locations.values()))

    def test_rejects_invalid_split_ratios(self):
        with self.assertRaises(ValueError):
            split_records([], train=0.9, validation=0.2)

    def test_small_dataset_keeps_available_partitions_nonempty(self):
        one = split_records([{"content_sha256": "a"}])
        self.assertEqual(len(one["train"]), 1)
        three = split_records([{"content_sha256": str(i)} for i in range(3)])
        self.assertTrue(all(three[name] for name in ("train", "validation", "test")))

    def test_benchmark_compares_only_identical_tasks_and_verifies_claims(self):
        base = [{"task_id": "a", "independently_verified": False, "model_claimed_success": True, "tests_passed": False}]
        candidate = [{"task_id": "a", "independently_verified": True, "model_claimed_success": True, "tests_passed": True}]
        result = compare_results(base, candidate)
        self.assertEqual(result["base"]["unsupported_success_claim_rate"], 1.0)
        self.assertEqual(result["candidate"]["completion_rate"], 1.0)
        with self.assertRaises(ValueError):
            compare_results(base, [{"task_id": "b"}])

    def test_trajectory_export_keeps_verified_examples_and_failed_records_separate(self):
        rows = [
            {"verification_status": "PASS", "user_request": "Add a sum", "workspace_context": "file: calculator.py",
             "final_report": "Tests passed", "source_changes": {"calculator.py": "def add(a,b): return a+b"},
             "workspace_id": "workspace-1"},
            {"verification_status": "PARTIAL", "user_request": "Broken request", "final_report": "Incomplete"},
        ]
        verified, diagnostics = curate_trajectories(rows)
        self.assertEqual(len(verified), 1)
        self.assertIn("calculator.py", verified[0]["prompt"])
        self.assertIn("def add", verified[0]["response"])
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]["failure_category"], "incomplete_or_unverified")
        normalized, report = normalize_records(verified)
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(normalized[0]["metadata"]["verification_status"], "PASS")


if __name__ == "__main__":
    unittest.main()
