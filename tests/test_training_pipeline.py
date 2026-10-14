import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.pipeline import append_record, export_sft, read_jsonl, review, safe, split, trajectory, validate


class TrainingPipelineTests(unittest.TestCase):
    def record(self, task_id="a", status="success"):
        return {"schema_version":1,"task_id":task_id,"timestamp":"2026-01-01T00:00:00Z","request":"Build a calculator",
                "category":"coding","workspace_summary":"Python project","relevant_paths":[],"plan":[],
                "events":[{"tool":"write_file","arguments":{"path":"calc.py"},"result":{"success":True,"path":"calc.py","operation":"write_file"}}],
                "commands":[],"changed_files":["calc.py"],"file_changes":[{"path":"calc.py","operation":"write_file","success":True}],
                "tests":{"discovered":0,"passed":0,"failed":0},"verification":{"tests_discovered":0,"tests_passed":0,"tests_failed":0,"evidence":[{"kind":"syntax","passed":True}]},
                "final_status":status,"review_status":"pending","final_report":"Implemented and verified."}

    def test_collection_sanitizes_secret_and_keeps_actual_tools(self):
        report={"task_id":"id-1","task":"Use api_key=abcdefgh12345678","verification_status":"PASS","tool_events":[{"tool":"write_file","arguments":{"path":"x.py"},"result":{"success":True}}]}
        row=trajectory(report)
        self.assertIn("[REDACTED]",row["request"])
        self.assertEqual(row["events"][0]["tool"],"write_file")

    def test_validation_detects_duplicate_ids_and_unsupported_success(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"data.jsonl"; row=self.record(); row["verification"]["evidence"]=[]
            p.write_text(json.dumps(row)+"\n"+json.dumps(row)+"\n",encoding="utf-8")
            errors=validate(p)["errors"]
        self.assertTrue(any("duplicate task_id" in e for e in errors))
        self.assertTrue(any("unsupported" in e for e in errors))

    def test_review_rejects_partial_and_routes_decisions(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"raw.jsonl"; row=self.record(status="partial"); p.write_text(json.dumps(row)+"\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"only successful"):
                review(p,"a","approved")

    def test_grouped_split_reproducible_and_no_leakage(self):
        rows=[dict(self.record(str(i)),duplicate_group=str(i//2)) for i in range(12)]
        a=split(rows,23); b=split(rows,23); self.assertEqual(a,b)
        names={}
        for part,subset in a.items():
            for row in subset: names.setdefault(row["duplicate_group"],set()).add(part)
        self.assertTrue(all(len(v)==1 for v in names.values()))

    def test_sft_requires_review_and_preserves_tool_results(self):
        row=self.record(); row["review_status"]="approved"
        output=export_sft([row,self.record("b","partial")])
        self.assertEqual(len(output),1)
        self.assertEqual(output[0]["messages"][1]["tool_call"]["tool"],"write_file")
        self.assertEqual(output[0]["messages"][2]["role"],"tool")

    def test_append_record_is_single_valid_json_line(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"events.jsonl"; append_record(path,self.record())
            self.assertEqual(len(read_jsonl(path)),1)

    def test_logging_failure_never_raises_from_task_finish(self):
        from agent.workspace_memory import WorkspaceMemory
        with tempfile.TemporaryDirectory() as d:
            class Context: pass
            memory=WorkspaceMemory(d,Context(),{"persistent_memory_enabled":True,"training_data_collection_enabled":True})
            memory._data["file_index"]={"files":[]}
            with patch("training.pipeline.append_record",side_effect=OSError("disk full")):
                memory.record_task_finish({"task_id":"id","task":"t","verification_status":"FAIL","commands":[],"static_checks":[]})
            self.assertIn("training collection failed",memory.status)


if __name__=="__main__": unittest.main()
