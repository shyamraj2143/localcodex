"""Summarize independently recorded benchmark outcomes; never runs generated code itself."""
import argparse, json
from pathlib import Path
from training.data import load_records

def evaluate(rows):
    if not rows: return {"tasks":0,"completion_rate":0,"test_pass_rate":0,"tool_call_success_rate":0,"false_success_rate":0,"invalid_or_empty_file_rate":0,"repair_success_rate":0,"hosted_api_calls_offline":0,"gui_verification_unavailable":True}
    n=len(rows); completed=sum(bool(r.get("independently_verified")) for r in rows); test_rows=[r for r in rows if r.get("tests_discovered",0)>0]; tools=[t for r in rows for t in r.get("tool_events",[]) if isinstance(t,dict)]; repairs=[r for r in rows if r.get("repair_attempted")]
    return {"tasks":n,"completion_rate":completed/n,"test_pass_rate":sum(bool(r.get("tests_passed")) for r in test_rows)/len(test_rows) if test_rows else None,"tool_call_success_rate":sum(bool(t.get("result",{}).get("success")) for t in tools)/len(tools) if tools else None,"false_success_rate":sum(bool(r.get("model_claimed_success")) and not bool(r.get("independently_verified")) for r in rows)/n,"invalid_or_empty_file_rate":sum(bool(r.get("invalid_or_empty_file")) for r in rows)/n,"repair_success_rate":sum(bool(r.get("repair_succeeded")) for r in repairs)/len(repairs) if repairs else None,"hosted_api_calls_offline":sum(int(r.get("hosted_api_calls",0)) for r in rows if r.get("offline_mode")),"gui_verification_unavailable":any(r.get("gui_verification_available") is False for r in rows)}

def main():
 p=argparse.ArgumentParser(); p.add_argument("--results",default=str(Path(__file__).with_name("results.jsonl"))); p.add_argument("--output",default=str(Path(__file__).with_name("results.summary.json"))); a=p.parse_args(); result=evaluate(load_records(a.results) if Path(a.results).exists() else []); Path(a.output).write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
