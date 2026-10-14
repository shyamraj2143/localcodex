"""Offline trajectory collection, review, validation, grouped splitting and SFT export."""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

from agent.orchestration import redact_secrets

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "trajectories.jsonl"
REVIEWED = ROOT / "data" / "reviewed"
PROCESSED = ROOT / "data" / "processed"
SECRET_FILE = re.compile(r"(^|[/\\])\.env(?:\.|$)|secret|credential|private[_-]?key", re.I)
BLOCKED = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".localcodex"}


def safe(value, limit=4000):
    if isinstance(value, str):
        return redact_secrets(value)[:limit]
    if isinstance(value, dict):
        return {str(k)[:100]: safe(v) for k, v in list(value.items())[:100] if not SECRET_FILE.search(str(k))}
    if isinstance(value, list):
        return [safe(v) for v in value[:200]]
    return value if value is None or isinstance(value, (bool, int, float)) else redact_secrets(str(value))[:limit]


def append_record(path, record):
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(safe(record), ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, line)
    finally:
        os.close(descriptor)


def trajectory(report):
    commands = report.get("commands", [])
    tests = [r for r in commands if re.search(r"pytest|unittest|test|build", str(r.get("command", "")), re.I)]
    passed = sum(bool(r.get("success")) for r in tests)
    failed = sum(not bool(r.get("success")) for r in tests)
    status = report.get("verification_status", "NOT VERIFIED")
    events = report.get("tool_events", [])
    file_changes = []
    for event in events:
        if not isinstance(event, dict): continue
        result = event.get("result", {})
        path = result.get("path") if isinstance(result, dict) else None
        operation = result.get("operation") if isinstance(result, dict) else None
        if path and operation in {"write_file", "edit_file", "delete_file"}:
            file_changes.append({"path": path, "operation": operation, "success": result.get("success")})
    return safe({
        "schema_version": 1, "task_id": report.get("task_id") or secrets.token_hex(16),
        "timestamp": report.get("finished_at") or datetime.now(timezone.utc).isoformat(),
        "request": report.get("task", ""), "category": report.get("category", "coding"),
        "workspace_summary": report.get("context_snapshot", "")[:12000],
        "relevant_paths": report.get("retrieved_paths", []), "plan": report.get("plan", []),
        "events": events, "commands": commands,
        "changed_files": report.get("changed_files", []), "file_changes": file_changes,
        "tests": {"discovered": len(tests), "passed": passed, "failed": failed, "results": tests},
        "verification": {"syntax_passed": report.get("syntax_passed"), "tests_discovered": len(tests),
                          "tests_passed": passed, "tests_failed": failed,
                          "gui_launch_verified": bool(report.get("gui_launch_verified", False)),
                          "evidence": report.get("static_checks", [])},
        "final_status": "success" if status == "PASS" and bool(report.get("static_checks")) else "failed" if status == "FAIL" else "partial",
        "review_status": "pending", "final_report": report.get("final_report", ""),
    })


def read_jsonl(path):
    rows=[]
    for n,line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(),1):
        if line.strip():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError as e: raise ValueError(f"line {n}: {e}") from e
    return rows


def validate(path):
    if not Path(path).exists():
        return {"records": 0, "errors": [], "valid": True, "warning": "No trajectories collected yet."}
    errors=[]; ids=set(); hashes=set(); count=0
    required={"schema_version","task_id","timestamp","request","category","workspace_summary","relevant_paths","plan","events","changed_files","tests","verification","final_status","review_status"}
    for n,line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(),1):
        if not line.strip(): continue
        count+=1
        try: r=json.loads(line)
        except json.JSONDecodeError as e: errors.append(f"line {n}: malformed JSON ({e.msg})"); continue
        if not isinstance(r,dict): errors.append(f"line {n}: record must be object"); continue
        missing=required-r.keys()
        if missing: errors.append(f"line {n}: missing {', '.join(sorted(missing))}")
        tid=r.get("task_id")
        if not tid: errors.append(f"line {n}: empty task_id")
        elif tid in ids: errors.append(f"line {n}: duplicate task_id {tid}")
        ids.add(tid)
        digest=hashlib.sha256(json.dumps(r,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if digest in hashes: errors.append(f"line {n}: duplicate record")
        hashes.add(digest)
        if r.get("final_status") not in {"success","partial","failed"}: errors.append(f"line {n}: invalid final_status")
        tests=r.get("tests",{})
        if not isinstance(tests,dict) or any(type(tests.get(k)) is not int or tests[k]<0 for k in ("discovered","passed","failed")):
            errors.append(f"line {n}: invalid test counts")
        elif tests["passed"]+tests["failed"]>tests["discovered"]: errors.append(f"line {n}: inconsistent test counts")
        verification=r.get("verification",{})
        if isinstance(verification,dict) and isinstance(tests,dict):
            if any(verification.get(k)!=tests.get(k2) for k,k2 in (("tests_discovered","discovered"),("tests_passed","passed"),("tests_failed","failed"))):
                errors.append(f"line {n}: verification test counts disagree with test results")
        if not isinstance(r.get("commands",[]),list): errors.append(f"line {n}: commands must be an array")
        else:
            for command in r.get("commands",[]):
                if not isinstance(command,dict) or not {"command","return_code","success"}.issubset(command):
                    errors.append(f"line {n}: malformed command result")
                    break
        if r.get("final_status")=="success" and (not isinstance(r.get("verification"),dict) or not r["verification"].get("evidence")):
            errors.append(f"line {n}: success unsupported by verification evidence")
        if r.get("final_status")=="success" and any(not isinstance(e,dict) or "tool" not in e or "result" not in e for e in r.get("events",[])):
            errors.append(f"line {n}: suspicious/incomplete tool trace")
    return {"records":count,"errors":errors,"valid":not errors}


def review(path, task_id, decision, note=""):
    if decision not in {"approved","rejected","pending"}: raise ValueError("decision must be approved, rejected, or pending")
    rows=read_jsonl(path); found=False
    for r in rows:
        if r.get("task_id")==task_id:
            found=True; r["review_status"]=decision; r["review_note"]=safe(note,1000)
            if decision=="approved" and r.get("final_status")!="success": raise ValueError("only successful trajectories can be approved")
    if not found: raise ValueError("task_id not found")
    approved=[r for r in rows if r.get("review_status")=="approved"]
    rejected=[r for r in rows if r.get("review_status")=="rejected"]
    pending=[r for r in rows if r.get("review_status")=="pending"]
    for name, subset in (("approved",approved),("rejected",rejected),("pending",pending)):
        write_jsonl(REVIEWED/f"{name}.jsonl",subset)


def write_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="\n") as f:
        for row in rows: f.write(json.dumps(safe(row),ensure_ascii=False)+"\n")


def split(records, seed=42):
    import random
    groups={}
    for r in records:
        key=r.get("duplicate_group") or hashlib.sha256((str(r.get("request",""))+"\0"+json.dumps(r.get("changed_files",[]),sort_keys=True)).encode()).hexdigest()
        groups.setdefault(key,[]).append(r)
    keys=sorted(groups); random.Random(seed).shuffle(keys)
    if len(keys)<3: raise ValueError("At least three distinct task groups are required for train/validation/test")
    nt=max(1,int(len(keys)*.8)); nv=max(1,int(len(keys)*.1))
    if nt+nv>=len(keys): nt=len(keys)-2; nv=1
    assignments={k:("train" if i<nt else "validation" if i<nt+nv else "test") for i,k in enumerate(keys)}
    return {part:[r for key,items in groups.items() if assignments[key]==part for r in items] for part in ("train","validation","test")}


def export_sft(records):
    result=[]
    for r in records:
        if r.get("review_status")!="approved" or r.get("final_status")!="success": continue
        if not r.get("verification",{}).get("evidence"): continue
        user=str(r.get("request","")).strip()
        assistant=str(r.get("final_report","")).strip()
        events=r.get("events",[])
        if not user or not assistant or not isinstance(events,list): continue
        messages=[{"role":"user","content":user}]
        for event in events:
            if not isinstance(event,dict) or not isinstance(event.get("result"),dict): continue
            messages.append({"role":"assistant","tool_call":event})
            messages.append({"role":"tool","name":str(event.get("tool","unknown")),"content":json.dumps(event["result"],ensure_ascii=False)})
        messages.append({"role":"assistant","content":assistant})
        result.append({"messages":messages,"metadata":{"task_id":r["task_id"],"format":"localcodex-tool-trace-v1"}})
    return result
