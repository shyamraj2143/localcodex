# Local Codex training data pipeline

All collection and preparation runs locally with the project's Python environment. No model training, checkpoint download, hosted API, or network access is needed. Task reports are appended best effort to `training/data/raw/trajectories.jsonl`; each workspace also retains the last 30 records in `.localcodex/memory/trajectories.jsonl`. Tool events contain bounded, redacted arguments/results. `.env`, environment variables and secret-named paths are excluded. Review sanitized records before approval: redaction is defense in depth.

## PowerShell workflow

```powershell
python -m training.scripts.collect
python -m training.scripts.sanitize --input training/data/raw/trajectories.jsonl --output training/data/raw/sanitized.jsonl
python -m training.scripts.validate --input training/data/raw/sanitized.jsonl
python -m training.scripts.review --input training/data/raw/sanitized.jsonl
python -m training.scripts.review --input training/data/raw/sanitized.jsonl --task-id <task-id> --decision approved --note "Reviewed evidence and privacy"
python -m training.scripts.split_dataset --input training/data/reviewed/approved.jsonl --seed 42
python -m training.scripts.export_sft --input training/data/processed/train.jsonl
python -m training.evaluation.evaluate --results training/evaluation/results.jsonl
```

Review without `--task-id` prints records. Decisions write `approved.jsonl`, `rejected.jsonl`, and `pending.jsonl` beneath `training/data/reviewed/`. Only a successful record with verification evidence can be approved or exported. Splitting needs at least three distinct task groups; duplicate groups stay together. `dataset_manifest.json` records source hash, seed and counts. Test is kept in `test.jsonl` and should remain isolated. Do not overwrite source files: sanitize always takes a distinct output path.

## Schema and training format

`schemas/trajectory.schema.json` defines schema version 1; the standard library validator in `scripts/validate.py` additionally checks duplicate IDs/records, test counts, unsupported success claims and trace completeness. `evaluation/tasks.jsonl` is a manually curated benchmark seed, separate from collected data. Record actual outcomes in `evaluation/results.jsonl`; `evaluate.py` computes completion, test, tool-call, false-success, invalid-file, repair and offline hosted-call metrics. GUI launch availability is reported separately. The benchmark evaluator summarizes recorded outcomes and deliberately does not execute generated code.

`export_sft.py` emits JSONL objects with `messages`. User instructions and final assistant reports are retained; observed tool calls/results appear as explicit `assistant.tool_call` and `tool` message fields. This is the documented local `localcodex-tool-trace-v1` format, not a claim of direct compatibility with every trainer's chat template. Convert the fields to the selected trainer's tool-call template and verify its parser before training. Plain text SFT does not itself guarantee tool execution.

## Optional future GPU training

On a separate compatible GPU machine, create a fresh environment and pin mutually compatible Transformers, TRL, PEFT, Accelerate, bitsandbytes, PyTorch and CUDA versions from their official compatibility guidance. Load the selected base checkpoint explicitly, apply LoRA/QLoRA adapters, train only reviewed `train.sft.jsonl`, tune evaluation on `validation.jsonl`, and report final metrics once on isolated `test.jsonl`. Verify license and data rights first. This project does not install or launch a trainer; 8 GB CPU-only hardware is not expected to run QLoRA efficiently. No checkpoints are downloaded automatically.
