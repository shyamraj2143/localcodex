"""Command-line interface for offline dataset work and safe training estimates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from training.data import DatasetError, load_records, normalize_records, split_records, write_jsonl
from training.doctor import hardware_report
from training.benchmark import compare_results
from training.trajectory import curate_trajectories, write_trajectory_jsonl

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "config.json"
DEFAULT_DATA = ROOT / "training_data"


def _config(path: str | None) -> dict:
    config_path = Path(path) if path else DEFAULT_CONFIG
    try:
        return json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Could not read configuration {config_path}: {exc}") from exc


def _report(path: str, records: list[dict]) -> tuple[list[dict], dict]:
    source = load_records(path)
    return normalize_records(source)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.cli", description="Offline-first Local Codex training data tools")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report hardware and Ollama training compatibility")
    datasets = sub.add_parser("datasets", help="Inspect and prepare local datasets")
    dataset_commands = datasets.add_subparsers(dest="dataset_command", required=True)
    validate = dataset_commands.add_parser("validate")
    validate.add_argument("--input", required=True)
    prepare = dataset_commands.add_parser("prepare")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", default=str(DEFAULT_DATA / "prepared"))
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--train", type=float, default=0.8)
    prepare.add_argument("--validation", type=float, default=0.1)
    stats = dataset_commands.add_parser("stats")
    stats.add_argument("--input", required=True)
    training = sub.add_parser("train", help="Estimate a training run; execution requires separately verified ML setup")
    training.add_argument("--config")
    training.add_argument("--dry-run", action="store_true")
    benchmark = sub.add_parser("benchmark", help="Compare recorded, independently verified benchmark outcomes")
    benchmark.add_argument("--base-results", required=True)
    benchmark.add_argument("--candidate-results", required=True)
    trajectories = sub.add_parser("trajectories", help="Curate workspace task trajectories for later training-data review")
    trajectory_commands = trajectories.add_subparsers(dest="trajectory_command", required=True)
    trajectory_export = trajectory_commands.add_parser("export")
    trajectory_export.add_argument("--input", required=True, help="Workspace .localcodex/memory/trajectories.jsonl")
    trajectory_export.add_argument("--output-dir", default=str(DEFAULT_DATA / "trajectories"))

    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            report = hardware_report(ROOT, _config(None))
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "datasets":
            source = load_records(args.input)
            records, report = normalize_records(source)
            if args.dataset_command in {"validate", "stats"}:
                print(json.dumps(report, indent=2))
                return 0 if report["rejected"] == 0 else 2
            partitions = split_records(records, seed=args.seed, train=args.train, validation=args.validation)
            output = Path(args.output).resolve()
            allowed_root = (ROOT / "training_data").resolve()
            if output != allowed_root and allowed_root not in output.parents:
                raise DatasetError(f"Output must stay under the application training-data directory: {allowed_root}")
            output.mkdir(parents=True, exist_ok=True)
            for name, rows in partitions.items():
                write_jsonl(output / f"{name}.jsonl", rows)
            manifest = {"source": str(Path(args.input).resolve()), "source_records": report["input"], "prepared_records": report["accepted"], "cleaning": report, "split_seed": args.seed, "split_counts": {name: len(rows) for name, rows in partitions.items()}, "provenance": "Imported local file; license and training rights must be verified by the dataset owner before training."}
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(json.dumps(manifest, indent=2))
            return 0
        if args.command == "train":
            config = _config(args.config)
            report = hardware_report(ROOT, config)
            summary = {"dry_run": args.dry_run, "base_model": report["model_identity"], "hardware_compatible": report["training_compatible"], "training_data_dir": str(DEFAULT_DATA.resolve()), "estimated_requirements": report["estimated_requirements"], "limitations": report["limitations"], "action": "No downloads or training started."}
            print(json.dumps(summary, indent=2))
            if args.dry_run:
                return 0
            print("Training was not started: this release implements data preparation and compatibility estimation; a verified trainer backend and source checkpoint are not configured.", file=sys.stderr)
            return 3
        if args.command == "benchmark":
            base = load_records(args.base_results)
            candidate = load_records(args.candidate_results)
            report = compare_results(base, candidate)
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "trajectories":
            rows = load_records(args.input)
            verified, diagnostics = curate_trajectories(rows)
            output = Path(args.output_dir).resolve()
            allowed_root = (ROOT / "training_data").resolve()
            if output != allowed_root and allowed_root not in output.parents:
                raise DatasetError(f"Output must stay under the application training-data directory: {allowed_root}")
            output.mkdir(parents=True, exist_ok=True)
            normalized, cleaning = normalize_records(verified)
            write_jsonl(output / "verified.jsonl", normalized)
            write_trajectory_jsonl(output / "diagnostics.jsonl", diagnostics)
            print(json.dumps({"verified_examples": len(normalized), "diagnostic_trajectories": len(diagnostics),
                              "cleaning": cleaning, "verified_output": str(output / "verified.jsonl"),
                              "diagnostics_output": str(output / "diagnostics.jsonl"),
                              "note": "Review provenance, privacy, and training rights before model training."}, indent=2))
            return 0
    except (DatasetError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
