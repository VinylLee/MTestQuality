#!/usr/bin/env python3
"""Run pinned NLI auditors sequentially, validate them, then ensemble-filter."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from evaluate_nli_quality import canonical_model_label_order, load_rows, sha256_file, write_json, write_jsonl
from filter_nli_ensemble import (
    PROJECT_ROOT, DEFAULT_CONFIG, AUDIT_FILES, OUTPUT_FILES, aggregate_summaries,
    config_fingerprint, filter_dataset, load_audit, load_config, project_path,
    protect_output, source_positions, pair_key,
)


def model_runtime(config, model):
    return {**config["runtime"], **{key: model[key] for key in config["runtime"] if key in model}}


def validate_runtime(manifest, config, model):
    expected = model_runtime(config, model)
    for key in ("device", "fp16", "batch_size", "max_length"):
        if manifest["runtime"][key] != expected[key]:
            raise ValueError(f"Cached audit runtime differs: {model['id']}/{key}")
    # Old main-based requests are reusable only when their resolved commit is pinned.
    if manifest["resolved_revision"] != model["revision"]:
        raise ValueError("Cached checkpoint revision differs")


def run_model(config, model, input_path, output_dir, log_path):
    from transformers import AutoConfig
    cache = PROJECT_ROOT / "nli_model_cache/hub"
    hf_config = AutoConfig.from_pretrained(model["name"], revision=model["revision"], cache_dir=str(cache))
    if hf_config.num_labels != 3 or list(canonical_model_label_order(model["name"], hf_config, None)) != model["label_order"]:
        raise ValueError(f"Official checkpoint mapping differs: {model['name']}")
    runtime = model_runtime(config, model)
    command = [sys.executable, str(PROJECT_ROOT / "evaluate_nli_quality.py"), str(input_path),
               "--output-dir", str(output_dir), "--model", model["name"], "--revision", model["revision"],
               "--device", runtime["device"], "--batch-size", str(runtime["batch_size"]),
               "--max-length", str(runtime["max_length"]), "--confidence-threshold", "0.95"]
    if not runtime["fp16"]:
        command.append("--no-fp16")
    environment = os.environ.copy()
    environment.update({"HF_HUB_CACHE": str(cache), "TRANSFORMERS_CACHE": str(cache),
                        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    print(f"Running {model['id']}: {model['name']}@{model['revision']} -> {output_dir}", flush=True)
    with Path(log_path).open("w", encoding="utf-8") as log:
        # A child process owns the model. Its exit releases every model tensor and
        # CUDA allocation, including on failure; no other auditor shares that process.
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)


def validate_completed(directory, rows, input_hash, config, model):
    checks, health, manifest = load_audit(directory, rows, input_hash, model, config)
    validate_runtime(manifest, config, model)
    return checks, health, manifest


def verify_ensemble_resume(directory, config, input_hash):
    for name in OUTPUT_FILES:
        if not (directory / name).is_file():
            raise ValueError("Incomplete ensemble output; use a fresh output root")
    summary = json.loads((directory / "ensemble_summary.json").read_text())
    manifest = json.loads((directory / "model_manifest.json").read_text())
    if summary["input"]["sha256"] != input_hash or summary["config_sha256"] != config_fingerprint(config):
        raise ValueError("Existing ensemble does not match input/config")
    for name, digest in manifest["output_sha256"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Ensemble output hash differs: {name}")
    for model in config["models"]:
        record = manifest["models"][model["id"]]
        for filename, key in (("predictions.jsonl", "predictions_sha256"), ("summary.json", "summary_sha256")):
            if sha256_file(directory / model["id"] / filename) != record[key]:
                raise ValueError("Existing ensemble uses different audit artifacts")
    return summary


def audit_dataset(config, dataset, output_root, resume=False):
    input_path = project_path(config["dataset_paths"][dataset])
    directory = Path(output_root) / dataset
    protect_output(directory, input_path)
    rows = load_rows(input_path)
    if any("nli_quality" in row or "ensemble_quality" in row for row in rows):
        raise ValueError("Input must contain original rows, not audit-enriched rows")
    input_hash = sha256_file(input_path)
    sources = source_positions(rows)
    invalid = sum(row.get("is_source") is not True and
                  (row.get("is_source") is not False or
                   pair_key(row) not in sources) for row in rows)
    print(f"Dataset {dataset}: {len(rows)} records, {invalid} invalid augmented pair structures", flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    audit_dirs = {}
    for model in config["models"]:
        target = directory / model["id"]
        if target.exists():
            if not resume:
                raise FileExistsError(f"Audit already exists: {target}; use --resume or a new output root")
            validate_completed(target, rows, input_hash, config, model)
            print(f"Validated cached audit: {target}", flush=True)
        else:
            stage = Path(tempfile.mkdtemp(prefix=f".{model['id']}-", dir=directory))
            reuse = config.get("reuse_audits", {}).get(dataset, {}).get(model["id"])
            try:
                if reuse:
                    source = project_path(reuse)
                    validate_completed(source, rows, input_hash, config, model)
                    for filename in AUDIT_FILES:
                        shutil.copy2(source / filename, stage / filename)
                    write_json(stage / "reuse_provenance.json", {"source_directory": str(source),
                               "input_sha256": input_hash, "resolved_revision": model["revision"]})
                    print(f"Reused validated legacy audit: {source}", flush=True)
                else:
                    run_model(config, model, input_path, stage, stage / "run.log")
                metadata = json.loads((stage / "summary.json").read_text())
                metadata["outputs"] = {key: str(target / filename) for key, filename in (
                    ("predictions", "predictions.jsonl"), ("disagreements", "disagreements.jsonl"),
                    ("high_confidence_agreements", "high_confidence_agreements.jsonl"),
                    ("pair_quality", "pair_quality.jsonl"))}
                write_json(stage / "summary.json", metadata)
                validate_completed(stage, rows, input_hash, config, model)
                if sha256_file(input_path) != input_hash:
                    raise ValueError("Input changed during audit")
                stage.rename(target)
            except Exception:
                print(f"Audit failed; diagnostic files preserved in {stage}. No ensemble was published.", flush=True)
                raise
        audit_dirs[model["id"]] = target
    if sha256_file(input_path) != input_hash:
        raise ValueError("Input changed during audit")
    if (directory / "ensemble_summary.json").exists() and resume:
        summary = verify_ensemble_resume(directory, config, input_hash)
    else:
        summary = filter_dataset(config, dataset, audit_dirs, directory)
    print(f"Completed {dataset}: removed {summary['removed_augmented_count']}/{summary['augmented_count']} augmented rows", flush=True)
    return summary


def smoke_test(config, output_root):
    root = Path(output_root) / "smoke"
    if root.exists():
        raise FileExistsError("Smoke output already exists; select a fresh output root")
    root.mkdir(parents=True)
    rows = []
    pairs = [("Someone is moving.", 0), ("The person has a red hat.", 1), ("No one is walking.", 2)]
    for position, (hypothesis, label) in enumerate(pairs):
        for source in (True, False):
            rows.append({"premise": "A person is walking.", "hypothesis": hypothesis, "label": label,
                         "pair_id": f"smoke-{position}", "is_source": source, "idx": len(rows),
                         "mr_id": "smoke", "mr_type": "smoke"})
    input_path = root / "input.jsonl"
    write_jsonl(input_path, rows)
    input_hash = sha256_file(input_path)
    evidence = {}
    for model in config["models"]:
        directory = root / model["id"]
        directory.mkdir()
        run_model(config, model, input_path, directory, directory / "run.log")
        _, health, manifest = validate_completed(directory, rows, input_hash, config, model)
        evidence[model["id"]] = {"source_health": health, "manifest": manifest}
    write_json(root / "smoke_summary.json", {"input_sha256": input_hash, "models": evidence})
    print("All four pinned checkpoints passed real three-class smoke inference", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", nargs="+", choices=("snli", "mnlim", "mnlimm", "sick"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true", help="Reuse completed, fully validated outputs")
    parser.add_argument("--smoke-only", action="store_true", help="Verify four real models using a six-row fixture")
    args = parser.parse_args()
    config = load_config(args.config)
    root = (args.output_root or project_path(config["output_root"])).resolve()
    if args.smoke_only:
        smoke_test(config, root)
        return
    requested = args.datasets or config["datasets"]
    if set(requested) - set(config["datasets"]) or len(set(requested)) != len(requested):
        raise ValueError("Requested datasets must be unique and configured")
    summaries = []
    for dataset in requested:
        summaries.append(audit_dataset(config, dataset, root, args.resume))
    # A total report represents every configured dataset only after all completed.
    if set(requested) == set(config["datasets"]):
        write_json(root / "summary_all_datasets.json", aggregate_summaries(summaries))
    else:
        write_json(root / "summary_selected_datasets.json", aggregate_summaries(summaries))


if __name__ == "__main__":
    main()
