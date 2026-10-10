#!/usr/bin/env python3
"""Audit merged MR data without modifying original data or historical outputs."""

import argparse
import collections
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
import unicodedata
from pathlib import Path

import evaluate_nli_quality as evaluator
import filter_nli_ensemble as ensemble
import refilter_ensemble_v1_1 as refilter

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs/mr_test_data_merged_quality_v1.json"
QUALITY = "merged_quality"
LABELS = evaluator.LABEL_NAME_TO_ID


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    evaluator.write_json(temporary, value)
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(refilter.canonical(value).encode()).hexdigest()


def exact_key(row):
    return row["premise"], row["hypothesis"]


def normalized_key(row):
    return tuple(" ".join(unicodedata.normalize("NFC", text).split()) for text in exact_key(row))


def audit_structure(rows):
    """Strict schema, with pair anomalies reported rather than repaired."""
    seen = set()
    sources = refilter.source_positions(rows)
    pair_groups = collections.defaultdict(list)
    issues = [[] for _ in rows]
    for i, row in enumerate(rows):
        for field in ("idx", "pair_id", "premise", "hypothesis", "label", "is_source", "mr_type", "mr_id"):
            if field not in row:
                raise ValueError(f"Missing {field} at row {i}")
        if any(k in row for k in (QUALITY, "nli_quality", "ensemble_quality", refilter.QUALITY)):
            raise ValueError(f"Reserved metadata at row {i}")
        if type(row["idx"]) is not int or row["idx"] in seen:
            raise ValueError(f"Invalid/duplicate idx at row {i}")
        seen.add(row["idx"])
        if type(row["is_source"]) is not bool:
            raise ValueError(f"Invalid source flag at row {i}")
        if type(row["label"]) is not int or row["label"] not in range(3):
            raise ValueError(f"Expected explicit 0/1/2 label encoding at row {i}")
        if not all(isinstance(row[k], str) for k in ("premise", "hypothesis", "mr_id", "mr_type")):
            raise ValueError(f"Invalid text/MR fields at row {i}")
        key = refilter.pair_key(row)
        if key is None:
            issues[i].append("invalid_pair_id")
        else:
            pair_groups[key].append(i)
        if not row["premise"].strip() or not row["hypothesis"].strip():
            issues[i].append("empty_text")
    for positions in pair_groups.values():
        source_count = sum(rows[i]["is_source"] for i in positions)
        if source_count != 1:
            for i in positions:
                issues[i].append("missing_source" if not source_count else "multiple_sources")
    exact, normalized = collections.defaultdict(list), collections.defaultdict(list)
    unchanged, format_only = [], []
    for i, row in enumerate(rows):
        exact[exact_key(row)].append(i)
        normalized[normalized_key(row)].append(i)
        source = sources.get(refilter.pair_key(row))
        if not row["is_source"] and source is not None:
            if exact_key(row) == exact_key(rows[source]):
                unchanged.append(i)
            elif normalized_key(row) == normalized_key(rows[source]):
                format_only.append(i)

    def groups(index, conflict):
        return [{"group_id": digest(key), "row_positions": positions,
                 "labels": sorted({rows[i]["label"] for i in positions}),
                 "source_positions": [i for i in positions if rows[i]["is_source"]]}
                for key, positions in index.items() if len(positions) > 1 and
                (len({rows[i]["label"] for i in positions}) > 1) is conflict]

    conflicts = groups(normalized, True)
    conflict_ids = {i: g["group_id"] for g in conflicts for i in g["row_positions"]}
    duplicates = []
    duplicate_ids = {}
    by_label = collections.defaultdict(list)
    for i, row in enumerate(rows):
        by_label[(*normalized_key(row), row["label"])].append(i)
    for key, positions in by_label.items():
        if len(positions) > 1:
            group = {"group_id": digest(key), "row_positions": positions,
                     "source_positions": [i for i in positions if rows[i]["is_source"]]}
            duplicates.append(group)
            duplicate_ids.update({i: group["group_id"] for i in positions})
    report = {"record_count": len(rows), "source_count": sum(r["is_source"] for r in rows),
              "structural_issues": [{"row_position": i, "issues": v} for i, v in enumerate(issues) if v],
              "exact_duplicates": groups(exact, False), "exact_conflicts": groups(exact, True),
              "normalized_duplicates": duplicates, "normalized_conflicts": conflicts,
              "normalization_only_groups": [{"group_id": digest(k), "row_positions": v}
                                             for k, v in normalized.items() if len({exact_key(rows[i]) for i in v}) > 1],
              "unchanged_augmentation_positions": unchanged, "format_only_augmentation_positions": format_only}
    return report, issues, conflict_ids, duplicate_ids


def rebuild_check(row, position, probabilities, token_count, provenance):
    if type(token_count) is not int or not 1 <= token_count <= 256:
        raise ValueError("Invalid tokenizer count")
    probs = {label: refilter.probability(probabilities[label]) for label in LABELS}
    if set(probabilities) != set(LABELS) or abs(sum(probs.values()) - 1) > 1e-5:
        raise ValueError("Invalid cached probabilities")
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    prediction, confidence = ranked[0]
    gold = evaluator.normalize_label(row["label"])
    check = {"row_position": position, "expected_label": gold, "predicted_label": prediction,
             "predicted_label_id": LABELS[prediction], "agreement": prediction == gold,
             "confidence": confidence, "probabilities": probs,
             "margin": evaluator.round_float(confidence - ranked[1][1]),
             "gold_probability": probs[gold], "gold_rank": next(i for i, (k, _) in enumerate(ranked, 1) if k == gold),
             "negative_log_likelihood": evaluator.round_float(-math.log(max(probs[gold], 1e-12))),
             "normalized_entropy": evaluator.round_float(-sum(p * math.log(max(p, 1e-12)) for p in probs.values()) / math.log(3)),
             "input_token_count": token_count, "at_max_length": token_count >= 256,
             "prediction_provenance": provenance}
    ensemble.validate_check(check, row, position, 1e-5)
    return check


def add_candidate(index, key, check, origin):
    """Disagreement in persisted probabilities makes this exact input ambiguous."""
    candidate = {"probabilities": check["probabilities"], "input_token_count": check["input_token_count"], "origins": [origin]}
    if key not in index:
        index[key] = candidate
    elif index[key] is not None:
        if (index[key]["probabilities"] != candidate["probabilities"] or
                index[key]["input_token_count"] != candidate["input_token_count"]):
            index[key] = None
        else:
            index[key]["origins"].append(origin)


def history_index(config, model):
    history = ensemble.load_config(ROOT / "configs/ensemble_filter_v1.json")
    historical_evaluator = read_json(ROOT / config["history_root"] / "validation_receipt.json")["delivered_code_sha256"]["evaluate_nli_quality.py"]
    if evaluator.sha256_file(ROOT / "evaluate_nli_quality.py") != historical_evaluator:
        raise ValueError("Cannot verify historical tokenizer implementation")
    result, receipts = {}, []
    for dataset in config["datasets"]:
        directory = ROOT / config["history_root"] / dataset
        manifest = read_json(directory / "model_manifest.json")
        record = manifest["models"][model["id"]]
        audit = directory / model["id"]
        for filename, field in (("predictions.jsonl", "predictions_sha256"), ("summary.json", "summary_sha256")):
            if evaluator.sha256_file(audit / filename) != record[field]:
                raise ValueError(f"Historical hash mismatch: {audit / filename}")
        rows = evaluator.load_rows(ROOT / history["dataset_paths"][dataset])
        checks, _, verified = ensemble.load_audit(audit, rows, evaluator.sha256_file(ROOT / history["dataset_paths"][dataset]), model, history)
        runtime = verified["runtime"]
        if runtime["max_length"] != 256 or runtime["fp16"] is not True:
            raise ValueError("Historical tokenizer/precision mismatch")
        receipts.append({"dataset": dataset, **verified, "tokenizer": {"revision": model["revision"], "use_fast": False,
                          "padding": True, "truncation": True, "max_length": 256},
                          "tokenizer_evidence": {"evaluator_sha256": historical_evaluator, "use_fast": False}})
        for i, (row, check) in enumerate(zip(rows, checks)):
            if check is None:
                continue
            if type(check.get("input_token_count")) is not int or not 1 <= check["input_token_count"] <= 256:
                raise ValueError("Historical token count invalid")
            add_candidate(result, exact_key(row), check, {"dataset": dataset, "row_position": i,
                          "predictions_sha256": verified["predictions_sha256"], "audit_directory": str(audit.relative_to(ROOT))})
    return result, receipts


def validate_artifacts(directory):
    hashes = read_json(directory / "artifact_hashes.json")
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file() and p.name != "artifact_hashes.json"}
    if set(hashes) != actual:
        raise ValueError("Artifact file set changed")
    for name, sha in hashes.items():
        if evaluator.sha256_file(directory / name) != sha:
            raise ValueError(f"Artifact hash mismatch: {name}")


def checkpoint_load(directory, identity, keys):
    path = directory / "checkpoint.json"
    if not path.exists():
        if any(directory.glob("batch-*.json")):
            raise ValueError("Unreceipted checkpoint shards")
        return {}, {"identity": identity, "shards": [], "oom_events": []}
    receipt = read_json(path)
    if receipt["identity"] != identity:
        raise ValueError("Checkpoint input/config/code identity mismatch")
    expected = {r["file"] for r in receipt["shards"]}
    # A crash between a shard rename and its receipt leaves one uncommitted shard;
    # keep it as diagnostics, and never use its predictions.
    for orphan in directory.glob("batch-*.json"):
        if orphan.name not in expected:
            orphan.rename(orphan.with_suffix(".uncommitted"))
    values = {}
    for shard in receipt["shards"]:
        path = directory / shard["file"]
        if evaluator.sha256_file(path) != shard["sha256"]:
            raise ValueError("Checkpoint shard hash mismatch")
        for item in read_json(path):
            position = item["position"]
            if type(position) is not int or position != len(values) or position >= len(keys):
                raise ValueError("Checkpoint positions invalid")
            rebuild_check({"label": 0}, position, item["probabilities"], item["input_token_count"], {})
            values[keys[position]] = item
    return values, receipt


def next_batch_size(size):
    return max(1, size // 2)


def infer_missing(config, model, keys, directory, identity):
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    directory.mkdir(parents=True, exist_ok=True)
    inputs = directory / "inputs.jsonl"
    if inputs.exists():
        saved = list(refilter.iter_jsonl(inputs))
        expected = [{"position": i, "premise": k[0], "hypothesis": k[1]} for i, k in enumerate(keys)]
        if saved != expected:
            raise ValueError("Checkpoint exact inputs changed")
    else:
        evaluator.write_jsonl(inputs, ({"position": i, "premise": k[0], "hypothesis": k[1]} for i, k in enumerate(keys)))
    values, receipt = checkpoint_load(directory, identity, keys)
    if len(values) == len(keys):
        return values, receipt
    runtime = config["runtime"]
    device = torch.device(runtime["device"])
    cache = ROOT / "nli_model_cache/hub"
    tokenizer = AutoTokenizer.from_pretrained(model["name"], revision=model["revision"], cache_dir=str(cache), use_fast=False, local_files_only=True)
    while True:
        free, _ = torch.cuda.mem_get_info(device)
        if free >= config["min_free_gpu_mib"] * 1024 * 1024:
            break
        print(f"{model['id']}: waiting for shared GPU memory ({free // 1048576} MiB free)", flush=True)
        time.sleep(config["gpu_wait_seconds"])
    network = AutoModelForSequenceClassification.from_pretrained(model["name"], revision=model["revision"], cache_dir=str(cache), local_files_only=True)
    if (network.config.num_labels != 3 or list(evaluator.canonical_model_label_order(model["name"], network.config, None)) != model["label_order"] or
            network.config._commit_hash != model["revision"]):
        raise ValueError("Local pinned checkpoint mapping/revision mismatch")
    network.to(device).eval()
    batch_size = receipt["shards"][-1]["batch_size"] if receipt["shards"] else runtime["batch_size"]
    receipt["runtime"] = {**runtime, "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                           "python_version": platform.python_version(), "device_name": torch.cuda.get_device_name(device),
                           "tokenizer_use_fast": False, "local_files_only": True}
    start = len(values)
    try:
        while start < len(keys):
            batch = keys[start:start + batch_size]
            try:
                with torch.inference_mode():
                    encoded = tokenizer([k[0] for k in batch], [k[1] for k in batch], padding=True, truncation=True, max_length=256, return_tensors="pt")
                    counts = encoded["attention_mask"].sum(dim=1).tolist()
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                    with torch.cuda.amp.autocast(enabled=True):
                        logits = network(**encoded).logits
                    probabilities = torch.softmax(logits.float(), dim=-1).cpu()
                    if not torch.isfinite(probabilities).all():
                        raise ValueError("Non-finite inference probabilities")
                    output = [{"position": start + i,
                               "probabilities": {label: evaluator.round_float(p[model["label_order"].index(label)]) for label in LABELS},
                               "input_token_count": int(n), "batch_size": batch_size}
                              for i, (p, n) in enumerate(zip(probabilities.tolist(), counts))]
                del encoded, logits, probabilities
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                receipt["oom_events"].append({"position": start, "batch_size": batch_size, "message": str(exc)})
                write_json(directory / "checkpoint.json", receipt)
                for name in ("encoded", "logits", "probabilities"):
                    if name in locals():
                        # Release tensors from the failed batch before retrying.
                        if name == "encoded": encoded = None
                        elif name == "logits": logits = None
                        else: probabilities = None
                torch.cuda.empty_cache()
                if batch_size == 1:
                    print(f"{model['id']}: OOM at batch 1, waiting for shared GPU", flush=True)
                    time.sleep(config["gpu_wait_seconds"])
                else:
                    batch_size = next_batch_size(batch_size)
                    print(f"{model['id']}: OOM, retrying batch size {batch_size}", flush=True)
                continue
            filename = f"batch-{len(receipt['shards']):06d}.json"
            write_json(directory / filename, output)
            receipt["shards"].append({"file": filename, "sha256": evaluator.sha256_file(directory / filename),
                                       "start": start, "count": len(output), "batch_size": batch_size})
            write_json(directory / "checkpoint.json", receipt)
            for key, item in zip(batch, output):
                values[key] = item
            start += len(batch)
            if len(receipt["shards"]) % 50 == 0 or start == len(keys):
                print(f"{model['id']}: inferred {start}/{len(keys)} unique inputs, batch {batch_size}", flush=True)
    finally:
        del network
        torch.cuda.empty_cache()
    return values, receipt


def write_audit(config, model, dataset, rows, checks, directory, history, checkpoint):
    directory.mkdir(parents=True)
    enriched = [{**r, "nli_quality": c} for r, c in zip(rows, checks)]
    evaluator.write_jsonl(directory / "predictions.jsonl", enriched)
    evaluator.write_jsonl(directory / "disagreements.jsonl", (r for r in enriched if not r["nli_quality"]["agreement"]))
    evaluator.write_jsonl(directory / "high_confidence_agreements.jsonl", (r for r in enriched if r["nli_quality"]["agreement"] and r["nli_quality"]["confidence"] >= .95))
    evaluator.write_jsonl(directory / "pair_quality.jsonl", evaluator.build_pair_rows(rows, checks, .95))
    runtime = {**config["runtime"], "torch_version": evaluator.torch.__version__, "transformers_version": evaluator.transformers.__version__,
               "mixed_prediction_origins": True, "batch_size_note": "requested batch size; actual historical/batch parameters in provenance"}
    summary = {"schema_version": 1, "input": {"path": config["dataset_paths"][dataset],
                  "sha256": evaluator.sha256_file(ROOT / config["dataset_paths"][dataset]), "record_count": len(rows)},
               "model": {"name": model["name"], "requested_revision": model["revision"], "resolved_revision": model["revision"], "model_logit_label_order": model["label_order"]},
               "runtime": runtime, "overall": evaluator.metric_summary(checks),
               "source_only": evaluator.metric_summary([c for r, c in zip(rows, checks) if r["is_source"]]),
               "augmented_only": evaluator.metric_summary([c for r, c in zip(rows, checks) if not r["is_source"]]),
               "augmentation_vs_source": evaluator.augmentation_vs_source_summary(rows, checks),
               "reuse_counts": dict(collections.Counter(c["prediction_provenance"]["kind"] for c in checks))}
    for field in ("label", "mr_type", "mr_id", "is_source"):
        summary["by_" + field] = evaluator.grouped_summaries(rows, checks, field)
    write_json(directory / "summary.json", summary)
    write_json(directory / "reuse_provenance.json", {"historical_audits": history,
               "checkpoint_identity": checkpoint["identity"], "actual_new_runtime": checkpoint.get("runtime"),
               "actual_batch_sizes": dict(collections.Counter(str(s["batch_size"]) for s in checkpoint["shards"])),
               "oom_events": checkpoint["oom_events"], "reuse_counts": summary["reuse_counts"]})
    write_json(directory / "completion.json", {"identity": checkpoint["identity"],
               "files": {p.name: evaluator.sha256_file(p) for p in sorted(directory.iterdir())}})


def finalize_rows(rows, structure):
    report, issues, conflicts, duplicates = structure
    representatives = {}
    for i, row in enumerate(rows):
        if row["is_source"]:
            representatives[(*normalized_key(row), row["label"])] = i
    result = []
    for i, row in enumerate(rows):
        key = (*normalized_key(row), row["label"])
        representative = representatives.get(key)
        if row["is_source"]:
            reason = "source_preserved"
        elif i in conflicts:
            reason = "label_conflict"
        elif row[refilter.QUALITY]["decision"] == "drop":
            reason = "model_v1_1"
        elif representative is not None:
            reason = "duplicate"
        else:
            reason = "augmentation_preserved"
            representatives[key] = i
            representative = i
        result.append({**row, QUALITY: {"row_position": i, "decision": "drop" if reason in ("label_conflict", "model_v1_1", "duplicate") else "keep",
                      "reason": reason, "structural_issues": issues[i], "conflict_group_id": conflicts.get(i),
                      "duplicate_group_id": duplicates.get(i), "duplicate_representative_position": representative,
                      "model_decision": row[refilter.QUALITY]["decision"], "model_reason": row[refilter.QUALITY]["reason"]}})
    return result


def validate_final(raw, rows, dataset, policy, evidence, structure):
    if len(raw) != len(rows):
        raise ValueError("Final row count mismatch")
    expected = finalize_rows(rows, structure)
    kept_keys = collections.defaultdict(list)
    for i, (original, row) in enumerate(zip(raw, rows)):
        if refilter.canonical(original) != refilter.canonical({k: v for k, v in row.items() if k not in (QUALITY, "ensemble_quality", refilter.QUALITY)}):
            raise ValueError("Raw fields changed")
        if row[QUALITY] != expected[i][QUALITY] or row[refilter.QUALITY] != refilter.decide_row(row, dataset, evidence[i], policy):
            raise ValueError("Final decision does not reproduce")
        if original["is_source"] and row[QUALITY]["decision"] != "keep":
            raise ValueError("Source removed")
        if row[QUALITY]["decision"] == "keep":
            kept_keys[(*normalized_key(row), row["label"])].append(i)
            if not row["is_source"] and i in structure[2]:
                raise ValueError("Conflicting augmentation retained")
    for positions in kept_keys.values():
        augmented = [i for i in positions if not rows[i]["is_source"]]
        if augmented and (len(positions) != 1 or len(augmented) != 1):
            raise ValueError("Redundant augmentation retained")
    return {"verified": True, "rows_checked": len(rows), "source_preserved": sum(r["is_source"] for r in raw),
            "raw_fields_unchanged": True, "decisions_reproduced": True, "conflicts_and_duplicates_checked": True}


def verify_backup(config):
    import tarfile
    manifest_path = ROOT / config["backup_manifest"]
    manifest = read_json(manifest_path)
    archive = ROOT / manifest["archive"]
    if evaluator.sha256_file(archive) != manifest["archive_sha256"]:
        raise ValueError("Backup archive hash mismatch")
    with tarfile.open(archive) as handle:
        for record in manifest["files"]:
            path = ROOT / record["path"]
            data = path.read_bytes()
            if (hashlib.sha256(data).hexdigest() != record["sha256"] or len(data) != record["size"] or
                    len(data.splitlines()) != record["line_count"] or handle.extractfile(record["path"]).read() != data):
                raise ValueError("Backup/original mismatch")
    receipt = read_json(ROOT / config["remote_backup_receipt"])
    expected = {r["path"]: r["sha256"] for r in manifest["files"]}
    expected.update({str(archive.relative_to(ROOT)): evaluator.sha256_file(archive),
                     str(manifest_path.relative_to(ROOT)): evaluator.sha256_file(manifest_path)})
    if receipt.get("verified") is not True or receipt["file_sha256"] != expected:
        raise ValueError("Missing verified remote backup receipt")
    return receipt


def run(config, output_root=None, resume=False):
    config = copy.deepcopy(config)
    output = Path(output_root or ROOT / config["output_root"]).resolve()
    protected = [ROOT / "data", ROOT / "backups", ROOT / "outputs/ensemble_filter_v1", ROOT / "outputs/ensemble_filter_v1_1", ROOT / "outputs/snli_v3_3_cross_encoder_deberta_v3_large"]
    if output.exists() and resume:
        validate_artifacts(output)
        receipt = read_json(output / "validation_receipt.json")
        if receipt["identity"]["config_sha256"] != digest(config):
            raise ValueError("Published resume config differs")
        for name, sha in receipt["identity"]["implementation_sha256"].items():
            if evaluator.sha256_file(ROOT / name) != sha:
                raise ValueError("Published resume implementation differs")
        for path, before in read_json(output / "run_state.json")["protected"].items():
            refilter.verify_integrity(path, before)
        print(f"Verified completed output: {output}", flush=True)
        return read_json(output / "summary_all_datasets.json")["totals"]
    refilter.protect_output(output, protected)
    if config["datasets"] != list(refilter.DATASETS) or config["runtime"] != {"device": "cuda:1", "batch_size": 16, "max_length": 256, "fp16": True}:
        raise ValueError("Merged run requires all four datasets and approved runtime")
    backup_receipt = verify_backup(config)
    stage = output.with_name("." + output.name + ".inprogress")
    if stage.exists() and not resume:
        raise FileExistsError(f"Work directory exists; use --resume: {stage}")
    stage.mkdir(parents=True, exist_ok=True)
    identity = {"config_sha256": digest(config), "implementation_sha256": {p: evaluator.sha256_file(ROOT / p) for p in
                ("run_merged_quality_audit.py", "evaluate_nli_quality.py", "filter_nli_ensemble.py", "refilter_ensemble_v1_1.py")},
                "input_sha256": {d: evaluator.sha256_file(ROOT / config["dataset_paths"][d]) for d in config["datasets"]}}
    state_path = stage / "run_state.json"
    if state_path.exists():
        state = read_json(state_path)
        if state["identity"] != identity:
            raise ValueError("Resume identity changed")
        for path, snapshot in state["protected"].items():
            refilter.verify_integrity(path, snapshot)
    else:
        state = {"identity": identity, "protected": {str(p): refilter.snapshot_tree(p) for p in protected}, "backup_receipt": backup_receipt}
        write_json(state_path, state)
    datasets = {d: evaluator.load_rows(ROOT / config["dataset_paths"][d]) for d in config["datasets"]}
    structures = {d: audit_structure(rows) for d, rows in datasets.items()}
    baseline_config = ensemble.load_config(ROOT / "configs/ensemble_filter_v1.json")
    baseline_config.update(dataset_paths=config["dataset_paths"], runtime=config["runtime"])
    baseline_config.pop("reuse_audits", None)
    for model in baseline_config["models"]:
        model.pop("batch_size", None)
    all_keys = list(dict.fromkeys(exact_key(r) for rows in datasets.values() for r in rows))
    for model in baseline_config["models"]:
        index, receipts = history_index(config, model)
        missing = [key for key in all_keys if key not in index or index[key] is None]
        print(f"{model['id']}: {len(all_keys)} unique inputs, {len(all_keys) - len(missing)} reusable, {len(missing)} inference ({sum(k in index and index[k] is None for k in missing)} ambiguous)", flush=True)
        checkpoint_identity = {**identity, "model": model, "missing_inputs_sha256": digest(missing), "history_receipts_sha256": digest(receipts)}
        new_values, checkpoint = infer_missing(config, model, missing, stage / "inference" / model["id"], checkpoint_identity)
        for dataset, rows in datasets.items():
            directory = stage / "baseline_v1" / dataset / model["id"]
            if directory.exists() and not (directory / "completion.json").exists():
                directory.rename(directory.with_name(directory.name + f".incomplete-{time.time_ns()}"))
            if directory.exists():
                complete = read_json(directory / "completion.json")
                if complete["identity"] != checkpoint_identity:
                    raise ValueError("Audit resume identity mismatch")
                for name, sha in complete["files"].items():
                    if evaluator.sha256_file(directory / name) != sha:
                        raise ValueError("Completed audit changed")
                ensemble.load_audit(directory, rows, identity["input_sha256"][dataset], model, baseline_config)
                continue
            checks = []
            for i, row in enumerate(rows):
                key = exact_key(row)
                old = index.get(key)
                if old is not None:
                    probs, count = old["probabilities"], old["input_token_count"]
                    origin = {"kind": "historical_reuse", "origins": old["origins"]}
                else:
                    item = new_values[key]
                    probs, count = item["probabilities"], item["input_token_count"]
                    origin = {"kind": "new_inference", "unique_input_position": item["position"], "actual_batch_size": item["batch_size"],
                              "checkpoint_identity_sha256": digest(checkpoint_identity), "cache_status": "ambiguous" if key in index else "miss"}
                checks.append(rebuild_check(row, i, probs, count, origin))
            write_audit(config, model, dataset, rows, checks, directory, receipts, checkpoint)
    for dataset in config["datasets"]:
        directory = stage / "baseline_v1" / dataset
        if (directory / "ensemble_summary.json").exists():
            from run_ensemble_audit import verify_ensemble_resume
            verify_ensemble_resume(directory, baseline_config, identity["input_sha256"][dataset])
        else:
            ensemble.filter_dataset(baseline_config, dataset, {m["id"]: directory / m["id"] for m in baseline_config["models"]}, directory)
        # Store stable published paths before V1.1 snapshots these manifests.
        path = directory / "model_manifest.json"
        manifest = read_json(path)
        for model_id, record in manifest["models"].items():
            record["audit_directory"] = str(output / "baseline_v1" / dataset / model_id)
        write_json(path, manifest)
    policy = refilter.load_config(ROOT / "configs/ensemble_filter_v1_1.json")
    policy.update(dataset_paths=config["dataset_paths"], source_predictions_root=str(stage / "baseline_v1"), output_root=str(stage / "model_v1_1"))
    if not (stage / "model_v1_1").exists():
        refilter.run_refilter(policy)
    model_summary = read_json(stage / "model_v1_1/summary_all_datasets.json")
    summaries, validations = {}, {}
    for dataset, raw in datasets.items():
        model_dir = stage / "model_v1_1" / dataset
        manifest = read_json(model_dir / "model_manifest.json")
        for name, sha in manifest["output_sha256"].items():
            if evaluator.sha256_file(model_dir / name) != sha:
                raise ValueError("V1.1 artifact changed")
        model_rows = list(refilter.iter_jsonl(model_dir / "ensemble_predictions.jsonl"))
        rows = finalize_rows(model_rows, structures[dataset])
        evidence, _ = refilter.prepare_evidence(model_rows, manifest["models"], stage / "baseline_v1" / dataset)
        validations[dataset] = validate_final(raw, rows, dataset, policy, evidence, structures[dataset])
        directory = stage / dataset
        directory.mkdir(exist_ok=True)
        kept = [r for r in rows if r[QUALITY]["decision"] == "keep"]
        removed = [r for r in rows if r[QUALITY]["decision"] == "drop"]
        conflicts = [r for r in removed if r[QUALITY]["reason"] == "label_conflict"]
        warnings = [r for r in rows if r["is_source"] and (r[QUALITY]["conflict_group_id"] or r[QUALITY]["duplicate_group_id"] or r[QUALITY]["structural_issues"])]
        for name, selected in (("ensemble_predictions.jsonl", rows), ("filtered.jsonl", kept), ("removed.jsonl", removed),
                               ("conflict_quarantine.jsonl", conflicts), ("source_warnings.jsonl", warnings)):
            evaluator.write_jsonl(directory / name, selected)
        summary = {"input_count": len(rows), "source_count": sum(r["is_source"] for r in rows),
                   "augmented_count": sum(not r["is_source"] for r in rows), "filtered_count": len(kept), "removed_count": len(removed),
                   "removed_source_count": 0, "exclusive_removal_reasons": dict(collections.Counter(r[QUALITY]["reason"] for r in removed)),
                   "source_warning_count": len(warnings), "conflict_quarantine_count": len(conflicts),
                   "conflict_quarantine_is_removed_subset": True, "model_audit": model_summary["datasets"][dataset],
                   "before_distribution": refilter.group_counts(rows, refilter.dimensions()),
                   "after_distribution": refilter.group_counts(kept, refilter.dimensions()),
                   "removed_distribution": refilter.group_counts(removed, refilter.dimensions()), "threshold_sensitivity": {}}
        for threshold in (.90, .95, .99):
            alternate = [{**r, refilter.QUALITY: refilter.decide_row(r, dataset, e, policy, threshold)} for r, e in zip(model_rows, evidence)]
            final = finalize_rows(alternate, structures[dataset])
            summary["threshold_sensitivity"][f"{threshold:.2f}"] = {"filtered_count": sum(r[QUALITY]["decision"] == "keep" for r in final),
                     "exclusive_removal_reasons": dict(collections.Counter(r[QUALITY]["reason"] for r in final if r[QUALITY]["decision"] == "drop"))}
        if len(rows) != len(kept) + len(removed) or sum(summary["exclusive_removal_reasons"].values()) != len(removed):
            raise ValueError("Final conservation failure")
        write_json(directory / "group_report.json", structures[dataset][0])
        write_json(directory / "ensemble_summary.json", summary)
        write_json(directory / "model_manifest.json", {"input_sha256": identity["input_sha256"][dataset], "config_sha256": identity["config_sha256"],
                   "models": manifest["models"], "validation": validations[dataset], "decision_order": ["source", "label_conflict", "model_v1_1", "duplicate"],
                   "normalization": "NFC Unicode and collapsed whitespace, comparison only, per dataset",
                   "baseline_v1_manifest_sha256": evaluator.sha256_file(stage / "baseline_v1" / dataset / "model_manifest.json"),
                   "model_v1_1_manifest_sha256": evaluator.sha256_file(model_dir / "model_manifest.json"),
                   "output_sha256": {p.name: evaluator.sha256_file(p) for p in directory.iterdir() if p.is_file() and p.name != "model_manifest.json"}})
        summaries[dataset] = summary
        print(f"{dataset}: kept {len(kept)}, removed {len(removed)} {summary['exclusive_removal_reasons']}", flush=True)
    integrity = {path: refilter.verify_integrity(path, snapshot) for path, snapshot in state["protected"].items()}
    totals = {key: sum(s[key] for s in summaries.values()) for key in ("input_count", "source_count", "augmented_count", "filtered_count", "removed_count", "removed_source_count")}
    write_json(stage / "summary_all_datasets.json", {"datasets": summaries, "totals": totals, "integrity": integrity, "identity": identity})
    write_json(stage / "validation_receipt.json", {"verified": True, "datasets": validations, "integrity": integrity, "identity": identity, "remote_backup": backup_receipt})
    # A portable receipt binds every staged artifact, including checkpoints.
    write_json(stage / "artifact_hashes.json", {str(p.relative_to(stage)): evaluator.sha256_file(p) for p in sorted(stage.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"})
    report = ["# Merged MR quality audit", "", f"Input {totals['input_count']}; source {totals['source_count']}; augmented {totals['augmented_count']}.", "",
              "| Dataset | Input | Kept | Conflict | Model V1.1 | Duplicate |", "|---|---:|---:|---:|---:|---:|"]
    for dataset, s in summaries.items():
        c = s["exclusive_removal_reasons"]
        report.append(f"| {dataset} | {s['input_count']} | {s['filtered_count']} | {c.get('label_conflict', 0)} | {c.get('model_v1_1', 0)} | {c.get('duplicate', 0)} |")
    report.extend(["", "All source rows retained. Original fields and order retained. Conflict quarantine is a subset of removed rows.",
                   "", "Same-label duplicates retain every source; otherwise the earliest augmentation surviving model filtering is retained.",
                   "", "V1 and V1.1 evidence, historical/new prediction provenance, actual batch sizes, threshold sensitivity and integrity receipts are included."])
    (stage / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    # Include the final report too; no self-referential hash.
    hashes = read_json(stage / "artifact_hashes.json")
    hashes["REPORT.md"] = evaluator.sha256_file(stage / "REPORT.md")
    write_json(stage / "artifact_hashes.json", hashes)
    validate_artifacts(stage)
    refilter.protect_output(output, protected)
    os.rename(stage, output)
    print(f"Published {output}: {totals}", flush=True)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    try:
        run(config, args.output_root, args.resume)
    except Exception:
        output = Path(args.output_root or ROOT / config["output_root"])
        stage = output.with_name("." + output.name + ".inprogress")
        if stage.is_dir():
            with (stage / "failures.log").open("a", encoding="utf-8") as handle:
                traceback.print_exc(file=handle)
        raise


if __name__ == "__main__":
    main()
