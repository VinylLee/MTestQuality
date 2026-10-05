#!/usr/bin/env python3
"""Validate independent NLI audits and conservatively filter augmented rows."""

import argparse
import collections
import json
import math
import os
import re
import tempfile
from pathlib import Path

from evaluate_nli_quality import (
    LABEL_NAME_TO_ID, load_rows, normalize_label, round_float,
    sha256_file, write_json, write_jsonl,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs/ensemble_filter_v1.json"
OUTPUT_FILES = ("ensemble_predictions.jsonl", "filtered.jsonl", "removed.jsonl",
                "ensemble_summary.json", "model_manifest.json")
AUDIT_FILES = ("predictions.jsonl", "disagreements.jsonl", "high_confidence_agreements.jsonl",
               "pair_quality.jsonl", "summary.json")


def project_path(value):
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def config_fingerprint(config):
    import hashlib
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("policy_name") != "conservative_consensus_v1":
        raise ValueError("Unsupported filtering policy")
    datasets = config.get("datasets", [])
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("Dataset ids must be nonempty and unique")
    if set(datasets) - {"snli", "mnlim", "mnlimm", "sick"}:
        raise ValueError("Unknown dataset id")
    for dataset in datasets:
        project_path(config["dataset_paths"][dataset])
    models = config.get("models", [])
    if len(models) != 4 or len({m["id"] for m in models}) != 4:
        raise ValueError("Exactly four distinct model ids are required")
    if len({m["name"] for m in models}) != 4:
        raise ValueError("Auditor checkpoints must be distinct")
    for model in models:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", model["id"]):
            raise ValueError("Unsafe model output id")
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise ValueError("Each auditor must be pinned to a full commit SHA")
        if len(model["label_order"]) != 3 or set(model["label_order"]) != set(LABEL_NAME_TO_ID):
            raise ValueError("Invalid model label order")
    for key in ("source_confidence_threshold", "followup_confidence_threshold", "source_accuracy_floor"):
        if not isinstance(config[key], (int, float)) or isinstance(config[key], bool) or not 0 <= config[key] <= 1:
            raise ValueError(f"Invalid {key}")
    for key in ("min_eligible_voters", "min_wrong_consensus"):
        if type(config[key]) is not int or not 1 <= config[key] <= 4:
            raise ValueError(f"Invalid {key}")
    if type(config["block_on_high_conf_gold_support"]) is not bool:
        raise ValueError("Gold-blocker setting must be boolean")
    tolerance = config["probability_sum_tolerance"]
    if not isinstance(tolerance, (int, float)) or not 0 < tolerance <= 0.001:
        raise ValueError("Invalid probability sum tolerance")
    runtime = config["runtime"]
    for model in models:
        effective = {**runtime, **{key: model[key] for key in runtime if key in model}}
        if (type(effective["batch_size"]) is not int or effective["batch_size"] < 1 or
                type(effective["max_length"]) is not int or effective["max_length"] < 2):
            raise ValueError("Invalid inference sizes")
        if type(effective["fp16"]) is not bool:
            raise ValueError("fp16 must be boolean")
        if effective["device"] == "cpu" and effective["fp16"]:
            raise ValueError("CPU inference requires fp16=false")
    return config


def protect_output(output_dir, input_path, filenames=OUTPUT_FILES):
    output_dir, input_path = Path(output_dir).resolve(), Path(input_path).resolve()
    if output_dir == input_path or output_dir in input_path.parents:
        raise ValueError("Output directory must not contain the input dataset")
    for name in filenames:
        target = output_dir / name
        if target.is_symlink() or target.resolve() == input_path:
            raise ValueError(f"Unsafe output target: {target}")


def pair_key(row):
    value = row.get("pair_id")
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value == "":
        return None
    # Do not accidentally merge integer 1 with string "1".
    return (type(value).__name__, value)


def source_positions(rows):
    grouped = collections.defaultdict(list)
    for position, row in enumerate(rows):
        key = pair_key(row)
        if row.get("is_source") is True and key is not None:
            grouped[key].append(position)
    return {key: positions[0] for key, positions in grouped.items() if len(positions) == 1}


def finite_probability(value, context):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid probability: {context}")
    return float(value)


def validate_check(check, row, position, tolerance):
    if not isinstance(check, dict) or type(check.get("row_position")) is not int or check["row_position"] != position:
        raise ValueError(f"Missing or mismatched row_position at row {position}")
    gold = normalize_label(row["label"])
    if "expected_label" in check and normalize_label(check["expected_label"]) != gold:
        raise ValueError(f"Gold label mismatch at row {position}")
    # A position-aligned but incomplete individual prediction is not deletion evidence.
    required = {"expected_label", "predicted_label", "confidence", "probabilities"}
    if not required.issubset(check):
        return None
    prediction = normalize_label(check["predicted_label"])
    probabilities = check["probabilities"]
    if not isinstance(probabilities, dict) or set(probabilities) != set(LABEL_NAME_TO_ID):
        raise ValueError(f"Invalid three-class probabilities at row {position}")
    probs = {label: finite_probability(value, f"row {position}/{label}") for label, value in probabilities.items()}
    if abs(sum(probs.values()) - 1.0) > tolerance:
        raise ValueError(f"Probability sum mismatch at row {position}")
    confidence = finite_probability(check["confidence"], f"row {position}/confidence")
    if abs(confidence - probs[prediction]) > tolerance or max(probs.values()) > probs[prediction]:
        raise ValueError(f"Top-1/confidence mismatch at row {position}")
    if "agreement" in check and check["agreement"] is not (prediction == gold):
        raise ValueError(f"Agreement field mismatch at row {position}")
    # Threshold the persisted class probability itself, not a potentially
    # inconsistent confidence field within the serialization tolerance.
    return {**check, "predicted_label": prediction, "expected_label": gold,
            "confidence": probs[prediction]}


def source_health(rows, checks, threshold):
    positions = [i for i, row in enumerate(rows) if row.get("is_source") is True]
    correct = sum(checks[i] is not None and checks[i]["predicted_label"] == normalize_label(rows[i]["label"]) for i in positions)
    confident = [i for i in positions if checks[i] is not None and checks[i]["confidence"] >= threshold]
    high_correct = sum(checks[i]["predicted_label"] == normalize_label(rows[i]["label"]) for i in confident)
    by_label = {}
    for label in LABEL_NAME_TO_ID:
        selected = [i for i in positions if normalize_label(rows[i]["label"]) == label]
        n_correct = sum(checks[i] is not None and checks[i]["predicted_label"] == label for i in selected)
        by_label[label] = {"count": len(selected), "correct": n_correct,
                           "accuracy": round_float(n_correct / len(selected)) if selected else None}
    return {"count": len(positions), "correct": correct,
            "accuracy": round_float(correct / len(positions)) if positions else None,
            "high_confidence_threshold": threshold, "high_confidence_count": len(confident),
            "high_confidence_correct": high_correct,
            "high_confidence_accuracy": round_float(high_correct / len(confident)) if confident else None,
            "high_confidence_coverage": round_float(len(confident) / len(positions)) if positions else None,
            "by_label": by_label}


def load_audit(directory, rows, input_sha256, model, config):
    directory = Path(directory)
    for name in AUDIT_FILES:
        if not (directory / name).is_file():
            raise ValueError(f"Incomplete audit: {directory / name}")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary["input"]["sha256"] != input_sha256 or summary["input"]["record_count"] != len(rows):
        raise ValueError(f"Audit input SHA-256/count mismatch: {directory}")
    metadata = summary["model"]
    if metadata["name"] != model["name"] or metadata["resolved_revision"] != model["revision"]:
        raise ValueError(f"Checkpoint/revision mismatch: {directory}")
    if metadata["model_logit_label_order"] != model["label_order"]:
        raise ValueError(f"Label order mismatch: {directory}")
    for key in ("requested_revision", "resolved_revision"):
        if not metadata.get(key):
            raise ValueError(f"Missing {key}: {directory}")
    runtime = summary["runtime"]
    for key in ("transformers_version", "torch_version", "device", "fp16", "batch_size", "max_length"):
        if key not in runtime:
            raise ValueError(f"Missing runtime metadata {key}: {directory}")
    checks = []
    with (directory / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise ValueError(f"Blank prediction row: {directory}")
            position = len(checks)
            if position >= len(rows):
                raise ValueError(f"Excess predictions: {directory}")
            enriched = json.loads(line)
            if not isinstance(enriched, dict):
                raise ValueError("Prediction must be an object")
            original = {key: value for key, value in enriched.items() if key != "nli_quality"}
            if (original != rows[position] or
                    json.dumps(original, sort_keys=True, allow_nan=False) !=
                    json.dumps(rows[position], sort_keys=True, allow_nan=False)):
                raise ValueError(f"Original fields differ at row {position}: {directory}")
            checks.append(validate_check(enriched.get("nli_quality"), rows[position], position,
                                         config["probability_sum_tolerance"]))
    if len(checks) != len(rows):
        raise ValueError(f"Missing predictions: {directory}")
    health = source_health(rows, checks, config["source_confidence_threshold"])
    if health["accuracy"] is not None and health["accuracy"] < config["source_accuracy_floor"]:
        raise ValueError(f"Source accuracy below safety floor: {directory}: {health['accuracy']}")
    manifest = {"model_name": model["name"], "requested_revision": metadata["requested_revision"],
                "resolved_revision": metadata["resolved_revision"], "label_order": model["label_order"],
                "mapping_source": model.get("mapping_source"), "runtime": runtime,
                "input_sha256": input_sha256, "record_count": len(rows),
                "audit_directory": str(directory.resolve()),
                "predictions_sha256": sha256_file(directory / "predictions.jsonl"),
                "summary_sha256": sha256_file(directory / "summary.json")}
    return checks, health, manifest


def decide_row(row, position, rows, sources, audits, config, followup_threshold=None):
    threshold = config["followup_confidence_threshold"] if followup_threshold is None else followup_threshold
    result = {"policy": config["policy_name"], "decision": "keep", "reason": "source_preserved",
              "row_position": position, "eligible_voter_count": 0, "wrong_consensus_label": None,
              "wrong_consensus_count": 0, "high_conf_gold_support_count": 0,
              "supporting_models": [], "per_model": {}}
    if row.get("is_source") is True:
        for model_id, checks in audits.items():
            check = checks[position]
            result["per_model"][model_id] = {"source_prediction": check["predicted_label"] if check else None,
                                                "source_confidence": check["confidence"] if check else None}
        return result
    source_position = sources.get(pair_key(row))
    if source_position is None or row.get("is_source") is not False:
        result["reason"] = "invalid_pair_structure"
        return result
    source_gold = normalize_label(rows[source_position]["label"])
    gold = normalize_label(row["label"])
    wrong_votes = collections.defaultdict(list)
    incomplete = set(audits) != {model["id"] for model in config["models"]}
    for model_id, checks in audits.items():
        source, augmented = checks[source_position], checks[position]
        eligible = bool(source and source["predicted_label"] == source_gold and
                        source["confidence"] >= config["source_confidence_threshold"])
        wrong = bool(eligible and augmented and augmented["predicted_label"] != gold and
                     augmented["confidence"] >= threshold)
        supports_gold = bool(eligible and augmented and augmented["predicted_label"] == gold and
                             augmented["confidence"] >= threshold)
        incomplete |= source is None or augmented is None
        result["eligible_voter_count"] += int(eligible)
        result["high_conf_gold_support_count"] += int(supports_gold)
        if wrong:
            wrong_votes[augmented["predicted_label"]].append(model_id)
        result["per_model"][model_id] = {
            "source_eligible": eligible, "source_prediction": source["predicted_label"] if source else None,
            "source_confidence": source["confidence"] if source else None,
            "augmented_prediction": augmented["predicted_label"] if augmented else None,
            "augmented_confidence": augmented["confidence"] if augmented else None,
            "high_conf_wrong_vote": wrong, "high_conf_gold_support": supports_gold}
    if wrong_votes:
        # Stable tie break; a split vote never meets the default 3-voter criterion.
        consensus = sorted(wrong_votes, key=lambda label: (-len(wrong_votes[label]), label))[0]
        result["wrong_consensus_label"] = consensus
        result["wrong_consensus_count"] = len(wrong_votes[consensus])
        result["supporting_models"] = wrong_votes[consensus]
    if incomplete:
        result["reason"] = "incomplete_model_results"
    elif result["eligible_voter_count"] < config["min_eligible_voters"]:
        result["reason"] = "insufficient_eligible_voters"
    elif config["block_on_high_conf_gold_support"] and result["high_conf_gold_support_count"]:
        result["reason"] = "high_confidence_gold_support"
    elif result["wrong_consensus_count"] < config["min_wrong_consensus"]:
        result["reason"] = "insufficient_wrong_consensus"
    else:
        result["decision"] = "drop"
        result["reason"] = "high_confidence_wrong_consensus"
    return result


def group_value(value):
    return "<missing>" if value is None else str(value)


def count_summary(rows, decisions, dataset):
    totals = {"input_count": len(rows), "source_count": 0, "augmented_count": 0,
              "kept_augmented_count": 0, "removed_augmented_count": 0, "removed_source_count": 0}
    groups = {key: {} for key in ("label", "mr_type", "mr_id", "dataset", "wrong_consensus_label",
                                   "decision_reason", "eligible_voter_count")}
    for row, decision in zip(rows, decisions):
        source = row.get("is_source") is True
        removed = decision["decision"] == "drop"
        totals["source_count" if source else "augmented_count"] += 1
        if source:
            totals["removed_source_count"] += int(removed)
        else:
            totals["removed_augmented_count" if removed else "kept_augmented_count"] += 1
        dimensions = {"label": normalize_label(row["label"]), "mr_type": row.get("mr_type"),
                      "mr_id": row.get("mr_id"), "dataset": dataset,
                      "wrong_consensus_label": decision["wrong_consensus_label"],
                      "decision_reason": decision["reason"], "eligible_voter_count": decision["eligible_voter_count"]}
        for key, value in dimensions.items():
            bucket = groups[key].setdefault(group_value(value), {"input_count": 0, "source_count": 0,
                       "augmented_count": 0, "kept_augmented_count": 0, "removed_augmented_count": 0})
            bucket["input_count"] += 1
            bucket["source_count" if source else "augmented_count"] += 1
            if not source:
                bucket["removed_augmented_count" if removed else "kept_augmented_count"] += 1
    totals["removal_rate"] = round_float(totals["removed_augmented_count"] / totals["augmented_count"]) if totals["augmented_count"] else 0.0
    for grouped in groups.values():
        for bucket in grouped.values():
            bucket["removal_rate"] = round_float(bucket["removed_augmented_count"] / bucket["augmented_count"]) if bucket["augmented_count"] else 0.0
    return {**totals, "by": groups}


def filter_dataset(config, dataset, audit_dirs, output_dir):
    input_path = project_path(config["dataset_paths"][dataset])
    output_dir = Path(output_dir).resolve()
    protect_output(output_dir, input_path)
    if any((output_dir / name).exists() for name in OUTPUT_FILES):
        raise FileExistsError(f"Ensemble outputs already exist; use a new directory: {output_dir}")
    input_hash = sha256_file(input_path)
    rows = load_rows(input_path)
    if any("nli_quality" in row or "ensemble_quality" in row for row in rows):
        raise ValueError("Input contains reserved audit metadata fields")
    sources = source_positions(rows)
    audits, health, manifests = {}, {}, {}
    for model in config["models"]:
        model_id = model["id"]
        audits[model_id], health[model_id], manifests[model_id] = load_audit(
            audit_dirs[model_id], rows, input_hash, model, config)
    decisions = [decide_row(row, i, rows, sources, audits, config) for i, row in enumerate(rows)]
    summary = count_summary(rows, decisions, dataset)
    summary.update({"schema_version": 1, "dataset": dataset, "policy": config["policy_name"],
                    "config_sha256": config_fingerprint(config), "policy_config": {k: config[k] for k in (
                        "source_confidence_threshold", "followup_confidence_threshold", "min_eligible_voters",
                        "min_wrong_consensus", "block_on_high_conf_gold_support")},
                    "input": {"path": str(input_path), "sha256": input_hash, "record_count": len(rows)},
                    "source_auditor_health": health, "threshold_sensitivity": {}})
    for threshold in (0.90, 0.95, 0.99):
        alternate = [decide_row(row, i, rows, sources, audits, config, threshold) for i, row in enumerate(rows)]
        counts = count_summary(rows, alternate, dataset)
        summary["threshold_sensitivity"][f"{threshold:.2f}"] = {key: counts[key] for key in (
            "removed_augmented_count", "kept_augmented_count", "removal_rate", "by")}
    enriched = [{**row, "ensemble_quality": decision} for row, decision in zip(rows, decisions)]
    kept = [row for row in enriched if row["ensemble_quality"]["decision"] == "keep"]
    removed = [row for row in enriched if row["ensemble_quality"]["decision"] == "drop"]
    assert len(kept) + len(removed) == len(rows) and summary["removed_source_count"] == 0
    for row in removed:
        d = row["ensemble_quality"]
        assert row.get("is_source") is False and d["eligible_voter_count"] >= config["min_eligible_voters"]
        assert d["wrong_consensus_count"] >= config["min_wrong_consensus"]
        assert not config["block_on_high_conf_gold_support"] or d["high_conf_gold_support_count"] == 0
    if sha256_file(input_path) != input_hash:
        raise ValueError("Input changed during filtering")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Validate everything before staging or publishing any filtered dataset.
    with tempfile.TemporaryDirectory(prefix=".ensemble-", dir=output_dir) as temporary:
        stage = Path(temporary)
        write_jsonl(stage / "ensemble_predictions.jsonl", enriched)
        write_jsonl(stage / "filtered.jsonl", kept)
        write_jsonl(stage / "removed.jsonl", removed)
        manifest = {"schema_version": 1, "dataset": dataset, "input": summary["input"],
                    "config_sha256": summary["config_sha256"], "models": manifests,
                    "output_sha256": {name: sha256_file(stage / name) for name in OUTPUT_FILES[:3]}}
        write_json(stage / "model_manifest.json", manifest)
        write_json(stage / "ensemble_summary.json", summary)
        # Summary is the completion marker and is always published last.
        for name in OUTPUT_FILES:
            if name != "ensemble_summary.json":
                os.replace(stage / name, output_dir / name)
        os.replace(stage / "ensemble_summary.json", output_dir / "ensemble_summary.json")
    return summary


def aggregate_summaries(summaries):
    datasets = {s["dataset"]: s for s in summaries}
    result = {"schema_version": 1, "datasets": datasets,
              "table": [{"dataset": s["dataset"], "source": s["source_count"], "augmented": s["augmented_count"],
                         "kept": s["kept_augmented_count"], "removed": s["removed_augmented_count"],
                         "removal_rate": s["removal_rate"]} for s in summaries],
              "by_mr_type": {}, "by_mr_id": {}, "by_gold_label": {}, "threshold_sensitivity": {}}
    for dimension, output in (("mr_type", "by_mr_type"), ("mr_id", "by_mr_id"), ("label", "by_gold_label")):
        grouped = result[output]
        for summary in summaries:
            for value, bucket in summary["by"][dimension].items():
                dest = grouped.setdefault(value, {"augmented_count": 0, "kept_augmented_count": 0, "removed_augmented_count": 0})
                for key in dest:
                    dest[key] += bucket[key]
        for bucket in grouped.values():
            bucket["removal_rate"] = round_float(bucket["removed_augmented_count"] / bucket["augmented_count"]) if bucket["augmented_count"] else 0.0
    augmented = sum(s["augmented_count"] for s in summaries)
    for threshold in ("0.90", "0.95", "0.99"):
        count = sum(s["threshold_sensitivity"][threshold]["removed_augmented_count"] for s in summaries)
        result["threshold_sensitivity"][threshold] = {"removed_augmented_count": count,
            "removal_rate": round_float(count / augmented) if augmented else 0.0,
            "by_dataset": {s["dataset"]: s["threshold_sensitivity"][threshold]["removed_augmented_count"] for s in summaries}}
    result["totals"] = {key: sum(s[key] for s in summaries) for key in ("input_count", "source_count", "augmented_count",
        "kept_augmented_count", "removed_augmented_count", "removed_source_count")}
    result["totals"]["removal_rate"] = round_float(result["totals"]["removed_augmented_count"] / augmented) if augmented else 0.0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", required=True, choices=("snli", "mnlim", "mnlimm", "sick"))
    parser.add_argument("--audit-root", type=Path, help="Directory containing model_a through model_d")
    parser.add_argument("--output-dir", type=Path, help="New output directory; defaults to audit-root")
    args = parser.parse_args()
    config = load_config(args.config)
    root = args.audit_root or project_path(config["output_root"]) / args.dataset
    summary = filter_dataset(config, args.dataset, {m["id"]: root / m["id"] for m in config["models"]}, args.output_dir or root)
    print(json.dumps({key: summary[key] for key in ("dataset", "input_count", "removed_augmented_count", "removal_rate")}, indent=2))


if __name__ == "__main__":
    main()
