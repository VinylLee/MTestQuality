#!/usr/bin/env python3
"""Pure-standard-library post-hoc V1.1 refilter; no model/inference imports.

Reuse persisted evaluator predictions and V1's typed pair keys, label mapping,
source gating and same-label voting semantics. Small data helpers are kept here
because importing the original evaluator/filter would import torch/transformers.
"""

import argparse
import collections
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs/ensemble_filter_v1_1.json"
DATASETS = ("snli", "mnlim", "mnlimm", "sick")
LABELS = ("entailment", "neutral", "contradiction")
WANLI = "alisawuffles/roberta-large-wanli"
QUALITY = "ensemble_quality_v1_1"
DIFF_COUNTS = ("v1_removed", "v1_1_removed", "drop_to_keep", "keep_to_drop",
               "unchanged_drop", "unchanged_keep")
PROTECTION_REASONS = ("gold_support_block", "conditional_clause_auto_drop_disabled",
                      "mnli_wrong_consensus_missing_wanli_anchor", "sick_requires_unanimous_consensus")


def project_path(value):
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{position + 1}")
            canonical(row)  # Reject non-finite numbers, including in raw fields.
            yield row


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_label(value):
    # Same canonical labels and aliases as evaluate_nli_quality.normalize_label.
    if isinstance(value, bool):
        raise ValueError("Boolean NLI label")
    if isinstance(value, int):
        if value not in range(3):
            raise ValueError("Unknown numeric NLI label")
        return LABELS[value]
    aliases = {"0": "entailment", "1": "neutral", "2": "contradiction",
               "entail": "entailment", "entails": "entailment",
               "contradict": "contradiction", "contradicts": "contradiction"}
    text = str(value).strip().lower()
    label = aliases.get(text, text)
    if label not in LABELS:
        raise ValueError(f"Unknown NLI label: {value!r}")
    return label


def pair_key(row):
    value = row.get("pair_id")
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value == "":
        return None
    return (type(value).__name__, value)


def source_positions(rows):
    grouped = collections.defaultdict(list)
    for i, row in enumerate(rows):
        if row.get("is_source") is True and pair_key(row) is not None:
            grouped[pair_key(row)].append(i)
    return {key: values[0] for key, values in grouped.items() if len(values) == 1}


def conditional_clause(row):
    return row.get("mr_type") == "conditional_clause" or row.get("mr_id") == "conditional_clause"


def probability(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("Invalid confidence/probability")
    return float(value)


def load_config(path=DEFAULT_CONFIG):
    config = read_json(path)
    validate_config(config)
    return config


def validate_config(config):
    # Formal V1.1 parameters are fixed; sensitivity overrides are diagnostic only.
    if config.get("policy_name") != "conservative_consensus_v1_1":
        raise ValueError("Unsupported policy")
    if config.get("datasets") != list(DATASETS):
        raise ValueError("V1.1 requires all four datasets in canonical order")
    for key, expected in (("source_confidence_threshold", .80), ("wrong_vote_threshold", .95),
                          ("gold_support_block_threshold", .90)):
        if probability(config[key]) != expected:
            raise ValueError(f"Formal V1.1 fixes {key}={expected}")
    if config.get("conditional_clause_auto_drop") is not False:
        raise ValueError("Conditional-clause auto-drop must be disabled")
    for dataset in DATASETS:
        policy = config["dataset_policies"][dataset]
        count = 4 if dataset == "sick" else 3
        if (type(policy.get("min_eligible_voters")) is not int or
                type(policy.get("min_wrong_consensus")) is not int or
                policy["min_eligible_voters"] != count or policy["min_wrong_consensus"] != count or
                policy.get("require_unanimous") is not (dataset == "sick") or
                policy.get("require_wanli_anchor") is not (dataset in ("mnlim", "mnlimm"))):
            raise ValueError(f"Invalid V1.1 dataset policy: {dataset}")
        if policy["require_wanli_anchor"] and policy.get("wanli_anchor_model") != WANLI:
            raise ValueError("MNLI anchor must be the WANLI-only checkpoint")
        project_path(config["dataset_paths"][dataset])


def snapshot_tree(root):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Missing/unsafe protected directory: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in protected tree: {path}")
        if path.is_file():
            stat = path.stat()
            result[str(path.relative_to(root))] = {
                "sha256": sha256_file(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def verify_integrity(root, before):
    after = snapshot_tree(root)
    if before != after:
        raise ValueError(f"Protected inputs changed (file count/SHA-256/size/mtime): {root}")
    return {"verified": True, "file_count_before": len(before), "file_count_after": len(after),
            "files": {name: {f"{key}_{phase}": value
                             for phase, info in (("before", before[name]), ("after", after[name]))
                             for key, value in info.items()} for name in before}}


def protect_output(output, protected):
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output directory: {target}")
    target = target.resolve()
    for root in protected:
        root = Path(root).resolve()
        if target == root or target in root.parents or root in target.parents:
            raise ValueError(f"Output must not overlap protected inputs: {target} / {root}")


def validate_manifest(manifest, dataset, directory, snapshot, input_path, rows):
    if manifest.get("dataset") != dataset or manifest.get("input", {}).get("record_count") != len(rows):
        raise ValueError("V1 manifest dataset/record count mismatch")
    input_hash = sha256_file(input_path)
    if manifest["input"].get("sha256") != input_hash:
        raise ValueError("Original dataset SHA-256 differs from V1")
    models = manifest.get("models", {})
    if not isinstance(models, dict) or len(models) != 4:
        raise ValueError("Exactly four persisted model identities required")
    names = []
    for model_id, model in models.items():
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", model_id):
            raise ValueError("Unsafe model id")
        name = model.get("model_name")
        if not isinstance(name, str) or not name:
            raise ValueError("Missing persisted model identity")
        names.append(name)
        if (not re.fullmatch(r"[0-9a-f]{40}", model.get("resolved_revision", "")) or
                set(model.get("label_order", [])) != set(LABELS) or len(model["label_order"]) != 3 or
                model.get("input_sha256") != input_hash or model.get("record_count") != len(rows)):
            raise ValueError("Incomplete/mismatched persisted model provenance")
        for filename, key in (("predictions.jsonl", "predictions_sha256"), ("summary.json", "summary_sha256")):
            saved = snapshot.get(f"{dataset}/{model_id}/{filename}")
            if saved is None or saved["sha256"] != model.get(key):
                raise ValueError(f"Persisted model artifact SHA-256 mismatch: {model_id}/{filename}")
    if len(set(names)) != 4 or (dataset in ("mnlim", "mnlimm") and names.count(WANLI) != 1):
        raise ValueError("Auditors must be distinct; MNLI requires an identifiable WANLI-only anchor")
    for filename in ("ensemble_predictions.jsonl", "filtered.jsonl", "removed.jsonl"):
        saved = snapshot.get(f"{dataset}/{filename}")
        if saved is None or saved["sha256"] != manifest.get("output_sha256", {}).get(filename):
            raise ValueError(f"V1 output SHA-256 mismatch: {filename}")
    return models


def prediction(check, prefix):
    if not isinstance(check, dict):
        return None
    try:
        return {"prediction": normalize_label(check[f"{prefix}_prediction"]),
                "confidence": probability(check[f"{prefix}_confidence"])}
    except (KeyError, ValueError, TypeError):
        return None


def missing_prediction(check, prefix):
    return not isinstance(check, dict) or any(check.get(f"{prefix}_{key}") is None
                                              for key in ("prediction", "confidence"))


def prepare_evidence(rows, models, directory):
    """Prefer ensemble evidence; read individual predictions only for missing fields."""
    sources, fallback, joined_models = source_positions(rows), {}, set()

    def saved_check(position, model_id):
        per_model = rows[position]["ensemble_quality"].get("per_model", {})
        if not isinstance(per_model, dict):
            return None
        if set(per_model) - set(models):
            raise ValueError("Ensemble evidence contains an unidentified auditor")
        return per_model.get(model_id)

    def offline_prediction(model_id, position):
        if model_id not in fallback:
            checks = []
            for i, row in enumerate(iter_jsonl(directory / model_id / "predictions.jsonl")):
                if i >= len(rows) or canonical({k: v for k, v in row.items() if k != "nli_quality"}) != canonical(
                        {k: v for k, v in rows[i].items() if k != "ensemble_quality"}):
                    raise ValueError("Offline join raw row/position mismatch")
                check = row.get("nli_quality", {})
                if type(check.get("row_position")) is not int or check["row_position"] != i:
                    raise ValueError("Offline join row_position mismatch")
                result = None
                required = ("expected_label", "predicted_label", "confidence", "probabilities")
                if all(key in check for key in required):
                    gold, label = normalize_label(check["expected_label"]), normalize_label(check["predicted_label"])
                    probs = check["probabilities"]
                    if gold != normalize_label(rows[i]["label"]) or not isinstance(probs, dict) or set(probs) != set(LABELS):
                        raise ValueError("Offline join gold label/probability mapping mismatch")
                    probs = {key: probability(value) for key, value in probs.items()}
                    if (abs(sum(probs.values()) - 1) > 1e-5 or
                            abs(probability(check["confidence"]) - probs[label]) > 1e-5 or
                            probs[label] < max(probs.values())):
                        raise ValueError("Offline join invalid top-1 probability")
                    result = {"prediction": label, "confidence": probs[label]}
                checks.append(result)
            if len(checks) != len(rows):
                raise ValueError("Offline join record count mismatch")
            fallback[model_id] = checks
            joined_models.add(model_id)
        return fallback[model_id][position]

    def get_prediction(position, model_id, prefix):
        check = saved_check(position, model_id)
        if missing_prediction(check, prefix):
            return offline_prediction(model_id, position)
        return prediction(check, prefix)

    evidence = []
    for i, row in enumerate(rows):
        per_model, complete = {}, True
        source_position = i if row.get("is_source") is True else sources.get(pair_key(row))
        valid_pair = row.get("is_source") is True or (row.get("is_source") is False and source_position is not None)
        if not valid_pair:
            evidence.append({"valid_pair": False, "complete": False, "per_model": {}})
            continue
        for model_id, model in models.items():
            source = get_prediction(source_position, model_id, "source")
            augmented = None if row.get("is_source") is True else get_prediction(i, model_id, "augmented")
            repeated = saved_check(i, model_id)
            if row.get("is_source") is False and not missing_prediction(repeated, "source"):
                if prediction(repeated, "source") != source:
                    raise ValueError(f"Repeated source evidence differs at row {i}/{model_id}")
            complete &= source is not None and (row.get("is_source") is True or augmented is not None)
            per_model[model_id] = {"model_name": model["model_name"], "source_prediction": source["prediction"] if source else None,
                "source_confidence": source["confidence"] if source else None,
                "source_gold_label": normalize_label(rows[source_position]["label"]),
                "augmented_prediction": augmented["prediction"] if augmented else None,
                "augmented_confidence": augmented["confidence"] if augmented else None}
        evidence.append({"valid_pair": valid_pair, "complete": bool(complete), "per_model": per_model})
    return evidence, sorted(joined_models)


def decide_row(row, dataset, evidence, config, wrong_threshold=None,
               protect_conditional=True, force_unanimous=False):
    threshold = config["wrong_vote_threshold"] if wrong_threshold is None else wrong_threshold
    policy = config["dataset_policies"][dataset]
    old = row["ensemble_quality"]["decision"]
    result = {"policy": config["policy_name"], "decision": "keep", "reason": "insufficient_wrong_consensus",
        "dataset": dataset, "row_position": row["ensemble_quality"]["row_position"],
        "source_confidence_threshold": config["source_confidence_threshold"],
        "wrong_vote_threshold": threshold, "gold_support_block_threshold": config["gold_support_block_threshold"],
        "eligible_voter_count": 0, "wrong_consensus_label": None, "wrong_consensus_count": 0,
        "gold_support_block_count": 0, "wrong_supporting_models": [], "gold_supporting_models": [],
        "wanli_anchor_required": policy["require_wanli_anchor"],
        "wanli_anchor_present_in_wrong_consensus": False,
        "conditional_clause_protected": conditional_clause(row) and row.get("is_source") is not True,
        "evidence_complete": evidence["complete"], "per_model": {}}
    votes = collections.defaultdict(list)
    gold = normalize_label(row["label"])
    if row.get("is_source") is not True:
        for model_id, check in evidence["per_model"].items():
            eligible = (check["source_prediction"] == check["source_gold_label"] and
                        check["source_confidence"] is not None and
                        check["source_confidence"] >= config["source_confidence_threshold"])
            wrong = (eligible and check["augmented_prediction"] is not None and check["augmented_prediction"] != gold and
                     check["augmented_confidence"] is not None and check["augmented_confidence"] >= threshold)
            blocker = (eligible and check["augmented_prediction"] == gold and check["augmented_confidence"] is not None and
                       check["augmented_confidence"] >= config["gold_support_block_threshold"])
            result["eligible_voter_count"] += int(eligible)
            if wrong:
                votes[check["augmented_prediction"]].append(model_id)
            if blocker:
                result["gold_supporting_models"].append(model_id)
            result["per_model"][model_id] = {**check, "source_eligible": bool(eligible),
                                           "high_conf_wrong_vote": bool(wrong), "gold_support_block": bool(blocker)}
    else:
        result["per_model"] = evidence["per_model"]
    if votes:
        label = sorted(votes, key=lambda key: (-len(votes[key]), key))[0]
        result.update(wrong_consensus_label=label, wrong_consensus_count=len(votes[label]), wrong_supporting_models=votes[label])
        result["wanli_anchor_present_in_wrong_consensus"] = any(
            evidence["per_model"][m]["model_name"] == WANLI for m in votes[label])
    result["gold_support_block_count"] = len(result["gold_supporting_models"])
    unanimous = policy["require_unanimous"] or force_unanimous
    if row.get("is_source") is True:
        result["reason"] = "source_never_filtered"
    elif not evidence["valid_pair"] or not evidence["complete"]:
        result["reason"] = "invalid_or_incomplete_evidence"
    elif protect_conditional and result["conditional_clause_protected"]:
        result["reason"] = "conditional_clause_auto_drop_disabled"
    elif result["gold_support_block_count"]:
        result["reason"] = "gold_support_block"
    elif unanimous and (result["eligible_voter_count"] != 4 or result["wrong_consensus_count"] != 4):
        result["reason"] = "sick_requires_unanimous_consensus" if dataset == "sick" else "diagnostic_requires_unanimous_consensus"
    elif (result["eligible_voter_count"] >= policy["min_eligible_voters"] and
          result["wrong_consensus_count"] >= policy["min_wrong_consensus"]):
        if policy["require_wanli_anchor"] and not result["wanli_anchor_present_in_wrong_consensus"]:
            result["reason"] = "mnli_wrong_consensus_missing_wanli_anchor"
        else:
            result.update(decision="drop", reason="high_confidence_wrong_consensus")
    result.update(v1_decision=old, v1_1_decision=result["decision"], decision_changed=old != result["decision"])
    return result


def group_counts(rows, dimensions):
    return {dimension: dict(sorted(collections.Counter(
        str(getter(row)) if getter(row) is not None else "<missing>" for row in rows).items()))
            for dimension, getter in dimensions.items()}


def dimensions():
    return {"dataset": lambda r: r[QUALITY]["dataset"], "mr_type": lambda r: r.get("mr_type"),
        "mr_id": lambda r: r.get("mr_id"), "gold_label": lambda r: normalize_label(r["label"]),
        "reason": lambda r: r[QUALITY]["reason"], "wrong_consensus_label": lambda r: r[QUALITY]["wrong_consensus_label"],
        "consensus_size": lambda r: f'{r[QUALITY]["wrong_consensus_count"]}/4'}


def confidence_ranges(rows):
    values = {"source_confidence": [], "augmented_confidence": []}
    for row in rows:
        quality = row[QUALITY]
        for model_id in quality["wrong_supporting_models"]:
            for key in values:
                values[key].append(quality["per_model"][model_id][key])
    return {key: {"vote_count": len(probs), "min": min(probs) if probs else None,
                  "max": max(probs) if probs else None} for key, probs in values.items()}


def policy_diff(rows, dataset):
    diff = {"dataset": dataset, **dict.fromkeys(DIFF_COUNTS, 0)}
    rescued, changed = [], []
    for row in rows:
        quality = row[QUALITY]
        old, new = quality["v1_decision"], quality["decision"]
        diff["v1_removed"] += int(old == "drop")
        diff["v1_1_removed"] += int(new == "drop")
        diff["unchanged_" + old if old == new else old + "_to_" + new] += 1
        if old != new:
            changed.append(row)
        if old == "drop" and new == "keep":
            rescued.append(row)
    diff["changes_by_reason"] = group_counts(changed, {"reason": dimensions()["reason"]})["reason"]
    diff["rescued_by"] = group_counts(rescued, dimensions())
    for reason in PROTECTION_REASONS:
        diff["changes_by_reason"].setdefault(reason, 0)
        diff["rescued_by"]["reason"].setdefault(reason, 0)
    if diff["drop_to_keep"] + diff["unchanged_drop"] != diff["v1_removed"] or sum(
            diff[key] for key in ("drop_to_keep", "keep_to_drop", "unchanged_drop", "unchanged_keep")) != len(rows):
        raise ValueError("Decision diff conservation failure")
    return diff


def dataset_summary(rows, evidence, config, dataset, old_summary, joined):
    augmented = [r for r in rows if r.get("is_source") is not True]
    removed = [r for r in rows if r[QUALITY]["decision"] == "drop"]
    counts = collections.Counter(r[QUALITY]["eligible_voter_count"] for r in augmented)
    summary = {"schema_version": "1.1", "policy": config["policy_name"], "dataset": dataset,
        "input": old_summary["input"], "input_count": len(rows), "source_count": len(rows) - len(augmented),
        "augmented_count": len(augmented), "kept_augmented_count": len(augmented) - len(removed),
        "removed_augmented_count": len(removed), "removed_source_count": sum(r.get("is_source") is True for r in removed),
        "removal_rate": len(removed) / len(augmented) if augmented else 0,
        "source_auditor_health": old_summary["source_auditor_health"],
        "policy_config": config, "config_sha256": hashlib.sha256(canonical(config).encode()).hexdigest(),
        "evidence_provenance": {"primary": "ensemble_predictions.jsonl", "offline_join_model_ids": joined,
                                "inference_performed": False},
        "by": {"all": group_counts(rows, dimensions()), "augmented": group_counts(augmented, dimensions()),
               "removed": group_counts(removed, dimensions())},
        "remaining_removed_confidence_ranges": confidence_ranges(removed),
        "eligibility_diagnostics": {str(i): {"count": counts[i], "proportion": counts[i] / len(augmented) if augmented else 0}
                                    for i in range(5)},
        "eligible_count_lt_3": sum(counts[i] for i in range(3)), "threshold_sensitivity": {}}
    for threshold in (.90, .95, .99):
        alternate = [decide_row(row, dataset, item, config, wrong_threshold=threshold)
                     for row, item in zip(rows, evidence)]
        n = sum(q["decision"] == "drop" for q in alternate)
        summary["threshold_sensitivity"][f"{threshold:.2f}"] = {
            "wrong_vote_threshold": threshold, "gold_support_block_threshold": .90,
            "removed_augmented_count": n, "kept_augmented_count": len(augmented) - n,
            "removal_rate": n / len(augmented) if augmented else 0}
    conditional = [(r, e) for r, e in zip(rows, evidence) if r.get("is_source") is not True and conditional_clause(r)]
    summary["conditional_clause_diagnostic"] = {"protected_count": len(conditional),
        "would_drop_under_standard_policy": sum(decide_row(r, dataset, e, config, protect_conditional=False)["decision"] == "drop"
                                               for r, e in conditional),
        "would_drop_under_4_of_4_099": sum(decide_row(r, dataset, e, config, wrong_threshold=.99,
            protect_conditional=False, force_unanimous=True)["decision"] == "drop" for r, e in conditional)}
    return summary


def process_dataset(config, dataset, source_root, stage, snapshot):
    directory = source_root / dataset
    rows = list(iter_jsonl(directory / "ensemble_predictions.jsonl"))
    raw = list(iter_jsonl(project_path(config["dataset_paths"][dataset])))
    manifest, old_summary = read_json(directory / "model_manifest.json"), read_json(directory / "ensemble_summary.json")
    models = validate_manifest(manifest, dataset, directory, snapshot, project_path(config["dataset_paths"][dataset]), rows)
    if len(raw) != len(rows) or old_summary.get("input") != manifest["input"] or old_summary.get("dataset") != dataset:
        raise ValueError("V1 input/summary metadata mismatch")
    for i, (original, row) in enumerate(zip(raw, rows)):
        if QUALITY in row or "ensemble_quality" in original or QUALITY in original:
            raise ValueError("Reserved quality fields already exist")
        if canonical(original) != canonical({k: v for k, v in row.items() if k != "ensemble_quality"}):
            raise ValueError(f"V1/raw original fields mismatch at {dataset}/{i}")
        normalize_label(original["label"])
        quality = row.get("ensemble_quality", {})
        if (quality.get("policy") != "conservative_consensus_v1" or quality.get("decision") not in ("keep", "drop") or
                type(quality.get("row_position")) is not int or quality["row_position"] != i):
            raise ValueError("Invalid persisted V1 decision/row_position")
    # Validate the actual V1 partitions; never infer/recreate its old decisions.
    for name, decision in (("filtered.jsonl", "keep"), ("removed.jsonl", "drop")):
        expected = (r for r in rows if r["ensemble_quality"]["decision"] == decision)
        for actual in iter_jsonl(directory / name):
            if canonical(actual) != canonical(next(expected, None)):
                raise ValueError(f"V1 partition/order mismatch: {name}")
        if next(expected, None) is not None:
            raise ValueError(f"Incomplete V1 partition: {name}")
    if old_summary["removed_augmented_count"] != sum(r["ensemble_quality"]["decision"] == "drop" for r in rows):
        raise ValueError("V1 removal count mismatch")
    evidence, joined = prepare_evidence(rows, models, directory)
    enriched = [{**row, QUALITY: decide_row(row, dataset, item, config)} for row, item in zip(rows, evidence)]
    diff = policy_diff(enriched, dataset)
    if diff["keep_to_drop"]:
        raise ValueError("V1.1 must not newly remove a V1 KEEP record")
    kept = [r for r in enriched if r[QUALITY]["decision"] == "keep"]
    removed = [r for r in enriched if r[QUALITY]["decision"] == "drop"]
    if len(kept) + len(removed) != len(rows) or any(r.get("is_source") is True or conditional_clause(r) for r in removed):
        raise ValueError("Output/source/conditional conservation failure")
    summary = dataset_summary(enriched, evidence, config, dataset, old_summary, joined)
    target = stage / dataset
    target.mkdir()
    for name, selected in (("ensemble_predictions.jsonl", enriched), ("filtered.jsonl", kept), ("removed.jsonl", removed),
                           ("rescued_from_v1.jsonl", [r for r in enriched if r[QUALITY]["v1_decision"] == "drop" and r[QUALITY]["decision"] == "keep"])):
        write_jsonl(target / name, selected)
    write_json(target / "policy_diff_from_v1.json", diff)
    write_json(target / "model_manifest.json", {"schema_version": "1.1", "dataset": dataset, "input": manifest["input"],
        "models": models, "source_v1_manifest_sha256": snapshot[f"{dataset}/model_manifest.json"]["sha256"],
        "inference_performed": False, "output_sha256": {p.name: sha256_file(p) for p in sorted(target.iterdir())}})
    return summary, diff


def merge_counts(groups):
    result = collections.Counter()
    for group in groups:
        result.update(group)
    return dict(sorted(result.items()))


def aggregate(summaries, diffs):
    total = {key: sum(s[key] for s in summaries.values()) for key in (
        "input_count", "source_count", "augmented_count", "kept_augmented_count", "removed_augmented_count", "removed_source_count")}
    total["removal_rate"] = total["removed_augmented_count"] / total["augmented_count"] if total["augmented_count"] else 0
    combined_diff = {"dataset": "TOTAL", **{key: sum(d[key] for d in diffs.values()) for key in DIFF_COUNTS},
        "changes_by_reason": merge_counts(d["changes_by_reason"] for d in diffs.values()),
        "rescued_by": {key: merge_counts(d["rescued_by"][key] for d in diffs.values()) for key in dimensions()}}
    result = {"schema_version": "1.1", "policy": "conservative_consensus_v1_1", "datasets": summaries,
        "totals": total, "remaining_removed_by": {key: merge_counts(s["by"]["removed"][key] for s in summaries.values()) for key in dimensions()},
        "threshold_sensitivity": {key: {"gold_support_block_threshold": .90,
            "removed_augmented_count": sum(s["threshold_sensitivity"][key]["removed_augmented_count"] for s in summaries.values()),
            "by_dataset": {d: s["threshold_sensitivity"][key]["removed_augmented_count"] for d, s in summaries.items()}}
            for key in ("0.90", "0.95", "0.99")}}
    return result, {**{d.upper(): value for d, value in diffs.items()}, "TOTAL": combined_diff}


def report_markdown(summary, diffs):
    lines = ["# Ensemble filter V1.1 — offline refilter report", "",
        "No model was loaded or rerun. All evidence comes from persisted V1 predictions.", "",
        "## Overall", "", "| dataset | augmented | V1 removed | V1.1 removed | rescued by V1.1 | V1.1 removal rate |",
        "|---|---:|---:|---:|---:|---:|"]
    for dataset, item in summary["datasets"].items():
        diff = diffs[dataset.upper()]
        lines.append(f'| {dataset.upper()} | {item["augmented_count"]} | {diff["v1_removed"]} | {diff["v1_1_removed"]} | {diff["drop_to_keep"]} | {item["removal_rate"]:.4%} |')
    total, diff = summary["totals"], diffs["TOTAL"]
    lines.extend([f'| TOTAL | {total["augmented_count"]} | {diff["v1_removed"]} | {diff["v1_1_removed"]} | {diff["drop_to_keep"]} | {total["removal_rate"]:.4%} |', "",
        f'Source records preserved: {total["source_count"]}; removed source: {total["removed_source_count"]}; KEEP→DROP: {diff["keep_to_drop"]}.', "",
        "## Policy changes", "",
        "Source gating remains correct source prediction with confidence ≥0.80; same-wrong-label votes remain ≥0.95. Gold-support blocker changes 0.95 → 0.90 (eligible auditors only). SNLI retains ≥3/4. MNLIM/MNLIMM additionally require the actual `alisawuffles/roberta-large-wanli` checkpoint among wrong supporters, resolved by model manifest identity, not a fixed model id. SICK changes 3/4 → unanimous 4/4, with all four source-eligible. Exact `mr_type` OR `mr_id == conditional_clause` disables auto-drop.", "",
        "Decision priority: source → invalid/incomplete evidence → conditional protection → gold blocker → dataset-specific rule → insufficient consensus.", "",
        "## Rescued samples (V1 DROP → V1.1 KEEP)", "",
        "Primary protection reasons are mutually exclusive due to decision priority; counts are not causal ablations.", ""])
    for key in ("reason", "dataset", "mr_type", "mr_id", "gold_label"):
        lines.extend([f'### By {key}', "", "| value | rescued |", "|---|---:|"])
        for value, count in diffs["TOTAL"]["rescued_by"][key].items():
            lines.append(f"| {value} | {count} |")
        lines.append("")
    lines.extend(["## Remaining removed samples", ""])
    for key in ("dataset", "mr_type", "mr_id", "gold_label", "wrong_consensus_label", "consensus_size"):
        lines.extend([f"### By {key}", "", "| value | removed |", "|---|---:|"])
        for value, count in summary["remaining_removed_by"][key].items():
            lines.append(f"| {value} | {count} |")
        lines.append("")
    lines.extend(["### Supporting-vote confidence ranges", "", "Ranges cover only auditors in the winning wrong-label consensus, not cross-model averaged probabilities.", "",
        "| dataset | source min–max | augmented min–max | votes |", "|---|---|---|---:|"])
    for dataset, item in summary["datasets"].items():
        ranges = item["remaining_removed_confidence_ranges"]
        source, aug = ranges["source_confidence"], ranges["augmented_confidence"]
        lines.append(f'| {dataset} | {source["min"]}–{source["max"]} | {aug["min"]}–{aug["max"]} | {aug["vote_count"]} |')
    lines.extend(["", "## Eligibility diagnostics", "", "| dataset | eligible 0 | eligible 1 | eligible 2 | eligible 3 | eligible 4 | eligible <3 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for dataset, item in summary["datasets"].items():
        values = [f'{item["eligibility_diagnostics"][str(i)]["count"]} ({item["eligibility_diagnostics"][str(i)]["proportion"]:.2%})' for i in range(5)]
        lines.append(f'| {dataset} | ' + " | ".join(values) + f' | {item["eligible_count_lt_3"]} |')
    lines.extend(["", "KEEP does not mean the ensemble proved the label correct. This filter only deletes records with strong consistent contrary evidence. Confidence is uncalibrated model softmax confidence, not a probability that the label is wrong.", "",
        "## Wrong-vote threshold sensitivity", "", "Source threshold=0.80 and gold blocker=0.90 stay fixed; anchor, unanimity, and conditional protection stay enabled. Actual filtered files always use wrong threshold=0.95.", "",
        "| wrong threshold | SNLI | MNLIM | MNLIMM | SICK | TOTAL |", "|---|---:|---:|---:|---:|---:|"])
    for key, sensitivity in summary["threshold_sensitivity"].items():
        lines.append(f"| {key} | " + " | ".join(str(sensitivity["by_dataset"][d]) for d in DATASETS) + f' | {sensitivity["removed_augmented_count"]} |')
    lines.extend(["", "## Conditional-clause diagnostic (not applied)", "", "| dataset | protected | would drop without protection | would drop with 4/4 at 0.99 |", "|---|---:|---:|---:|"])
    for dataset, item in summary["datasets"].items():
        diagnostic = item["conditional_clause_diagnostic"]
        lines.append(f'| {dataset} | {diagnostic["protected_count"]} | {diagnostic["would_drop_under_standard_policy"]} | {diagnostic["would_drop_under_4_of_4_099"]} |')
    lines.extend(["", "## Integrity and offline execution", "",
        f'All {summary["source_v1_integrity"]["file_count_before"]} V1 files were checked before/after: file count, SHA-256, size, and mtime_ns unchanged. Original data files were checked the same way. Detailed receipts: `source_v1_integrity` and `original_data_integrity` in `summary_all_datasets.json`.', "",
        "The refilter imports only Python standard-library modules; it never imports the evaluator, audit runner, torch, transformers, or any model loader. Individual model JSONL files are joined only if ensemble prediction fields are missing. Per-dataset summaries record any offline join model ids.", "",
        "## Artifacts and reproduction", "",
        "Each dataset contains `ensemble_predictions.jsonl`, `filtered.jsonl`, `removed.jsonl`, `rescued_from_v1.jsonl`, `ensemble_summary.json`, `policy_diff_from_v1.json`, and a provenance `model_manifest.json`. Raw fields and the complete V1 `ensemble_quality` remain unchanged; only `ensemble_quality_v1_1` is added. All subsets preserve original order.", "",
        "Use `outputs/ensemble_filter_v1_1/{snli,mnlim,mnlimm,sick}/filtered.jsonl` preferentially for future experiments; V1 remains available for policy comparisons. Root `diff_v1_vs_v1_1.json` contains per-dataset and TOTAL transitions and rescued groups.", "",
        "```bash", "# Existing output roots are refused; use a fresh directory to reproduce.",
        "python refilter_ensemble_v1_1.py --output-root outputs/ensemble_filter_v1_1_reproduced", "python -m unittest discover -s tests -v", "```", ""])
    return "\n".join(lines)


def run_refilter(config, original_root=None):
    validate_config(config)
    source_root, output_root = project_path(config["source_predictions_root"]), project_path(config["output_root"])
    original_root = project_path(original_root or PROJECT_ROOT / "data")
    input_paths = [project_path(config["dataset_paths"][d]) for d in DATASETS]
    protect_output(output_root, [source_root, original_root, *input_paths])
    before, original_before = snapshot_tree(source_root), snapshot_tree(original_root)
    input_before = {str(path): {"sha256": sha256_file(path), "size": path.stat().st_size,
                              "mtime_ns": path.stat().st_mtime_ns} for path in input_paths}
    output_root.parent.mkdir(parents=True, exist_ok=True)
    # Stage the entire run outside both protected roots. Publish only on success.
    with tempfile.TemporaryDirectory(prefix=".refilter-v1_1-", dir=output_root.parent) as temporary:
        stage, summaries, diffs = Path(temporary), {}, {}
        for dataset in DATASETS:
            summaries[dataset], diffs[dataset] = process_dataset(config, dataset, source_root, stage, before)
            print(f'{dataset}: V1 removed={diffs[dataset]["v1_removed"]}, V1.1 removed={diffs[dataset]["v1_1_removed"]}, rescued={diffs[dataset]["drop_to_keep"]}', flush=True)
        summary, combined_diff = aggregate(summaries, diffs)
        summary["source_v1_integrity"] = verify_integrity(source_root, before)
        summary["original_data_integrity"] = verify_integrity(original_root, original_before)
        for path, info in input_before.items():
            current = Path(path)
            if info != {"sha256": sha256_file(current), "size": current.stat().st_size, "mtime_ns": current.stat().st_mtime_ns}:
                raise ValueError("Original input changed during refilter")
        summary["input_integrity"] = {"verified": True, "files": input_before}
        summary["inference_performed"] = False
        for dataset, item in summaries.items():
            item["source_v1_integrity"] = {**summary["source_v1_integrity"], "files": {
                name: info for name, info in summary["source_v1_integrity"]["files"].items() if name.startswith(dataset + "/")}}
            item["source_v1_integrity"]["file_count_before"] = item["source_v1_integrity"]["file_count_after"] = len(item["source_v1_integrity"]["files"])
            write_json(stage / dataset / "ensemble_summary.json", item)
            manifest_path = stage / dataset / "model_manifest.json"
            manifest = read_json(manifest_path)
            manifest["config_sha256"] = item["config_sha256"]
            manifest["output_sha256"]["ensemble_summary.json"] = sha256_file(stage / dataset / "ensemble_summary.json")
            write_json(manifest_path, manifest)
        write_json(stage / "diff_v1_vs_v1_1.json", combined_diff)
        (stage / "REPORT.md").write_text(report_markdown(summary, combined_diff), encoding="utf-8")
        write_json(stage / "summary_all_datasets.json", summary)
        # Recheck output target immediately before atomic directory publication.
        protect_output(output_root, [source_root, original_root, *input_paths])
        os.rename(stage, output_root)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, help="Read-only existing V1 results")
    parser.add_argument("--output-root", type=Path, help="Fresh output root; never overwritten")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.source_root is not None:
        config["source_predictions_root"] = str(args.source_root.resolve())
    if args.output_root is not None:
        config["output_root"] = str(args.output_root.resolve())
    summary = run_refilter(config)
    print(json.dumps(summary["totals"], indent=2))


if __name__ == "__main__":
    main()
