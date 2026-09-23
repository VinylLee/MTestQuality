#!/usr/bin/env python3
"""Audit an NLI JSONL dataset with a pretrained three-way NLI classifier.

The input rows are preserved verbatim in ``predictions.jsonl`` and receive one
additional ``nli_quality`` object.  The script also writes compact views for
high-confidence agreements, disagreements, per-pair quality, and aggregate
metrics.  It intentionally treats model confidence as a screening signal, not
as a calibrated probability or ground truth.
"""

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import transformers
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_ID_TO_NAME = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL_NAME_TO_ID = {value: key for key, value in LABEL_ID_TO_NAME.items()}
KNOWN_MODEL_LABEL_ORDERS = {
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli": (
        "entailment",
        "neutral",
        "contradiction",
    ),
    "cross-encoder/nli-deberta-v3-large": (
        "contradiction",
        "entailment",
        "neutral",
    ),
}
MODEL_TRAINING_DATA_NOTES = {
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli": (
        "Model card lists MNLI, FEVER-NLI, ANLI, LingNLI, and WANLI; "
        "it does not list SNLI."
    ),
    "cross-encoder/nli-deberta-v3-large": (
        "Model card lists SNLI and MultiNLI. This is intentionally an in-domain "
        "quality screen for this SNLI-derived dataset."
    ),
}
DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-large"
CONFIDENCE_THRESHOLDS = (0.50, 0.70, 0.80, 0.90, 0.95, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input JSONL file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument(
        "--model-label-order",
        help="Comma-separated model output order, e.g. entailment,neutral,contradiction",
    )
    parser.add_argument("--no-fp16", action="store_true", help="Disable CUDA fp16 inference")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_label(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError(f"Boolean is not a valid NLI label: {value!r}")
    if isinstance(value, int):
        if value not in LABEL_ID_TO_NAME:
            raise ValueError(f"Unknown numeric NLI label: {value!r}")
        return LABEL_ID_TO_NAME[value]
    text = str(value).strip().lower()
    aliases = {
        "0": "entailment",
        "1": "neutral",
        "2": "contradiction",
        "entail": "entailment",
        "entails": "entailment",
        "contradict": "contradiction",
        "contradicts": "contradiction",
    }
    text = aliases.get(text, text)
    if text not in LABEL_NAME_TO_ID:
        raise ValueError(f"Unknown NLI label: {value!r}")
    return text


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            for field in ("premise", "hypothesis", "label"):
                if field not in row:
                    raise ValueError(f"Missing {field!r} at {path}:{line_number}")
            if not isinstance(row["premise"], str) or not isinstance(row["hypothesis"], str):
                raise ValueError(f"premise/hypothesis must be strings at {path}:{line_number}")
            normalize_label(row["label"])
            rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def canonical_model_label_order(
    model_name: str, config: Any, explicit_order: str
) -> Tuple[str, str, str]:
    if explicit_order:
        order = tuple(normalize_label(item) for item in explicit_order.split(","))
    else:
        raw_id2label = getattr(config, "id2label", {}) or {}
        try:
            order = tuple(normalize_label(raw_id2label[index]) for index in range(3))
        except (KeyError, ValueError):
            try:
                order = tuple(normalize_label(raw_id2label[str(index)]) for index in range(3))
            except (KeyError, ValueError):
                order = KNOWN_MODEL_LABEL_ORDERS.get(model_name, ())
    if len(order) != 3 or set(order) != set(LABEL_NAME_TO_ID):
        raise ValueError(
            "Could not establish a safe three-way model label mapping. "
            "Pass --model-label-order with all three labels in model-logit order. "
            f"config.id2label={getattr(config, 'id2label', None)!r}"
        )
    return order  # type: ignore[return-value]


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - position) + ordered[upper] * (position - lower))


def round_float(value: float) -> float:
    return round(float(value), 8)


def metric_summary(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(items)
    if not count:
        return {"count": 0}
    agreements = [bool(item["agreement"]) for item in items]
    confidences = [float(item["confidence"]) for item in items]
    gold_probabilities = [float(item["gold_probability"]) for item in items]
    margins = [float(item["margin"]) for item in items]
    entropies = [float(item["normalized_entropy"]) for item in items]
    nlls = [float(item["negative_log_likelihood"]) for item in items]
    by_gold = collections.Counter(str(item["expected_label"]) for item in items)
    by_prediction = collections.Counter(str(item["predicted_label"]) for item in items)
    confusion: Dict[str, Dict[str, int]] = {
        gold: {prediction: 0 for prediction in LABEL_NAME_TO_ID}
        for gold in LABEL_NAME_TO_ID
    }
    for item in items:
        confusion[str(item["expected_label"])][str(item["predicted_label"])] += 1
    threshold_metrics = {}
    for threshold in CONFIDENCE_THRESHOLDS:
        selected = [item for item in items if float(item["confidence"]) >= threshold]
        selected_agree = sum(bool(item["agreement"]) for item in selected)
        threshold_metrics[f"{threshold:.2f}"] = {
            "count": len(selected),
            "coverage": round_float(len(selected) / count),
            "agreement_rate": round_float(selected_agree / len(selected)) if selected else None,
            "agreed_count": selected_agree,
        }
    return {
        "count": count,
        "agreed_count": sum(agreements),
        "agreement_rate": round_float(sum(agreements) / count),
        "mean_confidence": round_float(statistics.fmean(confidences)),
        "median_confidence": round_float(statistics.median(confidences)),
        "confidence_percentiles": {
            "p05": round_float(percentile(confidences, 0.05)),
            "p25": round_float(percentile(confidences, 0.25)),
            "p50": round_float(percentile(confidences, 0.50)),
            "p75": round_float(percentile(confidences, 0.75)),
            "p95": round_float(percentile(confidences, 0.95)),
        },
        "mean_gold_probability": round_float(statistics.fmean(gold_probabilities)),
        "mean_margin": round_float(statistics.fmean(margins)),
        "mean_normalized_entropy": round_float(statistics.fmean(entropies)),
        "mean_negative_log_likelihood": round_float(statistics.fmean(nlls)),
        "gold_label_counts": dict(sorted(by_gold.items())),
        "predicted_label_counts": dict(sorted(by_prediction.items())),
        "confusion_matrix_gold_to_prediction": confusion,
        "confidence_thresholds": threshold_metrics,
    }


def grouped_summaries(
    rows: Sequence[Mapping[str, Any]], checks: Sequence[Mapping[str, Any]], key: str
) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = collections.defaultdict(list)
    for row, check in zip(rows, checks):
        value = row.get(key)
        if isinstance(value, bool):
            group_name = str(value).lower()
        elif value is None:
            group_name = "<missing>"
        else:
            group_name = str(value)
        groups[group_name].append(check)
    return {name: metric_summary(groups[name]) for name in sorted(groups)}


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def build_pair_rows(
    rows: Sequence[Mapping[str, Any]], checks: Sequence[Mapping[str, Any]], threshold: float
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[int]] = collections.defaultdict(list)
    for position, row in enumerate(rows):
        grouped[str(row.get("pair_id", f"<missing:{position}>"))].append(position)
    results = []
    for pair_id, positions in grouped.items():
        source_positions = [pos for pos in positions if rows[pos].get("is_source") is True]
        augmented_positions = [pos for pos in positions if rows[pos].get("is_source") is not True]
        augmented_checks = [checks[pos] for pos in augmented_positions]
        pair_checks = [checks[pos] for pos in positions]
        source_check = checks[source_positions[0]] if len(source_positions) == 1 else None
        results.append(
            {
                "pair_id": rows[positions[0]].get("pair_id"),
                "row_count": len(positions),
                "source_row_count": len(source_positions),
                "augmented_row_count": len(augmented_positions),
                "source_agreement": source_check["agreement"] if source_check else None,
                "source_confidence": source_check["confidence"] if source_check else None,
                "all_rows_agree": all(check["agreement"] for check in pair_checks),
                "all_augmented_rows_agree": (
                    all(check["agreement"] for check in augmented_checks)
                    if augmented_checks
                    else None
                ),
                "augmented_agreement_rate": (
                    round_float(sum(check["agreement"] for check in augmented_checks) / len(augmented_checks))
                    if augmented_checks
                    else None
                ),
                "min_confidence": round_float(min(check["confidence"] for check in pair_checks)),
                "mean_confidence": round_float(
                    statistics.fmean(check["confidence"] for check in pair_checks)
                ),
                "all_rows_agree_at_threshold": all(
                    check["agreement"] and check["confidence"] >= threshold
                    for check in pair_checks
                ),
                "row_positions": positions,
                "record_indices": [rows[pos].get("idx") for pos in positions],
                "mr_ids": [rows[pos].get("mr_id") for pos in augmented_positions],
            }
        )
    return results


def augmentation_vs_source_summary(
    rows: Sequence[Mapping[str, Any]], checks: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    source_by_pair: Dict[str, Mapping[str, Any]] = {}
    duplicate_source_pairs = set()
    for row, check in zip(rows, checks):
        if row.get("is_source") is not True:
            continue
        pair_key = str(row.get("pair_id"))
        if pair_key in source_by_pair:
            duplicate_source_pairs.add(pair_key)
        source_by_pair[pair_key] = check

    outcome_groups: Dict[str, List[Mapping[str, Any]]] = collections.defaultdict(list)
    by_mr: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    missing_source_count = 0
    for row, check in zip(rows, checks):
        if row.get("is_source") is True:
            continue
        source_check = source_by_pair.get(str(row.get("pair_id")))
        if source_check is None:
            missing_source_count += 1
            continue
        source_state = "source_agrees" if source_check["agreement"] else "source_disagrees"
        augmented_state = "augmented_agrees" if check["agreement"] else "augmented_disagrees"
        outcome = f"{source_state}__{augmented_state}"
        outcome_groups[source_state].append(check)
        by_mr[str(row.get("mr_id", "<missing>"))][outcome] += 1

    return {
        "interpretation": (
            "Conditions each augmented record on this same model's decision for its source "
            "record. source_agrees__augmented_disagrees is a useful transformation-induced "
            "mismatch signal, but still is not manually verified ground truth."
        ),
        "missing_source_count": missing_source_count,
        "duplicate_source_pair_count": len(duplicate_source_pairs),
        "when_source_agrees": metric_summary(outcome_groups["source_agrees"]),
        "when_source_disagrees": metric_summary(outcome_groups["source_disagrees"]),
        "by_mr_id_outcome_counts": {
            mr_id: dict(sorted(counts.items())) for mr_id, counts in sorted(by_mr.items())
        },
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be between 0 and 1")
    if args.batch_size < 1 or args.max_length < 2:
        raise ValueError("--batch-size and --max-length must be positive")

    started_at = dt.datetime.now(dt.timezone.utc)
    start_time = time.monotonic()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(input_path)
    input_sha256 = sha256_file(input_path)

    print(f"Loading {args.model}@{args.revision} ...", flush=True)
    # ``use_fast=False`` also supports older tokenizers packages and produces
    # the same model inputs for this SentencePiece-based DeBERTa checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, use_fast=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=args.revision
    )
    if model.config.num_labels != 3:
        raise ValueError(f"Expected a 3-label NLI model, got {model.config.num_labels}")
    model_label_order = canonical_model_label_order(
        args.model, model.config, args.model_label_order
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    use_fp16 = device.type == "cuda" and not args.no_fp16
    model.to(device)
    model.eval()

    checks: List[Dict[str, Any]] = []
    at_max_length_count = 0
    progress = tqdm(range(0, len(rows), args.batch_size), desc="NLI inference", unit="batch")
    with torch.inference_mode():
        for start in progress:
            batch = rows[start : start + args.batch_size]
            encoded = tokenizer(
                [row["premise"] for row in batch],
                [row["hypothesis"] for row in batch],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            token_counts = encoded["attention_mask"].sum(dim=1).tolist()
            at_max_length_count += sum(count >= args.max_length for count in token_counts)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.cuda.amp.autocast(enabled=use_fp16):
                logits = model(**encoded).logits
            probabilities = torch.softmax(logits.float(), dim=-1).cpu()
            for row, probs, token_count in zip(batch, probabilities.tolist(), token_counts):
                probability_by_label = {
                    label: float(probs[model_label_order.index(label)])
                    for label in LABEL_NAME_TO_ID
                }
                ranked = sorted(
                    probability_by_label.items(), key=lambda item: item[1], reverse=True
                )
                predicted_label, confidence = ranked[0]
                expected_label = normalize_label(row["label"])
                gold_probability = probability_by_label[expected_label]
                entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in probs) / math.log(3)
                checks.append(
                    {
                        "expected_label": expected_label,
                        "predicted_label": predicted_label,
                        "predicted_label_id": LABEL_NAME_TO_ID[predicted_label],
                        "agreement": predicted_label == expected_label,
                        "confidence": round_float(confidence),
                        "margin": round_float(confidence - ranked[1][1]),
                        "gold_probability": round_float(gold_probability),
                        "gold_rank": next(
                            rank
                            for rank, (label, _) in enumerate(ranked, 1)
                            if label == expected_label
                        ),
                        "negative_log_likelihood": round_float(-math.log(max(gold_probability, 1e-12))),
                        "normalized_entropy": round_float(entropy),
                        "probabilities": {
                            label: round_float(probability_by_label[label])
                            for label in LABEL_NAME_TO_ID
                        },
                        "input_token_count": int(token_count),
                        "at_max_length": bool(token_count >= args.max_length),
                    }
                )

    enriched_rows = []
    for position, (row, check) in enumerate(zip(rows, checks)):
        enriched = dict(row)
        enriched["nli_quality"] = {"row_position": position, **check}
        enriched_rows.append(enriched)

    predictions_path = output_dir / "predictions.jsonl"
    disagreements_path = output_dir / "disagreements.jsonl"
    high_confidence_path = output_dir / "high_confidence_agreements.jsonl"
    pairs_path = output_dir / "pair_quality.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(predictions_path, enriched_rows)
    disagreement_count = write_jsonl(
        disagreements_path,
        (row for row in enriched_rows if not row["nli_quality"]["agreement"]),
    )
    high_confidence_count = write_jsonl(
        high_confidence_path,
        (
            row
            for row in enriched_rows
            if row["nli_quality"]["agreement"]
            and row["nli_quality"]["confidence"] >= args.confidence_threshold
        ),
    )
    pair_rows = build_pair_rows(rows, checks, args.confidence_threshold)
    write_jsonl(pairs_path, pair_rows)

    finished_at = dt.datetime.now(dt.timezone.utc)
    resolved_revision = getattr(model.config, "_commit_hash", None)
    source_checks = [check for row, check in zip(rows, checks) if row.get("is_source") is True]
    augmented_checks = [check for row, check in zip(rows, checks) if row.get("is_source") is not True]
    summary = {
        "schema_version": 1,
        "created_at_utc": finished_at.isoformat(),
        "elapsed_seconds": round(time.monotonic() - start_time, 3),
        "input": {
            "path": str(input_path),
            "sha256": input_sha256,
            "record_count": len(rows),
        },
        "model": {
            "name": args.model,
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "model_logit_label_order": list(model_label_order),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "training_data_note": MODEL_TRAINING_DATA_NOTES.get(
                args.model, "Consult the model card for training-data provenance."
            ),
        },
        "runtime": {
            "started_at_utc": started_at.isoformat(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "fp16": use_fp16,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "at_max_length_count": at_max_length_count,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "interpretation_warning": (
            "Agreement is agreement with one pretrained model, not verified ground truth. "
            "Softmax confidence can be miscalibrated, especially under distribution shift."
        ),
        "screening": {
            "confidence_threshold": args.confidence_threshold,
            "high_confidence_agreement_count": high_confidence_count,
            "disagreement_count": disagreement_count,
        },
        "overall": metric_summary(checks),
        "source_only": metric_summary(source_checks),
        "augmented_only": metric_summary(augmented_checks),
        "by_is_source": grouped_summaries(rows, checks, "is_source"),
        "by_label": grouped_summaries(rows, checks, "label"),
        "by_mr_type": grouped_summaries(rows, checks, "mr_type"),
        "by_mr_id": grouped_summaries(rows, checks, "mr_id"),
        "augmentation_vs_source": augmentation_vs_source_summary(rows, checks),
        "pair_summary": {
            "pair_count": len(pair_rows),
            "pairs_all_rows_agree": sum(row["all_rows_agree"] for row in pair_rows),
            "pairs_all_rows_agree_at_threshold": sum(
                row["all_rows_agree_at_threshold"] for row in pair_rows
            ),
        },
        "outputs": {
            "predictions": str(predictions_path),
            "disagreements": str(disagreements_path),
            "high_confidence_agreements": str(high_confidence_path),
            "pair_quality": str(pairs_path),
        },
    }
    write_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), "overall": summary["overall"]}, indent=2))


if __name__ == "__main__":
    main()
