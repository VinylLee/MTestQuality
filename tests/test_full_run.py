"""Acceptance checks for real four-dataset outputs; skipped before the full run."""

import json
import unittest
from pathlib import Path

from evaluate_nli_quality import normalize_label, sha256_file
from filter_nli_ensemble import PROJECT_ROOT, load_config, pair_key, source_positions


def jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class FullRunAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.root = PROJECT_ROOT / cls.config["output_root"]
        if not (cls.root / "summary_all_datasets.json").is_file():
            raise unittest.SkipTest("Real four-dataset run has not finished")

    def test_original_data_and_legacy_audit_unchanged(self):
        before = json.loads((self.root / "input_integrity_before.json").read_text())
        self.assertGreaterEqual(len(before), 18)
        for path, digest in before.items():
            with self.subTest(path=path):
                self.assertEqual(sha256_file(PROJECT_ROOT / path), digest)

    def test_all_real_decisions_and_output_conservation(self):
        for dataset in self.config["datasets"]:
            with self.subTest(dataset=dataset):
                directory = self.root / dataset
                raw = jsonl(PROJECT_ROOT / self.config["dataset_paths"][dataset])
                enriched = jsonl(directory / "ensemble_predictions.jsonl")
                kept, removed = jsonl(directory / "filtered.jsonl"), jsonl(directory / "removed.jsonl")
                summary = json.loads((directory / "ensemble_summary.json").read_text())
                manifest = json.loads((directory / "model_manifest.json").read_text())
                self.assertEqual(len(enriched), len(raw))
                self.assertEqual(len(kept) + len(removed), len(raw))
                self.assertEqual(kept, [r for r in enriched if r["ensemble_quality"]["decision"] == "keep"])
                self.assertEqual(removed, [r for r in enriched if r["ensemble_quality"]["decision"] == "drop"])
                self.assertEqual(summary["removed_source_count"], 0)
                self.assertEqual(summary["removed_augmented_count"], len(removed))
                self.assertEqual(manifest["input"]["sha256"], sha256_file(PROJECT_ROOT / self.config["dataset_paths"][dataset]))
                for filename, digest in manifest["output_sha256"].items():
                    self.assertEqual(sha256_file(directory / filename), digest)
                sources = source_positions(raw)
                predictions = {}
                for model in self.config["models"]:
                    model_id = model["id"]
                    predictions[model_id] = jsonl(directory / model_id / "predictions.jsonl")
                    self.assertEqual(len(predictions[model_id]), len(raw))
                    self.assertEqual(manifest["models"][model_id]["resolved_revision"], model["revision"])
                    self.assertGreaterEqual(summary["source_auditor_health"][model_id]["accuracy"], 0.40)
                    self.assertEqual(sha256_file(directory / model_id / "predictions.jsonl"), manifest["models"][model_id]["predictions_sha256"])
                kept_positions = [r["ensemble_quality"]["row_position"] for r in kept]
                removed_positions = [r["ensemble_quality"]["row_position"] for r in removed]
                self.assertEqual(kept_positions, sorted(kept_positions))
                self.assertEqual(removed_positions, sorted(removed_positions))
                self.assertEqual(sorted(kept_positions + removed_positions), list(range(len(raw))))
                for position, (original, audited) in enumerate(zip(raw, enriched)):
                    self.assertEqual({k: v for k, v in audited.items() if k != "ensemble_quality"}, original)
                    decision = audited["ensemble_quality"]
                    self.assertEqual(decision["row_position"], position)
                    if original.get("is_source") is True:
                        self.assertEqual(decision["decision"], "keep")
                    if decision["decision"] != "drop":
                        continue
                    self.assertIs(original["is_source"], False)
                    source_position = sources[pair_key(original)]
                    gold = normalize_label(original["label"])
                    source_gold = normalize_label(raw[source_position]["label"])
                    votes, blockers, eligible = {}, 0, 0
                    for model_id, model_rows in predictions.items():
                        source = model_rows[source_position]["nli_quality"]
                        augmented = model_rows[position]["nli_quality"]
                        qualifies = source["predicted_label"] == source_gold and source["confidence"] >= 0.80
                        eligible += int(qualifies)
                        if not qualifies or augmented["confidence"] < 0.95:
                            continue
                        if augmented["predicted_label"] == gold:
                            blockers += 1
                        else:
                            votes.setdefault(augmented["predicted_label"], []).append(model_id)
                    self.assertGreaterEqual(eligible, 3)
                    self.assertEqual(blockers, 0)
                    self.assertNotEqual(decision["wrong_consensus_label"], gold)
                    self.assertGreaterEqual(len(votes[decision["wrong_consensus_label"]]), 3)
                    self.assertEqual(decision["supporting_models"], votes[decision["wrong_consensus_label"]])

    def test_cross_dataset_summary_and_sensitivity(self):
        total = json.loads((self.root / "summary_all_datasets.json").read_text())
        self.assertEqual(set(total["datasets"]), set(self.config["datasets"]))
        self.assertEqual(total["totals"]["input_count"], 74636)
        self.assertEqual(total["totals"]["source_count"], 34377)
        self.assertEqual(total["totals"]["removed_source_count"], 0)
        self.assertEqual(set(total["threshold_sensitivity"]), {"0.90", "0.95", "0.99"})
        for key in ("by_mr_type", "by_mr_id", "by_gold_label"):
            self.assertEqual(sum(v["removed_augmented_count"] for v in total[key].values()), total["totals"]["removed_augmented_count"])


if __name__ == "__main__":
    unittest.main()
