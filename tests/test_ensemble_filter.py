import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evaluate_nli_quality as evaluator
import filter_nli_ensemble as filtering
import run_ensemble_audit as runner


def check(label="entailment", confidence=0.99, position=0, gold="entailment"):
    probabilities = {value: (1 - confidence) / 2 for value in evaluator.LABEL_NAME_TO_ID}
    probabilities[label] = confidence
    return {"row_position": position, "expected_label": gold, "predicted_label": label,
            "confidence": confidence, "probabilities": probabilities, "agreement": label == gold}


def row(source, label=0, pair="p", idx=0):
    return {"premise": "A person is walking.", "hypothesis": "Someone is moving.",
            "label": label, "is_source": source, "pair_id": pair, "idx": idx,
            "mr_id": "none" if source else "test_mr", "mr_type": "none" if source else "test_type"}


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = filtering.load_config()
        self.rows = [row(True), row(False, idx=1)]
        self.audits = {m["id"]: [check(), check("neutral", position=1)] for m in self.config["models"]}

    def decide(self, position=1):
        return filtering.decide_row(self.rows[position], position, self.rows,
                                    filtering.source_positions(self.rows), self.audits, self.config)

    def test_source_never_removed(self):
        for predictions in self.audits.values():
            predictions[0] = check("contradiction")
        self.assertEqual(self.decide(0)["reason"], "source_preserved")
        self.assertEqual(self.decide(0)["decision"], "keep")

    def test_three_same_wrong_votes_drop(self):
        self.audits["model_d"][1] = check("entailment", 0.70, 1)
        result = self.decide()
        self.assertEqual(result["decision"], "drop")
        self.assertEqual(result["supporting_models"], ["model_a", "model_b", "model_c"])

    def test_only_two_wrong_votes_keep(self):
        for model in ("model_c", "model_d"):
            self.audits[model][1] = check("entailment", 0.70, 1)
        self.assertEqual(self.decide()["decision"], "keep")

    def test_split_wrong_labels_keep(self):
        self.audits["model_c"][1] = check("contradiction", position=1)
        self.audits["model_d"][1] = check("entailment", 0.70, 1)
        self.assertEqual(self.decide()["wrong_consensus_count"], 2)
        self.assertEqual(self.decide()["decision"], "keep")

    def test_high_confidence_gold_blocker(self):
        self.audits["model_d"][1] = check(position=1)
        self.assertEqual(self.decide()["reason"], "high_confidence_gold_support")
        self.assertEqual(self.decide()["decision"], "keep")

    def test_wrong_source_does_not_vote(self):
        for model in ("model_c", "model_d"):
            self.audits[model][0] = check("neutral", 0.999)
        result = self.decide()
        self.assertEqual(result["eligible_voter_count"], 2)
        self.assertEqual(result["decision"], "keep")

    def test_low_confidence_source_does_not_vote(self):
        for model in ("model_c", "model_d"):
            self.audits[model][0] = check(confidence=0.799999)
        self.assertEqual(self.decide()["eligible_voter_count"], 2)
        self.assertEqual(self.decide()["decision"], "keep")

    def test_missing_source_keep(self):
        self.rows[1]["pair_id"] = "missing"
        self.assertEqual(self.decide()["reason"], "invalid_pair_structure")

    def test_duplicate_source_keep(self):
        self.rows.append(row(True, idx=2))
        self.assertEqual(self.decide()["reason"], "invalid_pair_structure")

    def test_missing_pair_and_nonboolean_flag_keep(self):
        for value in (None, "", True, []):
            self.rows[1]["pair_id"] = value
            self.assertEqual(self.decide()["reason"], "invalid_pair_structure")
        self.rows[1]["pair_id"] = "p"
        self.rows[1]["is_source"] = "false"
        self.assertEqual(self.decide()["reason"], "invalid_pair_structure")

    def test_integer_and_string_pair_ids_distinct(self):
        self.rows[0]["pair_id"] = 1
        self.rows[1]["pair_id"] = "1"
        self.assertEqual(self.decide()["reason"], "invalid_pair_structure")

    def test_incomplete_individual_prediction_keep(self):
        self.audits["model_d"][1] = None
        self.assertEqual(self.decide()["reason"], "incomplete_model_results")

    def test_threshold_boundaries_inclusive(self):
        for predictions in self.audits.values():
            predictions[0] = check(confidence=0.80)
            predictions[1] = check("neutral", 0.95, 1)
        self.assertEqual(self.decide()["decision"], "drop")
        for predictions in self.audits.values():
            predictions[1] = check("neutral", 0.949999, 1)
        self.assertEqual(self.decide()["decision"], "keep")

    def test_ineligible_gold_support_does_not_block(self):
        self.audits["model_d"] = [check("neutral"), check(position=1)]
        self.assertEqual(self.decide()["decision"], "drop")

    def test_missing_auditor_cannot_delete(self):
        self.audits.pop("model_d")
        self.assertEqual(self.decide()["reason"], "incomplete_model_results")
        self.assertEqual(self.decide()["decision"], "keep")

    def test_source_and_augmented_gold_labels_can_differ(self):
        self.rows[1]["label"] = 2
        for predictions in self.audits.values():
            predictions[1] = check("entailment", 0.99, 1, "contradiction")
        result = self.decide()
        self.assertEqual(result["eligible_voter_count"], 4)
        self.assertEqual(result["wrong_consensus_label"], "entailment")
        self.assertEqual(result["decision"], "drop")


class FileIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = filtering.load_config()
        self.config["reuse_audits"] = {}
        # Repeated idx values verify that join is by row_position, not idx.
        self.rows = [row(False, pair="q", idx=0), row(True, idx=0), row(False, idx=0),
                     row(False, pair="p", idx=0), row(True, pair="q", idx=0)]
        self.input = self.root / "original.jsonl"
        evaluator.write_jsonl(self.input, self.rows)
        self.input_bytes = self.input.read_bytes()
        self.config["dataset_paths"]["snli"] = str(self.input)
        self.dirs = {}
        for model in self.config["models"]:
            directory = self.root / model["id"]
            self.dirs[model["id"]] = directory
            self.make_audit(directory, model)

    def make_audit(self, directory, model):
        directory.mkdir(parents=True, exist_ok=True)
        rows = evaluator.load_rows(self.input)
        enriched = []
        for position, original in enumerate(rows):
            gold = evaluator.normalize_label(original["label"])
            prediction = "neutral" if position == 2 else gold
            enriched.append({**original, "nli_quality": check(prediction, position=position, gold=gold)})
        evaluator.write_jsonl(directory / "predictions.jsonl", enriched)
        for filename in filtering.AUDIT_FILES[1:-1]:
            evaluator.write_jsonl(directory / filename, [])
        evaluator.write_json(directory / "summary.json", {
            "input": {"sha256": evaluator.sha256_file(self.input), "record_count": len(rows)},
            "model": {"name": model["name"], "requested_revision": model["revision"],
                      "resolved_revision": model["revision"], "model_logit_label_order": model["label_order"]},
            "runtime": {"torch_version": "test", "transformers_version": "test", **runner.model_runtime(self.config, model)}})

    def filter(self, name="output"):
        return filtering.filter_dataset(self.config, "snli", self.dirs, self.root / name)

    def mutate_predictions(self, model_id, mutation):
        path = self.dirs[model_id] / "predictions.jsonl"
        values = [json.loads(line) for line in path.read_text().splitlines()]
        mutation(values)
        evaluator.write_jsonl(path, values)

    def assert_rejected(self):
        with self.assertRaises((ValueError, KeyError)):
            self.filter()
        self.assertFalse((self.root / "output/filtered.jsonl").exists())
        self.assertEqual(self.input.read_bytes(), self.input_bytes)

    def test_conservation_order_and_source_preservation(self):
        summary = self.filter()
        kept = evaluator.load_rows(self.root / "output/filtered.jsonl")
        removed = evaluator.load_rows(self.root / "output/removed.jsonl")
        self.assertEqual(len(kept) + len(removed), len(self.rows))
        self.assertEqual([r["ensemble_quality"]["row_position"] for r in kept], [0, 1, 3, 4])
        self.assertEqual(len([r for r in kept if r["is_source"]]), 2)
        self.assertEqual(summary["removed_source_count"], 0)
        self.assertEqual(summary["removed_augmented_count"], 1)
        self.assertEqual(removed[0]["ensemble_quality"]["wrong_consensus_count"], 4)
        self.assertEqual(self.input.read_bytes(), self.input_bytes)

    def test_deterministic_outputs(self):
        first, second = self.filter("one"), self.filter("two")
        self.assertEqual(first, second)
        for filename in filtering.OUTPUT_FILES[:3]:
            self.assertEqual((self.root / "one" / filename).read_bytes(), (self.root / "two" / filename).read_bytes())

    def test_sensitivity_and_aggregate_statistics(self):
        summary = self.filter()
        self.assertEqual(set(summary["threshold_sensitivity"]), {"0.90", "0.95", "0.99"})
        self.assertEqual(summary["by"]["mr_id"]["test_mr"]["removed_augmented_count"], 1)
        combined = filtering.aggregate_summaries([summary])
        self.assertEqual(combined["totals"]["augmented_count"], 3)
        self.assertEqual(combined["by_gold_label"]["entailment"]["removed_augmented_count"], 1)

    def test_hash_mismatch_rejected(self):
        path = self.dirs["model_a"] / "summary.json"
        metadata = json.loads(path.read_text())
        metadata["input"]["sha256"] = "0" * 64
        evaluator.write_json(path, metadata)
        self.assert_rejected()

    def test_revision_mismatch_rejected(self):
        path = self.dirs["model_a"] / "summary.json"
        metadata = json.loads(path.read_text())
        metadata["model"]["resolved_revision"] = "0" * 40
        evaluator.write_json(path, metadata)
        self.assert_rejected()

    def test_missing_and_excess_records_rejected(self):
        self.mutate_predictions("model_a", lambda rows: rows.pop())
        self.assert_rejected()
        self.make_audit(self.dirs["model_a"], self.config["models"][0])
        self.mutate_predictions("model_a", lambda rows: rows.append(rows[-1]))
        self.assert_rejected()

    def test_row_position_and_original_fields_rejected(self):
        self.mutate_predictions("model_a", lambda rows: rows[1]["nli_quality"].update(row_position=0))
        self.assert_rejected()
        self.make_audit(self.dirs["model_a"], self.config["models"][0])
        self.mutate_predictions("model_a", lambda rows: rows[1].update(pair_id="different"))
        self.assert_rejected()

    def test_original_field_types_cannot_change(self):
        self.mutate_predictions("model_a", lambda rows: rows[1].update(is_source=1))
        self.assert_rejected()

    def test_threshold_uses_probability_not_inconsistent_confidence(self):
        for model_id in self.dirs:
            def mutate(rows):
                rows[2]["nli_quality"] = check("neutral", 0.949999, 2)
                rows[2]["nli_quality"]["confidence"] = 0.95
            self.mutate_predictions(model_id, mutate)
        self.assertEqual(self.filter()["removed_augmented_count"], 0)

    def test_nan_inf_bad_sum_and_top1_rejected(self):
        for value in (float("nan"), float("inf"), 0.2, -0.1):
            self.make_audit(self.dirs["model_a"], self.config["models"][0])
            self.mutate_predictions("model_a", lambda rows: rows[2]["nli_quality"]["probabilities"].update(neutral=value))
            self.assert_rejected()

    def test_low_source_accuracy_rejected(self):
        def mutate(rows):
            for i in (1, 4):
                rows[i]["nli_quality"] = check("neutral", position=i)
        self.mutate_predictions("model_a", mutate)
        self.assert_rejected()

    def test_incomplete_individual_record_kept(self):
        self.mutate_predictions("model_a", lambda rows: rows[2]["nli_quality"].pop("probabilities"))
        summary = self.filter()
        self.assertEqual(summary["removed_augmented_count"], 0)
        self.assertEqual(summary["by"]["decision_reason"]["incomplete_model_results"]["input_count"], 1)

    def test_dataset_without_sources_keeps_all_invalid_pairs(self):
        for original in self.rows:
            original["is_source"] = False
        evaluator.write_jsonl(self.input, self.rows)
        for model in self.config["models"]:
            self.make_audit(self.dirs[model["id"]], model)
        summary = self.filter()
        self.assertEqual(summary["removed_augmented_count"], 0)
        self.assertEqual(summary["source_count"], 0)
        self.assertEqual(summary["by"]["decision_reason"]["invalid_pair_structure"]["input_count"], len(self.rows))
        self.assertIsNone(summary["source_auditor_health"]["model_a"]["accuracy"])

    def test_cannot_overwrite_inputs_or_existing_output(self):
        with self.assertRaises(ValueError):
            filtering.filter_dataset(self.config, "snli", self.dirs, self.input.parent)
        self.filter()
        with self.assertRaises(FileExistsError):
            self.filter()
        self.assertEqual(self.input.read_bytes(), self.input_bytes)

    def test_runner_sequence_resume_and_failure(self):
        seen = []
        def run(config, model, input_path, output_dir, log_path):
            seen.append(model["id"])
            self.make_audit(output_dir, model)
        with patch.object(runner, "run_model", side_effect=run):
            result = runner.audit_dataset(self.config, "snli", self.root / "runner")
        self.assertEqual(seen, [m["id"] for m in self.config["models"]])
        with patch.object(runner, "run_model", side_effect=AssertionError("Should not rerun")):
            self.assertEqual(result, runner.audit_dataset(self.config, "snli", self.root / "runner", resume=True))
        with patch.object(runner, "run_model", side_effect=subprocess.CalledProcessError(1, "evaluator")):
            with self.assertRaises(subprocess.CalledProcessError):
                runner.audit_dataset(self.config, "snli", self.root / "failure")
        self.assertFalse((self.root / "failure/snli/filtered.jsonl").exists())


class EvaluatorCompatibilityTests(unittest.TestCase):
    def test_verified_model_orders_and_generic_fallback(self):
        config = filtering.load_config()
        for model in config["models"]:
            mapping = SimpleNamespace(id2label={str(i): label for i, label in enumerate(model["label_order"])})
            self.assertEqual(list(evaluator.canonical_model_label_order(model["name"], mapping, None)), model["label_order"])
            generic = SimpleNamespace(id2label={i: f"LABEL_{i}" for i in range(3)})
            self.assertEqual(list(evaluator.canonical_model_label_order(model["name"], generic, None)), model["label_order"])
        with self.assertRaises(ValueError):
            evaluator.canonical_model_label_order("unknown/model", generic, None)

    def test_normalized_labels(self):
        for original, expected in ((0, "entailment"), ("1", "neutral"), ("contradicts", "contradiction")):
            self.assertEqual(evaluator.normalize_label(original), expected)
        for invalid in (True, 3, "unknown"):
            with self.assertRaises(ValueError):
                evaluator.normalize_label(invalid)

    def test_unpinned_config_rejected(self):
        config = filtering.load_config()
        config["models"][0]["revision"] = "main"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            evaluator.write_json(path, config)
            with self.assertRaises(ValueError):
                filtering.load_config(path)


if __name__ == "__main__":
    unittest.main()
