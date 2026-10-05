"""Offline V1.1 policy, provenance, conservation and no-inference regression tests."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import refilter_ensemble_v1_1 as refilter


def example_row(source=False, old="drop", mr_id="test", position=1):
    return {"idx": position, "pair_id": "pair", "is_source": source,
        "premise": "A person walks.", "hypothesis": "Nobody walks.", "label": 0 if source else 2,
        "mr_id": mr_id, "mr_type": "flip", "ensemble_quality": {
            "policy": "conservative_consensus_v1", "row_position": position,
            "decision": "keep" if source else old, "reason": "persisted_v1_reason", "per_model": {}}}


def example_evidence(predictions=None, source_confidences=None, source_predictions=None, wanli_id="model_c"):
    predictions = predictions or [("neutral", .99)] * 4
    source_confidences = source_confidences or [.99] * 4
    source_predictions = source_predictions or ["entailment"] * 4
    ids = ("model_a", "model_b", "model_c", "model_d")
    models = {m: {"model_name": refilter.WANLI if m == wanli_id else f"test/{m}"} for m in ids}
    return {"valid_pair": True, "complete": True, "per_model": {m: {
        "model_name": models[m]["model_name"], "source_prediction": source_predictions[i],
        "source_confidence": source_confidences[i], "source_gold_label": "entailment",
        "augmented_prediction": predictions[i][0], "augmented_confidence": predictions[i][1]}
        for i, m in enumerate(ids)}}


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = refilter.load_config()

    def decide(self, dataset="snli", row=None, evidence=None, **kwargs):
        return refilter.decide_row(row or example_row(), dataset, evidence or example_evidence(), self.config, **kwargs)

    def test_gold_blocker_091_blocks_three_wrong_votes(self):
        evidence = example_evidence([("neutral", .99), ("neutral", .98), ("neutral", .97), ("contradiction", .91)])
        result = self.decide(evidence=evidence)
        self.assertEqual((result["decision"], result["reason"]), ("keep", "gold_support_block"))
        self.assertEqual(result["wrong_consensus_count"], 3)
        self.assertEqual(result["gold_support_block_count"], 1)

    def test_gold_089_does_not_block_snli_three_votes(self):
        result = self.decide(evidence=example_evidence([("neutral", .99)] * 3 + [("contradiction", .89)]))
        self.assertEqual(result["decision"], "drop")
        self.assertEqual(result["gold_support_block_count"], 0)

    def test_snli_three_votes_without_wanli_still_drop(self):
        evidence = example_evidence([("neutral", .99), ("neutral", .99), ("entailment", .70), ("neutral", .99)])
        result = self.decide(evidence=evidence)
        self.assertEqual(result["decision"], "drop")
        self.assertFalse(result["wanli_anchor_required"])
        self.assertFalse(result["wanli_anchor_present_in_wrong_consensus"])

    def test_mnli_abd_missing_wanli_anchor_keeps_both_splits(self):
        evidence = example_evidence([("neutral", .99), ("neutral", .99), ("entailment", .70), ("neutral", .99)])
        for dataset in ("mnlim", "mnlimm"):
            with self.subTest(dataset=dataset):
                result = self.decide(dataset, evidence=evidence)
                self.assertEqual((result["decision"], result["reason"]), ("keep", "mnli_wrong_consensus_missing_wanli_anchor"))

    def test_mnli_abc_with_anchor_drops_both_splits(self):
        evidence = example_evidence([("neutral", .99)] * 3 + [("entailment", .70)])
        for dataset in ("mnlim", "mnlimm"):
            result = self.decide(dataset, evidence=evidence)
            self.assertEqual(result["decision"], "drop")
            self.assertTrue(result["wanli_anchor_present_in_wrong_consensus"])

    def test_anchor_uses_actual_identity_not_model_c(self):
        evidence = example_evidence([("neutral", .99), ("neutral", .99), ("entailment", .70), ("neutral", .99)], wanli_id="model_d")
        self.assertEqual(self.decide("mnlim", evidence=evidence)["decision"], "drop")
        evidence = example_evidence([("neutral", .99)] * 3 + [("entailment", .70)], wanli_id="model_d")
        self.assertEqual(self.decide("mnlimm", evidence=evidence)["reason"], "mnli_wrong_consensus_missing_wanli_anchor")

    def test_sick_three_wrong_votes_kept(self):
        result = self.decide("sick", evidence=example_evidence([("neutral", .99)] * 3 + [("entailment", .70)]))
        self.assertEqual((result["decision"], result["reason"]), ("keep", "sick_requires_unanimous_consensus"))

    def test_sick_four_eligible_same_wrong_drops(self):
        self.assertEqual(self.decide("sick")["decision"], "drop")

    def test_sick_four_predictions_but_one_source_ineligible_keeps(self):
        self.assertEqual(self.decide("sick", evidence=example_evidence(source_confidences=[.99] * 3 + [.79]))["reason"],
                         "sick_requires_unanimous_consensus")

    def test_conditional_both_fields_protected_all_datasets(self):
        for dataset in refilter.DATASETS:
            for field in ("mr_id", "mr_type"):
                with self.subTest(dataset=dataset, field=field):
                    row = example_row()
                    row[field] = "conditional_clause"
                    result = self.decide(dataset, row, example_evidence([("neutral", .999)] * 4))
                    self.assertEqual((result["decision"], result["reason"]), ("keep", "conditional_clause_auto_drop_disabled"))

    def test_conditional_match_exact_not_fuzzy(self):
        for mr_id in ("conditional_clause_extra", "Conditional_clause", "conditional"):
            self.assertEqual(self.decide(row=example_row(mr_id=mr_id))["decision"], "drop")

    def test_source_always_first_priority_even_incomplete_conditional(self):
        row = example_row(source=True, mr_id="conditional_clause")
        evidence = {"valid_pair": False, "complete": False, "per_model": {}}
        for dataset in refilter.DATASETS:
            result = self.decide(dataset, row, evidence)
            self.assertEqual((result["decision"], result["reason"]), ("keep", "source_never_filtered"))

    def test_invalid_incomplete_before_conditional(self):
        for key in ("valid_pair", "complete"):
            evidence = example_evidence()
            evidence[key] = False
            result = self.decide(row=example_row(mr_id="conditional_clause"), evidence=evidence)
            self.assertEqual(result["reason"], "invalid_or_incomplete_evidence")

    def test_conditional_before_gold_blocker(self):
        result = self.decide(row=example_row(mr_id="conditional_clause"),
            evidence=example_evidence([("neutral", .99)] * 3 + [("contradiction", .99)]))
        self.assertEqual(result["reason"], "conditional_clause_auto_drop_disabled")
        self.assertEqual(result["gold_support_block_count"], 1)

    def test_source_and_wrong_threshold_inclusive(self):
        for dataset in refilter.DATASETS:
            self.assertEqual(self.decide(dataset, evidence=example_evidence([("neutral", .95)] * 4, [.80] * 4))["decision"], "drop")
        self.assertEqual(self.decide(evidence=example_evidence([("neutral", .949999)] * 4))["decision"], "keep")
        self.assertEqual(self.decide(evidence=example_evidence(source_confidences=[.799999] * 4))["decision"], "keep")

    def test_gold_threshold_inclusive(self):
        evidence = example_evidence([("neutral", .99)] * 3 + [("contradiction", .90)])
        self.assertEqual(self.decide(evidence=evidence)["reason"], "gold_support_block")

    def test_ineligible_gold_support_does_not_block(self):
        evidence = example_evidence([("neutral", .99)] * 3 + [("contradiction", .99)], [.99] * 3 + [.79])
        self.assertEqual(self.decide(evidence=evidence)["decision"], "drop")
        evidence = example_evidence([("neutral", .99)] * 3 + [("contradiction", .99)], source_predictions=["entailment"] * 3 + ["neutral"])
        self.assertEqual(self.decide(evidence=evidence)["decision"], "drop")

    def test_split_wrong_labels_never_combined(self):
        evidence = example_evidence([("neutral", .99)] * 2 + [("entailment", .99)] * 2)
        self.assertEqual(self.decide(evidence=evidence)["wrong_consensus_count"], 2)
        self.assertEqual(self.decide(evidence=evidence)["decision"], "keep")

    def test_gold_sensitivity_fixed_at_090(self):
        evidence = example_evidence([("neutral", .99)] * 3 + [("contradiction", .91)])
        for threshold in (.90, .95, .99):
            result = self.decide(evidence=evidence, wrong_threshold=threshold)
            self.assertEqual(result["gold_support_block_threshold"], .90)
            self.assertEqual(result["reason"], "gold_support_block")

    def test_v1_decision_is_persisted_not_simulated(self):
        row = example_row(old="keep")
        result = self.decide(row=row)
        self.assertEqual(result["v1_decision"], "keep")
        self.assertTrue(result["decision_changed"])

    def test_config_rejects_nonformal_policies(self):
        for key, value in (("wrong_vote_threshold", .90), ("gold_support_block_threshold", .95),
                           ("source_confidence_threshold", True), ("conditional_clause_auto_drop", True)):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.assertRaises(ValueError):
                refilter.validate_config(config)
        config = copy.deepcopy(self.config)
        config["dataset_policies"]["sick"]["min_wrong_consensus"] = 3
        with self.assertRaises(ValueError):
            refilter.validate_config(config)

    def test_typed_pairs_missing_and_duplicate_sources(self):
        rows = [{"pair_id": 1, "is_source": True}, {"pair_id": "1", "is_source": True},
                {"pair_id": "duplicate", "is_source": True}, {"pair_id": "duplicate", "is_source": True}]
        self.assertEqual(refilter.source_positions(rows), {("int", 1): 0, ("str", "1"): 1})
        for value in (None, True, "", []):
            self.assertIsNone(refilter.pair_key({"pair_id": value}))


def make_fixture(root):
    """Persist already-computed synthetic evidence, never instantiate a model."""
    config = refilter.load_config()
    config.update(source_predictions_root=str(root / "v1"), output_root=str(root / "v1_1"))
    names = ["test/a", "test/b", refilter.WANLI, "test/d"]
    predictions = [[("neutral", .999)] * 4, [("neutral", .999)] * 4,
        [("neutral", .99), ("neutral", .99), ("entailment", .70), ("neutral", .99)],
        [("neutral", .99)] * 3 + [("contradiction", .91)],
        [("neutral", .99)] * 3 + [("contradiction", .89)], [("contradiction", .99)] * 4]
    for dataset in refilter.DATASETS:
        directory = root / "v1" / dataset
        directory.mkdir(parents=True)
        rows = [example_row(source=True, position=0)] + [example_row(position=i + 1,
            mr_id="conditional_clause" if i == 1 else "test", old="keep" if i == 5 else "drop") for i in range(6)]
        raw = [{k: v for k, v in r.items() if k != "ensemble_quality"} for r in rows]
        input_path = root / "data" / dataset / "augmented.jsonl"
        input_path.parent.mkdir(parents=True)
        refilter.write_jsonl(input_path, raw)
        config["dataset_paths"][dataset] = str(input_path)
        digest = refilter.sha256_file(input_path)
        metadata = {"path": str(input_path), "sha256": digest, "record_count": len(rows)}
        manifest = {"schema_version": 1, "dataset": dataset, "input": metadata, "models": {}, "output_sha256": {}}
        for i, name in enumerate(names):
            model_id = f"model_{'abcd'[i]}"
            model_dir = directory / model_id
            model_dir.mkdir()
            audit_rows = []
            for position, row in enumerate(rows):
                label, confidence = ("entailment", .99) if position == 0 else predictions[position - 1][i]
                probabilities = {l: confidence if l == label else (1 - confidence) / 2 for l in refilter.LABELS}
                audit_rows.append({**raw[position], "nli_quality": {"row_position": position,
                    "expected_label": refilter.normalize_label(row["label"]), "predicted_label": label,
                    "confidence": confidence, "probabilities": probabilities}})
                check = {"source_prediction": "entailment", "source_confidence": .99}
                if position:
                    check.update(source_eligible=True, augmented_prediction=label, augmented_confidence=confidence)
                row["ensemble_quality"]["per_model"][model_id] = check
            refilter.write_jsonl(model_dir / "predictions.jsonl", audit_rows)
            refilter.write_json(model_dir / "summary.json", {"model": name})
            manifest["models"][model_id] = {"model_name": name, "resolved_revision": "a" * 40,
                "requested_revision": "a" * 40, "label_order": list(refilter.LABELS),
                "input_sha256": digest, "record_count": len(rows),
                "predictions_sha256": refilter.sha256_file(model_dir / "predictions.jsonl"),
                "summary_sha256": refilter.sha256_file(model_dir / "summary.json")}
        refilter.write_jsonl(directory / "ensemble_predictions.jsonl", rows)
        refilter.write_jsonl(directory / "filtered.jsonl", [r for r in rows if r["ensemble_quality"]["decision"] == "keep"])
        refilter.write_jsonl(directory / "removed.jsonl", [r for r in rows if r["ensemble_quality"]["decision"] == "drop"])
        manifest["output_sha256"] = {name: refilter.sha256_file(directory / name) for name in (
            "ensemble_predictions.jsonl", "filtered.jsonl", "removed.jsonl")}
        refilter.write_json(directory / "model_manifest.json", manifest)
        refilter.write_json(directory / "ensemble_summary.json", {"dataset": dataset, "input": metadata,
            "removed_augmented_count": 5, "source_auditor_health": {m: {"accuracy": 1} for m in manifest["models"]}})
    return config


class OfflineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = make_fixture(self.root)

    def run_refilter(self):
        return refilter.run_refilter(self.config, self.root / "data")

    def rewrite_ensemble(self, dataset, mutation):
        directory = self.root / "v1" / dataset
        rows = list(refilter.iter_jsonl(directory / "ensemble_predictions.jsonl"))
        mutation(rows)
        for filename, selected in (("ensemble_predictions.jsonl", rows),
            ("filtered.jsonl", [r for r in rows if r["ensemble_quality"]["decision"] == "keep"]),
            ("removed.jsonl", [r for r in rows if r["ensemble_quality"]["decision"] == "drop"])):
            refilter.write_jsonl(directory / filename, selected)
        manifest = refilter.read_json(directory / "model_manifest.json")
        for filename in manifest["output_sha256"]:
            manifest["output_sha256"][filename] = refilter.sha256_file(directory / filename)
        refilter.write_json(directory / "model_manifest.json", manifest)

    def assert_failed_unpublished(self):
        with self.assertRaises((ValueError, KeyError, FileNotFoundError)):
            self.run_refilter()
        self.assertFalse((self.root / "v1_1").exists())
        self.assertFalse(list(self.root.glob(".refilter-v1_1-*")))

    def test_all_datasets_output_conservation_order_diff_and_integrity(self):
        before = refilter.snapshot_tree(self.root / "v1")
        originals = refilter.snapshot_tree(self.root / "data")
        summary = self.run_refilter()
        self.assertEqual(before, refilter.snapshot_tree(self.root / "v1"))
        self.assertEqual(originals, refilter.snapshot_tree(self.root / "data"))
        self.assertEqual(summary["totals"]["removed_augmented_count"], 8)
        self.assertFalse(summary["inference_performed"])
        for dataset, item in summary["datasets"].items():
            directory = self.root / "v1_1" / dataset
            old = list(refilter.iter_jsonl(self.root / "v1" / dataset / "ensemble_predictions.jsonl"))
            all_rows = list(refilter.iter_jsonl(directory / "ensemble_predictions.jsonl"))
            kept = list(refilter.iter_jsonl(directory / "filtered.jsonl"))
            removed = list(refilter.iter_jsonl(directory / "removed.jsonl"))
            rescued = list(refilter.iter_jsonl(directory / "rescued_from_v1.jsonl"))
            diff = refilter.read_json(directory / "policy_diff_from_v1.json")
            self.assertEqual([r["ensemble_quality"]["row_position"] for r in all_rows], list(range(len(old))))
            self.assertEqual(len(kept) + len(removed), len(old))
            self.assertEqual(kept, [r for r in all_rows if r[refilter.QUALITY]["decision"] == "keep"])
            self.assertEqual(removed, [r for r in all_rows if r[refilter.QUALITY]["decision"] == "drop"])
            self.assertEqual([{k: v for k, v in r.items() if k != refilter.QUALITY} for r in all_rows], old)
            self.assertEqual(len(rescued), diff["drop_to_keep"])
            self.assertEqual(diff["drop_to_keep"] + diff["unchanged_drop"], diff["v1_removed"])
            self.assertEqual(diff["keep_to_drop"] + diff["unchanged_keep"], len(old) - diff["v1_removed"])
            self.assertEqual(diff["keep_to_drop"], 0)
            self.assertTrue(item["source_v1_integrity"]["verified"])
            self.assertEqual(item["evidence_provenance"]["offline_join_model_ids"], [])
            self.assertEqual(item["conditional_clause_diagnostic"]["protected_count"], 1)
            self.assertEqual(item["threshold_sensitivity"]["0.95"]["removed_augmented_count"], len(removed))
        diff = refilter.read_json(self.root / "v1_1/diff_v1_vs_v1_1.json")
        self.assertEqual(set(diff), {d.upper() for d in refilter.DATASETS} | {"TOTAL"})
        self.assertEqual(diff["TOTAL"]["v1_removed"], 20)
        self.assertEqual(diff["TOTAL"]["drop_to_keep"], 12)

    def test_no_individual_prediction_read_when_ensemble_complete(self):
        original = refilter.iter_jsonl
        def guarded(path):
            if Path(path).name == "predictions.jsonl":
                raise AssertionError("Primary ensemble evidence should suffice")
            return original(path)
        with patch.object(refilter, "iter_jsonl", side_effect=guarded):
            self.run_refilter()

    def test_missing_ensemble_field_offline_join(self):
        self.rewrite_ensemble("mnlim", lambda rows: rows[1]["ensemble_quality"]["per_model"]["model_c"].pop("augmented_confidence"))
        summary = self.run_refilter()
        self.assertEqual(summary["datasets"]["mnlim"]["evidence_provenance"]["offline_join_model_ids"], ["model_c"])
        self.assertEqual(summary["datasets"]["mnlim"]["removed_augmented_count"], 2)

    def test_missing_source_field_offline_join(self):
        self.rewrite_ensemble("snli", lambda rows: rows[0]["ensemble_quality"]["per_model"]["model_a"].pop("source_confidence"))
        summary = self.run_refilter()
        self.assertEqual(summary["datasets"]["snli"]["evidence_provenance"]["offline_join_model_ids"], ["model_a"])

    def test_mismatched_repeated_source_rejected(self):
        self.rewrite_ensemble("snli", lambda rows: rows[1]["ensemble_quality"]["per_model"]["model_a"].update(source_confidence=.80))
        self.assert_failed_unpublished()

    def test_model_artifact_hash_tamper_rejected(self):
        path = self.root / "v1/snli/model_a/predictions.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        self.assert_failed_unpublished()

    def test_v1_partition_hash_tamper_rejected(self):
        path = self.root / "v1/snli/filtered.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        self.assert_failed_unpublished()

    def test_v1_partition_order_rejected_even_updated_hash(self):
        directory = self.root / "v1/snli"
        rows = list(refilter.iter_jsonl(directory / "filtered.jsonl"))
        refilter.write_jsonl(directory / "filtered.jsonl", list(reversed(rows)))
        manifest = refilter.read_json(directory / "model_manifest.json")
        manifest["output_sha256"]["filtered.jsonl"] = refilter.sha256_file(directory / "filtered.jsonl")
        refilter.write_json(directory / "model_manifest.json", manifest)
        self.assert_failed_unpublished()

    def test_original_dataset_hash_tamper_rejected(self):
        path = self.root / "data/snli/augmented.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        self.assert_failed_unpublished()

    def test_manifest_identity_missing_anchor_rejected(self):
        path = self.root / "v1/mnlim/model_manifest.json"
        manifest = refilter.read_json(path)
        manifest["models"]["model_c"]["model_name"] = "unknown/not-wanli"
        refilter.write_json(path, manifest)
        self.assert_failed_unpublished()

    def test_original_fields_type_change_rejected(self):
        self.rewrite_ensemble("snli", lambda rows: rows[0].update(is_source=1))
        self.assert_failed_unpublished()

    def test_mtime_change_during_run_fails_without_publish(self):
        actual = refilter.process_dataset
        path = self.root / "v1/snli/ensemble_summary.json"
        def mutate(*args, **kwargs):
            result = actual(*args, **kwargs)
            if args[1] == "sick":
                stat = path.stat()
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            return result
        with patch.object(refilter, "process_dataset", side_effect=mutate):
            self.assert_failed_unpublished()

    def test_file_count_integrity_detects_added_file(self):
        before = refilter.snapshot_tree(self.root / "v1")
        refilter.write_json(self.root / "v1/extra.json", {})
        with self.assertRaises(ValueError):
            refilter.verify_integrity(self.root / "v1", before)

    def test_output_overlap_and_overwrite_refused(self):
        self.run_refilter()
        with self.assertRaises(FileExistsError):
            self.run_refilter()
        for target in (self.root / "v1/new", self.root / "data/new"):
            self.config["output_root"] = str(target)
            with self.assertRaises(ValueError):
                self.run_refilter()

    def test_invalid_pair_and_incomplete_evidence_keep(self):
        directory = self.root / "v1/snli"
        rows = list(refilter.iter_jsonl(directory / "ensemble_predictions.jsonl"))
        models = refilter.read_json(directory / "model_manifest.json")["models"]
        rows[1]["pair_id"] = "missing-source"
        evidence, _ = refilter.prepare_evidence(rows, models, directory)
        self.assertEqual(refilter.decide_row(rows[1], "snli", evidence[1], self.config)["reason"], "invalid_or_incomplete_evidence")
        rows[1]["pair_id"] = "pair"
        rows[1]["ensemble_quality"]["per_model"]["model_a"]["augmented_confidence"] = -1
        evidence, _ = refilter.prepare_evidence(rows, models, directory)
        self.assertEqual(refilter.decide_row(rows[1], "snli", evidence[1], self.config)["reason"], "invalid_or_incomplete_evidence")

    def test_no_inference_modules_imported_in_complete_offline_run(self):
        # In a fresh interpreter, reject any attempt to import loaders or old runners.
        refilter.write_json(self.root / "config.json", self.config)
        code = """
import sys, importlib.abc
class ForbiddenImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'huggingface_hub',
                'evaluate_nli_quality', 'filter_nli_ensemble', 'run_ensemble_audit'}:
            raise AssertionError('Forbidden inference-related import: ' + fullname)
sys.meta_path.insert(0, ForbiddenImports())
import refilter_ensemble_v1_1 as r
r.run_refilter(r.load_config(sys.argv[1]), sys.argv[2])
"""
        result = subprocess.run([sys.executable, "-c", code, str(self.root / "config.json"), str(self.root / "data")],
            cwd=refilter.PROJECT_ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "v1_1/summary_all_datasets.json").exists())


if __name__ == "__main__":
    unittest.main()
