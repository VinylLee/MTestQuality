import copy
import json
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import run_merged_quality_audit as merged


def row(i, source=False, premise="A person walks.", hypothesis="Someone moves.", label=0, pair=0, decision="keep", conditional=False):
    return {"idx": i, "pair_id": pair, "is_source": source, "premise": premise, "hypothesis": hypothesis,
            "label": label, "mr_id": "conditional_clause" if conditional else "test", "mr_type": "inv",
            "ensemble_quality_v1_1": {"decision": decision, "reason": "test"}}


def structure(rows):
    return merged.audit_structure([{k: v for k, v in r.items() if k != "ensemble_quality_v1_1"} for r in rows])


class DeterministicQualityTests(unittest.TestCase):
    def test_source_priority_and_all_sources_retained(self):
        rows = [row(0), row(1, True), row(2, True, pair=1), row(3)]
        result = merged.finalize_rows(rows, structure(rows))
        self.assertEqual([r[merged.QUALITY]["decision"] for r in result], ["drop", "keep", "keep", "drop"])
        self.assertEqual(rows[0]["premise"], result[0]["premise"])

    def test_conflict_quarantines_every_augmentation_including_conditional(self):
        rows = [row(0, True), row(1, label=1), row(2, conditional=True)]
        result = merged.finalize_rows(rows, structure(rows))
        self.assertEqual([r[merged.QUALITY]["reason"] for r in result], ["source_preserved", "label_conflict", "label_conflict"])
        self.assertIsNotNone(result[0][merged.QUALITY]["conflict_group_id"])

    def test_unicode_whitespace_comparison_preserves_raw_text(self):
        rows = [row(0, True, premise="Else", hypothesis="Other"), row(1, premise="Cafe\u0301   x\t"), row(2, premise="Caf\u00e9 x")]
        result = merged.finalize_rows(rows, structure(rows))
        self.assertEqual(result[2][merged.QUALITY]["reason"], "duplicate")
        self.assertEqual(result[1]["premise"], "Cafe\u0301   x\t")
        self.assertTrue(structure(rows)[0]["normalization_only_groups"])

    def test_no_case_punctuation_or_cross_dataset_deduplication(self):
        rows = [row(0, premise="A"), row(1, premise="a"), row(2, premise="A!")]
        result = merged.finalize_rows(rows, structure(rows))
        self.assertTrue(all(r[merged.QUALITY]["decision"] == "keep" for r in result))
        separate = [row(0, premise="A")]
        self.assertEqual(merged.finalize_rows(separate, structure(separate))[0][merged.QUALITY]["decision"], "keep")

    def test_representative_chosen_after_model_filtering(self):
        rows = [row(0, True, premise="Other"), row(1, decision="drop"), row(2), row(3)]
        result = merged.finalize_rows(rows, structure(rows))
        self.assertEqual([r[merged.QUALITY]["reason"] for r in result], ["source_preserved", "model_v1_1", "augmentation_preserved", "duplicate"])
        self.assertEqual(result[3][merged.QUALITY]["duplicate_representative_position"], 2)

    def test_structure_does_not_guess_or_repair(self):
        for changes in ({"label": "0"}, {"is_source": 1}, {"idx": True}, {"premise": 2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                structure([{**row(0), **changes}])
        with self.assertRaises(ValueError):
            structure([row(0), row(0)])
        rows = [row(0, pair=99)]
        self.assertEqual(structure(rows)[1], [["missing_source"]])

    def test_no_change_and_format_only_reports(self):
        rows = [row(0, True), row(1), row(2, premise="A  person walks. ")]
        report = structure(rows)[0]
        self.assertEqual(report["unchanged_augmentation_positions"], [1])
        self.assertEqual(report["format_only_augmentation_positions"], [2])


class ProbabilityAndResumeTests(unittest.TestCase):
    def test_cache_ambiguity_is_sticky(self):
        index = {}
        a = {"probabilities": {"entailment": .9, "neutral": .05, "contradiction": .05}, "input_token_count": 7}
        b = {**a, "probabilities": {"entailment": .8, "neutral": .15, "contradiction": .05}}
        merged.add_candidate(index, ("a", "b"), a, {"file": "a"})
        merged.add_candidate(index, ("a", "b"), a, {"file": "b"})
        self.assertEqual(len(index[("a", "b")]["origins"]), 2)
        merged.add_candidate(index, ("a", "b"), b, {"file": "c"})
        merged.add_candidate(index, ("a", "b"), a, {"file": "d"})
        self.assertIsNone(index[("a", "b")])

    def test_label_metrics_rebuilt(self):
        probs = {"entailment": .9, "neutral": .08, "contradiction": .02}
        correct = merged.rebuild_check({"label": 0}, 7, probs, 12, {"kind": "historical_reuse"})
        changed = merged.rebuild_check({"label": 1}, 13, probs, 12, {"kind": "historical_reuse"})
        self.assertTrue(correct["agreement"])
        self.assertFalse(changed["agreement"])
        self.assertEqual(changed["row_position"], 13)
        self.assertEqual(changed["gold_probability"], .08)
        self.assertEqual(changed["gold_rank"], 2)
        self.assertGreater(changed["negative_log_likelihood"], correct["negative_log_likelihood"])

    def test_invalid_probability_rejected(self):
        for p in ({"entailment": .8, "neutral": .3, "contradiction": 0},
                  {"entailment": float("nan"), "neutral": 0, "contradiction": 0}):
            with self.assertRaises(ValueError):
                merged.rebuild_check({"label": 0}, 0, p, 12, {})

    def test_oom_batch_sequence(self):
        sizes = [16]
        for _ in range(5):
            sizes.append(merged.next_batch_size(sizes[-1]))
        self.assertEqual(sizes, [16, 8, 4, 2, 1, 1])

    def test_real_inference_loop_oom_retries_and_resumes_without_model_loading(self):
        torch = merged.evaluator.torch
        cpu = torch.device("cpu")
        config = {"runtime": {"device": "cuda:1", "batch_size": 16, "max_length": 256, "fp16": True},
                  "min_free_gpu_mib": 1, "gpu_wait_seconds": 1}
        model = {"name": "cross-encoder/nli-deberta-v3-large", "revision": "a" * 40,
                 "id": "model_a", "label_order": ["contradiction", "entailment", "neutral"]}
        calls = []

        class Network:
            config = SimpleNamespace(num_labels=3, _commit_hash=model["revision"])

            def to(self, device): return self
            def eval(self): return self
            def __call__(self, **encoded):
                count = encoded["attention_mask"].shape[0]
                calls.append(count)
                if len(calls) <= 2:
                    raise RuntimeError("CUDA out of memory")
                return SimpleNamespace(logits=torch.tensor([[1., 4., 2.]] * count))

        def tokenizer(premises, hypotheses, **kwargs):
            return {"attention_mask": torch.ones((len(premises), 5), dtype=torch.long)}

        with tempfile.TemporaryDirectory() as temp, patch.object(torch, "device", return_value=cpu), \
                patch.object(torch.cuda, "mem_get_info", return_value=(10**10, 10**10)), \
                patch.object(torch.cuda, "get_device_name", return_value="fixture"), \
                patch.object(torch.cuda.amp, "autocast", return_value=nullcontext()), \
                patch.object(merged.evaluator.AutoTokenizer, "from_pretrained", return_value=tokenizer), \
                patch.object(merged.evaluator.AutoModelForSequenceClassification, "from_pretrained", return_value=Network()) as loader:
            keys = [(str(i), "h") for i in range(19)]
            values, receipt = merged.infer_missing(config, model, keys, Path(temp), {"fixed": True})
            self.assertEqual(calls[:3], [16, 8, 4])
            self.assertEqual(len(values), 19)
            self.assertEqual(len(receipt["oom_events"]), 2)
            self.assertTrue(all(s["batch_size"] == 4 for s in receipt["shards"]))
            loader.reset_mock()
            resumed, _ = merged.infer_missing(config, model, keys, Path(temp), {"fixed": True})
            self.assertEqual(resumed, values)
            loader.assert_not_called()

    def test_checkpoint_resume_hash_and_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            identity = {"model": "fixed", "input": "fixed"}
            keys = [("a", "b"), ("c", "d")]
            item = {"position": 0, "probabilities": {"entailment": .9, "neutral": .08, "contradiction": .02}, "input_token_count": 10, "batch_size": 1}
            merged.write_json(directory / "batch-000000.json", [item])
            receipt = {"identity": identity, "shards": [{"file": "batch-000000.json", "sha256": merged.evaluator.sha256_file(directory / "batch-000000.json")}], "oom_events": []}
            merged.write_json(directory / "checkpoint.json", receipt)
            values, actual = merged.checkpoint_load(directory, identity, keys)
            self.assertEqual(values[keys[0]], item)
            with self.assertRaises(ValueError):
                merged.checkpoint_load(directory, {"input": "changed"}, keys)
            merged.write_json(directory / "batch-000001.json", [item])
            merged.checkpoint_load(directory, identity, keys)
            self.assertTrue((directory / "batch-000001.uncommitted").exists())
            (directory / "batch-000000.json").write_text("[]")
            with self.assertRaises(ValueError):
                merged.checkpoint_load(directory, identity, keys)


class RealMergedAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = merged.ROOT / "outputs/mr_test_data_merged_quality_v1"
        if not cls.root.exists():
            raise unittest.SkipTest("Merged audit not yet published")

    def test_all_artifact_hashes(self):
        hashes = merged.read_json(self.root / "artifact_hashes.json")
        actual = {str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file() and p.name != "artifact_hashes.json"}
        self.assertEqual(set(hashes), actual)
        for name, sha in hashes.items():
            self.assertEqual(merged.evaluator.sha256_file(self.root / name), sha, name)

    def test_all_rows_fields_decisions_and_partitions(self):
        summary = merged.read_json(self.root / "summary_all_datasets.json")
        self.assertEqual(summary["totals"]["input_count"], 78042)
        self.assertEqual(summary["totals"]["source_count"], 34377)
        policy = merged.refilter.load_config()
        for dataset in policy["datasets"]:
            raw = merged.evaluator.load_rows(merged.ROOT / "data/mr_test_data_merged" / (dataset + ".jsonl"))
            directory = self.root / dataset
            rows = list(merged.refilter.iter_jsonl(directory / "ensemble_predictions.jsonl"))
            manifest = merged.read_json(directory / "model_manifest.json")
            evidence, _ = merged.refilter.prepare_evidence(rows, manifest["models"], self.root / "baseline_v1" / dataset)
            merged.validate_final(raw, rows, dataset, policy, evidence, merged.audit_structure(raw))
            kept, removed = list(merged.refilter.iter_jsonl(directory / "filtered.jsonl")), list(merged.refilter.iter_jsonl(directory / "removed.jsonl"))
            self.assertEqual(kept, [r for r in rows if r[merged.QUALITY]["decision"] == "keep"])
            self.assertEqual(removed, [r for r in rows if r[merged.QUALITY]["decision"] == "drop"])
            self.assertEqual(len(raw), len(kept) + len(removed))
            self.assertEqual(list(merged.refilter.iter_jsonl(directory / "conflict_quarantine.jsonl")), [r for r in removed if r[merged.QUALITY]["reason"] == "label_conflict"])

    def test_every_prediction_has_exact_input_provenance_and_current_gold_metrics(self):
        config = merged.read_json(merged.DEFAULT_CONFIG)
        baseline = merged.ensemble.load_config()
        for model in baseline["models"]:
            model_id = model["id"]
            history = {d: list(merged.refilter.iter_jsonl(merged.ROOT / config["history_root"] / d / model_id / "predictions.jsonl"))
                       for d in config["datasets"]}
            checkpoint_dir = self.root / "inference" / model_id
            inputs = list(merged.refilter.iter_jsonl(checkpoint_dir / "inputs.jsonl"))
            new_predictions = {}
            checkpoint = merged.read_json(checkpoint_dir / "checkpoint.json")
            for shard in checkpoint["shards"]:
                for item in merged.read_json(checkpoint_dir / shard["file"]):
                    self.assertNotIn(item["position"], new_predictions)
                    new_predictions[item["position"]] = item
            self.assertEqual(set(new_predictions), set(range(len(inputs))))
            for dataset in config["datasets"]:
                raw = merged.evaluator.load_rows(merged.ROOT / config["dataset_paths"][dataset])
                directory = self.root / "baseline_v1" / dataset / model_id
                checks, _, _ = merged.ensemble.load_audit(directory, raw, merged.evaluator.sha256_file(merged.ROOT / config["dataset_paths"][dataset]), model, baseline)
                for i, (row, check) in enumerate(zip(raw, checks)):
                    gold = merged.evaluator.normalize_label(row["label"])
                    self.assertEqual(check["gold_probability"], check["probabilities"][gold])
                    ranked = sorted(check["probabilities"], key=check["probabilities"].get, reverse=True)
                    self.assertEqual(check["gold_rank"], ranked.index(gold) + 1)
                    self.assertEqual(check["agreement"], check["predicted_label"] == gold)
                    origin = check["prediction_provenance"]
                    if origin["kind"] == "historical_reuse":
                        for source in origin["origins"]:
                            old = history[source["dataset"]][source["row_position"]]
                            self.assertEqual(merged.exact_key(old), merged.exact_key(row))
                            self.assertEqual(check["probabilities"], old["nli_quality"]["probabilities"])
                            self.assertEqual(check["input_token_count"], old["nli_quality"]["input_token_count"])
                    else:
                        position = origin["unique_input_position"]
                        self.assertEqual(merged.exact_key(inputs[position]), merged.exact_key(row))
                        self.assertEqual(check["probabilities"], new_predictions[position]["probabilities"])
                        self.assertEqual(origin["actual_batch_size"], new_predictions[position]["batch_size"])


if __name__ == "__main__":
    unittest.main()
