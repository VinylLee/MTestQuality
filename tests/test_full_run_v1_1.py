"""Acceptance tests against real V1.1 outputs (no model loading)."""

import unittest

import refilter_ensemble_v1_1 as refilter


class V11FullRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = refilter.load_config()
        cls.root = refilter.project_path(cls.config["output_root"])
        if not (cls.root / "summary_all_datasets.json").is_file():
            raise unittest.SkipTest("Real four-dataset V1.1 run has not finished")
        cls.summary = refilter.read_json(cls.root / "summary_all_datasets.json")

    def test_all_real_rows_evidence_raw_fields_partitions_and_diffs(self):
        for dataset in refilter.DATASETS:
            with self.subTest(dataset=dataset):
                directory = self.root / dataset
                v1 = list(refilter.iter_jsonl(refilter.project_path(self.config["source_predictions_root"]) / dataset / "ensemble_predictions.jsonl"))
                rows = list(refilter.iter_jsonl(directory / "ensemble_predictions.jsonl"))
                kept, removed = list(refilter.iter_jsonl(directory / "filtered.jsonl")), list(refilter.iter_jsonl(directory / "removed.jsonl"))
                rescued = list(refilter.iter_jsonl(directory / "rescued_from_v1.jsonl"))
                diff = refilter.read_json(directory / "policy_diff_from_v1.json")
                self.assertEqual(len(v1), len(rows))
                self.assertEqual(len(kept) + len(removed), len(rows))
                self.assertEqual(kept, [r for r in rows if r[refilter.QUALITY]["decision"] == "keep"])
                self.assertEqual(removed, [r for r in rows if r[refilter.QUALITY]["decision"] == "drop"])
                self.assertEqual(rescued, [r for r in rows if r[refilter.QUALITY]["v1_decision"] == "drop" and r[refilter.QUALITY]["decision"] == "keep"])
                self.assertEqual(diff, refilter.policy_diff(rows, dataset))
                self.assertEqual(diff["keep_to_drop"], 0)
                for old, row in zip(v1, rows):
                    self.assertEqual(refilter.canonical({k: v for k, v in row.items() if k != refilter.QUALITY}), refilter.canonical(old))
                    quality = row[refilter.QUALITY]
                    self.assertEqual(quality["v1_decision"], old["ensemble_quality"]["decision"])
                    if row.get("is_source") is True:
                        self.assertEqual(quality["decision"], "keep")
                    if refilter.conditional_clause(row):
                        self.assertEqual(quality["decision"], "keep")
                source_positions = refilter.source_positions(v1)
                for row in removed:
                    quality = row[refilter.QUALITY]
                    self.assertIs(row["is_source"], False)
                    self.assertFalse(refilter.conditional_clause(row))
                    self.assertEqual(quality["v1_decision"], "drop")
                    self.assertEqual(quality["gold_support_block_count"], 0)
                    gold = refilter.normalize_label(row["label"])
                    source = v1[source_positions[refilter.pair_key(row)]]
                    votes, blockers = {}, []
                    for model_id, check in row["ensemble_quality"]["per_model"].items():
                        original_source = source["ensemble_quality"]["per_model"][model_id]
                        eligible = original_source["source_prediction"] == refilter.normalize_label(source["label"]) and original_source["source_confidence"] >= .80
                        if not eligible:
                            continue
                        if check["augmented_prediction"] == gold and check["augmented_confidence"] >= .90:
                            blockers.append(model_id)
                        if check["augmented_prediction"] != gold and check["augmented_confidence"] >= .95:
                            votes.setdefault(check["augmented_prediction"], []).append(model_id)
                    self.assertFalse(blockers)
                    self.assertGreaterEqual(len(votes[quality["wrong_consensus_label"]]), 4 if dataset == "sick" else 3)
                    self.assertEqual(votes[quality["wrong_consensus_label"]], quality["wrong_supporting_models"])
                    if dataset in ("mnlim", "mnlimm"):
                        self.assertTrue(any(quality["per_model"][m]["model_name"] == refilter.WANLI for m in quality["wrong_supporting_models"]))
                    if dataset == "sick":
                        self.assertEqual(quality["eligible_voter_count"], 4)
                manifest = refilter.read_json(directory / "model_manifest.json")
                for name, digest in manifest["output_sha256"].items():
                    self.assertEqual(refilter.sha256_file(directory / name), digest)

    def test_real_protected_inputs_match_before_after_receipts(self):
        for receipt_key, root in (("source_v1_integrity", refilter.project_path(self.config["source_predictions_root"])),
                                  ("original_data_integrity", refilter.PROJECT_ROOT / "data")):
            receipt = self.summary[receipt_key]
            actual = refilter.snapshot_tree(root)
            self.assertTrue(receipt["verified"])
            if receipt_key == "original_data_integrity":
                # New datasets may be added; every historically receipted file
                # must still exist with unchanged bytes. V1 stays strictly frozen.
                self.assertLessEqual(set(receipt["files"]), set(actual))
            else:
                self.assertEqual(receipt["file_count_before"], len(actual))
                self.assertEqual(set(receipt["files"]), set(actual))
            for name in receipt["files"]:
                info = actual[name]
                for key, value in info.items():
                    self.assertEqual(receipt["files"][name][f"{key}_before"], receipt["files"][name][f"{key}_after"])
                    # A Git checkout changes mtimes. Receipts prove run-time mtime
                    # preservation; persisted contents/count/size stay portable.
                    if key != "mtime_ns":
                        self.assertEqual(receipt["files"][name][f"{key}_after"], value)

    def test_real_aggregate_sensitivity_and_rescue_accounting(self):
        summary = self.summary
        diff = refilter.read_json(self.root / "diff_v1_vs_v1_1.json")
        self.assertEqual(summary["totals"]["input_count"], 74636)
        self.assertEqual(summary["totals"]["source_count"], 34377)
        self.assertEqual(summary["totals"]["removed_source_count"], 0)
        self.assertFalse(summary["inference_performed"])
        self.assertEqual(diff["TOTAL"]["keep_to_drop"], 0)
        self.assertEqual(diff["TOTAL"]["drop_to_keep"] + diff["TOTAL"]["unchanged_drop"], diff["TOTAL"]["v1_removed"])
        for key in refilter.DIFF_COUNTS:
            self.assertEqual(diff["TOTAL"][key], sum(diff[d.upper()][key] for d in refilter.DATASETS))
        for dimension in refilter.dimensions():
            self.assertEqual(sum(diff["TOTAL"]["rescued_by"][dimension].values()), diff["TOTAL"]["drop_to_keep"])
            self.assertEqual(sum(summary["remaining_removed_by"][dimension].values()), summary["totals"]["removed_augmented_count"])
        for dataset, item in summary["datasets"].items():
            self.assertEqual(item["threshold_sensitivity"]["0.95"]["removed_augmented_count"], item["removed_augmented_count"])
            self.assertEqual(sum(v["count"] for v in item["eligibility_diagnostics"].values()), item["augmented_count"])
            self.assertEqual(item["evidence_provenance"]["offline_join_model_ids"], [])
            for sensitivity in item["threshold_sensitivity"].values():
                self.assertEqual(sensitivity["gold_support_block_threshold"], .90)


if __name__ == "__main__":
    unittest.main()
