import json
import unittest

from pipeline.tests.test_mapper_safety import Node, _cluster, loaded_mapper


class MappingExplanationTests(unittest.TestCase):

    @staticmethod
    def _engines(loaded):
        return {
            "cva6": loaded.engines.Cva6HostEngine(),
            "snitch": _cluster(loaded, "snitch"),
            "spatz": _cluster(loaded, "spatz"),
        }

    def test_safe_automatic_selection_is_unchanged_and_all_engines_are_explained(self):
        with loaded_mapper() as loaded:
            engines = self._engines(loaded)
            mapper = loaded.mapper.CostEngineMapper(engines)
            node = Node(left=(32, 32), right=(32, 32), output=(32, 32))
            self.assertIs(mapper.mapNodeToEngine(node, None), engines["spatz"])

            explanation = mapper.explanations[-1]
            self.assertEqual(explanation["strategy"], "measured_rate_greedy")
            self.assertEqual(explanation["node"], "node")
            self.assertEqual(explanation["operator"], "MatMul")
            self.assertEqual(explanation["selected_engine"], "spatz")
            self.assertEqual(explanation["selection_rule"], "minimum_safe_estimated_cost")
            self.assertIsNotNone(explanation["selected_estimated_cost_cycles"])
            self.assertEqual(
                [row["engine"] for row in explanation["engines"]],
                ["cva6", "snitch", "spatz"],
            )
            self.assertTrue(all(row["compatible"] for row in explanation["engines"]))
            spatz = explanation["engines"][2]
            self.assertEqual(spatz["cost"]["rate_key"], "MatMul")
            self.assertEqual(spatz["cost"]["rate_kind"], "operator_rate")
            self.assertEqual(spatz["cost"]["rate_row"], {
                "kind": "engine", "key": "spatz",
            })

    def test_incompatible_engine_has_source_grounded_reason(self):
        with loaded_mapper() as loaded:
            engines = self._engines(loaded)
            mapper = loaded.mapper.CostEngineMapper(engines)
            self.assertIs(mapper.mapNodeToEngine(Node(op="Add"), None), engines["cva6"])
            rows = {row["engine"]: row for row in mapper.explanations[-1]["engines"]}
            self.assertFalse(rows["snitch"]["compatible"])
            self.assertEqual(
                rows["snitch"]["incompatibility_reason"],
                "operator_not_in_cluster_mapping",
            )
            self.assertEqual(rows["snitch"]["cost"], {
                "evaluated": False, "reason": "engine_incompatible",
            })

    def test_compatible_pin_is_recorded_without_fabricating_selected_cost(self):
        with loaded_mapper() as loaded:
            engines = self._engines(loaded)
            mapper = loaded.mapper.CostEngineMapper(engines, pin="snitch")
            node = Node(right=(3, 7), output=(2, 7))
            self.assertIs(mapper.mapNodeToEngine(node, None), engines["snitch"])

            explanation = mapper.explanations[-1]
            rows = {row["engine"]: row for row in explanation["engines"]}
            self.assertEqual(explanation["selection_rule"], "compatible_pinned_engine_selected")
            self.assertEqual(explanation["requested_pin"], "snitch")
            self.assertEqual(
                explanation["pin_semantics"],
                "compatible_node_preference_not_whole_graph",
            )
            self.assertIsNone(explanation["selected_estimated_cost_cycles"])
            self.assertTrue(rows["snitch"]["candidate_after_pin"])
            self.assertFalse(rows["spatz"]["candidate_after_pin"])
            self.assertEqual(rows["snitch"]["pin_influence"], "retained_by_pin")
            self.assertEqual(rows["spatz"]["exclusion_reason"], "excluded_by_compatible_pin")
            self.assertFalse(rows["snitch"]["cost"]["evaluated"])
            self.assertEqual(
                rows["snitch"]["cost"]["reason"],
                "deterministic_generic_fallback_has_no_generic_rate",
            )

    def test_explanation_cost_failure_cannot_change_compatible_pin_selection(self):
        with loaded_mapper() as loaded:
            engines = self._engines(loaded)
            mapper = loaded.mapper.CostEngineMapper(engines, pin="snitch")

            def fail_cost(_engine, _node):
                raise LookupError("observation-only failure")

            mapper.cost_breakdown = fail_cost
            self.assertIs(mapper.mapNodeToEngine(Node(), None), engines["snitch"])
            rows = {row["engine"]: row for row in mapper.explanations[-1]["engines"]}
            self.assertEqual(
                rows["snitch"]["cost"]["reason"],
                "cost_breakdown_error_during_observation",
            )

    def test_incompatible_pin_leaves_normal_automatic_candidates(self):
        with loaded_mapper() as loaded:
            engines = self._engines(loaded)
            engines["snitch"].enabled = False
            mapper = loaded.mapper.CostEngineMapper(engines, pin="snitch")
            selected = mapper.mapNodeToEngine(Node(), None)
            self.assertIs(selected, engines["cva6"])

            explanation = mapper.explanations[-1]
            rows = {row["engine"]: row for row in explanation["engines"]}
            self.assertEqual(explanation["selection_rule"], "minimum_safe_estimated_cost")
            self.assertEqual(rows["snitch"]["incompatibility_reason"], "cluster_disabled")
            self.assertEqual(rows["snitch"]["pin_influence"], "pinned_engine_incompatible")
            self.assertTrue(rows["cva6"]["candidate_after_pin"])
            self.assertTrue(rows["spatz"]["candidate_after_pin"])

    def test_no_safe_cost_remains_unselected_and_explained(self):
        with loaded_mapper() as loaded:
            snitch = _cluster(loaded, "snitch")
            mapper = loaded.mapper.CostEngineMapper({"snitch": snitch})
            self.assertIsNone(
                mapper.mapNodeToEngine(Node(right=(3, 7), output=(2, 7)), None)
            )
            explanation = mapper.explanations[-1]
            self.assertEqual(
                explanation["selection_rule"],
                "compatible_engines_without_safe_automatic_cost",
            )
            self.assertIsNone(explanation["selected_engine"])
            self.assertFalse(explanation["engines"][0]["cost"]["evaluated"])

    def test_no_compatible_engine_is_distinct_from_no_safe_cost(self):
        with loaded_mapper() as loaded:
            engines = self._engines(loaded)
            engines["cva6"].Mapping.pop("MatMul")
            engines["snitch"].enabled = False
            engines["spatz"].enabled = False
            mapper = loaded.mapper.CostEngineMapper(engines)
            self.assertIsNone(mapper.mapNodeToEngine(Node(), None))
            explanation = mapper.explanations[-1]
            self.assertEqual(explanation["selection_rule"], "no_compatible_engine")
            self.assertTrue(all(
                not row["candidate_before_pin"] for row in explanation["engines"]
            ))

    def test_default_rate_is_visibly_a_committed_proxy(self):
        with loaded_mapper() as loaded:
            host = loaded.engines.Cva6HostEngine()
            mapper = loaded.mapper.CostEngineMapper({"cva6": host})
            mapper.mapNodeToEngine(Node(op="Add"), None)
            cost = mapper.explanations[-1]["engines"][0]["cost"]
            self.assertEqual(cost["rate_key"], "_default")
            self.assertEqual(cost["rate_kind"], "explicit_default_proxy")
            self.assertEqual(cost["rate_row"], {
                "kind": "host_profile", "key": "cva6",
            })
            self.assertEqual(cost["rate_table"], {
                "kind": "committed_rate_table",
                "path": "pipeline/hetero_platform/mapper.py",
            })

    def test_external_rate_table_is_labeled_without_calibration_claim(self):
        measured = {
            "RATES": {"cva6": {"MatMul": 1.0}},
            "OFFLOAD_FIXED": {},
            "OFFLOAD_PER_BYTE": {},
        }
        with loaded_mapper(measured) as loaded:
            host = loaded.engines.Cva6HostEngine()
            mapper = loaded.mapper.CostEngineMapper({"cva6": host})
            mapper.mapNodeToEngine(Node(), None)
            source = mapper.explanations[-1]["engines"][0]["cost"]["rate_table"]
            self.assertEqual(source["kind"], "external_rate_table")
            self.assertTrue(source["path"].endswith("rates.json"))
            self.assertNotIn("calibration", json.dumps(source).lower())

    def test_explanation_serialization_is_deterministic(self):
        with loaded_mapper() as loaded:
            first = loaded.mapper.CostEngineMapper(self._engines(loaded))
            second = loaded.mapper.CostEngineMapper(self._engines(loaded))
            first.mapNodeToEngine(Node(), None)
            second.mapNodeToEngine(Node(), None)
            self.assertEqual(
                json.dumps(first.explanations, sort_keys=True, separators=(",", ":")),
                json.dumps(second.explanations, sort_keys=True, separators=(",", ":")),
            )


if __name__ == "__main__":
    unittest.main()
