import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.experiment.actual_implementation import (  # noqa: E402
    EVIDENCE_TYPE,
    annotate_completed_nodes,
    resolve_actual_implementation,
)
from pipeline.experiment.kernel_implementation_catalog import build_catalog  # noqa: E402
from pipeline.experiment.matmul_gemm_implementation import (  # noqa: E402
    CVA6_GENERIC,
    SNITCH_GENERIC,
    SNITCH_TUNED,
    SPATZ_GENERIC,
    SPATZ_TUNED,
)


def _completed(index, engine, op="MatMul", cycles=99):
    return {"node": index, "op": op, "engine": engine, "cycles": cycles}


def _generated(index, engine, op="MatMul", **arguments):
    return {
        "index": index,
        "node": f"{op.lower()}_{index}",
        "op": op,
        "engine": engine,
        "kernel_arguments": arguments,
    }


class ActualImplementationTests(unittest.TestCase):

    def resolve(self, engine, op="MatMul", index=0, **arguments):
        return resolve_actual_implementation(
            _completed(index, engine, op),
            _generated(index, engine, op, **arguments),
        )

    def test_reported_ids_are_accepted_catalog_ids(self):
        ids = {entry["id"] for entry in build_catalog(ROOT)}
        self.assertTrue({
            CVA6_GENERIC,
            SNITCH_GENERIC,
            SNITCH_TUNED,
            SPATZ_GENERIC,
            SPATZ_TUNED,
        }.issubset(ids))

    def test_completed_cva6_matmul_and_gemm_are_generic(self):
        for op in ("MatMul", "Gemm"):
            with self.subTest(op=op):
                result = resolve_actual_implementation(_completed(0, "cva6", op))
                self.assertEqual(result["implementation_id"], CVA6_GENERIC)
                self.assertFalse(result["fallback_used"])
                self.assertEqual(result["evidence"]["type"], EVIDENCE_TYPE)
                self.assertNotIn("generated_dispatch_arguments", result["evidence"]["sources"])

    def test_snitch_tuned_scalar_tail_and_source_fallback(self):
        tuned = self.resolve("snitch", M=2, N=3, O=8)
        tail = self.resolve("snitch", M=2, N=3, O=9)
        small = self.resolve("snitch", M=2, N=3, O=7)
        empty = self.resolve("snitch", M=0, N=3, O=8)
        self.assertEqual(tuned["implementation_id"], SNITCH_TUNED)
        self.assertEqual(tail["implementation_id"], SNITCH_TUNED)
        self.assertEqual(
            tail["implementation_detail"],
            "scalar_tail_for_remaining_output_columns",
        )
        self.assertEqual(small["implementation_id"], SNITCH_GENERIC)
        self.assertEqual(small["fallback_reason"], "output_columns_below_ssr_unroll")
        self.assertEqual(empty["fallback_reason"], "empty_input_dimension")

    def test_spatz_tuned_transpose_and_empty_paths(self):
        tuned = self.resolve("spatz", "Gemm", M=2, N=3, O=8, transA=0, transB=0)
        transpose = self.resolve(
            "spatz", "Gemm", M=2, N=3, O=8, transA=1, transB=0
        )
        empty = self.resolve("spatz", M=2, N=0, O=8)
        self.assertEqual(tuned["implementation_id"], SPATZ_TUNED)
        self.assertEqual(transpose["implementation_id"], SPATZ_GENERIC)
        self.assertEqual(transpose["fallback_reason"], "transposed_operand")
        self.assertEqual(empty["implementation_id"], SPATZ_GENERIC)
        self.assertEqual(empty["fallback_reason"], "empty_dimension")

    def test_missing_dispatch_or_runtime_evidence_remains_unknown(self):
        missing_dispatch = resolve_actual_implementation(_completed(0, "spatz"))
        missing_dimensions = resolve_actual_implementation(
            _completed(0, "snitch"), _generated(0, "snitch", M=2, N=3)
        )
        mapper_placement_only = resolve_actual_implementation({
            "op": "MatMul", "engine": "cva6",
        })
        missing_transpose = self.resolve("spatz", "Gemm", M=2, N=3, O=8)
        unsupported = self.resolve("spatz", "Conv", M=2, N=3, O=8)
        for result in (
            missing_dispatch,
            missing_dimensions,
            mapper_placement_only,
            missing_transpose,
            unsupported,
        ):
            self.assertIsNone(result["implementation_id"])
            self.assertEqual(result["evidence"]["type"], "unavailable")

    def test_generated_runtime_operator_and_engine_mismatches_remain_unknown(self):
        op_mismatch = resolve_actual_implementation(
            _completed(0, "spatz", "MatMul"),
            _generated(0, "spatz", "Gemm", M=2, N=3, O=8, transA=0, transB=0),
        )
        engine_mismatch = resolve_actual_implementation(
            _completed(0, "snitch"),
            _generated(0, "spatz", M=2, N=3, O=8),
        )
        self.assertIn("operator mismatch", op_mismatch["evidence"]["reason"])
        self.assertIn("engine mismatch", engine_mismatch["evidence"]["reason"])

    def test_join_uses_stable_index_and_does_not_mutate_inputs(self):
        mapping = {"nodes": [
            _generated(7, "spatz", "Gemm", M=2, N=3, O=8, transA=0, transB=0),
            _generated(2, "snitch", M=2, N=3, O=9),
        ]}
        result = {"nodes": [_completed(2, "snitch"), _completed(7, "spatz", "Gemm")]}
        original_mapping = copy.deepcopy(mapping)
        original_result = copy.deepcopy(result)
        annotated = annotate_completed_nodes(mapping, result)
        self.assertEqual(
            annotated["nodes"][0]["implementation"]["implementation_id"],
            SNITCH_TUNED,
        )
        self.assertEqual(
            annotated["nodes"][1]["implementation"]["implementation_id"],
            SPATZ_TUNED,
        )
        self.assertEqual(mapping, original_mapping)
        self.assertEqual(result, original_result)

    def test_missing_or_duplicate_generated_index_remains_unknown(self):
        duplicate = _generated(3, "snitch", M=2, N=3, O=8)
        mapping = {"nodes": [duplicate, dict(duplicate, node="duplicate_name")]}
        annotated = annotate_completed_nodes(
            mapping,
            {"nodes": [_completed(3, "snitch"), _completed(4, "snitch")]},
        )
        self.assertIn("ambiguous", annotated["nodes"][0]["implementation"]["evidence"]["reason"])
        self.assertIn("unavailable", annotated["nodes"][1]["implementation"]["evidence"]["reason"])

    def test_partial_stalled_result_annotates_only_completed_nodes_and_keeps_status(self):
        mapping = {"nodes": [
            _generated(0, "snitch", M=2, N=3, O=8),
            _generated(1, "spatz", M=2, N=3, O=8),
        ]}
        result = {
            "status": "stalled",
            "nodes": [_completed(0, "snitch")],
            "log_tail": ["still waiting"],
        }
        annotated = annotate_completed_nodes(mapping, result)
        self.assertEqual(annotated["status"], "stalled")
        self.assertEqual(annotated["log_tail"], ["still waiting"])
        self.assertEqual(len(annotated["nodes"]), 1)
        self.assertEqual(annotated["nodes"][0]["node"], 0)
        self.assertEqual(
            annotated["nodes"][0]["implementation"]["implementation_id"],
            SNITCH_TUNED,
        )
        self.assertNotIn("implementation", mapping["nodes"][0])
        self.assertEqual(len(mapping["nodes"]), 2)


if __name__ == "__main__":
    unittest.main()
