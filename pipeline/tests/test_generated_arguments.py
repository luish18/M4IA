import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.experiment.generated_arguments import (  # noqa: E402
    emitted_kernel_arguments,
    generated_mapping_nodes,
)


class _Parser:
    def __init__(self, representation):
        self.operatorRepresentation = representation


class _Mapper:
    def __init__(self, representation):
        self.parser = _Parser(representation)


class _Layer:
    def __init__(self, representation):
        self.mapper = _Mapper(representation)


class GeneratedKernelArgumentsTests(unittest.TestCase):

    def test_matmul_reads_selected_bound_parser_representation(self):
        layer = _Layer({
            "M": 32,
            "N": np.int64(64),
            "O": 16,
            "A": "input",
            "B": "weights",
        })
        self.assertEqual(
            emitted_kernel_arguments(layer, "MatMul"),
            {"M": 32, "N": 64, "O": 16},
        )

    def test_gemm_keeps_transpose_dispatch_arguments(self):
        layer = _Layer({
            "M": 16, "N": 64, "O": 32, "transA": 0, "transB": 1,
        })
        self.assertEqual(
            emitted_kernel_arguments(layer, "Gemm"),
            {"M": 16, "N": 64, "O": 32, "transA": 0, "transB": 1},
        )

    def test_missing_non_integer_and_bool_evidence_is_omitted(self):
        layer = _Layer({
            "M": 2,
            "N": "3",
            "O": 8.0,
            "transA": False,
            "transB": 1,
        })
        self.assertEqual(emitted_kernel_arguments(layer, "Gemm"), {"M": 2, "transB": 1})
        self.assertEqual(emitted_kernel_arguments(object(), "MatMul"), {})

    def test_unsupported_operator_has_no_argument_contract(self):
        self.assertIsNone(emitted_kernel_arguments(_Layer({"M": 1}), "Conv"))

    def test_generated_mapping_metadata_keeps_index_placement_arguments_and_explanation(self):
        explanation = {
            "strategy": "measured_rate_greedy",
            "node": "matmul_0",
            "operator": "MatMul",
            "selected_engine": "spatz",
        }
        decided = {
            "matmul_0": {
                "index": 4,
                "op": "MatMul",
                "engine": "spatz",
                "layer": _Layer({"M": 2, "N": 3, "O": 9}),
            },
        }
        nodes = generated_mapping_nodes(decided, [explanation])
        self.assertEqual(nodes, [{
            "index": 4,
            "node": "matmul_0",
            "op": "MatMul",
            "engine": "spatz",
            "kernel_arguments": {"M": 2, "N": 3, "O": 9},
            "mapping_explanation": explanation,
        }])
        self.assertEqual(decided["matmul_0"]["engine"], "spatz")

    def test_missing_mapper_explanation_is_explicitly_unavailable(self):
        decided = {
            "add_0": {
                "index": 0,
                "op": "Add",
                "engine": "cva6",
                "layer": _Layer({}),
            },
        }
        node = generated_mapping_nodes(decided, [])[0]
        self.assertNotIn("kernel_arguments", node)
        self.assertEqual(node["mapping_explanation"]["status"], "unavailable")

    def test_disagreeing_mapper_explanation_is_not_attached_as_fact(self):
        decided = {
            "matmul_0": {
                "index": 1,
                "op": "MatMul",
                "engine": "spatz",
                "layer": _Layer({"M": 2, "N": 3, "O": 8}),
            },
        }
        explanation = {
            "strategy": "measured_rate_greedy",
            "node": "matmul_0",
            "operator": "MatMul",
            "selected_engine": "snitch",
        }
        node = generated_mapping_nodes(decided, [explanation])[0]
        self.assertEqual(
            node["mapping_explanation"]["reason"],
            "mapper_explanation_disagrees_with_generated_placement",
        )

    def test_generator_uses_progress_index_and_retained_mapper_instance(self):
        source = (ROOT / "pipeline/hetero_platform/generate.py").read_text()
        self.assertIn('"index": progress.index_of(name)', source)
        self.assertIn("mapper_factory.last_instance", source)
        self.assertIn("generated_mapping_nodes(decided, explanations)", source)


if __name__ == "__main__":
    unittest.main()
