import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.experiment.kernel_implementation_catalog import build_catalog  # noqa: E402
from pipeline.experiment.matmul_gemm_implementation import (  # noqa: E402
    CVA6_GENERIC,
    SNITCH_GENERIC,
    SNITCH_TUNED,
    SPATZ_GENERIC,
    SPATZ_TUNED,
    resolve_matmul_gemm_implementation,
)


class MatMulGemmImplementationTests(unittest.TestCase):

    def resolve(self, *args, **kwargs):
        return resolve_matmul_gemm_implementation(*args, **kwargs)

    def test_ids_are_the_accepted_catalog_ids(self):
        catalog_ids = {entry["id"] for entry in build_catalog(ROOT)}
        self.assertTrue({
            CVA6_GENERIC,
            SNITCH_GENERIC,
            SNITCH_TUNED,
            SPATZ_GENERIC,
            SPATZ_TUNED,
        }.issubset(catalog_ids))

    def test_snitch_fallback_threshold_and_scalar_tail(self):
        small = self.resolve("snitch", "MatMul", 2, 3, 7)
        self.assertEqual(small["implementation_id"], SNITCH_GENERIC)
        self.assertEqual(small["fallback_reason"], "output_columns_below_ssr_unroll")

        exact = self.resolve("snitch", "MatMul", 2, 3, 8)
        self.assertEqual(exact["implementation_id"], SNITCH_TUNED)
        self.assertFalse(exact["fallback_used"])

        tail = self.resolve("snitch", "MatMul", 2, 3, 9)
        self.assertEqual(tail["implementation_id"], SNITCH_TUNED)
        self.assertEqual(
            tail["implementation_detail"],
            "scalar_tail_for_remaining_output_columns",
        )

    def test_snitch_empty_required_input_dimension_falls_back(self):
        for dimensions in ((0, 3, 8), (2, 0, 8)):
            with self.subTest(dimensions=dimensions):
                result = self.resolve("snitch", "Gemm", *dimensions)
                self.assertEqual(result["implementation_id"], SNITCH_GENERIC)
                self.assertEqual(result["fallback_reason"], "empty_input_dimension")

    def test_spatz_empty_or_transposed_gemm_falls_back(self):
        empty = self.resolve("spatz", "MatMul", 2, 3, 0)
        self.assertEqual(empty["implementation_id"], SPATZ_GENERIC)
        self.assertEqual(empty["fallback_reason"], "empty_dimension")

        for attributes in ({"transA": 1}, {"transB": 1}):
            with self.subTest(attributes=attributes):
                result = self.resolve("spatz", "Gemm", 2, 3, 8, **attributes)
                self.assertEqual(result["implementation_id"], SPATZ_GENERIC)
                self.assertEqual(result["fallback_reason"], "transposed_operand")

        tuned = self.resolve("spatz", "Gemm", 2, 3, 8)
        self.assertEqual(tuned["implementation_id"], SPATZ_TUNED)

    def test_unknown_dimensions_and_attributes_remain_unknown(self):
        missing_dimension = self.resolve("spatz", "Gemm", None, 3, 8)
        self.assertIsNone(missing_dimension["implementation_id"])
        self.assertIsNotNone(missing_dimension["unknown_reason"])

        missing_attribute = self.resolve(
            "spatz", "Gemm", 2, 3, 8, transA="dynamic"
        )
        self.assertIsNone(missing_attribute["implementation_id"])
        self.assertIsNotNone(missing_attribute["unknown_reason"])

    def test_interpretation_conditions_stay_grounded_in_current_sources(self):
        snitch = (ROOT / "runtime/snitch/kernels/gemm_fp32_ssr.c").read_text()
        spatz = (ROOT / "runtime/spatz/kernels/gemm_fp32_rvv.c").read_text()
        build = (ROOT / "pipeline/build_mesh.py").read_text()

        self.assertIn("M == 0 || N == 0 || O < UNROLL", snitch)
        self.assertIn("for (; j < O; ++j)", snitch)
        self.assertIn("M == 0 || N == 0 || O == 0 || transA || transB", spatz)
        self.assertIn("kernel_overrides", build)
        self.assertIn("f\"-D{sym}={sym}_generic\"", build)


if __name__ == "__main__":
    unittest.main()
