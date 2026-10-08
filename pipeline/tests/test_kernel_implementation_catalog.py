import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.experiment.kernel_implementation_catalog import build_catalog  # noqa: E402


class KernelImplementationCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = build_catalog(ROOT)
        cls.by_id = {row["id"]: row for row in cls.rows}

    def test_stable_current_implementation_ids(self):
        self.assertEqual(
            [row["id"] for row in self.rows],
            [
                "cva6.fp32.matmul_gemm.deeploy_generic",
                "snitch.fp32.matmul_gemm.ssr_frep",
                "snitch.fp32.matmul_gemm.deeploy_generic_fallback",
                "spatz.fp32.matmul_gemm.rvv_tuned",
                "spatz.fp32.matmul_gemm.deeploy_generic_fallback",
            ],
        )

    def test_optimized_sources_symbols_fallbacks_and_unknowns(self):
        for implementation_id in (
            "snitch.fp32.matmul_gemm.ssr_frep",
            "spatz.fp32.matmul_gemm.rvv_tuned",
        ):
            row = self.by_id[implementation_id]
            source = ROOT / row["source_file"]
            self.assertTrue(source.is_file())
            for symbol in row["symbols"]:
                self.assertIn(f"{symbol}(", source.read_text())
            self.assertIsNotNone(row["fallback_id"])
        generic = self.by_id["cva6.fp32.matmul_gemm.deeploy_generic"]
        self.assertIsNone(generic["source_file"])
        self.assertIsNone(generic["compatibility_conditions"])

    def test_catalog_is_deterministic_and_refuses_source_drift(self):
        self.assertEqual(self.rows, build_catalog(ROOT))
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "source is missing"):
                build_catalog(Path(temporary))


if __name__ == "__main__":
    unittest.main()
