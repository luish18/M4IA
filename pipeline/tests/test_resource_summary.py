import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline" / "sweep"))

import design as design_mod  # noqa: E402
from pipeline.experiment.resource_summary import build_resource_summary  # noqa: E402


class ResourceSummaryTests(unittest.TestCase):
    def summary(self, overrides=None):
        return build_resource_summary(design_mod.resolve(overrides or {}))

    def test_baseline_distinguishes_modeled_compute_and_control_cores(self):
        resources = self.summary()
        self.assertEqual(resources["platform"]["total_modeled_cores"], 19)
        for name in ("snitch", "spatz"):
            cluster = resources["clusters"][name]
            self.assertEqual(cluster["modeled_cores"], 9)
            self.assertEqual(cluster["compute_cores"], 8)
            self.assertEqual(cluster["control_dma_cores"], 1)

    def test_geometry_changes_are_derived(self):
        resources = self.summary({"SPATZ_NB_LANES": 8, "TCDM_SIZE": 0x40000})
        vector = resources["clusters"]["spatz"]["spatz_vector"]
        self.assertEqual(vector["modeled_vector_lanes"], 72)
        self.assertEqual(vector["useful_vector_lanes"], 64)
        for cluster in resources["clusters"].values():
            self.assertEqual(cluster["tcdm"]["bank_count"], 32)
            self.assertEqual(cluster["tcdm"]["bank_size_bytes"], 0x2000)

    def test_summary_does_not_invent_memory_kind_or_ppa(self):
        encoded = repr(self.summary()).lower()
        for absent in ("dram", "lpddr", "hyperram", "area", "power", "peak"):
            self.assertNotIn(absent, encoded)


if __name__ == "__main__":
    unittest.main()
