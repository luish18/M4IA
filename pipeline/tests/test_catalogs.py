import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline" / "sweep"))

import design as design_mod  # noqa: E402
from pipeline.experiment.discovery import (  # noqa: E402
    current_engine_catalog,
    current_host_profile_catalog,
    current_parameter_catalog,
)
from pipeline.experiment.engine_catalog import build_catalog as build_engine_catalog  # noqa: E402
from pipeline.experiment.host_profile_catalog import build_catalog as build_host_catalog  # noqa: E402
from pipeline.experiment.mapping_strategy_catalog import META as MAPPING_META  # noqa: E402
from pipeline.experiment.mapping_strategy_catalog import build_catalog as build_mapping_catalog  # noqa: E402
from pipeline.experiment.parameter_catalog import META as PARAMETER_META  # noqa: E402
from pipeline.experiment.parameter_catalog import build_catalog as build_parameter_catalog  # noqa: E402


class CatalogTests(unittest.TestCase):
    def test_parameter_catalog_matches_numeric_design_source(self):
        self.assertEqual(set(PARAMETER_META), set(design_mod.DEFAULTS))
        rows = current_parameter_catalog(ROOT)
        by_name = {row["name"]: row for row in rows}
        self.assertEqual(set(by_name), set(design_mod.DEFAULTS))
        for name, value in design_mod.DEFAULTS.items():
            self.assertEqual(by_name[name]["default"], value)
            self.assertEqual(by_name[name]["build_time"], name in design_mod.BUILD_TIME)
        self.assertNotIn("DRAM_KIND", by_name)

    def test_parameter_catalog_refuses_drift(self):
        bad = dict(design_mod.DEFAULTS, NEW_UNDOCUMENTED_KNOB=1)
        with self.assertRaisesRegex(RuntimeError, "catalog drift"):
            build_parameter_catalog(bad, design_mod.BUILD_TIME)

    def test_engine_catalog_matches_operational_ids_and_macros(self):
        rows = current_engine_catalog(ROOT)
        self.assertEqual(
            {row["system_id"]: row["name"] for row in rows},
            {0: "cva6", 1: "snitch", 2: "spatz"},
        )
        macros = {row["name"]: row["runtime_macro"] for row in rows}
        with self.assertRaisesRegex(RuntimeError, "metadata drift"):
            build_engine_catalog({0: "cva6", 1: "snitch", 2: "spatz", 3: "future"}, macros)

    def test_host_profiles_match_build_mesh_and_refuse_drift(self):
        profiles = current_host_profile_catalog(ROOT)
        self.assertEqual(set(profiles), {"cva6", "ara"})
        self.assertEqual(profiles["cva6"]["target"], "hetero_soc")
        self.assertEqual(profiles["ara"]["target"], "hetero_ara")
        self.assertEqual(profiles["ara"]["logical_engine"], "cva6")
        with self.assertRaisesRegex(RuntimeError, "metadata drift"):
            build_host_catalog({"cva6": "hetero_soc", "ara": "hetero_ara", "future": "x"})

    def test_mapping_catalog_points_to_current_mapper_and_generated_explanations(self):
        rows = build_mapping_catalog()
        self.assertEqual([row["id"] for row in rows], ["measured_rate_greedy"])
        self.assertNotIn("pin", MAPPING_META)
        self.assertEqual(rows[0]["decision_explanation"], {
            "available": True,
            "path": "mapping.nodes[].mapping_explanation",
            "evidence": "descriptive_shadow_of_mapper_decision",
        })

        mapper = ast.parse((ROOT / "pipeline" / "hetero_platform" / "mapper.py").read_text())
        classes = {node.name for node in mapper.body if isinstance(node, ast.ClassDef)}
        functions = {node.name for node in mapper.body if isinstance(node, ast.FunctionDef)}
        meta = MAPPING_META["measured_rate_greedy"]
        self.assertIn(meta.implementation, classes)
        self.assertIn(meta.factory, functions)


if __name__ == "__main__":
    unittest.main()
