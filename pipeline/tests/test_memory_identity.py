import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline" / "sweep"))

import design as design_mod  # noqa: E402
from pipeline.experiment.discovery import current_experiment_schema  # noqa: E402
from pipeline.experiment.memory_catalog import (  # noqa: E402
    build_catalog,
    current_memory_catalog,
    load_operational_memory_source,
    resolve_memory,
)
from pipeline.experiment.resolve import resolve_experiment  # noqa: E402
from pipeline.experiment.schema import ExperimentRequest, WorkloadSpec  # noqa: E402

HOST_PROFILES = {
    "cva6": {"target": "hetero_soc", "logical_engine": "cva6", "vector": False},
    "ara": {"target": "hetero_ara", "logical_engine": "cva6", "vector": True},
}


def fake_workload(path, application=None):
    return WorkloadSpec(
        path=str(path),
        content_digest="sha256:model",
        workload_fingerprint="sha256:workload",
        artifact_digests=(),
        application=application,
        opset=(),
        inputs=(),
        outputs=(),
        node_count=0,
        op_counts=(),
        initializer_count=0,
    )


def resolved(request):
    return resolve_experiment(
        request,
        design_api=design_mod,
        host_profiles=HOST_PROFILES,
        workload_path="synthetic/network.onnx",
        repository_root=ROOT,
        workload_inspector=fake_workload,
    )


class MemoryIdentityTests(unittest.TestCase):
    def test_omitted_and_explicit_fixed_have_one_canonical_resolved_identity(self):
        omitted = resolved(ExperimentRequest(workload="synthetic/network.onnx"))
        explicit = resolved(ExperimentRequest(workload="synthetic/network.onnx", dram="fixed"))
        self.assertEqual(omitted.memory["kind"], "fixed")
        self.assertEqual(omitted.memory, explicit.memory)
        self.assertEqual(omitted.resolved_fingerprint, explicit.resolved_fingerprint)
        self.assertIsNone(ExperimentRequest().dram)

    def test_real_memory_kinds_resolve_distinctly(self):
        kinds = ("lpddr4", "lpddr4x", "lpddr5", "hyperram")
        experiments = {kind: resolved(ExperimentRequest(workload="w", dram=kind)) for kind in kinds}
        self.assertEqual({item.memory["kind"] for item in experiments.values()}, set(kinds))
        self.assertEqual(len({item.memory["fingerprint"] for item in experiments.values()}), len(kinds))
        self.assertEqual(len({item.resolved_fingerprint for item in experiments.values()}), len(kinds))

    def test_unsupported_memory_fails_instead_of_falling_back(self):
        with self.assertRaisesRegex(ValueError, "unknown main memory"):
            resolved(ExperimentRequest(workload="w", dram="ddr42"))

    def test_numeric_design_identity_is_unchanged_and_memory_independent(self):
        base = resolved(ExperimentRequest(workload="w", dram="fixed"))
        other_memory = resolved(ExperimentRequest(workload="w", dram="lpddr5"))
        changed_design = resolved(ExperimentRequest(
            workload="w",
            dram="fixed",
            design_overrides=(("SPATZ_NB_LANES", 8),),
        ))
        self.assertNotIn("DRAM_KIND", design_mod.DEFAULTS)
        self.assertEqual(base.hardware, other_memory.hardware)
        self.assertEqual(base.hardware["design_slug"], "baseline")
        self.assertEqual(other_memory.hardware["design_slug"], "baseline")
        self.assertNotEqual(base.hardware["fingerprint"], changed_design.hardware["fingerprint"])
        self.assertNotEqual(base.hardware["design_slug"], changed_design.hardware["design_slug"])

    def test_resource_summary_is_unchanged_by_memory_only_selection(self):
        summaries = [resolved(ExperimentRequest(workload="w", dram=kind)).resources for kind in (
            "fixed", "lpddr4", "lpddr5", "hyperram",
        )]
        self.assertTrue(all(summary == summaries[0] for summary in summaries[1:]))

    def test_memory_discovery_matches_current_operational_kinds(self):
        source = load_operational_memory_source(ROOT)
        rows = current_memory_catalog(ROOT)
        self.assertEqual(tuple(row["name"] for row in rows), source.KINDS)
        schema = current_experiment_schema(ROOT)
        self.assertEqual(tuple(row["name"] for row in schema["main_memories"]), source.KINDS)
        self.assertEqual(source.KINDS, ("fixed", "lpddr4", "lpddr4x", "lpddr5", "hyperram"))
        self.assertEqual(schema["presets"], ["m4ia_current"])

    def test_memory_metadata_has_accurate_ownership_and_no_copied_rate_labels(self):
        source = load_operational_memory_source(ROOT)
        rows = {row["name"]: row for row in current_memory_catalog(ROOT)}
        for name, row in rows.items():
            sources = {item["path"]: item for item in row["operational_sources"]}
            self.assertEqual(
                set(sources),
                {"pipeline/common.py", "targets/hetero/memsys.py", "targets/hetero/dram_presets.py"},
            )
            self.assertIn("DRAM_KIND", sources["targets/hetero/memsys.py"]["symbols"])
            display = f'{row["label"]} {row["description"]}'
            self.assertNotIn("MHz", display)
            self.assertNotIn("x8", display)
            self.assertNotIn("x16", display)
            if name == "fixed":
                self.assertEqual(row["model"], "fixed_latency")
                self.assertEqual(sources["targets/hetero/dram_presets.py"]["symbols"], ["KINDS"])
            else:
                self.assertEqual(row["model"], source.PRESETS[name]["kind"])
                self.assertIn("PRESETS", sources["targets/hetero/dram_presets.py"]["symbols"])
                mtps = source.PRESETS[name].get("mtps")
                if mtps is not None:
                    self.assertNotIn(str(mtps), display)

    def test_memory_catalog_refuses_operational_drift(self):
        source = load_operational_memory_source(ROOT)
        with self.assertRaisesRegex(RuntimeError, "metadata drift"):
            build_catalog((*source.KINDS, "future"), source.PRESETS, source.FIELDS)

    def test_operational_overrides_affect_identity_and_are_validated(self):
        base = resolved(ExperimentRequest(workload="w", dram="lpddr5"))
        changed = resolved(ExperimentRequest(
            workload="w",
            dram="lpddr5",
            dram_overrides=(("ctrl_ps", 30000),),
        ))
        self.assertNotEqual(base.memory["fingerprint"], changed.memory["fingerprint"])
        self.assertNotEqual(base.resolved_fingerprint, changed.resolved_fingerprint)
        with self.assertRaisesRegex(ValueError, "unknown DRAM parameter"):
            resolve_memory(ROOT, "lpddr5", {"not_a_parameter": 1})
        with self.assertRaisesRegex(ValueError, "not operational for fixed"):
            resolve_memory(ROOT, "fixed", {"ctrl_ps": 30000})

    def test_request_dict_canonicalizes_override_mappings(self):
        request = ExperimentRequest.from_dict({
            "workload": "w",
            "dram": "lpddr5",
            "design_overrides": {"SPATZ_NB_LANES": 8},
            "dram_overrides": {"mapping": "RoBaCoBg", "ctrl_ps": 30000},
        })
        self.assertEqual(request.overrides_dict(), {"SPATZ_NB_LANES": 8})
        self.assertEqual(
            request.dram_overrides_dict(),
            {"ctrl_ps": 30000, "mapping": "RoBaCoBg"},
        )


if __name__ == "__main__":
    unittest.main()
