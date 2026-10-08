import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline" / "sweep"))

import design as design_mod  # noqa: E402
from pipeline.experiment.resolve import resolve_experiment  # noqa: E402
from pipeline.experiment.schema import ExperimentRequest, WorkloadSpec  # noqa: E402

HOST_PROFILES = {
    "cva6": {"target": "hetero_soc", "logical_engine": "cva6", "vector": False},
    "ara": {"target": "hetero_ara", "logical_engine": "cva6", "vector": True},
}


def inspect_fake(path, application=None):
    return WorkloadSpec(
        path=str(path), content_digest="sha256:model", workload_fingerprint="sha256:workload",
        artifact_digests=(), application=application, opset=(), inputs=(), outputs=(),
        node_count=1, op_counts=(("MatMul", 1),), initializer_count=1,
    )


class ResolveExperimentTests(unittest.TestCase):
    def resolve(self, request):
        return resolve_experiment(
            request,
            design_api=design_mod,
            host_profiles=HOST_PROFILES,
            workload_path="synthetic/network.onnx",
            repository_root=ROOT,
            workload_inspector=inspect_fake,
        )

    def test_current_platform_resolves_current_design_defaults(self):
        resolved = self.resolve(ExperimentRequest(workload="synthetic/network.onnx"))
        self.assertEqual(resolved.platform, "m4ia_current")
        self.assertEqual(resolved.hardware["design"], design_mod.DEFAULTS)
        self.assertEqual(resolved.hardware["design_slug"], "baseline")
        self.assertEqual(resolved.memory["kind"], "fixed")
        self.assertEqual(resolved.host_profile, "cva6")
        self.assertEqual(resolved.simulator["target"], "hetero_soc")
        self.assertEqual(resolved.mapping["strategy"], "measured_rate_greedy")

    def test_existing_design_validation_remains_authoritative(self):
        with self.assertRaisesRegex(ValueError, "unknown design keys"):
            self.resolve(ExperimentRequest(workload="w", design_overrides=(("NOT_A_KNOB", 1),)))
        with self.assertRaisesRegex(ValueError, "invalid design"):
            self.resolve(ExperimentRequest(workload="w", design_overrides=(("SPATZ_NB_LANES", 3),)))

    def test_missing_workload_fails_before_inspection(self):
        inspected = False

        def must_not_inspect(path, application=None):
            nonlocal inspected
            inspected = True
            raise AssertionError("workload inspector must not run")

        with self.assertRaisesRegex(ValueError, "workload path is required"):
            resolve_experiment(
                ExperimentRequest(),
                design_api=design_mod,
                host_profiles=HOST_PROFILES,
                repository_root=ROOT,
                workload_inspector=must_not_inspect,
            )
        self.assertFalse(inspected)

    def test_unknown_host_strategy_pin_and_frontend_are_explicit(self):
        cases = (
            (ExperimentRequest(workload="w", host="future"), "unknown host profile"),
            (ExperimentRequest(workload="w", mapping_strategy="future"), "unknown mapping strategy"),
            (ExperimentRequest(workload="w", pin="ara"), "unknown pinned engine"),
            (ExperimentRequest(workload="w", frontend="future"), "frontend must be"),
        )
        for request, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.resolve(request)


if __name__ == "__main__":
    unittest.main()
