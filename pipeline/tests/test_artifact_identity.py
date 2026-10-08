import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline" / "sweep"))

import design as design_mod  # noqa: E402
from pipeline.experiment.artifact_identity import (  # noqa: E402
    build_machine_context,
    build_run_input_context,
)
from pipeline.experiment.fingerprint import fingerprint  # noqa: E402
from pipeline.experiment.memory_catalog import resolve_memory  # noqa: E402


class ArtifactIdentityTests(unittest.TestCase):
    def resolved(self, *, host="cva6", memory="fixed", overrides=None,
                 workload="sha256:workload-a", design=None):
        design_overrides = design or {}
        numeric = design_mod.resolve(design_overrides)
        target = "hetero_ara" if host == "ara" else "hetero_soc"
        payload = {
            "platform": "m4ia_current",
            "hardware": {
                "design": numeric,
                "design_slug": design_mod.slug(design_overrides),
                "build_key": design_mod.build_key(design_overrides),
                "fingerprint": fingerprint(numeric),
            },
            "memory": resolve_memory(ROOT, memory, overrides),
            "host_profile": host,
            "simulator": {"kind": "gvsoc", "target": target},
            "workload": {"workload_fingerprint": workload},
            "mapping": {"strategy": "measured_rate_greedy", "pin": None},
            "execution": {"frontend": None, "serial": False, "power": False},
        }
        payload["resolved_fingerprint"] = fingerprint(payload)
        payload["schema_version"] = 2
        return payload

    def run_input(self, resolved=None, *, images=16, source="sha256:run-source"):
        return build_run_input_context(
            resolved or self.resolved(),
            calibration_input_fingerprint="sha256:calibration",
            run_source_set_digest=source,
            execution_controls={
                "images": images,
                "application": None,
                "frontend": None,
                "serial": False,
                "power": False,
                "pin": None,
            },
            workload_label="fixture",
        )

    def test_identical_inputs_have_identical_artifact_key(self):
        self.assertEqual(self.run_input(), self.run_input())

    def test_images_change_run_input_not_machine_identity(self):
        sixteen = self.run_input(images=16)
        thirty_two = self.run_input(images=32)
        self.assertEqual(
            sixteen["machine"]["machine_fingerprint"],
            thirty_two["machine"]["machine_fingerprint"],
        )
        self.assertNotEqual(
            sixteen["run_input_fingerprint"],
            thirty_two["run_input_fingerprint"],
        )
        self.assertNotEqual(sixteen["artifact_key"], thirty_two["artifact_key"])

    def test_host_changes_machine_and_run_identity(self):
        cva6 = self.run_input(self.resolved(host="cva6"))
        ara = self.run_input(self.resolved(host="ara"))
        self.assertNotEqual(cva6["machine"], ara["machine"])
        self.assertNotEqual(cva6["run_input_fingerprint"], ara["run_input_fingerprint"])

    def test_fixed_and_lpddr5_have_distinct_machine_and_cell_keys(self):
        fixed = self.run_input(self.resolved(memory="fixed"))
        lpddr5 = self.run_input(self.resolved(memory="lpddr5"))
        self.assertEqual(fixed["machine"]["identity"]["memory"]["kind"], "fixed")
        self.assertNotEqual(fixed["machine"]["machine_key"], lpddr5["machine"]["machine_key"])
        self.assertNotEqual(fixed["artifact_key"], lpddr5["artifact_key"])

    def test_memory_override_changes_machine_and_run_identity(self):
        default = self.run_input(self.resolved(memory="lpddr5"))
        changed = self.run_input(self.resolved(
            memory="lpddr5", overrides={"ctrl_ps": 30000},
        ))
        self.assertNotEqual(default["machine"], changed["machine"])
        self.assertNotEqual(default["run_input_fingerprint"], changed["run_input_fingerprint"])

    def test_workload_change_changes_run_but_not_machine_identity(self):
        first = self.run_input(self.resolved(workload="sha256:workload-a"))
        second = self.run_input(self.resolved(workload="sha256:workload-b"))
        self.assertEqual(first["machine"], second["machine"])
        self.assertNotEqual(first["run_input_fingerprint"], second["run_input_fingerprint"])

    def test_causal_run_source_change_changes_run_input(self):
        first = self.run_input(source="sha256:source-a")
        second = self.run_input(source="sha256:source-b")
        self.assertNotEqual(first["run_input_fingerprint"], second["run_input_fingerprint"])

    def test_calibration_change_changes_run_input(self):
        first = self.run_input()
        second = build_run_input_context(
            self.resolved(),
            calibration_input_fingerprint="sha256:other-calibration",
            run_source_set_digest="sha256:run-source",
            execution_controls=first["identity"]["execution_controls"],
            workload_label="fixture",
        )
        self.assertNotEqual(first["run_input_fingerprint"], second["run_input_fingerprint"])

    def test_design_slug_remains_numeric_design_only(self):
        fixed = self.resolved(memory="fixed")
        lpddr5 = self.resolved(memory="lpddr5")
        self.assertEqual(fixed["hardware"]["design_slug"], lpddr5["hardware"]["design_slug"])
        self.assertEqual(fixed["hardware"]["design_slug"], design_mod.slug({}))

    def test_post_run_evidence_is_not_an_input(self):
        identity = self.run_input()
        mutated = copy.deepcopy(identity)
        mutated["post_run_fixture"] = {"cycles": 123}
        self.assertEqual(
            identity["run_input_fingerprint"],
            mutated["run_input_fingerprint"],
        )

    def test_non_positive_or_boolean_images_are_rejected(self):
        for value in (0, -1, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "images"):
                self.run_input(images=value)


if __name__ == "__main__":
    unittest.main()
