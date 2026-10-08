import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.experiment.artifact_identity import build_run_input_context  # noqa: E402
from pipeline.experiment.calibration_artifact import write_metadata  # noqa: E402
from pipeline.experiment.fingerprint import file_digest, fingerprint  # noqa: E402
from pipeline.experiment.run_manifest import (  # noqa: E402
    KIND,
    build_run_manifest,
    write_run_manifest,
)


class RunManifestTests(unittest.TestCase):
    def resolved(self):
        payload = {
            "schema_version": 2,
            "platform": "m4ia_current",
            "hardware": {
                "design": {"HOST_NB_LANES": 4},
                "design_slug": "baseline",
                "build_key": "vlen-128",
                "fingerprint": "sha256:numeric-design",
            },
            "memory": {
                "kind": "fixed",
                "fingerprint": "sha256:fixed-memory",
                "preset_fingerprint": None,
                "overrides": {},
            },
            "host_profile": "cva6",
        }
        payload["resolved_fingerprint"] = fingerprint(payload)
        return payload

    def run_input(self):
        return build_run_input_context(
            self.resolved(),
            calibration_input_fingerprint="sha256:calibration-input",
            run_source_set_digest="sha256:run-sources",
            execution_controls={"images": 16, "frontend": None, "power": False},
            workload_label="mnist",
        )

    def make_artifacts(self, root, result=None):
        rates = root / "machine" / "rates.json"
        log = root / "machine" / "calib" / "run" / "calib.log"
        calibration = root / "machine" / "calibration.json"
        rates.parent.mkdir(parents=True)
        log.parent.mkdir(parents=True)
        rates.write_text('{"cva6": {"MatMul": 1.0}}\n')
        log.write_text("calibration output\n")
        input_identity = {"kind": "m4ia.calibration", "fixture": True}
        context = {
            "input_identity": input_identity,
            "input_fingerprint": "sha256:calibration-input",
            "provenance": {"toolchain": {"binary_digest": "sha256:gcc"}},
        }
        # The CP5 writer validates that fingerprint against the identity.
        context["input_fingerprint"] = fingerprint(input_identity)
        run_input = self.run_input()
        run_input["identity"]["calibration_input_fingerprint"] = context["input_fingerprint"]
        run_input["run_input_fingerprint"] = fingerprint(run_input["identity"])
        write_metadata(calibration, context, rates, log)

        result_path = root / "cells" / "fixture" / "result.json"
        result_path.parent.mkdir(parents=True)
        result = result or {
            "op": "fixture",
            "mapping": {
                "host": "cva6",
                "dram": "fixed",
                "nodes": [{
                    "index": 0,
                    "node": "matmul",
                    "op": "MatMul",
                    "engine": "cva6",
                    "kernel_arguments": {"M": 2, "N": 2, "O": 2},
                }],
            },
            "result": {
                "status": "ok",
                "cycles": 100,
                "caches": [{"name": "l2", "read_hits": 3}],
                "nodes": [{
                    "node": 0,
                    "op": "MatMul",
                    "engine": "cva6",
                    "cycles": 80,
                    "implementation": {
                        "implementation_id": "cva6.fp32.matmul_gemm.deeploy_generic",
                        "evidence": {"type": "deterministic"},
                    },
                }],
            },
        }
        result_path.write_text(json.dumps(result, sort_keys=True) + "\n")
        provenance = {
            "source_set": {"digest": "sha256:run-sources", "manifest": []},
            "m4ia": {"available": False},
        }
        return calibration, result_path, run_input, provenance

    def inputs(self, root, result=None):
        calibration, result_path, run_input, provenance = self.make_artifacts(root, result)
        return {
            "requested": {"dram": None, "images": 16},
            "resolved": {
                "experiment": self.resolved(),
                "machine": run_input["machine"],
                "execution": {"images": 16},
            },
            "run_input": run_input,
            "artifact_key": run_input["artifact_key"],
            "calibration_path": calibration,
            "result_path": result_path,
            "run_provenance": provenance,
            "base_dir": root,
        }

    def build(self, root, result=None):
        inputs = self.inputs(root, result)
        manifest = build_run_manifest(**inputs)
        return (
            manifest,
            inputs["calibration_path"],
            inputs["result_path"],
            inputs["run_input"],
            inputs["run_provenance"],
        )

    def test_manifest_kind_and_semantic_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, *_ = self.build(Path(tmp))
        self.assertEqual(manifest["kind"], KIND)
        self.assertEqual(KIND, "m4ia.run")
        for layer in ("requested", "resolved", "generated", "actual", "measured"):
            self.assertIn(layer, manifest)
        self.assertIsNone(manifest["requested"]["dram"])
        self.assertEqual(
            manifest["resolved"]["experiment"]["memory"]["kind"], "fixed",
        )
        self.assertNotIn("k9r5", json.dumps(manifest).lower())

    def test_actual_is_completed_only_and_contains_no_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, *_ = self.build(Path(tmp))
        actual = manifest["actual"]["nodes"]
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0]["node"], 0)
        self.assertIn("implementation", actual[0])
        self.assertNotIn("cycles", actual[0])

    def test_measured_preserves_observations_without_duplicating_implementation(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, *_ = self.build(Path(tmp))
        measured = manifest["measured"]
        self.assertEqual(measured["status"], "ok")
        self.assertEqual(measured["cycles"], 100)
        self.assertIn("caches", measured)
        self.assertEqual(measured["nodes"][0]["cycles"], 80)
        self.assertNotIn("implementation", measured["nodes"][0])

    def test_partial_result_preserves_status_and_completed_only_evidence(self):
        result = {
            "op": "fixture",
            "mapping": {"dram": "fixed", "nodes": [
                {"index": 0, "node": "a", "op": "MatMul", "engine": "cva6"},
                {"index": 1, "node": "b", "op": "MatMul", "engine": "snitch"},
            ]},
            "result": {"status": "stalled", "nodes": [{
                "node": 0, "op": "MatMul", "engine": "cva6", "cycles": 10,
                "implementation": {"implementation_id": "generic"},
            }]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            manifest, *_ = self.build(Path(tmp), result)
        self.assertEqual(manifest["measured"]["status"], "stalled")
        self.assertEqual([node["node"] for node in manifest["actual"]["nodes"]], [0])

    def test_generated_actual_and_measured_each_affect_completed_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, *_ = self.build(root / "base")
            generated_result = None
            _, _, source, _, _ = self.build(root / "source")
            generated_result = json.loads(source.read_text())
            generated_result["mapping"]["nodes"][0]["engine"] = "snitch"
            generated, *_ = self.build(root / "generated", generated_result)
            actual_result = json.loads(source.read_text())
            actual_result["result"]["nodes"][0]["implementation"]["implementation_id"] = "other"
            actual, *_ = self.build(root / "actual", actual_result)
            measured_result = json.loads(source.read_text())
            measured_result["result"]["cycles"] = 101
            measured, *_ = self.build(root / "measured", measured_result)
        self.assertNotEqual(base["run_fingerprint"], generated["run_fingerprint"])
        self.assertNotEqual(base["run_fingerprint"], actual["run_fingerprint"])
        self.assertNotEqual(base["run_fingerprint"], measured["run_fingerprint"])
        self.assertEqual(base["run_input_fingerprint"], measured["run_input_fingerprint"])

    def test_artifact_paths_and_digests_are_linked(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, calibration, result, *_ = self.build(Path(tmp))
            self.assertEqual(manifest["calibration"]["metadata_digest"], file_digest(calibration))
            self.assertEqual(manifest["artifacts"]["result"]["digest"], file_digest(result))
            self.assertFalse(Path(manifest["calibration"]["path"]).is_absolute())
            self.assertFalse(Path(manifest["artifacts"]["result"]["path"]).is_absolute())

    def test_result_mutation_is_detectable(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _, result, *_ = self.build(Path(tmp))
            recorded = manifest["artifacts"]["result"]["digest"]
            result.write_text(result.read_text() + "\n")
            self.assertNotEqual(recorded, file_digest(result))

    def test_invalid_calibration_metadata_is_rejected(self):
        cases = (
            ("not-json", "cannot read calibration metadata"),
            (json.dumps({"schema_version": 1, "kind": "legacy"}), "kind"),
            (json.dumps({"schema_version": 1, "kind": "m4ia.calibration"}), "input_fingerprint"),
        )
        for text, reason in cases:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                inputs = self.inputs(root)
                inputs["calibration_path"].write_text(text)
                with self.assertRaisesRegex(RuntimeError, reason):
                    build_run_manifest(**inputs)

    def test_artifact_key_must_match_run_input_addressing(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = self.inputs(Path(tmp))
            inputs["artifact_key"] = "other-artifact"
            with self.assertRaisesRegex(RuntimeError, "artifact key"):
                build_run_manifest(**inputs)

    def test_resolved_experiment_must_match_run_input_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = self.inputs(Path(tmp))
            inputs["resolved"]["experiment"]["resolved_fingerprint"] = "sha256:other"
            with self.assertRaisesRegex(RuntimeError, "resolved experiment"):
                build_run_manifest(**inputs)

    def test_resolved_machine_must_match_run_input_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = self.inputs(Path(tmp))
            inputs["resolved"]["machine"]["machine_fingerprint"] = "sha256:other"
            with self.assertRaisesRegex(RuntimeError, "resolved machine"):
                build_run_manifest(**inputs)

    def test_resolved_machine_fingerprint_must_match_machine_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = self.inputs(Path(tmp))
            inputs["resolved"]["machine"]["identity"]["host_profile"] = "ara"
            with self.assertRaisesRegex(RuntimeError, "machine fingerprint"):
                build_run_manifest(**inputs)

    def test_calibration_must_match_run_input_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = self.inputs(Path(tmp))
            calibration = json.loads(inputs["calibration_path"].read_text())
            calibration["input_fingerprint"] = "sha256:other"
            inputs["calibration_path"].write_text(json.dumps(calibration))
            with self.assertRaisesRegex(RuntimeError, "calibration artifact"):
                build_run_manifest(**inputs)

    def test_runtime_mapping_host_must_match_resolved_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = self.inputs(Path(tmp))
            document = json.loads(inputs["result_path"].read_text())
            document["mapping"]["host"] = "ara"
            inputs["result_path"].write_text(json.dumps(document))
            with self.assertRaisesRegex(RuntimeError, "host contradicts"):
                build_run_manifest(**inputs)

    def test_runtime_mapping_memory_must_match_resolved_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = self.inputs(Path(tmp))
            document = json.loads(inputs["result_path"].read_text())
            document["mapping"]["dram"] = "lpddr5"
            inputs["result_path"].write_text(json.dumps(document))
            with self.assertRaisesRegex(RuntimeError, "memory contradicts"):
                build_run_manifest(**inputs)

    def test_no_result_does_not_create_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self.inputs(root)
            inputs["result_path"].unlink()
            manifest = root / "manifest.json"
            with self.assertRaisesRegex(RuntimeError, "result artifact"):
                write_run_manifest(manifest, **inputs)
            self.assertFalse(manifest.exists())

    def test_building_manifest_does_not_modify_result_outer_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, result, *_ = self.build(root)
            document = json.loads(result.read_text())
        self.assertEqual(set(document), {"op", "mapping", "result"})


if __name__ == "__main__":
    unittest.main()
