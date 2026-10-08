import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline" / "sweep"))

import design as design_mod  # noqa: E402
from pipeline.experiment.calibration_artifact import (  # noqa: E402
    KIND,
    PLATFORM_ID,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    build_calibration_context,
    cache_status,
    calibration_source_files,
    write_metadata,
)
from pipeline.experiment.fingerprint import file_set_digest, file_set_manifest  # noqa: E402
from pipeline.experiment.memory_catalog import resolve_memory  # noqa: E402


class CalibrationArtifactTests(unittest.TestCase):
    def resolved(
        self,
        *,
        host="cva6",
        memory="fixed",
        dram_overrides=None,
        workload="sha256:workload-a",
        design_overrides=None,
    ):
        design = design_mod.resolve(design_overrides or {})
        target = "hetero_ara" if host == "ara" else "hetero_soc"
        return {
            "schema_version": 2,
            "resolved_fingerprint": f"resolved-for-{workload}",
            "platform": PLATFORM_ID,
            "hardware": {
                "design": design,
                "design_slug": design_mod.slug(design),
                "build_key": design_mod.build_key(design),
            },
            "memory": resolve_memory(ROOT, memory, dram_overrides),
            "host_profile": host,
            "simulator": {"kind": "gvsoc", "target": target},
            "workload": {"workload_fingerprint": workload},
            "mapping": {"strategy": "measured_rate_greedy"},
            "execution": {},
        }

    def dependency(self, name, remote=None):
        return {
            "available": True,
            "name": name,
            "repository": remote or f"https://example.invalid/{name}.git",
            "revision": f"{name}-revision",
            "dirty": False,
            "diff_digest": None,
            "untracked_digest": None,
        }

    def context(self, resolved=None, dependencies=None):
        dependencies = dependencies or {
            "deeploy": self.dependency("deeploy"),
            "gvsoc": self.dependency("gvsoc"),
            "gvsoc_core": self.dependency("gvsoc-core"),
        }
        return build_calibration_context(
            resolved or self.resolved(),
            source_set_digest="sha256:calibration-sources",
            dependencies=dependencies,
            patches=[
                {"path": "deps/patches/b.patch", "digest": "sha256:patch-b"},
                {"path": "deps/patches/a.patch", "digest": "sha256:patch-a"},
            ],
            toolchain={
                "executable": "/different/paths/do/not/matter/gcc",
                "version": "gcc fixture 15.2",
                "binary_digest": "sha256:gcc",
            },
            simulator_binary={
                "executable": "gvsoc",
                "binary_digest": "sha256:gvsoc-binary",
            },
            provenance={"m4ia": {"revision": "fixture"}},
        )

    def test_identical_inputs_have_identical_fingerprint(self):
        self.assertEqual(
            self.context()["input_fingerprint"],
            self.context()["input_fingerprint"],
        )

    def test_numeric_design_change_changes_fingerprint(self):
        baseline = self.context(self.resolved())
        changed = self.context(self.resolved(
            design_overrides={"SPATZ_NB_LANES": 8},
        ))
        self.assertNotEqual(baseline["input_fingerprint"], changed["input_fingerprint"])

    def test_causal_calibration_source_change_changes_fingerprint(self):
        baseline = self.context()
        changed = build_calibration_context(
            self.resolved(),
            source_set_digest="sha256:changed-calibration-sources",
            dependencies={
                "deeploy": self.dependency("deeploy"),
                "gvsoc": self.dependency("gvsoc"),
                "gvsoc_core": self.dependency("gvsoc-core"),
            },
            patches=[
                {"path": "deps/patches/b.patch", "digest": "sha256:patch-b"},
                {"path": "deps/patches/a.patch", "digest": "sha256:patch-a"},
            ],
            toolchain={
                "version": "gcc fixture 15.2",
                "binary_digest": "sha256:gcc",
            },
            simulator_binary={"binary_digest": "sha256:gvsoc-binary"},
        )
        self.assertNotEqual(baseline["input_fingerprint"], changed["input_fingerprint"])

    def test_host_and_target_change_changes_fingerprint(self):
        cva6 = self.context(self.resolved(host="cva6"))
        ara = self.context(self.resolved(host="ara"))
        self.assertNotEqual(cva6["input_fingerprint"], ara["input_fingerprint"])

    def test_fixed_and_lpddr5_have_distinct_calibration_identity(self):
        fixed = self.context(self.resolved(memory="fixed"))
        lpddr5 = self.context(self.resolved(memory="lpddr5"))
        self.assertEqual(fixed["input_identity"]["memory"]["kind"], "fixed")
        self.assertNotEqual(fixed["input_fingerprint"], lpddr5["input_fingerprint"])

    def test_operational_dram_override_changes_calibration_identity(self):
        default = self.context(self.resolved(memory="lpddr5"))
        overridden = self.context(self.resolved(
            memory="lpddr5", dram_overrides={"ctrl_ps": 30000},
        ))
        self.assertNotEqual(default["input_fingerprint"], overridden["input_fingerprint"])

    def test_workload_only_difference_does_not_change_calibration_identity(self):
        first = self.context(self.resolved(workload="sha256:workload-a"))
        second = self.context(self.resolved(workload="sha256:workload-b"))
        self.assertEqual(first["input_fingerprint"], second["input_fingerprint"])
        self.assertNotIn("workload", first["input_identity"])
        self.assertNotIn("resolved_fingerprint", first["input_identity"])

    def test_remote_and_descriptive_paths_do_not_change_semantic_identity(self):
        upstream = {
            "deeploy": self.dependency("deeploy", "https://upstream/deeploy"),
            "gvsoc": self.dependency("gvsoc", "https://upstream/gvsoc"),
            "gvsoc_core": self.dependency("gvsoc-core", "https://upstream/core"),
        }
        mirror = {
            name: dict(snapshot, repository=f"ssh://mirror/{name}")
            for name, snapshot in upstream.items()
        }
        self.assertEqual(
            self.context(dependencies=upstream)["input_fingerprint"],
            self.context(dependencies=mirror)["input_fingerprint"],
        )

    def test_calibration_source_set_covers_current_memory_cpp_and_excludes_noise(self):
        paths = calibration_source_files(ROOT, "lpddr5")
        relative = [path.relative_to(ROOT).as_posix() for path in paths]
        self.assertEqual(relative, sorted(relative))
        self.assertIn("targets/hetero/dram.cpp", relative)
        self.assertIn("targets/hetero/dram_core.hpp", relative)
        self.assertIn("targets/hetero/timing_cache.cpp", relative)
        self.assertIn("runtime/tests/mesh_calib.c", relative)
        self.assertIn("pipeline/sweep/calibrate.py", relative)
        self.assertIn("pipeline/sweep/run.py", relative)
        # These committed defaults are overwritten in the per-design mesh
        # copy. Their generator and inputs are the causal sources.
        self.assertNotIn("runtime/mesh/hes_system.h", relative)
        self.assertNotIn("runtime/mesh/host.ld", relative)
        self.assertNotIn("README.md", relative)
        self.assertFalse(any(path.startswith("docs/") for path in relative))
        self.assertFalse(any(path.startswith("results/") for path in relative))
        self.assertFalse(any(path.startswith("work/") for path in relative))
        self.assertFalse(any(path.startswith("gui/") for path in relative))

    def test_fixed_source_set_is_explicit_without_uninstantiated_dram_model(self):
        relative = {
            path.relative_to(ROOT).as_posix()
            for path in calibration_source_files(ROOT, "fixed")
        }
        self.assertIn("targets/hetero/timing_cache.cpp", relative)
        self.assertNotIn("targets/hetero/dram.cpp", relative)
        self.assertNotIn("targets/hetero/dram_core.hpp", relative)

    def test_source_manifest_and_digest_are_deterministic(self):
        paths = calibration_source_files(ROOT, "lpddr5")
        manifest = file_set_manifest(ROOT, paths)
        self.assertEqual(manifest, file_set_manifest(ROOT, reversed(paths)))
        self.assertEqual(
            file_set_digest(ROOT, paths),
            file_set_digest(ROOT, reversed(paths)),
        )

    def _artifacts(self, root):
        rates = root / "rates.json"
        log = root / "run" / "calib.log"
        metadata = root / "calibration.json"
        log.parent.mkdir()
        rates.write_text('{"RATES": {}}\n')
        log.write_text("raw calibration\n")
        return rates, log, metadata

    def test_matching_metadata_and_rates_are_a_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            rates, log, metadata = self._artifacts(Path(tmp))
            context = self.context()
            write_metadata(metadata, context, rates, log)
            self.assertEqual(
                cache_status(rates, metadata, context["input_fingerprint"]),
                (True, "hit"),
            )

    def test_cache_miss_reasons_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rates = root / "rates.json"
            metadata = root / "calibration.json"
            self.assertEqual(cache_status(rates, metadata, "sha256:x"),
                             (False, "rates-missing"))
            rates.write_text("{}\n")
            self.assertEqual(cache_status(rates, metadata, "sha256:x"),
                             (False, "metadata-missing"))

            metadata.write_text("not json")
            self.assertEqual(cache_status(rates, metadata, "sha256:x"),
                             (False, "metadata-invalid"))
            metadata.write_text("[]")
            self.assertEqual(cache_status(rates, metadata, "sha256:x"),
                             (False, "metadata-invalid"))

    def test_schema_kind_input_and_rates_digest_mismatches_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rates, log, metadata = self._artifacts(root)
            context = self.context()
            write_metadata(metadata, context, rates, log)
            original = json.loads(metadata.read_text())

            cases = (
                (dict(original, schema_version=SCHEMA_VERSION + 1), "schema-mismatch"),
                (dict(original, kind="legacy.calibration"), "kind-mismatch"),
                (dict(original, input_fingerprint="sha256:other"),
                 "input-identity-mismatch"),
            )
            for blob, reason in cases:
                with self.subTest(reason=reason):
                    metadata.write_text(json.dumps(blob))
                    self.assertEqual(
                        cache_status(rates, metadata, context["input_fingerprint"]),
                        (False, reason),
                    )

            metadata.write_text(json.dumps(original))
            rates.write_text('{"RATES": {"changed": true}}\n')
            self.assertEqual(
                cache_status(rates, metadata, context["input_fingerprint"]),
                (False, "rates-digest-mismatch"),
            )

    def test_protocol_and_internal_identity_are_validated_before_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rates, log, metadata = self._artifacts(root)
            context = self.context()
            write_metadata(metadata, context, rates, log)
            original = json.loads(metadata.read_text())

            cases = (
                (dict(original, protocol="legacy_protocol"), "protocol-mismatch"),
                ({key: value for key, value in original.items()
                  if key != "input_identity"}, "input-identity-invalid"),
                (dict(original, input_identity=[]), "input-identity-invalid"),
                (dict(original, input_identity={"tampered": True}),
                 "input-identity-mismatch"),
            )
            for blob, reason in cases:
                with self.subTest(reason=reason):
                    metadata.write_text(json.dumps(blob))
                    self.assertEqual(
                        cache_status(rates, metadata, context["input_fingerprint"]),
                        (False, reason),
                    )

            metadata.write_text(json.dumps(original))
            self.assertEqual(
                cache_status(rates, metadata, "sha256:different-expected-input"),
                (False, "input-mismatch"),
            )
            self.assertEqual(
                cache_status(rates, metadata, context["input_fingerprint"]),
                (True, "hit"),
            )

    def test_metadata_records_identity_provenance_and_artifact_digests(self):
        with tempfile.TemporaryDirectory() as tmp:
            rates, log, metadata = self._artifacts(Path(tmp))
            context = self.context()
            write_metadata(metadata, context, rates, log)
            blob = json.loads(metadata.read_text())
            self.assertEqual(blob["kind"], KIND)
            self.assertEqual(blob["protocol"], PROTOCOL_ID)
            self.assertEqual(blob["input_fingerprint"], context["input_fingerprint"])
            self.assertEqual(blob["input_identity"], context["input_identity"])
            self.assertTrue(blob["artifacts"]["rates"]["digest"].startswith("sha256:"))
            self.assertEqual(blob["artifacts"]["raw_log"]["path"], "run/calib.log")
            self.assertNotIn("k9r5", json.dumps(blob).lower())

    def test_writer_does_not_create_sidecar_for_incomplete_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rates = root / "rates.json"
            metadata = root / "calibration.json"
            rates.write_text("{}\n")
            with self.assertRaises(FileNotFoundError):
                write_metadata(metadata, self.context(), rates, root / "missing.log")
            self.assertFalse(metadata.exists())

    def test_writer_rejects_internally_inconsistent_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            rates, log, metadata = self._artifacts(Path(tmp))
            context = dict(self.context(), input_fingerprint="sha256:not-the-input")
            with self.assertRaisesRegex(ValueError, "does not match"):
                write_metadata(metadata, context, rates, log)
            self.assertFalse(metadata.exists())


if __name__ == "__main__":
    unittest.main()
