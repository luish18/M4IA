import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline" / "sweep"))

import design as design_mod  # noqa: E402
from pipeline.experiment.artifact_identity import (  # noqa: E402
    build_machine_context,
    build_run_input_context,
)
from pipeline.experiment.calibration_artifact import write_metadata  # noqa: E402
from pipeline.experiment.fingerprint import fingerprint  # noqa: E402
from pipeline.experiment.memory_catalog import resolve_memory  # noqa: E402
from pipeline.sweep import run as sweep_run  # noqa: E402


class SweepArtifactTests(unittest.TestCase):
    def resolved(self, *, memory="fixed", host="cva6"):
        numeric = design_mod.resolve({})
        value = {
            "schema_version": 2,
            "platform": "m4ia_current",
            "hardware": {
                "design": numeric,
                "design_slug": design_mod.slug({}),
                "build_key": design_mod.build_key({}),
                "fingerprint": fingerprint(numeric),
            },
            "memory": resolve_memory(ROOT, memory),
            "host_profile": host,
            "simulator": {
                "kind": "gvsoc",
                "target": "hetero_ara" if host == "ara" else "hetero_soc",
            },
            "workload": {"path": "fixture", "workload_fingerprint": "sha256:workload"},
            "mapping": {"strategy": "measured_rate_greedy", "pin": None},
            "execution": {"frontend": None, "serial": False, "power": False},
        }
        value["resolved_fingerprint"] = fingerprint(value)
        return value

    def calibration_context(self):
        identity = {"kind": "m4ia.calibration", "fixture": True}
        return {
            "input_identity": identity,
            "input_fingerprint": fingerprint(identity),
            "provenance": {},
        }

    def test_prepare_replaces_stale_mesh_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source-mesh"
            destination = root / "machine"
            source.mkdir()
            (source / "current.c").write_text("first\n")
            with mock.patch.object(sweep_run, "MESH", source), mock.patch.object(
                sweep_run, "sh", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")
            ):
                mesh = sweep_run.prepare({}, destination, {})
                (mesh / "stale.c").write_text("stale\n")
                (source / "current.c").write_text("second\n")
                sweep_run.prepare({}, destination, {})
            self.assertEqual((mesh / "current.c").read_text(), "second\n")
            self.assertFalse((mesh / "stale.c").exists())

    def test_calibrate_reuses_only_matching_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            design_dir = Path(tmp)
            rates = design_dir / "rates.json"
            log = design_dir / "calib" / "run" / "calib.log"
            metadata = design_dir / "calibration.json"
            log.parent.mkdir(parents=True)
            rates.write_text("{}\n")
            log.write_text("calibration\n")
            context = self.calibration_context()
            write_metadata(metadata, context, rates, log)
            with mock.patch.object(sweep_run, "sh") as command:
                found, returned, reason = sweep_run.calibrate(
                    design_dir, design_dir / "mesh", {"PATH": os.environ.get("PATH", "")},
                    "cva6", self.resolved(), context=context,
                )
            self.assertEqual(found, rates)
            self.assertEqual(returned, context)
            self.assertEqual(reason, "hit")
            command.assert_not_called()

    def _recalibrate(self, design_dir, context):
        def command(args, **kwargs):
            args = [str(arg) for arg in args]
            if "-o" in args:
                Path(args[args.index("-o") + 1]).write_text('{"rates": true}\n')
            return SimpleNamespace(returncode=0, stdout="calibration output\n", stderr="")

        with mock.patch.object(sweep_run, "sh", side_effect=command) as invoked:
            result = sweep_run.calibrate(
                design_dir, design_dir / "mesh", {"PATH": os.environ.get("PATH", "")},
                "cva6", self.resolved(), context=context,
            )
        return result, invoked.call_count

    def test_legacy_rates_without_metadata_recalibrate(self):
        with tempfile.TemporaryDirectory() as tmp:
            design_dir = Path(tmp)
            (design_dir / "rates.json").write_text('{"legacy": true}\n')
            (design_dir / "calib").mkdir()
            (design_dir / "calib" / "stale.o").write_text("old\n")
            (rates, _, reason), calls = self._recalibrate(
                design_dir, self.calibration_context(),
            )
            self.assertEqual(reason, "metadata-missing")
            self.assertEqual(calls, 3)
            self.assertFalse((design_dir / "calib" / "stale.o").exists())
            self.assertTrue(rates.is_file())
            self.assertTrue((design_dir / "calibration.json").is_file())

    def test_modified_rates_recalibrate(self):
        with tempfile.TemporaryDirectory() as tmp:
            design_dir = Path(tmp)
            rates = design_dir / "rates.json"
            log = design_dir / "calib" / "run" / "calib.log"
            metadata = design_dir / "calibration.json"
            log.parent.mkdir(parents=True)
            rates.write_text("{}\n")
            log.write_text("old\n")
            context = self.calibration_context()
            write_metadata(metadata, context, rates, log)
            rates.write_text('{"modified": true}\n')
            (_, _, reason), calls = self._recalibrate(design_dir, context)
            self.assertEqual(reason, "rates-digest-mismatch")
            self.assertEqual(calls, 3)

    def test_calibration_input_mismatch_recalibrates(self):
        with tempfile.TemporaryDirectory() as tmp:
            design_dir = Path(tmp)
            rates = design_dir / "rates.json"
            log = design_dir / "calib" / "run" / "calib.log"
            metadata = design_dir / "calibration.json"
            log.parent.mkdir(parents=True)
            rates.write_text("{}\n")
            log.write_text("old\n")
            old_context = self.calibration_context()
            write_metadata(metadata, old_context, rates, log)
            new_identity = {"kind": "m4ia.calibration", "fixture": "changed"}
            new_context = {
                "input_identity": new_identity,
                "input_fingerprint": fingerprint(new_identity),
                "provenance": {},
            }
            (_, _, reason), calls = self._recalibrate(design_dir, new_context)
            self.assertEqual(reason, "input-mismatch")
            self.assertEqual(calls, 3)

    def test_failed_calibration_simulation_does_not_write_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            design_dir = Path(tmp)
            responses = iter((
                SimpleNamespace(returncode=0, stdout="", stderr=""),
                SimpleNamespace(returncode=1, stdout="partial", stderr="failed"),
            ))
            with mock.patch.object(sweep_run, "sh", side_effect=lambda *a, **k: next(responses)):
                with self.assertRaisesRegex(RuntimeError, "simulation failed"):
                    sweep_run.calibrate(
                        design_dir, design_dir / "mesh",
                        {"PATH": os.environ.get("PATH", "")}, "cva6", self.resolved(),
                        context=self.calibration_context(),
                    )
            self.assertFalse((design_dir / "calibration.json").exists())

    def test_calibration_host_reaches_build_target_and_rate_parser(self):
        for host, target in (("cva6", "hetero_soc"), ("ara", "hetero_ara")):
            with self.subTest(host=host), tempfile.TemporaryDirectory() as tmp:
                design_dir = Path(tmp)
                commands = []

                def command(args, **kwargs):
                    args = [str(arg) for arg in args]
                    commands.append(args)
                    if "-o" in args:
                        Path(args[args.index("-o") + 1]).write_text("{}\n")
                    return SimpleNamespace(returncode=0, stdout="calibration\n", stderr="")

                with mock.patch.object(sweep_run, "sh", side_effect=command):
                    sweep_run.calibrate(
                        design_dir, design_dir / "mesh",
                        {"PATH": os.environ.get("PATH", "")},
                        host, self.resolved(host=host),
                        context=self.calibration_context(),
                    )

                build, simulate, parse = commands
                self.assertEqual(build[build.index("--host") + 1], host)
                self.assertIn(f"--target={target}", simulate)
                self.assertEqual(parse[parse.index("--host") + 1], host)

    def test_unknown_calibration_host_is_rejected_before_commands(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            sweep_run, "sh",
        ) as command:
            with self.assertRaisesRegex(RuntimeError, "unknown calibration host"):
                sweep_run.calibrate(
                    Path(tmp), Path(tmp) / "mesh",
                    {"PATH": os.environ.get("PATH", "")},
                    "unknown", self.resolved(), context=self.calibration_context(),
                )
        command.assert_not_called()

    def test_resolve_cell_passes_requested_memory_and_effective_kws_frontend(self):
        captured = []

        class Resolved:
            def __init__(self, request):
                self.request = request

            def to_dict(self):
                value = self_outer.resolved(memory=self.request.dram or "fixed")
                value["execution"] = {
                    "frontend": self.request.frontend,
                    "serial": self.request.serial,
                    "power": self.request.power,
                }
                return value

        self_outer = self

        def resolve(request, **kwargs):
            captured.append(request)
            return Resolved(request)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            sweep_run, "resolve_workload_path", return_value=Path(tmp)
        ), mock.patch.object(
            sweep_run, "detect_app", return_value=("kws", {"header": "kws_data.h"})
        ), mock.patch.object(sweep_run, "current_host_profile_catalog", return_value={}), \
                mock.patch.object(sweep_run, "resolve_experiment", side_effect=resolve):
            requested, resolved, effective = sweep_run.resolve_cell_experiment(
                {}, "fixture", "cva6", False, "16", dram="lpddr5",
            )
        self.assertEqual(captured[0].dram, "lpddr5")
        self.assertEqual(resolved["memory"]["kind"], "lpddr5")
        self.assertEqual(requested["dram"], "lpddr5")
        self.assertIsNone(requested["frontend"])
        self.assertEqual(effective["frontend"], "snitch")
        self.assertEqual(captured[0].frontend, "snitch")

    def _run_cell_dependencies(self, resolved, out_dir, *, runner_result=None):
        requested = {
            "platform": "m4ia_current", "model": "fixture", "dram": None,
            "images": 16,
        }
        effective = {
            "images": 16, "application": None, "frontend": None,
            "serial": False, "power": False, "pin": None,
        }
        calibration = self.calibration_context()
        run_provenance = {
            "source_set": {"digest": "sha256:run-source", "manifest": []},
            "m4ia": {"available": False},
        }

        def command(args, **kwargs):
            args = [str(arg) for arg in args]
            command.calls.append(args)
            if runner_result is not None and "--out" in args:
                path = Path(args[args.index("--out") + 1])
                path.write_text(json.dumps(runner_result) + "\n")
            return SimpleNamespace(returncode=0 if runner_result is not None else 1,
                                   stdout="", stderr="new run failed")

        command.calls = []

        return requested, effective, calibration, run_provenance, command

    def test_failed_new_run_cannot_consume_stale_result_or_manifest(self):
        resolved = self.resolved()
        requested, effective, calibration, provenance, command = self._run_cell_dependencies(
            resolved, None,
        )
        run_input = build_run_input_context(
            resolved,
            calibration_input_fingerprint=calibration["input_fingerprint"],
            run_source_set_digest=provenance["source_set"]["digest"],
            execution_controls=effective,
            workload_label="fixture",
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cell = out / "cells" / run_input["artifact_key"]
            cell.mkdir(parents=True)
            stale_result = cell / "result.json"
            stale_manifest = cell / "manifest.json"
            stale_result.write_text('{"old": true}\n')
            stale_manifest.write_text('{"old": true}\n')
            with mock.patch.object(
                sweep_run, "resolve_cell_experiment",
                return_value=(requested, resolved, effective),
            ), mock.patch.object(
                sweep_run, "prepare", return_value=out / "mesh",
            ), mock.patch.object(
                sweep_run, "calibrate",
                return_value=(out / "rates.json", calibration, "hit"),
            ), mock.patch.object(
                sweep_run, "capture_run_provenance", return_value=provenance,
            ), mock.patch.object(sweep_run, "sh", side_effect=command), mock.patch.object(
                sweep_run, "write_run_manifest",
            ) as writer:
                row = sweep_run.run_cell({}, "fixture", out, "cva6", False, 16)
            self.assertEqual(row["status"], "failed")
            self.assertFalse(stale_result.exists())
            self.assertFalse(stale_manifest.exists())
            writer.assert_not_called()

    def test_successful_row_preserves_legacy_fields_and_adds_identity(self):
        resolved = self.resolved()
        result_document = {
            "op": "fixture",
            "mapping": {"dram": "fixed", "nodes": []},
            "result": {
                "status": "ok", "cycles": 100,
                "caches": [{"dynamic_pj": 2.5}], "nodes": [],
            },
        }
        requested, effective, calibration, provenance, command = self._run_cell_dependencies(
            resolved, None, runner_result=result_document,
        )

        def manifest(path, **kwargs):
            Path(path).write_text('{"kind": "m4ia.run"}\n')
            return {"run_fingerprint": "sha256:completed"}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with mock.patch.object(
                sweep_run, "resolve_cell_experiment",
                return_value=(requested, resolved, effective),
            ), mock.patch.object(
                sweep_run, "prepare", return_value=out / "mesh",
            ), mock.patch.object(
                sweep_run, "calibrate",
                return_value=(out / "rates.json", calibration, "hit"),
            ), mock.patch.object(
                sweep_run, "capture_run_provenance", return_value=provenance,
            ), mock.patch.object(sweep_run, "sh", side_effect=command), mock.patch.object(
                sweep_run, "write_run_manifest", side_effect=manifest,
            ), mock.patch.object(
                sweep_run.area_model, "breakdown", return_value={"host": 1.0},
            ), mock.patch.object(
                sweep_run.area_model, "COEFFS", {"host": {"sourced": False}},
            ):
                row = sweep_run.run_cell({}, "fixture", out, "cva6", False, 16)
                cell_result = out / "cells" / row["artifact_key"] / "result.json"
                outer = json.loads(cell_result.read_text())
        for field in ("design_slug", "design", "model", "host", "status",
                      "cycles", "area_au", "cache_dynamic_pj"):
            self.assertIn(field, row)
        for field in ("dram_kind", "artifact_key", "run_input_fingerprint",
                      "run_fingerprint", "manifest"):
            self.assertIn(field, row)
        self.assertEqual(row["dram_kind"], "fixed")
        self.assertEqual(set(outer), {"op", "mapping", "result"})
        runner_command = next(args for args in command.calls if "--out" in args)
        self.assertEqual(runner_command[runner_command.index("--tag") + 1], row["artifact_key"])


if __name__ == "__main__":
    unittest.main()
