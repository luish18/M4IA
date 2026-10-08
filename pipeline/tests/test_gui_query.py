import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "pipeline"
GUI_QUERY = PIPELINE / "gui_query.py"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PIPELINE))

import gui_query  # noqa: E402
from experiment.schema import WorkloadSpec  # noqa: E402

REAL_RESOLVE_EXPERIMENT = gui_query.resolve_experiment


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


def resolve_without_onnx(request, **kwargs):
    return REAL_RESOLVE_EXPERIMENT(
        request,
        workload_inspector=fake_workload,
        **kwargs,
    )


def run_query(*args):
    return subprocess.run(
        [sys.executable, str(GUI_QUERY), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class GuiQueryIntegrationTests(unittest.TestCase):
    def resolve_request(self, payload):
        with tempfile.TemporaryDirectory() as temporary:
            request_path = Path(temporary) / "request.json"
            request_path.write_text(json.dumps(payload))
            args = SimpleNamespace(request=str(request_path))
            with mock.patch.object(
                gui_query,
                "resolve_experiment",
                side_effect=resolve_without_onnx,
            ):
                return gui_query.cmd_resolve_experiment(args)

    def test_existing_and_new_commands_remain_present(self):
        result = run_query("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in (
            "env", "ops", "inspect", "knobs", "validate",
            "experiment-schema", "resolve-experiment",
        ):
            self.assertIn(command, result.stdout)

    def test_existing_command_shapes_are_unchanged(self):
        self.assertEqual(
            set(gui_query.cmd_env(None)),
            {"root", "python", "have", "ready", "build_key"},
        )
        self.assertEqual(set(gui_query.cmd_ops(None)), {"ops", "apps"})
        self.assertEqual(
            set(gui_query.cmd_knobs(None)),
            {"defaults", "build_time", "ofat", "build_key"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            designs = Path(temporary) / "designs.json"
            designs.write_text("[{}]")
            validated = gui_query.cmd_validate(SimpleNamespace(designs=str(designs)))
        self.assertEqual(set(validated), {"designs", "build_key"})
        self.assertEqual(
            set(validated["designs"][0]),
            {"design", "slug", "resolved", "reasons", "build_key", "needs_build"},
        )

    def test_experiment_schema_is_json_and_exposes_current_choices(self):
        result = run_query("experiment-schema")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        schema = json.loads(result.stdout)
        self.assertEqual(schema["presets"], ["m4ia_current"])
        self.assertEqual(
            [row["name"] for row in schema["main_memories"]],
            ["fixed", "lpddr4", "lpddr4x", "lpddr5", "hyperram"],
        )
        for key in (
            "parameters", "engines", "host_profiles", "mapping_strategies",
            "kernel_implementations", "main_memories",
        ):
            self.assertIn(key, schema)

    def test_omitted_and_explicit_fixed_resolve_identically(self):
        workload = {"workload": "synthetic/network.onnx"}
        omitted = self.resolve_request(workload)
        explicit = self.resolve_request({**workload, "dram": "fixed"})
        self.assertEqual(omitted["memory"]["kind"], "fixed")
        self.assertEqual(omitted["memory"], explicit["memory"])
        self.assertEqual(omitted["resolved_fingerprint"], explicit["resolved_fingerprint"])

    def test_lpddr5_resolves_distinctly(self):
        workload = {"workload": "synthetic/network.onnx"}
        fixed = self.resolve_request(workload)
        lpddr5 = self.resolve_request({**workload, "dram": "lpddr5"})
        self.assertEqual(lpddr5["memory"]["kind"], "lpddr5")
        self.assertNotEqual(fixed["memory"]["fingerprint"], lpddr5["memory"]["fingerprint"])
        self.assertNotEqual(fixed["resolved_fingerprint"], lpddr5["resolved_fingerprint"])

    def test_invalid_memory_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "unknown main memory"):
            self.resolve_request({"workload": "synthetic/network.onnx", "dram": "ddr42"})

    def test_malformed_and_invalid_requests_return_json_errors_without_tracebacks(self):
        cases = (
            ("{not-json", "JSONDecodeError"),
            (json.dumps({"future": 1}), "unknown ExperimentRequest fields"),
            (json.dumps({}), "workload path is required"),
        )
        for contents, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                request_path = Path(temporary) / "request.json"
                request_path.write_text(contents)
                result = run_query("resolve-experiment", str(request_path))
                self.assertEqual(result.returncode, 1)
                error = json.loads(result.stdout)
                self.assertIn(message, error["error"])
                self.assertNotIn("Traceback", result.stderr)

    def test_new_queries_do_not_spawn_build_or_simulation_commands(self):
        with mock.patch("subprocess.run") as run, mock.patch("subprocess.Popen") as popen:
            schema = gui_query.cmd_experiment_schema(None)
            resolved = self.resolve_request({"workload": "synthetic/network.onnx"})
        self.assertEqual(schema["presets"], ["m4ia_current"])
        self.assertEqual(resolved["memory"]["kind"], "fixed")
        run.assert_not_called()
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
