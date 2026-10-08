import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import run_hetero  # noqa: E402
from pipeline.experiment.actual_implementation import (  # noqa: E402
    annotate_completed_nodes as actual_annotate_completed_nodes,
)
from pipeline.experiment.matmul_gemm_implementation import CVA6_GENERIC  # noqa: E402


def _mapping():
    return {
        "pin": None,
        "host": "cva6",
        "nodes": [{
            "index": 0,
            "node": "matmul_0",
            "op": "MatMul",
            "engine": "cva6",
            "kernel_arguments": {"M": 2, "N": 3, "O": 8},
        }],
    }


def _completed_result():
    return {
        "status": "ok",
        "wall_s": 1.0,
        "nodes": [{
            "node": 0,
            "op": "MatMul",
            "engine": "cva6",
            "cycles": 99,
        }],
        "per_engine_cycles": {"cva6": 99},
        "waits": [],
        "cycles": 99,
        "errors": 0,
        "outputs": 1,
        "maxdiff": 0.0,
        "offload_failures": 0,
    }


class RunHeteroEvidenceTests(unittest.TestCase):
    def _run_main(self, memory):
        generated = _mapping()
        simulated = _completed_result()
        events = []

        def simulate(*args, **kwargs):
            events.append("simulate")
            return simulated

        def annotate(mapping, result):
            self.assertEqual(events, ["simulate"])
            events.append("annotate")
            return actual_annotate_completed_nodes(mapping, result)

        def report(op_name, mapping, result, out_path=None):
            self.assertEqual(events, ["simulate", "annotate"])
            events.append("report")
            return None

        with tempfile.TemporaryDirectory() as tmp:
            test_dir = Path(tmp) / "case"
            test_dir.mkdir()
            argv = ["run_hetero.py", str(test_dir), "--out", str(Path(tmp) / "result.json")]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(run_hetero, "use_dram", return_value=memory),
                mock.patch.object(run_hetero, "resolve_op", return_value=test_dir),
                mock.patch.object(run_hetero, "generate", return_value=generated),
                mock.patch.object(run_hetero, "detect_app", return_value=(None, None)),
                mock.patch.object(run_hetero, "build_network", return_value={
                    "host": "host.elf", "snitch": "snitch.elf", "spatz": "spatz.elf",
                }),
                mock.patch.object(run_hetero, "simulate", side_effect=simulate),
                mock.patch.object(
                    run_hetero, "annotate_completed_nodes", side_effect=annotate,
                ),
                mock.patch.object(run_hetero, "report", side_effect=report) as report_mock,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as stopped:
                    run_hetero.main()

        self.assertEqual(stopped.exception.code, 0)
        self.assertEqual(events, ["simulate", "annotate", "report"])
        _, reported_mapping, reported_result = report_mock.call_args.args
        return generated, simulated, reported_mapping, reported_result

    def test_real_runner_retains_annotation_return_after_simulation_before_report(self):
        generated, simulated, mapping, result = self._run_main("fixed")
        self.assertEqual(mapping["dram"], "fixed")
        self.assertEqual(
            result["nodes"][0]["implementation"]["implementation_id"],
            CVA6_GENERIC,
        )
        self.assertNotIn("implementation", simulated["nodes"][0])
        self.assertNotIn("implementation", generated["nodes"][0])

    def test_effective_fixed_and_nonfixed_memory_are_both_explicit(self):
        for memory in ("fixed", "lpddr5"):
            with self.subTest(memory=memory):
                _, _, mapping, _ = self._run_main(memory)
                self.assertEqual(mapping["dram"], memory)

    def test_report_preserves_outer_json_contract(self):
        mapping = {"pin": None, "host": "cva6", "dram": "fixed", "nodes": []}
        result = {
            "status": "stalled",
            "wall_s": 1.0,
            "nodes": [],
            "per_engine_cycles": {},
            "waits": [],
            "log_tail": ["partial run"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            with contextlib.redirect_stdout(io.StringIO()):
                run_hetero.report("fixture", mapping, result, out_path=path)
            payload = json.loads(path.read_text())
        self.assertEqual(set(payload), {"op", "mapping", "result"})
        self.assertEqual(payload["op"], "fixture")
        self.assertEqual(payload["mapping"], mapping)
        self.assertEqual(payload["result"], result)

    def test_production_result_classifies_every_offload_failure_as_wrong_result(self):
        lines = (
            "[HES] core=cva6 cycles=10 instret=9 errors=0 total=1 "
            "maxdiff_e6=0 offload_failures=1\n",
            "[HES-MNIST] images=1 correct=1 agree_with_onnx=1 "
            "cycles_total=10 cycles_per_image=10 offload_failures=1\n",
            "[HES-KWS] clips=1 correct=1 agree_with_onnx=1 mfcc_maxdiff_e6=0 "
            "cycles_total=10 cycles_per_clip=10 frontend_engine=1 "
            "frontend_busy=5 frontend_wait=0 hidden=2 snitch_busy=5 "
            "spatz_busy=5 pipelined=1 offload_failures=1\n",
        )

        class Process:
            def __init__(self, line):
                self.stdout = iter((line,))

            def poll(self):
                return 0

            def kill(self):
                return None

            def wait(self):
                return 0

        for line in lines:
            with self.subTest(prefix=line.split("]", 1)[0]), tempfile.TemporaryDirectory() as tmp, \
                    mock.patch.object(
                        run_hetero.subprocess, "Popen", return_value=Process(line),
                    ):
                result = run_hetero.simulate(
                    {"host": "host", "snitch": "snitch", "spatz": "spatz"},
                    Path(tmp), total_nodes=0, timeout_s=5, stall_s=5, quiet=True,
                )
            self.assertGreater(result["offload_failures"], 0)
            self.assertEqual(result["status"], "wrong-result")


if __name__ == "__main__":
    unittest.main()
