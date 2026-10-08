import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ModuleNotFoundError:
    np = onnx = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.experiment.workload import inspect_workload, workload_to_legacy_inspect  # noqa: E402


def write_matmul(path: Path, dynamic: bool = False) -> Path:
    a_shape = [None, 3] if dynamic else [2, 3]
    y_shape = [None, 4] if dynamic else [2, 4]
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, a_shape)
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, y_shape)
    weights = np.array([[0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]], dtype=np.float32)
    b = numpy_helper.from_array(weights, name="B")
    node = helper.make_node("MatMul", ["A", "B"], ["Y"])
    graph = helper.make_graph([node], "g", [a], [y], [b])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)
    return path


@unittest.skipIf(onnx is None, "environment limitation: onnx and/or numpy are unavailable")
class WorkloadSpecTests(unittest.TestCase):
    def test_static_descriptors_and_legacy_surface(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = inspect_workload(write_matmul(Path(temporary) / "network.onnx"), application="demo")
        descriptors = spec.descriptors
        self.assertEqual(descriptors["estimated_macs"]["value"], 24)
        self.assertEqual(descriptors["static_tensor_bytes"]["known_bytes"], 104)
        self.assertEqual(descriptors["working_set"]["max_static_node_bytes"], 104)
        self.assertEqual(workload_to_legacy_inspect(spec)["app"], "demo")

    def test_dynamic_shape_remains_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = inspect_workload(write_matmul(Path(temporary) / "network.onnx", dynamic=True))
        descriptors = spec.descriptors
        self.assertFalse(descriptors["static_tensor_bytes"]["complete"])
        self.assertEqual(descriptors["working_set"]["unknown_nodes"], [0])
        self.assertIsNone(descriptors["working_set"]["max_static_node_bytes"])
        self.assertIsNone(descriptors["theoretical_arithmetic_intensity"]["mac_per_byte"])

    def test_package_fingerprint_tracks_causal_fixtures_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_matmul(root / "network.onnx")
            (root / "inputs.npz").write_bytes(b"inputs-one")
            (root / "outputs.npz").write_bytes(b"outputs-one")
            (root / "README.txt").write_text("one")
            baseline = inspect_workload(root)
            (root / "README.txt").write_text("two")
            unchanged = inspect_workload(root)
            (root / "inputs.npz").write_bytes(b"inputs-two")
            changed = inspect_workload(root)
        self.assertEqual(baseline.workload_fingerprint, unchanged.workload_fingerprint)
        self.assertNotEqual(baseline.workload_fingerprint, changed.workload_fingerprint)


if __name__ == "__main__":
    unittest.main()
