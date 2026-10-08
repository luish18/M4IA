import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline import build_mesh  # noqa: E402


class BuildMeshHostTests(unittest.TestCase):
    def _selected_host_image(self, host):
        selected = []

        def build(image, sources, out_dir, **kwargs):
            selected.append(image)
            return Path(out_dir) / f"{image.name}.elf"

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            build_mesh, "build", side_effect=build,
        ):
            result = build_mesh.build_test(
                "mesh_calib",
                Path(tmp),
                cluster_src=Path("cluster_main.c"),
                host_extra=(Path("hes_host.c"),),
                host=host,
            )
        self.assertEqual(set(result), {"host", "snitch", "spatz"})
        return selected[0]

    def test_cva6_calibration_selects_scalar_host_image(self):
        self.assertIs(self._selected_host_image("cva6"), build_mesh.HOST)

    def test_ara_calibration_selects_vector_host_image(self):
        selected = self._selected_host_image("ara")
        self.assertIs(selected, build_mesh.HOST_ARA)
        self.assertIsNot(selected, build_mesh.HOST)
        self.assertIn("_v", selected.march)

    def test_unknown_calibration_host_is_rejected_before_build(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            build_mesh, "build",
        ) as build:
            with self.assertRaisesRegex(ValueError, "unknown host"):
                build_mesh.build_test("mesh_calib", Path(tmp), host="unknown")
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
