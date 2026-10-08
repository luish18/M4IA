import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.experiment.fingerprint import canonical_json_bytes, file_set_digest, fingerprint  # noqa: E402


class FingerprintTests(unittest.TestCase):
    def test_dict_order_does_not_change_fingerprint(self):
        self.assertEqual(
            fingerprint({"hardware": {"lanes": 4, "vlen": 512}, "host": "cva6"}),
            fingerprint({"host": "cva6", "hardware": {"vlen": 512, "lanes": 4}}),
        )

    def test_sequence_order_does_change_fingerprint(self):
        self.assertNotEqual(fingerprint([1, 2, 3]), fingerprint([3, 2, 1]))

    def test_set_order_is_canonicalized(self):
        self.assertEqual(fingerprint({"x": {3, 1, 2}}), fingerprint({"x": {2, 3, 1}}))

    def test_non_finite_float_is_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                canonical_json_bytes({"value": value})

    def test_file_set_identity_tracks_only_selected_files(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first, second = Path(first), Path(second)
            for root in (first, second):
                (root / "a.txt").write_text("alpha")
                (root / "b.txt").write_text("beta")
            baseline = file_set_digest(first, ["a.txt", "b.txt"])
            self.assertEqual(baseline, file_set_digest(first, ["b.txt", "a.txt"]))
            self.assertEqual(baseline, file_set_digest(second, ["a.txt", "b.txt"]))
            (first / "ignored.txt").write_text("irrelevant")
            self.assertEqual(baseline, file_set_digest(first, ["a.txt", "b.txt"]))
            (first / "a.txt").write_text("changed")
            self.assertNotEqual(baseline, file_set_digest(first, ["a.txt", "b.txt"]))


if __name__ == "__main__":
    unittest.main()
