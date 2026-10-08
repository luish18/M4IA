import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.experiment.schema import ExperimentRequest  # noqa: E402


class ExperimentRequestBoundaryTests(unittest.TestCase):
    def test_valid_public_field_types_are_preserved(self):
        request = ExperimentRequest.from_dict({
            "platform": "m4ia_current",
            "workload": "ops/demo",
            "host": "cva6",
            "mapping_strategy": "measured_rate_greedy",
            "pin": None,
            "frontend": "snitch",
            "serial": False,
            "power": True,
            "dram": "lpddr5",
            "design_overrides": {"SPATZ_NB_LANES": 8},
            "dram_overrides": {"ctrl_ps": 30000, "mapping": "RoBaCoBg"},
        })
        self.assertEqual(request.platform, "m4ia_current")
        self.assertIs(request.serial, False)
        self.assertIs(request.power, True)

    def test_required_string_fields_reject_non_strings(self):
        for name in ("platform", "workload", "host", "mapping_strategy"):
            with self.subTest(field=name):
                with self.assertRaisesRegex(ValueError, rf"{name} must be a string"):
                    ExperimentRequest.from_dict({name: 3})

    def test_nullable_string_fields_reject_non_strings(self):
        for name in ("pin", "frontend", "dram"):
            with self.subTest(field=name):
                with self.assertRaisesRegex(ValueError, rf"{name} must be a string or null"):
                    ExperimentRequest.from_dict({name: 7})
            self.assertIsNone(getattr(ExperimentRequest.from_dict({name: None}), name))

    def test_boolean_fields_require_actual_booleans(self):
        for name in ("serial", "power"):
            for value in (0, 1, "false", None):
                with self.subTest(field=name, value=value):
                    with self.assertRaisesRegex(ValueError, rf"{name} must be a boolean"):
                        ExperimentRequest.from_dict({name: value})

    def test_design_overrides_require_integer_values_but_not_bool(self):
        for value in (True, False, 1.5, "8", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    ExperimentRequest.from_dict({"design_overrides": {"SPATZ_NB_LANES": value}})

    def test_dram_overrides_accept_only_flat_json_scalars(self):
        scalars = {"string": "x", "integer": 1, "float": 1.5, "boolean": True, "null": None}
        request = ExperimentRequest.from_dict({"dram_overrides": scalars})
        self.assertEqual(request.dram_overrides_dict(), scalars)
        for value in ([1], {"nested": 1}, (1,)):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must be a JSON scalar"):
                    ExperimentRequest.from_dict({"dram_overrides": {"field": value}})

    def test_request_must_be_an_object_with_string_known_fields(self):
        with self.assertRaisesRegex(ValueError, "must be an object"):
            ExperimentRequest.from_dict([])
        with self.assertRaisesRegex(ValueError, "field names must be strings"):
            ExperimentRequest.from_dict({1: "value"})
        with self.assertRaisesRegex(ValueError, "unknown ExperimentRequest fields"):
            ExperimentRequest.from_dict({"future": 1})


if __name__ == "__main__":
    unittest.main()
