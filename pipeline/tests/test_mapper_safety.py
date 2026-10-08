import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HETERO_PLATFORM = ROOT / "pipeline" / "hetero_platform"
_NO_RATES_FILE = object()


class _AcceptArguments:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _DeploymentEngine:
    def __init__(self, name, mapping, _init_code="", include_list=()):
        self.name = name
        self.Mapping = mapping
        self.includeList = include_list


class _EngineMapper:
    def __init__(self, engine_dict):
        self.engineDict = engine_dict


def _module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _install_dependency_stubs(package_name):
    generic_mapping = {
        "Add": object(),
        "Conv": object(),
        "Gemm": object(),
        "MatMul": object(),
    }

    packages = (
        "onnx_graphsurgeon",
        "Deeploy",
        "Deeploy.CommonExtensions",
        "Deeploy.EngineExtension",
        "Deeploy.EngineExtension.OptimizationPasses",
        "Deeploy.EngineExtension.OptimizationPasses.TopologyOptimizationPasses",
        "Deeploy.Targets",
        "Deeploy.Targets.Generic",
        "hetero",
    )
    for name in packages:
        module = _module(name)
        module.__path__ = []
        sys.modules[name] = module

    sys.modules["onnx_graphsurgeon"].Node = object
    sys.modules["onnx_graphsurgeon"].Graph = object
    sys.modules["Deeploy.AbstractDataTypes"] = _module(
        "Deeploy.AbstractDataTypes", PointerClass=lambda value: ("pointer", value)
    )
    sys.modules["Deeploy.CommonExtensions.DataTypes"] = _module(
        "Deeploy.CommonExtensions.DataTypes", float32_t=object()
    )
    sys.modules["Deeploy.DeeployTypes"] = _module(
        "Deeploy.DeeployTypes",
        DeploymentEngine=_DeploymentEngine,
        NodeBinding=_AcceptArguments,
        NodeMapper=_AcceptArguments,
    )
    sys.modules["Deeploy.Targets.Generic.Bindings"] = _module(
        "Deeploy.Targets.Generic.Bindings", BasicTransformer=object()
    )
    sys.modules["Deeploy.Targets.Generic.Layers"] = _module(
        "Deeploy.Targets.Generic.Layers",
        ConvLayer=_AcceptArguments,
        GEMMLayer=_AcceptArguments,
    )
    sys.modules["Deeploy.Targets.Generic.Parsers"] = _module(
        "Deeploy.Targets.Generic.Parsers",
        GenericConv2DParser=_AcceptArguments,
        GenericGEMMParser=_AcceptArguments,
        MatMulParser=_AcceptArguments,
    )
    sys.modules["Deeploy.Targets.Generic.Platform"] = _module(
        "Deeploy.Targets.Generic.Platform", GenericMapping=generic_mapping
    )
    sys.modules["Deeploy.Targets.Generic.TypeCheckers"] = _module(
        "Deeploy.Targets.Generic.TypeCheckers",
        ConvChecker=_AcceptArguments,
        GEMMChecker=_AcceptArguments,
        MatMulChecker=_AcceptArguments,
    )
    sys.modules[
        "Deeploy.EngineExtension.OptimizationPasses.TopologyOptimizationPasses.EngineColoringPasses"
    ] = _module(
        "Deeploy.EngineExtension.OptimizationPasses.TopologyOptimizationPasses.EngineColoringPasses",
        EngineMapper=_EngineMapper,
    )

    system = types.SimpleNamespace(
        TCDM_SIZE=128 * 1024,
        MAILBOX_SIZE=256,
        CLUSTER_STACK_SIZE=1024,
        SNITCH_CLUSTER=types.SimpleNamespace(nb_core=9),
        SPATZ_CLUSTER=types.SimpleNamespace(nb_core=9),
    )
    sys.modules["hetero"].system = system

    package = _module(package_name)
    package.__path__ = [str(HETERO_PLATFORM)]
    sys.modules[package_name] = package
    sys.modules[f"{package_name}.templates"] = _module(
        f"{package_name}.templates",
        matmul=lambda _macro: object(),
        gemm=lambda _macro: object(),
        conv2d=lambda _macro: object(),
    )


def _load_file(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def loaded_mapper(rate_tables=_NO_RATES_FILE):
    previous_modules = dict(sys.modules)
    previous_path = list(sys.path)
    temporary = tempfile.TemporaryDirectory()
    package_name = f"_mapper_safety_{uuid.uuid4().hex}"
    try:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HES_RATES", None)
            if rate_tables is not _NO_RATES_FILE:
                table_path = Path(temporary.name) / "rates.json"
                table_path.write_text(json.dumps(rate_tables))
                os.environ["HES_RATES"] = str(table_path)

            _install_dependency_stubs(package_name)
            engines = _load_file(
                f"{package_name}.engines", HETERO_PLATFORM / "engines.py"
            )
            mapper = _load_file(
                f"{package_name}.mapper", HETERO_PLATFORM / "mapper.py"
            )
            yield types.SimpleNamespace(engines=engines, mapper=mapper)
    finally:
        sys.path[:] = previous_path
        for name in list(sys.modules):
            if name not in previous_modules:
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
        temporary.cleanup()


class Tensor:
    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype


class Node:
    def __init__(
        self,
        *,
        op="MatMul",
        left=(2, 3),
        right=(3, 8),
        output=(2, 8),
        attrs=None,
        name="node",
    ):
        self.name = name
        self.op = op
        self.inputs = [Tensor(list(left)), Tensor(list(right))]
        self.outputs = [Tensor(list(output))]
        self.attrs = dict(attrs or {})


def _cluster(loaded, name):
    return loaded.engines.ClusterEngine(
        name, f"HES_ENGINE_{name.upper()}", tcdm_budget=1_000_000
    )


class MapperSafetyTests(unittest.TestCase):

    def test_unknown_extent_is_not_zero_and_cluster_refuses_it(self):
        with loaded_mapper() as loaded:
            node = Node(left=(None, 3), output=(None, 8))
            self.assertIsNone(loaded.engines._tensor_bytes(node.inputs[0]))
            self.assertIsNone(loaded.engines.working_set_bytes(node))
            self.assertFalse(_cluster(loaded, "snitch").canExecute(node))

    def test_known_static_working_set_is_unchanged(self):
        with loaded_mapper() as loaded:
            node = Node()
            self.assertEqual(loaded.engines.working_set_bytes(node), (6 + 24 + 16) * 4)
            self.assertTrue(_cluster(loaded, "snitch").canExecute(node))

    def test_incomplete_mac_estimate_is_unavailable(self):
        with loaded_mapper() as loaded:
            self.assertIsNone(loaded.mapper.node_macs(Node(output=(None, 8))))
            self.assertIsNone(loaded.mapper.node_macs(Node(left=(None, 3))))
            self.assertEqual(loaded.mapper.node_macs(Node()), 48)

    def test_unknown_engine_cannot_borrow_cva6_rate(self):
        with loaded_mapper() as loaded:
            class FutureEngine(_DeploymentEngine):
                def canExecute(self, _node):
                    return True

            future = FutureEngine("future", {"MatMul": object()})
            mapper = loaded.mapper.CostEngineMapper({"future": future})
            cost = mapper.cost_breakdown(future, Node())
            self.assertFalse(cost["evaluated"])
            self.assertEqual(cost["reason"], "missing_engine_rate_row")
            self.assertIsNone(mapper.mapNodeToEngine(Node(), None))

    def test_missing_or_invalid_named_rate_is_unavailable(self):
        with loaded_mapper() as loaded:
            snitch = _cluster(loaded, "snitch")
            mapper = loaded.mapper.CostEngineMapper({"snitch": snitch})
            loaded.mapper.RATES["snitch"] = {}
            self.assertEqual(
                mapper.cost_breakdown(snitch, Node())["reason"],
                "missing_or_invalid_operator_rate",
            )

            loaded.mapper.RATES["snitch"] = {"MatMul": 0, "_default": 99}
            invalid = mapper.cost_breakdown(snitch, Node())
            self.assertFalse(invalid["evaluated"])
            self.assertEqual(invalid["rate_key"], "MatMul")

    def test_explicit_default_is_identified_as_proxy(self):
        with loaded_mapper() as loaded:
            host = loaded.engines.Cva6HostEngine()
            mapper = loaded.mapper.CostEngineMapper({"cva6": host})
            node = Node(op="Add")
            result = mapper.cost_breakdown(host, node)
            self.assertTrue(result["evaluated"])
            self.assertEqual(result["rate_key"], "_default")
            self.assertEqual(result["rate_kind"], "explicit_default_proxy")

    def test_missing_cluster_offload_cost_is_unavailable(self):
        with loaded_mapper() as loaded:
            snitch = _cluster(loaded, "snitch")
            mapper = loaded.mapper.CostEngineMapper({"snitch": snitch})
            loaded.mapper.OFFLOAD_FIXED.pop("snitch")
            result = mapper.cost_breakdown(snitch, Node())
            self.assertFalse(result["evaluated"])
            self.assertEqual(result["reason"], "missing_or_invalid_cluster_offload_cost")

    def test_external_tables_replace_all_committed_rows(self):
        measured = {
            "RATES": {"ara": {"MatMul": 2.0}},
            "OFFLOAD_FIXED": {"snitch": 10},
            "OFFLOAD_PER_BYTE": {"snitch": 0.25},
        }
        with loaded_mapper(measured) as loaded:
            self.assertEqual(loaded.mapper.RATES, measured["RATES"])
            self.assertEqual(loaded.mapper.OFFLOAD_FIXED, measured["OFFLOAD_FIXED"])
            self.assertEqual(
                loaded.mapper.OFFLOAD_PER_BYTE, measured["OFFLOAD_PER_BYTE"]
            )
            host = loaded.engines.Cva6HostEngine()
            result = loaded.mapper.CostEngineMapper(
                {"cva6": host}, host="ara"
            ).cost_breakdown(host, Node())
            self.assertTrue(result["evaluated"])

    def test_malformed_external_tables_fail_before_replacement(self):
        cases = (
            [],
            {"RATES": {}, "OFFLOAD_FIXED": {}},
            {"RATES": [], "OFFLOAD_FIXED": {}, "OFFLOAD_PER_BYTE": {}},
        )
        for blob in cases:
            with self.subTest(blob=blob):
                with self.assertRaises(RuntimeError):
                    with loaded_mapper(blob):
                        pass

    def test_generic_fallbacks_are_not_priced_as_tuned(self):
        with loaded_mapper() as loaded:
            snitch = _cluster(loaded, "snitch")
            snitch_mapper = loaded.mapper.CostEngineMapper({"snitch": snitch})
            small = Node(right=(3, 7), output=(2, 7))
            small_cost = snitch_mapper.cost_breakdown(snitch, small)
            self.assertFalse(small_cost["evaluated"])
            self.assertEqual(
                small_cost["reason"],
                "deterministic_generic_fallback_has_no_generic_rate",
            )

            tail = Node(right=(3, 9), output=(2, 9))
            self.assertTrue(snitch_mapper.cost_breakdown(snitch, tail)["evaluated"])

            spatz = _cluster(loaded, "spatz")
            spatz_mapper = loaded.mapper.CostEngineMapper({"spatz": spatz})
            transposed = Node(op="Gemm", attrs={"transA": 1, "transB": 0})
            self.assertFalse(
                spatz_mapper.cost_breakdown(spatz, transposed)["evaluated"]
            )
            empty = Node(left=(0, 3), output=(0, 8))
            self.assertFalse(spatz_mapper.cost_breakdown(spatz, empty)["evaluated"])

    def test_unknown_implementation_state_is_not_priced(self):
        with loaded_mapper() as loaded:
            spatz = _cluster(loaded, "spatz")
            mapper = loaded.mapper.CostEngineMapper({"spatz": spatz})
            node = Node(op="Gemm", attrs={"transA": "dynamic"})
            result = mapper.cost_breakdown(spatz, node)
            self.assertFalse(result["evaluated"])
            self.assertEqual(result["reason"], "matmul_gemm_implementation_unavailable")

    def test_compatible_pin_still_selects_unpriced_engine(self):
        with loaded_mapper() as loaded:
            host = loaded.engines.Cva6HostEngine()
            snitch = _cluster(loaded, "snitch")
            engines = {"cva6": host, "snitch": snitch}
            mapper = loaded.mapper.CostEngineMapper(engines, pin="snitch")
            selected = mapper.mapNodeToEngine(
                Node(right=(3, 7), output=(2, 7)), None
            )
            self.assertIs(selected, snitch)
            self.assertEqual(mapper.decisions[-1], ("node", "MatMul", "snitch", None))

    def test_valid_static_automatic_mapping_preserves_spatz_selection(self):
        with loaded_mapper() as loaded:
            host = loaded.engines.Cva6HostEngine()
            snitch = _cluster(loaded, "snitch")
            spatz = _cluster(loaded, "spatz")
            engines = {"cva6": host, "snitch": snitch, "spatz": spatz}
            mapper = loaded.mapper.CostEngineMapper(engines)
            node = Node(left=(32, 32), right=(32, 32), output=(32, 32))
            self.assertIs(mapper.mapNodeToEngine(node, None), spatz)

    def test_no_safe_automatic_cost_returns_no_selection(self):
        with loaded_mapper() as loaded:
            snitch = _cluster(loaded, "snitch")
            mapper = loaded.mapper.CostEngineMapper({"snitch": snitch})
            node = Node(right=(3, 7), output=(2, 7))
            self.assertTrue(snitch.canExecute(node))
            self.assertIsNone(mapper.mapNodeToEngine(node, None))
            self.assertEqual(mapper.decisions, [])


if __name__ == "__main__":
    unittest.main()
