# SPDX-License-Identifier: Apache-2.0
"""Choosing which core runs each node.

Deeploy's default EngineMapper takes the first engine that says it can execute
a node, which makes the answer depend on the order the engines happen to be
listed in. This one picks the cheapest, from a cost model seeded with the
per-core measurements this repository already has in results/.

The model is deliberately crude, because it only has to get the *ordering*
right, and the ordering is what the measurements establish:

    cost = offload_overhead(engine) + macs(node) / rate(engine, op)

`rate` is MACs per cycle, derived from results/: for each measured operator,
the MAC count of that benchmark divided by the cycles each core took. The
Snitch cluster's rate is the single-core Xssr/Xfrep rate multiplied by the
eight compute cores it splits the work over. `offload_overhead` is what a job
costs before any arithmetic happens -- mailbox write, doorbell, DMA staging,
completion -- measured by make mesh-test.

The overhead term is the part that matters: it is what stops a small node
being shipped to a cluster that would finish the arithmetic quickly and spend
ten times longer getting the data there and back.

Named operator rates are measurements.  A row's explicit ``_default`` remains
available as a documented proxy, and is kept distinguishable from a named
operator rate.  Missing shapes, rates, offload terms, or implementation facts
stay unavailable rather than being converted into a fabricated cost.  --pin
keeps its existing compatible-node preference semantics, which is how the
mapper's choice gets checked against an alternative rather than assumed.
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, Mapping, Optional

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import DeploymentEngine
from Deeploy.EngineExtension.OptimizationPasses.TopologyOptimizationPasses.EngineColoringPasses import EngineMapper

from .engines import ClusterEngine, _all_fp32, working_set_bytes

try:  # Pipeline scripts import hetero_platform as a top-level package.
    from experiment.matmul_gemm_implementation import resolve_matmul_gemm_implementation
except ImportError:  # Package imports used by unit tests and tooling.
    from pipeline.experiment.matmul_gemm_implementation import resolve_matmul_gemm_implementation

# MACs per cycle, per engine and operator class.
#
# The two cluster rows are measured by `make mesh-test`, which runs exactly the
# configuration this model is predicting: eight compute cores, operands staged
# into TCDM, on the SoC. A rate is that benchmark's MAC count over its measured
# cycles, so it already carries the in-cluster overhead at that problem size;
# OFFLOAD_FIXED below covers the host-side round trip on top.
#
#   GEMM/MatMul  32x32x32          =  32,768 MACs
#   Conv2d       4x16x16 -> 8x3x3  =  56,448 MACs
#
# The cva6 row comes from results/, where the same shapes were measured on the
# standalone board.
#
# Both clusters have eight compute cores, so these are like-for-like. Spatz
# leads on all three because its kernels vectorize the output columns -- see
# runtime/spatz/kernels/gemm_fp32_rvv.c for why that matters so much.
#
# The ara row prices the same host with its Ara vector unit (--host ara): the
# cva6 row scaled by how many fewer cycles each benchmark took under
# `pipeline/run.py --cores cva6,ara` -- GEMM/Regular 1.65x and mymatmul 1.64x
# faster, Conv/Regular_2D_Bias 0.63x, because GCC gathers the convolution
# window and the vector unit issues a gather one element per burst.
RATES: Dict[str, Dict[str, float]] = {
    "cva6": {"Gemm": 0.067, "MatMul": 0.075, "Conv": 0.070, "_default": 0.070},
    "ara": {"Gemm": 0.111, "MatMul": 0.123, "Conv": 0.044, "_default": 0.044},
    "snitch": {"Gemm": 4.52, "MatMul": 4.07, "Conv": 0.59, "_default": 0.59},
    "spatz": {"Gemm": 11.91, "MatMul": 16.79, "Conv": 0.65, "_default": 0.65},
}

# Cycles a job costs before any arithmetic: the mailbox write, the doorbell,
# waking the control core, and the completion handshake. Charged per offload so
# a node too small to be worth shipping stays on the host.
OFFLOAD_FIXED = {"cva6": 0, "snitch": 1200, "spatz": 1200}

# Cycles per byte staged into TCDM and back, from the same measurements.
OFFLOAD_PER_BYTE = {"cva6": 0.0, "snitch": 0.10, "spatz": 0.10}

RATE_TABLE_SOURCE = {
    "kind": "committed_rate_table",
    "path": "pipeline/hetero_platform/mapper.py",
}


# A design-space sweep changes the very hardware these numbers measure, so a
# swept design would otherwise be mapped by another machine's arithmetic -- a
# 16-lane Spatz kept being handed work at a 4-lane Spatz's prices. Each design
# point re-measures the whole table with runtime/tests/mesh_calib.c and points
# HES_RATES at the result; see pipeline/sweep/calibrate.py.
#
# Unset, the committed tables above are used unchanged, so nothing about the
# existing pipeline moves.
def _load_measured_tables() -> None:
    global RATE_TABLE_SOURCE
    path = os.environ.get("HES_RATES")
    if not path:
        return
    blob = json.loads(Path(path).read_text())
    if not isinstance(blob, dict):
        raise RuntimeError(f"{path} must contain a JSON object of measured tables")

    active = (
        ("RATES", RATES),
        ("OFFLOAD_FIXED", OFFLOAD_FIXED),
        ("OFFLOAD_PER_BYTE", OFFLOAD_PER_BYTE),
    )
    replacements = {}
    for name, _table in active:
        if name not in blob:
            raise RuntimeError(f"{path} has no {name} -- it is not a table "
                               "produced by pipeline/sweep/calibrate.py")
        measured = blob[name]
        if not isinstance(measured, dict):
            raise RuntimeError(f"{path} has a non-object {name} table")
        replacements[name] = measured

    # Validate all three before mutating any module global. Replacement is
    # deliberate: merging a partial calibration with committed rows would mix
    # measurements from different machines. A selected Ara host legitimately
    # supplies an `ara` row instead of `cva6`; the mapper selects that row via
    # self.host rather than retaining a stale scalar-host row.
    for name, table in active:
        table.clear()
        table.update(replacements[name])
    RATE_TABLE_SOURCE = {
        "kind": "external_rate_table",
        "path": str(Path(path)),
    }


_load_measured_tables()


def _static_shape(tensor):
    shape = getattr(tensor, "shape", None)
    if not shape or any(
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension < 0
        for dimension in shape
    ):
        return None
    return list(shape)


def node_macs(node: gs.Node) -> float | None:
    """Current MAC estimate, or None when any required extent is unknown."""
    output = _static_shape(node.outputs[0]) if node.outputs else None
    if output is None:
        return None
    output_elements = 1
    for dimension in output:
        output_elements *= dimension

    if node.op in ("Gemm", "MatMul"):
        if len(node.inputs) < 2:
            return None
        left = _static_shape(node.inputs[0])
        return output_elements * left[-1] if left else None
    if node.op == "Conv":
        if len(node.inputs) < 2:
            return None
        weights = _static_shape(node.inputs[1])
        if weights is None:
            return None
        taps = 1
        for dimension in weights[1:]:
            taps *= dimension
        return output_elements * taps
    return float(output_elements)


def _integer_attribute(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return None
    return None


def matmul_gemm_implementation(engine_name: str, node: gs.Node) -> dict | None:
    """Audited cluster implementation state, or None when not applicable."""
    if engine_name not in {"snitch", "spatz"} or node.op not in {"MatMul", "Gemm"}:
        return None
    if len(node.inputs) < 2:
        return resolve_matmul_gemm_implementation(engine_name, node.op, None, None, None)

    left = _static_shape(node.inputs[0])
    right = _static_shape(node.inputs[1])
    if left is None or right is None or len(left) < 2 or len(right) < 2:
        return resolve_matmul_gemm_implementation(engine_name, node.op, None, None, None)

    trans_a = trans_b = 0
    if node.op == "Gemm":
        attributes = getattr(node, "attrs", None)
        if not isinstance(attributes, Mapping):
            return resolve_matmul_gemm_implementation(
                engine_name, node.op, None, None, None, transA=None, transB=None,
            )
        trans_a = _integer_attribute(attributes.get("transA", 0))
        trans_b = _integer_attribute(attributes.get("transB", 0))
        if trans_a is None or trans_b is None:
            return resolve_matmul_gemm_implementation(
                engine_name, node.op, None, None, None,
                transA=attributes.get("transA"), transB=attributes.get("transB"),
            )

    m = left[-1] if trans_a else left[-2]
    n = left[-2] if trans_a else left[-1]
    o = right[-2] if trans_b else right[-1]
    return resolve_matmul_gemm_implementation(
        engine_name, node.op, m, n, o, transA=trans_a, transB=trans_b,
    )


class CostEngineMapper(EngineMapper):
    """Pick the engine whose modelled cost for the node is lowest.

    `pin` forces every node the named engine can execute onto it, which is what
    the mapped-versus-pinned comparison uses.

    `host` is the orchestrator the board carries -- the scalar CVA6, or the
    same core with an Ara vector unit -- and picks the rates the host engine is
    priced at. The engine is called cva6 on both boards, but the two cores run
    the same node at very different speeds, so pricing the vector host at the
    scalar rates would offload work it does faster in place.
    """

    def __init__(self, engineDict: Dict[str, DeploymentEngine],
                 pin: Optional[str] = None, host: str = "cva6") -> None:
        super().__init__(engineDict)
        self.pin = pin
        self.host = host
        self.decisions = []  #: (node name, op, engine, cost) for the report
        self.explanations = []  #: descriptive shadows of completed decisions

    def cost_breakdown(self, engine: DeploymentEngine, node: gs.Node) -> dict:
        """Return a defensible numeric cost or an explicit unavailable reason."""
        implementation = matmul_gemm_implementation(engine.name, node)
        if implementation is not None:
            if implementation["unknown_reason"] is not None:
                return {
                    "evaluated": False,
                    "reason": "matmul_gemm_implementation_unavailable",
                    "implementation": implementation,
                }
            if implementation["fallback_used"]:
                return {
                    "evaluated": False,
                    "reason": "deterministic_generic_fallback_has_no_generic_rate",
                    "implementation": implementation,
                }

        if engine.name == "cva6":
            rates = RATES.get(self.host)
            missing_row = "missing_host_rate_row"
            rate_row = {"kind": "host_profile", "key": self.host}
        else:
            rates = RATES.get(engine.name)
            missing_row = "missing_engine_rate_row"
            rate_row = {"kind": "engine", "key": engine.name}
        if not isinstance(rates, dict):
            return {
                "evaluated": False,
                "reason": missing_row,
                "rate_row": rate_row,
                "rate_table": dict(RATE_TABLE_SOURCE),
            }

        if node.op in rates:
            rate_key = node.op
            rate_kind = "operator_rate"
        elif "_default" in rates:
            rate_key = "_default"
            rate_kind = "explicit_default_proxy"
        else:
            rate_key = node.op
            rate_kind = "unavailable"
        rate = rates.get(rate_key)
        if (
            not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or not math.isfinite(rate)
            or rate <= 0
        ):
            return {
                "evaluated": False,
                "reason": "missing_or_invalid_operator_rate",
                "rate_key": rate_key,
                "rate_kind": rate_kind,
                "rate_row": rate_row,
                "rate_table": dict(RATE_TABLE_SOURCE),
            }

        macs = node_macs(node)
        if macs is None:
            return {"evaluated": False, "reason": "shape_derived_macs_unavailable"}

        fixed = 0
        per_byte = 0.0
        staged_bytes = None
        if isinstance(engine, ClusterEngine):
            staged_bytes = working_set_bytes(node)
            if staged_bytes is None:
                return {"evaluated": False, "reason": "working_set_bytes_unavailable"}
            fixed = OFFLOAD_FIXED.get(engine.name)
            per_byte = OFFLOAD_PER_BYTE.get(engine.name)
            if (
                not isinstance(fixed, (int, float))
                or isinstance(fixed, bool)
                or not math.isfinite(fixed)
                or fixed < 0
                or not isinstance(per_byte, (int, float))
                or isinstance(per_byte, bool)
                or not math.isfinite(per_byte)
                or per_byte < 0
            ):
                return {
                    "evaluated": False,
                    "reason": "missing_or_invalid_cluster_offload_cost",
                }

        transfer = per_byte * staged_bytes if staged_bytes is not None else 0.0
        total = macs / rate + fixed + transfer
        return {
            "evaluated": True,
            "macs": macs,
            "rate_macs_per_cycle": rate,
            "rate_key": rate_key,
            "rate_kind": rate_kind,
            "rate_row": rate_row,
            "rate_table": dict(RATE_TABLE_SOURCE),
            "fixed_offload_cycles": fixed,
            "working_set_bytes": staged_bytes,
            "per_byte_offload_cycles": per_byte if staged_bytes is not None else None,
            "transfer_offload_cycles": transfer if staged_bytes is not None else None,
            "total_estimated_cycles": total,
        }

    @staticmethod
    def _incompatibility_reason(engine: DeploymentEngine, node: gs.Node) -> str:
        """Explain a failed current canExecute check without changing it."""
        if isinstance(engine, ClusterEngine):
            if not engine.enabled:
                return "cluster_disabled"
            if node.op not in engine.Mapping:
                return "operator_not_in_cluster_mapping"
            if not _all_fp32(node):
                return "node_not_fp32"
            staged_bytes = working_set_bytes(node)
            if staged_bytes is None:
                return "working_set_bytes_unavailable"
            if staged_bytes > engine.tcdm_budget:
                return "working_set_exceeds_tcdm_budget"
        if node.op not in engine.Mapping:
            return "operator_not_in_engine_mapping"
        return "compatibility_reason_unknown"

    def _record_explanation(
        self,
        node: gs.Node,
        candidates,
        selected,
        selected_cost,
        *,
        pin_selected: bool,
        cost_details: dict | None = None,
    ) -> None:
        """Record descriptive state after a selection decision has been made."""
        cost_details = cost_details or {}
        entries = []
        for engine in self.engineDict.values():
            compatible = engine in candidates
            retained = compatible and (
                not pin_selected or engine.name == self.pin
            )
            incompatibility_reason = (
                None if compatible else self._incompatibility_reason(engine, node)
            )
            if compatible:
                cost = cost_details.get(engine.name)
                if cost is None:
                    try:
                        cost = self.cost_breakdown(engine, node)
                    except Exception as error:
                        # Explanation collection must never change compatible
                        # pin behavior merely because an unused cost cannot be
                        # described. Automatic selection still evaluates costs
                        # through the unchanged fail-loud path above.
                        cost = {
                            "evaluated": False,
                            "reason": "cost_breakdown_error_during_observation",
                            "error_type": type(error).__name__,
                        }
            else:
                cost = {
                    "evaluated": False,
                    "reason": "engine_incompatible",
                }

            if not compatible:
                pin_influence = (
                    "pinned_engine_incompatible"
                    if engine.name == self.pin
                    else "incompatible_before_pin"
                )
            elif self.pin is None:
                pin_influence = "not_requested"
            elif pin_selected and engine.name == self.pin:
                pin_influence = "retained_by_pin"
            elif pin_selected:
                pin_influence = "excluded_by_compatible_pin"
            else:
                pin_influence = "pin_not_applied"

            entries.append({
                "engine": engine.name,
                "compatible": compatible,
                "candidate_before_pin": compatible,
                "candidate_after_pin": retained,
                "incompatibility_reason": incompatibility_reason,
                "exclusion_reason": (
                    incompatibility_reason
                    if not compatible
                    else "excluded_by_compatible_pin" if not retained else None
                ),
                "pin_influence": pin_influence,
                "cost": cost,
                "cost_used_for_selection": (
                    selected is engine and not pin_selected and selected_cost is not None
                ),
            })

        if selected is None:
            selection_rule = (
                "compatible_engines_without_safe_automatic_cost"
                if candidates
                else "no_compatible_engine"
            )
        elif pin_selected:
            selection_rule = "compatible_pinned_engine_selected"
        else:
            selection_rule = "minimum_safe_estimated_cost"

        self.explanations.append({
            "strategy": "measured_rate_greedy",
            "node": node.name,
            "operator": node.op,
            "requested_pin": self.pin,
            "pin_semantics": "compatible_node_preference_not_whole_graph",
            "engines": entries,
            "selected_engine": selected.name if selected is not None else None,
            "selected_estimated_cost_cycles": selected_cost,
            "selection_rule": selection_rule,
        })

    def cost(self, engine: DeploymentEngine, node: gs.Node) -> float | None:
        breakdown = self.cost_breakdown(engine, node)
        return breakdown["total_estimated_cycles"] if breakdown["evaluated"] else None

    def mapNodeToEngine(self, node: gs.Node, graph: gs.Graph) -> Optional[DeploymentEngine]:
        _ = graph
        candidates = [e for e in self.engineDict.values() if e.canExecute(node)]
        if not candidates:
            self._record_explanation(
                node, candidates, None, None, pin_selected=False,
            )
            return None

        if self.pin is not None:
            pinned = self.engineDict.get(self.pin)
            if pinned is not None and pinned in candidates:
                self.decisions.append((node.name, node.op, pinned.name, None))
                self._record_explanation(
                    node, candidates, pinned, None, pin_selected=True,
                )
                return pinned

        priced = []
        cost_details = {}
        for engine in candidates:
            breakdown = self.cost_breakdown(engine, node)
            cost_details[engine.name] = breakdown
            if breakdown["evaluated"]:
                priced.append((engine, breakdown["total_estimated_cycles"]))
        if not priced:
            self._record_explanation(
                node,
                candidates,
                None,
                None,
                pin_selected=False,
                cost_details=cost_details,
            )
            return None

        best, best_cost = min(priced, key=lambda item: item[1])
        self.decisions.append((node.name, node.op, best.name, best_cost))
        self._record_explanation(
            node,
            candidates,
            best,
            best_cost,
            pin_selected=False,
            cost_details=cost_details,
        )
        return best


def make_mapper(pin: Optional[str] = None, host: str = "cva6"):
    """A CostEngineMapper factory the deployer can instantiate."""

    class _Mapper(CostEngineMapper):

        def __init__(self, engineDict):
            super().__init__(engineDict, pin = pin, host = host)
            _Mapper.last_instance = self

    _Mapper.last_instance = None
    return _Mapper
