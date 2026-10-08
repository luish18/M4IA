# SPDX-License-Identifier: Apache-2.0
"""Resolve actual MatMul/Gemm implementations from completed-node evidence."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

from .matmul_gemm_implementation import (
    CVA6_GENERIC,
    resolve_matmul_gemm_implementation,
)


EVIDENCE_TYPE = "deterministic_compiled_dispatch_and_runtime_evidence"
UNAVAILABLE_EVIDENCE_TYPE = "unavailable"


def _unknown(reason: str) -> dict:
    return {
        "implementation_id": None,
        "fallback_used": None,
        "fallback_from": None,
        "fallback_to": None,
        "fallback_reason": None,
        "implementation_detail": None,
        "evidence": {"type": UNAVAILABLE_EVIDENCE_TYPE, "reason": reason},
    }


def _known(resolved: Mapping[str, object], sources: list[str]) -> dict:
    return {
        "implementation_id": resolved["implementation_id"],
        "fallback_used": resolved["fallback_used"],
        "fallback_from": resolved["fallback_from"],
        "fallback_to": resolved["fallback_to"],
        "fallback_reason": resolved["fallback_reason"],
        "implementation_detail": resolved["implementation_detail"],
        "evidence": {"type": EVIDENCE_TYPE, "sources": sources},
    }


def _integer(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return int(value)


def _completed_beacon_reason(node: Mapping[str, object]) -> str | None:
    index = _integer(node.get("node"))
    cycles = _integer(node.get("cycles"))
    if index is None or index < 0 or cycles is None or cycles < 0:
        return "completed runtime beacon evidence is unavailable"
    if not isinstance(node.get("op"), str) or not isinstance(node.get("engine"), str):
        return "completed runtime beacon identity is unavailable"
    return None


def _generated_match_reason(completed, generated) -> str | None:
    generated_index = _integer(generated.get("index"))
    if generated_index is None or generated_index != completed.get("node"):
        return "generated/runtime node index mismatch"
    if generated.get("op") != completed.get("op"):
        return "generated/runtime operator mismatch"
    if generated.get("engine") != completed.get("engine"):
        return "generated/runtime engine mismatch"
    return None


def resolve_actual_implementation(
    completed_node: Mapping[str, object],
    generated_node: Mapping[str, object] | None = None,
) -> dict:
    """Resolve implementation only from a completed beacon and dispatch facts.

    Mapper placement is insufficient.  CVA6's current compiled host path is
    unambiguously Generic after a completed beacon.  Cluster MatMul/Gemm also
    require matching generated metadata and every argument needed by the
    audited compiled dispatch rule.
    """
    if not isinstance(completed_node, Mapping):
        return _unknown("completed runtime beacon evidence is unavailable")
    if reason := _completed_beacon_reason(completed_node):
        return _unknown(reason)

    op = completed_node["op"]
    engine = completed_node["engine"]
    if generated_node is not None:
        if not isinstance(generated_node, Mapping):
            return _unknown("generated node metadata is unavailable")
        if reason := _generated_match_reason(completed_node, generated_node):
            return _unknown(reason)

    if op not in {"MatMul", "Gemm"}:
        return _unknown(f"no deterministic implementation rule for operator {op!r}")
    if engine == "cva6":
        return _known({
            "implementation_id": CVA6_GENERIC,
            "fallback_used": False,
            "fallback_from": None,
            "fallback_to": None,
            "fallback_reason": None,
            "implementation_detail": None,
        }, ["completed_runtime_beacon", "audited_compiled_host_path"])
    if engine not in {"snitch", "spatz"}:
        return _unknown(f"no deterministic implementation rule for engine {engine!r}")
    if generated_node is None:
        return _unknown("generated node metadata is unavailable")

    arguments = generated_node.get("kernel_arguments")
    if not isinstance(arguments, Mapping):
        return _unknown("generated dispatch arguments are unavailable")
    dimensions = {key: _integer(arguments.get(key)) for key in ("M", "N", "O")}
    if any(value is None for value in dimensions.values()):
        return _unknown("generated matrix dimensions are incomplete or non-integer")

    trans_a = trans_b = 0
    if engine == "spatz" and op == "Gemm":
        trans_a = _integer(arguments.get("transA"))
        trans_b = _integer(arguments.get("transB"))
        if trans_a is None or trans_b is None:
            return _unknown("generated GEMM transpose arguments are unavailable")

    resolved = resolve_matmul_gemm_implementation(
        engine,
        op,
        dimensions["M"],
        dimensions["N"],
        dimensions["O"],
        transA=trans_a,
        transB=trans_b,
    )
    if resolved["implementation_id"] is None:
        return _unknown(resolved["unknown_reason"])
    return _known(
        resolved,
        [
            "completed_runtime_beacon",
            "generated_dispatch_arguments",
            "audited_compiled_dispatch_rules",
        ],
    )


def annotate_completed_nodes(mapping: Mapping[str, object], result: Mapping[str, object]) -> dict:
    """Return a copy of ``result`` annotated by unambiguous generated indices."""
    planned = mapping.get("nodes") if isinstance(mapping, Mapping) else None
    planned = planned if isinstance(planned, list) else []

    by_index = {}
    ambiguous = set()
    for entry in planned:
        if not isinstance(entry, Mapping):
            continue
        index = _integer(entry.get("index"))
        if index is None:
            continue
        if index in by_index:
            ambiguous.add(index)
        else:
            by_index[index] = entry

    output = dict(result) if isinstance(result, Mapping) else {}
    runtime_nodes = result.get("nodes") if isinstance(result, Mapping) else None
    runtime_nodes = runtime_nodes if isinstance(runtime_nodes, list) else []
    annotated = []
    for runtime_node in runtime_nodes:
        if not isinstance(runtime_node, Mapping):
            continue
        node = dict(runtime_node)
        index = _integer(node.get("node"))
        if index in ambiguous:
            implementation = _unknown("generated node index is ambiguous")
        else:
            generated = by_index.get(index)
            if generated is None:
                implementation = _unknown("generated node metadata is unavailable")
            else:
                implementation = resolve_actual_implementation(node, generated)
        node["implementation"] = implementation
        annotated.append(node)
    output["nodes"] = annotated
    return output
