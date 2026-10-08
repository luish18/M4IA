# SPDX-License-Identifier: Apache-2.0
"""Extract generated matrix dispatch evidence from Deeploy's bound parser."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral


def _integer_constant(value):
    """Return a concrete emitted integer, without coercing bool or text."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return int(value)


def emitted_kernel_arguments(layer, op):
    """Return constants used by a bound MatMul/Gemm call, when available.

    Deeploy renders the selected binding from
    ``layer.mapper.parser.operatorRepresentation``.  Reading that state avoids
    reconstructing generated arguments from the earlier ONNX graph.  Missing
    or non-integer evidence is omitted deliberately.
    """
    if op not in {"MatMul", "Gemm"}:
        return None

    mapper = getattr(layer, "mapper", None)
    parser = getattr(mapper, "parser", None)
    representation = getattr(parser, "operatorRepresentation", None)
    if not isinstance(representation, Mapping):
        return {}

    keys = ("M", "N", "O")
    if op == "Gemm":
        keys += ("transA", "transB")

    arguments = {}
    for key in keys:
        value = _integer_constant(representation.get(key))
        if value is not None:
            arguments[key] = value
    return arguments


def generated_mapping_nodes(decided, explanations):
    """Build serializable mapping nodes from post-binding generated state."""
    by_node = {
        entry.get("node"): entry
        for entry in explanations
        if isinstance(entry, Mapping)
    }
    nodes = []
    for name, decision in decided.items():
        entry = {
            "index": decision["index"],
            "node": name,
            "op": decision["op"],
            "engine": decision["engine"],
        }
        arguments = emitted_kernel_arguments(decision["layer"], decision["op"])
        if arguments is not None:
            entry["kernel_arguments"] = arguments
        explanation = by_node.get(name)
        if explanation is None:
            explanation = {
                "strategy": "measured_rate_greedy",
                "status": "unavailable",
                "reason": "mapper_explanation_not_retained_for_generated_node",
            }
        elif (
            explanation.get("operator") != decision["op"]
            or explanation.get("selected_engine") != decision["engine"]
        ):
            explanation = {
                "strategy": "measured_rate_greedy",
                "status": "unavailable",
                "reason": "mapper_explanation_disagrees_with_generated_placement",
            }
        entry["mapping_explanation"] = explanation
        nodes.append(entry)
    return nodes
