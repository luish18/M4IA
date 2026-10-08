"""Descriptive identities for mapping policies already present in M4IA."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MappingStrategyMeta:
    implementation: str
    factory: str
    cost_model: str
    description: str


META: dict[str, MappingStrategyMeta] = {
    "measured_rate_greedy": MappingStrategyMeta(
        implementation="CostEngineMapper",
        factory="make_mapper",
        cost_model="measured_rates_plus_offload",
        description=(
            "Current node-local greedy mapper: among engines that can execute a node, "
            "choose the minimum estimated cost using measured rates plus fixed/per-byte "
            "offload costs. pin prefers a compatible node on the requested engine; it "
            "is not a whole-graph guarantee or a strategy."
        ),
    ),
}


def build_catalog() -> list[dict]:
    return [
        {
            "id": strategy_id,
            **asdict(meta),
            "implementation_source": "pipeline/hetero_platform/mapper.py",
            "integration_source": "pipeline/hetero_platform/generate.py",
            "decision_explanation": {
                "available": True,
                "path": "mapping.nodes[].mapping_explanation",
                "evidence": "descriptive_shadow_of_mapper_decision",
            },
        }
        for strategy_id, meta in sorted(META.items())
    ]
