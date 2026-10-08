"""Pure request-to-resolved experiment logic for current M4IA."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping

from .engine_catalog import META as ENGINE_META
from .fingerprint import fingerprint
from .mapping_strategy_catalog import META as MAPPING_META
from .memory_catalog import resolve_memory
from .resource_summary import build_resource_summary
from .schema import ExperimentRequest, ResolvedExperiment, WorkloadSpec
from .workload import inspect_workload

SCHEMA_VERSION = 2
CURRENT_PLATFORM = "m4ia_current"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def resolve_experiment(
    request: ExperimentRequest,
    *,
    design_api,
    host_profiles: Mapping[str, Mapping[str, object]],
    workload_path: Path | str | None = None,
    application: str | None = None,
    simulator_identity: Mapping[str, object] | None = None,
    software_identity: Mapping[str, object] | None = None,
    repository_root: Path | str = REPOSITORY_ROOT,
    workload_inspector: Callable[..., WorkloadSpec] = inspect_workload,
) -> ResolvedExperiment:
    if request.platform != CURRENT_PLATFORM:
        raise ValueError(
            f"unknown platform {request.platform!r}; current schema supports only "
            f"{CURRENT_PLATFORM!r}"
        )
    if request.host not in host_profiles:
        raise ValueError(
            f"unknown host profile {request.host!r}; known profiles: {sorted(host_profiles)}"
        )
    if request.mapping_strategy not in MAPPING_META:
        raise ValueError(
            f"unknown mapping strategy {request.mapping_strategy!r}; "
            f"known strategies: {sorted(MAPPING_META)}"
        )
    if request.pin is not None and request.pin not in ENGINE_META:
        raise ValueError(
            f"unknown pinned engine {request.pin!r}; known engines: {sorted(ENGINE_META)}"
        )
    if request.frontend is not None and request.frontend not in {"snitch", "spatz"}:
        raise ValueError("frontend must be one of: snitch, spatz")

    overrides = request.overrides_dict()
    resolved_design = design_api.resolve(overrides)
    reasons = design_api.validate(overrides)
    if reasons:
        raise ValueError("invalid design: " + "; ".join(reasons))

    profile = dict(host_profiles[request.host])
    if "target" not in profile:
        raise ValueError(f"host profile {request.host!r} has no resolved target")

    chosen_spelling = workload_path if workload_path is not None else request.workload
    if chosen_spelling is None or (
        isinstance(chosen_spelling, str) and not chosen_spelling
    ):
        raise ValueError("workload path is required")
    chosen_path = Path(chosen_spelling)
    workload = workload_inspector(chosen_path, application=application)
    if request.workload:
        workload = replace(workload, path=request.workload)

    memory = resolve_memory(
        repository_root,
        request.dram,
        request.dram_overrides_dict(),
    )
    strategy = MAPPING_META[request.mapping_strategy]
    hardware = {
        "design": resolved_design,
        "design_slug": design_api.slug(overrides),
        "build_key": design_api.build_key(overrides),
        # Deliberately numeric-design-only. Memory belongs to the separate
        # resolved experiment identity below.
        "fingerprint": fingerprint(resolved_design),
    }
    mapping = {
        "strategy": request.mapping_strategy,
        "implementation": strategy.implementation,
        "factory": strategy.factory,
        "cost_model": strategy.cost_model,
        "pin": request.pin,
        "pin_semantics": "compatible_node_preference_not_whole_graph",
        "host_cost_profile": request.host,
    }
    execution = {
        "frontend": request.frontend,
        "serial": request.serial,
        "power": request.power,
    }
    simulator = dict(simulator_identity) if simulator_identity is not None else {
        "kind": "gvsoc",
        "target": profile["target"],
    }
    software = dict(software_identity) if software_identity is not None else {}
    resources = build_resource_summary(resolved_design)

    identity_payload = {
        "schema_version": SCHEMA_VERSION,
        "platform": request.platform,
        "hardware_fingerprint": hardware["fingerprint"],
        "memory_fingerprint": memory["fingerprint"],
        "host_profile": request.host,
        "simulator": simulator,
        "software": software,
        "workload_fingerprint": workload.workload_fingerprint,
        "mapping": mapping,
        "execution": execution,
    }

    return ResolvedExperiment(
        schema_version=SCHEMA_VERSION,
        resolved_fingerprint=fingerprint(identity_payload),
        platform=request.platform,
        hardware=hardware,
        memory=memory,
        resources=resources,
        host_profile=request.host,
        simulator=simulator,
        software=software,
        workload=workload,
        mapping=mapping,
        execution=execution,
    )
