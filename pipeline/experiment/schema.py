"""Pure experiment-description types for the M4IA workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


def _mapping_pairs(data: object, field_name: str, *, integers: bool) -> tuple[tuple[str, object], ...]:
    if not isinstance(data, Mapping):
        raise ValueError(f"{field_name} must be an object")
    pairs = []
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings")
        if integers and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"{field_name} value {key!r} must be an integer")
        if not integers and not (
            value is None or isinstance(value, (str, int, float, bool))
        ):
            raise ValueError(f"{field_name} value {key!r} must be a JSON scalar")
        pairs.append((key, value))
    return tuple(sorted(pairs))


@dataclass(frozen=True)
class WorkloadSpec:
    path: str
    content_digest: str
    workload_fingerprint: str
    artifact_digests: tuple[dict, ...]
    application: str | None
    opset: tuple[dict, ...]
    inputs: tuple[dict, ...]
    outputs: tuple[dict, ...]
    node_count: int
    op_counts: tuple[tuple[str, int], ...]
    initializer_count: int
    descriptors: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "content_digest": self.content_digest,
            "workload_fingerprint": self.workload_fingerprint,
            "artifact_digests": [dict(item) for item in self.artifact_digests],
            "application": self.application,
            "opset": [dict(item) for item in self.opset],
            "inputs": [dict(item) for item in self.inputs],
            "outputs": [dict(item) for item in self.outputs],
            "node_count": self.node_count,
            "op_counts": dict(self.op_counts),
            "initializer_count": self.initializer_count,
            "descriptors": dict(self.descriptors),
        }


@dataclass(frozen=True)
class ExperimentRequest:
    platform: str = "m4ia_current"
    workload: str = ""
    design_overrides: tuple[tuple[str, int], ...] = ()
    host: str = "cva6"
    mapping_strategy: str = "measured_rate_greedy"
    pin: str | None = None
    frontend: str | None = None
    serial: bool = False
    power: bool = False
    # Mirrors the current public --dram spelling. None records omission; the
    # resolver canonicalizes it to the operational default, fixed.
    dram: str | None = None
    # Current M4IA can carry DRAM_OVERRIDES in HES_DESIGN. This is identity
    # only here; no public override CLI or execution wiring is introduced.
    dram_overrides: tuple[tuple[str, object], ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ExperimentRequest":
        if not isinstance(data, Mapping):
            raise ValueError("ExperimentRequest must be an object")
        if not all(isinstance(name, str) for name in data):
            raise ValueError("ExperimentRequest field names must be strings")
        allowed = {
            "platform", "workload", "design_overrides", "host",
            "mapping_strategy", "pin", "frontend", "serial", "power",
            "dram", "dram_overrides",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown ExperimentRequest fields: {sorted(unknown)}")

        for name in ("platform", "workload", "host", "mapping_strategy"):
            if name in data and not isinstance(data[name], str):
                raise ValueError(f"{name} must be a string")
        for name in ("pin", "frontend", "dram"):
            if name in data and data[name] is not None and not isinstance(data[name], str):
                raise ValueError(f"{name} must be a string or null")
        for name in ("serial", "power"):
            if name in data and not isinstance(data[name], bool):
                raise ValueError(f"{name} must be a boolean")

        kwargs = dict(data)
        kwargs["design_overrides"] = _mapping_pairs(
            data.get("design_overrides", {}), "design_overrides", integers=True,
        )
        kwargs["dram_overrides"] = _mapping_pairs(
            data.get("dram_overrides", {}), "dram_overrides", integers=False,
        )
        return cls(**kwargs)

    def overrides_dict(self) -> dict[str, int]:
        return dict(self.design_overrides)

    def dram_overrides_dict(self) -> dict[str, object]:
        return dict(self.dram_overrides)

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "workload": self.workload,
            "design_overrides": self.overrides_dict(),
            "host": self.host,
            "mapping_strategy": self.mapping_strategy,
            "pin": self.pin,
            "frontend": self.frontend,
            "serial": self.serial,
            "power": self.power,
            "dram": self.dram,
            "dram_overrides": self.dram_overrides_dict(),
        }


@dataclass(frozen=True)
class ResolvedExperiment:
    schema_version: int
    resolved_fingerprint: str
    platform: str
    hardware: Mapping[str, object]
    memory: Mapping[str, object]
    resources: Mapping[str, object]
    host_profile: str
    simulator: Mapping[str, object]
    software: Mapping[str, object]
    workload: WorkloadSpec
    mapping: Mapping[str, object]
    execution: Mapping[str, object]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "resolved_fingerprint": self.resolved_fingerprint,
            "platform": self.platform,
            "hardware": dict(self.hardware),
            "memory": dict(self.memory),
            "resources": dict(self.resources),
            "host_profile": self.host_profile,
            "simulator": dict(self.simulator),
            "software": dict(self.software),
            "workload": self.workload.to_dict(),
            "mapping": dict(self.mapping),
            "execution": dict(self.execution),
        }
