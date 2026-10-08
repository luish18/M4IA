"""Descriptive catalog for current M4IA host hardware/software profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class HostProfileMeta:
    label: str
    logical_engine: str
    vector: bool
    description: str


META: dict[str, HostProfileMeta] = {
    "cva6": HostProfileMeta("Scalar CVA6 host", "cva6", False, "Scalar CVA6 orchestrator used by hetero_soc."),
    "ara": HostProfileMeta("CVA6 + Ara host", "cva6", True, "The CVA6 logical host with Ara/RVV vector execution, used by hetero_ara."),
}


def build_catalog(host_targets: Mapping[str, str]) -> list[dict]:
    operational = set(host_targets)
    metadata = set(META)
    if operational != metadata:
        raise RuntimeError(
            "host-profile metadata drift against pipeline/build_mesh.py::HOSTS: "
            f"operational_only={sorted(operational - metadata)}, "
            f"metadata_only={sorted(metadata - operational)}"
        )
    return [
        {
            "name": name,
            **asdict(META[name]),
            "target": host_targets[name],
            "operational_source": "pipeline/build_mesh.py::HOSTS",
        }
        for name in sorted(host_targets)
    ]
