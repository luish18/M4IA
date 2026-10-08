"""Descriptive catalog for current M4IA logical execution engines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class EngineMeta:
    label: str
    kind: str
    deployment_engine: str
    description: str


META: dict[str, EngineMeta] = {
    "cva6": EngineMeta("CVA6 host", "host", "Cva6HostEngine", "Logical host engine; scalar CVA6 and CVA6+Ara are profiles of it."),
    "snitch": EngineMeta("Snitch cluster", "cluster", "SnitchClusterEngine", "Snitch cluster offload engine."),
    "spatz": EngineMeta("Spatz cluster", "cluster", "SpatzClusterEngine", "Spatz/RVV cluster offload engine."),
}


def build_catalog(engine_names: Mapping[int, str], runtime_macros: Mapping[str, str]) -> list[dict]:
    ids_by_name = {name: system_id for system_id, name in engine_names.items()}
    if len(ids_by_name) != len(engine_names):
        raise RuntimeError("architectural engine names are not unique")
    metadata_names = set(META)
    architectural_names = set(ids_by_name)
    runtime_names = set(runtime_macros)
    if architectural_names != metadata_names:
        raise RuntimeError(
            "engine metadata drift against targets/hetero/system.py: "
            f"architecture_only={sorted(architectural_names - metadata_names)}, "
            f"metadata_only={sorted(metadata_names - architectural_names)}"
        )
    if runtime_names != metadata_names:
        raise RuntimeError(
            "engine metadata drift against pipeline/hetero_platform/progress.py: "
            f"runtime_only={sorted(runtime_names - metadata_names)}, "
            f"metadata_only={sorted(metadata_names - runtime_names)}"
        )
    return [
        {
            "name": name,
            "system_id": system_id,
            **asdict(META[name]),
            "runtime_macro": runtime_macros[name],
            "architectural_source": "targets/hetero/system.py",
            "runtime_macro_source": "pipeline/hetero_platform/progress.py",
        }
        for system_id, name in sorted(engine_names.items())
    ]
