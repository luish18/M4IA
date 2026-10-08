"""Descriptive main-memory identity derived from M4IA's DRAM presets.

This module neither selects memory nor configures the simulator. Operational
ownership remains in ``targets/hetero/memsys.py`` and
``targets/hetero/dram_presets.py``.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence

from .fingerprint import fingerprint


@dataclass(frozen=True)
class MemoryMeta:
    label: str
    description: str


META: dict[str, MemoryMeta] = {
    "fixed": MemoryMeta(
        label="Fixed main memory",
        description="The current fixed main-memory model.",
    ),
    "lpddr4": MemoryMeta(
        label="LPDDR4 main memory",
        description="Main memory using the current operational LPDDR4 preset.",
    ),
    "lpddr4x": MemoryMeta(
        label="LPDDR4X main memory",
        description="Main memory using the current operational LPDDR4X preset.",
    ),
    "lpddr5": MemoryMeta(
        label="LPDDR5 main memory",
        description="Main memory using the current operational LPDDR5 preset.",
    ),
    "hyperram": MemoryMeta(
        label="HyperRAM main memory",
        description="Main memory using the current operational HyperRAM preset.",
    ),
}


def _operational_sources(name: str) -> list[dict]:
    preset_symbols = ["KINDS"] if name == "fixed" else ["KINDS", "PRESETS", "params"]
    return [
        {
            "path": "pipeline/common.py",
            "symbols": ["DRAM_KINDS", "use_dram"],
            "role": "public selection validation and HES_DESIGN injection",
        },
        {
            "path": "targets/hetero/memsys.py",
            "symbols": ["DRAM_KIND", "DRAM_LATENCY", "DRAM_WIDTH", "L2", "make_main_memory"],
            "role": "selection consumption and memory/cache behavioral consequences",
        },
        {
            "path": "targets/hetero/dram_presets.py",
            "symbols": preset_symbols,
            "role": (
                "supported-kind registry"
                if name == "fixed" else "supported-kind registry and device preset expansion"
            ),
        },
    ]


def load_operational_memory_source(root: Path | str) -> ModuleType:
    """Load the pure operational preset module without importing GVSoC."""
    path = Path(root) / "targets" / "hetero" / "dram_presets.py"
    spec = importlib.util.spec_from_file_location("m4ia_experiment_dram_presets", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load operational memory presets from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_catalog(
    kinds: Sequence[str],
    presets: Mapping[str, Mapping[str, object]],
    fields: Sequence[str],
) -> list[dict]:
    """Attach descriptions while refusing drift from operational identities."""
    operational = set(kinds)
    metadata = set(META)
    if operational != metadata:
        raise RuntimeError(
            "memory metadata drift against targets/hetero/dram_presets.py::KINDS: "
            f"operational_only={sorted(operational - metadata)}, "
            f"metadata_only={sorted(metadata - operational)}"
        )
    expected_presets = operational - {"fixed"}
    if set(presets) != expected_presets:
        raise RuntimeError(
            "memory preset drift: "
            f"kinds_without_presets={sorted(expected_presets - set(presets))}, "
            f"presets_without_kinds={sorted(set(presets) - expected_presets)}"
        )
    rows = []
    for name in kinds:
        model = "fixed_latency" if name == "fixed" else presets[name]["kind"]
        row = {
            "name": name,
            **asdict(META[name]),
            "model": model,
            "operational_key": "DRAM_KIND",
            "operational_sources": _operational_sources(name),
            "supports_overrides": name != "fixed",
            "override_fields": list(fields) if name != "fixed" else [],
        }
        rows.append(row)
    return rows


def current_memory_catalog(root: Path | str) -> list[dict]:
    source = load_operational_memory_source(root)
    return build_catalog(source.KINDS, source.PRESETS, source.FIELDS)


def resolve_memory(
    root: Path | str,
    requested_kind: str | None,
    overrides: Mapping[str, object] | None = None,
) -> dict:
    """Canonicalize one descriptive memory identity from operational presets."""
    source = load_operational_memory_source(root)
    catalog = {row["name"]: row for row in build_catalog(
        source.KINDS, source.PRESETS, source.FIELDS,
    )}
    kind = "fixed" if requested_kind is None else requested_kind
    if kind not in catalog:
        raise ValueError(
            f"unknown main memory {kind!r}; choose one of {', '.join(source.KINDS)}"
        )

    overrides = dict(overrides or {})
    if kind == "fixed" and overrides:
        raise ValueError("dram_overrides are not operational for fixed main memory")

    row = catalog[kind]
    if kind == "fixed":
        preset_fingerprint = None
        configuration = {"kind": kind}
    else:
        # params() is the operational expansion and also refuses unknown fields.
        preset_parameters = source.params(kind)
        resolved_parameters = source.params(kind, overrides)
        preset_fingerprint = fingerprint(preset_parameters)
        configuration = {"kind": kind, "parameters": resolved_parameters}

    return {
        "kind": kind,
        "model": row["model"],
        "label": row["label"],
        "overrides": overrides,
        "preset_fingerprint": preset_fingerprint,
        "fingerprint": fingerprint(configuration),
        "operational_key": row["operational_key"],
        "operational_sources": row["operational_sources"],
    }
