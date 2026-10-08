"""Descriptive metadata for current M4IA sweep parameters.

This module does not own values, validation, or grids. Those remain in
``pipeline/sweep/design.py`` and ``pipeline/sweep/run.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ParameterMeta:
    label: str
    group: str
    unit: str | None
    description: str


META: dict[str, ParameterMeta] = {
    "HOST_VLEN": ParameterMeta("Ara VLEN", "host_vector", "bits", "Vector register length of the Ara unit attached to the CVA6 host."),
    "HOST_NB_LANES": ParameterMeta("Ara lanes", "host_vector", "lanes", "Number of vector lanes in the Ara unit attached to the CVA6 host."),
    "HOST_LANE_WIDTH": ParameterMeta("Ara lane width", "host_vector", "bytes", "Datapath width of each Ara vector lane."),
    "SPATZ_VLEN": ParameterMeta("Spatz VLEN", "spatz_vector", "bits", "Vector register length of each Spatz vector unit."),
    "SPATZ_NB_LANES": ParameterMeta("Spatz lanes", "spatz_vector", "lanes", "Number of vector lanes in each Spatz vector unit."),
    "SPATZ_LANE_WIDTH": ParameterMeta("Spatz lane width", "spatz_vector", "bytes", "Datapath width of each Spatz vector lane."),
    "SNITCH_NB_CORE": ParameterMeta("Snitch cluster cores", "cluster_shape", "cores", "Total Snitch-cluster core count, including the control/DMA core."),
    "SPATZ_NB_CORE": ParameterMeta("Spatz cluster cores", "cluster_shape", "cores", "Total Spatz-cluster core count, including the control/DMA core."),
    "TCDM_SIZE": ParameterMeta("TCDM size", "cluster_memory", "bytes", "Scratchpad capacity assigned to each cluster in the heterogeneous SoC."),
    "ICACHE_SIZE": ParameterMeta("L1 I-cache size", "host_cache", "bytes", "Capacity of the CVA6 host instruction cache."),
    "ICACHE_WAYS": ParameterMeta("L1 I-cache ways", "host_cache", "ways", "Associativity of the CVA6 host instruction cache."),
    "DCACHE_SIZE": ParameterMeta("L1 D-cache size", "host_cache", "bytes", "Capacity of the CVA6 host data cache."),
    "DCACHE_WAYS": ParameterMeta("L1 D-cache ways", "host_cache", "ways", "Associativity of the CVA6 host data cache."),
    "L2_SIZE": ParameterMeta("L2 size", "host_cache", "bytes", "Capacity of the shared host-side L2 cache."),
    "L2_WAYS": ParameterMeta("L2 ways", "host_cache", "ways", "Associativity of the shared host-side L2 cache."),
    "LINE_SIZE": ParameterMeta("Cache line size", "host_cache", "bytes", "Cache-line size used by the modelled host cache hierarchy."),
    "NARROW_AXI_WIDTH": ParameterMeta("Narrow AXI width", "interconnect", "bytes", "Width of the narrow AXI path used for host-to-cluster accesses."),
    "WIDE_AXI_WIDTH": ParameterMeta("Wide AXI width", "interconnect", "bytes", "Width of the wide AXI path used toward shared main memory."),
}


def build_catalog(
    defaults: Mapping[str, int],
    build_time: Sequence[str] = (),
    screening_values: Mapping[str, Sequence[int]] | None = None,
) -> list[dict]:
    default_names = set(defaults)
    meta_names = set(META)
    if default_names != meta_names:
        raise RuntimeError(
            "parameter catalog drift: "
            f"missing metadata={sorted(default_names - meta_names)}, "
            f"metadata without supported knob={sorted(meta_names - default_names)}"
        )
    build_time = set(build_time)
    if unknown := sorted(build_time - default_names):
        raise RuntimeError(f"unknown build-time parameters: {unknown}")
    screening_values = screening_values or {}
    if unknown := sorted(set(screening_values) - default_names):
        raise RuntimeError(f"screening grid contains unknown parameters: {unknown}")
    return [
        {
            "name": name,
            **asdict(META[name]),
            "value_kind": "integer",
            "default": defaults[name],
            "build_time": name in build_time,
            "screening_values": list(screening_values.get(name, ())),
            "operational_source": "pipeline/sweep/design.py",
        }
        for name in defaults
    ]
