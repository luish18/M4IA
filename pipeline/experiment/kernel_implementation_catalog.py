"""Descriptive inventory of concrete fp32 MatMul/Gemm implementation paths.

The catalog shadows current build/runtime mechanisms. It neither selects a
kernel nor claims which implementation ran in an experiment.
"""

from __future__ import annotations

from pathlib import Path


_IMPLEMENTATIONS = (
    {
        "id": "cva6.fp32.matmul_gemm.deeploy_generic",
        "engine": "cva6",
        "operators": ("MatMul", "Gemm"),
        "classification": "generic_c",
        "source_file": None,
        "source_tree": "deps/deeploy/TargetLibraries/Generic/src",
        "symbols": ("MatMul_fp32_fp32_fp32", "Gemm_fp32_fp32_fp32_fp32"),
        "build_selection": {
            "mechanism": "Generic library compiled without symbol rename for the host image",
            "selector": None,
        },
        "runtime_dispatch": "Deeploy-generated Network.c calls the Generic symbols on host-mapped nodes.",
        "compatibility_conditions": None,
        "fallback_id": None,
    },
    {
        "id": "snitch.fp32.matmul_gemm.ssr_frep",
        "engine": "snitch",
        "operators": ("MatMul", "Gemm"),
        "classification": "handwritten_ssr_frep",
        "source_file": "runtime/snitch/kernels/gemm_fp32_ssr.c",
        "source_tree": None,
        "symbols": ("MatMul_fp32_fp32_fp32", "Gemm_fp32_fp32_fp32_fp32"),
        "build_selection": {
            "mechanism": "Generic symbols are renamed while the Snitch source retains the public symbols",
            "selector": "-D<symbol>=<symbol>_generic",
        },
        "runtime_dispatch": "cluster_main.c dispatches HES_K_MATMUL_FP32 and HES_K_GEMM_FP32 to the public symbols.",
        "compatibility_conditions": {
            "optimized_entry": "M != 0, N != 0, and O >= 8",
            "scalar_tail": "O % 8 columns execute the scalar tail in the same source",
        },
        "fallback_id": "snitch.fp32.matmul_gemm.deeploy_generic_fallback",
    },
    {
        "id": "snitch.fp32.matmul_gemm.deeploy_generic_fallback",
        "engine": "snitch",
        "operators": ("MatMul", "Gemm"),
        "classification": "generic_c_fallback",
        "source_file": None,
        "source_tree": "deps/deeploy/TargetLibraries/Generic/src",
        "symbols": ("MatMul_fp32_fp32_fp32_generic", "Gemm_fp32_fp32_fp32_fp32_generic"),
        "build_selection": {
            "mechanism": "Generic library is compiled with public MatMul/Gemm symbols renamed",
            "selector": "-D<symbol>=<symbol>_generic",
        },
        "runtime_dispatch": "Called directly by the Snitch SSR/FREP source for empty inputs or O < 8.",
        "compatibility_conditions": None,
        "fallback_id": None,
    },
    {
        "id": "spatz.fp32.matmul_gemm.rvv_tuned",
        "engine": "spatz",
        "operators": ("MatMul", "Gemm"),
        "classification": "handwritten_rvv",
        "source_file": "runtime/spatz/kernels/gemm_fp32_rvv.c",
        "source_tree": None,
        "symbols": ("MatMul_fp32_fp32_fp32", "Gemm_fp32_fp32_fp32_fp32"),
        "build_selection": {
            "mechanism": "Generic symbols are renamed while the Spatz RVV source retains the public symbols",
            "selector": "-D<symbol>=<symbol>_generic",
        },
        "runtime_dispatch": "cluster_main.c dispatches HES_K_MATMUL_FP32 and HES_K_GEMM_FP32 to the public symbols.",
        "compatibility_conditions": {
            "matmul_optimized_entry": "M != 0, N != 0, and O != 0",
            "gemm_optimized_entry": "M != 0, N != 0, O != 0, transA == 0, and transB == 0",
        },
        "fallback_id": "spatz.fp32.matmul_gemm.deeploy_generic_fallback",
    },
    {
        "id": "spatz.fp32.matmul_gemm.deeploy_generic_fallback",
        "engine": "spatz",
        "operators": ("MatMul", "Gemm"),
        "classification": "generic_c_fallback",
        "source_file": None,
        "source_tree": "deps/deeploy/TargetLibraries/Generic/src",
        "symbols": ("MatMul_fp32_fp32_fp32_generic", "Gemm_fp32_fp32_fp32_fp32_generic"),
        "build_selection": {
            "mechanism": "Generic library is compiled with public MatMul/Gemm symbols renamed",
            "selector": "-D<symbol>=<symbol>_generic",
        },
        "runtime_dispatch": "Called directly by the Spatz RVV source for empty inputs; GEMM also falls back for either transpose.",
        "compatibility_conditions": None,
        "fallback_id": None,
    },
)


def _validate_entry(root: Path, entry: dict) -> None:
    source_file = entry["source_file"]
    if source_file is None:
        return
    path = root / source_file
    if not path.is_file():
        raise RuntimeError(f"kernel implementation source is missing: {source_file}")
    text = path.read_text()
    missing = [symbol for symbol in entry["symbols"] if f"{symbol}(" not in text]
    if missing:
        raise RuntimeError(f"kernel implementation symbols missing from {source_file}: {missing}")


def _validate_integration(root: Path) -> None:
    build = (root / "pipeline" / "build_mesh.py").read_text()
    dispatch = (root / "runtime" / "mesh" / "cluster_main.c").read_text()
    for fragment in (
        'RUNTIME / "snitch" / "kernels"',
        'RUNTIME / "spatz" / "kernels"',
        "MatMul_fp32_fp32_fp32",
        "Gemm_fp32_fp32_fp32_fp32",
    ):
        if fragment not in build:
            raise RuntimeError(f"kernel build integration drifted: missing {fragment!r}")
    for fragment in (
        "HES_K_MATMUL_FP32",
        "HES_K_GEMM_FP32",
        "MatMul_fp32_fp32_fp32(",
        "Gemm_fp32_fp32_fp32_fp32(",
    ):
        if fragment not in dispatch:
            raise RuntimeError(f"kernel runtime dispatch drifted: missing {fragment!r}")


def build_catalog(root: Path | str) -> list[dict]:
    root = Path(root)
    for entry in _IMPLEMENTATIONS:
        _validate_entry(root, entry)
    _validate_integration(root)
    return [
        {
            **entry,
            "operators": list(entry["operators"]),
            "symbols": list(entry["symbols"]),
            "build_selection": dict(entry["build_selection"]),
            "compatibility_conditions": (
                dict(entry["compatibility_conditions"])
                if entry["compatibility_conditions"] is not None else None
            ),
        }
        for entry in _IMPLEMENTATIONS
    ]
