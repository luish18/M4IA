# SPDX-License-Identifier: Apache-2.0
"""Small source, Git, patch, and executable provenance primitives for M4IA."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .fingerprint import (
    file_digest,
    file_set_manifest,
    fingerprint,
    sha256_bytes,
)


SOURCE_SUFFIXES = frozenset({
    ".S",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".json",
    ".ld",
    ".py",
    ".s",
})


def source_files_under(path: Path | str) -> list[Path]:
    """Return deterministic source files under ``path`` using safe suffixes."""
    path = Path(path)
    if not path.exists():
        return []
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix in SOURCE_SUFFIXES
    )


_RUN_EXPERIMENT_SOURCES = (
    "actual_implementation.py",
    "artifact_identity.py",
    "discovery.py",
    "engine_catalog.py",
    "fingerprint.py",
    "generated_arguments.py",
    "host_profile_catalog.py",
    "mapping_strategy_catalog.py",
    "matmul_gemm_implementation.py",
    "memory_catalog.py",
    "resolve.py",
    "schema.py",
    "workload.py",
)


_RUN_COMMON_SOURCES = (
    "pipeline/build_mesh.py",
    "pipeline/common.py",
    "pipeline/gen_system_header.py",
    "pipeline/run_hetero.py",
    "runtime/common/bench.h",
    "runtime/common/mfcc.h",
    "runtime/common/miniio.h",
    "runtime/common/semihost.h",
    "runtime/common/syscalls.c",
    "runtime/mesh/cluster_main.c",
    "runtime/mesh/crt0_cluster.S",
    "runtime/mesh/crt0_host.S",
    "runtime/mesh/hes_cluster.h",
    "runtime/mesh/hes_dma.h",
    "runtime/mesh/hes_host.c",
    "runtime/mesh/hes_host.h",
    "runtime/mesh/hes_job.h",
    "runtime/mesh/hes_mailbox.h",
    "runtime/snitch/snitch_ssr.h",
    "targets/hetero/__init__.py",
    "targets/hetero/design.py",
    "targets/hetero/dram_presets.py",
    "targets/hetero/memsys.py",
    "targets/hetero/power.py",
    "targets/hetero/soc.py",
    "targets/hetero/system.py",
    "targets/hetero/timing_cache.cpp",
    "targets/hetero/timing_cache.py",
)

_REAL_DRAM_RUN_SOURCES = (
    "targets/hetero/dram.cpp",
    "targets/hetero/dram.py",
    "targets/hetero/dram_core.hpp",
)

_APPLICATION_MAIN = {
    "mnist": "runtime/mesh/mnist_main.c",
    "kws": "runtime/mesh/kws_main.c",
}


def _resolved_run_dimensions(resolved_experiment) -> tuple[str, str, str | None]:
    if hasattr(resolved_experiment, "to_dict"):
        resolved_experiment = resolved_experiment.to_dict()
    if not isinstance(resolved_experiment, Mapping):
        raise ValueError("resolved_experiment must be an object")

    simulator = resolved_experiment.get("simulator")
    memory = resolved_experiment.get("memory")
    workload = resolved_experiment.get("workload", {})
    if not isinstance(simulator, Mapping) or not isinstance(simulator.get("target"), str):
        raise ValueError("resolved_experiment.simulator.target must be a string")
    if not isinstance(memory, Mapping) or not isinstance(memory.get("kind"), str):
        raise ValueError("resolved_experiment.memory.kind must be a string")
    if not isinstance(workload, Mapping):
        raise ValueError("resolved_experiment.workload must be an object")
    application = workload.get("application")
    if application is not None and not isinstance(application, str):
        raise ValueError("resolved_experiment.workload.application must be a string or null")
    return simulator["target"], memory["kind"], application


def run_source_files(
    root: Path | str,
    resolved_experiment,
    *,
    include_sweep: bool = False,
) -> list[Path]:
    """Return first-party sources causal to this resolved heterogeneous run."""
    root = Path(root).resolve()
    target, memory_kind, application = _resolved_run_dimensions(resolved_experiment)
    target_adapter = root / "targets" / f"{target}.py"
    if not target_adapter.is_file():
        raise FileNotFoundError(f"resolved simulator target adapter is missing: {target_adapter}")

    relative = list(_RUN_COMMON_SOURCES)
    relative.append(f"targets/{target}.py")
    relative.append(_APPLICATION_MAIN.get(application, "runtime/mesh/host_main.c"))
    if memory_kind != "fixed":
        relative.extend(_REAL_DRAM_RUN_SOURCES)
    explicit = [root / path for path in relative]
    explicit.extend(
        root / "pipeline" / "experiment" / name
        for name in _RUN_EXPERIMENT_SOURCES
    )
    if include_sweep:
        explicit.extend([
            root / "pipeline" / "sweep" / "calibrate.py",
            root / "pipeline" / "sweep" / "design.py",
            root / "pipeline" / "sweep" / "run.py",
        ])

    trees = (
        root / "pipeline" / "hetero_platform",
        root / "runtime" / "common" / "kernels",
        root / "runtime" / "snitch" / "kernels",
        root / "runtime" / "spatz" / "kernels",
    )
    paths = [path for path in explicit if path.is_file()]
    for tree in trees:
        paths.extend(source_files_under(tree))
    return sorted(set(paths))


def capture_run_provenance(
    root: Path | str,
    resolved_experiment,
    *,
    include_sweep: bool = False,
) -> dict:
    """Capture causal run-source content and descriptive repository state."""
    root = Path(root).resolve()
    target, memory_kind, _ = _resolved_run_dimensions(resolved_experiment)
    sources = run_source_files(
        root, resolved_experiment, include_sweep=include_sweep,
    )
    manifest = file_set_manifest(root, sources)
    return {
        "m4ia": git_snapshot_optional(root, name="m4ia"),
        "source_set": {
            "manifest": manifest,
            "digest": fingerprint(manifest),
            "include_sweep": include_sweep,
            "simulator_target": target,
            "memory_kind": memory_kind,
        },
    }


def _run_bytes(repo: Path, args: Sequence[str]) -> bytes:
    process = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process.stdout


def _run_text(repo: Path, args: Sequence[str]) -> str:
    return _run_bytes(repo, args).decode("utf-8", errors="strict").strip()


def _untracked_manifest(repo: Path) -> list[dict]:
    raw = _run_bytes(repo, ["ls-files", "--others", "--exclude-standard", "-z"])
    names = [item.decode("utf-8") for item in raw.split(b"\0") if item]
    records = []
    for name in sorted(names):
        path = repo / name
        if path.is_symlink():
            payload = ("symlink\0" + os.readlink(path)).encode("utf-8")
            digest = sha256_bytes(payload)
            kind = "symlink"
        elif path.is_file():
            digest = file_digest(path)
            kind = "file"
        else:
            digest = fingerprint({"kind": "other"})
            kind = "other"
        records.append({"path": Path(name).as_posix(), "kind": kind, "digest": digest})
    return records


def git_snapshot(repo: Path | str, name: str | None = None) -> dict:
    """Capture Git provenance, including dirty tracked and untracked content."""
    repo = Path(repo).resolve()
    revision = _run_text(repo, ["rev-parse", "HEAD"])
    try:
        repository = _run_text(repo, ["config", "--get", "remote.origin.url"]) or None
    except subprocess.CalledProcessError:
        repository = None
    tracked_diff = _run_bytes(repo, ["diff", "--binary", "HEAD", "--", "."])
    untracked = _untracked_manifest(repo)
    return {
        "name": name or repo.name,
        "repository": repository,
        "revision": revision,
        "dirty": bool(tracked_diff or untracked),
        "diff_digest": sha256_bytes(tracked_diff) if tracked_diff else None,
        "untracked_digest": fingerprint(untracked) if untracked else None,
    }


def git_snapshot_optional(repo: Path | str, name: str | None = None) -> dict:
    """Capture Git provenance or explicitly report absent Git metadata."""
    repo = Path(repo).resolve()
    try:
        snapshot = git_snapshot(repo, name=name)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "name": name or repo.name,
            "available": False,
            "reason": "git-metadata-unavailable",
            "repository": None,
            "revision": None,
            "dirty": None,
            "diff_digest": None,
            "untracked_digest": None,
        }
    return {"available": True, **snapshot}


def semantic_git_identity(snapshot: dict) -> dict:
    """Return causal Git fields, deliberately excluding name and remote URL."""
    if snapshot.get("available") is False:
        return {
            "available": False,
            "revision": None,
            "dirty": None,
            "diff_digest": None,
            "untracked_digest": None,
        }
    return {
        "available": True,
        "revision": snapshot.get("revision"),
        "dirty": snapshot.get("dirty"),
        "diff_digest": snapshot.get("diff_digest"),
        "untracked_digest": snapshot.get("untracked_digest"),
    }


def patch_digests(root: Path | str, paths: Iterable[Path | str]) -> list[dict]:
    """Digest patches relative to ``root``; reject paths outside that root."""
    root = Path(root).resolve()
    records = []
    for value in paths:
        path = Path(value)
        absolute = (path if path.is_absolute() else root / path).resolve()
        try:
            relative = absolute.relative_to(root)
        except ValueError as error:
            raise ValueError(f"patch is outside provenance root: {value}") from error
        records.append({"path": relative.as_posix(), "digest": file_digest(absolute)})
    return sorted(records, key=lambda item: item["path"])


def toolchain_snapshot(
    executable: Path | str,
    version_args: Sequence[str] = ("--version",),
) -> dict:
    """Capture executable basename, version output, and binary content digest."""
    executable = Path(executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"required executable is missing: {executable}")
    process = subprocess.run(
        [str(executable), *version_args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {
        "executable": executable.name,
        "version": (process.stdout or process.stderr).strip(),
        "binary_digest": file_digest(executable),
    }


def binary_snapshot(executable: Path | str) -> dict:
    """Capture a binary identity without assuming it has a version command."""
    executable = Path(executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"required executable is missing: {executable}")
    return {
        "executable": executable.name,
        "binary_digest": file_digest(executable),
    }
