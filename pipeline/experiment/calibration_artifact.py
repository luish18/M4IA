# SPDX-License-Identifier: Apache-2.0
"""Identity, provenance, and validation for M4IA calibration artifacts.

The sweep runner uses this module to decide whether ``rates.json`` is reusable
and to write its validated sidecar only after successful calibration.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .fingerprint import file_digest, file_set_manifest, fingerprint
from .provenance import (
    binary_snapshot,
    git_snapshot,
    git_snapshot_optional,
    patch_digests,
    semantic_git_identity,
    source_files_under,
    toolchain_snapshot,
)


SCHEMA_VERSION = 1
KIND = "m4ia.calibration"
PROTOCOL_ID = "mesh_calib_v1"
PLATFORM_ID = "m4ia_current"

HOST_TARGETS = {
    "cva6": "hetero_soc",
    "ara": "hetero_ara",
}

# These are the first-party files actually used to build the calibration
# images, construct their target, generate their shared system definition, and
# turn the completed beacon stream into a measured rate table.  Keeping this
# list explicit prevents unrelated workloads, GUIs, reports, and documentation
# from becoming calibration inputs.
_COMMON_CALIBRATION_SOURCES = (
    "pipeline/build_mesh.py",
    "pipeline/common.py",
    "pipeline/gen_system_header.py",
    "pipeline/sweep/calibrate.py",
    "pipeline/sweep/design.py",
    # prepare()/calibrate() choose the generated mesh copy, calibration and
    # cluster programs, host glue, target, ELF environment, and parser command.
    # Those choices can change what is measured, so this is more than a cache-
    # path wrapper even though the rest of this large file is orchestration.
    "pipeline/sweep/run.py",
    "runtime/tests/mesh_calib.c",
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
    "targets/hetero/design.py",
    "targets/hetero/dram_presets.py",
    "targets/hetero/memsys.py",
    "targets/hetero/soc.py",
    "targets/hetero/system.py",
    "targets/hetero/timing_cache.cpp",
    "targets/hetero/timing_cache.py",
    "targets/hetero_ara.py",
    "targets/hetero_soc.py",
)

_REAL_DRAM_SOURCES = (
    "targets/hetero/dram.cpp",
    "targets/hetero/dram.py",
    "targets/hetero/dram_core.hpp",
)

_CALIBRATION_KERNEL_TREES = (
    "runtime/common/kernels",
    "runtime/snitch/kernels",
    "runtime/spatz/kernels",
)


def calibration_source_files(root: Path | str, memory_kind: str) -> list[Path]:
    """Return the audited first-party sources causal to one calibration.

    The real-DRAM model is included only when it is instantiated.  The timing
    cache is causal for both fixed and real memory.  Kernel directories are
    included because ``build_mesh.py`` compiles those sets into calibration
    images.  Missing expected files fail explicitly instead of silently
    weakening the identity.
    """
    root = Path(root).resolve()
    if not isinstance(memory_kind, str) or not memory_kind:
        raise ValueError("memory_kind must be a non-empty string")

    relative_paths = list(_COMMON_CALIBRATION_SOURCES)
    if memory_kind != "fixed":
        relative_paths.extend(_REAL_DRAM_SOURCES)

    paths = [root / relative for relative in relative_paths]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        rendered = ", ".join(path.relative_to(root).as_posix() for path in missing)
        raise FileNotFoundError(f"required calibration source is missing: {rendered}")

    for relative in _CALIBRATION_KERNEL_TREES:
        tree = root / relative
        if not tree.is_dir():
            raise FileNotFoundError(f"required calibration source tree is missing: {tree}")
        paths.extend(source_files_under(tree))

    return sorted(set(paths))


def _as_mapping(value: Any, field: str) -> dict:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _resolved_machine(resolved_experiment: Any) -> dict:
    resolved = _as_mapping(resolved_experiment, "resolved_experiment")
    if resolved.get("platform") != PLATFORM_ID:
        raise ValueError(f"calibration platform must be {PLATFORM_ID!r}")

    host = resolved.get("host_profile")
    if host not in HOST_TARGETS:
        raise ValueError(f"unsupported calibration host profile: {host!r}")

    simulator = _as_mapping(resolved.get("simulator"), "simulator")
    target = simulator.get("target")
    if target != HOST_TARGETS[host]:
        raise ValueError(
            f"host profile {host!r} requires simulator target {HOST_TARGETS[host]!r}, "
            f"not {target!r}"
        )

    hardware = _as_mapping(resolved.get("hardware"), "hardware")
    design = _as_mapping(hardware.get("design"), "hardware.design")
    for key, value in design.items():
        if not isinstance(key, str):
            raise ValueError("hardware.design keys must be strings")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"hardware.design value {key!r} must be an integer")

    memory = _as_mapping(resolved.get("memory"), "memory")
    kind = memory.get("kind")
    memory_fingerprint = memory.get("fingerprint")
    if not isinstance(kind, str) or not kind:
        raise ValueError("resolved memory.kind must be a non-empty string")
    if not isinstance(memory_fingerprint, str) or not memory_fingerprint:
        raise ValueError("resolved memory.fingerprint must be a non-empty string")

    # Preserve explicit, accepted memory identity without copying its
    # operational parameter parser into this module.
    memory_identity = {
        "kind": kind,
        "fingerprint": memory_fingerprint,
        "preset_fingerprint": memory.get("preset_fingerprint"),
        "overrides": _as_mapping(memory.get("overrides", {}), "memory.overrides"),
    }
    return {
        "platform": PLATFORM_ID,
        "host_profile": host,
        "simulator_target": target,
        "hardware": {
            "design": design,
            "design_fingerprint": fingerprint(design),
        },
        "memory": memory_identity,
    }


def _semantic_executable_identity(snapshot: Mapping[str, object]) -> dict:
    digest = snapshot.get("binary_digest")
    if not isinstance(digest, str) or not digest:
        raise ValueError("executable snapshot requires binary_digest")
    result = {"binary_digest": digest}
    if "version" in snapshot:
        result["version"] = snapshot.get("version")
    return result


def _semantic_patch_identity(patches: list[Mapping[str, object]]) -> list[str]:
    digests = []
    for patch in patches:
        digest = patch.get("digest")
        if not isinstance(digest, str) or not digest:
            raise ValueError("patch provenance requires a digest")
        digests.append(digest)
    return sorted(digests)


def build_calibration_context(
    resolved_experiment: Any,
    *,
    source_set_digest: str,
    dependencies: Mapping[str, Mapping[str, object]],
    patches: list[Mapping[str, object]],
    toolchain: Mapping[str, object],
    simulator_binary: Mapping[str, object],
    provenance: Mapping[str, object] | None = None,
) -> dict:
    """Build a workload-independent calibration input identity.

    Values are injectable so the identity contract can be tested without a
    local GVSoC/Deeploy/toolchain installation.  Production capture remains
    strict in :func:`capture_calibration_context`.
    """
    if not isinstance(source_set_digest, str) or not source_set_digest:
        raise ValueError("source_set_digest must be a non-empty string")
    if not isinstance(dependencies, Mapping):
        raise ValueError("dependencies must be an object")

    dependency_identity = {}
    for name, snapshot in sorted(dependencies.items()):
        if not isinstance(name, str) or not isinstance(snapshot, Mapping):
            raise ValueError("dependency snapshots must be named objects")
        dependency_identity[name] = semantic_git_identity(dict(snapshot))

    input_identity = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "protocol": PROTOCOL_ID,
        **_resolved_machine(resolved_experiment),
        "calibration_source_set_digest": source_set_digest,
        "dependencies": dependency_identity,
        "patches": _semantic_patch_identity(patches),
        "toolchain": _semantic_executable_identity(toolchain),
        "simulator_binary": _semantic_executable_identity(simulator_binary),
    }
    return {
        "input_identity": input_identity,
        "input_fingerprint": fingerprint(input_identity),
        "provenance": dict(provenance or {}),
    }


def _dependency_snapshot(path: Path, name: str) -> dict:
    if not path.is_dir():
        raise FileNotFoundError(f"required dependency checkout is missing: {path}")
    return git_snapshot(path, name=name)


def capture_calibration_context(root: Path | str, resolved_experiment: Any) -> dict:
    """Strictly capture available causal inputs and descriptive provenance."""
    root = Path(root).resolve()
    machine = _resolved_machine(resolved_experiment)
    memory_kind = machine["memory"]["kind"]
    sources = calibration_source_files(root, memory_kind)
    source_manifest = file_set_manifest(root, sources)
    source_digest = fingerprint(source_manifest)

    dependency_snapshots = {
        "deeploy": _dependency_snapshot(root / "deps" / "deeploy", "deeploy"),
        "gvsoc": _dependency_snapshot(root / "deps" / "gvsoc", "gvsoc"),
        "gvsoc_core": _dependency_snapshot(
            root / "deps" / "gvsoc" / "core", "gvsoc-core"
        ),
    }
    patch_paths = sorted((root / "deps" / "patches").glob("gvsoc-core-*.patch"))
    patches = patch_digests(root, patch_paths)
    gcc = (
        root / "toolchains" / "xpack-riscv-none-elf-gcc-15.2.0-1" /
        "bin" / "riscv-none-elf-gcc"
    )
    simulator = root / "deps" / "gvsoc" / "install" / "bin" / "gvsoc"
    toolchain = toolchain_snapshot(gcc)
    simulator_identity = binary_snapshot(simulator)

    provenance = {
        "m4ia": git_snapshot_optional(root, name="m4ia"),
        "calibration_sources": {
            "manifest": source_manifest,
            "digest": source_digest,
        },
        "dependencies": dependency_snapshots,
        "patches": patches,
        "toolchain": toolchain,
        "simulator_binary": simulator_identity,
    }
    return build_calibration_context(
        resolved_experiment,
        source_set_digest=provenance["calibration_sources"]["digest"],
        dependencies=dependency_snapshots,
        patches=patches,
        toolchain=toolchain,
        simulator_binary=simulator_identity,
        provenance=provenance,
    )


def cache_status(
    rates_path: Path | str,
    metadata_path: Path | str,
    input_fingerprint: str,
) -> tuple[bool, str]:
    """Return whether a rates table and sidecar match current inputs."""
    rates_path = Path(rates_path)
    metadata_path = Path(metadata_path)
    if not rates_path.is_file():
        return False, "rates-missing"
    if not metadata_path.is_file():
        return False, "metadata-missing"

    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, ValueError):
        return False, "metadata-invalid"
    if not isinstance(metadata, dict):
        return False, "metadata-invalid"
    if metadata.get("schema_version") != SCHEMA_VERSION:
        return False, "schema-mismatch"
    if metadata.get("kind") != KIND:
        return False, "kind-mismatch"
    if metadata.get("protocol") != PROTOCOL_ID:
        return False, "protocol-mismatch"

    input_identity = metadata.get("input_identity")
    if not isinstance(input_identity, dict):
        return False, "input-identity-invalid"
    stored_fingerprint = metadata.get("input_fingerprint")
    if not isinstance(stored_fingerprint, str) or (
        fingerprint(input_identity) != stored_fingerprint
    ):
        return False, "input-identity-mismatch"
    if stored_fingerprint != input_fingerprint:
        return False, "input-mismatch"

    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict):
        return False, "rates-digest-mismatch"
    rates = artifacts.get("rates")
    if not isinstance(rates, dict):
        return False, "rates-digest-mismatch"
    if rates.get("digest") != file_digest(rates_path):
        return False, "rates-digest-mismatch"
    return True, "hit"


def _relative_artifact(base: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.name


def write_metadata(
    metadata_path: Path | str,
    context: Mapping[str, object],
    rates_path: Path | str,
    log_path: Path | str,
) -> Path:
    """Write ``calibration.json`` after both successful artifacts exist."""
    metadata_path = Path(metadata_path)
    rates_path = Path(rates_path)
    log_path = Path(log_path)

    # Digest first: a failed or incomplete calibration must not leave a valid-
    # looking sidecar behind.
    rates_digest = file_digest(rates_path)
    log_digest = file_digest(log_path)
    input_identity = context.get("input_identity")
    input_fingerprint = context.get("input_fingerprint")
    if not isinstance(input_identity, Mapping):
        raise ValueError("calibration context requires input_identity")
    if not isinstance(input_fingerprint, str) or not input_fingerprint:
        raise ValueError("calibration context requires input_fingerprint")
    if fingerprint(input_identity) != input_fingerprint:
        raise ValueError("calibration context input_fingerprint does not match input_identity")

    blob = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "protocol": PROTOCOL_ID,
        "input_fingerprint": input_fingerprint,
        "input_identity": dict(input_identity),
        "provenance": dict(context.get("provenance") or {}),
        "artifacts": {
            "rates": {
                "path": _relative_artifact(metadata_path.parent, rates_path),
                "digest": rates_digest,
            },
            "raw_log": {
                "path": _relative_artifact(metadata_path.parent, log_path),
                "digest": log_digest,
            },
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")
    return metadata_path
