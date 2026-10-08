# SPDX-License-Identifier: Apache-2.0
"""First-class M4IA run manifests built from authoritative artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .calibration_artifact import KIND as CALIBRATION_KIND
from .calibration_artifact import SCHEMA_VERSION as CALIBRATION_SCHEMA_VERSION
from .fingerprint import file_digest, fingerprint


SCHEMA_VERSION = 1
KIND = "m4ia.run"


def _read_object(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} {path} must contain a JSON object")
    return value


def _object(value, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{description} is missing or is not an object")
    return value


def _relative(base: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.name


def _actual_projection(result: Mapping[str, object]) -> dict:
    nodes = result.get("nodes")
    nodes = nodes if isinstance(nodes, list) else []
    actual_nodes = []
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        actual = {
            key: node[key]
            for key in ("node", "op", "engine")
            if key in node
        }
        if "implementation" in node:
            actual["implementation"] = node["implementation"]
        actual_nodes.append(actual)
    return {"nodes": actual_nodes}


def _measured_projection(result: Mapping[str, object]) -> dict:
    measured = dict(result)
    nodes = result.get("nodes")
    nodes = nodes if isinstance(nodes, list) else []
    measured["nodes"] = [
        {key: value for key, value in node.items() if key != "implementation"}
        for node in nodes
        if isinstance(node, Mapping)
    ]
    return measured


def build_run_manifest(
    *,
    requested: Mapping[str, object],
    resolved: Mapping[str, object],
    run_input: Mapping[str, object],
    artifact_key: str,
    calibration_path: Path | str,
    result_path: Path | str,
    run_provenance: Mapping[str, object],
    base_dir: Path | str | None = None,
) -> dict:
    """Build requested/resolved/generated/actual/measured run evidence."""
    if not isinstance(requested, Mapping) or not isinstance(resolved, Mapping):
        raise TypeError("requested and resolved must be objects")
    if not isinstance(run_input, Mapping):
        raise TypeError("run_input must be an object")
    if not isinstance(run_provenance, Mapping):
        raise TypeError("run_provenance must be an object")
    if not isinstance(artifact_key, str) or not artifact_key:
        raise ValueError("artifact_key must be a non-empty string")

    run_input_identity = run_input.get("identity")
    run_input_fingerprint = run_input.get("run_input_fingerprint")
    if not isinstance(run_input_identity, Mapping):
        raise RuntimeError("run input identity is missing")
    if not isinstance(run_input_fingerprint, str) or not run_input_fingerprint:
        raise RuntimeError("run input fingerprint is missing")
    if fingerprint(run_input_identity) != run_input_fingerprint:
        raise RuntimeError("run input fingerprint does not match its identity")
    recorded_artifact_key = run_input.get("artifact_key")
    if recorded_artifact_key is not None and recorded_artifact_key != artifact_key:
        raise RuntimeError("artifact key does not match run input addressing")

    resolved_experiment = _object(
        resolved.get("experiment"), "resolved experiment",
    )
    resolved_machine = _object(resolved.get("machine"), "resolved machine")
    if resolved_experiment.get("resolved_fingerprint") != run_input_identity.get(
        "resolved_experiment_fingerprint"
    ):
        raise RuntimeError("resolved experiment does not match run input identity")
    machine_fingerprint = resolved_machine.get("machine_fingerprint")
    if machine_fingerprint != run_input_identity.get("machine_fingerprint"):
        raise RuntimeError("resolved machine does not match run input identity")
    machine_identity = _object(
        resolved_machine.get("identity"), "resolved machine identity",
    )
    if fingerprint(machine_identity) != machine_fingerprint:
        raise RuntimeError("resolved machine fingerprint does not match its identity")

    resolved_memory = _object(
        resolved_experiment.get("memory"), "resolved experiment memory",
    )
    resolved_host = resolved_experiment.get("host_profile")
    if not isinstance(resolved_host, str) or not resolved_host:
        raise RuntimeError("resolved experiment host_profile is missing")
    resolved_memory_kind = resolved_memory.get("kind")
    if not isinstance(resolved_memory_kind, str) or not resolved_memory_kind:
        raise RuntimeError("resolved experiment memory.kind is missing")

    source_set = run_provenance.get("source_set")
    if not isinstance(source_set, Mapping) or not isinstance(source_set.get("digest"), str):
        raise RuntimeError("run provenance source-set digest is missing")
    if source_set["digest"] != run_input_identity.get("run_source_set_digest"):
        raise RuntimeError("run provenance does not match run input identity")

    calibration_path = Path(calibration_path)
    result_path = Path(result_path)
    calibration = _read_object(calibration_path, "calibration metadata")
    if calibration.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise RuntimeError("calibration metadata schema does not match")
    if calibration.get("kind") != CALIBRATION_KIND:
        raise RuntimeError(
            f"unexpected calibration artifact kind: {calibration.get('kind')!r}"
        )
    calibration_fingerprint = calibration.get("input_fingerprint")
    if not isinstance(calibration_fingerprint, str) or not calibration_fingerprint:
        raise RuntimeError("calibration artifact has no input_fingerprint")
    if calibration_fingerprint != run_input_identity.get("calibration_input_fingerprint"):
        raise RuntimeError("calibration artifact does not match run input identity")

    result_document = _read_object(result_path, "result artifact")
    generated = result_document.get("mapping")
    result = result_document.get("result")
    if not isinstance(generated, dict) or not isinstance(result, dict):
        raise RuntimeError("result artifact must contain mapping and result objects")
    if "host" in generated and generated["host"] != resolved_host:
        raise RuntimeError("runtime mapping host contradicts resolved host profile")
    if "dram" in generated and generated["dram"] != resolved_memory_kind:
        raise RuntimeError("runtime mapping memory contradicts resolved memory kind")

    actual = _actual_projection(result)
    measured = _measured_projection(result)
    calibration_digest = file_digest(calibration_path)
    result_digest = file_digest(result_path)
    completed_identity = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "run_input_fingerprint": run_input_fingerprint,
        "generated": generated,
        "actual": actual,
        "measured": measured,
        "calibration_metadata_digest": calibration_digest,
        "result_artifact_digest": result_digest,
        "run_source_set_digest": source_set["digest"],
    }

    base = Path(base_dir) if base_dir is not None else result_path.parent
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "artifact_key": artifact_key,
        "run_input_fingerprint": run_input_fingerprint,
        "run_fingerprint": fingerprint(completed_identity),
        "requested": dict(requested),
        "resolved": dict(resolved),
        "generated": generated,
        "actual": actual,
        "measured": measured,
        "run_input": dict(run_input_identity),
        "calibration": {
            "path": _relative(base, calibration_path),
            "input_fingerprint": calibration_fingerprint,
            "metadata_digest": calibration_digest,
            "provenance": calibration.get("provenance", {}),
        },
        "run_provenance": dict(run_provenance),
        "artifacts": {
            "result": {
                "path": _relative(base, result_path),
                "digest": result_digest,
            },
        },
    }


def write_run_manifest(path: Path | str, **kwargs) -> dict:
    """Build and write a manifest only after linked artifacts validate."""
    path = Path(path)
    manifest = build_run_manifest(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
