# SPDX-License-Identifier: Apache-2.0
"""Pre-run machine and artifact identities for M4IA experiments."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .fingerprint import fingerprint


SCHEMA_VERSION = 1
MACHINE_KIND = "m4ia.machine"
RUN_INPUT_KIND = "m4ia.run_input"


def _mapping(value: Any, field: str) -> dict:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _short(value: str, length: int = 12) -> str:
    return value.split(":", 1)[-1][:length]


def _component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return cleaned or "artifact"


def build_machine_context(resolved_experiment: Any) -> dict:
    """Build workload-independent numeric-design + host + memory identity."""
    resolved = _mapping(resolved_experiment, "resolved_experiment")
    hardware = _mapping(resolved.get("hardware"), "resolved_experiment.hardware")
    memory = _mapping(resolved.get("memory"), "resolved_experiment.memory")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "kind": MACHINE_KIND,
        "platform": _nonempty_string(resolved.get("platform"), "platform"),
        "design_slug": _nonempty_string(hardware.get("design_slug"), "design_slug"),
        "design_fingerprint": _nonempty_string(
            hardware.get("fingerprint"), "hardware.fingerprint",
        ),
        "host_profile": _nonempty_string(
            resolved.get("host_profile"), "host_profile",
        ),
        "memory": {
            "kind": _nonempty_string(memory.get("kind"), "memory.kind"),
            "fingerprint": _nonempty_string(
                memory.get("fingerprint"), "memory.fingerprint",
            ),
            "preset_fingerprint": memory.get("preset_fingerprint"),
            "overrides": _mapping(memory.get("overrides", {}), "memory.overrides"),
        },
    }
    machine_fingerprint = fingerprint(identity)
    key = "__".join((
        _component(identity["design_slug"]),
        _component(identity["host_profile"]),
        f"{_component(identity['memory']['kind'])}-{_short(machine_fingerprint)}",
    ))
    return {
        "identity": identity,
        "machine_fingerprint": machine_fingerprint,
        "machine_key": key,
    }


def build_run_input_context(
    resolved_experiment: Any,
    *,
    calibration_input_fingerprint: str,
    run_source_set_digest: str,
    execution_controls: Mapping[str, object],
    workload_label: str,
) -> dict:
    """Build the stable pre-execution identity used to address one cell."""
    resolved = _mapping(resolved_experiment, "resolved_experiment")
    resolved_fingerprint = _nonempty_string(
        resolved.get("resolved_fingerprint"), "resolved_fingerprint",
    )
    calibration_input_fingerprint = _nonempty_string(
        calibration_input_fingerprint, "calibration_input_fingerprint",
    )
    run_source_set_digest = _nonempty_string(
        run_source_set_digest, "run_source_set_digest",
    )
    controls = _mapping(execution_controls, "execution_controls")
    images = controls.get("images")
    if not isinstance(images, int) or isinstance(images, bool) or images <= 0:
        raise ValueError("execution_controls.images must be a positive integer")

    machine = build_machine_context(resolved)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "kind": RUN_INPUT_KIND,
        "resolved_experiment_fingerprint": resolved_fingerprint,
        "machine_fingerprint": machine["machine_fingerprint"],
        "calibration_input_fingerprint": calibration_input_fingerprint,
        "run_source_set_digest": run_source_set_digest,
        "execution_controls": controls,
    }
    run_input_fingerprint = fingerprint(identity)
    artifact_key = "__".join((
        _component(machine["identity"]["design_slug"]),
        f"{_component(workload_label)}-{_short(run_input_fingerprint)}",
    ))
    return {
        "identity": identity,
        "run_input_fingerprint": run_input_fingerprint,
        "artifact_key": artifact_key,
        "machine": machine,
    }
