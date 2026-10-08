"""Read current M4IA identities without importing heavy execution backends."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from .engine_catalog import build_catalog as build_engine_catalog
from .host_profile_catalog import build_catalog as build_host_profile_catalog
from .kernel_implementation_catalog import build_catalog as build_kernel_catalog
from .mapping_strategy_catalog import build_catalog as build_mapping_catalog
from .memory_catalog import current_memory_catalog
from .parameter_catalog import build_catalog as build_parameter_catalog


def _simple_assignments(path: Path, wanted: set[str]) -> dict:
    tree = ast.parse(path.read_text(), filename=str(path))
    env = {}

    def value(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, ast.Dict):
            return {value(key): value(item) for key, item in zip(node.keys, node.values)}
        raise ValueError(f"unsupported expression in {path}: {ast.dump(node)}")

    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            env[target.id] = value(statement.value)
    if missing := wanted - set(env):
        raise RuntimeError(f"{path} is missing expected assignments: {sorted(missing)}")
    return env


def _load_pure_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_engine_catalog(root: Path | str) -> list[dict]:
    root = Path(root)
    names = _simple_assignments(
        root / "targets" / "hetero" / "system.py",
        {"ENGINE_HOST", "ENGINE_SNITCH", "ENGINE_SPATZ", "ENGINE_NAMES"},
    )["ENGINE_NAMES"]
    macros = _simple_assignments(
        root / "pipeline" / "hetero_platform" / "progress.py", {"ENGINE_MACRO"},
    )["ENGINE_MACRO"]
    return build_engine_catalog(names, macros)


def _host_targets(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or target.id != "HOSTS":
            continue
        if not isinstance(statement.value, ast.Dict):
            raise RuntimeError("pipeline/build_mesh.py::HOSTS is no longer a dict literal")
        out = {}
        for key_node, value_node in zip(statement.value.keys, statement.value.values):
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                raise RuntimeError("HOSTS profile names must remain string literals")
            if not isinstance(value_node, ast.Tuple) or len(value_node.elts) != 2:
                raise RuntimeError("HOSTS values must remain (image, target) tuples")
            target_node = value_node.elts[1]
            if not isinstance(target_node, ast.Constant) or not isinstance(target_node.value, str):
                raise RuntimeError("HOSTS target names must remain string literals")
            out[key_node.value] = target_node.value
        return out
    raise RuntimeError("pipeline/build_mesh.py does not define HOSTS")


def current_host_profile_catalog(root: Path | str) -> dict[str, dict]:
    root = Path(root)
    rows = build_host_profile_catalog(_host_targets(root / "pipeline" / "build_mesh.py"))
    return {row["name"]: row for row in rows}


def current_parameter_catalog(
    root: Path | str,
    screening_values: dict[str, list[int]] | None = None,
) -> list[dict]:
    root = Path(root)
    design = _load_pure_module(
        root / "pipeline" / "sweep" / "design.py", "m4ia_experiment_design",
    )
    return build_parameter_catalog(design.DEFAULTS, design.BUILD_TIME, screening_values)


def current_kernel_implementation_catalog(root: Path | str) -> list[dict]:
    return build_kernel_catalog(root)


def current_experiment_schema(root: Path | str) -> dict:
    """Machine-readable query surface; it does not execute an experiment."""
    root = Path(root)
    return {
        "parameters": current_parameter_catalog(root),
        "engines": current_engine_catalog(root),
        "host_profiles": list(current_host_profile_catalog(root).values()),
        "mapping_strategies": build_mapping_catalog(),
        "kernel_implementations": current_kernel_implementation_catalog(root),
        "main_memories": current_memory_catalog(root),
        "presets": ["m4ia_current"],
    }
