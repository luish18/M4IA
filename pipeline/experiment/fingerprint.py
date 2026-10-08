"""Deterministic identity primitives for M4IA experiment artifacts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

SHA256_PREFIX = "sha256:"


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON mappings require string keys")
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        items = [_jsonable(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported value for canonical JSON: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return SHA256_PREFIX + hashlib.sha256(data).hexdigest()


def fingerprint(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def file_digest(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return SHA256_PREFIX + digest.hexdigest()


def file_set_manifest(root: Path | str, paths: Iterable[Path | str]) -> list[dict]:
    root = Path(root).resolve()
    records = []
    for path in paths:
        path = Path(path)
        absolute = path if path.is_absolute() else root / path
        absolute = absolute.resolve()
        try:
            relative = absolute.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path is outside fingerprint root: {path}") from exc
        if not absolute.is_file():
            raise FileNotFoundError(absolute)
        records.append({"path": relative.as_posix(), "digest": file_digest(absolute)})
    records.sort(key=lambda item: item["path"])
    return records


def file_set_digest(root: Path | str, paths: Iterable[Path | str]) -> str:
    return fingerprint(file_set_manifest(root, paths))
