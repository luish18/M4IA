"""Static ONNX workload inspection shared by experiment discovery."""

from __future__ import annotations

from pathlib import Path

from .fingerprint import file_digest, file_set_digest, file_set_manifest
from .schema import WorkloadSpec


def _product(shape):
    if shape is None:
        return None
    result = 1
    for dimension in shape:
        if not isinstance(dimension, int) or dimension <= 0:
            return None
        result *= dimension
    return result


def resolve_workload_path(path: Path | str, roots=()) -> Path:
    """Resolve a workload spelling using explicit search roots, without mutation."""
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path]
    candidates.extend(Path(root) / path for root in roots)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def inspect_workload(model_path: Path | str, application: str | None = None) -> WorkloadSpec:
    # Optional dependencies stay local so catalog/schema discovery remains
    # usable in a checkout whose full runtime environment is not installed.
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model_path = Path(model_path)
    if model_path.is_dir():
        model_path = model_path / "network.onnx"
    if not model_path.is_file():
        raise FileNotFoundError(f"workload model not found: {model_path}")

    package = model_path.parent
    artifact_paths = [model_path.name]
    for name in ("inputs.npz", "outputs.npz"):
        if (package / name).is_file():
            artifact_paths.append(name)
    try:
        from pipeline.common import detect_app
        _detected, app = detect_app(package)
    except ImportError:
        app = None
    if app is not None and (package / app["header"]).is_file():
        artifact_paths.append(app["header"])
    artifacts = file_set_manifest(package, artifact_paths)

    model = onnx.load(str(model_path))
    original_graph = model.graph
    try:
        graph = onnx.shape_inference.infer_shapes(model).graph
    except Exception:
        graph = original_graph

    initializer_names = {initializer.name for initializer in original_graph.initializer}

    def vi_tensor(value_info):
        tensor_type = value_info.type.tensor_type
        dimensions = [
            dimension.dim_value if dimension.HasField("dim_value") else (dimension.dim_param or "?")
            for dimension in tensor_type.shape.dim
        ]
        return {
            "name": value_info.name,
            "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type).lower(),
            "shape": dimensions,
        }

    inputs = tuple(
        vi_tensor(value_info)
        for value_info in original_graph.input
        if value_info.name not in initializer_names
    )
    outputs = tuple(vi_tensor(value_info) for value_info in original_graph.output)
    op_counts = {}
    for node in original_graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    tensor_meta = {}

    def register_vi(value_info):
        tensor_type = value_info.type.tensor_type
        shape = []
        for dimension in tensor_type.shape.dim:
            shape.append(
                int(dimension.dim_value)
                if dimension.HasField("dim_value") and dimension.dim_value > 0 else None
            )
        tensor_meta[value_info.name] = (tuple(shape), int(tensor_type.elem_type))

    for sequence in (graph.input, graph.value_info, graph.output):
        for value_info in sequence:
            register_vi(value_info)
    for initializer in original_graph.initializer:
        tensor_meta[initializer.name] = (
            tuple(int(dimension) for dimension in initializer.dims),
            int(initializer.data_type),
        )

    def tensor_bytes(name):
        meta = tensor_meta.get(name)
        if meta is None:
            return None
        shape, element_type = meta
        elements = _product(shape)
        if elements is None:
            return None
        try:
            dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(element_type))
        except Exception:
            return None
        return int(elements * dtype.itemsize)

    known_tensor_bytes = {}
    unknown_tensors = []
    for name in sorted(tensor_meta):
        size = tensor_bytes(name)
        if size is None:
            unknown_tensors.append(name)
        else:
            known_tensor_bytes[name] = size

    working_sets = {}
    unknown_working_set_nodes = []
    for index, node in enumerate(original_graph.node):
        names = {name for name in (*node.input, *node.output) if name}
        sizes = [tensor_bytes(name) for name in names]
        if any(size is None for size in sizes):
            unknown_working_set_nodes.append(index)
        else:
            working_sets[index] = sum(sizes)
    maximum_working_set = max(working_sets.values()) if working_sets else None

    def shape_of(name):
        meta = tensor_meta.get(name)
        if meta is None:
            return None
        shape, _ = meta
        return shape if _product(shape) is not None else None

    covered_ops = {"MatMul", "Gemm", "Conv"}
    mac_total = 0
    mac_known_nodes = []
    mac_unknown_nodes = []
    for index, node in enumerate(original_graph.node):
        if node.op_type not in covered_ops:
            continue
        estimate = None
        output_shape = shape_of(node.output[0]) if node.output else None
        output_elements = _product(output_shape)
        if node.op_type == "MatMul" and len(node.input) >= 2 and output_elements is not None:
            lhs = shape_of(node.input[0])
            if lhs is not None and len(lhs) >= 1:
                estimate = output_elements * lhs[-1]
        elif node.op_type == "Gemm" and len(node.input) >= 2 and output_elements is not None:
            lhs = shape_of(node.input[0])
            attributes = {
                attribute.name: onnx.helper.get_attribute_value(attribute)
                for attribute in node.attribute
            }
            transpose_lhs = int(attributes.get("transA", 0))
            if lhs is not None and len(lhs) >= 2:
                estimate = output_elements * (lhs[-2] if transpose_lhs else lhs[-1])
        elif node.op_type == "Conv" and len(node.input) >= 2 and output_elements is not None:
            weights = shape_of(node.input[1])
            if weights is not None and len(weights) >= 2:
                taps = _product(weights[1:])
                if taps is not None:
                    estimate = output_elements * taps
        if estimate is None:
            mac_unknown_nodes.append(index)
        else:
            mac_total += int(estimate)
            mac_known_nodes.append(index)

    initializer_elements = 0
    initializer_zeros = 0
    for initializer in original_graph.initializer:
        array = numpy_helper.to_array(initializer)
        initializer_elements += int(array.size)
        initializer_zeros += int(np.count_nonzero(array == 0))

    static_known_bytes = sum(known_tensor_bytes.values())
    static_complete = not unknown_tensors
    mac_complete = not mac_unknown_nodes
    theoretical_ai = None
    if static_complete and mac_complete and static_known_bytes > 0 and mac_known_nodes:
        theoretical_ai = mac_total / static_known_bytes

    descriptors = {
        "estimated_macs": {
            "value": mac_total,
            "covered_ops": sorted(covered_ops),
            "known_nodes": mac_known_nodes,
            "unknown_nodes": mac_unknown_nodes,
            "complete_for_covered_ops": mac_complete,
            "method": "static-shape formulas; unsupported ops are not assigned zero",
        },
        "static_tensor_bytes": {
            "known_bytes": static_known_bytes,
            "known_tensors": len(known_tensor_bytes),
            "unknown_tensors": unknown_tensors,
            "complete": static_complete,
            "method": "unique graph tensors counted once",
        },
        "working_set": {
            "max_static_node_bytes": maximum_working_set,
            "known_nodes": sorted(working_sets),
            "unknown_nodes": unknown_working_set_nodes,
            "method": "unique node inputs+outputs; not measured memory traffic",
        },
        "theoretical_arithmetic_intensity": {
            "mac_per_byte": theoretical_ai,
            "method": "estimated covered-op MACs / unique static graph tensor bytes",
            "is_measured": False,
        },
        "initializer_sparsity": {
            "zero_fraction": (
                initializer_zeros / initializer_elements if initializer_elements else None
            ),
            "zero_elements": initializer_zeros,
            "elements": initializer_elements,
            "scope": "ONNX initializers only; no claim of hardware sparsity support",
        },
    }

    return WorkloadSpec(
        path=str(model_path),
        content_digest=file_digest(model_path),
        workload_fingerprint=file_set_digest(package, artifact_paths),
        artifact_digests=tuple(artifacts),
        application=application,
        opset=tuple(
            {"domain": opset.domain or "ai.onnx", "version": opset.version}
            for opset in model.opset_import
        ),
        inputs=inputs,
        outputs=outputs,
        node_count=len(original_graph.node),
        op_counts=tuple(sorted(op_counts.items())),
        initializer_count=len(initializer_names),
        descriptors=descriptors,
    )


def workload_to_legacy_inspect(spec: WorkloadSpec) -> dict:
    result = {
        "path": spec.path,
        "opset": [dict(item) for item in spec.opset],
        "inputs": [dict(item) for item in spec.inputs],
        "outputs": [dict(item) for item in spec.outputs],
        "nodes": spec.node_count,
        "op_types": dict(sorted(spec.op_counts, key=lambda item: -item[1])),
        "initializers": spec.initializer_count,
    }
    if spec.application is not None:
        result["app"] = spec.application
    return result
