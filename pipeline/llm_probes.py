#!/usr/bin/env python3
"""Op probes for the planned LLM application, at the shapes it will use.

  python pipeline/llm_probes.py              # writes ops/llm_probe_*/
  python pipeline/run_hetero.py ops/llm_probe_layernorm

The LLM decode step is a graph this pipeline has never compiled: MNIST and KWS
are Conv/Gemm/Relu/MaxPool, while a transformer block also needs
LayerNormalization, Gelu and a masked Softmax on the host, plus M=1 MatMuls
chained through the clusters. Deeploy's Generic platform lists fp32 bindings
for all of them, but only Softmax has ever run here (results/). So before any
training code is written, each piece is checked on its own, then together:

  layernorm    LayerNormalization over d_model             host only
  gelu         Gelu over the FFN hidden width              host only
  softmax      scores + additive mask -> Softmax           host only; the mask
               value is chosen non-integer, see MASK_OFF
  head         one attention head of one decode step:      M=1 MatMuls
               q = x Wq; p = softmax(q Kt / 4 + mask);     chained through the
               o = (p V) Wo                                clusters and host
  ffn          x W1 + b1 -> Relu -> W2 + b2                W1 is 64 KiB: the
                                                           largest job the
                                                           model will stage
  block        one whole pre-LN decoder layer, 4 heads,    the integration
               emitting the new k_t / v_t per head         probe

Geometry is the planned model's: d_model 64, 4 heads of 16, FFN 256, a
64-token context. Weights are random but scaled so activations stay O(1); the
point is numerical agreement with onnxruntime, not a trained model. No Concat
or Slice anywhere -- neither has an fp32 binding in the pinned Deeploy -- so
heads are separate weight matrices and their outputs are summed through Wo.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent

D_MODEL = 64
N_HEAD = 4
D_HEAD = D_MODEL // N_HEAD
D_FFN = 4 * D_MODEL
CTX = 64
# Filled cache slots in the probe. Not the whole context, so the mask has both
# values in it and a wrong mask shows up as a numerical error.
FILLED = 37
# Deeploy types each network input from the *values* of its sample
# (testUtils.typeMapping.inferMinimalType), and generate.is_fp32_network turns
# every cluster engine off if any input comes out non-float. A 0 / -1e9 mask is
# all integers, so it was typed as one and the mask Add had no binding. Any
# large negative works for the softmax -- exp(-30000) is 0 in fp32 either way
# -- so pick one with a fractional part. The C driver must keep using it.
MASK_OFF = -30000.5

# LayerNormalization is opset 17 and Gelu opset 20; MNIST/KWS export at 13.
OPSET = 20


class Graph:
    """Collects nodes, initializers and I/O for one probe graph."""

    def __init__(self, rng):
        self.rng = rng
        self.nodes = []
        self.inits = []
        self.inputs = []
        self.outputs = []
        self.feed = {}
        self._n = 0

    def _name(self, stem):
        self._n += 1
        return f"{stem}_{self._n}"

    def input(self, name, value):
        value = np.asarray(value, dtype = np.float32)
        self.inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT,
                                                         list(value.shape)))
        self.feed[name] = value
        return name

    def output(self, name, shape):
        self.outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT,
                                                          list(shape)))

    def const(self, stem, value):
        name = self._name(stem)
        self.inits.append(numpy_helper.from_array(np.asarray(value, dtype = np.float32),
                                                  name))
        return name

    def weight(self, stem, rows, cols):
        w = self.rng.standard_normal((rows, cols)) / np.sqrt(rows)
        return self.const(stem, w)

    def op(self, op_type, inputs, stem = None, out = None, **attrs):
        out = out or self._name(stem or op_type.lower())
        self.nodes.append(helper.make_node(op_type, inputs, [out],
                                           name = self._name(op_type), **attrs))
        return out

    # --- the transformer pieces -------------------------------------------

    def layernorm(self, x, width):
        gamma = self.const("ln_g", 1.0 + 0.1 * self.rng.standard_normal(width))
        beta = self.const("ln_b", 0.1 * self.rng.standard_normal(width))
        return self.op("LayerNormalization", [x, gamma, beta], axis = -1,
                       epsilon = 1e-5)

    def head(self, x, kt, v, mask, out_k = None, out_v = None):
        q = self.op("MatMul", [x, self.weight("wq", D_MODEL, D_HEAD)])
        if out_k:
            self.op("MatMul", [x, self.weight("wk", D_MODEL, D_HEAD)], out = out_k)
            self.op("MatMul", [x, self.weight("wv", D_MODEL, D_HEAD)], out = out_v)
        s = self.op("MatMul", [q, kt])
        s = self.op("Mul", [s, self.const("scale", np.full((1, CTX),
                                                           1.0 / np.sqrt(D_HEAD)))])
        s = self.op("Add", [s, mask])
        p = self.op("Softmax", [s], axis = -1)
        o = self.op("MatMul", [p, v])
        return self.op("MatMul", [o, self.weight("wo", D_HEAD, D_MODEL)])

    def gelu(self, x):
        # Deeploy's GELU_fp32_fp32 is the tanh form. ONNX's default is the erf
        # form, and the two differ by up to 4.7e-4 -- enough to fail the run.
        return self.op("Gelu", [x], approximate = "tanh")

    def ffn(self, x):
        h = self.op("MatMul", [x, self.weight("w1", D_MODEL, D_FFN)])
        h = self.op("Add", [h, self.const("b1", 0.1 * self.rng.standard_normal((1, D_FFN)))])
        # Relu, as pipeline/llm.py uses: the gelu probe measured Gelu at ~72k
        # cycles for 256 elements on the host, 40% of a whole layer.
        h = self.op("Relu", [h])
        y = self.op("MatMul", [h, self.weight("w2", D_FFN, D_MODEL)])
        return self.op("Add", [y, self.const("b2",
                                             0.1 * self.rng.standard_normal((1, D_MODEL)))])

    def model(self):
        graph = helper.make_graph(self.nodes, "llm_probe", self.inputs, self.outputs,
                                  self.inits)
        model = helper.make_model(graph, producer_name = "m4ia-llm-probe",
                                  opset_imports = [helper.make_opsetid("", OPSET)])
        model.ir_version = 9
        # Deeploy's lowering passes, and the mapper's MAC count, read every
        # intermediate tensor's shape; without this no node past the first is
        # priced and the deployer stops with "Missing engine color".
        model = onnx.shape_inference.infer_shapes(model, strict_mode = True)
        missing = [v.name for v in model.graph.value_info
                   if not v.type.tensor_type.HasField("shape")]
        if missing:
            sys.exit(f"error: shape inference left {missing} without a shape")
        onnx.checker.check_model(model, full_check = True)
        return model


def _mask():
    m = np.full((1, CTX), MASK_OFF, dtype = np.float32)
    m[0, :FILLED] = 0.0
    return m


def _cache(rng):
    """One head's cache: K transposed (d_head x ctx), V (ctx x d_head)."""
    kt = rng.standard_normal((D_HEAD, CTX)).astype(np.float32)
    v = rng.standard_normal((CTX, D_HEAD)).astype(np.float32)
    # Unfilled slots are zero, as the C driver will leave them.
    kt[:, FILLED:] = 0.0
    v[FILLED:, :] = 0.0
    return kt, v


def _x(rng, width = D_MODEL):
    return rng.standard_normal((1, width)).astype(np.float32)


def probe_layernorm(g):
    x = g.input("x", _x(g.rng))
    g.output(g.layernorm(x, D_MODEL), (1, D_MODEL))


def probe_gelu(g):
    x = g.input("x", 2.0 * _x(g.rng, D_FFN))
    g.output(g.gelu(x), (1, D_FFN))


def probe_softmax(g):
    s = g.input("scores", 3.0 * _x(g.rng, CTX))
    mask = g.input("mask", _mask())
    g.output(g.op("Softmax", [g.op("Add", [s, mask])], axis = -1), (1, CTX))


def probe_head(g):
    x = g.input("x", _x(g.rng))
    kt, v = _cache(g.rng)
    kt, v = g.input("kt", kt), g.input("v", v)
    mask = g.input("mask", _mask())
    g.output(g.head(x, kt, v, mask), (1, D_MODEL))


def probe_ffn(g):
    x = g.input("x", _x(g.rng))
    g.output(g.ffn(x), (1, D_MODEL))


def probe_block(g):
    """One pre-LN decoder layer for one token, as the decode graph will hold it."""
    x = g.input("x", _x(g.rng))
    caches = [_cache(g.rng) for _ in range(N_HEAD)]
    kts = [g.input(f"kt{h}", c[0]) for h, c in enumerate(caches)]
    vs = [g.input(f"v{h}", c[1]) for h, c in enumerate(caches)]
    mask = g.input("mask", _mask())

    # The heads are accumulated straight into the residual, each Add taking the
    # head's (MatMul) output FIRST. Deeploy's _merge_matmul_add_fun fuses
    # MatMul+Add into a Gemm and takes the bias as "add.inputs[0] if it is a
    # Constant else add.inputs[1]" -- so with a Variable on both sides and the
    # MatMul at index 1 it makes the MatMul's own output the bias, computing
    # 2*h instead of h + rest. Order is the only thing that keeps it correct.
    a = g.layernorm(x, D_MODEL)
    for h in range(N_HEAD):
        head = g.head(a, kts[h], vs[h], mask, out_k = f"k_new{h}", out_v = f"v_new{h}")
        x = g.op("Add", [head, x])
    y = g.op("Add", [g.ffn(g.layernorm(x, D_MODEL)), x], out = "y")

    g.output(y, (1, D_MODEL))
    for h in range(N_HEAD):
        g.output(f"k_new{h}", (1, D_HEAD))
        g.output(f"v_new{h}", (1, D_HEAD))


PROBES = {
    "layernorm": probe_layernorm,
    "gelu": probe_gelu,
    "softmax": probe_softmax,
    "head": probe_head,
    "ffn": probe_ffn,
    "block": probe_block,
}


def write(name, build, seed):
    g = Graph(np.random.default_rng(seed))
    build(g)
    model = g.model()

    out = ROOT / "ops" / f"llm_probe_{name}"
    out.mkdir(parents = True, exist_ok = True)
    onnx.save(model, out / "network.onnx")

    sess = ort.InferenceSession(model.SerializeToString(),
                                providers = ["CPUExecutionProvider"])
    feed = {i.name: g.feed[i.name] for i in sess.get_inputs()}
    results = sess.run(None, feed)
    for r in results:
        if not np.all(np.isfinite(r)):
            sys.exit(f"{name}: onnxruntime produced non-finite outputs")

    # Same layout make_op.py writes: arrays in graph input/output order.
    np.savez(out / "inputs.npz", **{f"input_{i}": v for i, v in enumerate(feed.values())})
    np.savez(out / "outputs.npz", **{f"output_{i}": v for i, v in enumerate(results)})
    ops = sorted({n.op_type for n in model.graph.node})
    print(f"{out.relative_to(ROOT)}: {len(model.graph.node)} nodes {ops}")


def main():
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probes", nargs = "*",
                    help = f"which probes to write, of {list(PROBES)} (default: all)")
    ap.add_argument("--seed", type = int, default = 0)
    args = ap.parse_args()
    unknown = set(args.probes) - set(PROBES)
    if unknown:
        ap.error(f"unknown probes: {sorted(unknown)}")
    for name in args.probes or PROBES:
        write(name, PROBES[name], args.seed)


if __name__ == "__main__":
    main()
