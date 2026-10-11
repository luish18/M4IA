#!/usr/bin/env python3
"""A tiny decoder-only language model, generating text one token at a time.

  python pipeline/llm.py --problems 8

MNIST and KWS are classifiers: one input, one forward pass, every node the same
shape every time. An LLM is the first application here whose work changes as it
runs. It generates autoregressively -- each token is one pass of a *decode*
graph whose matrix products all have M=1 (GEMV: every weight streamed for one
row of output, no reuse) -- and each step attends over a KV cache that grows by
one position per token. That is the regime the mapper's offload cost, the
TCDM working set and the memory model have never been exercised in.

  token --[C: tok_emb + pos_emb]--> x --[decode graph]--> logits --argmax--> next
                                          |    ^
                               new k, v   v    |  K/V caches, mask, sel
                                     [C: KV cache, owned by the host]

The task is two-digit addition, generated procedurally (no download, as with
MNIST's and KWS's constraints): "07+48=" prompts the model, which answers with
the sum's three digits *least significant first* and a terminator, "550;".
Reversed digits are what make addition learnable left-to-right by a model this
small -- each digit then depends only on the digits and carry already seen.
The task gives the run a correctness check of its own (is the sum right?) next
to the one that can fail it (does the chip pick the same tokens onnxruntime
does?).

Writes ops/llm/:

  network.onnx   the decode graph Deeploy compiles: one token, all layers
  inputs.npz     one mid-sequence decode step, for single-inference codegen
  outputs.npz    what onnxruntime produces for it
  llm_data.h     embeddings, prompts, true answers and onnxruntime's greedy
                 tokens -- what runtime/mesh/llm_main.c loops over

Geometry: 2 layers, d_model 64, 4 heads of 16, ReLU FFN of 256, context 16.
The widest cluster job is W1 at 64x256 fp32 = 64 KiB, inside the 85,504-byte
TCDM budget (pipeline/hetero_platform/engines.py). Training is plain numpy with
hand-written backward passes; --gradcheck verifies them numerically.

The graph is shaped around what the pinned Deeploy can deploy in fp32 -- see
pipeline/llm_probes.py, which established each constraint:

  * no Concat / Slice (no fp32 bindings): heads are separate weight matrices,
    and their outputs reach the residual through per-head Wo slices, summed;
  * every MatMul+Add written as Add(matmul_out, other), because Deeploy's
    MatMul+Add->Gemm merge takes the wrong bias when the MatMul is second;
  * every input's codegen sample must hold a value only fp32 represents, or
    Deeploy types it as an integer or bfloat16 and the clusters are switched
    off (see MASK_OFF, SEL_OFF).

The current token's own key and value are produced *inside* the graph, so the
cache inputs cannot hold them yet. They are added in with a one-hot `sel` row:
Kt_full = k_new^T sel + Kt_cache, V_full = sel^T v_new + V_cache -- an outer
product and an add, which is all MatMul/Add. The C driver then writes k_new and
v_new into slot t for the next step.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent
OP_DIR = ROOT / "ops" / "llm"

# --- geometry ----------------------------------------------------------------

N_LAYER = 2
D_MODEL = 64
N_HEAD = 4
D_HEAD = D_MODEL // N_HEAD
D_FFN = 4 * D_MODEL
CTX = 16
LN_EPS = 1e-5

# --- the task ----------------------------------------------------------------

TOKENS = "0123456789+=;"
VOCAB = len(TOKENS)
PLUS, EQUALS, END = TOKENS.index("+"), TOKENS.index("="), TOKENS.index(";")
OPERAND_DIGITS = 2
ANSWER_DIGITS = OPERAND_DIGITS + 1
PROMPT_LEN = 2 * OPERAND_DIGITS + 2          # "07+48="
ANSWER_LEN = ANSWER_DIGITS + 1               # "550;"
SEQ_LEN = PROMPT_LEN + ANSWER_LEN            # 10
# Decode steps per problem: every token but the last is fed once. The logits of
# step t predict token t+1, so the answer comes from steps PROMPT_LEN-1 onward.
STEPS = SEQ_LEN - 1
assert SEQ_LEN <= CTX

# Deeploy types each network input from its codegen sample's *values*
# (testUtils.typeMapping.inferMinimalType): the narrowest type that holds them
# all exactly. All-integer values become an integer type, and values that all
# fit bfloat16 become bfloat16_t -- and either one makes
# hetero_platform/generate.is_fp32_network switch every cluster off. So each
# input's sample needs at least one value only fp32 represents exactly;
# check_fp32_typing() enforces it before anything is written.
#
# Additive mask value for cache slots the current token may not see. Any large
# negative zeroes the softmax (exp(-30000) underflows in fp32); this one has a
# fractional part bfloat16 cannot hold.
MASK_OFF = -30000.5
# What `sel` holds in the slots *after* the current one. Those slots are
# masked out of the scores, and their softmax weight is exactly zero, so
# whatever the outer product writes there is never read -- which leaves them
# free to carry the value that types `sel` as fp32 (0.5 did not: 0, 1 and 0.5
# all fit bfloat16). Slots before the current one hold real cache entries and
# must be exactly 0.
SEL_OFF = 0.1


def check_fp32_typing(name, values):
    """Exit unless Deeploy will type this sample as fp32 (see MASK_OFF)."""
    v = np.asarray(values, dtype = np.float32).reshape(-1)
    as_bf16 = (v.view(np.uint32) & 0xFFFF0000).view(np.float32)
    with np.errstate(over = "ignore"):      # out of float16 range just means "no"
        as_fp16 = v.astype(np.float16).astype(np.float32)
    narrower = {
        "an integer type": np.all(v == np.round(v)),
        "bfloat16": np.array_equal(v, as_bf16),
        "float16": np.array_equal(v, as_fp16),
    }
    for kind, fits in narrower.items():
        if fits:
            sys.exit(f"error: every value in the codegen sample for '{name}' fits "
                     f"{kind}, so Deeploy would type it that way and disable the "
                     f"clusters")


def encode(a, b):
    """The full token sequence for a + b, answer reversed."""
    digits = lambda v, n: [int(ch) for ch in f"{v:0{n}d}"]
    answer = digits(a + b, ANSWER_DIGITS)[::-1]
    return (digits(a, OPERAND_DIGITS) + [PLUS] + digits(b, OPERAND_DIGITS) + [EQUALS]
            + answer + [END])


def decode_answer(tokens):
    """The integer an answer's tokens spell, or None if they are malformed."""
    if len(tokens) != ANSWER_LEN or tokens[-1] != END or any(t > 9 for t in tokens[:-1]):
        return None
    return int("".join(str(t) for t in tokens[:-1][::-1]))


def make_split(rng, n_test):
    """Every a+b pair, split into train and a held-out test set."""
    top = 10 ** OPERAND_DIGITS
    pairs = np.array([(a, b) for a in range(top) for b in range(top)])
    order = rng.permutation(len(pairs))
    test, train = pairs[order[:n_test]], pairs[order[n_test:]]
    seqs = lambda p: np.array([encode(a, b) for a, b in p], dtype = np.int64)
    return seqs(train), train, seqs(test), test


# --- the model, in numpy -----------------------------------------------------
#
# float64 for training, so --gradcheck can use central differences; the export
# is fp32. Weight layout is the decode graph's: Wq/Wk/Wv are [D, D] and head h
# owns columns h*D_HEAD..; Wo is [D, D] and head h owns the matching rows.

def init_params(rng):
    def w(rows, cols, scale = 1.0):
        return rng.standard_normal((rows, cols)) * scale / np.sqrt(rows)

    # Residual-branch outputs scaled down with depth, GPT-2 style, so the
    # residual stream does not grow with the number of layers.
    resid = 1.0 / np.sqrt(2 * N_LAYER)
    p = {
        "tok": 0.5 * rng.standard_normal((VOCAB, D_MODEL)),
        "pos": 0.5 * rng.standard_normal((CTX, D_MODEL)),
        "lnf_g": np.ones(D_MODEL), "lnf_b": np.zeros(D_MODEL),
        "wout": w(D_MODEL, VOCAB), "bout": np.zeros(VOCAB),
    }
    for l in range(N_LAYER):
        p.update({
            f"{l}.ln1_g": np.ones(D_MODEL), f"{l}.ln1_b": np.zeros(D_MODEL),
            f"{l}.wq": w(D_MODEL, D_MODEL), f"{l}.wk": w(D_MODEL, D_MODEL),
            f"{l}.wv": w(D_MODEL, D_MODEL), f"{l}.wo": w(D_MODEL, D_MODEL, resid),
            f"{l}.ln2_g": np.ones(D_MODEL), f"{l}.ln2_b": np.zeros(D_MODEL),
            f"{l}.w1": w(D_MODEL, D_FFN), f"{l}.b1": np.zeros(D_FFN),
            f"{l}.w2": w(D_FFN, D_MODEL, resid), f"{l}.b2": np.zeros(D_MODEL),
        })
    return p


def _ln(x, g, b):
    mu = x.mean(-1, keepdims = True)
    xc = x - mu
    rstd = 1.0 / np.sqrt((xc * xc).mean(-1, keepdims = True) + LN_EPS)
    xh = xc * rstd
    return xh * g + b, (xh, rstd, g)


def _ln_back(dy, cache):
    xh, rstd, g = cache
    dg = (dy * xh).reshape(-1, xh.shape[-1]).sum(0)
    db = dy.reshape(-1, xh.shape[-1]).sum(0)
    dxh = dy * g
    dx = rstd * (dxh - dxh.mean(-1, keepdims = True)
                 - xh * (dxh * xh).mean(-1, keepdims = True))
    return dx, dg, db


def _heads(x):
    """(B, T, D) -> (B, H, T, D_HEAD)."""
    B, T, _ = x.shape
    return x.reshape(B, T, N_HEAD, D_HEAD).transpose(0, 2, 1, 3)


def _merge(x):
    """(B, H, T, D_HEAD) -> (B, T, D)."""
    B, _, T, _ = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, D_MODEL)


def _wgrad(x, dy):
    """dW for y = x @ W over any leading batch dimensions."""
    return x.reshape(-1, x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])


def forward(p, ids):
    """Logits for every position of ids (B, T), and what backward needs."""
    B, T = ids.shape
    scale = 1.0 / np.sqrt(D_HEAD)
    causal = np.triu(np.full((T, T), -1e9), 1)
    x = p["tok"][ids] + p["pos"][:T]
    caches = []
    for l in range(N_LAYER):
        a, ln1 = _ln(x, p[f"{l}.ln1_g"], p[f"{l}.ln1_b"])
        q = _heads(a @ p[f"{l}.wq"])
        k = _heads(a @ p[f"{l}.wk"])
        v = _heads(a @ p[f"{l}.wv"])
        s = q @ k.transpose(0, 1, 3, 2) * scale + causal
        s = s - s.max(-1, keepdims = True)
        e = np.exp(s)
        att = e / e.sum(-1, keepdims = True)
        o = _merge(att @ v)
        x = x + o @ p[f"{l}.wo"]

        c, ln2 = _ln(x, p[f"{l}.ln2_g"], p[f"{l}.ln2_b"])
        h = c @ p[f"{l}.w1"] + p[f"{l}.b1"]
        r = np.maximum(h, 0.0)
        x = x + r @ p[f"{l}.w2"] + p[f"{l}.b2"]
        caches.append((a, ln1, q, k, v, att, o, c, ln2, h, r))
    xf, lnf = _ln(x, p["lnf_g"], p["lnf_b"])
    logits = xf @ p["wout"] + p["bout"]
    return logits, (ids, caches, xf, lnf)


def loss_and_grads(p, ids, targets, weight):
    """Cross-entropy over the positions `weight` selects, and its gradient."""
    logits, (ids, caches, xf, lnf) = forward(p, ids)
    z = logits - logits.max(-1, keepdims = True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims = True))
    count = weight.sum()
    picked = np.take_along_axis(logp, targets[..., None], -1)[..., 0]
    loss = -(picked * weight).sum() / count

    g = {}
    dlogits = np.exp(logp)
    np.put_along_axis(dlogits, targets[..., None],
                      np.take_along_axis(dlogits, targets[..., None], -1) - 1.0, -1)
    dlogits *= (weight / count)[..., None]

    g["wout"] = _wgrad(xf, dlogits)
    g["bout"] = dlogits.reshape(-1, VOCAB).sum(0)
    dx, g["lnf_g"], g["lnf_b"] = _ln_back(dlogits @ p["wout"].T, lnf)

    scale = 1.0 / np.sqrt(D_HEAD)
    for l in reversed(range(N_LAYER)):
        a, ln1, q, k, v, att, o, c, ln2, h, r = caches[l]
        # FFN branch.
        g[f"{l}.w2"] = _wgrad(r, dx)
        g[f"{l}.b2"] = dx.reshape(-1, D_MODEL).sum(0)
        dh = (dx @ p[f"{l}.w2"].T) * (h > 0)
        g[f"{l}.w1"] = _wgrad(c, dh)
        g[f"{l}.b1"] = dh.reshape(-1, D_FFN).sum(0)
        dc, g[f"{l}.ln2_g"], g[f"{l}.ln2_b"] = _ln_back(dh @ p[f"{l}.w1"].T, ln2)
        dx = dx + dc
        # Attention branch.
        g[f"{l}.wo"] = _wgrad(o, dx)
        do = _heads(dx @ p[f"{l}.wo"].T)
        datt = do @ v.transpose(0, 1, 3, 2)
        dv = att.transpose(0, 1, 3, 2) @ do
        ds = att * (datt - (datt * att).sum(-1, keepdims = True)) * scale
        dq = ds @ k
        dk = ds.transpose(0, 1, 3, 2) @ q
        dq, dk, dv = _merge(dq), _merge(dk), _merge(dv)
        g[f"{l}.wq"] = _wgrad(a, dq)
        g[f"{l}.wk"] = _wgrad(a, dk)
        g[f"{l}.wv"] = _wgrad(a, dv)
        da = dq @ p[f"{l}.wq"].T + dk @ p[f"{l}.wk"].T + dv @ p[f"{l}.wv"].T
        dln, g[f"{l}.ln1_g"], g[f"{l}.ln1_b"] = _ln_back(da, ln1)
        dx = dx + dln

    g["tok"] = np.zeros_like(p["tok"])
    np.add.at(g["tok"], ids, dx)
    g["pos"] = np.zeros_like(p["pos"])
    g["pos"][:ids.shape[1]] = dx.sum(0)
    return loss, g


def batch_arrays(seqs):
    """Model inputs, next-token targets and the answer-position weights."""
    ids, targets = seqs[:, :-1], seqs[:, 1:]
    weight = np.zeros(targets.shape)
    weight[:, PROMPT_LEN - 1:] = 1.0     # only the answer is learned
    return ids, targets, weight


def greedy(p, prompts):
    """Answer tokens for each prompt, decoding the way the chip will."""
    seq = prompts.copy()
    for _ in range(ANSWER_LEN):
        logits, _ = forward(p, seq)
        seq = np.concatenate([seq, logits[:, -1].argmax(-1)[:, None]], 1)
    return seq[:, PROMPT_LEN:]


def accuracy(p, seqs, pairs):
    answers = greedy(p, seqs[:, :PROMPT_LEN])
    got = [decode_answer(list(t)) for t in answers]
    return np.mean([g == a + b for g, (a, b) in zip(got, pairs)])


def train(p, seqs, test_seqs, test_pairs, steps, batch, lr, rng):
    """Adam with linear warmup and cosine decay."""
    m = {k: np.zeros_like(v) for k, v in p.items()}
    s = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2, eps, warm = 0.9, 0.99, 1e-8, min(200, steps // 10)
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(seqs), batch)
        loss, g = loss_and_grads(p, *batch_arrays(seqs[idx]))
        rate = lr * min(1.0, step / max(warm, 1)) * 0.5 * (1 + np.cos(np.pi * step / steps))
        for k in p:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            s[k] = b2 * s[k] + (1 - b2) * g[k] ** 2
            mh, sh = m[k] / (1 - b1 ** step), s[k] / (1 - b2 ** step)
            p[k] -= rate * mh / (np.sqrt(sh) + eps)
        if step % 500 == 0 or step == steps:
            acc = accuracy(p, test_seqs[:200], test_pairs[:200])
            print(f"  step {step:5d}  loss {loss:.4f}  held-out exact-match {100 * acc:.1f}%")


def gradcheck(rng):
    """Compare every parameter's analytic gradient against central differences."""
    p = init_params(rng)
    seqs = np.array([encode(*rng.integers(0, 100, 2)) for _ in range(3)])
    ids, targets, weight = batch_arrays(seqs)
    _, g = loss_and_grads(p, ids, targets, weight)
    worst = 0.0
    for k in sorted(p):
        for _ in range(4):
            i = tuple(rng.integers(0, n) for n in p[k].shape)
            if k == "pos" and i[0] >= ids.shape[1]:
                i = (int(rng.integers(0, ids.shape[1])),) + i[1:]
            old = p[k][i]
            p[k][i] = old + 1e-6
            lp, _ = loss_and_grads(p, ids, targets, weight)
            p[k][i] = old - 1e-6
            lm, _ = loss_and_grads(p, ids, targets, weight)
            p[k][i] = old
            num = (lp - lm) / 2e-6
            rel = abs(num - g[k][i]) / max(1e-8, abs(num) + abs(g[k][i]))
            worst = max(worst, rel)
            if rel > 1e-4:
                sys.exit(f"gradcheck failed: {k}{i} analytic {g[k][i]:.6e} "
                         f"numeric {num:.6e}")
    print(f"gradcheck ok: worst relative error {worst:.2e}")


# --- the decode graph --------------------------------------------------------

def input_names():
    """Graph inputs, in the order runtime/mesh/llm_main.c fills them."""
    names = ["x", "mask", "sel"]
    for l in range(N_LAYER):
        for h in range(N_HEAD):
            names += [f"kt_{l}_{h}", f"v_{l}_{h}"]
    return names


def output_names():
    """Graph outputs: the logits, then each head's new key (as a column) and value."""
    names = ["logits"]
    for l in range(N_LAYER):
        for h in range(N_HEAD):
            names += [f"k_new_{l}_{h}", f"v_new_{l}_{h}"]
    return names


def export_onnx(p, path: Path):
    f32 = lambda v: np.ascontiguousarray(v, dtype = np.float32)
    init, nodes = [], []
    counter = [0]

    def const(name, value):
        init.append(numpy_helper.from_array(f32(value), name))
        return name

    def op(op_type, inputs, out = None, **attrs):
        counter[0] += 1
        out = out or f"t{counter[0]}"
        nodes.append(helper.make_node(op_type, inputs, [out],
                                      name = f"{op_type.lower()}_{counter[0]}", **attrs))
        return out

    def layernorm(x, prefix):
        return op("LayerNormalization",
                  [x, const(f"{prefix}_g", p[f"{prefix}_g"]),
                   const(f"{prefix}_b", p[f"{prefix}_b"])],
                  axis = -1, epsilon = LN_EPS)

    score_scale = const("score_scale", np.full((1, CTX), 1.0 / np.sqrt(D_HEAD)))
    sel_t = op("Transpose", ["sel"], perm = [1, 0])          # [CTX, 1]

    x = "x"
    for l in range(N_LAYER):
        a = layernorm(x, f"{l}.ln1")
        a_t = op("Transpose", [a], perm = [1, 0])             # [D, 1]
        for h in range(N_HEAD):
            cols = slice(h * D_HEAD, (h + 1) * D_HEAD)
            q = op("MatMul", [a, const(f"{l}.wq.{h}", p[f"{l}.wq"][:, cols])])
            # The key comes out as a column, the layout it has in the cache.
            k_new = op("MatMul", [const(f"{l}.wkT.{h}", p[f"{l}.wk"][:, cols].T), a_t],
                       out = f"k_new_{l}_{h}")
            v_new = op("MatMul", [a, const(f"{l}.wv.{h}", p[f"{l}.wv"][:, cols])],
                       out = f"v_new_{l}_{h}")
            kt = op("Add", [op("MatMul", [k_new, "sel"]), f"kt_{l}_{h}"])
            v = op("Add", [op("MatMul", [sel_t, v_new]), f"v_{l}_{h}"])
            s = op("Mul", [op("MatMul", [q, kt]), score_scale])
            att = op("Softmax", [op("Add", [s, "mask"])], axis = -1)
            o = op("MatMul", [att, v])
            # Head output first: see the module docstring on the Gemm merge.
            x = op("Add", [op("MatMul", [o, const(f"{l}.wo.{h}", p[f"{l}.wo"][cols])]), x])
        c = layernorm(x, f"{l}.ln2")
        hid = op("Add", [op("MatMul", [c, const(f"{l}.w1", p[f"{l}.w1"])]),
                         const(f"{l}.b1", p[f"{l}.b1"][None])])
        hid = op("Relu", [hid])
        ffn = op("Add", [op("MatMul", [hid, const(f"{l}.w2", p[f"{l}.w2"])]),
                         const(f"{l}.b2", p[f"{l}.b2"][None])])
        x = op("Add", [ffn, x])
    xf = layernorm(x, "lnf")
    op("Add", [op("MatMul", [xf, const("wout", p["wout"])]), const("bout", p["bout"][None])],
       out = "logits")

    shapes = {"x": [1, D_MODEL], "mask": [1, CTX], "sel": [1, CTX]}
    for name in input_names()[3:]:
        shapes[name] = [D_HEAD, CTX] if name.startswith("kt") else [CTX, D_HEAD]
    out_shapes = {"logits": [1, VOCAB]}
    for name in output_names()[1:]:
        out_shapes[name] = [D_HEAD, 1] if name.startswith("k_new") else [1, D_HEAD]

    graph = helper.make_graph(
        nodes, "llm_decode",
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, shapes[n])
         for n in input_names()],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, out_shapes[n])
         for n in output_names()],
        initializer = init)
    # LayerNormalization is opset 17; MNIST/KWS export at 13.
    model = helper.make_model(graph, producer_name = "hetero-sim/pipeline/llm.py",
                              opset_imports = [helper.make_opsetid("", 17)])
    model.ir_version = 8
    # Deeploy's lowering passes and the mapper's MAC count read every
    # intermediate tensor's shape.
    model = onnx.shape_inference.infer_shapes(model, strict_mode = True)
    missing = [v.name for v in model.graph.value_info
               if not v.type.tensor_type.HasField("shape")]
    if missing:
        sys.exit(f"error: shape inference left {missing} without a shape")
    onnx.checker.check_model(model, full_check = True)
    onnx.save(model, str(path))
    return model


# --- the decode loop, as the chip runs it ------------------------------------

class DecodeState:
    """The host's side of decoding: the caches, and the per-step inputs.

    This is runtime/mesh/llm_main.c in numpy, slot for slot, so that what
    onnxruntime predicts here is what the chip is expected to produce.
    """

    def __init__(self, tok, pos):
        self.tok, self.pos = tok, pos
        self.kt = np.zeros((N_LAYER, N_HEAD, D_HEAD, CTX), dtype = np.float32)
        self.v = np.zeros((N_LAYER, N_HEAD, CTX, D_HEAD), dtype = np.float32)

    def feed(self, token, t):
        mask = np.full((1, CTX), MASK_OFF, dtype = np.float32)
        mask[0, :t + 1] = 0.0
        sel = np.zeros((1, CTX), dtype = np.float32)
        sel[0, t] = 1.0
        sel[0, t + 1:] = SEL_OFF
        feed = {"x": (self.tok[token] + self.pos[t])[None].astype(np.float32),
                "mask": mask, "sel": sel}
        for l in range(N_LAYER):
            for h in range(N_HEAD):
                feed[f"kt_{l}_{h}"] = self.kt[l, h]
                feed[f"v_{l}_{h}"] = self.v[l, h]
        return feed

    def absorb(self, outputs, t):
        it = iter(outputs[1:])
        for l in range(N_LAYER):
            for h in range(N_HEAD):
                self.kt[l, h, :, t] = next(it)[:, 0]
                self.v[l, h, t, :] = next(it)[0]


def ort_decode(sess, tok, pos, prompt, capture = None):
    """Greedy decode of one prompt on the exported graph.

    Returns the answer tokens and every step's logits. `capture`, if given,
    is the step whose feed and outputs are returned as the codegen sample.
    """
    state = DecodeState(tok, pos)
    seq = list(prompt)
    logits, sample = [], None
    for t in range(STEPS):
        feed = state.feed(seq[t], t)
        out = sess.run(None, feed)
        if t == capture:
            # Copies: the cache inputs are views of the state's arrays, which
            # absorb() and every later step write into.
            sample = ({k: v.copy() for k, v in feed.items()}, out)
        state.absorb(out, t)
        logits.append(out[0][0])
        if t >= PROMPT_LEN - 1:
            seq.append(int(out[0].argmax()))
    return seq[PROMPT_LEN:], np.array(logits), sample


# --- the C data header -------------------------------------------------------

def _c_floats(rows):
    return ",\n".join("  {" + ", ".join(f"{v:.9g}f" for v in row) + "}" for row in rows)


def write_data_header(path: Path, tok, pos, prompts, answers, reference):
    n = len(prompts)
    ints = lambda rows: ",\n".join("  {" + ", ".join(str(int(v)) for v in row) + "}"
                                   for row in rows)
    out = [
        "/* Generated by pipeline/llm.py. Do not edit.",
        " *",
        " * The embeddings the host adds up for each token (the graph starts after",
        " * them), the prompts runtime/mesh/llm_main.c decodes, the true answers,",
        " * and the tokens onnxruntime picks greedily on the same graph -- so the",
        " * simulation can be checked against the reference model and not only",
        " * against arithmetic.",
        " */",
        "#ifndef LLM_DATA_H",
        "#define LLM_DATA_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define LLM_NUM_PROBLEMS {n}",
        f"#define LLM_N_LAYER {N_LAYER}",
        f"#define LLM_N_HEAD {N_HEAD}",
        f"#define LLM_D_MODEL {D_MODEL}",
        f"#define LLM_D_HEAD {D_HEAD}",
        f"#define LLM_CTX {CTX}",
        f"#define LLM_VOCAB {VOCAB}",
        f"#define LLM_PROMPT_LEN {PROMPT_LEN}",
        f"#define LLM_ANSWER_LEN {ANSWER_LEN}",
        f"#define LLM_STEPS {STEPS}",
        f"#define LLM_MASK_OFF ({MASK_OFF}f)",
        f"#define LLM_SEL_OFF ({SEL_OFF}f)",
        "",
        "/* Network input order: x, mask, sel, then (kt, v) per layer and head.",
        " * Output order: logits, then (k_new, v_new) per layer and head. */",
        "#define LLM_IN_X 0",
        "#define LLM_IN_MASK 1",
        "#define LLM_IN_SEL 2",
        "#define LLM_IN_KT(l, h) (3 + 2 * ((l) * LLM_N_HEAD + (h)))",
        "#define LLM_IN_V(l, h) (4 + 2 * ((l) * LLM_N_HEAD + (h)))",
        "#define LLM_OUT_LOGITS 0",
        "#define LLM_OUT_K(l, h) (1 + 2 * ((l) * LLM_N_HEAD + (h)))",
        "#define LLM_OUT_V(l, h) (2 + 2 * ((l) * LLM_N_HEAD + (h)))",
        "",
        f"static const float llm_tok_emb[{VOCAB}][{D_MODEL}] = {{",
        _c_floats(tok), "};", "",
        f"static const float llm_pos_emb[{CTX}][{D_MODEL}] = {{",
        _c_floats(pos), "};", "",
        f"static const uint8_t llm_prompts[{n}][{PROMPT_LEN}] = {{",
        ints(prompts), "};", "",
        "/* The true sum, as answer tokens. */",
        f"static const uint8_t llm_answers[{n}][{ANSWER_LEN}] = {{",
        ints(answers), "};", "",
        "/* onnxruntime's greedy answer on the same graph and the same cache",
        " * protocol. The chip has to reproduce these, right or wrong. */",
        f"static const uint8_t llm_reference[{n}][{ANSWER_LEN}] = {{",
        ints(reference), "};", "",
        "#endif /* LLM_DATA_H */", "",
    ]
    path.write_text("\n".join(out))


# --- driver ------------------------------------------------------------------

def load_embeddings(path: Path):
    with np.load(path) as d:
        return d["tok"].astype(np.float32), d["pos"].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", type = int, default = 8,
                    help = "held-out problems to embed for the simulation "
                           "(default: %(default)s)")
    ap.add_argument("--steps", type = int, default = 2000)
    ap.add_argument("--batch", type = int, default = 64)
    ap.add_argument("--lr", type = float, default = 3e-3)
    ap.add_argument("--test", type = int, default = 1000,
                    help = "held-out problems never trained on (default: %(default)s)")
    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("--gradcheck", action = "store_true",
                    help = "verify the backward passes numerically, then exit")
    ap.add_argument("--reuse", action = "store_true",
                    help = "keep the network.onnx and embeddings already in the op "
                           "directory and only rebuild the evaluation set, instead of "
                           "training again")
    ap.add_argument("-o", "--out", default = str(OP_DIR))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.gradcheck:
        gradcheck(rng)
        return

    out = Path(args.out)
    out.mkdir(parents = True, exist_ok = True)
    seqs, pairs, test_seqs, test_pairs = make_split(rng, args.test)
    print(f"addition: {len(seqs)} train, {len(test_seqs)} held-out problems")

    onnx_path, emb_path = out / "network.onnx", out / "embeddings.npz"
    params = None
    if args.reuse:
        if not onnx_path.exists() or not emb_path.exists():
            sys.exit(f"error: --reuse needs an existing {onnx_path} and {emb_path}")
        print(f"reusing {onnx_path.name}, not training")
    else:
        params = init_params(rng)
        print(f"training {args.steps} steps, batch {args.batch}, lr {args.lr}")
        train(params, seqs, test_seqs, test_pairs, args.steps, args.batch, args.lr, rng)
        acc = accuracy(params, test_seqs, test_pairs)
        print(f"held-out exact-match over all {len(test_seqs)} problems: {100 * acc:.2f}%")
        export_onnx(params, onnx_path)
        np.savez(emb_path, tok = params["tok"].astype(np.float32),
                 pos = params["pos"].astype(np.float32))

    tok, pos = load_embeddings(emb_path)
    sess = ort.InferenceSession(str(onnx_path), providers = ["CPUExecutionProvider"])
    problems = test_seqs[:args.problems]
    prompts = problems[:, :PROMPT_LEN]

    # A mid-answer step: the cache is partly filled, so every input holds the
    # kind of values it will at run time -- and, through MASK_OFF and SEL_OFF,
    # a non-integer, which is what gets each one typed as fp32.
    capture = PROMPT_LEN
    reference, sample = [], None
    for i, prompt in enumerate(prompts):
        answer, logits, s = ort_decode(sess, tok, pos, prompt,
                                       capture = capture if i == 0 else None)
        reference.append(answer)
        sample = sample or s
        if params is not None:
            # The cached decode graph has to agree with the full-sequence model
            # it was exported from, step for step.
            ids = np.array([list(prompt) + answer[:-1]])
            full, _ = forward(params, ids)
            drift = np.abs(full[0] - logits).max()
            if drift > 1e-3:
                sys.exit(f"error: decode graph drifts from the trained model by "
                         f"{drift:.3e} on problem {i}")
    reference = np.array(reference)
    if params is not None:
        print(f"decode graph (fp32, KV cache) vs full-sequence model (fp64): "
              f"logits agree to within 1e-3 on all {len(prompts)} problems")

    answers = problems[:, PROMPT_LEN:]
    right = sum(decode_answer(list(r)) == a + b
                for r, (a, b) in zip(reference, test_pairs[:args.problems]))
    print(f"onnxruntime answers {right}/{len(prompts)} embedded problems correctly")

    feed, results = sample
    for name in input_names():
        check_fp32_typing(name, feed[name])
    np.savez(out / "inputs.npz",
             **{f"input_{i}": feed[n] for i, n in enumerate(input_names())})
    np.savez(out / "outputs.npz", **{f"output_{i}": r for i, r in enumerate(results)})
    write_data_header(out / "llm_data.h", tok, pos, prompts, answers, reference)

    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nop directory ready: {shown}")
    print("run it with:\n  python pipeline/run_hetero.py ops/llm")


if __name__ == "__main__":
    main()
