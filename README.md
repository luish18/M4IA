# M4IA (hetero-sim) — CVA6 / Snitch / Spatz experimentation platform

GVSoC simulation of three heterogeneous RISC-V core types, driven from ONNX via Deeploy:

```
ONNX op + inputs  ──Deeploy──►  C code  ──riscv-gcc──►  3 ELFs  ──GVSoC──►  per-core metrics
```

| core   | what is simulated                                    | GVSoC target | ISA               |
|--------|------------------------------------------------------|--------------|-------------------|
| cva6   | CVA6 64-bit host core, L1 I$/D$ + L2 + DRAM          | `cva6_real` (in `targets/`) | rv64imafdc |
| snitch | Snitch integer core + FP subsystem, SSR streamers and FREP sequencer, data in cluster TCDM | `snitch_real` (in `targets/`) | rv32imafd + Xssr/Xfrep |
| spatz  | Snitch + Spatz vector unit (4 lanes), RVV kernels    | `spatz_real` (in `targets/`) | rv32imafd + V |
| ara    | the CVA6 host with an Ara vector unit (vlen 4096, 4 lanes), autovectorized kernels; opt-in with `--cores` | `ara_host` (in `targets/`) | rv64imafdc + V |

Each core runs the best code the pipeline can give it, not the same code: CVA6
scalar `-O3`, and both accelerator cores' FP kernels hand-written against what
they actually have — Snitch against its two custom extensions, Spatz against
RVV. See [The Snitch FP extensions](#the-snitch-fp-extensions-xssr--xfrep) and
[The Spatz RVV kernels](#the-spatz-rvv-kernels).

## Documentation

This README remains the substantial technical entry point: setup, commands,
measured results, kernel rationale, memory behavior, and simulator fixes live
here. The experiment-facing and maintainer contracts are split into focused
documents:

- [Documentation map](docs/README.md) — choose a guide by reader goal and role.
- [Architecture](docs/architecture.md) — current end-to-end operational flow.
- [Experimental Workbench](docs/experimental-workbench.md) — request,
  resolution, evidence, identity, provenance, and scientific-safety contracts.
- [Workbench quickstart](docs/experimental-workbench-quickstart.md) — discover,
  resolve, run, and inspect one experiment.
- [Chip design parameters](docs/chip-design-parameters.md) — current design
  inventory and causal-validation boundary.
- [Extension guide](docs/extending-m4ia.md) and
  [extension-seam audit](docs/dev/extension-seams.md) — how to extend M4IA and
  why some seams remain fragile or blocked.
- [Metric semantics](docs/metrics.md) — measurement scopes and non-claims.
- [GUI integration](docs/gui-integration.md) — current thin-client boundary and
  the schema-driven continuation path.

Historical context remains intentionally separate: the Portuguese
[development report](docs/relatorio-desenvolvimento.md), the original
[heterogeneous-mesh proposal](docs/hetero-mesh-plan.md), the
[Foundation construction history](docs/dev/foundation-checkpoints.md), and the
[modern integration notes](docs/dev/integration-notes.md), which record what
already existed in M4IA, what the Foundation changed, and why it was ported
semantically. Future scientific work is sequenced in the
[research roadmap](docs/dev/research-roadmap.md).

M4IA has two related execution contexts:

- `pipeline/run.py` builds and measures one core type at a time. It is the
  standalone/per-core comparison path used by the baseline operator tables in
  this README.
- `pipeline/run_hetero.py` runs the heterogeneous SoC: a CVA6 host dispatches
  work to the configured Snitch and Spatz clusters. The sweep and application
  paths build on this heterogeneous runner.

**The memory system is simulated,** not assumed away: cache misses, DRAM latency and
refill bandwidth all land in the cycle counts. `--memory ideal` switches back to the
zero-latency targets (`cva6_ideal`, `snitch`, `spatz`), where every access completes
in a single cycle. This is an idealized-memory comparison; instruction execution,
runtime, synchronization, instrumentation, and other modeled effects remain. See
[The memory system](#the-memory-system).

## Setup

Dependencies (GVSoC, Deeploy, the RISC-V toolchain) are large and are not tracked in
git. One command fetches them at pinned commits, applies the local GVSoC patches, and
builds the simulator (including the targets in `targets/` and their cache model):

```bash
./setup.sh
```

It needs `uv`, `cmake`, `git`, `curl`, and a C++ toolchain; re-running it is safe.

### Docker

`Dockerfile` runs the same `./setup.sh` inside an Ubuntu 22.04 image, so it works the
same on any platform Docker runs on:

```bash
docker build -t hetero-sim .
docker run --rm hetero-sim                                                    # default op
docker run --rm hetero-sim python pipeline/run.py Tests/Kernels/FP32/MatMul   # any op
docker run --rm -it hetero-sim bash                                           # shell
```

The image bakes in the built GVSoC targets, so no fetching happens at container start;
only the `docker build` needs network access.

The build is not quick. `setup.sh` clones GVSoC *with all its submodules*, downloads the
toolchain, applies the patches in `deps/patches/`, and compiles ten targets in four
flavours each (optim, debug, profile, asserts -- `--trace`, `--vcd`, `--gui` and
`--power` select between them at run time, so none of them is optional). Budget tens of
minutes and about 6 GB.

#### Developing against the image

`COPY . .` and `RUN ./setup.sh` are one layer, so touching any tracked file invalidates
it and re-clones GVSoC from scratch. That is the wrong loop for changing a model. Mount
the working tree into a long-lived container instead, and rebuild only what changed:

```bash
docker run -dit --name hetero-sim-dev -w /workspace \
  -v "$PWD/Makefile:/workspace/Makefile" -v "$PWD/ops:/workspace/ops" \
  -v "$PWD/pipeline:/workspace/pipeline" -v "$PWD/results:/workspace/results" \
  -v "$PWD/runtime:/workspace/runtime" -v "$PWD/targets:/workspace/targets" \
  -v "$PWD/work:/workspace/work" -v "$PWD/deps/patches:/workspace/deps/patches" \
  hetero-sim bash

docker exec hetero-sim-dev bash -lc 'cd /workspace && make gvsoc'   # ~1 min incremental
docker exec hetero-sim-dev bash -lc 'cd /workspace && make mesh-probe'
```

Everything under those mounts is the host's, so an edit is visible immediately: Python
targets and pipeline changes need no rebuild at all, and a C++ model (say
`targets/hetero/timing_cache.cpp`) needs only `make gvsoc`, which recompiles the one
component rather than the world.

Two things to know about that container:

- **Its output is owned by root.** `work/` files it writes cannot be removed from the
  host shell; clean them from inside (`docker exec <ctr> rm -rf /workspace/work/<dir>`).
- **The image goes stale.** It carries the GVSoC that was built from `targets/` at image
  build time, so after a change to a C++ model the image and the container disagree
  until one of them is rebuilt. Rebuild the image when you want a clean reproducible
  environment or to hand it to someone else; use `make gvsoc` while developing.

## Usage

Run a bundled Deeploy single-op test (`network.onnx` + `inputs.npz` + `outputs.npz`):

```bash
.venv/bin/python pipeline/run.py Tests/Kernels/FP32/GEMM/Regular
```

Bring your own ONNX op (expected outputs computed with onnxruntime):

```bash
.venv/bin/python pipeline/make_op.py myop.onnx myinputs.npz -o ops/myop
.venv/bin/python pipeline/run.py ops/myop
```

Output: a per-core table (cycles, speedup vs CVA6, numerical error vs the ONNX
reference), the cache counters behind it, and the same numbers as JSON in `results/`.

Swap Spatz's hand-written kernels for whatever GCC autovectorizes:

```bash
.venv/bin/python pipeline/run.py <op> --spatz-kernels autovec
```

Results for the non-default choice land in `results/<op>-spatz-autovec.json`.

Compare the two memory models on the same op:

```bash
.venv/bin/python pipeline/run.py Tests/Kernels/FP32/MatMul                  # modelled memory
.venv/bin/python pipeline/run.py Tests/Kernels/FP32/MatMul --memory ideal   # zero-latency
```

Results land in `results/<op>.json` and `results/<op>-ideal.json`.

Put a modeled LPDDR/HyperRAM device-timing model behind the caches instead of
the fixed main-memory latency (see
[Modeled device memories](#modeled-device-memories---dram)):

```bash
.venv/bin/python pipeline/run.py Tests/Kernels/FP32/MatMul --cores cva6,snitch,spatz,ara --dram lpddr4
.venv/bin/python pipeline/run_hetero.py ops/mnist --dram hyperram        # also: make mnist DRAM=hyperram
```

`--dram` takes `fixed` (the default), `lpddr4`, `lpddr4x`, `lpddr5` or `hyperram`;
results get a `-<kind>` suffix so they never overwrite the fixed-latency baselines.

### Debugging a run

`--debug` (or `-d`, or `HES_DEBUG=1`, or `make run DEBUG=1`) traces every command
the pipeline runs — Deeploy codegen, each compile, the link, GVSoC — followed by
the files that command generated, so a failure can be reproduced by hand:

```bash
.venv/bin/python pipeline/run.py ops/mymatmul --debug
```

```
[dbg] $ .venv/bin/python generateNetwork.py -t ops/mymatmul -p Generic -d work/mymatmul/gen
[dbg]   (cwd: deps/deeploy/DeeployTest)
[dbg]   exit=0 in 1.0s
[dbg]   -> work/mymatmul/gen/Network.c  (25.0 KiB)
[dbg]   -> work/mymatmul/gen/testinputs.h  (23.6 KiB)
...
[dbg] $ toolchains/.../riscv-none-elf-gcc -march=rv64imafdc_zicsr_zifencei ... -o work/mymatmul/cva6/net.elf
[dbg]   exit=0 in 0.0s
[dbg]   -> work/mymatmul/cva6/net.elf  (27.2 KiB)
```

The trace goes to stderr, so stdout still carries only the report:
`run.py <op> --debug 2>trace.log`. Files the driver itself writes (`sim.log`,
the results JSON) are traced too. Commands that produce nothing are shown as
`(not created)`, which is what a failed compile or a timed-out simulation looks
like.

## Desktop GUI

`gui/` is a desktop front-end (Rust, [iced](https://iced.rs)) for everything
above: import ONNX models, run ops on a chosen set of cores and compare them,
define a sweep's parameter space and run it over a group of models such as
MNIST and KWS, and read the results. It runs on Linux, macOS and Windows.

```bash
cd gui && cargo run --release
```

It is a thin client: it runs the same scripts you would (`make_op.py`,
`run.py`, `run_hetero.py`, `sweep/run.py`, `sweep/report.py`, plus
`pipeline/gui_query.py`, which answers its questions about ops and knobs as
JSON) and draws what they write. Where they run is a setting:

- **Docker** (any OS; the only choice on macOS and Windows, since GVSoC builds
  on Linux only). Each job is a `docker run` of the `hetero-sim` image with the
  workspace's `ops/`, `results/` and `work/` mounted, so the workspace can be a
  plain clone with no `setup.sh`. Build the image first (see
  [Docker](#docker)); an image older than the GUI helpers is caught by
  *Settings → Check environment*. Output is chowned back to you on Linux.
- **Native**: a Linux checkout where `./setup.sh` has run.

The tabs:

| tab | what it does |
|-----|--------------|
| Models & ops | list `ops/` (and Deeploy's kernel tests), inspect a graph's inputs, outputs and node types, import an `.onnx` with an `inputs.npz` or random inputs, regenerate `ops/mnist` and `ops/kws` |
| Compare cores | run ops on any of cva6 / snitch / spatz / ara standalone (`run.py`), or on the SoC mapped or pinned (`run_hetero.py`), with the main memory of your choice (`--dram`); bar charts, speedups, cache counters, node mapping |
| Sweep | pick models, host, samples and main memory; set the parameter space as one-factor-at-a-time, full factorial or an explicit list of points; check the points against `design.py` before launching; save/load the spec |
| Sweep results | live progress, the sensitivity table and Pareto front from `report.py`, every cell, CSV export |
| Jobs | the queue, live logs, cancel, and the exact command of each job |

GUI output goes under `work/gui/` (runs, sweeps), never over the committed
`results/`. The flags it relies on are ordinary CLI flags, usable by hand:
`run.py --out FILE`, `sweep/run.py --designs FILE --progress json`, and
`sweep/report.py --json`.

`cargo test` covers parsing, the space expansion and command building; the
tests that drive the pipeline through Docker are opt-in:
`cargo test -- --ignored --test-threads=1` (image from `HETERO_GUI_TEST_IMAGE`,
default `hetero-sim:gui`). CI builds and tests the GUI on all three platforms.

## Results

### Preserved standalone baseline results

These are preserved simulated baselines for the listed operator shapes and
recorded default modeled-memory configuration. They use the standalone
`pipeline/run.py` path and measure one core of each type; they are not universal
rankings of the configurable heterogeneous clusters. `×` is against CVA6; the
fastest core in each row is in bold. Every row is verified against the ONNX
reference and reproduced by
`results/<op>.json`. The `ara` column is the CVA6 host with its Ara vector unit,
opt-in with `--cores cva6,ara` — see [The vector host](#the-vector-host-cva6--ara).

Unless a later result section states another configuration explicitly, its
numbers should likewise be read as preserved simulated measurements for the
named shapes, software, and model state—not as universal architecture rankings.

| operator | shape | cva6 | snitch | spatz | ara |
|----------|-------|------|--------|-------|-----|
| Add                | 64 × fp32, elementwise                       | **863**  | 1053 (0.8×)        | 1035 (0.8×)      | 1061 (0.8×)       |
| MatMul             | 2 × (16×32 · 32×8) fp32                      | 119.0k   | 12.6k (9.4×)       | **6.8k (17.5×)** | 74.5k (1.6×)      |
| MatMul (custom op) | 32×32×32 fp32                                | 477.5k   | 43.1k (11.1×)      | **7.5k (63.3×)** | 291.2k (1.6×)     |
| GEMM               | 32×32×32 fp32 + bias                         | 489.0k   | 43.5k (11.2×)      | **8.4k (58.5×)** | 296.5k (1.6×)     |
| GEMM (int8)        | 32×32×32, s8·s8 → s32                        | 575.9k   | 451.7k (1.3×)      | **84.1k (6.8×)** | 276.9k (2.1×)     |
| Conv2D + bias      | 2×64×32 fp32, 4 filters 2×8×8, stride 2×4    | 1.86M    | **173.9k (10.7×)** | 457.0k (4.1×)    | 2.93M (0.6×)      |
| Softmax            | 512 fp32, 32 rows of 16                      | 36.5k    | 51.0k (0.7×)       | 32.5k (1.1×)     | **31.3k (1.2×)**  |
| ReLU               | 2×8×8 fp32, elementwise                      | 1695     | 2071 (0.8×)        | 584 (2.9×)       | **530 (3.2×)**    |
| MaxPool 2D         | 16×16 → 8×8 fp32, 3×3 window, stride 2       | 23.4k    | 17.2k (1.4×)       | **10.9k (2.2×)** | 11.3k (2.1×)      |

Reading it:

- **In these shapes, Spatz takes every fp32 matmul-shaped row**, by 17-63× over CVA6 and 2-6×
  over Snitch, once its kernels are hand-written against RVV rather than
  autovectorized — see [The Spatz RVV kernels](#the-spatz-rvv-kernels). While
  Spatz ran compiler output this table had Snitch ahead on all of them.
- **Snitch takes the listed Conv2D case**, and its extensions are what put it 10.7× over
  CVA6 everywhere: Xssr removes the loads and the address arithmetic and Xfrep
  removes the loop, which leaves an FMA rate the stream bandwidth sets — see
  [The Snitch FP extensions](#the-snitch-fp-extensions-xssr--xfrep). Spatz has
  no hand-written convolution yet, which is the obvious next kernel.
- **Spatz takes the listed int8 GEMM case**, by 5.4× over Snitch: SSR feeds the FP regfile and
  FREP replays FP instructions, so neither helps an integer reduction, while
  RVV vectorizes it directly.
- **This Softmax shape is `expf`-bound** on every core, so they land within 1.7× of each
  other and no amount of streaming or vectorizing moves it much. Snitch is last
  because the libm code is scalar work on its small integer core, and FREP
  replays FP instructions from a 16-entry buffer, not a libm call.
- **Add is too small to say anything** — 64 elements, where the result is
  dominated by call and loop overhead rather than the 64 additions.
- **The vector host is not a fourth accelerator.** It gains most on the work a
  network leaves on the host — ReLU 3.2×, MaxPool and int8 GEMM 2.1× — and
  loses Conv2D (0.6×): GCC reads the convolution window through a gather, and
  the Ara unit issues a gather one element per bus burst. On every dense fp32
  op Spatz is still 11-39× ahead of it.

For these baseline cases, the useful dividing line is not simply integer versus
float. It is whether the work reduces to a dense FP multiply-accumulate loop
that Snitch's sequencer can replay. Broader claims require additional shapes,
completed implementation evidence, and controlled experiments.

`--memory ideal` (zero-latency memory, `results/<op>-ideal.json`) keeps the same
ordering everywhere except Add, where Spatz's 635 cycles edge out CVA6's 715
once its instruction fetches stop paying DRAM. The cost of the modelled memory
system per core is broken down under
[What it costs](#what-it-costs).

## Keyword spotting: using both clusters at once

The table above, and the MNIST network built on it, measure one thing: which
core is fastest at a given operator. That is operator selection, and it leaves
two things on the table. MNIST gives the Snitch cluster **zero** cycles — the
cost model correctly sends every dense node to Spatz — and `hes_offload()`
blocks, so even when two clusters could work at once they take turns and the
total is a sum rather than a maximum.

`ops/kws` is an application built to need both. It is keyword spotting on
synthetic speech, and it has two stages that want different hardware and are
available at the same time:

```
 clip N+1 ─┐
           ▼
   [SNITCH cluster]  window → FFT → |·|² → mel → log → DCT      one job, 8 cores,
           │         (HES_K_MFCC_FP32)                          sliced by frame
           ▼  features 32×13   (main memory, double-buffered)
   [SPATZ cluster]   Conv 3×3 → Conv 3×3 → Gemm → Gemm          Deeploy-generated,
           │                                                    existing kernels
           ▼  logits
   [CVA6]            ReLU / MaxPool / Softmax / argmax / control

 t0:  snitch(clip 0)
 t1:  snitch(clip 1)  ||  spatz+cva6(clip 0)      ← both clusters live
 t2:  snitch(clip 2)  ||  spatz+cva6(clip 1)
```

Because clips keep arriving, the front-end for clip N+1 is independent of the
classifier for clip N. `runtime/mesh/kws_main.c` posts the one before running
the other, and collects it after. `RunNetwork()` offloads its Conv and Gemm
nodes to a *different* cluster's mailbox, so the overlap needs nothing from the
generated code — only that `hes_offload()` be split into `hes_post()` +
`hes_wait()`, which is all `runtime/mesh/hes_host.c` changed. Each cluster
already had its own `seq`/`done_seq` pair; one job per cluster in flight was
always expressible, `hes_offload()` just never used it.

Run it with `make kws`. 16 clips, all numbers from the modelled-memory board:

| | cycles/clip | vs. pipelined |
|---|---|---|
| front-end on snitch, classifier on spatz | **256,652** | — |
| the same work, serial (`SERIAL=1`) | 468,217 | 1.82× slower |
| front-end on spatz, classifier on snitch (`FE=spatz PIN=snitch`) | 269,436 | 1.05× slower |
| classifier on the host (`PIN=cva6`, 8 clips) | 1,539,711 | 6.0× slower |

and, for the first time in this repository, all three engines do real work:

| engine | cycles (16 clips) | share | |
|---|---|---|---|
| snitch | 3,614,890 | 50.2% | the MFCC front-end |
| spatz | 1,831,164 | 25.4% | Conv and Gemm |
| cva6 | 1,755,330 | 24.4% | ReLU, MaxPool, Softmax, control |

against MNIST's `spatz 53% / cva6 47% / snitch 0%`. 93.4% of the front-end's
cycles overlap the classifier and cost no wall time at all.

### What the placement comparison says

The two rows in the middle of the first table are the same work with the two
stages swapped between the clusters, and the interesting part is that Spatz is
faster at **both** stages and still loses:

| stage | on snitch | on spatz | |
|---|---|---|---|
| MFCC front-end | 225.9k/clip | **185.6k/clip** | spatz, 1.22× |
| classifier (cluster share) | 129.8k/clip | **114.4k/clip** | spatz, 1.13× |

The two stages have to be on different clusters — a cluster's mailbox holds one
descriptor, so posting the next front-end to the cluster the classifier is using
would overwrite a job in flight, which `hes_post()` refuses and
`pipeline/run_hetero.py` catches before the build. So the choice is which stage
gets Spatz, and a pipeline's period is the *maximum* of its stages, not their
sum. The classifier is the slower stage under either assignment, so Spatz
belongs on the classifier and the front-end goes to Snitch by elimination.
"Put each stage on the core that runs it fastest" is not the rule; "give the
better core to the stage that sets the period" is.

### What the SSR front-end kernel does and does not buy

`runtime/snitch/kernels/mfcc_fp32_ssr.c` streams two of the four stages, and
they are the two the access-pattern argument rests on:

- the **mel filterbank** — 40 reductions, each over its own ragged 10–30 bin
  run. Short vectors *and* a horizontal reduction per filter is the worst case
  for RVV; for SSR the data-dependent trip count is just a register, and the
  accumulators stay live across the whole filter.
- the **DCT-II** — a small dense matrix-vector product, the same block shape as
  the GEMM kernel's inner loop.

It is worth 8.2% of the front-end (992.5k → 911.6k cycles over 4 clips), and no
more, because the FFT dominates and is **not** streamed. Its butterfly wants
four reads and four writes per iteration against three data movers, so streaming
it means splitting it into passes through a scratch buffer, and whether that
costs less than the loads it removes is a real question rather than a rhetorical
one. It is left open. That is also why Spatz wins the front-end above: the
measurement does not support the a-priori claim that the front-end is
Snitch-shaped work, and the claim is reported as refuted rather than quietly
dropped. A hand-written RVV front-end for Spatz — which does not exist either —
would likely widen its lead further.

### How it is checked

Three independent checks, and a misheard keyword is not one of them:

- every clip's prediction against **onnxruntime** on the same graph — 16/16, the
  same discipline `mnist_main.c` uses;
- every clip's **features** against the numpy front-end in `pipeline/kws.py`,
  embedded in `ops/kws/kws_data.h`. Max |chip − numpy| = 5×10⁻⁶. Without this a
  wrong FFT would only ever surface as a misclassification;
- `make ssr-test` runs `runtime/tests/ssr_mfcc.c`, which compares the streamed
  filterbank and DCT against a scalar reference on the ragged filter widths and
  cepstra counts the application never reaches — bit-identical on all five.

The audio is synthesized procedurally from formant templates (no download, no
new dependency), so the 98.8% test accuracy says the network trained, not that
the model competes with Speech Commands. What is being measured is the chip.

## The vector host (CVA6 + Ara)

The orchestrator can carry a vector unit of its own.
[`targets/ara_host.py`](targets/ara_host.py) is the `cva6_real` board with
GVSoC's Ara unit attached to the CVA6 — ISS v2, `vlen` 4096, 4 lanes × 8 B,
vector loads and stores going through the same cache hierarchy as scalar data —
and [`targets/hetero_ara.py`](targets/hetero_ara.py) is `hetero_soc` with that
core as orchestrator: the same Snitch cluster, Spatz pair, memory system and
addresses.

```bash
.venv/bin/python pipeline/run.py <op> --cores cva6,ara   # the host with and without its vector unit
make mnist HOST=ara                                     # the SoC with the vector host
make kws HOST=ara
make ara-test                                           # the checks the model fixes rest on
```

`ara` is opt-in and `HOST` defaults to `cva6`, so every number outside this
section and the `ara` column in [Results](#results) is the scalar host.

### What it buys

Per core, the `ara` column: 3.2× on ReLU, 2.1× on MaxPool and int8 GEMM, 1.6× on
the dense fp32 ops, 0.6× on Conv2D. On the SoC, on the same inputs as the
scalar-host runs and verified against onnxruntime:

| | scalar host | vector host | |
|---|---|---|---|
| MNIST, cycles/image (16 images) | 572,820 | **437,272** | 1.31× |
| host share of the MNIST run | 46.7% | 30.2% | |
| KWS, cycles/clip (16 clips) | 256,652 | **237,722** | 1.08× |
| host cycles in the KWS run | 1,755,330 | 846,548 | 2.07× |

MNIST also classifies all 64 images of its evaluation set on the vector host,
64/64 agreeing with onnxruntime. The placement does not change: Conv and Gemm
stay on Spatz, while ReLU, MaxPool and Softmax — the half of MNIST neither
cluster has kernels for — get faster in place, which is what the vector host is
for. KWS gains less because its period is set by the MFCC front-end on Snitch,
which the host does not run.

The mapper prices the vector host by its own measured rates (`RATES["ara"]` in
[`pipeline/hetero_platform/mapper.py`](pipeline/hetero_platform/mapper.py)),
and its placement still beats both alternatives on either host (8 MNIST images,
cycles/image):

| placement | scalar host | vector host |
|---|---|---|
| Conv and Gemm on spatz — the mapper's choice | **576,078** | **439,089** |
| Conv and Gemm on snitch | 621,456 | 485,313 |
| every node on the host | 4,214,984 | 13,376,745 |

The last row is the vector unit's weakness in one number: with the convolutions
on the host it is 3.2× *slower* than the scalar host, because GCC reads the
convolution window through a gather and the unit issues a gather one element
per bus burst. The measured rates are what keep the mapper from putting them
there.

### How it is checked

The Ara model computed wrong answers until five fixes, listed under
[GVSoC model fixes](#ara-vector-host), and the checks that found them stay:

- `make ara-test` runs `runtime/tests/ara_probe.c` — vector loads and stores
  over lengths, alignments and scalar interleavings, gathers and strided
  accesses, the vl and scalar-operand races, and each instruction GCC's dense
  kernels are built from, all against scalar references — and
  `runtime/tests/ara_kernels.c`, Deeploy's integer GEMM vectorized with the
  host's kernel flags against the same source built without the vector
  extension, element by element;
- `pipeline/run.py --cores cva6,snitch,spatz,ara` verifies every op in the
  Results table against the ONNX reference;
- MNIST and KWS check every sample against onnxruntime, as on the scalar host.

## Layout

```
setup.sh               fetch + patch + build all dependencies (pinned commits)
pipeline/run.py        the pipeline driver (codegen → build → simulate → report)
pipeline/make_op.py    wrap an ONNX model + inputs into a pipeline op directory
pipeline/sweep/        design-space sweeps: design rules, area model, driver, report
pipeline/gui_query.py  ops, knobs, graph inspection and design checks as JSON (for the GUI)
gui/                   the desktop front-end (Rust + iced), see "Desktop GUI"
runtime/               bare-metal glue: crt0, semihosting, linker scripts, bench main
runtime/snitch/snitch_ssr.h    Xssr/Xfrep intrinsics as raw instruction encodings
runtime/snitch/kernels/        the FP kernels Snitch overrides (MatMul/GEMM/Conv2d/MFCC)
runtime/spatz/kernels/         the FP kernels Spatz overrides, hand-written RVV
runtime/common/kernels/        first-party kernels every core gets (the MFCC front-end)
runtime/mesh/kws_main.c        keyword spotting: both clusters running at once
pipeline/kws.py                synthesize the audio, train the classifier, export ops/kws
runtime/tests/         standalone bare-metal checks (`make ssr-test` runs the SSR ones,
                       `make ara-test` ara_probe.c and ara_kernels.c on the vector host)
targets/*_real.py      GVSoC targets with the memory system modelled (the default)
targets/cva6_ideal.py  GVSoC target with zero-latency memory (--memory ideal)
targets/ara_host.py    CVA6 + Ara vector unit, modelled memory (--cores ara)
targets/hetero_soc.py  the SoC: CVA6 host + Snitch cluster + Spatz pair (make hetero/mnist/kws)
targets/hetero_ara.py  the same SoC with the vector host (HOST=ara)
targets/hetero/        memory-system parameters + the timing-cache model (C++ and generator)
targets/hetero/dram*   the main-memory device model: engine (dram_core.hpp), GVSoC wrapper
                       (dram.cpp/.py) and the LPDDR4/4X/5 and HyperRAM presets (dram_presets.py)
targets/hetero_models.py  build-only target that compiles the optional models
tools/dram/            DRAM engine unit tests, trace replay, preset check (`make dram-test`)
tools/dram/xcheck/     cross-check against DRAMSys and Ramulator2 (`make dram-xcheck`)
deps/patches/          local fixes to GVSoC models (tracked; applied by setup.sh)
deps/gvsoc             GVSoC checkout + build           (untracked)
deps/deeploy           Deeploy, installed editable      (untracked)
toolchains/            xPack riscv-none-elf-gcc 15.2    (untracked)
work/                  per-op build + simulation artifacts, disposable (untracked)
ops/                   input op directories (onnx + npz)
results/               metrics JSON per op — committed as the verified baseline
                       (<op>.json modelled memory, <op>-ideal.json zero-latency)
```

## How it works

1. **Deeploy codegen** — `generateNetwork.py -p Generic` turns the ONNX op into
   `Network.c` (kernel calls + weights) plus `testinputs.h` / `testoutputs.h`.
2. **Per-core build** — the same generated C is cross-compiled three times.
   Snitch replaces three of the Deeploy kernels with SSR/FREP versions of its
   own and Spatz replaces two with RVV versions (both below); the Deeploy ones
   stay linked in as `<name>_generic` and still run the shapes the rewrites do
   not cover. Spatz's remaining kernels are compiled `-O3 -ffast-math` so GCC
   autovectorizes them; glue code stays scalar on both, because the Spatz
   `-march` carries `v` and GCC will otherwise emit RVV for ordinary control
   loops.
   Snitch/Spatz builds avoid the C extension because
   the GVSoC Snitch model executes compressed FP loads on the integer core,
   diverging from the decoupled FP subsystem.
3. **Simulation** — each ELF runs on its GVSoC board; `bench_main.c` reads
   `mcycle` around `RunNetwork()`, verifies outputs against the ONNX reference,
   and prints a `[HES] ...` metrics line via semihosting. Which board depends on
   `--memory`; the binary does not, so both models run the same code.
4. **Report** — `run.py` parses the metrics and the `[HES-MEM]` cache counters and
   emits the comparison table + JSON.

## The Snitch FP extensions (Xssr / Xfrep)

A Snitch core is a small integer core in front of a decoupled FP subsystem, and
two vendor extensions are what make that split pay off:

- **Xssr** — stream semantic registers. `ft0`/`ft1`/`ft2` stop being registers:
  once a data mover is configured with a loop nest (up to four levels of bounds
  and strides, plus a repeat count), every read of `ft0` pops the next element
  of the stream out of memory and every write pushes one back. Loads, address
  arithmetic and pointer bumps leave the loop.
- **Xfrep** — FP repetition. `frep.o rs1, len, 0, 0` hands the next `len`
  instructions to the FPU sequencer, which replays them `rs1`+1 times from its
  16-entry buffer. The integer core issues the body once and is then done.

An inner loop that was load / load / fmadd / bump / branch becomes one `frep.o`
over a handful of `fmadd.s`, running at whatever rate the streams can feed.

### Emitting them from GCC

`riscv-none-elf-gcc` supports neither extension (there is no upstream binutils
support for either), so [`runtime/snitch/snitch_ssr.h`](runtime/snitch/snitch_ssr.h) emits
each one as a raw encoding via `.insn`, which needs nothing from the assembler:

| instruction | emitted as |
|-------------|------------|
| `scfgwi rs1, reg<<5\|ssr` — write an SSR config register | `.insn r 0x2b, 2, reg, x0, rs1, x<ssr>` |
| `scfgri rd, reg<<5\|ssr` — read one back                 | `.insn r 0x2b, 1, reg, rd, x0, x<ssr>` |
| `frep.o rs1, len, 0, 0` — repeat the next `len` insns    | `.insn i 0x0b, 0, x1, rs1, len-1` |
| `frep.i rs1, len, 0, 0` — repeat each of them in place   | `.insn i 0x0b, 0, x0, rs1, len-1` |

The `x1`/`x0` in the frep encodings is not a register: that field carries
`stagger_mask<<1 | is_outer`, and register staggering is unused here. Enabling
the streams is an ordinary CSR write (`csrsi 0x7c0, 1`), which GCC can already
assemble.

On top of those, the header provides the same calls the Snitch runtime does —
`ssr_loop_1d`..`4d`, `ssr_repeat`, `ssr_read`, `ssr_write` — including its
convention that a bound register holds count-1 and that stride *i*+1 is the
*delta* applied when loop *i* wraps, i.e. the next step minus the distance the
inner loop already travelled.

While SSR is enabled every FP instruction touching ft0-ft2 consumes stream
elements, including anything the compiler decides to emit there. The enable and
disable therefore sit inside the same `asm volatile` block as the loop body, so
the enabled window holds exactly the instructions written by hand, and ft0-ft2
are clobbered so the register allocator stays away from them.

### The kernels

`pipeline/run.py` compiles [`runtime/snitch/kernels/`](runtime/snitch/kernels)
for the snitch core and renames the Deeploy Generic definitions it replaces to
`<name>_generic` (a `-D` applied to the Generic library alone). The generated
network keeps calling the original names and so gets the streamed version; the
streamed version calls `<name>_generic` for the shapes it does not cover.

| kernel | ft0 | ft1 | FREP body |
|--------|-----|-----|-----------|
| `MatMul_fp32_fp32_fp32`, `Gemm_fp32_fp32_fp32_fp32` | `A[i][k]`, held for 8 FMAs | 8 consecutive `B[k][j]` | 8 `fmadd.s`, replayed N times |
| `Conv2d_fp32_fp32_fp32_NCHW` | input window element, held for 4 FMAs | tap *k* of 4 filters | 4 `fmadd.s`, replayed C·P·Q times |

Both put the *reused* operand on ft0 with an SSR repeat count, so the two
streams issue 1 + unroll accesses per unroll FMAs; both unroll wide enough that
the independent accumulators cover the 3-cycle FMA latency. That ratio is what
sets the speed: the model funnels the three SSR ports through one per-core
router, which `make ssr-test` measures at about one element per cycle, so
GEMM's 9 accesses per 8 FMAs puts the floor near 1.1 cycles/FMA. It measures
1.33, the difference being the per-block C loads, result stores and FREP setup
outside the streamed body.

Reductions keep the generic order, so MatMul and Conv2d come out bit-identical
to the scalar kernels; GEMM differs in the last bit only because it starts its
accumulator at C instead of adding C at the end.

Columns past the last full block, transposed GEMM operands and filters past the
last group of four run scalar. `make ssr-test` exercises exactly those paths
(plus a streaming probe) on `snitch_real` — the benchmark ops only reach the
shapes that divide evenly.

### What it buys

Snitch cycles for the timed op, everything else unchanged:

| op                     | scalar FP | Xssr + Xfrep |        |
|------------------------|-----------|--------------|--------|
| Conv/Regular_2D_Bias   | 1.14M     | 174k         | 6.5x   |
| GEMM/Regular           | 214k      | 43.5k        | 4.9x   |
| MatMul                 | 52.7k     | 12.6k        | 4.2x   |
| mymatmul (32×32×32)    | 206k      | 43.1k        | 4.8x   |

which was enough to put one Snitch core ahead of the 4-lane Spatz on all four
ops while Spatz ran compiler output. It now holds only on Conv2D (174k vs
457k), the one op Spatz has no hand-written kernel for; on the MatMul and GEMM
shapes, where both cores' kernels are hand-written, Spatz is 5-8x ahead — see
[The Spatz RVV kernels](#the-spatz-rvv-kernels).

Three ops in the baseline are untouched, and none of them can use the
extensions:

- **Softmax** spends its cycles inside `expf`; FREP replays FP instructions from
  a 16-entry buffer, not a libm call, and the surrounding max/sum/scale passes
  are a small part of the total.
- **Integer GEMM** reduces in integer registers. SSR feeds the FP regfile and
  FREP replays FP instructions, so neither applies.
- **Add** is emitted inline into the generated `Network.c` by Deeploy instead of
  being called as a kernel, so there is no function to override.

## The Spatz RVV kernels

Spatz used to run whatever GCC autovectorized out of the Deeploy sources, and
that cost about 5x. GCC vectorizes the innermost loop, which for a matmul is
the dot product, and that shape is wrong for this machine twice over: it loads
`B` with `vlse32.v` down a column — a strided gather whose stride is a multiple
of the TCDM bank interleave for every power-of-two row length, so the elements
serialize onto one bank — and it runs a `vfredusum` reduction per output
element, the one vector operation whose cost does not amortize over the vector
length.

[`runtime/spatz/kernels/`](runtime/spatz/kernels) keeps the accumulator in a
vector register across `k` and broadcasts the scalar `A` element instead:

```c
acc[r] = vfmacc_vf(acc[r], A[i+r][k], B[k][j..j+vl], vl);
```

Every load is then unit-stride, no reduction is executed at all, and four rows
of `A` share one `B` load. It runs at `LMUL=4`, which is worth 1.8x on its own:
a back-to-back `vfmacc` loop with every operand already in the vector register
file sustains 0.180 cycles/element at `LMUL=1` against 0.128 at `LMUL=4`, so a
third of the peak goes to issue overhead before the kernel touches memory.

### What it buys

Spatz cycles for the timed op, everything else unchanged:

| op                     | autovectorized | hand-written RVV |        |
|------------------------|----------------|------------------|--------|
| MatMul                 | 15.8k          | 6.8k             | 2.3x   |
| mymatmul (32x32x32)    | 57.6k          | 7.5k             | 7.6x   |
| GEMM/Regular           | 60.9k          | 8.4k             | 7.3x   |

Transposed operands break the unit-stride `B` load and fall back to the Deeploy
kernel, which stays reachable as `<name>_generic` exactly as it does for
Snitch. Conv2D is untouched and still autovectorized, which is why Snitch keeps
that row.

## The memory system

Each core keeps the memory architecture it actually has — CVA6 caches, Snitch and
Spatz a cluster scratchpad — and all three sit behind the same main memory, so a
cycle difference between them is a difference in how the core uses memory and not in
how the memory was configured. Fixed-path policy and selection live in
[`targets/hetero/memsys.py`](targets/hetero/memsys.py); modeled-device timing
presets live in
[`targets/hetero/dram_presets.py`](targets/hetero/dram_presets.py).

```
cva6_real     fetch ── L1 I$ 16K/4-way ─┐
                                        ├─ L2 512K/8-way ── DRAM (100 cycles, 8 B/cycle)
              data  ── L1 D$ 32K/8-way ─┘
                       write-through, 8-entry store buffer
                       (peripherals are mapped uncached)

snitch_real   fetch ── cluster I$ (L0 per core + 8K shared) ── HBM (100 cycles)
              data  ── banked TCDM, single cycle, bank conflicts modelled

spatz_real    same as snitch_real, plus the Spatz VLSU ports into the TCDM
```

Cache geometry follows the CVA6 defaults (16 KiB 4-way instruction cache, 32 KiB
8-way write-through data cache) with one deliberate deviation: 64-byte lines instead
of the 128-bit lines of the RTL, because the GVSoC ISS fetches instructions in
64-byte granules and a shorter line would turn one fetch into several refills the
real front-end never issues.

### The cache model

`targets/hetero/timing_cache.cpp` is a cache that models timing only. It holds no
data: every access is forwarded to the next level, so a store can never be lost in a
line that gets evicted and a read can never return something main memory does not
hold. What it owns is the latency — it keeps a tag array, decides hit or miss, and
rewrites what the master sees:

- **hit** → the hit latency. Read hits are forwarded to the next level marked as debug
  accesses, so the level below serves the bytes without counting them: its tag array
  and its counters only ever see this cache's miss stream.
- **miss** → the next level's latency plus the line transfer, with refills serialized
  through the port to that level, so a burst of misses pays bandwidth and not only
  latency.
- **store** → write-through into the store buffer, which drains to the next level.
  The core stalls only once the buffer is full. A store that misses does not pull the
  line in, matching CVA6's write-through L1.

Replacement is LRU. The `[HES-MEM]` line each cache prints at the end of a run is what
the report's cache table shows; those counters cover the whole program, while `cycles`
covers the timed op alone.

### What it costs

Same binaries, same ops, `--memory real` against `--memory ideal`:

| op                     | cva6                | snitch            | spatz              |
|------------------------|---------------------|-------------------|--------------------|
| Add/Regular            | 715 → 863 (+21%)    | 1053 (unchanged)  | 635 → 1035 (+63%)  |
| Conv/Regular_2D_Bias   | 1.74M → 1.86M (+7%) | 174k (unchanged)  | 452k → 457k (+1%)  |
| GEMM/Regular           | 473k → 489k (+3%)   | 43.5k (unchanged) | 6.36k → 8.36k (+31%)|
| MatMul                 | 118k → 119k (+1%)   | 12.6k (unchanged) | 5.02k → 6.82k (+36%)|
| Softmax/Regular        | 34.1k → 36.5k (+7%) | 51.0k (unchanged) | 30.4k → 32.5k (+7%)|
| Integer GEMM/Regular   | 575k → 576k (+0.2%) | 452k (unchanged)  | 82.4k → 84.1k (+2%)|

Snitch does not move because GVSoC's Snitch board already charged 100 cycles for HBM,
which is the DRAM latency the three targets now share; its instruction fetches do pay
it, and raising `DRAM_LATENCY` to 1000 lengthens the Conv run by 25% (it was 4% before
the SSR/FREP kernels: the compute shrank 6.5x, the instruction refills did not, so the
same memory system now costs proportionally more). Spatz moves
because its board mapped HBM with *zero* latency, so instruction fetches that missed
the 8 KiB cluster cache used to refill for free. CVA6 moves because it had no memory
system at all.

Spatz's MatMul and GEMM rows moved from +2% to +31-36% when its kernels were
hand-written: the compute shrank ~7x while the instruction refills did not, so
the same memory system now costs proportionally far more. That is the same
effect the Conv/`DRAM_LATENCY` note above describes for Snitch, and it is the
general shape of the thing — the better the kernel, the larger the share of
what remains that is memory.

The single-op kernels stay compute-bound: their working sets are at most 128 KiB by
construction (they have to fit the Snitch TCDM), and the operands are staged into
local memory before the timed region on every core — `bench_main` copies the inputs
into the network buffers, which warms the CVA6 caches the same way `crt0` fills the
TCDM. What the modelled memory system charges for is therefore the steady-state cost:
capacity misses, store traffic and instruction refills, not the cold start.

### Modeled device memories (`--dram`)

`DRAM_KIND` in [`memsys.py`](targets/hetero/memsys.py) (or `--dram` on `run.py`,
`run_hetero.py` and `sweep/run.py`, or `DRAM=` on `make run/hetero/mnist/kws`)
replaces the constant main-memory latency with a modeled device-timing instance:

| kind | device | source of the numbers | unloaded line read | peak |
|---|---|---|---|---|
| `fixed` | -- (the default) | -- | 100 cycles | 8 B/cycle |
| `lpddr4` | LPDDR4-3200, x16, 8 banks | DRAMSys memspec `JEDEC_8Gb_LPDDR4-3200_16bit` | 71.9 ns = 72 cycles | 6.4 GB/s |
| `lpddr4x` | LPDDR4X-4266, x16, 8 banks | DRAMSys memspec `JEDEC_1Gbx16_LPDDR4-4266` | 66.4 ns = 67 cycles | 8.5 GB/s |
| `lpddr5` | LPDDR5-6400, x16, 4 bank groups x 4 | Ramulator2 `LPDDR5_6400` / `LPDDR5_8Gb_x16` (JESD209-5C) | 68.8 ns = 69 cycles | 12.8 GB/s |
| `hyperram` | HyperRAM 2.0, x8 @ 200 MHz | Infineon S27KS/S70KS data sheets | 265 ns = 265 cycles | 0.4 GB/s |

The line read includes a 20 ns controller + PHY allowance (`ctrl_ps`, an
assumption, overridable like every other field through `DRAM_OVERRIDES`).
Cycles are at the 1 GHz core clock (`FREQUENCY`, which was a nominal 10 MHz before
anything in the model depended on time; with `fixed` the change is cycle-neutral,
checked against the committed MatMul per-core, MatMul SoC and KWS baselines).

What [`dram_core.hpp`](targets/hetero/dram_core.hpp) models: per-bank open rows
(open-page policy) with tRCD/tRP/tRAS/tRC, tRRD and the tFAW window, tCCD
(short/long across bank groups), read-to-write and write-to-read turnarounds,
write recovery, one data bus per channel, all-bank refresh every tREFI for tRFC
(scheduled before any access that would run into it), and a 32-entry write
queue: writes are posted, go to the DRAM when it would otherwise idle or in a
batch past a watermark, and a read that hits a queued write is answered from it.
Reads are served in arrival order -- the caches above miss in order, so a
reordering controller would rarely have anything to reorder -- and each request
is widened to the 64-byte line it refills. HyperRAM is a single bus: command/address phase,
fixed double initial latency (the device default), two bytes per clock on x8,
transfers split at tCSM. Not modelled: FR-FCFS reordering, per-bank refresh,
power-down, command-bus contention, the LPDDR5 two-part ACT.

Where it sits: between the L2 and main memory on the CVA6/Ara boards, on the wide
AXI port where host refills and cluster DMA meet on the SoC, and between the chip
and HBM on the stock Snitch/Spatz boards. The ELF loaders bypass it (they would
otherwise leave it busy when the cores start), and debug accesses -- including
the hits the timing caches forward to fetch bytes -- pass through untimed, the
lesson of the router bandwidth bug in the Ara section.

With a device, the L2 also becomes **write-back** (`L2_WRITEBACK`): only refills
and dirty evictions reach main memory. Under the original write-through L2 every
store would pay a DRAM write burst and a bus turnaround, which is a model artefact,
not a memory cost. The fixed model keeps write-through so its numbers stay what
they were.

Same binaries, `--dram` across the board (MatMul per core; MNIST and KWS on the
whole SoC, per sample):

| | fixed | lpddr4 | lpddr4x | lpddr5 | hyperram |
|---|---|---|---|---|---|
| MatMul, cva6 | 119,037 | 118,895 | 118,880 | 119,088 | 119,744 |
| MatMul, snitch | 12,603 | 11,857 | 12,169 | 11,849 | 15,408 |
| MatMul, spatz | 6,815 | 6,005 | 5,915 | 6,121 | 9,791 |
| MatMul, ara | 74,478 | 74,124 | 74,024 | 73,952 | 84,758 |
| MNIST, cycles/image | 570,385 | 580,927 | 578,073 | 574,748 | 750,740 (+32%) |
| KWS, cycles/clip | 256,652 | 261,545 | 259,822 | 256,840 | 395,141 (+54%) |

Two things show. LPDDR is close to the fixed model's 100-cycle guess for a lone
miss but different in shape: the cluster DMA streams that dominate MNIST and KWS
find their rows open 93-98% of the time and run at bus speed, which the fixed
model gets slightly wrong in both directions (cheaper per access, but 6.4 B/cycle
of LPDDR4 instead of the AXI's 64). And HyperRAM is what a memory system that is
actually slow looks like: 0.4 GB/s turns KWS, whose two clusters stream weights
and features at once, 54% slower, while a CVA6 kernel that lives in its 512 KiB
L2 barely notices.

How it is checked: `make dram-test` runs hand-computed JEDEC cases through the
engine (row hit/miss/conflict, tRRD, tFAW, tWTR, tRTW, refresh, HyperBus latency
and tCSM) and checks that the latency the header generator derives for every
preset equals what the engine does. `make dram-xcheck` replays synthetic access
patterns through DRAMSys (LPDDR4/4X, from the same memspec files) and Ramulator2
(LPDDR5, from the same presets) and compares read latencies request by request:

| pattern | compared | LPDDR4 vs DRAMSys | LPDDR4X vs DRAMSys | LPDDR5 vs Ramulator2 |
|---|---|---|---|---|
| `sequential` | latency | +8.3% | +5.3% | +5.2% |
| `bank_rotate` | latency | +15.4% (+0.3%\*) | +16.2% (+0.3%\*) | +7.1% |
| `row_conflict` | latency | -1.5% | -1.0% | +15.5% (-0.0%\*) |
| `random` | latency | +3.6% | +5.5% | +2.4% |
| `read_write` | latency | +5.0% | +3.5% | -2.9% |
| `stream` | throughput | -0.3% | +1.2% | -0.4% |
| `bank_rotate_sat` | throughput | -2.4% | -3.3% | -0.2% |
| `read_write_sat` | throughput | -8.5% | -8.2% | +0.9% |

Our engine minus the reference: mean read latency on the patterns that leave
the DRAM time to breathe, sustained read bandwidth on the saturating ones (where
latency only measures how deep a queue each tool lets build). Outside the three
starred cells everything is within 10%, and with the reads that arrive around a
refresh left out, every latency pattern is within 5% (the figure in brackets for
the starred ones). What remains is one behaviour, not a timing error: both
references reorder the queue that piles up behind a refresh -- DRAMSys's FIFO is
per bank, Ramulator2 is FR-FCFS -- so row hits on banks that are already open
overtake older requests still waiting for their ACT. This model fixes a request's
latency when it arrives and serves the backlog in order, which errs on the slow
side for a few hundred nanoseconds every 3.9 us.

The cross-check is also what fixed the presets: it showed LPDDR4 reads missing
the four clocks of the RD-1/CAS-2 command and tDQSCK, LPDDR5 accesses missing
the CAS (WCK-sync) clock and the longer same-bank-group turnarounds, and a
strictly in-order controller paying a turnaround for every write -- which is why
the presets now have a write queue (posted writes, drained when the DRAM is idle
or in a batch past a watermark, as Ramulator2's controller does). HyperRAM has
no open cycle-level reference; its arithmetic is in the unit tests.

`make dram-xcheck` builds the references into their own image
([`tools/dram/xcheck/Dockerfile`](tools/dram/xcheck/Dockerfile), ~3 GB); with
less disk, run [`build_refs.sh`](tools/dram/xcheck/build_refs.sh) inside the
hetero-sim dev container (~300 MB) and then `tools/dram/xcheck/run.py`, which is
how the table above was produced.

Why not the DRAM models GVSoC already ships: its DRAMSys bridge (`memory.dramsys`)
needs a SystemC build and answers reads asynchronously while ignoring debug
accesses, which the timing caches depend on; its Ramulator bridge
(`memory.ramulator`) speaks only the newer io_v2 beat protocol, and everything in
this hierarchy is io v1 with no bridge between the two; its HyperRAM model is a
pin-level modeled device behind PULP's uDMA, not something a core can load from. Both
simulators are used here instead as references, offline.

## Caveats

- Cache counters in the report cover the whole program (startup, timed op, output
  check); the cycle counts next to them cover the timed op only.
- Ops must fit in 128 KiB TCDM (data + heap + 8 KiB stack) for Snitch/Spatz.
- `minstret` is not implemented by these GVSoC core models; instruction counts
  are reported when available.
- Spatz executes whatever RVV reaches it; instructions the GVSoC model does not
  implement trap and are reported as a failed run rather than a wrong number.
  Four gaps hit by autovectorized code are fixed in `deps/patches/` — see
  `## GVSoC model fixes` below.
- The cores do not run identical code, on purpose: the numbers are meant to be
  each core at its best, not a controlled experiment on one source. Snitch uses
  Xssr/Xfrep, Spatz uses hand-written RVV, CVA6 runs the scalar kernels.
  `--spatz-kernels autovec` swaps Spatz's back for the compiler's output, which
  is how the comparison below was measured.
- The standalone operator table above uses one core of each type and therefore
  does not measure cluster scaling. That scope must not be generalized to all
  M4IA execution: `run_hetero.py` has a host/cluster runtime, Snitch and Spatz
  core counts are configurable design dimensions, and application paths such
  as KWS use cluster execution and cross-cluster concurrency. Per-core tiling
  and utilization still need to be stated for each heterogeneous experiment.
- The vector host's Ara unit reaches memory through the host's cache hierarchy
  on a single port, and issues gathers and strided accesses one element per
  burst. That is how the model is built rather than a property of every vector
  core, and it is why Conv2D is slower on it. ISS v2's semi-hosting has been
  seen to drop a string write and repeat the previous one, which garbles a log
  line but never a result: the progress beacon carries its engine as a number
  for that reason.

## GVSoC model fixes

`setup.sh` applies every `deps/patches/gvsoc-core-*.patch` to `deps/gvsoc/core`.

### CVA6 load latency

`gvsoc-core-cva6-load-latency.patch`. The CVA6 model runs each instruction's handler
at commit and then marks its destination registers ready in that same cycle — which
overwrote the timestamp the LSU had just written for a load, discarding the latency of
every data access. Memory delays reached the core's fetch path but never its loads: a
10-cycle L1 hit latency changed a run by exactly zero cycles. The commit stage now
keeps whichever is later, so a dependent instruction waits for the memory system.

This also affects `--memory ideal`, where a load takes one cycle to come back rather
than none. It moved exactly one number in the committed baseline: Conv on CVA6, by 5
cycles out of 1.74M.

### Spatz vector model

Autovectorized kernels exercise the Spatz model harder than the hand-written
benchmarks it ships with. `gvsoc-core-rvv-extensions-and-fixes.patch` carries four
fixes:

1. **Missing instructions** — added `vsext`/`vzext.vf*`, `vwadd`/`vwsub[u].w{v,x}`
   and `vmv<nr>r.v`; implemented `vl<nr>r.v` / `vs<nr>r.v`, which were `abort()` stubs.
2. **Stale `vtype` on deferred execution** — the VPU runs an instruction's functional
   handler at *completion* but read the live `vl`/`vtype`, which a later `vsetvli` on
   the scalar core had already changed. The configuration is now captured at dispatch
   and restored around the handler and in the VLSU.
3. **`vmv.x.s` writeback race** — declared with a vector-output format, so its integer
   destination was never scoreboarded and the scalar core read a stale register.
4. **TCDM bank-crossing bursts** — the Spatz VLSU issued full-width bursts from
   misaligned addresses, crossing a bank boundary and silently corrupting data (this
   was the Conv2D failure). Bursts are now clamped at the interleave boundary.

### Ara vector host

`gvsoc-core-vector-host-ara.patch`. The CVA6 + Ara host hung once MNIST grew past
16 images and returned wrong numbers on every kernel GCC vectorized densely. Each of
these five was reproduced before it was fixed: the first by tracing the hung MNIST
run, which the 64-image run on `hetero_ara` still exercises, and the other four by
the `make ara-test` cases that guard them now.

1. **Router bandwidth charged to debug accesses** — the timing cache fetches the
   bytes of every hit from the level below as a debug access, and the router's
   bandwidth limiter advanced its burst cyclestamp for those too. On the SoC boards,
   where `wide_axi` is bandwidth-limited, hit traffic pushed the cyclestamp ahead of
   simulated time; a CVA6 stalled on a full vector queue re-issues two fetches a
   cycle, so every wait doubled the next — 8M, 16M, 33M, 67M cycles, which looked
   like a hang. Debug accesses now cross a router without occupying its bandwidth.
   Against the unfixed router this moves each committed scalar-host SoC run by
   about 30 cycles.
2. **`vid.v`** had no decoding at all, which was the MatMul illegal-instruction
   trap.
3. **Strided and indexed accesses** — the Ara VLSU issued every load and store as
   one contiguous range from the base address, so `vlse`/`vsse` and the `vluxei`
   gathers GCC builds matmul and convolution on read the wrong memory. They are now
   issued an element per burst, the address advancing by the stride or read from
   the offset vector, as the Spatz VLSU does.
4. **`vsetvli zero, zero`** reset `vl` to VLMAX instead of keeping it: a write to
   `x0` is decoded to a discard register at `ISS_NB_REGS`, so the handler never saw
   `rd` as `x0`. It is the form a compiler uses to change SEW or LMUL mid-loop.
5. **Scalar operands read late** — the vector unit runs an instruction some cycles
   after the core issued it, and nine handlers read their scalar operand from the
   live register instead of the value captured at dispatch. `vmv.v.x v9, t1` ran
   after the core's `vsetvli t1, zero` and made the integer GEMM's B offset VLMAX.
   Eight of the nine are fixed here; `vw_op_wx_exec`, which
   `gvsoc-core-rvv-extensions-and-fixes.patch` added itself, is fixed in that patch
   so it still reverse-applies.

The last two are in handlers the Snitch and Spatz cores share. Every committed
Snitch and Spatz result is unchanged to the cycle: neither core's code reached
them.
