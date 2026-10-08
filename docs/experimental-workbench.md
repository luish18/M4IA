# M4IA Experimental Workbench

The Experimental Workbench is the experiment-facing layer around the existing
M4IA execution stack. It makes experiments discoverable, deterministic,
auditable, and reproducible without becoming a second simulator, mapper,
runtime, or configuration system.

Its core rule is:

> Minimal input, deterministic resolution, explicit ambiguity, provenanceable
> results.

## Why it exists

M4IA already had CVA6/Snitch/Spatz target construction, Deeploy generation,
heterogeneous mapping, specialized kernels, host/cluster dispatch, GVSoC
execution, calibration, sweeps, applications, and a GUI. Experimental facts,
however, were distributed across defaults, build choices, generated code,
runtime dispatch, calibration files, and results.

The Foundation groups its contribution into three responsibilities:

1. **Researcher interface and discovery** — `ExperimentRequest`, deterministic
   resolution, catalogs, and query APIs.
2. **Auditability and extension structure** — resource summaries, mapping
   explanations, implementation evidence, fingerprints, provenance, and
   manifests.
3. **Scientific safety** — unknown remains unknown; missing cost is not zero;
   fallback is not tuned execution; stale or contradictory artifacts are not
   silently accepted.

## The experimental contract

Current M4IA preserves five distinct layers:

```text
REQUESTED -> RESOLVED -> GENERATED -> ACTUAL -> MEASURED
```

### Requested

`ExperimentRequest` records researcher intent:

- platform and workload;
- numeric design overrides;
- host profile;
- mapping strategy and optional compatible-node pin;
- frontend, serial and power controls;
- main-memory kind and accepted DRAM overrides.

It is a strict JSON/API boundary. Unknown fields and semantically invalid
types are rejected. A request intentionally omits derived defaults.

### Resolved

`resolve_experiment()` validates and expands the request into:

- full numeric design, slug, build key, and hardware fingerprint;
- explicit host profile and simulator target;
- explicit memory identity;
- workload package identity and descriptors;
- `ResourceSummary`;
- mapping and execution semantics;
- one deterministic resolved fingerprint.

Omitted main memory and explicit `fixed` have the same resolved identity.
Unsupported values fail; they never silently resolve to fixed.

### Generated

Generated evidence comes from the actual mapper/code-generation path. Current
node records may include:

- generated index, node, operator and engine;
- structured mapping explanation;
- emitted MatMul/Gemm arguments that survived Deeploy binding.

Generated evidence proves what was emitted, not that execution completed.

### Actual

Actual evidence is projected only from completed runtime nodes. Audited
MatMul/Gemm implementation identity requires a matching completion beacon,
generated engine/operator metadata, required arguments, and the compiled
dispatch rules. Missing or contradictory evidence remains unknown.

### Measured

Measured evidence contains observations such as status, total and node cycles,
correctness, accuracy, waits, cache/DRAM counters, and wall time. Measurement
scope is part of the meaning; see [Metric semantics](metrics.md).

## Discovery and query API

`pipeline/gui_query.py` is a thin JSON bridge. Existing commands remain
available, and two Foundation commands are query-only:

```bash
.venv/bin/python pipeline/gui_query.py experiment-schema
.venv/bin/python pipeline/gui_query.py resolve-experiment request.json
```

`experiment-schema` exposes platform preset IDs in `presets`, plus parameters,
engines, host profiles, mapping strategies, kernel implementations, and memory
kinds. Memory discovery is ultimately derived from the operational DRAM preset
source rather than another list in the bridge.

`resolve-experiment` uses `ExperimentRequest.from_dict()` and the accepted
resolver. On success stdout contains one JSON document. Invalid JSON or request
semantics produce a JSON `error` object with nonzero exit and no build or
simulation side effect.

## Catalogs and operational truth

Catalogs are descriptive shadows of existing operational paths:

| Catalog | Purpose | Operational owner |
|---|---|---|
| Parameters | Names, units, defaults, screening metadata | `pipeline/sweep/design.py` and target consumers |
| Engines | Stable logical identities | target/runtime IDs and progress macros |
| Hosts | Experiment-facing host profiles | `pipeline/build_mesh.py::HOSTS` |
| Mapping | Stable identity of current strategy | `pipeline/hetero_platform/mapper.py` |
| Kernels | Audited source/build/dispatch paths | runtime/build/kernel sources |
| Memory | Supported kinds and override surface | `dram_presets.py`, `memsys.py`, `common.py` |

A catalog row cannot create support. Operational code remains authoritative if
metadata and behavior ever disagree.

## ResourceSummary

`ResourceSummary` derives compute/resource geometry from the resolved numeric
design. It distinguishes facts such as modeled cores, compute cores,
control/DMA cores, and modeled versus useful Spatz lanes.

It deliberately excludes main-memory kind and L2 policy: those are separate
resolved memory facts. It is not an area or power model and does not prove
resource-normalized equivalence.

## Mapping safety and explanation

The stable current strategy ID is `measured_rate_greedy`. Automatic selection
considers only compatible candidates with a defensible numeric cost. Safety
rules include:

- unknown tensor extent does not become zero bytes;
- incomplete shape-derived MAC count is unavailable;
- unknown engines do not borrow CVA6 rates;
- missing/non-positive rates and missing offload terms remain unavailable;
- an explicit `_default` rate is labeled as a proxy;
- external `HES_RATES` replaces rather than partially merges the active table;
- known Generic fallback is not priced with a tuned cluster rate;
- when no compatible candidate can be safely costed, no automatic selection
  is fabricated.

Mapping explanations report compatibility, pin filtering, safe/unavailable
cost evidence, rate source, selection rule, and selected engine. They describe
the decision; they do not influence it.

`--pin` retains compatible-node semantics. It is not a whole-graph promise.

## Implementation and fallback evidence

Stable implementation IDs cover the currently audited FP32 MatMul/Gemm paths:

```text
cva6.fp32.matmul_gemm.deeploy_generic
snitch.fp32.matmul_gemm.ssr_frep
snitch.fp32.matmul_gemm.deeploy_generic_fallback
spatz.fp32.matmul_gemm.rvv_tuned
spatz.fp32.matmul_gemm.deeploy_generic_fallback
```

The C/runtime implementation is operational truth. The shared Python helper is
an audited interpretation used by mapper safety and post-run classification.

Current rules include:

- Snitch empty required dimensions or `O < 8`: whole-kernel Generic fallback;
- Snitch `O >= 8`: tuned SSR/FREP; a scalar column tail is still the tuned
  implementation, not whole-kernel fallback;
- Spatz empty dimensions: whole-kernel Generic fallback;
- Spatz Gemm transpose: Generic fallback;
- CVA6 completed MatMul/Gemm: current Deeploy Generic host identity.

Unsupported operators and insufficient evidence remain unknown.

## Identity model

Different artifacts need different identities:

```text
numeric design identity
    design.slug

machine identity
    numeric design + host + resolved memory

calibration identity
    machine + protocol + causal calibration sources/dependencies/toolchain/simulator

run-input identity
    resolved experiment + execution controls + calibration fingerprint
    + causal run-source digest

completed-run identity
    run input + generated + actual + measured + linked artifact digests
```

Calibration is workload-independent. A workload change affects experiment and
run identity but does not invalidate a compatible characterization of the same
machine. Fixed and real memories, host profiles, and operational DRAM overrides
cannot share calibration identity when behavior differs.

The pre-run `run_input_fingerprint` addresses a cell. Measurements cannot
participate in this path key. The post-run `run_fingerprint` may change when
generated, actual, measured, or linked artifact evidence changes.

## Causal provenance

Calibration and run provenance answer different questions:

- calibration provenance: what produced these reusable rates?
- run provenance: what causal first-party source state produced this run?

Source sets are selected rather than whole-repository hashes. Documentation,
GUI files, results, and work outputs are excluded. Host- and memory-specific
sources participate only when selected. C++ model files are covered where
causal. Repository remote URLs are descriptive provenance, not semantic
identity.

## Calibration cache contract

A calibration cache hit requires compatible `rates.json` and
`calibration.json`, including schema, kind, protocol, internally consistent
input identity/fingerprint, expected fingerprint, and rates digest. Legacy
rates without metadata and contradictory/tampered sidecars are misses.

Calibration metadata is written only after successful rates and raw-log
production. The resolved host must agree with the calibration host image,
GVSoC target, and rate parser.

## Run manifest

Sweep cells produce `kind: "m4ia.run"` manifests with first-class:

```text
requested  resolved  generated  actual  measured
```

The manifest also records machine/run identities, calibration linkage and
provenance, run provenance, and the linked result digest. Construction rejects
contradictory artifact keys, experiment/machine fingerprints, calibration
identity, runtime host, or runtime memory.

Partial or stalled runs that produced a genuine result may receive a manifest;
actual evidence still contains only completed nodes. A failed invocation that
produced no new result does not receive a fabricated manifest.

## Stale-state protections

The sweep path:

- addresses machine artifacts by host and resolved memory as well as numeric
  design;
- replaces the copied runtime mesh before regenerating headers;
- reuses calibration only through the sidecar contract;
- addresses cells by pre-run identity;
- removes stale `result.json` and `manifest.json` immediately before launch;
- isolates mutable runner work by artifact key.

## Scientific-use guardrail

Configurable does not mean causally validated. Before using a knob as a
scientific axis, establish:

```text
request/override
    -> resolved design
    -> HES_DESIGN/generated header/target configuration
    -> GVSoC construction
    -> runtime-visible consequence
    -> measured behavior
```

Discovery and fingerprint changes prove description/identity propagation, not
the simulator consequence by themselves.

## Reproducible comparison checklist

Before comparing results:

1. identify the intended request difference;
2. compare resolved experiment, host, memory and machine identity;
3. verify workload identity;
4. inspect generated mapping rather than inferring it from a pin;
5. inspect completed implementation/fallback evidence;
6. verify calibration identity and cache status;
7. verify causal run provenance;
8. gate on correctness/feasibility;
9. compare measurements only within their documented scope;
10. preserve manifest and linked result/calibration artifacts together.

## Deliberate non-claims

The Foundation does not establish that:

- it created the underlying SoC, runtime, mapper, or tuned kernels;
- equal cores or lanes imply equal compute, area, or power;
- every exposed knob is causally validated;
- the current mapper is globally optimal;
- placement proves a tuned implementation;
- node cycles are pure kernel cycles;
- `area_au` is physical area;
- `cache_dynamic_pj` is total energy;
- GVSoC is equivalent to RTL synthesis or silicon.

See the [quickstart](experimental-workbench-quickstart.md),
[extension guide](extending-m4ia.md), and historical
[Foundation checkpoints](dev/foundation-checkpoints.md). The maintained
[research roadmap](dev/research-roadmap.md) sequences later scientific work.
