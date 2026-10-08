# M4IA extension-seam audit

> **Technical audit and rationale.** This document explains why an extension
> seam is ready, fragile, or blocked. It is not a second configuration system
> and not the step-by-step recipe. Operational source remains authoritative.
> For practical procedures, see [Extending M4IA](../extending-m4ia.md).

## Classification

- **ready** — a new instance has a local path and existing checks.
- **ready with local change** — the path is bounded, but named operational and
  experiment-facing sites must change together.
- **fragile** — manually coupled sites still require an end-to-end audit.
- **blocked** — no defensible current operational contract exists.

The classification concerns the complete path. A catalog entry or GUI option
alone never establishes support.

## Workload or model — ready with local change

`pipeline/common.py` and `pipeline/experiment/workload.py` resolve the consumed
ONNX package and its identity; Deeploy and application generation own actual
execution. Discovery is path-based rather than a second workload registry.

A new package must preserve requested spelling, resolved content identity,
tensor/shape facts, generated code, correctness data, and run provenance. A new
operator additionally needs a real binding/kernel path. Resolver, workload,
manifest, and a real correctness smoke are the important checks. GUI discovery
may expose the package, but that exposure cannot create operator support.

## Numeric hardware knob — ready with local change

`pipeline/sweep/design.py` owns numeric defaults, validation, slug, build key,
and sweep grids. `targets/hetero/system.py`, `targets/hetero/soc.py`, and
`pipeline/gen_system_header.py` own operational consequences.
`parameter_catalog.py`, resolution, `ResourceSummary`, and calibration/run
provenance describe the result.

The seam is bounded, but a knob is scientifically usable only after tracing it
through target construction and runtime-visible behavior. Parameter metadata
has a drift check; it is still descriptive. `design.slug` must remain numeric
design identity rather than absorbing host, memory, or workload.

## Main memory — ready with local change, cross-cutting

Operational ownership is split deliberately:

- `targets/hetero/dram_presets.py` defines public device presets;
- `pipeline/common.py::use_dram()` applies the selected `DRAM_KIND`;
- `targets/hetero/memsys.py` selects fixed/device memory and L2 policy;
- `targets/hetero/dram.py`, `dram.cpp`, and `dram_core.hpp` implement real
  device behavior.

Memory is explicit experiment and machine identity, not a numeric design knob.
Adding a kind therefore also requires resolution/discovery, effective preset
and override identity, calibration separation, conditional causal provenance,
artifact addressing, result/manifest metadata, tests, and later GUI discovery.
The current source-derived kind list prevents a separate Python list, but
operational timing correctness remains a model-validation responsibility.

## Kernel implementation — ready with local change

The concrete seam is source and symbol, build override in `build_mesh.py`,
runtime dispatch, Deeploy binding, fallback, catalog ID, and completed-runtime
evidence. Current catalog coverage is intentionally limited to audited FP32
MatMul/Gemm paths.

A new ID is justified only after the complete path exists. Calibration must not
label an engine/operator rate as implementation-specific unless the protocol
actually binds that implementation. Generated arguments and runtime completion
must be sufficient for any new actual-evidence rule.

## Mapping strategy — fragile

`mapping_strategy_catalog.py` gives the current strategy the stable
`measured_rate_greedy` ID, while `mapper.py::make_mapper()` and
`generate.py::build_deployer()` construct the only operational strategy.
Resolution can name the strategy, but generation does not yet select among a
registry of mapper factories.

A second catalog row would therefore be unsafe today. A future strategy must
first become an explicit runner/generator input bound to a vetted factory, then
extend explanations, identity, tests, and manifests. No generic plugin registry
is warranted while only one strategy exists.

## Host or simulator profile — fragile

`pipeline/build_mesh.py::HOSTS` owns the host image and matching GVSoC target.
The host catalog derives from that table, but sweep calibration, run-source
selection, host rates, build settings, CLI choices, and manifest evidence are
coupled around the two current profiles.

A third profile requires an image/target pair plus an audit that resolved host,
calibration build image, calibration target/parser, final target, provenance,
and GUI choice agree. The CP7 Ara calibration bug demonstrates why adding a
catalog row alone is insufficient.

## Metric — ready with local change; physical PPA remains blocked

Adding a scoped observation or derived field is bounded when the producer,
unit, scope, classification, parser, result/manifest projection, sweep row, and
tests agree. See [Metric semantics](../metrics.md).

Claims of physical area or total SoC energy are not ready: current `area_au` is
an estimated structural proxy with unsourced coefficients, and
`cache_dynamic_pj` covers modeled cache dynamic energy only. New labels must
not strengthen those claims without a source-backed model and validation.

## Precision — blocked

There is no first-class precision request. Compatibility, tensor bytes,
kernels, job kinds, calibration workload, rate rows, implementation IDs, and
correctness expectations are currently FP32-specific.

A future precision contract must propagate through request, resolution,
working set, engine capability, kernel dispatch, calibration and rate identity,
actual evidence, and measured error/accuracy. Unknown precision cannot silently
mean FP32.

## Future execution engine — fragile

Engine identity begins in `targets/hetero/system.py::ENGINE_NAMES` and runtime
beacon macros in `pipeline/hetero_platform/progress.py`. A real engine also
needs target/model, image and ELF handling, mailbox/dispatch, Deeploy engine and
capability rules, safe calibrated costs, resource description, kernels,
fallbacks, evidence, results, provenance, and integration validation.

Current catalog drift checks prevent name mismatch; they do not remove the
explicit Snitch/Spatz handling throughout build and runtime.

## GUI-facing capability — ready with local change

The backend contract is ready for incremental consumption:

```text
experiment-schema -> ExperimentRequest -> resolve-experiment
                  -> run/sweep -> m4ia.run manifest/result
```

Implement semantics in Python first, expose source-derived discovery and
strict resolution, then make Rust a consumer. Current Rust enums remain a
temporary UI representation. A frontend list becomes unsafe only when it is
treated as operational truth or can select behavior the backend does not
validate.

## Cross-cutting invariants

Every extension must preserve:

```text
REQUESTED != RESOLVED != GENERATED != ACTUAL != MEASURED
catalog/discovery metadata != operational support
unknown != zero/default/supported
proxy != physical quantity
```

It must identify its operational owner, discovery surface, request/resolution
identity, calibration/run provenance, generated and actual evidence, metrics,
tests, and GUI implications. Hashing the whole repository is not a substitute
for selecting causal sources.
