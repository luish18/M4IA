# Extending M4IA safely

This is the practical recipe guide for current M4IA. The companion
[extension-seam audit](dev/extension-seams.md) explains why each seam is
classified as ready with local change, fragile, or blocked.

The governing rule is:

> Operational support first; experiment metadata and discovery follow it.

A catalog row cannot create simulator, runtime, kernel, or mapping behavior.

## Adding a workload or model

1. Add the consumed ONNX package in the existing operation layout.
2. Verify `pipeline/run.py::resolve_test_dir()` and the corresponding
   `pipeline/run_hetero.py` workload/application path select it correctly.
3. Add application generation/runtime support only when required.
4. Ensure `pipeline/experiment/workload.py` includes every consumed artifact
   and excludes docs, logs, results, and work outputs.
5. Exercise `gui_query.py inspect` and `resolve-experiment`.
6. Verify direct run and sweep compatibility.
7. Add workload identity tests and a correctness smoke.

Changing a consumed input/output/data header must change workload identity;
changing an unrelated README must not.

## Adding a hardware design knob

1. Add the numeric default to `pipeline/sweep/design.py::DEFAULTS`.
2. Add explicit validation and build-time classification where applicable.
3. Propagate the resolved value through the actual target/header/build path.
4. Add descriptive metadata to `parameter_catalog.py`; keep its drift check.
5. Update `ResourceSummary` only for a defensible derived resource fact.
6. Audit design, machine, calibration, and run-input identity.
7. Add resolver, validation, propagation, and stale-cache tests.
8. Before a scientific campaign, prove the request-to-measurement causal path.

Do not add memory to `DEFAULTS`: main-memory kind is a separate experiment
dimension. Do not assume a discovery entry proves target consumption.

## Adding a memory kind or preset

1. Implement the operational behavior in the memory subsystem.
2. Extend the authoritative `KINDS`/preset/field definitions in
   `targets/hetero/dram_presets.py`.
3. Confirm `pipeline/common.py::use_dram()` and
   `targets/hetero/memsys.py` consume it.
4. Extend descriptive labels only with non-drifting wording or values derived
   from the operational preset.
5. Resolve its complete preset/override identity through `resolve_memory()`.
6. Preserve numeric-only `design.slug` semantics.
7. Include the memory in machine, calibration and run identities.
8. Audit host/L2/device construction and result/counter semantics.
9. Add discovery, override, collision, provenance, and smoke tests.

If a model uses new C/C++ files, include them in the relevant causal source
sets. Fixed-memory runs should not be invalidated by modeled-device-only sources.

## Adding a kernel implementation

Audit the complete operational path:

```text
source -> public symbol -> build override -> generated call
       -> runtime dispatch -> fallback
```

Then:

1. implement and test the runtime kernel;
2. define exact eligibility and fallback conditions;
3. add a stable implementation ID to the catalog;
4. prevent known fallback cases from using tuned cost;
5. capture only generated integer arguments actually emitted by Deeploy;
6. classify implementation only for matching completed runtime evidence;
7. keep missing/contradictory evidence unknown;
8. add tuned, tail, fallback, mismatch, and correctness tests.

The Python dispatch interpretation must remain aligned with C, which stays
operational truth.

## Adding a mapping strategy

This seam is fragile while generation has one operational factory.

1. Implement the strategy in the operational mapper.
2. Bind a stable request/catalog ID to a vetted factory selection path.
3. Define compatibility, unavailable-cost, tie, and pin semantics.
4. Make the decision explainable without using explanations as policy input.
5. Preserve unknown working sets/MACs/rates as unavailable.
6. Add automatic-selection, no-safe-cost, pin, and regression tests.
7. Expose the strategy only after a request can actually select it.

Adding only a row to `mapping_strategy_catalog.py` is not support.

## Adding a host or simulator profile

Maintain this invariant:

```text
resolved host
== host image built
== GVSoC target launched
== calibration host interpretation
```

1. Add the image/target pair to `pipeline/build_mesh.py::HOSTS`.
2. Add any required host compile flags and runtime compatibility.
3. Provide defensible cost/calibration data.
4. Check catalog discovery and resolved simulator identity.
5. Update calibration and run provenance source selection.
6. Test network build, standalone calibration build, target launch, cache
   separation, and manifest host consistency.

## Adding a metric

Define before exposing it:

```text
name, unit, scope, classification, source/provenance,
safe claim, unsafe claim
```

Use one of `measured`, `derived`, `modeled`, `estimated`, or
`external_reference`. Preserve existing result fields where consumers depend
on them; additive manifest metadata is preferable to renaming a value into a
stronger claim.

Examples: `area_au` remains an estimated structural proxy, and
`cache_dynamic_pj` remains modeled cache dynamic energy rather than total SoC
energy. See [Metric semantics](metrics.md).

## Adding precision support

Precision is currently blocked as a local metadata extension. A real feature
must propagate through:

```text
request -> workload dtype/tensor bytes -> working set -> engine capability
        -> kernel/fallback -> calibration/rate identity -> mapper cost
        -> actual implementation -> correctness/error interpretation
```

Unknown precision must be rejected, never silently treated as FP32. Add the
request field only as part of an end-to-end operational design.

## Adding an execution engine

1. Add target/model and build support.
2. Add runtime mailbox, image, dispatch, and beacon identity.
3. Add Deeploy engine/binding and capability rules.
4. Add kernels and explicit fallbacks.
5. Add safe cost/calibration support without borrowing another engine's data.
6. Extend engine and kernel catalogs after operational support exists.
7. Extend generated and actual evidence.
8. Include selected target/runtime sources in provenance.
9. Add end-to-end correctness/offload validation.

The current runtime has explicit host/Snitch/Spatz handling; a future engine is
not a one-line plugin.

## Adding a GUI-facing capability

1. Implement and validate the Python/backend semantics.
2. Expose authoritative discovery through `experiment-schema`.
3. Accept and validate the field through `ExperimentRequest` and
   `resolve-experiment` where it is part of experiment identity.
4. Add Python contract tests.
5. Update Rust models/controls as a consumer.
6. Preserve existing CLI behavior and display backend validation errors.
7. Add Rust command/parser compatibility tests.

Do not duplicate a new list in Rust when the backend can derive it from the
operational source. See [GUI integration](gui-integration.md).

## Tests and review

For each extension select the smallest tests that prove each layer:

- catalog drift and request validation;
- deterministic resolution and fingerprints;
- operational propagation;
- mapper safety and fallback behavior;
- generated/completed evidence;
- cache invalidation and artifact addressing;
- contradiction/stale-state rejection;
- a real correctness smoke when the environment supports it.

Also update causal calibration/run source sets when the new source can affect
those artifacts. Do not hash the whole repository as a shortcut.

## Patterns to avoid

- catalog entries before operational support;
- copied engine/memory/knob lists in frontends;
- unknown converted to zero/default/supported;
- tuned cost for known Generic fallback;
- partial external calibration merged with stale rows;
- post-run measurements used in a pre-run path key;
- a scientific sweep claim based only on exposed metadata;
- broad plugin frameworks that hide engine-specific behavior.
