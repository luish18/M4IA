# GUI integration and continuation guide

The Rust GUI is a thin client over the Python pipeline. It launches the same
runner/sweep commands used at the CLI and asks `pipeline/gui_query.py` for
machine-readable discovery and validation. It should remain a consumer of
backend semantics rather than a second experiment resolver.

## Current boundary

```text
Rust GUI
    -> gui/src/backend.rs::Invocation
    -> pipeline/gui_query.py or runner CLI
    -> Python operational stack
    -> result.json / sweep.jsonl
```

`backend::capture()` runs quick JSON queries and extracts the bridge's JSON
`error` value when a command fails. `gui/src/ui/mod.rs::query()` deserializes a
successful response. The GUI currently uses `env`, `ops`, `inspect`, `knobs`,
and `validate`.

The accepted Foundation added two query-only commands:

```text
experiment-schema
resolve-experiment REQUEST.json
```

No build, calibration, simulation, runner, or manifest side effect is involved
in either query.

## Backend contract

### Experiment schema

`experiment-schema` exposes current:

- platform preset IDs through `presets`;
- numeric design parameters;
- engines;
- host profiles;
- mapping strategies;
- kernel implementation catalog;
- main-memory kinds and accepted override fields.

Memory rows flow through the experiment discovery layer to the operational
DRAM preset definitions. `gui_query.py` does not maintain another DRAM list.

The current catalog document is additive JSON, not a complete JSON Schema for
every request field. A future versioned/multi-platform contract may benefit
from explicit schema/version and platform structures, but the current surface
is sufficient for incremental GUI consumption.

### Experiment resolution

`resolve-experiment` reads a request file, validates it through
`ExperimentRequest.from_dict()`, and returns the deterministic resolved
experiment. It exposes explicit host, mapping, execution, memory, resource,
workload and fingerprint facts.

Omitted and explicit fixed memory resolve identically. Real memories and
operational overrides change resolved identity. Invalid requests return a
machine-readable JSON error; the GUI should display that error rather than
silently reinterpret the request.

## Current memory path

The current Rust UI still has a `DramChoice` enum for:

```text
fixed  lpddr4  lpddr4x  lpddr5  hyperram
```

Compare sends the selected value directly to `run.py` or `run_hetero.py` as
`--dram`, including explicit `--dram fixed`. Making fixed explicit is
important for native GUI jobs: child processes inherit the environment, and an
omitted runner value otherwise preserves a pre-existing `HES_DESIGN` memory.

Sweep stores `SweepSpec.dram`. Existing saved specs without the field load as
fixed through Serde defaults. For compatibility, the current sweep command may
omit a fixed flag; the accepted sweep resolver canonicalizes it to fixed and
writes explicit `DRAM_KIND=fixed` into each machine design. Non-fixed values
are passed through `--dram`.

This path is behaviorally aligned today, but `DramChoice` remains duplicated
presentation state and should be migrated to discovery later.

## Current result parsing

The Rust models parse the established `result.json` and sweep row fields.
Serde ignores additive Foundation fields, so current displays remain backward
compatible.

The GUI does not yet have typed models/views for:

- mapping explanations and generated arguments;
- explicit mapping memory;
- sweep artifact/machine/run fingerprints;
- `m4ia.run` manifests;
- requested/resolved/generated/actual/measured views.

These are frontend implementation tasks, not missing backend contracts.

## Target continuation path

Migration should be incremental:

```text
experiment-schema
    -> controls
    -> ExperimentRequest
    -> resolve-experiment
    -> resolved preview
    -> existing run/sweep launch
    -> result + m4ia.run manifest views
```

### Stage 1: discover memory

Replace the hand-maintained memory choice list and quantitative display labels
with `main_memories` rows from `experiment-schema`. Preserve runner command
compatibility and explicit fixed selection.

### Stage 2: resolved preview

Serialize current user controls into an `ExperimentRequest`, resolve it, and
show workload, host, numeric design, memory, resources, mapping strategy, and
resolved fingerprint before launch.

### Stage 3: consume manifests

Add typed parsing for:

```text
requested -> user intent
resolved  -> canonical configuration/machine
generated -> placement/explanation/dispatch arguments
actual    -> completed execution/implementation evidence
measured  -> status/cycles/counters/correctness
```

Also expose calibration and artifact digests/fingerprints without recomputing
their semantics in Rust.

### Stage 4: migrate more controls

Host profiles, engine/pin choices, mapping strategies, and design-parameter
metadata can move to discovery as useful. Hand-designed widgets remain fine;
backend validation and identity stay authoritative.

## Hard-coded frontend values

| Current Rust value | Disposition |
|---|---|
| `DramChoice` names | Migrate to memory discovery |
| Quantitative memory labels | Migrate; they can drift from operational presets |
| `cva6` / `ara` host radios | Migrate to host-profile discovery |
| `cva6` / `snitch` / `spatz` placement choices | Migrate to engine and mapping catalogs |
| `mapped` | Harmless UI sentinel for no pin |
| KWS frontend choices | Keep temporarily; resolve through backend when preview is added |
| Isolated `ideal`/`real` mode and tuned/autovec choice | Runner-specific UI, not necessarily experiment catalog concepts |
| Numeric knobs | Already queried; migrate incrementally to richer parameter rows |

## Compatibility rules

- Evolve JSON additively; do not repurpose established fields.
- Keep success stdout as one JSON document.
- Keep ordinary validation failures machine-readable and free of tracebacks.
- Preserve old result/sweep parsing while adding typed Foundation fields.
- Do not infer actual implementation from placement.
- Do not merge measured fields into actual implementation identity.
- Keep requested memory spelling separate from canonical resolved memory.
- Add Python contract tests before depending on a field in Rust.

## Maintainer checklist

Before a GUI migration:

1. confirm the backend value is derived from an operational owner;
2. confirm request validation and canonical resolution;
3. add/adjust Python JSON-contract tests;
4. update the Rust type/control;
5. preserve CLI fallback compatibility;
6. test fixed and non-fixed memory explicitly;
7. test old saved `SweepSpec` and result fixtures;
8. keep manifest evidence layers distinct;
9. run `cargo test` in an existing Rust-capable environment.

The Foundation PR deliberately does not require a broad Rust GUI refactor.
