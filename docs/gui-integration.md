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
successful response, and `ui::resolve()` / `resolve_all()` write a request
under `work/` and resolve it. The GUI uses `env`, `ops`, `inspect`, `knobs`,
`validate`, `experiment-schema` and `resolve-experiment`.

The Foundation's two commands are query-only:

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

## Current memory, host and placement path

`gui/src/experiment.rs` holds the Rust types for `experiment-schema` and
`resolve-experiment`. The app loads the schema next to `knobs`.

The memory pick lists (Compare, Sweep), the host pick lists and the SoC pin
choices are built from these rows: `main_memories`, `host_profiles` and
`engines` (`ui/common.rs::memory_choices`, `host_choices`, `engine_choices`).
Selections are stored as the pipeline's spelling (`String`), so saved
`SweepSpec` files are unchanged. If the schema cannot be read, for example
from an image older than the Foundation, only `fixed` / `cva6` / `mapped` are
offered and the status line says why.

Compare still sends `--dram` explicitly, including `--dram fixed`. A native
child inherits the environment, and an omitted value would otherwise keep a
pre-existing `HES_DESIGN` memory. Sweep omits a fixed flag; the sweep resolver
canonicalizes it and writes `DRAM_KIND=fixed` into each machine design.

## Resolved preview

`ExperimentRequest::new(...).execution(app, frontend, serial, power)` builds
requests the same way `sweep/run.py::resolve_cell_experiment` does. A
front-end and `serial` are sent only for KWS, and the front-end defaults to
Snitch. As a result, a preview resolves to the fingerprint the driver later
records. The Docker app test checks this against a real manifest.

- **Compare, SoC mode.** *Preview* resolves one request per op and placement.
  *Run* resolves first and queues jobs only if every request resolves. A
  backend error is shown verbatim and nothing runs. The only exception is a
  bridge that predates `resolve-experiment` (argparse "invalid choice"): the
  jobs then run with a warning, as before. Requests and resolved experiments
  are written next to each result (`<tag>.request.json`,
  `<tag>.resolved.json`), and the result card shows the resolved fingerprint.
- **Compare, isolated mode.** `run.py` is not an `ExperimentRequest`, so these
  jobs launch directly.
- **Sweep.** *Resolve baseline* resolves the baseline design for each selected
  model. Per-design identity still comes from `validate`. Launch is not gated
  on it, because the driver resolves every cell and records failures as rows.

## Current result and manifest parsing

`model.rs` parses the Foundation fields additively. Every new field is
optional, so committed and older results still load.

- `mapping.nodes[]`: `index`, `kernel_arguments` and `mapping_explanation`.
  This is the generated layer.
- `result.nodes[]`: `cycles`, which is measured, and `implementation`, which
  is actual.
- Sweep rows: `dram_kind`, `machine_key`, `artifact_key`,
  `run_input_fingerprint`, `run_fingerprint`, `manifest` and
  `calibration_cache`.
- `RunManifest`: the five layers of an `m4ia.run` manifest, plus its
  calibration, artifact and run-source identity.

`ui/evidence.rs` joins generated and completed nodes on the generated `index`,
the same key `annotate_completed_nodes` uses. It shows them in separate
columns: generated engine and selection rule; actual implementation and
fallback; measured cycles. Each node can expand into the mapper's per-engine
explanation. The table never infers an implementation from a placement.

Compare's "node mapping" toggle shows this table for SoC results. Sweep
results have a *details* button per cell, which opens the cell's manifest
layer by layer: requested, resolved (with the machine), generated, actual and
measured, then identity.

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

### Stages 1–3 (done)

- **Memory, host and engine choices.** These come from discovery.
- **Resolved preview.** Shown before launch, and it gates SoC launches in
  Compare.
- **Typed views.** The generated, actual and measured evidence, and sweep
  manifests, each have their own view.

### Stage 4: migrate more controls

Mapping strategies (there is one today) and DRAM overrides
(`supports_overrides` / `override_fields`) have no UI yet. The OFAT grid and
the build-time knob list still come from `knobs`. The Sweep editor shows the
parameter catalog's label, unit and description next to each knob.
Hand-designed widgets remain fine; backend validation and identity stay
authoritative.

## Hard-coded frontend values

| Rust value | Disposition |
|---|---|
| `DramChoice` names and quantitative labels | Migrated to `main_memories` (removed) |
| `cva6` / `ara` host radios | Migrated to `host_profiles` |
| `cva6` / `snitch` / `spatz` placement choices | Migrated to `engines` |
| `mapped` | Harmless UI sentinel for no pin |
| KWS frontend choices | Kept; resolved through the backend in the preview |
| Isolated `ideal`/`real` mode and tuned/autovec choice | Runner-specific UI, not experiment catalog concepts |
| Isolated core list (`cva6`/`snitch`/`spatz`/`ara`) | Runner-specific (`run.py --cores`) |
| Numeric knobs | Grid from `knobs`; labels/units from `parameters` |

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

Tests: `pipeline/tests/test_gui_query.py` and `test_run_manifest.py` pin the
fields the GUI reads. `gui/tests/fixtures/` holds real bridge, result and
manifest output. The ignored Docker app tests mount the checkout's sources,
run Compare and Sweep end to end, and compare the Sweep preview fingerprint
with the driver's manifest.
