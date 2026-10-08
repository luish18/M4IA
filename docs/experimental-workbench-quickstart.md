# Experimental Workbench quickstart

This tutorial follows one workload through discovery, deterministic resolution,
direct execution, a one-cell sweep, and manifest inspection.

It assumes `./setup.sh` has produced the project virtual environment, Deeploy,
toolchain, and GVSoC installation. The lightweight discovery command works with
system Python, but resolving a real ONNX workload needs the project environment.

```text
discover -> request -> resolve -> run -> inspect evidence
```

## Discover current choices

From the repository root:

```bash
.venv/bin/python pipeline/gui_query.py experiment-schema \
  | .venv/bin/python -m json.tool
```

The response includes current design parameters, engines, host profiles,
mapping strategies, kernel implementations, and memory kinds. Useful existing
queries are:

```bash
.venv/bin/python pipeline/gui_query.py ops
.venv/bin/python pipeline/gui_query.py knobs
.venv/bin/python pipeline/gui_query.py inspect ops/mymatmul
```

## Write a minimal request

Create `/tmp/m4ia-request.json`:

```json
{
  "workload": "ops/mymatmul",
  "host": "cva6",
  "dram": "fixed",
  "pin": "spatz"
}
```

The request records intent. It does not repeat numeric defaults, derived
resources, operational preset parameters, or fingerprints.

## Resolve without executing

```bash
.venv/bin/python pipeline/gui_query.py resolve-experiment \
  /tmp/m4ia-request.json \
  | .venv/bin/python -m json.tool
```

Inspect:

- `platform` and `resolved_fingerprint`;
- `hardware.design`, `design_slug`, `build_key`, and hardware fingerprint;
- `host_profile` and simulator target;
- `memory.kind`, overrides, and memory fingerprint;
- workload fingerprint;
- `resources`;
- mapping strategy, pin semantics, and execution controls.

Resolution is query-only: it does not build, calibrate, simulate, or create a
manifest. Invalid JSON or request semantics return a JSON `error` object and a
nonzero status.

Try removing `dram` and resolving again. Omitted memory and explicit `fixed`
must produce the same resolved fingerprint. Changing it to `lpddr5` must change
the resolved memory and experiment fingerprints.

## Run directly

For a fast controlled run using the same public choices:

```bash
.venv/bin/python pipeline/run_hetero.py ops/mymatmul \
  --host cva6 \
  --dram fixed \
  --pin spatz \
  --out /tmp/m4ia-spatz-result.json
```

`--pin spatz` applies to compatible nodes; it does not promise whole-graph
placement. Inspect the generated mapping and completed node evidence rather
than inferring execution from the requested pin.

The direct runner preserves the existing outer result contract:

```json
{
  "op": "...",
  "mapping": {},
  "result": {}
}
```

`mapping.nodes` contains generated placement/explanation/arguments where
available. `result.nodes` contains completed runtime nodes and implementation
evidence where it can be established. A generated node without a completed
beacon is not added to the result.

Direct runs produce `result.json`; the sweep path below adds machine,
calibration, pre-run identity, provenance, and a first-class run manifest.

## Run one sweep cell

Choose a new output directory rather than reusing a historical result:

```bash
.venv/bin/python pipeline/sweep/run.py \
  --models ops/mymatmul \
  --dram fixed \
  --images 1 \
  --limit 1 \
  --progress lines \
  -o work/experimental-smoke-cp8
```

The sweep CLI does not currently expose a sweep-level pin. Use the direct
runner for controlled pin experiments rather than inventing a flag.

The exact directory names contain deterministic keys:

```text
work/experimental-smoke-cp8/
├── sweep.jsonl
├── designs/
│   └── <machine_key>/
│       ├── design.json
│       ├── mesh/
│       ├── rates.json
│       ├── calibration.json
│       └── calib/
└── cells/
    └── <artifact_key>/
        ├── result.json
        └── manifest.json
```

The machine key includes numeric design, host, and resolved memory, but not the
workload. The cell key uses the pre-run identity and therefore also separates
workload and execution controls such as sample count.

## Inspect sweep output

Each `sweep.jsonl` row preserves legacy performance fields and adds identity
links such as:

```text
dram_kind
machine_key
artifact_key
run_input_fingerprint
run_fingerprint
manifest
```

Use the row's `manifest` field to find the corresponding manifest. It has:

```text
requested
resolved
generated
actual
measured
```

plus calibration linkage, run provenance, artifact paths/digests, and the two
run fingerprints.

Check that:

- requested memory preserves the request spelling;
- resolved memory is explicit even for fixed;
- generated nodes describe code-generation evidence;
- actual nodes contain completed-only execution/implementation evidence;
- measured status/cycles/counters remain observations;
- calibration metadata and result digests match the linked files.

## Verify calibration reuse

Run the identical sweep into the same output directory a second time. A cache
hit is valid only if `rates.json` and `calibration.json` pass schema, kind,
protocol, input-identity, expected-fingerprint, and rates-digest checks.

A legacy `rates.json` without its sidecar is intentionally a miss. Different
host or memory configurations use different machine/calibration paths. Workload
changes alone do not invalidate calibration for the same machine.

## Compare responsibly

Before treating two runs as controlled:

1. compare requested differences;
2. compare resolved machine/memory/workload identity;
3. confirm generated placement;
4. confirm completed implementation evidence;
5. confirm calibration identity and reuse status;
6. confirm run-source provenance;
7. reject incorrect/infeasible results;
8. compare values only within their documented metric scope.

Do not call `area_au` physical area or `cache_dynamic_pj` total energy. See
[Metric semantics](metrics.md).

## Validation boundary

The Python contract and control flow are covered by the repository unit suite.
A publication or release checklist should additionally replay a direct run,
calibration miss/hit, fixed-memory cell, and one modeled-device cell in a complete
Deeploy/GVSoC/toolchain environment.
