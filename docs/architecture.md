# M4IA architecture

M4IA is a heterogeneous RISC-V experimentation platform for ONNX workloads.
It combines Deeploy code generation, a cost-based heterogeneous mapper, a CVA6
host, Snitch and Spatz clusters, a host-driven offload runtime, configurable
memory, and GVSoC execution.

The Experimental Foundation wraps that operational stack with discovery,
deterministic resolution, evidence, identity, and provenance. It does not
replace or reimplement the stack.

## Operational flow

```text
GUI / CLI / sweep
    -> workload discovery and ExperimentRequest
    -> deterministic experiment resolution
    -> numeric design + explicit main-memory selection
    -> Deeploy graph parsing and lowering
    -> compatibility and cost mapper / node colouring
    -> Deeploy code generation
    -> generated Network.c + mapping.json
    -> host/cluster ELF construction
    -> CVA6 host dispatch
    -> Snitch / Spatz cluster execution
    -> GVSoC heterogeneous target and memory hierarchy
    -> runtime beacons, counters, correctness and result.json
    -> sweep row and m4ia.run manifest
```

| Stage | Operational owner | Responsibility |
|---|---|---|
| Frontend queries | `pipeline/gui_query.py` | Workload/knob discovery, experiment schema, query-only resolution |
| Direct execution | `pipeline/run.py`, `pipeline/run_hetero.py` | Build, simulate, parse and report one workload |
| Sweep orchestration | `pipeline/sweep/run.py` | Resolve cells, prepare machine artifacts, calibrate, launch and manifest |
| Workload locating and identity | `pipeline/run.py::resolve_test_dir()`, application handling in `pipeline/run_hetero.py`, `pipeline/experiment/workload.py` | Locate direct/application inputs and describe the consumed package for resolution/sweeps |
| Experiment resolution | `pipeline/experiment/resolve.py` | Validate and canonicalize researcher intent |
| Numeric design | `pipeline/sweep/design.py` | Numeric defaults, validation, slug and build key |
| Main memory | `pipeline/common.py`, `targets/hetero/memsys.py`, `targets/hetero/dram_presets.py` | Select and instantiate fixed or device memory |
| Deeploy integration | `pipeline/hetero_platform/` | Engine capabilities, mapping, code generation and progress instrumentation |
| Image construction | `pipeline/build_mesh.py` | Build the selected host image and both cluster images |
| Runtime dispatch | `runtime/mesh/` | Mailboxes, staging, synchronization, progress and application control |
| Specialized kernels | `runtime/snitch/kernels/`, `runtime/spatz/kernels/` | Existing SSR/FREP and RVV implementations plus Generic fallback paths |
| Target construction | `targets/hetero/soc.py`, `targets/hetero_soc.py`, `targets/hetero_ara.py` | Compose host, clusters, interconnect and memory into GVSoC targets |
| Results/evidence | `pipeline/run_hetero.py`, `pipeline/experiment/run_manifest.py` | Preserve generated, completed, implementation and measured evidence |

## Host and engines

### CVA6 host

CVA6 owns orchestration and host-side execution. `RunNetwork()` runs on the
selected scalar-CVA6 or CVA6+Ara host image. Nodes unsupported by a cluster
remain on the logical `cva6` engine and use the host/Generic path.

### Snitch cluster

The baseline resolved design gives the Snitch cluster nine modeled cores:
eight compute cores and one control/DMA core. `SNITCH_NB_CORE` is a supported
numeric design dimension, so this is a default rather than an architecture
invariant. Its tuned FP32 MatMul/Gemm path uses the existing SSR/FREP
implementation when the compiled dispatch conditions hold. Other shapes can
take the renamed Deeploy Generic fallback.

### Spatz cluster

The baseline resolved design also gives Spatz nine modeled cores, eight useful
for compute. `SPATZ_NB_CORE` is independently sweepable. `ResourceSummary`
derives modeled, compute, and control/DMA counts from the resolved design rather
than freezing the baseline values. The manual FP32 MatMul/Gemm path uses RVV
when supported and retains the Deeploy Generic fallback. Modeled vector lanes
and useful compute lanes are different resource facts; neither is a physical
area or power measurement.

## Mapping and generation

`pipeline/hetero_platform/generate.py` constructs the current deployment
platform and mapper. The mapper first checks engine compatibility, then uses
the current measured-rate plus offload-cost model where a defensible numeric
cost exists. Unknown working sets, MAC counts, rates, or overheads stay
unavailable rather than being fabricated.

`--pin` means compatible-node preference: a compatible pinned engine is
selected even when automatic pricing is unavailable, but unsupported nodes may
remain elsewhere. It is not whole-graph isolation.

Generation writes `mapping.json`. Its node records can contain placement,
mapping explanation, and generated MatMul/Gemm arguments. These are generated
facts, not proof of runtime completion.

## Build and runtime

`pipeline/build_mesh.py::HOSTS` binds each host profile to both its image and
GVSoC target:

```text
cva6 -> HOST image     -> hetero_soc
ara  -> HOST_ARA image -> hetero_ara
```

The same binding is used for network construction and calibration. Each
cluster has its own image, mailbox and completion state. The host stages
arguments, posts jobs, waits for completion, and records progress beacons.
Applications such as KWS may use the two clusters concurrently, but each
cluster mailbox admits one job at a time.

## Memory architecture

Current public main-memory kinds are derived from
`targets/hetero/dram_presets.py::KINDS`:

```text
fixed  lpddr4  lpddr4x  lpddr5  hyperram
```

`pipeline/common.py::use_dram()` validates the requested kind and merges
`DRAM_KIND` into the operational `HES_DESIGN` document when an override is
needed. The sweep writes the resolved kind explicitly into each machine
design; a direct fixed run may use the operational fixed default.
`targets/hetero/memsys.py` consumes that value, constructs fixed or device
memory, and selects the associated L2 policy. The fixed path keeps the
historical fixed-latency/write-through behavior; modeled device memory uses the
C++ timing model and a write-back L2.

Memory is not an integer design override:

```text
design.slug = numeric chip-design identity

machine identity = numeric design + host profile + resolved memory
```

Omitted memory and explicit `fixed` canonicalize to the same resolved memory.
Modeled-device overrides are expanded through the operational preset code and
participate in identity.

## Evidence flow

M4IA deliberately separates intention, generation, completion, implementation
classification, and observations:

```text
mapper decision
    != generated placement/arguments
    != completed runtime beacon
    != concrete implementation evidence
    != measured cycles/counters
```

After simulation, `annotate_completed_nodes()` joins only completed runtime
nodes to matching generated metadata. For audited MatMul/Gemm paths it may
classify a tuned or Generic implementation. Missing, duplicate or contradictory
evidence stays unknown. Planned nodes without a completion beacon never become
actual execution evidence.

## Artifacts and provenance

The sweep preserves five identity scopes:

- **numeric design identity** — `design.slug`, containing only the resolved
  numeric chip-design values;
- **machine identity** — numeric design plus host profile and resolved memory;
- **calibration identity** — workload-independent machine/protocol identity
  plus causal sources, dependencies, patches, toolchain, and simulator;
- **run-input identity** — resolved experiment, execution controls,
  calibration fingerprint, and causal run-source digest; it addresses the cell;
- **completed-run identity** — run-input identity plus generated, actual,
  measured, and linked-artifact evidence.

Machine directories hold the effective design, refreshed mesh, rates,
calibration metadata, and calibration logs. Cell directories are addressed by
pre-run input identity and hold the authoritative `result.json` and
`manifest.json`. Stale result/manifest files are removed before a new launch.

The `m4ia.run` manifest separates:

```text
requested  resolved  generated  actual  measured
```

See [Experimental Workbench](experimental-workbench.md) for the contract and
[Metric semantics](metrics.md) for what the observations do and do not mean.

## Architectural non-claims

- Catalog metadata is not operational support.
- A configurable parameter is not automatically causally validated.
- Mapper placement is not concrete implementation evidence.
- Host-observed node cycles are not automatically kernel-body cycles.
- Equal core or lane counts do not imply equal area or power.
- `area_au` is not physical silicon area.
- `cache_dynamic_pj` is not total SoC energy.
- GVSoC execution is not a claim of RTL or silicon equivalence.

For implementation history, see
[the development report](relatorio-desenvolvimento.md), the historical
[mesh proposal](hetero-mesh-plan.md), and the maintainer-facing
[integration notes](dev/integration-notes.md).
