# Chip design optimization parameters

> **Scope and causal-validation notice**
>
> This is the maintained inventory of the current modeled design space. It is
> not proof that every listed parameter is a validated scientific sweep axis.
> Before drawing a causal claim, trace the value from request/override through
> resolved design, generated configuration, GVSoC construction, runtime-visible
> state, and measurement. See
> [Experimental Workbench](experimental-workbench.md#scientific-use-guardrail)
> and [Extending M4IA](extending-m4ia.md#adding-a-hardware-design-knob).
>
> Main-memory selection (`fixed`, `lpddr4`, `lpddr4x`, `lpddr5`, or
> `hyperram`) is explicit experiment/machine identity, not an integer entry in
> `pipeline/sweep/design.py::DEFAULTS`. Consequently it does not enter the
> numeric-only `design.slug`; it is resolved and fingerprinted separately.

This inventory covers the simulated `hetero_soc` chip (CVA6 host + Snitch and
Spatz clusters, described historically in
[relatorio-desenvolvimento.md](relatorio-desenvolvimento.md)). It includes
several kinds of value that must not be confused:

- **supported numeric design knobs** are exactly the integer keys in
  `pipeline/sweep/design.py::DEFAULTS`; `parameter_catalog.py` provides
  descriptive metadata and rejects drift from that set;
- **operational/model facts** are current constants or properties consumed by
  the target/runtime but are not standard sweep keys;
- **derived values** are computed from the resolved design and must not be
  overridden as independent constants;
- **main memory** is separate experiment/machine identity, not a numeric design
  key;
- **evaluation controls**, such as standalone `--memory`, select a comparison
  mode rather than changing `design.slug`;
- **future axes** require new modeling or runtime work before they are supported.

The SoC is a GVSoC model rather than synthesized RTL. Exposure in Python or a
catalog is still not proof that a knob's causal path has been scientifically
validated.

There is no vendored RTL in this repo (`deps/` holds only three GVSoC patches,
no submodules, no SystemVerilog). GVSoC itself — the simulator that supplies
the CVA6/Ara/Snitch/Spatz component models — is fetched by `setup.sh` at a
pinned commit, not checked into this worktree, so any microarchitectural
detail *inside* those upstream models that is not exposed by the supported
design keys below (for example CVA6 pipeline depth or branch predictor) is
fixed by that pinned version and out of scope for the standard sweep API.

## Supported numeric design keys

These are the complete keys accepted by `ExperimentRequest.design_overrides`
and the current sweep design resolver. Values shown are baseline defaults, not
claims that every axis has completed causal validation.

| Key | Baseline | Scope |
|---|---:|---|
| `HOST_VLEN` | 4096 bits | Ara vector-register length; build-time key |
| `HOST_NB_LANES` | 4 | Ara vector lanes |
| `HOST_LANE_WIDTH` | 8 bytes | Ara lane width |
| `SPATZ_VLEN` | 512 bits | Per-core Spatz vector-register length; build-time key |
| `SPATZ_NB_LANES` | 4 | Lanes per modeled Spatz core |
| `SPATZ_LANE_WIDTH` | 8 bytes | Spatz lane width |
| `SNITCH_NB_CORE` | 9 | Total modeled Snitch-cluster cores, including one control/DMA core |
| `SPATZ_NB_CORE` | 9 | Total modeled Spatz-cluster cores, including one control/DMA core |
| `TCDM_SIZE` | 128 KiB | Capacity assigned to each cluster |
| `ICACHE_SIZE` / `ICACHE_WAYS` | 16 KiB / 4 | Host L1 instruction-cache geometry |
| `DCACHE_SIZE` / `DCACHE_WAYS` | 32 KiB / 8 | Host L1 data-cache geometry |
| `L2_SIZE` / `L2_WAYS` | 512 KiB / 8 | Shared host-side L2 geometry |
| `LINE_SIZE` | 64 bytes | Line size shared by the modeled host caches |
| `NARROW_AXI_WIDTH` | 8 bytes/cycle | Host-to-cluster/narrow AXI width |
| `WIDE_AXI_WIDTH` | 64 bytes/cycle | Main-memory/DMA wide AXI width |

Authoritative ownership is
[`pipeline/sweep/design.py`](../pipeline/sweep/design.py); descriptive names
and units are in
[`pipeline/experiment/parameter_catalog.py`](../pipeline/experiment/parameter_catalog.py).
Operational consumers are identified in the sections below.

## 1. CVA6 host (orchestrator)

The vector and cache-geometry rows named in `DEFAULTS` are supported numeric
design knobs. ISA/profile choice, hart assignment, timing assumptions, and
address-space sizes are operational or derived facts.

| Parameter | Baseline/current value | Class and owner | What it controls |
|---|---|---|---|
| host ISA | `rv64imafdc` (`cva6`) / `rv64imafdcv` (`ara`) | Host-profile operational fact; [targets/hetero/soc.py](../targets/hetero/soc.py) | ISA decoded by the selected host model |
| `HOST_VLEN` | 4096 bits | Supported numeric knob; [targets/hetero/system.py](../targets/hetero/system.py) | Ara vector-register length; eight times baseline Spatz VLEN |
| `HOST_NB_LANES` | 4 | Supported numeric knob; [targets/hetero/system.py](../targets/hetero/system.py) | Parallel Ara vector lanes |
| `HOST_LANE_WIDTH` | 8 bytes | Supported numeric knob; [targets/hetero/system.py](../targets/hetero/system.py) | Bytes processed per Ara lane per cycle |
| `HOST_HARTID` | derived; 18 at baseline | Derived as `max(first_hartid + nb_core)` across clusters in [targets/hetero/system.py](../targets/hetero/system.py) | Keeps the host above every resolved cluster hart ID when core counts change |
| `ICACHE_SIZE` / `ICACHE_WAYS` | 16 KiB / 4 | Supported numeric knobs; [targets/hetero/memsys.py](../targets/hetero/memsys.py) | L1 instruction-cache geometry |
| `ICACHE_HIT_LATENCY` | 0 cycles | Operational model fact, not a standard sweep key; [targets/hetero/memsys.py](../targets/hetero/memsys.py) | Front-end stall charged on an I-cache hit |
| `DCACHE_SIZE` / `DCACHE_WAYS` | 32 KiB / 8 | Supported numeric knobs; [targets/hetero/memsys.py](../targets/hetero/memsys.py) | L1 data-cache geometry |
| `DCACHE_HIT_LATENCY` | 1 cycle | Operational model fact, not a standard sweep key; [targets/hetero/memsys.py](../targets/hetero/memsys.py) | Extra modeled D-cache hit latency on top of ISS behavior |
| D-cache write policy | no write allocate; write-through; 8-entry buffer | Operational model facts; [targets/hetero/memsys.py](../targets/hetero/memsys.py) | Store-miss allocation and draining to L2 |
| `LINE_SIZE` | 64 bytes | Supported numeric knob; [targets/hetero/memsys.py](../targets/hetero/memsys.py) | Line size shared by I-cache, D-cache, and L2 |
| `L2_SIZE` / `L2_WAYS` | 512 KiB / 8 | Supported numeric knobs; [targets/hetero/memsys.py](../targets/hetero/memsys.py) | Shared host-side L2 geometry |
| `L2_LATENCY` | 10 cycles | Operational model fact, not a standard sweep key; [targets/hetero/memsys.py](../targets/hetero/memsys.py) | L2 hit latency |
| `L2_WIDTH` | 16 bytes/cycle | Operational model fact, not a standard sweep key; [targets/hetero/memsys.py](../targets/hetero/memsys.py) | L1-to-L2 refill width |
| `HOST_HEAP_SIZE` | 8 MiB | Fixed operational fact; [targets/hetero/system.py](../targets/hetero/system.py) | Heap used by generated `InitNetwork` allocations |
| `HOST_STACK_SIZE` | 256 KiB | Fixed operational fact; [targets/hetero/system.py](../targets/hetero/system.py) | Host stack size |
| `HOST_LOAD_SIZE` | 64 MiB | Fixed operational fact; [targets/hetero/system.py](../targets/hetero/system.py) | Host ELF image window |

Every cache above (I$, D$, L2) is one instance of the same generic
`TimingCache` model, whose full property surface is
`size`, `line_size`, `ways`, `hit_latency`, `miss_latency`, `refill_cycles`,
`write_cycles`, `write_allocate`, `store_buffer_size` — see
[`targets/hetero/timing_cache.py`](../targets/hetero/timing_cache.py). Only the ones
each level actually sets non-default are listed in the table; e.g.
`miss_latency` is left at 0 everywhere today (the "next level" latency is
charged by that level's own mapping instead). A model property outside
`design.DEFAULTS` is not a supported standard sweep key; adding one requires
the operational, identity, provenance, and validation work in
[Extending M4IA](extending-m4ia.md#adding-a-hardware-design-knob).

## 2. Snitch cluster (integer core + Xssr/Xfrep FP subsystem)

The baseline count is not fixed: `SNITCH_NB_CORE` is a supported numeric key.
The runtime convention reserves one resolved core for control/DMA and treats
the remainder as compute cores.

| Parameter | Baseline/current value | Class and owner | What it controls |
|---|---|---|---|
| `SNITCH_NB_CORE` | 9 = 8 compute + 1 control/DMA | Supported numeric knob; [targets/hetero/system.py](../targets/hetero/system.py) | Total modeled cores; `ResourceSummary` derives the resolved split |
| `TCDM_SIZE` | 128 KiB | Supported numeric knob shared by both clusters; [targets/hetero/system.py](../targets/hetero/system.py) | Banked scratchpad capacity |
| ISA | `rv32imfdca` | Fixed operational fact; [targets/hetero/system.py](../targets/hetero/system.py) | Scalar ISA with the Snitch FP subsystem and no RVV |
| core model | `accurate` | Fixed operational fact; [targets/hetero/system.py](../targets/hetero/system.py) | Retains the decoupled FP subsystem needed by Xssr/Xfrep kernels |
| performance counters | 16 | Fixed operational fact; [targets/hetero/system.py](../targets/hetero/system.py) | Cluster register-file counter count |
| `CLUSTER_STACK_SIZE` | 4 KiB per resolved core | Fixed operational fact; [targets/hetero/system.py](../targets/hetero/system.py) | Per-core stack carved from TCDM |
| peripheral window | 64 KiB | Fixed operational fact; [targets/hetero/system.py](../targets/hetero/system.py) | Cluster peripheral register window |
| cluster base | `0x1000_0000` | Fixed operational fact; [targets/hetero/system.py](../targets/hetero/system.py) | Snitch cluster address-space base |
| `CLUSTER_IRQ` | 19 | Fixed operational fact; [targets/hetero/system.py](../targets/hetero/system.py) | Cluster-local interrupt used by the mailbox doorbell |
| TCDM banking (32 banks × 8 B, 256 B interleave) | fixed | GVSoC's `ClusterArch`/`SnitchCluster` model (not a constant in this repo — see [relatorio-desenvolvimento.md §6.4](relatorio-desenvolvimento.md)) | Bank-conflict behavior of strided TCDM accesses; changing it means forking the upstream GVSoC class, not editing a constant here |
| Xssr / Xfrep | fixed feature, not sized | — | Stream Semantic Registers / FP-repeat sequencer that removes load/address-generation and loop overhead — the reason a scalar Snitch core beats naive RVV in dense FMA loops |

## 3. Spatz cluster (Snitch core + RVV vector unit)

`SPATZ_NB_CORE`, vector geometry, and shared TCDM size are supported design
keys. The baseline again uses one control/DMA core and eight compute cores.

| Parameter | Baseline/current value | Class and owner | What it controls |
|---|---|---|---|
| `SPATZ_NB_CORE` | 9 = 8 compute + 1 control/DMA | Supported numeric knob; [targets/hetero/system.py](../targets/hetero/system.py) | Total modeled cores; independent of the Snitch count |
| `SPATZ_VLEN` | 512 bits | Supported numeric/build-time knob; [targets/hetero/system.py](../targets/hetero/system.py) | Vector-register length per modeled Spatz core |
| `SPATZ_NB_LANES` | 4 | Supported numeric knob; [targets/hetero/system.py](../targets/hetero/system.py) | Vector lanes per modeled Spatz core |
| `SPATZ_LANE_WIDTH` | 8 bytes | Supported numeric knob; [targets/hetero/system.py](../targets/hetero/system.py) | Datapath width of each lane |
| ISA | `rv32imfdcav` | Fixed operational fact; [targets/hetero/system.py](../targets/hetero/system.py) | Scalar ISA plus RVV |
| performance counters | 2 | Fixed operational fact; [targets/hetero/system.py](../targets/hetero/system.py) | Cluster register-file counter count |
| core model | `SnitchFast` operationally | Fixed operational fact; [targets/hetero/system.py](../targets/hetero/system.py) and [targets/hetero/soc.py](../targets/hetero/soc.py) | Spatz construction ignores the nominal `accurate` property for its core type |
| cluster base | `0x0010_0000` | Fixed operational fact; [targets/hetero/system.py](../targets/hetero/system.py) | Spatz cluster address-space base |
| TCDM / stack / peripheral sizing | shared rules, resolved per cluster | Supported TCDM key plus operational facts; [targets/hetero/system.py](../targets/hetero/system.py) | VLSU operands must fit TCDM; compatibility uses the cluster-specific resolved budget in [pipeline/hetero_platform/engines.py](../pipeline/hetero_platform/engines.py) |

At the baseline, Spatz has 36 modeled lanes and 32 useful compute lanes. Those
counts change with resolved core/lane knobs; `ResourceSummary` computes them and
does not treat the baseline as immutable.

## 4. Interconnect and shared memory system

AXI widths are supported numeric design knobs. Clock, addressing, mailbox, and
fixed-path timing are current model facts. Main-memory selection is resolved
separately from `design.DEFAULTS`.

| Parameter | Baseline/current value | Class and owner | What it controls |
|---|---|---|---|
| `NARROW_AXI_WIDTH` | 8 bytes/cycle | Supported numeric knob; [targets/hetero/system.py](../targets/hetero/system.py) | Host accesses to cluster address space |
| `WIDE_AXI_WIDTH` | 64 bytes/cycle | Supported numeric knob; [targets/hetero/system.py](../targets/hetero/system.py) | Cluster DMA, instruction refills, and shared main-memory traffic |
| `FREQUENCY` | 1 GHz | Operational model fact, not a standard sweep key; [targets/hetero/memsys.py](../targets/hetero/memsys.py) | Single clock domain; converts modeled device timings to cycles |
| `DRAM_KIND` | `fixed` by default | Separate resolved memory identity; [targets/hetero/memsys.py](../targets/hetero/memsys.py) and [targets/hetero/dram_presets.py](../targets/hetero/dram_presets.py) | Selects `fixed`, `lpddr4`, `lpddr4x`, `lpddr5`, or `hyperram` |
| `DRAM_OVERRIDES` | `{}` | Separate effective memory identity; [targets/hetero/dram_presets.py](../targets/hetero/dram_presets.py) | Valid preset-field overrides; not an integer design knob |
| L2 write policy | write-through for `fixed`; write-back for modeled devices | Derived default in [targets/hetero/memsys.py](../targets/hetero/memsys.py) | Prevents every store from becoming a modeled device write burst |
| `DRAM_LATENCY` | 100 cycles for `fixed`; derived nominal value for devices | Fixed-path model fact / device consequence; [targets/hetero/memsys.py](../targets/hetero/memsys.py) | Mapping latency for fixed memory and nominal header/peripheral value for a modeled device |
| `DRAM_WIDTH` | 8 bytes/cycle for `fixed`; device-owned otherwise | Fixed-path model fact; [targets/hetero/memsys.py](../targets/hetero/memsys.py) | Serializes fixed-path line refills; device model owns its own bandwidth |
| `HBM_SIZE` | 2 GiB | Fixed operational fact; [targets/hetero/system.py](../targets/hetero/system.py) | Shared main-memory capacity |
| `MAILBOX_SIZE` / `MAILBOX_MAX_ARGS` | 512 B / 16 | Fixed operational facts; [targets/hetero/system.py](../targets/hetero/system.py) | Per-cluster job descriptor and argument capacity |
| host/Snitch/Spatz ELF windows | 64 MiB each | Fixed operational facts; [targets/hetero/system.py](../targets/hetero/system.py) | Non-overlapping load-image windows in shared memory |
| DRAM/stdout/control peripheral windows | 256 MiB each | Fixed operational facts; [targets/hetero/system.py](../targets/hetero/system.py) | Host peripheral address-space regions; these are not the 64 MiB ELF windows |

## 5. Memory-model switch (evaluation knob, not a chip parameter)

| Parameter | Values | Where | What it controls |
|---|---|---|---|
| `--memory` | `real` (default) / `ideal` | Standalone [pipeline/run.py](../pipeline/run.py) control | Selects modeled-memory targets or the available idealized/zero-latency-memory targets. It supports memory-sensitivity comparison but does not isolate computation alone: instruction execution, runtime, instrumentation, synchronization, and other simulator effects remain. Ara has no separate ideal target and continues to use `ara_host`. |

`--dram` applies under the modeled (`real`) standalone path and on the
heterogeneous runner/sweep. It selects resolved main-memory identity; it is not
an alternative spelling for `--memory`.

## 6. Cost model (software layer that evaluates the hardware design)

These don't change the simulated chip, but they change how a design's numbers
get turned into a mapping decision — worth listing because tuning them changes
which core an operator lands on, i.e. they are part of the same optimization
loop.

| Parameter | Current value | Where | What it controls |
|---|---|---|---|
| `RATES` | per-host/engine, per-op MACs/cycle table | [pipeline/hetero_platform/mapper.py](../pipeline/hetero_platform/mapper.py) | Committed estimate table, replaced as a whole by a valid external `HES_RATES` table |
| `OFFLOAD_FIXED` | cva6: 0, snitch/spatz: 1200 cycles | [pipeline/hetero_platform/mapper.py](../pipeline/hetero_platform/mapper.py) | Fixed mailbox/offload cost charged per cluster node |
| `OFFLOAD_PER_BYTE` | cva6: 0.0, snitch/spatz: 0.10 cycles/byte | [pipeline/hetero_platform/mapper.py](../pipeline/hetero_platform/mapper.py) | Cost of staging operands into and out of TCDM |
| cluster TCDM budget | resolved `TCDM_SIZE` minus mailbox, resolved per-core stacks, and 8 KiB reserve | [pipeline/hetero_platform/engines.py](../pipeline/hetero_platform/engines.py) | Per-cluster operand capacity used by compatibility checks |

## 7. Not modelled yet (open design space, from `hetero-mesh-plan.md`)

The project's own roadmap ([hetero-mesh-plan.md](hetero-mesh-plan.md)) lists
further axes that would extend the design space but need new modeling work
before they become supported keys like the ones above. Core counts are omitted
from this table because `SNITCH_NB_CORE` and `SPATZ_NB_CORE` are already
supported numeric dimensions.

| Parameter | Planned value / range | Why it's not a constant yet |
|---|---|---|
| Number of clusters | two current named clusters; more proposed historically | Requires target address map, images, runtime/mailboxes, engine identity, mapping, provenance, and simulation-throughput work ([hetero-mesh-plan.md](hetero-mesh-plan.md)) |
| Mixed-core clusters | homogeneous only | Would need a new cluster class — GVSoC currently builds every core in a cluster from one class |
| D2D link `link_latency_ns` / `link_bandwidth_GBps` | unset | `D2DLink` exists in GVSoC (`pulp/chips/soft_hier_old/c2c_platform/`) but is wired to nothing in this board |
| L3 (die-stacked scratchpad) size/latency | unset | Memory hierarchy stops at L2 → HBM today; no L3 level modelled |
| Selectable Snitch core model | fixed to `accurate` | A faster ISS would drop the decoupled FP subsystem that Xssr/Xfrep measurements depend on; Spatz separately uses its required `SnitchFast` construction |

## Notes for the thesis

- Supported numeric keys are resolved through `pipeline/sweep/design.py`, read
  by the target model, and propagated to the C runtime where required through
  `pipeline/gen_system_header.py` and `runtime/mesh/hes_system.h`. The two VLEN
  keys are build-time dimensions; other supported keys reuse a compatible
  build. Operational propagation and scientific causality must still be
  validated per key.
- The **TCDM banking** of both clusters (32 banks × 8 B, 256 B interleave
  period) is the one hardware detail in this design that is *not* a constant
  in this repo — it's fixed inside GVSoC's own `SnitchCluster`/`ClusterArch`
  model. It matters for the report because it explains why a strided vector
  load collapses onto 1-2 banks for any power-of-two row width (see
  [relatorio-desenvolvimento.md §6.4](relatorio-desenvolvimento.md)) — a
  parameter worth calling out as "fixed by the platform" rather than omitting.
- The clock frequency (`FREQUENCY`) is a single domain for the whole chip by
  design ([targets/hetero/system.py](../targets/hetero/system.py)): a cycle
  difference between engines in the results is a difference in work per
  cycle, never a difference in clocking. Splitting it into per-engine domains
  would be a design change with its own trade-off (cross-domain
  synchronization cost) and is not currently modelled.
