# Metric semantics

M4IA reports timing, correctness, counters, a structural area proxy, and
limited energy-related values. A metric name is not enough to establish its
scope or evidential strength.

Use this template when documenting or exporting a metric:

```text
name | value | unit | scope | classification | source/provenance
safe claim | unsafe claim
```

Useful classifications are `measured`, `derived`, `modeled`, `estimated`, and
`external_reference`.

## Timing

### Node cycles

Generated host code brackets each node with progress instrumentation. For an
offloaded node, the host-observed interval can include dispatch, argument/data
staging, cluster execution, synchronization, and waiting.

Safe claim: end-to-end node latency under the current generated/runtime path.

Unsafe claim: pure kernel-body cycles.

### Total cycles

Total cycles cover the instrumented `RunNetwork()` interval. They include work
inside that call and its progress instrumentation.

Do not label:

```text
total cycles - sum(node cycles)
```

as pure architectural overhead without an independently defined measurement.

### Per-engine cycles

`per_engine_cycles` sums completed node intervals attributed to each logical
engine. It is useful for workload distribution, but concurrent application
stages and host-observed waits mean it is not automatically additive to wall
or total execution time.

### Calibration cycles

The calibration protocol has engine/job and host-observed scopes that are not
identical to node measurements. Keep the scope in the column/series name; do
not combine them silently.

### Wall time

`wall_s` is host/tool/simulator throughput. It is useful for regression and
capacity planning, not simulated architectural latency.

## Correctness and feasibility

Fields such as `status`, `errors`, `maxdiff`, application accuracy, and
agreement with the ONNX reference form the feasibility gate for performance.

```text
correctness/feasibility -> performance comparison
```

`wrong-result`, `stalled`, `no-metrics`, invalid, or failed runs must not be
ranked with correct results unless the analysis explicitly represents them as
infeasible.

## Cache and memory counters

`[HES-MEM]` records are emitted by the timing-cache model and parsed into cache
rows. Current fields include cache name, accesses, reads, writes, hits, misses,
latency cycles, and optional dynamic/leakage energy values when power
accounting is enabled.

These counters cover the whole simulated program, including startup and output
checking; timed operation/network cycles cover a narrower interval.

`[HES-DRAM]` records describe the instantiated modeled device-memory instance with fields
such as kind, reads, writes, bursts, row hits/misses/conflicts, refreshes, busy
time, and latency cycles. Fixed memory is still explicit configuration, but it
does not fabricate modeled-device counters.

Counter interpretation must state the selected memory and cache policy.

## `area_au`

`pipeline/sweep/area.py` computes a structural proxy from:

- I-cache, D-cache, L2, and cluster-TCDM capacities;
- one CVA6 core contribution;
- an Ara vector-unit contribution;
- modeled Snitch-family cores;
- Spatz vector units on every modeled Spatz core;
- a TCDM-crossbar port proxy;
- narrow/wide AXI widths.

Every coefficient in `area.py::COEFFS` is currently marked `sourced: False`:

```text
SRAM_AU_PER_BIT
FP32_FMA_AU
SNITCH_CORE_AU
CVA6_CORE_AU
VECTOR_CTRL_AU
XBAR_AU_PER_PORT
AXI_AU_PER_BYTE
```

Therefore `area_au` is an **estimated structural relative area proxy in
placeholder units**, not silicon area.

Safe claim: under this explicit proxy, configuration A has a larger/smaller
value than configuration B, with sensitivity disclosed.

Unsafe claims:

- the value is mm²;
- coefficients are technology-normalized physical areas;
- equal values establish equi-area designs;
- omitted structures are negligible.

### Report sensitivity

`pipeline/sweep/report.py` first reports measured cycle sensitivity. Its Pareto
pass can use cycles and `area_au`, and may add `cache_dynamic_pj` when every
area-bearing point has it.

When area coefficients are unsourced, the robustness pass perturbs SRAM-heavy
and logic-heavy portions in opposite directions by ±30%. This is a sensitivity
test, not a confidence interval. The report explicitly says to treat cycles as
the only measured objective.

## `--power`

`--power` selects GVSoC power-accounting execution. It causes configured
`vp::PowerSource` activity to be accumulated and lets instrumented timing
caches emit optional energy fields. It does not turn the current result into a
complete SoC energy measurement.

`targets/hetero/power.py` provides coefficients for cache SRAM and TCDM/main
memory component models. The cache/TCDM scaling anchor comes from the GVSoC
Siracusa L1 power tables at 25 °C and 1.2 V; the technology node is not recorded.
Dynamic access energy scales with the square root of capacity and leakage with
capacity. Those geometry rules are explicit first-order assumptions.

## `cache_dynamic_pj`

For sweep rows, `cache_dynamic_pj` is the sum of nonzero `dynamic_pj` values
from parsed cache rows. In the current heterogeneous result path those rows are
the instrumented I-cache, D-cache, and L2 timing-cache records.

Safe claim: modeled dynamic energy reported by the covered cache hierarchy for
the whole-run counter scope.

Unsafe claim: total SoC or application energy.

It does not include a complete accounting of cores, vector units, TCDM,
interconnect, main memory, or all static/background energy.

## Leakage fields

Timing-cache records can contain `leakage_pj` under `--power`, backed by the
scaled SRAM leakage coefficient. Sweep rows do not currently aggregate it into
a first-class energy objective. Treat it as a cache-model field with the same
anchor/node limitations, not as useful total-chip leakage.

## TCDM and main-memory energy

`power.py` can attach scaled SRAM power properties to discovered TCDM banks and
defines main-memory access properties. The current runner/result/sweep path
does not expose those components as a complete TCDM/main-memory energy total.
Their model existence must not be confused with an archived metric.

`DRAM_ACCESS_PJ_ESTIMATE = 20.0` is explicitly an order-of-magnitude,
technology-independent placeholder per memory access. It is not a measured or
device-specific LPDDR/HyperRAM energy value, and no off-chip leakage estimate
is provided.

## Resource counts

`ResourceSummary` reports architectural descriptors such as modeled and useful
cores/lanes. These are not physical costs:

```text
same core count != equi-area or equi-power
same useful lane count != equi-area or equi-power
```

## Recommended analysis table

| Metric | Unit | Scope | Classification |
|---|---|---|---|
| node cycles | cycles | host-observed node interval | measured model output |
| total cycles | cycles | `RunNetwork()` interval | measured model output |
| maxdiff | output units | reference comparison | derived/observed |
| cache counters | events/cycles | whole simulated program | modeled counters |
| cache dynamic energy | pJ | covered caches, whole-run scope | modeled |
| area proxy | `area_au` | current structural formula | estimated |

## Future PPA boundary

Future work may add sourced/reference area, synthesis-calibrated area, broader
energy, cross-calibration, and multiobjective studies. This documentation does
not change any area or energy implementation or strengthen current claims.
