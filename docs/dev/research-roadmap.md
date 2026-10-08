# M4IA scientific research roadmap

This is a forward-looking sequence after the Experimental Foundation. It does
not repeat the historical M1–M9 construction record and does not promise that
an exposed knob is already causally validated.

Guiding rule:

> Minimal input, deterministic resolution, explicit ambiguity, and preserved
> evidence before architectural claims.

## 1. Causal knob audit

For each proposed independent variable, trace:

```text
request -> resolved value -> generated/target configuration
        -> runtime-visible consequence -> measured response
```

Classify knobs as validated, descriptive-only, partially connected, or unsafe.
Discovery and fingerprint changes alone are not causal validation.

## 2. External and reference validation

Compare selected kernels, counters, dispatch rules, and memory effects against
the best available Deeploy, Snitch, Spatz, GVSoC, RTL, or literature reference.
Record tool/source versions and distinguish exact comparison from qualitative
sanity checking.

## 3. Deterministic pilot

Choose a small workload/shape set and freeze resolved experiments,
calibrations, provenance, correctness thresholds, and analysis scripts. Repeat
runs to establish determinism and variance before launching a campaign.

## 4. PE/core scaling

Study Snitch and Spatz modeled core/PE counts separately. Keep control/DMA
cores, useful compute cores, memory capacity, and mapper placement explicit.
Do not equate core count with equal area or power.

## 5. Lane scaling

Vary Spatz lanes only after confirming that generated headers, target model,
runtime/kernel assumptions, working-set constraints, and useful-lane reporting
all follow the resolved value.

## 6. PE × lane interaction

Run a controlled factorial study after the individual axes are defensible.
Look for saturation, TCDM pressure, issue/dispatch limits, and workload-shape
interactions rather than presenting a single universal ranking.

## 7. Shape and reuse regimes

Use matrix/tensor shapes that expose reuse, working-set, tail, and fallback
boundaries. Preserve generated arguments and actual implementation evidence so
tuned and Generic regimes are not mixed silently.

## 8. Mapping studies

Compare the current automatic strategy with controlled compatible-node direct
pins or future validated strategies. Separate predicted cost, generated
placement, completed implementation, and measured performance. The current
sweep CLI has no sweep-level `--pin`; campaigns must not invent one.

## 9. Memory and TCDM studies

Study fixed, LPDDR4, LPDDR4X, LPDDR5, HyperRAM, cache policy, TCDM capacity,
and bank/traffic effects only after model/reference validation appropriate to
each claim. Keep memory, calibration, run identity, counters, and effective
overrides explicit. Current energy fields are not total SoC energy.

## 10. Application studies

Apply validated mechanisms to MNIST, KWS, and later applications. Preserve
accuracy/correctness as a feasibility gate, report application-specific
frontend and sample semantics, and retain completed-only actual evidence.

## Later engineering and research

After the controlled sequence, possible directions include:

- Deeploy tiling and scheduling integration;
- double buffering and DMA overlap;
- precision/quantization with a first-class dtype contract;
- broader or source-backed PPA models;
- stronger mapper objectives and strategy selection;
- new engines and topologies;
- adaptive design-space exploration.

Each remains subject to the extension-seam audit. Catalog metadata must follow,
not precede, operational support. See [extension seams](extension-seams.md),
[practical extension recipes](../extending-m4ia.md), and
[metric semantics](../metrics.md).
