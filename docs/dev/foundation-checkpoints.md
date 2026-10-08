# Experimental Foundation construction history

> **Historical construction context — not the current operational reference.**
>
> This document records how the frozen Experimental Foundation was built and
> validated on the historical `lucas-ic` lineage. Current M4IA source and the
> maintained [Experimental Workbench](../experimental-workbench.md) define
> present behavior. See [integration notes](integration-notes.md) for the
> semantic port into current M4IA.

## Lineage and attribution

```text
historical M4IA/K9R5 base
6d40999f244c038f882c5746381f35d565432dcc
        |
        v
45bbb7da05b1df139f03ebf098ffcc2d751262ee
Freeze experimental foundation M1-M8
        |
        v
e97b2dced29352a519d320aeb0b8f871580a589e
Complete experimental provenance foundation M9
        |
        v
ab188c069157161a4a49cdfca0ebabd98c0f810a
frozen lucas-ic reference used for the M4IA port
```

M1–M8 were logical development checkpoints committed together at `45bbb7d`;
they are not nine independent Git commits. M9 was committed at `e97b2dc`.
Some fingerprint/provenance primitives were already in the working series when
the numbered milestones began and are described as Foundation components, not
retroactively attributed to a milestone that did not create them.

The historical base was already a substantial heterogeneous platform. It
contained CVA6/Snitch/Spatz target construction, Deeploy generation, the
host/cluster mailbox runtime, tuned Snitch SSR/FREP and Spatz RVV kernels,
Generic fallbacks, the cost mapper, design sweeps and calibration, application
paths, memory instrumentation, Docker workflow, and GUI support. The Foundation
organized, described, audited, and hardened those mechanisms; it did not create
or claim authorship of the underlying operational stack.

## Conceptual contribution

The work grouped naturally into three areas:

```text
RESEARCHER INTERFACE / DISCOVERY
ExperimentRequest, deterministic resolution, catalogs, workload inspection,
and machine-readable discovery

AUDITABILITY / EXTENSION STRUCTURE
ResolvedExperiment, ResourceSummary, mapping explanation, implementation
evidence, fingerprints, provenance, manifests, and extension seams

SCIENTIFIC SAFETY / HARDENING
unknown remains unknown; missing cost is not zero; fallback is not tuned;
stale or contradictory artifacts are not silently accepted
```

## M1 — controlled pinning

The milestone carried the existing direct-run compatible-node pin through the
historical sweep/request metadata. Its invariant was already important:

```text
pin == compatible-node preference
pin != whole-graph isolation
```

The current M4IA direct runner retains this meaning. Current sweep CLI does
**not** expose a sweep-level `--pin`; that historical surface was not blindly
ported. See the [quickstart](../experimental-workbench-quickstart.md).

## M2 — ResourceSummary

`ResourceSummary` made modeled resources explicit so labels such as “nine
cores” could not silently imply equal compute resources. It distinguished
modeled cores, useful compute cores, a control/DMA core, TCDM organization, and
modeled versus useful Spatz lanes. It deliberately did not infer area, power,
or architectural equivalence.

The current port keeps memory separate from this compute/resource summary.
Main memory is resolved machine identity, not a resource-summary field.

## M3 — kernel implementation catalog

The Foundation assigned stable IDs to audited FP32 MatMul/Gemm paths already
present in source and connected each identity to source, symbol, build
override, dispatch, and fallback assumptions. It did not introduce a second
selector. Current IDs and source checks live in
`pipeline/experiment/kernel_implementation_catalog.py`.

## M4 — actual implementation and fallback evidence

This checkpoint connected completed runtime nodes to a concrete implementation
only when generated dispatch evidence and audited rules made that conclusion
safe. Placement alone was insufficient. Missing or contradictory evidence
remained unknown; tuned-to-Generic fallback was explicit.

The current M4IA port further names generated evidence as its own stage:

```text
REQUESTED -> RESOLVED -> GENERATED -> ACTUAL -> MEASURED
```

## M5 — mapping explainability

Structured explanations recorded strategy identity, compatibility, pin
filtering, cost availability, rate source, offload terms, selection rule, and
selected engine. The metadata described the existing decision and was not
allowed to influence it. Exposing those inputs revealed the ambiguities then
fixed in M6.

## M6 — scientific-safety hardening

The source audit established the following rules:

1. unknown tensor size is not zero;
2. incomplete MAC estimates do not become fabricated costs;
3. an unknown engine does not borrow CVA6 rates;
4. missing or invalid rate/offload data remains unavailable;
5. `_default` is a labeled proxy, not an operator measurement;
6. an external rate table replaces rather than partially merges active data;
7. known whole-kernel Generic fallback is not priced as tuned execution;
8. a Snitch scalar tail remains part of the tuned implementation;
9. explicit pin semantics remain compatible-node preference.

Safely priced mappings were preserved. Only previously unsafe or ambiguous
automatic choices were intentionally allowed to change.

## M7 — extension-seam audit

The audit identified operational owners, identity and provenance effects,
tests, and limitations for workloads, knobs, kernels, mapping strategies,
hosts/simulators, precision, and future engines. Its conservative finding was:

- workloads, numeric knobs, and kernels: ready with local change;
- mapping strategies, hosts/simulators, and future engines: fragile;
- precision: blocked pending an end-to-end dtype contract.

Current memory and GUI-facing seams were added during the semantic M4IA port.
See the maintained [technical seam audit](extension-seams.md) and practical
[extension guide](../extending-m4ia.md).

## M8 — freeze validation and harness discoveries

The freeze combined unit, structural, golden/regression, adversarial, container
integration, and scientific-invariant checks. It found two harness defects
important beyond the historical branch:

1. an Ara experiment could select the Ara final target while calibration built
   the scalar CVA6 host image;
2. a reused sweep output could retain an old copied `runtime/mesh` tree.

The Foundation aligned the selected calibration host and refreshed the copied
mesh before header generation. The current port re-audited and retained both
protections against current M4IA APIs.

## M9 — provenance and completed-run identity

M9 separated actual from measured data, strengthened workload identity,
separated run provenance from calibration provenance, centralized the audited
MatMul/Gemm dispatch interpretation, and made causal run sources part of the
completed fingerprint. It also corrected a stale Spatz core-count comment
without changing the operational count.

Historical final validation recorded:

```text
Python suite: 106 tests passed

Pinned Spatz, 32x32x32 MatMul:
  status       ok
  node cycles  6694
  total cycles 8183
  maxdiff      0.0
```

Those numbers belong to the frozen environment and are evidence of a
historical regression check, not current universal performance claims.

## Audited dispatch facts retained by the port

For the frozen and current audited FP32 MatMul/Gemm path:

```text
Snitch:
  empty required dimension -> Deeploy Generic whole-kernel fallback
  O < 8                    -> Deeploy Generic whole-kernel fallback
  O >= 8                   -> tuned SSR/FREP
  O >= 8 with O % 8        -> tuned path with internal scalar tail

Spatz:
  empty dimension          -> Deeploy Generic whole-kernel fallback
  Gemm transA or transB     -> Deeploy Generic whole-kernel fallback
  otherwise                -> tuned RVV
```

C/runtime source remains operational truth. The Python interpretation exists
for safe costing and evidence classification, not to control dispatch.

## Deliberate non-claims

The historical Foundation did not claim:

- authorship of the underlying SoC, runtime, mapper, or tuned kernels;
- that mapper prediction or placement proves actual execution;
- that pinning isolates a whole graph;
- that node latency is pure kernel-body time;
- that equal cores or lanes imply equal area or power;
- that every exposed knob is causally validated;
- that the current mapper is globally optimal;
- that one architecture is generally superior.

The original milestone task prompts remain development records rather than
maintained product documentation. Their durable rationale is preserved here;
current behavior belongs in the Workbench, architecture, metric, and extension
documents.
