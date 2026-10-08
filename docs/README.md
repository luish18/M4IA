# M4IA documentation map

Use this page to choose the document that matches your goal. Current
operational source remains authoritative if documentation and behavior ever
disagree.

| Reader goal | Start here | Role |
|---|---|---|
| Understand M4IA and set up the repository | [Main README](../README.md), then [Architecture](architecture.md) | Current operational reference |
| Run and inspect an experiment | [Experimental Workbench quickstart](experimental-workbench-quickstart.md) | Executable tutorial |
| Understand request, resolution, evidence, identity, provenance, and manifests | [Experimental Workbench](experimental-workbench.md) | Scientific/experiment semantics |
| Interpret cycles, counters, area, and energy-related fields | [Metric semantics](metrics.md) | Scientific semantics and non-claims |
| Inspect current hardware/design parameters | [Chip design parameters](chip-design-parameters.md) | Current source-oriented inventory |
| Add a workload, knob, memory, kernel, strategy, host, metric, precision, engine, or GUI capability | [Extending M4IA](extending-m4ia.md), with the [extension-seam audit](dev/extension-seams.md) | Practical maintainer guide plus technical rationale |
| Continue the Rust GUI/backend integration | [GUI integration](gui-integration.md) | Maintainer continuation guide |
| Understand historical project development and architecture rationale | [Development report](relatorio-desenvolvimento.md) and [heterogeneous-mesh proposal](hetero-mesh-plan.md) | Historical/rationale |
| Understand Foundation construction and what changed from pre-Foundation M4IA | [Foundation checkpoints](dev/foundation-checkpoints.md) for M1–M9, then [integration notes](dev/integration-notes.md#before-and-after-the-experimental-foundation) for the modern before/after semantic-port record | Historical construction and maintainer integration record |
| Continue the scientific program | [Research roadmap](dev/research-roadmap.md) | Forward-looking roadmap |

## Role boundaries

- **Current operational reference:** the main README, architecture, and chip
  inventory describe current commands, owners, and modeled configuration.
- **Tutorial:** the quickstart provides source-checked commands; it does not
  replace the semantic reference.
- **Scientific semantics:** the Workbench and metrics documents define what
  identities and observations mean—and what they do not prove.
- **Maintainer guides:** the practical extension and GUI guides explain how to
  continue the system; the seam audit explains why some paths remain fragile
  or blocked.
- **Historical/rationale:** the development report and mesh proposal preserve
  project chronology; Foundation checkpoints preserve M1–M9 construction;
  integration notes preserve the modern semantic port and the pre/post state.
  None supersedes current operational source.
- **Forward-looking roadmap:** the research roadmap sequences future studies;
  it is not evidence that every proposed axis is already causally validated.

Across all roles, preserve:

```text
REQUESTED -> RESOLVED -> GENERATED -> ACTUAL -> MEASURED
```

Catalog or discovery metadata is not operational support, and an exposed
parameter is not automatically a validated scientific axis.
