# Documentation instructions

These instructions apply to everything under `docs/`.

## Start here

1. Read [context.md](context.md) for the project handoff.
2. Read only the source document that owns the requested decision:

| Document | Owns |
|---|---|
| [exp/explore.md](exp/explore.md) | Completed NF3 exploration and the data decisions carried into M0 |
| [plan/Model.md](plan/Model.md) | Model shapes, objectives, comparisons, branch order, and scaling |
| [plan/Architecture.md](plan/Architecture.md) | Feature view, causal contexts, tensors, state, inference, and optional interfaces |
| [plan/Thesis.md](plan/Thesis.md) | Datasets, evaluation, acceptance thresholds, supported claims, and schedule |
| [plan/Refs.md](plan/Refs.md) | Paper-to-model comparison and index of paper notes |
| [notes/](notes/) | How each cited paper works, its pros and cons here, sources, and read/reproduction status |

The plan documents are authoritative for exact research specifications.
`context.md` explains history and status; it must not silently replace a plan
contract.

## Writing rules

- Use plain language. Avoid business jargon, filler, sales language, and
  unsupported superlatives.
- State whether a claim is implemented, measured, proposed, conditional, or
  `UNVERIFIED`.
- Never infer that a planned experiment has run because its code or config
  exists.
- Keep each fact in its owning document and link to it elsewhere. Do not copy
  complete model tables, evaluation contracts, or reference records into
  multiple files.
- Every research paper read or used must have a note under `notes/`; link it
  from `Refs.md`. State what was checked and what remains unverified. Explain
  the method and its pros and cons for this project in short, plain language.
- Keep `Refs.md` as the comparison/index, preserve existing reference anchors,
  and prefer original papers, standards, and official dataset pages.
- Treat FlowTransformer as a supervised flow-sequence framework and Anomal-E
  as graph SSL. Neither establishes a Transformer foundation model for
  unlabeled NetFlow.
- Do not claim generic masked reconstruction, EMA teachers, JEPA, hierarchy,
  or relation-biased attention as novel. Preserve the MMAE, CMES, data2vec,
  I-JEPA, V-JEPA, TS-JEPA, and LeNEPA boundaries in `Refs.md`.
- A result on the four converted NF3 benchmarks is a benchmark result. It is
  not operational generalisation or foundation-model evidence.
- Preserve negative-result routes. Failed mechanisms are removed rather than
  retained as decoration.
- Use Markdown tables only for real comparisons or mappings; do not turn the
  whole document into tables.

## Handoff maintenance

Update `context.md` only for durable changes: a merged branch, completed
experiment, changed model contract, new verified evidence, an important
failure, or a new immediate next step. Do not append routine command logs,
verbatim chats, private reasoning, credentials, raw endpoint data, or unrelated
conversation.

When branch copies disagree, record the disagreement and reconcile the owning
plan files. Do not declare one copy canonical merely because it is newer.

## Scope and checks

- A documentation-only change must not alter model code, data, dependencies,
  workflows, or generated artifacts unless the user asks for that broader
  change.
- For Markdown-only edits, review links and formatting. Do not run training or
  unrelated Python checks.
- For code work, use the repository's Pixi commands. Do not substitute `uv`
  commands.
- Preserve unrelated files, local data, outputs, and stashes.
