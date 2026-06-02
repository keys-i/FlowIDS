# Architecture contract

This is a future implementation contract, not a public API or a claim about
the current repository. The model and research scope are specified alongside
[Model](Model.md), [Thesis](Thesis.md), and the cited prior-art anchors in
[Refs](Refs.md).

## Dataflow

`versioned traffic → audit → portable NetFlow → causal ego history → record
encoder → causal Transformer → final-flow representation → heads`

Every arrow is versioned and auditable. A source record is retained with its
conversion evidence; the portable record is the only primary model input;
context is formed only from completed earlier flows; and the representation
used by every head is the target flow's final causal state. No label is loaded
in self-supervised learning (SSL).

### Provenance and portability

A conceptual `DatasetManifest` records dataset/version, raw and converted
hashes, capture lineage, site, collector, exporter, meter and timeouts, time
range, schema, licence, and label provenance. A conceptual `FeatureSchema`
records each raw name, units, semantics, role, type, transform, mask group,
and compatibility status. These manifests are immutable inputs to conversion,
training, evaluation, and replay.

The primary portable view excludes raw IP and MAC addresses, flow ID, labels,
absolute timestamps, capture/day/scenario fields, and post-hoc metadata.
Endpoint addresses may be used only for split grouping, causal routing, and
equality tests. State holds only per-capture keyed endpoint hashes; those
hashes are never model inputs or embeddings. Source/destination orientation
must have stable, documented meter semantics; otherwise directed relation
experiments for that source are unsupported.

Ports are exact tokens only for 0–1023. Higher ports receive a train-only
frequency bucket plus a `registered` (1024–49151) or `dynamic` (49152–65535)
range token; `missing` and `unknown` remain distinct. Every result includes a
mandatory port-free ablation. `L7_PROTO` is excluded from the primary view
unless extractor semantics are equal across sources and it is not a label
proxy.

Numeric features use train-partition median imputation plus a missing indicator,
`log1p` for non-negative values, train-only 1st/99th-percentile clipping, then
z-scoring. Categorical vocabularies are train-only. Encode clipped `log1p`
time since the preceding observed flow and since the preceding flow sharing
either endpoint; never encode wall-clock or absolute capture time. Label maps
are explicitly reviewed; unsupported labels remain `unsupported`, never
silently remapped.

### Curation and splits

The source unit is `(site, collector, capture lineage, calendar block, schema)`.
Reject any source with unknown lineage, time, semantics, or licence. All
conversions of one underlying PCAP are atomic. Remove exact exported-record
duplicates before splitting; cluster near-identical deployable records without
deleting legitimate recurrence; and group similar contexts with
semantic-event MinHash. Near-duplicate and context-linked groups are
indivisible. Scored cross-partition overlap must be zero unresolved.

Entity, campaign, and time partitioning happens before preprocessing. A
conceptual `SplitManifest` stores files and hashes, time ranges, entity and
campaign groups, purge rules, row and context hashes, and pretraining
visibility. It is the authority for all preprocessing fits and for replay.
Synthetic flows are excluded from the foundation corpus and may appear only in
benchmarks. Sealed lineage is never visible during pretraining. Reject the
experiment if exact sealed-flow overlap exceeds 1%, high-similarity context
overlap exceeds 0.1%, held-out lineage appears in pretraining, an
identifier-only classifier has material predictive power, or manifest replay
does not reconstruct the split. Any residual scored overlap remains zero.

Source sampling is `p(d) ∝ n_d^α`, with `α ∈ {0, .5, 1}` selected once at S0;
use 0.5 only if it beats both controls. Each source has a 20% cap and excess
mass is redistributed across uncapped sources. The selected source mixture and
sequence are manifested.

### Causal ego history

For a target completed at `t`, select the latest 128 earlier source-incident
and 128 earlier destination-incident flows. Union, deduplicate, and order them
by flow end time, breaking ties with an audit-only deterministic key. The
source-selected horizon is one of `{1, 10, 60}` minutes. Context contains at
most 255 history records plus the target and never crosses data source,
capture, exporter, split, or time boundaries.

`ContextBatch` conceptually contains numeric, categorical, missingness,
group/padding, causal, and relative-time masks, plus an optional 16-relation
matrix. Streaming state has explicit expiry. Offline replay and streaming must
produce identical IDs, order, masks, and relations. Mutating future or padding
items must leave earlier outputs bitwise unchanged in deterministic mode.

For endpoint-disjoint evaluation, hold out principals, purge cross-boundary
flows, reset state, and admit only earlier history from the same test partition.

### Relation bias and encoder

Each causally visible ordered event pair has four equality bits:
`src-src`, `dst-dst`, `src-earlier-dst`, and `dst-earlier-src`. Their 16
combinations form relation types. A learned scalar per layer and head is added
to causal-attention logits for each type. Raw endpoint identities are never
embedded. Renaming endpoints consistently must preserve outputs; relation
destruction controls preserve relation and relative-time marginals.

The record encoder consumes the portable record and its masks. A causal
Transformer consumes the ordered context and returns the final-flow
representation. This differs from CMES: CMES uses four bits for endpoint,
destination-port, protocol, and feature similarity in supervised,
bidirectional, sorted-group context; this contract uses endpoint-equality bits
only in causal history ([Refs: CMES](Refs.md#cmescrossflow2026)). It also differs from MMAE:
MMAE's support flow is a corruption source for packet patches, not a separately
encoded endpoint-history record ([Refs: MMAE](Refs.md#mmae2026)).

## Heads and runtime

Conceptual output heads are binary attack, supported family, defensible
application/service, and one-class or energy scoring for zero-label settings.
Unsupported labels are not family targets. Bidirectional variants are
nondeployable and evaluation-only.

Runtime is causal per completed flow, with dynamic batching and wait no greater
than 5 ms. Report p50/p95/p99 latency, flows/s, device memory, state memory,
and optional-field missingness behaviour. Missing optional fields use the
defined missing tokens. A checkpoint bundle includes weights, preprocessing,
vocabularies, schemas, manifests, and selection evidence. A conceptual
`RunManifest` records git/config/data/split hashes, seeds, loss weights,
parameters, FLOPs, exposures, hardware, software, and checkpoint selection.

## Conditional extensions

M5 exists only if at least 20% of contexts truncate and long-horizon evidence
is deficient. It uses permutation-equivariant 1 s/10 s/60 s event windows, one
summary per window, a causal summary Transformer, and retains current fine
events. It adds no GNN, SSM/Mamba, memory module, or generative decoder.

X1 packet teacher exists only behind a privacy/legal gate. Source PCAP and its
generated NetFlow must join one-to-one by capture, bidirectional five-tuple,
and exact time overlap; reject ambiguous pairs. No evaluation capture may
train the teacher. A frozen teacher consumes sizes, directions, timings, and
protocol bytes; the student consumes portable NetFlow only; completed-flow
latents align; and no teacher is deployed. Required controls are flow SSL,
same-modality larger teacher, random teacher, shuffled alignment, and frozen
versus full-student training.
