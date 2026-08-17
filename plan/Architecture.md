# Architecture contract

This is the sole data, tensor, causal-state, and deployment contract for the
proposed system. It defines future implementation behaviour, not a public API.
Model objectives and promotion gates belong in [Model](Model.md); study claims,
datasets, and evaluation tracks belong in [Thesis](Thesis.md); source evidence
and prior art belong in [Refs](Refs.md).

## Boundary and dataflow

```text
versioned source -> provenance audit -> portable FlowRecord -> causal endpoint
history -> field tokenizer -> causal Transformer -> completed-flow embedding ->
task head
```

The prediction unit is one completed bidirectional flow. A unidirectional
source is usable only after a documented causal pairing rule produces a
completed bidirectional record; otherwise it is excluded. Source records,
conversions, and packet captures never cross a capture/exporter observation
boundary in context construction.

`explore` supplies useful DuckDB audit queries. Its query output is not
immutable D0 evidence: D0 requires versioned source files, frozen audit output,
and the artifacts below.

## Versioned artifacts and isolation

| Artifact | Required content | Isolation/invariant |
|---|---|---|
| `DatasetManifest` | Dataset/version; source and converted hashes; capture lineage; site, collector, exporter, meter and timeout/version; time range/timezone; schema hash; licence; label provenance. | A capture and every derivative share one lineage ID. |
| `FeatureSchema` | Raw name, units, semantics, type, portable role, transform, semantic mask group, and cross-corpus status. | Derived from source metadata, never inferred from labels. |
| `FlowRecord` | Lineage and capture IDs; interval and completion time; typed observable fields; field-presence mask; audit routing keys. | Raw endpoints are routing-only; labels and post-hoc fields are absent. |
| `PacketPair` | Flow-to-packet-span mapping; canonical pairing evidence; confidence; ambiguity status. | Exists only for Q3 sources; ambiguous records are discarded. |
| `SplitManifest` | Source hashes; partition; chronological range; entity/session/campaign groups; purge interval; retained event IDs; exact/near contract and membership hashes; context membership hashes; pretraining visibility. | It is the single authority for partition, fit scope, and state reset. |
| `LabelTable` | Reviewed raw-to-canonical label map; label provenance; supported/unsupported status. | Physically and logically unavailable to canonicalization and the SSL dataloader. |
| `ContextBatch` | Numeric values; categorical IDs; field-presence and missing masks; explicit semantic-group mask; padding and causal masks; elapsed-time features; optional 16-type relation matrix. | Target is final; every history event completed earlier than it. |
| `RunManifest` | Git/config/data/split hashes; seeds; loss definition and frozen weights; parameters, FLOPs, unique flows and exposures; hardware/software; checkpoint rule. | Written before training and retained with output. |
| Checkpoint bundle | Encoder weights; preprocessing state; vocabulary; feature/label schema; data, split, and run hashes. | Sufficient for deterministic offline replay of the served encoder. |

No artifact is added by this documentation change. A persisted artifact is
versioned, content-addressed, and immutable once referenced by a run.

## Source quality and capture lineage

Every source is classified before use.

A provenance-disjoint source unit is the smallest stable combination of site,
collector/exporter observation point, capture lineage, calendar block, and
schema. Units sharing any underlying capture are one source for scaling counts.

| Tier | Minimum evidence | Permitted use |
|---|---|---|
| Q0 | Provenance, timing, schema, or rights are missing or unresolved. | Exclude entirely when use rights or field semantics are unresolved. Otherwise allow quarantined benchmark reproduction/diagnostics only; exclude from the core corpus, source-diversity counts, ladder promotion, and transfer/operational/foundation claims. |
| Q1 | Valid provenance, schema, and completion-time ordering; incident state may be unknown. | General SSL only. |
| Q2 | Q1 plus independently reviewed clean interval. | Q1 uses plus benign-only fitting and threshold calibration. |
| Q3 | Q2 plus deterministic PCAP-to-flow pairing meeting the contract below. | Q2 uses plus optional packet-teacher work. |

A PCAP, its exported NetFlow/CSV/Parquet conversions, resampling, redaction,
and duplicate exports are atomic capture derivatives. They occupy one source
lineage and one partition. A source with unknown lineage, time range, feature
meaning, meter semantics, or licence is Q0. Q0 never enters the core corpus. A
source with unresolved use rights or field meaning is excluded entirely. Other
Q0 material may enter a quarantined benchmark run that executes only checks
whose prerequisites are present; missing grouping/provenance gates stay
explicitly blocked, and a duplicate match cannot raise the tier.

### Exact and NF3-v1 duplicate grouping

Exact-record and strict deterministic near groups are hard split groups, not a
row-deduplication instruction: every retained occurrence and its multiplicity
remain represented. Exact-record equality is HMAC-SHA-256 equality over the
canonical typed serialization of every non-target exported field, using the
same lineage-scoped secret as the endpoint routing keys. Labels and derived
targets are excluded; the plaintext serialization and an unkeyed digest
containing endpoint values are never persisted.

`NF3-v1` compares records only within the same declared capture lineage and
observation domain. Its fixed canonical typed tuple is lineage ID; observation
domain ID; `HMAC-SHA-256(K_lineage, canonical endpoint bytes)` in the declared
source and destination slots; exact `L4_SRC_PORT`, `L4_DST_PORT`, `PROTOCOL`,
`FLOW_START_MILLISECONDS`, `FLOW_END_MILLISECONDS`, `IN_PKTS`, `OUT_PKTS`,
`IN_BYTES`, `OUT_BYTES`, `TCP_FLAGS`, `CLIENT_TCP_FLAGS`, and
`SERVER_TCP_FLAGS`; then the presence mask for those fields. Timestamp
tolerance is zero. No other field participates: labels and exporter-derived
secondary fields are excluded. Sort by this tuple and assign each equal-key run
one group while retaining every member. MinHash, LSH, fuzzy distance, tolerance
windows, and transitive or connected-component closure are forbidden
([grouping evidence](Refs.md#duplicate-grouping-evidence)).

Freeze `contract_hash = SHA-256(JCS(contract))` for the exact-record and
`NF3-v1` contracts; each document includes its version, typed field order and
normalization, equality rule, and boundary policy, plus the HMAC algorithm and
non-secret key identifier/scope where applicable. For every group, freeze
`membership_hash = SHA-256(JCS(membership))`; that document contains the
contract hash, key hash, and sorted event IDs with one entry per retained
occurrence. HMAC secrets are never manifested.
If a hard group meets a chronological boundary, assign all retained members to
the later partition or purge the earlier members; never move a later member
into training. All hashes and parameters are frozen before model fitting.

Semantic context similarity is diagnostic only: record its frozen method and
score distribution, but do not use it to assign partitions or pass D0. It may
become a thresholded hard gate only after independently known repeat-export
positives justify a predeclared equivalence rule, threshold, and error
assessment in a versioned contract revision
([context-similarity evidence](Refs.md#context-similarity-evidence)).

On labelled sources, run a shallow tree for each candidate model field and a
separate identifier-only tree over audit fields, both on grouped validation
data. Balanced accuracy or AUROC at least 0.99 triggers a source/split
investigation; exclude any implicated model-visible proxy or revise the split,
then rerun the complete audit. Audit-only identifiers remain forbidden
regardless of score. Synthetic traffic is excluded from the core corpus.

During pretraining, source `d` is sampled with
`p(d) proportional to n_d^alpha`, where `n_d` is its post-filter unique-flow
count and `alpha` is selected once at S0 from `{0, 0.5, 1}`. No source may
exceed 20% probability; redistribute excess mass across uncapped sources.
Freeze the selected source sequence by seed and record its hash. Use
`alpha=0.5` only if it beats both controls under the S0 transfer screen.

### D0 acceptance evidence

Every applicable item below must pass before a source or result enters the claim-bearing
model ladder. A Q0 benchmark-only run sits outside that ladder: it may execute
only supported checks, must report every unavailable gate as blocked, and
cannot promote a rung. A query that merely prints a count is not a gate; its
frozen result and pass/fail rule are part of the run evidence.

| Check | Required evidence |
|---|---|
| Conversion parity | Reconcile source/output rows and every dropped record; verify units, directionality, missingness, completion timestamps, and chronological ordering against source metadata and sampled raw records. |
| Input isolation | Schema and batch assertions prove labels, raw identifiers, capture metadata, and post-hoc fields cannot enter canonicalization or SSL tensors. |
| Split isolation | Exact-record and `NF3-v1` groups never cross a partition in any track. Each track additionally isolates its declared factor: time with boundary purge, endpoint principals, complete family/campaign units, or source/capture lineage. Post-split contexts share zero event IDs across partitions. Semantic context similarity is diagnostic, not a pass/fail criterion. |
| Fit scope | Changing validation/test records cannot alter training transforms, vocabularies, buckets, duplicate parameters, or feature selection. |
| Causality and state | Future/padding mutation leaves earlier outputs unchanged; offline replay equals streaming construction; partition reset prevents state crossing. |
| Reproducibility | The same inputs and seeds reproduce artifact hashes, retained event IDs, contexts, and split assignments. |
| Q3 pairing (Q3/X1-Distill only) | Ambiguous or boundary-crossing matches are rejected and the audited precision gate below passes. |
| Release isolation | Checkpoints, logs, and released manifests contain no payload, raw endpoint, re-identifying pair map, or private per-event embedding. |

## Canonical feature view

The headline view is one schema frozen before training as the semantic
intersection of completion-observable fields across every domain in the
primary claim: protocol; service-aware ports; directional packet/byte volume;
duration; TCP flags; and compatible packet-size, retransmission, and
inter-arrival summaries. It is unchanged across primary domains and tasks so a
single checkpoint is testable. A pair-specific intersection is a secondary
sensitivity analysis, never headline evidence.

The secondary view may include source-specific compatible fields and has an
explicit field-presence mask. It must never replace the headline view, and its
results are labelled secondary. Missing is distinct from zero in both views.

The primary view excludes IP/MAC addresses, flow IDs, labels, absolute
timestamps, capture/day/scenario IDs, hostnames, collector/template IDs, and
post-hoc metadata. Endpoint values may produce only the lineage-scoped HMAC
routing keys defined above, retaining declared source/destination position, for
split grouping, causal lookup, and anonymous equality relations. Those keys,
raw values, and any stable endpoint embedding never enter model tensors.
Directed experiments require stable, documented source/destination semantics.

Ports use exact tokens for 0--1023. A higher training port receives one of
eight quantile buckets by training-only occurrence frequency plus a
`REGISTERED` (1024--49151) or `DYNAMIC` (49152--65535) range token; an unseen
higher port receives `UNK` plus its range. Keep separate `PAD` and `MISSING`
tokens. A port-free result is mandatory.
`L7_PROTO` is excluded from the headline view. It may enter the secondary view
only when extractor semantics are equivalent across the relevant sources and it
is not a task-label proxy.

Numeric processing is fit on the permitted training partition only: median
imputation, a missingness indicator, `log1p` for non-negative heavy-tailed
fields, 1st/99th percentile clipping, then z-score scaling. Categorical
vocabularies and port-frequency buckets are likewise training-only. Time input
is clipped `log1p` elapsed time since the preceding observed flow and since the
preceding flow sharing either endpoint; wall-clock and absolute capture time
are forbidden. Reviewed labels remain in `LabelTable`; unsupported classes stay
unsupported and are never fuzzy-matched or silently folded into another class.

## Causal context and tensors

Construct contexts separately for each partition only after the frozen split
and boundary policy have been applied and endpoint state has been reset. The
cross-partition context audit must find zero shared event IDs and zero exact or
`NF3-v1` group IDs represented on both sides. Canonical context membership
hashes are replay and shared-event evidence, not semantic group IDs.

Every context draws from flows completed before target time `t` in the same
source, capture, exporter, partition, and selected 1/10/60-minute horizon.
Break equal completion times with a deterministic audit-only event key. Append
the completed target last and cap the full sequence at 256 tokens. No control
may change these boundaries, ordering rules, target position, or state reset.

M0 through M2-F use flat history: retain the latest 255 eligible collector
events irrespective of endpoint. M3-Ego replaces that selection rule by
retrieving the latest 128 eligible events incident to the target source and the
latest 128 incident to its destination, then unioning, deduplicating, ordering,
and retaining the latest 255. For the M3-Ego screen, let `h` be that ego
history's event count and use exactly `h` events in every history-matched
control; target-only remains the deliberate zero-history ablation:

- `flat-matched`: the latest `h` eligible collector events;
- `random-matched`: the `h` non-incident candidates with the lowest
  `SHA-256(run_seed || target_event_id || candidate_event_id)` values;
- `time-feature-matched`: for each ego event from newest to oldest, select the
  unused non-incident candidate minimizing protocol mismatch, port-range
  mismatch count, absolute completion-lag difference, then L1 distance over
  frozen train-only transformed duration/byte/packet fields, with event ID as
  final tie-breaker;
- `target-only`: no history event.

If a non-incident control has fewer than `h` candidates, mark that target
unsupported for the matched comparison; never backfill across a boundary.

For the endpoint-disjoint track, assign held-out endpoint principals before
context construction. A flow whose two endpoints are held belongs to the
target partition; a flow joining a held and unheld endpoint is purged; all
other flows belong to the source partition. Reset state, and admit only earlier
flows from the same target partition at test time.

Offline replay and streaming construction must emit identical event IDs, order,
padding, causal masks, relation types, and elapsed-time features. Future events
and padding must leave an earlier deterministic output bitwise unchanged.
Endpoint state expires at the selected horizon. At every adaptation,
calibration, or test boundary, clear state; within a partition, test state may
contain only its own earlier completed flows.

`ContextBatch` represents semantic masking as a distinct `[batch, event,
group]` boolean mask, not missingness or padding. The five groups are transport
and flags; service and ports; directional volume; packet-size and
retransmission statistics; and duration and inter-arrival statistics. Group
membership comes only from `FeatureSchema`.

For each causally visible ordered pair, encode the four bits `src-src`,
`dst-dst`, `src-earlier-dst`, and `dst-earlier-src` as one of 16 relation
types. A learned scalar for each layer, head, and type is added to the causal
attention logit. Raw identities, port/protocol similarity, and feature
similarity are not relation inputs in the proposed architecture.
Consistently renaming endpoint routing keys must leave model outputs unchanged.
The destruction control permutes relation types only within
source/day/relative-time strata so relation and timing marginals are preserved.

## Encoder and inference envelope

One completed flow is one token. The exact record encoder, Transformer shape,
parameter accounting, and inference-time removal of SSL heads are defined once
in [Model](Model.md#locked-deployable-backbone). The final completed-flow state
feeds every downstream head. Bidirectional models are offline-oracle upper
bounds, never deployable equivalents.

Primary scoring happens at flow completion from that flow and strictly earlier
completed flows. The only offline oracle may use full bidirectional context and
is explicitly labelled nondeployable. Live packet-prefix scoring is excluded:
it requires a versioned active-flow snapshot source with point-in-time field
availability and separate feature semantics; without that source no
packet-prefix or real-time claim is evaluated.

The service target is p95 CPU inference no greater than 2 ms per completed
flow, exclusive of dynamic-batching wait; batching may wait at most 5 ms.
Endpoint state is at most 32 KiB per active endpoint under the configured
horizon. Report p50/p95/p99 latency, completed flows/s, device memory,
endpoint-state memory, and missing-field behaviour. Missing optional fields
map to the declared tokens and masks, never to another architecture.

## Conditional packet teacher (X1-Distill)

X1-Distill is disabled unless Q3 paired data are legally usable and a refreshed
novelty review permits the experiment. A pair is accepted only when canonical
bidirectional five-tuple, protocol, overlapping time interval, and packet/byte
evidence identify exactly one flow-to-packet span. On 500 manually audited
pairs, the one-sided 95% lower confidence bound of pairing precision must be at
least 99%. Ambiguous, unmatched, and boundary-crossing records are excluded.

No teacher pretraining, packet content, or pair from an evaluation capture may
enter training. The frozen teacher may use packet sizes, directions, timings,
and protocol bytes; the student receives the headline NetFlow view only and
aligns completed-flow latents. The teacher is absent at deployment.

Packet material and pair maps are access-controlled, retention-limited,
encrypted at rest where required, and never released with endpoint keys,
payload bytes, or re-identifying join evidence. Any released X1-Distill artifact is a
flow-only checkpoint plus aggregate pairing statistics and approved,
non-reidentifying metadata. Before release, perform a predeclared membership
inference test, canary-exposure/retrieval test, nearest-neighbour payload audit,
and representation-inversion review; a failed privacy or governance review
removes X1-Distill without blocking the NetFlow-only study.

## Conditional hierarchy (M5-Hier)

M5-Hier exists only after the trigger in [Model](Model.md#evidence-ladder). It uses
fine 1-second, medium 10-second, and coarse 60-second windows. Within each
closed history window, permutation-equivariant attention uses elapsed-time
features and the anonymous endpoint relation tensor; one summary represents
each non-empty window. Current fine-window events remain ordered causal tokens.
It receives the identical M3-Ego history and M4-Rel relation tensor as its
non-hierarchical control; only the aggregation path changes.

M5-Hier reallocates rather than adds Transformer blocks: the first two of the fixed
eight blocks form one weight-shared window encoder at all three scales, with
RoPE disabled inside the set and a learned scale token; the remaining six
blocks apply causal attention with RoPE to time-ordered summaries followed by
the current fine-window events. There is no second independent Transformer
stack. Scale tokens, summary projections, record encoder, and all eight blocks
count toward the same complete 25M ±5% deployable band; reduce FFN width before
the first M5-Hier run if necessary, then retrain the M4-Rel control at that same width
and freeze it across the matched comparison. FLOP reports include every
repeated window-encoder pass and the six-block causal stage.

M5-Hier does not add a GNN, SSM/Mamba block, memory bank, or generative decoder.
Those remain challengers only if a measured bottleneck justifies a separately
gated study.
