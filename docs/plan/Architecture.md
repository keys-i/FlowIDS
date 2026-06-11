# Architecture contract

This document fixes the model-facing data and inference contract for the
proposed system. It is not a public API. Model objectives, sizes, and promotion
criteria live in [Model](Model.md); datasets, evaluation tracks, and claim
limits live in [Thesis](Thesis.md); prior work lives in [Refs](Refs.md).

## Scope

The supplied datasets are accepted as benchmark inputs. Results based only on
these datasets are benchmark results, not proof of live-network performance,
cross-network transfer, or a broad foundation model.

## Exploration

Exploration is complete. The existing DuckDB queries were used to inspect
schema, missingness, timestamps, labels, duplicates, and feature choices. The
[recorded findings](../exp/explore.md) constrain M0 implementation.

## M0 implementation

At the M0 implementation stage, code under `src` constructs evaluation splits,
fits preprocessing on training data only, checks causal context and state
behaviour, and records the checkpoint details needed by the implementation.

**M0 Small** is causal with seven earlier completed flows plus its target, for
eight tokens. **M0 Base** is the unrestricted 25M offline model with up to 255
earlier completed flows plus its target, for 256 tokens. **M0 Matched** is the
causal 25M/256-token model. It has the same parameterization as M0 Base except
for its attention mask. Promoted M0 and every later rung mean M0 Matched.

## Prediction unit and feature view

The prediction unit is one completed bidirectional flow. A unidirectional
source may be used only when a documented pairing rule produces a completed
bidirectional record; otherwise it is excluded. Source records and packet
captures do not cross their capture/exporter observation boundary while context
is built.

The headline feature view is the completion-observable semantic intersection
used by the primary benchmark: protocol; service-aware ports; directional
packet/byte volume; duration; TCP flags; and compatible packet-size,
retransmission, and inter-arrival summaries. It is fixed across the primary
domains and tasks. Pair-specific intersections are secondary sensitivity
analyses, not headline evidence.

A secondary view may add compatible source-specific fields with an explicit
field-presence mask. Missing is distinct from zero. The headline view excludes
IP/MAC addresses, flow IDs, labels, absolute timestamps, capture/day/scenario
IDs, hostnames, collector/template IDs, and post-hoc metadata. Raw endpoint
values and stable endpoint embeddings never enter model tensors.

Ports use exact tokens for 0--1023. Higher training ports use one of eight
training-frequency buckets plus a `REGISTERED` (1024--49151) or `DYNAMIC`
(49152--65535) range token; unseen higher ports use `UNK` plus their range.
`PAD` and `MISSING` remain separate. A port-free result is required.
`L7_PROTO` stays out of the headline view; it may be used only in the
secondary view when its semantics are compatible and it is not a label proxy.

M0 fits numeric processing on training data only: median imputation with a
missingness indicator, `log1p` for non-negative heavy-tailed fields, 1st/99th
percentile clipping, then z-score scaling. Categorical vocabularies and port
buckets are training-only. Time is clipped `log1p` elapsed time since the
preceding observed flow and preceding flow sharing either endpoint; wall-clock
and absolute capture time are forbidden. These elapsed values remain batch
fields reserved for later ablation; M0 does not project or otherwise consume
them.

## Context and relations

M0 Base and M0 Matched contexts contain at most 255 earlier completed flows and
the completed target, appended last. Context uses a selected 1-, 10-, or
60-minute horizon; equal completion times have a deterministic tie-break. M0
defines partition boundaries and resets state at each boundary. Future data,
padding, labels, raw identifiers, and post-hoc fields must not affect the
encoder input.

M0--M2-F use the latest 255 eligible collector events. M3-Ego takes the latest
128 eligible events incident to the target source and the latest 128 incident
to its destination, unions and deduplicates them, orders them, and retains the
latest 255. With ego-history size `h`, matched controls use: the latest `h`
collector events (`flat-matched`); the `h` non-incident candidates with the
lowest `SHA-256(run_seed || target_event_id || candidate_event_id)`
(`random-matched`); or non-incident events matched from newest to oldest by
protocol mismatch, port-range mismatch count, absolute completion-lag
difference, then L1 distance over train-transformed duration/byte/packet
fields, with event ID as final tie-breaker (`time-feature-matched`).
`target-only` has no history. A comparison target without `h` non-incident
candidates is unsupported.

For endpoint-disjoint evaluation, assign held-out endpoint principals before
context construction: flows with two held endpoints are test; flows joining a
held and unheld endpoint are purged; the rest are training. Test context uses
only its own earlier completed flows.

Offline replay and streaming construction must produce identical event order,
padding, causal masks, relation types, and elapsed-time batch fields. State
expires at the selected horizon and is cleared at adaptation, calibration, and
test boundaries.

Semantic group masking is distinct from missingness and padding. The groups
are transport/flags, service/ports, directional volume, packet-size and
retransmission statistics, and duration/inter-arrival statistics.

For each causally visible ordered pair, the four endpoint-equality bits
`src-src`, `dst-dst`, `src-earlier-dst`, and `dst-earlier-src` select one of 16
relation types. They are anonymous equality relations only: raw identities,
port/protocol similarity, and feature similarity are not relation inputs.
Each layer and head adds a learned scalar for its relation type to the causal
attention logit. Renaming endpoint routing keys must not change outputs. The
destruction control permutes relation types only within source/day/relative-time
strata.

## Encoder and inference envelope

One completed flow is one token. The record encoder, Transformer shape,
parameter accounting, objectives, and removal of SSL heads are defined once in
[Model](Model.md#locked-deployable-backbone). The final completed-flow state
feeds downstream heads.

M0 Base and M0 Matched have the same factorized encoder, post-LN Transformer,
no positional signal, downstream head, fields, context length, and training
contract. The factorized numeric projection plus categorical lookup tables is
mathematically equivalent to a concatenated one-hot linear projection for valid
categories under this project's PAD/UNK/missing scheme, but is not the official
FlowTransformer implementation. The sole architectural difference is attention
masking: M0 Matched uses the causal mask; M0 Base uses unrestricted attention
and is offline-only. M0 Matched has no RoPE and no learned elapsed-time
projection.

The 25M match applies only to M0 Base and M0 Matched: their complete trainable
classifier, including factorized encoder, Transformer, and head, must be within
±5% parameters. M0 Small deliberately keeps its project-selected scale and is
not a capacity-matched comparator.

Primary scoring occurs at flow completion, from that flow and strictly earlier
completed flows. Live packet-prefix scoring is out of scope unless a suitable
point-in-time active-flow source is supplied and separately evaluated.

The service target is p95 CPU inference at most 2 ms per completed flow,
excluding dynamic-batching wait; batching may wait at most 5 ms. Report
p50/p95/p99 latency, completed flows/s, device memory, endpoint-state memory,
and missing-field behaviour.

## Optional extensions

**X1-Distill** is optional. It is considered only when usable paired packet and
flow data are supplied and a documented rule establishes an exact one-to-one
flow-to-packet-span pairing; ambiguous, unmatched, and boundary-crossing pairs
are excluded. The teacher may use packet sizes, directions, timings, and
protocol bytes; the student sees only the headline flow view and aligns
completed-flow latents. The teacher is absent at deployment. No packet data or
pair from an evaluation capture enters training.

**M5-Hier** is considered only after the trigger in
[Model](Model.md#evidence-ladder). It uses fine (1-second), medium
(10-second), and coarse (60-second) windows while preserving the same causal
history and anonymous relation inputs as its non-hierarchical control. It
uses two weight-shared window-encoder blocks at all three scales and six causal
blocks for ordered summaries and current fine-window events. All eight blocks,
the record encoder, scale tokens, and projections remain within the same 25M
±5% capacity; it adds no second model, GNN, SSM/Mamba block, memory bank, or
generative decoder.
