# Architecture

This file describes the data seen by the models and how inference should work.
It is not a public API. [Model](Model.md) covers objectives, model sizes, and
the checks each model must pass. [Thesis](Thesis.md) covers datasets,
evaluation, and the claims the results could support. [Refs](Refs.md) covers
earlier work.

## Limits

The supplied datasets are benchmark inputs. Results based only on these
datasets are benchmark results; they do not establish live-network performance,
cross-network transfer, or a foundation model.

## Exploration

Exploration is complete. The existing DuckDB queries covered schema,
missingness, timestamps, labels, duplicates, and feature choices. M0 must
follow the [recorded findings](../exp/explore.md).

## M0

Code under `src` builds the evaluation splits, fits preprocessing on training
data only, checks causal context and state, and saves the checkpoint details
used by the implementation.

**M0 Small** is causal with seven earlier completed flows plus its target, for
eight tokens. **M0 Base** is the unrestricted 25M offline model with up to 255
earlier completed flows plus its target, for 256 tokens. **M0 Matched** is the
causal 25M/256-token model. It has the same parameters as M0 Base except for
the attention mask. Any later reference to M0 means M0 Matched.

## Prediction unit and input features

The prediction unit is one completed bidirectional flow. A unidirectional
source may be used only when a documented pairing rule produces a completed
bidirectional record; otherwise it is excluded. Source records and packet
captures do not cross their capture/exporter observation boundary while context
is built.

The main feature set contains fields that are available when a flow completes
and mean the same thing across the primary benchmarks: protocol, service-aware
ports, packet and byte counts by direction, duration, TCP flags, and compatible
packet-size, retransmission, and inter-arrival summaries. This set stays fixed
across the primary datasets and tasks. Pair-specific feature sets are secondary
sensitivity checks.

A secondary feature set may add compatible source-specific fields with an
explicit field-presence mask. Missing is distinct from zero. The main feature
set excludes IP/MAC addresses, flow IDs, labels, absolute timestamps,
capture/day/scenario IDs, hostnames, collector/template IDs, and post-hoc
metadata. Raw endpoint
values and stable endpoint embeddings never enter model tensors.

Ports use exact tokens for 0--1023. Higher training ports use one of eight
training-frequency buckets plus a `REGISTERED` (1024--49151) or `DYNAMIC`
(49152--65535) range token; unseen higher ports use `UNK` plus their range.
`PAD` and `MISSING` remain separate. A port-free result is required.
`L7_PROTO` stays out of the main feature set. The secondary set may use it only
when its meaning is compatible across sources and it is not a label proxy.

M0 fits numeric processing on training data only: median imputation with a
missingness indicator, `log1p` for non-negative heavy-tailed fields, 1st/99th
percentile clipping, then z-score scaling. Categorical vocabularies and port
buckets come from training data only. The two time fields are clipped `log1p`
elapsed time since the previous observed flow and since the previous flow that
shares either endpoint. Wall-clock time and absolute capture time are never
model inputs. The elapsed values stay in the batch for a later ablation; M0
does not use them.

## Context and endpoint relationships

M0 Base and M0 Matched see at most 255 earlier completed flows followed by the
target flow. Choose a 1-, 10-, or 60-minute horizon, and break equal completion
times deterministically. Reset state at every M0 partition boundary. Future
data, padding, labels, raw identifiers, and post-hoc fields must not affect the
encoder input.

M0--M2-F use the latest 255 qualifying collector events. M3-Ego takes the
latest 128 qualifying events incident to the target source and the latest 128 incident
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

M2-F builds future targets separately from the student's context. Here,
`corpus` means one dataset or capture stream, and `source endpoint` means the
flow's source-address routing key. For each anchor, take only flows that come
strictly later in the deterministic completion-time and event-ID order, belong
to the same corpus, capture or exporter stream, and pretraining partition, and
touch either anchor endpoint. After deduplication, only candidates 1, 4, and 16
become targets. The EMA teacher sees the same frozen flat 255-event context as
M0--M2-F, limited to records available when that target completed and ending
with the target flow. The student never sees target flows or metadata derived
from the future.

Missing target positions are omitted, not padded. Every point-horizon arm uses
the same anchors and horizon masks, and candidate state resets wherever student
state resets. The separate 60-second aggregate diagnostic uses the same anchors
but no event-horizon mask. [Model](Model.md) defines the objective, controls,
and failure checks.

M3-Ego remains the proposed context method. Compare it with three routing-only
controls that use the same history size and horizon:

- `flat` uses earlier collector flows.
- `conversation` uses earlier flows between the same unordered endpoint pair
  while keeping each flow's direction fields.
- `source-host` uses earlier flows sent by the target source endpoint to any
  destination. Use it only where source or initiator direction means the same
  thing across the relevant corpora.

Raw endpoint values may choose records but never enter model tensors. Neither
`conversation` nor `source-host` can support the endpoint-ego claim. If either
matches or beats endpoint-ego, reject that claim. Test one builder at a time;
do not combine them or give any control a second encoder.

The optional context-length check uses total-token limits `W={3,8,20}`, target
included, and left-pads every arm to 20 tokens. Keep target anchors and
optimizer updates fixed so history length is the only change. This is not a
random-window policy and does not replace the default context.

Time-gap sessionisation is only a diagnostic. Inspect training completion-gap
distributions separately for conversation, source-host, and endpoint-ego
streams. `FLOW_GROUP` is never an input, label, or embedding. Choose at most
one threshold from `{1,10,60}` seconds on development validation data. Use it
only if the same threshold beats an equal-size unsessionised history on at
least two development corpora, then freeze it before external evaluation.

For endpoint-disjoint evaluation, assign held-out endpoint principals before
context construction: flows with two held endpoints are test; flows joining a
held and unheld endpoint are purged; the rest are training. Test context uses
only its own earlier completed flows.

Offline replay and streaming must produce the same event order, padding,
causal masks, relationship types, and elapsed-time batch fields. State expires
at the chosen horizon and clears at adaptation, calibration, and test
boundaries.

Semantic group masking is distinct from missingness and padding. The groups
are transport/flags, service/ports, directional volume, packet-size and
retransmission statistics, and duration/inter-arrival statistics.

For each causally visible ordered pair, four endpoint-equality bits
`src-src`, `dst-dst`, `src-earlier-dst`, and `dst-earlier-src` select one of 16
relation types. They are anonymous equality relations only: raw identities,
port/protocol similarity, and feature similarity are not relation inputs.
Each layer and head adds a learned scalar for its relation type to the causal
attention logit. Renaming endpoint routing keys must not change outputs. The
destruction control permutes relation types only within source/day/relative-time
strata.

## Encoder and inference

One completed flow is one token. The record encoder, Transformer shape,
parameter accounting, objectives, and removal of SSL heads are defined once in
[Model](Model.md#shared-deployable-model). The final completed-flow state
feeds downstream heads.

M0 Base and M0 Matched have the same factorized encoder, post-LN Transformer,
no positional signal, downstream head, fields, context length, and training
setup. The factorized numeric projection plus categorical lookup tables is
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

The target is p95 CPU inference at or below 2 ms per completed flow, excluding
dynamic-batching wait. A batch may wait at most 5 ms. Report
p50/p95/p99 latency, completed flows/s, device memory, endpoint-state memory,
and missing-field behaviour.

## Optional work

Consider **X1-Distill** only when usable paired packet and flow data exist and a
documented rule gives an exact one-to-one flow-to-packet-span match. Exclude
ambiguous, unmatched, and boundary-crossing pairs. The teacher may use packet
sizes, directions, timings, and protocol bytes. The student sees only the main
flow features and aligns completed-flow latents. Deployment uses no teacher,
packet data, or pair drawn from an evaluation capture.

**M5-Hier** is considered only after the trigger in
[Model](Model.md#experiment-sequence). It uses fine (1-second), medium
(10-second), and coarse (60-second) windows while preserving the same causal
history and anonymous relation inputs as its non-hierarchical control. It
uses two weight-shared window-encoder blocks at all three scales and six causal
blocks for ordered summaries and current fine-window events. All eight blocks,
the record encoder, scale tokens, and projections remain within the same 25M
±5% capacity; it adds no second model, GNN, SSM/Mamba block, memory bank, or
generative decoder.
