# Architecture

I define inputs, causal context, and inference. See [Model](Model.md) for
objectives, [Thesis](Thesis.md) for evaluation, and [Refs](Refs.md) for prior
work.

## Limits

Supplied datasets are benchmarks. Results do not prove live-network
performance, broad transfer, or a foundation model.

## Exploration

I follow the [NF3 exploration](../exp/explore.md) findings on schema,
missingness, timestamps, duplicates, and features.

## M0

I start with **M0 Base**, the only FlowTransformer-style baseline: an offline,
unrestricted 25M model with 255 earlier flows and its target. **M0 Matched** is
the same model with a causal mask. **M0 Small** is an eight-token causal smoke
check.

I fit preprocessing on training only and reset causal state at partition
boundaries. Code is not experimental evidence.

## Prediction unit and input features

One completed bidirectional flow is one token. I exclude sources without a
documented bidirectional pairing rule. Context stays inside its capture or
exporter boundary.

During SSL I mask five semantic groups: transport and flags; service and ports;
directional volume; packet-size and retransmission statistics; and duration and
inter-arrival statistics. I use only completion-time fields with compatible
meanings.

I exclude addresses, flow IDs, labels, absolute timestamps, capture
or scenario IDs, hostnames, collector or template IDs, and post-hoc metadata.
Routing keys can choose records but never enter tensors.

Ports 0 to 1023 stay exact. Higher training ports use eight training-frequency
buckets plus `REGISTERED` (1024 to 49151) or `DYNAMIC` (49152 to 65535); unseen
higher ports use `UNK` plus a range. `PAD` and `MISSING` differ. I require a
port-free result. `L7_PROTO` stays out unless a compatible secondary
sensitivity check proves it is not a label proxy.

I fit numeric transforms, vocabularies, and port buckets on training only.
Numeric fields use median imputation with a missingness bit, `log1p` for heavy
tails, 1st/99th-percentile clipping, and z-scores. Unavailable elapsed times are
missing, not zero. Batches retain elapsed time since the previous flow and
endpoint-sharing flow; M0 ignores both.

## Context and endpoint relationships

I order completed flows deterministically and select at most 255 earlier
qualifying events before the target. I test 1-, 10-, or 60-minute horizons and
reset state at every partition boundary. Future events, padding, labels, raw
identities, and post-hoc fields never affect an encoder input.

M3-Ego takes up to 128 earlier events incident to each target endpoint, unions
and deduplicates them, then keeps the latest 255. Same-size controls use the
latest collector events (`flat-matched`), deterministic non-incident events
(`random-matched`), or non-incident events ranked by protocol mismatch,
port-range mismatch count, completion-lag difference, then transformed
duration, byte, and packet L1 distance
(`time-feature-matched`).

`random-matched` keeps the lowest
`SHA-256(run_seed || target_event_id || candidate_event_id)`. `target-only` has
no history. `conversation` uses earlier flows between the same unordered
endpoint pair and keeps direction fields. `source-host` uses earlier flows sent
by the target source to any destination; I use it only where source/initiator
direction has the same meaning across corpora.

Too few non-incident candidates makes a target unsupported. Any control
equalling or beating ego rejects the claim.

For M2-F, I order candidates by completion time and event ID. I keep later
flows in the same corpus, stream, and pretraining partition that touch either
anchor endpoint. Deduplicated positions 1, 4, and 16 are targets. Each teacher
sees its target and causal 255-event prefix. The student sees neither. I omit
missing positions.

For endpoint-disjoint evaluation, I assign held-out principals before building
context: held-out to held-out flows are test, mixed flows are purged, and the
rest are training. Test context uses only earlier completed test flows. Offline
replay and streaming must preserve ordering, padding, masks, relation types,
and state resets. State also clears at adaptation, calibration, and test
boundaries.

I use 16 anonymous directed relation types. Four equality bits between the
current and earlier flow endpoints choose the type: `src-src`, `dst-dst`,
`src-earlier-dst`, and `dst-earlier-src`. Each attention head adds its learned
type scalar to the causal logit.
Raw identities and feature, port, or protocol similarity are excluded.
Renaming routing keys must not change output.

I also permute relation types within source, day, and relative-time strata. A
surviving gain fails the relation claim.

## Encoder and inference

During SSL I feed the encoder only completed source flows and causal contexts.
[Model](Model.md#training-objectives-and-controls) defines the temporary heads.
The deployed encoder is a factorized record encoder plus a post-LN Transformer
with no positional signal. Its completed target state feeds the downstream
head. I remove SSL heads and teachers before inference.

M0 Base and M0 Matched use the same fields, factorized encoder, post-LN
Transformer, head, context length, and training setup. The sole architectural
difference is attention masking. The complete classifier must be within 5
percent of 25M trainable parameters. M0 Small is deliberately not matched.

I score at flow completion from that flow and strictly earlier completed flows.
Packet-prefix scoring is out of scope. The target is p95 CPU inference at or
below 2 ms per completed flow, excluding a dynamic-batching wait of at most 5
ms. I report p50, p95, p99, throughput, device memory, endpoint-state memory,
and missing-field behavior.

## Optional work

I run **X1-Distill** only with safe, exact packet-to-flow pairs. The deployed
student reads only the main flow view.

I consider **M5-Hier** only after [Model's trigger](Model.md#experiment-sequence).
It uses 1-, 10-, and 60-second causal windows with the same anonymous
relations.

Two weight-shared window encoders and six causal summary blocks,
plus the record encoder and projections, stay within the 25M plus or minus 5
percent capacity. I do not add a second model, GNN, SSM, memory bank, or
generative decoder.
