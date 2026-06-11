# Inputs and context

This page defines what the [models](Model.md) can see. Each completed flow
becomes a vector; the Transformer reads it with selected past flows and
produces an attack score.

## Prediction unit and input features

One token represents one completed bidirectional flow: both directions of
an exchange. Exclude sources without a documented pairing rule. Keep
history within its capture/exporter and [data partition](Thesis.md#evaluation-and-leakage-checks).
Current code groups by dataset and partition; finer exporter boundaries need
checking before adding multi-exporter data.

The five groups used for masking are transport/flags, service/ports,
directional volume, packet-size/retransmission statistics, and
duration/inter-arrival statistics. Use completion-time fields with
compatible meanings.

M1's exact field assignment is in `src/m1/masking.py`. TTL and TCP-window
fields join transport; port IDs and ranges share the service mask. Numeric
values and their missing flags are hidden together. Three-event spans are
selected with probability 0.3, with a random offset per group. Padding is
excluded; an otherwise unmasked example hides one group on its last event.
M1 reconstruct excludes imputed numeric targets, averages each loss type within a group,
then averages selected groups. M1 teacher scores only events with a masked group.

Exclude addresses, flow IDs, labels, absolute timestamps, capture/scenario
IDs, hostnames, collector/template IDs, and post-hoc metadata. Addresses may
select history but their values never enter model inputs. Test renamed and
unseen endpoints because relationships can still reveal dataset shortcuts.

<details>
<summary>Field encoding</summary>

Fit transforms and vocabularies on training data only.

- Keep ports 0–1023 exact. Other training ports use eight frequency buckets
  plus `REGISTERED` (1024–49151) or `DYNAMIC` (49152–65535); unseen ports use
  `UNK` plus a range. `PAD` and `MISSING` differ. Require a port-free result.
  Exclude `L7_PROTO` unless a compatible secondary test shows it is not a
  label proxy.
- Numeric fields use median imputation, a missingness bit, `log1p` for heavy
  tails, 1st/99th-percentile clipping, and z-scores.
- M0 and M1 do not use gaps between flows. Duration and within-flow
  inter-arrival fields describe the flow itself. Later models may add gaps
  since the previous flow or endpoint-sharing flow, retaining missingness.

</details>

## Context and endpoint relationships

Flat history uses recent flows; endpoint history uses flows involving either
host in the target. See [the example](Model.md#which-past-flows-should-the-model-see).

Order completed flows deterministically. M0 uses a ten-minute window and
breaks completion-time ties by source-local row index. The 256-event models
keep the latest 255 qualifying earlier flows, then the target. Future events
and labels cannot choose or populate history. Ignore padding in attention and
losses; reset history at every partition boundary.

<details>
<summary>History selection and relation types</summary>

M3-Ego takes up to 128 earlier events per endpoint, unions and deduplicates
them, then keeps the latest 255. Same-size controls use:

- `flat-matched`: latest collector events;
- `random-matched`: non-endpoint events with the lowest
  `SHA-256(run_seed || target_event_id || candidate_event_id)`;
- `time-feature-matched`: non-endpoint events ranked by protocol mismatch,
  port-range mismatch count, completion-lag difference, then transformed
  duration, byte, and packet L1 distance.

Exclude targets with too few non-endpoint candidates from matched tests.
Endpoint history must beat every control. Also compare `target-only` (no
history), `conversation` (same unordered endpoint pair, keeping direction),
and `source-host` (target source to any destination). Use the latter only
where source/initiator direction means the same thing across datasets.

For endpoint-disjoint tests, assign held-out hosts before building history:
both endpoints held out means test; mixed pairs are purged; the rest are
training. Test history uses earlier completed test flows. Offline replay and
streaming must agree on order, padding, masks, relations, and state resets,
including adaptation/calibration/test boundaries.

M4-Rel uses four endpoint-equality bits: `src-src`, `dst-dst`,
`src-earlier-dst`, and `dst-earlier-src`. These select one of 16 anonymous
directed types. Each attention head adds a learned type scalar to its causal
score. Exclude address values and feature/port/protocol similarity. Renaming
routing keys must leave output unchanged.

Shuffle types within source, day, and relative-time groups. A gain that
survives shuffling is not explained by the true relations.

### Future targets

M2 future-hybrid and future-jepa target deduplicated later endpoint-related flows at positions 1, 4,
and 16, ordered by completion time then event ID, within the same corpus,
stream, and pretraining partition. Targets must complete strictly after the
anchor; equal completion times are excluded. The teacher has dropout off and sees each
target with up to 255 earlier events within the configured history age limit. The student sees only masked history
through the anchor; later records cannot enter its input. Omit missing
positions. [Model](Model.md#training-objectives-and-controls) defines the loss.

</details>

## Encoder and inference

After pretraining, keep the record encoder, Transformer, and attack head;
remove prediction heads and teachers. [Model](Model.md#m0-supervised-models)
owns the shapes and attention masks.

Score at flow completion. The target is p95 CPU inference ≤2 ms per flow,
excluding a batching wait ≤5 ms. Report p50/p95/p99 latency, throughput,
device memory, endpoint-state memory, and missing-field behaviour.
Packet-prefix/live scoring needs active snapshots and is outside this plan.

<details>
<summary>Conditional model inputs</summary>

X1's deployed student reads only the main flow fields. M5 uses 1-, 10-, and
60-second causal windows with the same anonymous relations: two weight-shared
window encoders, six causal summary blocks, record encoder, and projections,
all within 25M ±5% parameters. Neither adds a second model, GNN, SSM, memory
bank, or generative decoder. [Model](Model.md#experiment-sequence) owns the
conditions for trying them.

</details>
