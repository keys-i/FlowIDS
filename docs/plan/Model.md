# Model specification

## Scope and decision boundary

This file owns model comparisons, objectives, promotion, collapse, and scaling.
The feature view and causal-context contract are in
[Architecture](Architecture.md#prediction-unit-and-feature-view) and
[Architecture](Architecture.md#context-and-relations); claims, task
tracks, metrics, and operational, versatility, and resource gates are in
[Thesis](Thesis.md); prior-art and reproducibility evidence are in
[Refs](Refs.md#prior-art-and-reproduction-boundaries).

The non-negotiable claim and prior-art boundary is stated once in
[Thesis](Thesis.md#starting-point-and-claim-boundary).

The [historical prototype](Refs.md#historicalprototype828582d) is forensic
evidence, not an eligible result. If evaluated, recreate only its documented
architecture and window choices inside the new harness and label the result
`historical-recreation`; do not restore its data, feature, label, or split
pipeline. Use the frozen evaluation contract in
[Thesis](Thesis.md#evaluation-and-leakage-contract).

A benchmark result is limited to that benchmark. Only the frozen external
transfer evaluation can support a transfer claim.

## Locked deployable backbone

All promoted neural rungs use one completed bidirectional flow per token and
the same deployable encoder unless size is the isolated variable. The M0
Matched record encoder is a factorized equivalent to one-hot record projection
under this project's PAD/UNK/missing scheme: it linearly projects transformed
numeric values and their missingness, then combines them with categorical
embeddings. The M0 Matched Transformer has d_model=512, eight post-LN blocks,
eight heads, FFN 1792, ReLU, dropout 0.1, no positional encoding, and a
last-token binary head. It consumes at most 256 events under the Architecture
context contract. Count the record encoder, embeddings, and backbone in the complete deployable total; it
must remain within 25M ±5%, then freeze that measured architecture for every
rung. SSL decoders, predictors, and EMA copies are removed at inference; their
training FLOPs and memory are reported. A bidirectional model is an offline
upper bound, never a deployable equivalent.

### Supervised M0 models

The models below use the same permitted fields, preprocessing, labels, and
splits. They are FlowTransformer-style PyTorch models under this project
contract, not exact reproductions of the original Keras framework.

- **M0 Base:** the 25M FlowTransformer-style PyTorch model with unrestricted
  attention. It is the offline comparator.
- **M0 Small:** the causal eight-flow version for cheap screening: d_model=64,
  two post-LN blocks, two heads, FFN 128, ReLU, dropout 0.1, and no positional
  encoding. It is not capacity-matched to M0 Matched.
- **M0 Matched:** the 25M causal, padding-safe model. It has exactly the same
  parameterization, architecture, capacity, record encoder, and last-token
  head as M0 Base except for the attention mask. The causal change is a
  hypothesis under test, not an established improvement.

One development-data search selects, then freezes: context horizon {1, 10,
60} minutes; learning rate {1e-4, 3e-4}; weight decay {0.01, 0.05}; and
dropout {0, 0.1}. No test-visible or rung-specific expansion is permitted.
Use AdamW, gradient clipping at 1.0, 5% linear warm-up, then cosine decay.
For each model-size stage, select the effective context batch once on the
reference training hardware and hold event exposure per update constant across
its matched rungs; mixed precision is allowed only after numerical parity with
FP32 on the validation smoke test.

### Exact S0 screening instantiation

S0 is not a second architecture: use the same token semantics, causal mask,
post-LN blocks, ReLU, no positional encoding, optimizer schedule, objectives,
and downstream protocol at d_model=256, four blocks, eight heads, FFN 1024,
and maximum 256 events. Its four blocks contain about 3.16M parameters. Cap
the tokenizer at 1.5M parameters so the complete trainable encoder is 3--5M;
report the exact count before screening. Screen at least 5M unique flows and
exactly 20M event exposures. S0 changes only capacity, so a mechanism passing
S0 must be confirmed at S1 before any claim.

## Objectives and objective controls

Use exactly the five semantic groups and target exclusions in
[Architecture](Architecture.md#context-and-relations).

For each causal sequence, select 30% of event/group units in contiguous spans
of mean length three; replace the whole selected group with its learned group
mask token; and reconstruct only selected values. L_raw uses categorical
cross-entropy, missingness BCE, and Smooth L1 on train-only transformed numeric
values. Average within a group and then across groups, so group width cannot
set the loss weight. Padding is not reconstructed; observed-value absence is
represented by the missingness target.

L_latent makes the masked student predict the same-event representation from an
unmasked, causal, stop-gradient EMA teacher. The target is the mean of the
teacher's top four layer-normalized states; the student predictor has two
layers. Teacher dropout is disabled and its momentum rises from 0.99 to 0.9999
on a cosine schedule after optimizer updates.

L_future predicts EMA-teacher latents for the next causally subsequent event
incident to either target endpoint at horizons 1, 4, and 16. Future events are
targets only, never encoder input; missing horizons are ignored. It is compared
with three controls before it may join the hybrid: shuffled-future targets
shuffled within source/day/horizon, raw next-flow-value prediction, and a
60-second endpoint aggregate forecast (log-binned flow, byte, packet,
protocol-mix, and unique-remote-count summaries). A compact aggregate forecast
is a control, not a promoted replacement mechanism; a shuffled-time gain is a
shortcut failure.

The core objective is L_hybrid = lambda_raw L_raw + lambda_latent L_latent.
Estimate each constituent's median on 200 train-only warm-up batches, set its
weight to inverse median, normalize weights to sum to two, freeze them, and
count warm-up compute. Every 100 updates log unweighted encoder-gradient norms
and pairwise objective-gradient cosines. If conflict occurs on more than 50%
of sampled updates for three epochs, test exactly one compute-matched
alternating schedule; do not conduct a loss-weight search. Generic contrastive
augmentation is excluded from the proposed model because no universal NetFlow
augmentation preserves ports, direction, timing, flags, and rare attacks. It
may appear only as a semantically audited baseline.

## Comparator contract

All comparators use the identical permitted fields and frozen splits from
[Thesis](Thesis.md#evaluation-and-leakage-contract). Required controls are:
always-benign; a robust novelty score fitted without attack labels (the mean of
the two largest absolute median/MAD z-scores over log-duration and directional
log-bytes/log-packets, stratified by protocol and port range with
protocol/global fallback); regularized logistic regression; Isolation Forest;
OC-SVM on a predeclared feasible sample; CatBoost or LightGBM target-flow
classifier; that tree with deterministic past-context aggregates; per-flow
MLP; M0 Small; M0 Base; M0 Matched; matched random initialization of M0
Matched; and frozen linear probe plus full fine-tuning for every pretrained
encoder. No candidate advances by beating only a weak neural control.

Where code and schema permit, run E-GraphSAGE, Anomal-E, NEGSC, GraphIDS, and
the verified Van Langendonck graph-foundation model. The requested GNNet name
is unresolved and must not be silently mapped to another paper. The CMES-style
bidirectional relation-bias model is an explicit offline upper bound. NetFlowGen
and unreproducible models remain literature comparisons. YaTC and exact MMAE
are packet-view upper bounds only on a paired subset; they are not NetFlow
competitors. The YaTC packet teacher is eligible for X1-Distill only after the paired
data and refreshed-novelty gates in
[Architecture](Architecture.md#optional-extensions).

Matched mechanism controls are deliberate and named. MMAE-NF is a 25M NetFlow
adaptation, not a reproduction: map each semantic group to a patch unit and use
the released-code FlowMix pairing, dynamic masking, and
teacher/reconstruction/alignment logic without endpoint context; record any
operation that cannot map exactly and mark the control UNVERIFIED rather than
silently approximating it. CMES-25M has matched data, fields, backbone, and
heads with CMES context/predicates; CMES-Causal uses ego context with CMES
predicates. They never enter production by default. Reproduction status and
claimed overlap are governed by
[Refs](Refs.md#prior-art-and-reproduction-boundaries).

## Evidence ladder

| Rung | Sole promoted difference | Required evidence | Immediate rejection |
|---|---|---|---|
| Exploration | No model | Complete: the existing DuckDB queries inspected schema, distributions, usable features, duplicates, time shift, and leakage risks. See the [findings](../exp/explore.md). | Exploration is not promotion evidence. Carry its exclusions and data-handling decisions into M0 Matched. |
| M0 Matched | 25M backbone trained only on labels | Before training, [build the actual splits, train-only preprocessing, and causal-context checks under `src`](Architecture.md#m0-implementation); then establish the reference versus every applicable classical control. | If it fails the [classical-parity gate](#classical-parity-discriminator), permit only that restricted S0 discriminator; do not scale. |
| M1-R | L_raw pretraining on flat causal contexts | Pass the common frozen promotion gate below against random-init M0 Matched. | Reject if gain needs identifiers or ports, vanishes chronologically, or is in-domain only. |
| M1-L | L_latent instead of raw; parallel constituent | Pass collapse screen and beat M0 Matched in the separate exposure-matched and FLOP-matched comparisons. | Reject if raw matches it or teacher/student collapse persists. |
| M2-H | Add the other constituent: L_raw + L_latent | Beat M1-R and M1-L, not merely M0 Matched, under exposure and FLOP matching. | Reject if either constituent or MMAE-NF matches it. |
| M2-F | Conditionally add L_future | Beat shuffled-future, raw-next, and 60-second aggregate-forecast controls on external transfer. | Reject for persistent gradient conflict, source/day identity, shortcut under time shuffle, or no external gain. |
| M3-Ego | Replace flat history with causal endpoint-ego history | Positive context main effect and positive hybrid×ego interaction. | Reject if random, time/feature-matched, or CMES grouping matches it. |
| M4-Rel | Add directed identity-free endpoint relation bias | Beat no-bias and CMES-Causal; pass relation destruction and endpoint-renaming tests. | Reject if identity is required, relation is unused, or endpoint-held-out performance falls. |
| M5-Hier | Conditional multi-resolution summaries | Run only if ≥20% valid contexts truncate at 256 and long-horizon recall is deficient; beat non-hierarchical M4-Rel on identical ego history within Thesis resource gates. | Reject if non-hierarchical M4-Rel matches it or runtime cost fails its gate. |
| X1-Distill | Conditional frozen YaTC/packet teacher to flow student | Leakage-safe paired data, refreshed novelty search, and flow-only external gain over same-modality controls. | Reject if pairing/privacy fails, packet alignment is shuffled-equivalent, or gain is <1 absolute AUPRC point. |
| Final | No mechanism | Five pretraining-seed replication on untouched domains/tasks. | Any primary failure narrows the claim. |

### Classical-parity discriminator

M0 Matched clears the classical gate only when all three conditions hold at `k=10`
labelled support groups per class against the best applicable classical model:
at least 1.0 absolute macro one-vs-rest AUPRC point improvement; paired
hierarchical-bootstrap 95% lower bound above zero; and TPR at 10 false alerts
per million non-inferior within one absolute point. Any other result is
classical parity, including a classical win.

Under parity, run only three flat-context S0 arms—scratch, M1-R, and M1-L—at
20M event exposures, plus their separate fixed-FLOP comparison. Use the frozen
screening seeds and adaptation repeats from Thesis. Do not run hybrid, future,
ego, relation, hierarchy, or scaling yet. Continue only if M1-R or M1-L beats
both S0 scratch and the best classical model by the same AUPRC, confidence, and
operational gates on at least two development networks. Otherwise stop
all Transformer and SSL progression. If one constituent passes, complete the
remaining standard S0 screen below before entering the 25M ladder.

M3-Ego uses the context and matched-control definitions in
[Architecture](Architecture.md#context-and-relations). Its controls are
flat collector history, same-size random earlier history,
time-and-feature-matched history without a shared endpoint, target-only, CMES
grouping. M4-Rel uses Architecture's anonymous relation tensor and causal-logit
bias; it does not introduce another relation taxonomy.

Before full progression, run the S0 screen over the fixed 20M exposures:
{scratch, raw, latent, hybrid} × {flat, ego}, plus the named paper controls;
test relation bias separately. Estimate
(EgoHybrid − EgoScratch) − (FlatHybrid − FlatScratch). The proposed combined
mechanism survives only if its paired block-bootstrap 95% lower bound is above
zero on the predeclared low-label cross-domain aggregate. Non-canonical
alternatives and their comparison rules are fixed in
[Thesis](Thesis.md#excluded-or-conditional-alternatives).

## Promotion, collapse, and scaling

A rung promotes only if all conditions hold: at least 1.0 absolute macro
one-vs-rest AUPRC point over the preceding promoted model at 10 labelled
support groups per supported class and on area under the frozen
label-efficiency grid in
[Thesis](Thesis.md#evaluation-and-leakage-contract); paired hierarchical
block-bootstrap 95% lower bound above zero; improvement on at least two
independent targets; no target loses more
than one macro-AUPRC point; no critical family loses more than two recall
points; and it beats the best applicable same-input comparator. Fixed-data-
exposure and fixed-total-FLOP comparisons are separate mandatory runs. Use the
frozen screening/final seed and adaptation-repeat policy in
[Thesis](Thesis.md#evaluation-and-leakage-contract). Lower SSL loss is never
promotion evidence. Apply the precommitted operational,
versatility, and resource non-inferiority gates without redefining them here:
[Thesis evaluation](Thesis.md#evaluation-and-leakage-contract).
Compute label-efficiency area by trapezoidal integration against `log10(k)`,
normalized by the grid's log-span; do not average the five points equally.

During pretraining, sample source `d` with `p(d) ∝ n_d^alpha`. Choose `alpha`
once at S0 from `{0, 0.5, 1}`, cap each source at 20%, redistribute excess
mass, then freeze the choice and sampling sequence.

At every validation checkpoint record embedding standard-deviation quantiles,
covariance eigenvalues, active/effective rank, off-diagonal correlation,
mean/nearest-neighbour cosine, alignment/uniformity, encoder and head gradient
norms, and source/day/entity distributions. Reject a checkpoint if any persists
for three validations: median embedding standard deviation <1e-3; active or
effective rank <10% of width; more than half of dimensions with standard
deviation <1e-3; rank falls at least 50% while SSL loss improves; or a
constant-vector baseline is indistinguishable at every label budget. These are
preregistered engineering thresholds, not literature constants.

| Stage | Model | Minimum unique flows / independent sources | Event exposures | Promotion gate |
|---|---:|---:|---:|---|
| S0 | 3--5M | 5M / — | 20M | mechanism screen only |
| S1 | 25M | 50M / 10 | 100M | external transfer |
| S2 | 25M | 250M / 20 | 500M | scale confirmation |
| S3 | 90M | 1B / 30 | after S2 | only if S2 passes |
| S4 | 300M | — | — | only if 90M beats 25M by ≥2% relative external-transfer AUPRC |

Do not substitute repeated epochs or larger models for source diversity. Scale
data before width. Stop after two consecutive doubled-exposure runs lower SSL
loss without positive transfer; retain the smallest promoted model.
