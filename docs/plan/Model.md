# Models and experiments

## What this file covers

This file defines the models, training objectives, comparisons, failure checks,
and scaling steps.
[Architecture](Architecture.md#prediction-unit-and-input-features) defines the
input features, and
[Architecture](Architecture.md#context-and-endpoint-relationships) defines the
causal context. [Thesis](Thesis.md) defines the claims, tasks, metrics, and
resource limits. [Refs](Refs.md#closest-work-and-reproduction-status) records
the earlier work and what can be reproduced.

The intended claim and its limits appear once in
[Thesis](Thesis.md#starting-point-and-intended-claim).

The [historical prototype](Refs.md#historicalprototype828582d) is useful only
as a record of earlier work. It cannot count as a result. If it is evaluated,
recreate only its documented architecture and window choices in the new
harness, label it `historical-recreation`, and do not restore its data,
features, labels, or split pipeline. Use the fixed evaluation rules in
[Thesis](Thesis.md#evaluation-and-leakage-checks).

A benchmark result is limited to that benchmark. Only an external transfer
evaluation fixed in advance can support a transfer claim.

## Shared deployable model

Every neural model that passes to the next stage uses one completed
bidirectional flow per token and the same deployable encoder unless model size
is the variable being tested. The M0 Matched record encoder is a factorized
equivalent of a one-hot record projection
under this project's PAD/UNK/missing scheme: it linearly projects transformed
numeric values and their missingness, then combines them with categorical
embeddings. The M0 Matched Transformer has d_model=512, eight post-LN blocks,
eight heads, FFN 1792, ReLU, dropout 0.1, no positional encoding, and a
last-token binary head. It consumes at most 256 events under the Architecture
rules. Count the record encoder, embeddings, backbone, and head in the total
deployable model. The total must stay within 25M ±5%. Once measured, keep that
architecture fixed across later stages. Remove SSL decoders, predictors, and
EMA copies for inference, but report their training FLOPs and memory. A
bidirectional model is only an offline upper bound.

### M0 supervised models

These models use the same fields, preprocessing, labels, and splits. They are
FlowTransformer-style PyTorch models built for this project, not exact copies
of the original Keras framework.

- **M0 Base:** the 25M FlowTransformer-style PyTorch model with unrestricted
  attention. It is the offline comparator.
- **M0 Small:** the causal eight-flow version for cheap screening: d_model=64,
  two post-LN blocks, two heads, FFN 128, ReLU, dropout 0.1, and no positional
  encoding. It is not capacity-matched to M0 Matched.
- **M0 Matched:** the 25M causal, padding-safe model. It has exactly the same
  parameters, architecture, capacity, record encoder, and last-token head as
  M0 Base except for the attention mask. Whether causality helps is a question
  for the experiment, not an assumed improvement.

One development-data search chooses and then fixes the context horizon
`{1, 10, 60}` minutes, learning rate `{1e-4, 3e-4}`, weight decay
`{0.01, 0.05}`, and dropout `{0, 0.1}`. Do not expand the search for a later
stage or after seeing test data.
Use AdamW, gradient clipping at 1.0, 5% linear warm-up, then cosine decay.
For each model size, choose the effective context batch once on the reference
hardware and keep event exposure per update fixed across matched models. Use
mixed precision only after it matches FP32 on the validation smoke test.

### S0 screening model

S0 is the same architecture at a smaller size. Keep the token meaning, causal
mask, post-LN blocks, ReLU, no positional encoding, optimizer schedule,
objectives, and downstream protocol at d_model=256, four blocks, eight heads, FFN 1024,
and maximum 256 events. Its four blocks contain about 3.16M parameters. Cap
the tokenizer at 1.5M parameters so the complete trainable encoder is 3--5M;
report the exact count before screening. Screen at least 5M unique flows and
exactly 20M event exposures. S0 changes only capacity, so a mechanism passing
S0 must be confirmed at S1 before any claim.

## Training objectives and controls

Use exactly the five semantic groups and target exclusions in
[Architecture](Architecture.md#context-and-endpoint-relationships).

For each causal sequence, select 30% of event/group units in contiguous spans
with mean length three. Replace each selected group with its learned mask token
and reconstruct only the selected values. $L_{\mathrm{raw}}$ uses categorical
cross-entropy, missingness BCE, and Smooth L1 on train-only transformed numeric
values. Average within a group and then across groups, so group width cannot
set the loss weight. Padding is not reconstructed; observed-value absence is
represented by the missingness target.

$L_{\mathrm{latent}}$ makes the masked student predict the same-event
representation from an unmasked, causal, stop-gradient EMA teacher. The target
is the mean of the teacher's top four layer-normalized states; the student predictor has two
layers. Disable teacher dropout. After each optimizer update, raise teacher
momentum from 0.99 to 0.9999 on a cosine schedule. This is a data2vec-style
same-event comparison, not future prediction.

Use $L_{\mathrm{future}}$ only as a conditional JEPA-family comparison in M2-F.
It is not a separate architecture or part of the claimed contribution. LeNEPA
is the closest earlier method because it also predicts a future latent state,
although this project keeps an EMA teacher. For anchor flow $i$, let $z_i$ be
the masked student's final anchor state and let $H_i$ contain the available
horizons from $\{1,4,16\}$:

$$
L_{\mathrm{future}}(i)=\frac{1}{|H_i|}\sum_{h\in H_i}
\left\|\operatorname{norm}\!\left(p(z_i+e_h)\right)-
\operatorname{sg}\!\left(\operatorname{norm}(\bar z_{i,h})\right)\right\|_2^2.
$$

Here, $\operatorname{norm}(u)=u/(\lVert u\rVert_2+10^{-6})$. $e_h$ is a
learned horizon embedding with the encoder width, $p$ is one shared two-layer
predictor, and $\bar z_{i,h}$ is the mean of the same top four normalized EMA
teacher layers used by $L_{\mathrm{latent}}$, taken from the $h$-th later flow
that touches either anchor endpoint. The teacher sees that target flow and only
the causal prefix available when it completed, with dropout disabled. The
student never sees later flows or their metadata.

Order target candidates by completion time and then event ID within the same
corpus, capture or exporter stream, and pretraining partition. Skip missing
horizons and anchors with no available horizon. Average the available horizons
for each anchor before averaging the batch so busy endpoints do not dominate.
Remove the predictor and EMA teacher at inference.

M2-F may add $L_{\mathrm{future}}$ only after M2-H passes:

$$
L_{\mathrm{M2-F}}=
\lambda_{\mathrm{raw}}L_{\mathrm{raw}}+
\lambda_{\mathrm{latent}}L_{\mathrm{latent}}+
\lambda_{\mathrm{future}}L_{\mathrm{future}}.
$$

Before warm-up, fix the eligible anchor/horizon sequence and all activity
strata. Use 200 batches from that sequence to set each weight to the inverse
median loss, normalize the three weights to sum to two, and then keep them
fixed. Retrain the eligible-anchor M2-H control with its own 200-batch,
two-loss warm-up under the same sum-to-two rule. Count warm-up and training
events for every arm. Each point-horizon control changes only the third target
and head. Keep the weighting rule, anchors, horizon masks, optimizer updates,
trainable-parameter budget, and the separate fixed-exposure and fixed-FLOP
comparisons the same. Do not search for better weights.

Compare M2-F with both the M2-H model that passed the earlier checks and M2-H
retrained on exactly the future-eligible anchors. Every future arm uses the
same anchors and horizon-availability masks. The point-horizon controls are:

- A shuffled latent target, stratified by corpus, capture or exporter,
  partition, day, horizon, relative-time bucket, candidate-count bucket, and
  training-only activity and degree buckets from the causal pre-anchor history
  of both endpoints.
- Prediction of the allowed raw semantic fields at horizons 1, 4, and 16, with
  the same shared horizon conditioning and group-balanced categorical,
  missingness, and numeric losses.

Fit every stratum boundary on the pretraining partition and then keep it fixed.
Choose one shuffle seed per run before training and use one fixed within-stratum
derangement of target event IDs. Do not reshuffle by batch or epoch. If a
stratum cannot be deranged, remove its anchor/horizon pairs from every
point-horizon arm before fixing the common sequence.

Run two more diagnostics: a training-only mean teacher target for each corpus
and horizon, and a 60-second raw endpoint aggregate forecast with log-binned
flow, byte, packet, protocol-mix, and unique-remote-count summaries. The
aggregate is not matched to an event horizon. Train it on the same anchors as a
separate shortcut check, use the same inverse-median weighting for its third
loss, and match its head size, updates, and total FLOPs within 5%. Where target
widths differ, match point-horizon predictor parameters and total FLOPs within
5%.

The true future target must beat the stratified shuffle on both predictive
validation loss and external low-label transfer. Report eligibility and results by
corpus, day, activity and degree of both endpoints, horizon, and target
elapsed-time bucket. At each horizon, report teacher and predictor rank,
variance, and nearest-neighbour cosine. A lower SSL loss on its own proves
nothing.

The core objective is
$L_{\mathrm{hybrid}}=\lambda_{\mathrm{raw}}L_{\mathrm{raw}}+
\lambda_{\mathrm{latent}}L_{\mathrm{latent}}$. Estimate each loss's median on
200 training-only warm-up batches, set its weight to the inverse median,
normalize the weights to sum to two, and then keep them fixed. Count the warm-up
compute. Every 100 updates, log unweighted encoder-gradient norms and pairwise
objective-gradient cosines. If more than 50% of sampled updates conflict for
three epochs, test one compute-matched alternating schedule. Do not search over
loss weights. Generic contrastive augmentation is not part of the proposed
model because no universal NetFlow augmentation preserves ports, direction,
timing, flags, and rare attacks. It may appear only as a baseline after its
semantics have been checked.

## Comparisons

Use the same allowed fields and fixed splits from
[Thesis](Thesis.md#evaluation-and-leakage-checks) for every comparison. Include:

- an always-benign baseline;
- a robust novelty score fitted without attack labels: the mean of the two
  largest absolute median/MAD z-scores over log-duration and directional
  log-bytes/log-packets, stratified by protocol and port range with
  protocol/global fallback;
- regularized logistic regression, Isolation Forest, and OC-SVM on a sample
  whose feasible size is chosen in advance;
- a CatBoost or LightGBM target-flow classifier, plus the same tree with
  deterministic past-context aggregates;
- a per-flow MLP, M0 Small, M0 Base, and M0 Matched;
- M0 Matched with random initialization under the same setup; and
- a frozen linear probe and full fine-tuning for every pretrained encoder.

Beating only a weak neural baseline is not enough to continue.

Where code and schema permit, run E-GraphSAGE, Anomal-E, NEGSC, GraphIDS, and
the verified Van Langendonck graph-foundation model. The requested GNNet name
is unresolved; do not substitute a different paper. The CMES-style
bidirectional relation-bias model is an explicit offline upper bound. NetFlowGen
and unreproducible models remain literature comparisons. YaTC and exact MMAE
are packet-view upper bounds only on a paired subset; they are not NetFlow
competitors. Use the YaTC packet teacher for X1-Distill only after the paired
data and updated literature checks in
[Architecture](Architecture.md#optional-work) pass.

Keep the matched controls separate and name them clearly. MMAE-NF is a 25M
NetFlow adaptation, not a reproduction. Map each semantic group to one patch
unit and use the released FlowMix pairing, dynamic masking, and teacher,
reconstruction, and alignment logic without endpoint context. If an operation
cannot be mapped exactly, mark the control `UNVERIFIED` instead of inventing an
approximation. CMES-25M matches the data, fields, backbone, and heads while
using CMES context and predicates. CMES-Causal uses ego context with CMES
predicates. Neither is a deployment model.
[Refs](Refs.md#closest-work-and-reproduction-status) records what can be
reproduced and where the methods overlap.

## Experiment sequence

| Stage | What changes | What it must show | Stop if |
|---|---|---|---|
| Exploration | No model | Complete: the existing DuckDB queries covered schema, distributions, useful features, duplicates, time shift, and leakage risks. See the [findings](../exp/explore.md). | This stage supplies data rules, not evidence that a model works. |
| M0 Matched | Train the 25M backbone only on labels | Before training, [build the splits, training-only preprocessing, and causal-context checks under `src`](Architecture.md#m0), then compare with every applicable classical baseline. | It fails the [classical model check](#classical-model-check). Only the restricted S0 check may then run; do not scale. |
| M1-R | Pretrain with $L_{\mathrm{raw}}$ on flat causal contexts | Pass the common checks below against randomly initialized M0 Matched. | The gain needs identifiers or ports, disappears chronologically, or exists only in-domain. |
| M1-L | Replace raw reconstruction with $L_{\mathrm{latent}}$ | Pass the collapse checks and beat M0 Matched in separate exposure-matched and FLOP-matched comparisons. | Raw reconstruction matches it or teacher/student collapse persists. |
| M2-H | Combine $L_{\mathrm{raw}}$ and $L_{\mathrm{latent}}$ | Beat M1-R and M1-L under both exposure and FLOP matching. | Either constituent or MMAE-NF matches it. |
| M2-F | Conditionally add $L_{\mathrm{future}}$ | Beat the M2-H model that passed earlier checks, eligible-anchor M2-H, and every named future control on external low-label transfer under fixed exposure and fixed FLOPs, then pass the common checks. | Real targets fail to beat stratified shuffle on predictive validation and external transfer; gains need ports, stay in-domain, fail a time shuffle, need extra heads or a loss sweep; or any common check fails. |
| M3-Ego | Replace flat history with causal endpoint-ego history | Positive context main effect and positive hybrid×ego interaction. | Random, time/feature-matched, or CMES grouping matches it. |
| M4-Rel | Add directed identity-free endpoint relation bias | Beat no-bias and CMES-Causal; pass relation destruction and endpoint-renaming tests. | Identity is required, the relation goes unused, or endpoint-held-out performance falls. |
| M5-Hier | Conditional multi-resolution summaries | Run only if ≥20% of valid contexts truncate at 256 and long-horizon recall is weak; beat non-hierarchical M4-Rel on identical ego history within the Thesis resource limits. | Non-hierarchical M4-Rel matches it or runtime exceeds the limit. |
| X1-Distill | Frozen YaTC/packet teacher used only if paired data exists | Show safe pairing, pass an updated literature search, and improve external results for the flow-only student over same-modality controls. | Pairing or privacy checks fail, shuffled packet alignment works just as well, or the gain is <1 absolute AUPRC point. |
| Final | No mechanism | Five pretraining-seed replication on untouched domains/tasks. | Any primary failure narrows the claim. |

### Classical model check

At `k=10` labelled support groups per class, M0 Matched must meet all three of
these conditions against the best applicable classical model:

- improve macro one-vs-rest AUPRC by at least 1.0 absolute point;
- have a paired hierarchical-bootstrap 95% lower bound above zero; and
- keep TPR at 10 false alerts per million within one absolute point.

Anything else counts as parity, including a classical win.

Under parity, run only three flat-context S0 arms—scratch, M1-R, and M1-L—at
20M event exposures, plus a separate fixed-FLOP comparison. Use the screening
seeds and adaptation repeats fixed in Thesis. Do not run hybrid, future, ego,
relation, hierarchy, or scaling yet. Continue only if M1-R or M1-L beats both
S0 scratch and the best classical model by the same AUPRC, confidence, and
false-alert checks on at least two development networks. Otherwise stop all
Transformer and SSL work. If one objective passes, finish the standard S0
screen below before moving to the 25M models.

M3-Ego uses the context and matched-control definitions in
[Architecture](Architecture.md#context-and-endpoint-relationships). Its
controls are flat collector history, same-size random earlier history,
time-and-feature-matched history without a shared endpoint, target-only, CMES
grouping. M4-Rel uses Architecture's anonymous relation tensor and causal-logit
bias; it does not introduce another relation taxonomy.

Before full progression, run the S0 screen over the fixed 20M exposures:
{scratch, raw, latent, hybrid} × {flat, ego}, plus the named paper controls;
test relation bias separately. Estimate
(EgoHybrid − EgoScratch) − (FlatHybrid − FlatScratch). The proposed combined
mechanism survives only if its paired block-bootstrap 95% lower bound is above
zero on the low-label cross-domain aggregate chosen in advance. Other options
and their comparison rules are in
[Thesis](Thesis.md#options-outside-the-main-plan).

M2-F is not required before M3-Ego. If M2-F fails, start M3-Ego from M2-H. If
M2-F and M3-Ego each pass on their own, their combination must beat both
branches under the common checks before it can continue.

Context length is not part of the JEPA comparison. If it is revisited, run one
separate source-validation ablation with total-token limits $W=\{3,8,20\}$,
including the target. Left-pad every arm to 20 tokens, keep target anchors and
optimizer updates fixed, and change only the number of visible earlier events.
Do not sample a random range, combine this check with M2-F, or replace the
256-event setup without separate evidence.

## Passing checks, collapse checks, and scaling

A stage continues only when all of these conditions hold:

- it improves macro one-vs-rest AUPRC by at least 1.0 absolute point over the
  previous model at 10 labelled groups per supported class and on the area
  under the fixed label-efficiency grid;
- its paired hierarchical block-bootstrap 95% lower bound is above zero;
- it improves at least two independent targets;
- no target loses more than one macro-AUPRC point;
- no critical family loses more than two recall points; and
- it beats the best applicable comparison that uses the same inputs.

Run fixed-data-exposure and fixed-total-FLOP comparisons separately. Use the
screening seeds, final seeds, and adaptation repeats fixed in
[Thesis](Thesis.md#evaluation-and-leakage-checks). Lower SSL loss alone is not
enough. The false-alert, task-coverage, and resource limits remain those in
[Thesis](Thesis.md#evaluation-and-leakage-checks). Compute label-efficiency
area by trapezoidal integration against `log10(k)`, normalized by the grid's
log span; do not average the five points equally.

During pretraining, sample source `d` with `p(d) ∝ n_d^alpha`. Choose `alpha`
once at S0 from `{0, 0.5, 1}`, cap each source at 20%, redistribute excess
mass, then freeze the choice and sampling sequence.

At every validation checkpoint, record embedding standard-deviation quantiles,
covariance eigenvalues, active/effective rank, off-diagonal correlation,
mean/nearest-neighbour cosine, alignment/uniformity, encoder and head gradient
norms, and source/day/entity distributions. Reject a checkpoint if any persists
for three validations: median embedding standard deviation <1e-3; active or
effective rank <10% of width; more than half of dimensions with standard
deviation <1e-3; rank falls at least 50% while SSL loss improves; or a
constant-vector baseline is indistinguishable at every label budget. These are
project thresholds fixed before any results are seen, not constants from the
literature.

| Stage | Model | Minimum unique flows / independent sources | Event exposures | Next step |
|---|---:|---:|---:|---|
| S0 | 3--5M | 5M / — | 20M | mechanism screen only |
| S1 | 25M | 50M / 10 | 100M | external transfer |
| S2 | 25M | 250M / 20 | 500M | scale confirmation |
| S3 | 90M | 1B / 30 | after S2 | only if S2 passes |
| S4 | 300M | — | — | only if 90M beats 25M by ≥2% relative external-transfer AUPRC |

Do not substitute repeated epochs or larger models for source diversity. Scale
data before width. Stop after two consecutive doubled-exposure runs lower SSL
loss without positive transfer; keep the smallest model that passed.
