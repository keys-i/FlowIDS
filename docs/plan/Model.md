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

All neural rungs use one completed bidirectional flow per token and the same
deployable encoder unless size is the isolated variable. The record encoder has
train-only transformed numeric fields with field-specific projections,
categorical embeddings with PAD/UNK, missingness embeddings, and a two-layer
projection to event width. The causal Transformer is d_model=512, eight pre-LN
blocks, eight heads, FFN 2048, GELU, nominal dropout 0.1, RoPE sequence
position, and learned elapsed-time features. It consumes at most 256 events
under the Architecture context contract and uses the final completed-flow
state. The Transformer
blocks alone are approximately 25.2M weights. Count the record encoder,
embeddings, and backbone in the complete deployable total; if it falls outside
25M ±5%, reduce FFN width only enough to enter the band before any experiment,
then freeze that measured architecture for every rung. SSL decoders,
predictors, and EMA copies are removed at inference; their training FLOPs and
memory are reported. A bidirectional model is an offline upper bound, never a
deployable equivalent.

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
RoPE, elapsed-time projection, optimizer schedule, objectives, and downstream
protocol at d_model=256, four pre-LN blocks, eight heads, FFN 1024, and maximum
256 events. Its four blocks contain about 3.16M parameters. Cap the tokenizer
at 1.5M parameters so the complete trainable encoder is 3--5M; report the exact
count before screening. Screen at least 5M unique flows and exactly 20M event
exposures. S0 changes only capacity, so a mechanism passing S0 must be
confirmed at S1 before any claim.

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
on a cosine schedule after optimizer updates. This is a data2vec-style
same-event comparator, not a future-prediction method.

$L_{\mathrm{future}}$ is a conditional JEPA-family future-latent comparator in
the existing M2-F rung, not a contribution or distinct architectural
mechanism. Its causal next-latent geometry most directly collides with LeNEPA,
although this candidate retains an EMA teacher. For anchor flow $i$, let $z_i$
be the masked student's final anchor state and $H_i$ be the available horizons
in $\{1,4,16\}$:

$$
L_{\mathrm{future}}(i)=\frac{1}{|H_i|}\sum_{h\in H_i}
\left\|\operatorname{norm}\!\left(p(z_i+e_h)\right)-
\operatorname{sg}\!\left(\operatorname{norm}(\bar z_{i,h})\right)\right\|_2^2.
$$

$\operatorname{norm}(u)=u/(\lVert u\rVert_2+10^{-6})$. $e_h$ is a learned
horizon embedding with the encoder width, $p$ is one shared two-layer predictor,
and $\bar z_{i,h}$ is the same top-four-layer EMA-teacher target used by
$L_{\mathrm{latent}}$, now taken from the $h$-th later completed flow incident
to either anchor endpoint. The teacher sees that target flow's own causal prefix
ending at its completion, with dropout disabled. The student never sees a later
flow or its metadata. Order candidates by completion time then deterministic
event identifier within the same dataset corpus, capture/exporter observation
stream, and pretraining partition; omit missing horizons and anchors for which
$H_i$ is empty; average available horizons per anchor before averaging the
batch so busy endpoints cannot dominate. Remove the predictor and EMA teacher
at inference.

M2-F may add $L_{\mathrm{future}}$ only after M2-H promotes:

$$
L_{\mathrm{M2-F}}=\lambda_{\mathrm{raw}}L_{\mathrm{raw}}+
\lambda_{\mathrm{latent}}L_{\mathrm{latent}}+
\lambda_{\mathrm{future}}L_{\mathrm{future}}.
$$

Before warm-up, freeze the eligible anchor/horizon sequence and all activity
strata. On 200 batches from that sequence, set all three weights to inverse
median loss and normalize them to sum to two, then freeze them. The
eligible-anchor M2-H arm receives its own 200-batch two-loss warm-up under the
existing sum-to-two rule; count warm-up and training events for every arm.
Every point-horizon control replaces only the third target/head and uses the
same weighting rule, anchor/horizon sequence, optimizer updates,
trainable-parameter budget, and separate fixed-event-exposure and
fixed-total-FLOP comparisons. Do not search weights.

Compare M2-F against both promoted M2-H and M2-H retrained on the exact
future-eligible anchors. Every future/control arm uses the same anchor and
horizon-availability masks. Point-horizon controls are: a stratified shuffled
latent target preserving dataset corpus, capture/exporter, partition, day,
horizon, relative-time bucket, candidate-count bucket, and train-only
activity/degree buckets computed from causal pre-anchor histories for both
anchor endpoints; and future prediction of the permitted raw semantic fields
at all three horizons using the same shared horizon conditioning and the
group-balanced categorical, missingness, and numeric losses. Fit every stratum
edge on the pretraining training partition and freeze it. For each run, use one
precommitted shuffle seed and one fixed within-stratum derangement of target
event IDs; never reshuffle by batch or epoch. A stratum without a valid
derangement is unsupported, and its anchor/horizon pairs are removed from every
point-horizon arm before the common sequence is frozen.

Two additional diagnostics are: a frozen train-only teacher-target mean for
each corpus and horizon; and a 60-second raw endpoint aggregate forecast
(log-binned flow, byte, packet, protocol-mix, and unique-remote-count
summaries). The aggregate is not a horizon-matched target: train it on the same
eligible anchor set as a separate shortcut diagnostic, apply the same
inverse-median third-loss weighting, and match its head parameters, updates,
and total FLOPs within 5%. Match point-horizon predictor parameters and total
FLOPs within 5% where target widths differ.

The real target must beat stratified shuffle on predictive validation loss
*and* external low-label transfer. Report eligibility and results by corpus,
day, both-endpoint activity/degree, horizon, and target elapsed-time bucket. At
each horizon, report teacher and predictor rank, variance, and nearest-neighbour
cosine. Lower SSL loss alone is not evidence.

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
MLP; supervised flat causal Transformer; the `historical-recreation`; matched
random initialization; and frozen linear probe plus full fine-tuning for every
pretrained encoder. No candidate advances by beating only a weak neural
control.

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
| Exploration | No model | Complete: the existing DuckDB queries inspected schema, distributions, usable features, duplicates, time shift, and leakage risks. See the [findings](../exp/explore.md). | Exploration is not promotion evidence. Carry its exclusions and data-handling decisions into M0. |
| M0 | 25M backbone trained only on labels | Before training, [build the actual splits, train-only preprocessing, and causal-context checks under `src`](Architecture.md#m0-implementation); then establish the reference versus every applicable classical control. | If it fails the [classical-parity gate](#classical-parity-discriminator), permit only that restricted S0 discriminator; do not scale. |
| M1-R | L_raw pretraining on flat causal contexts | Pass the common frozen promotion gate below against random-init M0. | Reject if gain needs identifiers or ports, vanishes chronologically, or is in-domain only. |
| M1-L | L_latent instead of raw; parallel constituent | Pass collapse screen and beat M0 in the separate exposure-matched and FLOP-matched comparisons. | Reject if raw matches it or teacher/student collapse persists. |
| M2-H | Add the other constituent: L_raw + L_latent | Beat M1-R and M1-L, not merely M0, under exposure and FLOP matching. | Reject if either constituent or MMAE-NF matches it. |
| M2-F | Conditionally add L_future | Beat promoted M2-H, eligible-anchor M2-H, and every named future control on external low-label transfer under fixed exposure and fixed FLOPs, then pass the common promotion and non-inferiority gates. | Reject if real targets do not beat stratified shuffle on predictive validation and external transfer; if gains are port-dependent, in-domain-only, time-shuffle-sensitive, or need extra heads or a loss sweep; or if any ordinary promotion gate fails. |
| M3-Ego | Replace flat history with causal endpoint-ego history | Positive context main effect and positive hybrid×ego interaction. | Reject if random, time/feature-matched, or CMES grouping matches it. |
| M4-Rel | Add directed identity-free endpoint relation bias | Beat no-bias and CMES-Causal; pass relation destruction and endpoint-renaming tests. | Reject if identity is required, relation is unused, or endpoint-held-out performance falls. |
| M5-Hier | Conditional multi-resolution summaries | Run only if ≥20% valid contexts truncate at 256 and long-horizon recall is deficient; beat non-hierarchical M4-Rel on identical ego history within Thesis resource gates. | Reject if non-hierarchical M4-Rel matches it or runtime cost fails its gate. |
| X1-Distill | Conditional frozen YaTC/packet teacher to flow student | Leakage-safe paired data, refreshed novelty search, and flow-only external gain over same-modality controls. | Reject if pairing/privacy fails, packet alignment is shuffled-equivalent, or gain is <1 absolute AUPRC point. |
| Final | No mechanism | Five pretraining-seed replication on untouched domains/tasks. | Any primary failure narrows the claim. |

### Classical-parity discriminator

M0 clears the classical gate only when all three conditions hold at `k=10`
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

M2-F is not a prerequisite for M3-Ego. If M2-F rejects, M3-Ego starts from
M2-H. If M2-F and M3-Ego each pass separately, their combination must beat both
branches under the common gate before it enters the promoted stack.

Context length is not a JEPA component. If revisited, run one separate
source-validation ablation using total-token ceilings $W=\{3,8,20\}$, including
the final target. Left-pad every arm to 20 tokens, hold target anchors and
optimizer updates fixed, and change only the number of visible earlier events.
Do not sample a random range, combine this change with M2-F, or replace the
256-event contract without separate promotion evidence.

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
