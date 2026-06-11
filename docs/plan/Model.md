# Models and experiments

## What this file covers

[Architecture](Architecture.md) covers inputs, [Thesis](Thesis.md) evaluation,
and [Refs](Refs.md) evidence.

## Training objectives and controls

`unlabelled source flows -> masked causal context -> shared encoder -> temporary raw/EMA/future loss -> remove SSL heads -> low-label transfer`

I pretrain source flows by masking 30% of semantic groups in contiguous spans
averaging three. The encoder sees the masked sequence. Targets stay evaluation-only under
[Thesis](Thesis.md#evaluation-and-leakage-checks).
[Architecture](Architecture.md#prediction-unit-and-input-features) lists five
semantic groups and exclusions.

| Objective | Signal | Plain analogy | Target |
|---|---|---|---|
| $L_{\mathrm{raw}}$ | Hidden feature groups | Fill blacked-out boxes from surrounding flows. | Reconstruct selected categorical, missingness, and numeric values with cross-entropy, BCE, and Smooth L1. Average by group then across groups. Never reconstruct padding. |
| $L_{\mathrm{latent}}$ | Same-event EMA teacher state | Match a teacher's summary, not every value. | Predict the mean of top four layer-normalized teacher states. Student predictor: two layers. Teacher dropout: off. Momentum: 0.99 to 0.9999 cosine. |
| $L_{\mathrm{hybrid}}$ | Raw values and teacher state | Learn detail and summary together. | $\lambda_{\mathrm{raw}}L_{\mathrm{raw}}+\lambda_{\mathrm{latent}}L_{\mathrm{latent}}$. |
| $L_{\mathrm{future}}$ | Later endpoint-incident flow state | Predict a later relevant event. | M2-F only. LeNEPA covers causal next-latent prediction; HEPA covers causal horizon-conditioned future representations. I test only whether endpoint selection adds value. |

For each anchor and $h\in\{1,4,16\}$, I take the $h$-th later endpoint-incident
flow in the same corpus, stream, and pretraining partition. Completion time,
then event ID, breaks ties.

The student sees only masked past context. The dropout-free EMA teacher sees
the target and its causal completion prefix. Let $z_i$ be the student anchor
state and $H_i$ its available horizons.

$$
L_{\mathrm{future}}(i)=\frac{1}{|H_i|}\sum_{h\in H_i}
\left\|\mathrm{norm}\!\left(p(z_i+e_h)\right)-
\mathrm{sg}\!\left(\mathrm{norm}(\bar z_{i,h})\right)\right\|_2^2.
$$

$e_h$ is a learned horizon embedding, $p$ is one shared two-layer predictor,
and $\bar z_{i,h}$ is the mean of the teacher's top four layer-normalized
states. $\mathrm{norm}(u)=u/(\lVert u\rVert_2+10^{-6})$. I skip unavailable
horizons and anchors with none, then average horizons within each anchor.

In M2-F I train
$L_{\mathrm{M2-F}}=\lambda_{\mathrm{raw}}L_{\mathrm{raw}}+
\lambda_{\mathrm{latent}}L_{\mathrm{latent}}+
\lambda_{\mathrm{future}}L_{\mathrm{future}}$.

For every hybrid arm, I set
weights from 200 training-only warm-up batches by inverse median loss,
normalize them to sum to two, then freeze them. M2-F has three weights;
eligible-anchor M2-H warms separately. I count warm-up compute.

## Shared deployable model

M0 Base and M0 Matched share a factorized one-hot-equivalent record encoder
under PAD, UNK, and missing values.

### M0 supervised models

| Model | Role | Shape |
|---|---|---|
| M0 Base | The only FlowTransformer-style baseline. Offline unrestricted-attention upper bound. | Same fields and record encoder as M0 Matched; $d_{model}=512$, 8 post-LN blocks, 8 heads, FFN 1792, ReLU, dropout 0.1, no positions, 256 events, $25M\pm5\%$. |
| M0 Small | Cheap causal check. | $d_{model}=64$, 2 post-LN blocks, 2 heads, FFN 128, ReLU, dropout 0.1, no positions, 8 flows. |
| M0 Matched | Causal 25M model. It differs from M0 Base only by attention mask. | Same shape and capacity as M0 Base. |

### S0 screening model

S0 screens SSL: $d_{model}=256$, 4 blocks, 8 heads, FFN 1024, 256 events,
tokenizer at most 1.5M, encoder 3 to 5M.

I count trainable parameters. One search fixes
context horizon $\{1,10,60\}$ minutes, learning rate
$\{10^{-4},3\times10^{-4}\}$, weight decay $\{0.01,0.05\}$, and dropout
$\{0,0.1\}$. I use AdamW, clipping 1.0, 5% linear warm-up, cosine decay, and
fixed event exposure per update. AMP must match FP32 first.

## Experiment sequence

| Stage | SSL signal and reason | Run trigger | Stop trigger |
|---|---|---|---|
| Exploration | None. I set rules from [findings](../exp/explore.md). | Before models. | It is not model evidence. |
| M0 Matched | Labels only. I establish causal supervised performance. | Before SSL. | Fail [classical model check](#classical-model-check), except restricted S0. |
| M1-R | $L_{\mathrm{raw}}$ tests portable reconstruction. | M0 permits SSL. | Gains need identifiers/ports, vanish chronologically, or stay in-domain. |
| M1-L | $L_{\mathrm{latent}}$ tests representation learning. | Run beside M1-R. | Collapse persists or raw reconstruction matches it. |
| M2-H | $L_{\mathrm{hybrid}}$ tests both signals. | M1 tests finish. | A constituent or MMAE-NF matches it. |
| M2-F | $L_{\mathrm{future}}$ tests endpoint-incident future targets. | M2-H passes and targets exist. | It must beat passed M2-H, eligible-anchor M2-H, and every future control, or it stops. |
| M3-Ego | Reuse passing M2-H objective over endpoint-ego history. | After M2-H, even if M2-F fails. | Flat, random, matched non-endpoint, or CMES grouping matches it. |
| M4-Rel | Reuse passing M2-H objective with directed identity-free relation bias. | M3-Ego passes. | It needs identity, goes unused, or endpoint-held-out performance falls. |
| M5-Hier | Reuse passing M2-H objective at several time scales. | At least 20% truncate at 256 and long-horizon recall is weak. | M4-Rel matches it or runtime fails Thesis limits. |
| X1-Distill | Paired packet-teacher state trains a flow-only student. | Exact safe pairs and refreshed literature check. | Pairing/privacy fails, shuffled alignment works, or gain is under 1 absolute AUPRC point. |
| Final | No new mechanism. | Choices frozen. | A primary failure narrows my claim. |

Under M0 parity I run only flat scratch, M1-R, and M1-L plus fixed-FLOP
comparison. One must beat
scratch and the best classical control on M0 criteria in two development
networks. Otherwise I stop Transformer and SSL work.

Before promotion I run 20M-exposure S0
`{scratch, raw, latent, hybrid} x {flat, ego}`, paper controls, and relation
separately. The paired block-bootstrap lower bound of
$(EgoHybrid-EgoScratch)-(FlatHybrid-FlatScratch)$ on the fixed low-label
cross-domain aggregate must exceed zero.

If M2-F and M3-Ego pass alone, I test $L_{\mathrm{M2-F}}$ on ego context once;
it must beat flat M2-F and ego M2-H. The separate context check uses
$W=\{3,8,20\}$, left-padding to 20, fixed anchors and updates, and only visible
earlier events.

## Comparisons

I fix eligible anchors, horizons, and strata before warm-up. Point-horizon arms
match encoder, predictor, masks, exposure, FLOPs, and time/activity strata;
only the third target and head change. I run fixed-exposure and fixed-FLOP
versions, matching predictor parameters and FLOPs within 5%.

- **Endpoint-incident:** the M2-F target.
- **Generic future-latent:** use the $h$-th later eligible flow in the same
  corpus, stream, partition, and elapsed-time bucket without requiring an
  endpoint match. Keep anchors, masks, teacher prefix, weights, updates, and
  FLOPs fixed. This is the HEPA-motivated generic temporal control.
- **Stratified shuffle:** use one fixed within-stratum derangement by corpus,
  stream, partition, day, horizon, elapsed time, candidate count, and
  pre-anchor endpoint activity and degree. A stratum that cannot be deranged
  leaves every point-horizon arm before I fix the common sequence.
- **Raw future:** predict allowed semantic fields at horizons 1, 4, and 16 with
  the same horizon embedding and group-balanced categorical, missingness, and
  numeric losses.
- **Shortcut checks:** predict the training-only corpus-horizon mean teacher
  state. Separately forecast 60-second endpoint aggregates over log-binned
  flows, bytes, packets, protocol mix, and unique remotes. The aggregate arm
  keeps anchors, third-loss weighting, updates, and FLOPs matched.

I report eligibility by corpus, day, activity, degree, horizon, and elapsed
time. The true target must beat stratified shuffle on validation and external
transfer. I report teacher/predictor rank, variance, and nearest-neighbour
cosine by horizon. Lower SSL loss proves nothing.

Using Thesis splits, I compare always-benign; median/MAD novelty over log
duration, bytes, and packets by protocol and port range; logistic regression;
Isolation Forest; OC-SVM; boosted trees with and without past aggregates;
per-flow MLP; M0 variants; scratch; frozen probes; full fine-tuning; and usable
E-GraphSAGE, Anomal-E, NEGSC, GraphIDS, and Van Langendonck implementations.

GNNet is unresolved. MMAE-NF is an adaptation. CMES-25M uses CMES context and
predicates; CMES-Causal uses ego context with the same predicates. Both are
offline comparators. M0 Base is the only baseline; all others are controls,
ablations, or comparators.

## Classical model check

At $k=10$ labelled support groups per class, M0 Matched must beat the best
classical control by 1.0 macro one-vs-rest AUPRC point, have paired
hierarchical-bootstrap 95% lower bound above zero, and keep TPR at 10 false
alerts per million within one absolute point. Otherwise it is parity.

## Passing checks, collapse checks, and scaling

A stage needs all of: 1.0 absolute macro-AUPRC gain at $k=10$ and the fixed
label-efficiency area; paired hierarchical-bootstrap 95% lower bound above
zero; gains on two independent targets; no target loss over one macro-AUPRC
point; no critical-family recall loss over two points; and a win over the best
same-input comparator.

I use trapezoids against $\log_{10}(k)$, normalized by log span.

I sample source $d$ with $p(d)\propto n_d^\alpha$. At S0 I choose
$\alpha\in\{0,0.5,1\}$ once, cap a source at 20%, redistribute excess, then
freeze the sequence. Every 100 updates I log gradients and objective-gradient
cosines. If over 50% conflict for three epochs, I test one compute-matched
alternating schedule.

At validation I record embedding spread, covariance eigenvalues, active/effective
rank, correlation, cosine, alignment, uniformity, gradients, and source/day/entity
distributions.

I reject a checkpoint after three validations if median standard
deviation is below $10^{-3}$, active/effective rank below 10% of width, over half
of dimensions below $10^{-3}$, rank falls 50% while SSL loss improves, or a
constant vector matches every label budget.

| Stage | Model | Minimum flows / sources | Exposures | Next step |
|---|---:|---:|---:|---|
| S0 | 3 to 5M | 5M / n.a. | 20M | mechanism screen |
| S1 | 25M | 50M / 10 | 100M | external transfer |
| S2 | 25M | 250M / 20 | 500M | scale confirmation |
| S3 | 90M | 1B / 30 | after S2 | only if S2 passes |
| S4 | 300M | n.a. | n.a. | only if 90M beats 25M by at least 2% relative external-transfer AUPRC |

I stop after two doubled-exposure runs lower SSL loss without positive transfer
and keep the smallest passing model.
