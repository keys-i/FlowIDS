# Model ladder

A flow record summarises a network exchange using counts, sizes, and timings.
We want to learn from unlabelled records, then detect attacks on another
network using fewer labels. This first step is called **pretraining**.

M0 is the FlowTransformer-style baseline. M1 and M2 have local code and CPU
checks. M3 onward remains planned; no real-data comparison is recorded.
[Architecture](Architecture.md) defines the inputs, [Thesis](Thesis.md) defines
the tests, and [Refs](Refs.md) links the papers.

## Experiment sequence

`reconstruct` predicts field values; `teacher` predicts a learned vector;
`hybrid` combines both. A **teacher** supplies training vectors instead of attack labels.

“Closest work” means the closest comparison in our notes. The differences
are design choices to test; the analogies explain the training task.

| Model | What we add | Closest work in our notes | How ours differs | Analogy |
|---|---|---|---|---|
| **M0** | Baseline trained on attack labels | [FlowTransformer](../notes/traffic/flowtransformer.md) | Our PyTorch baseline uses train-only preprocessing and a purged time split. Base and Matched hold the model shape fixed and change the attention mask. | Learn to flag entries in a labelled traffic log. |
| **M1 reconstruct** | Reconstruct hidden groups of flow fields | [MMAE](../notes/traffic/mmae.md), reconstruction part | Predict flow counts, categories, and missingness. MMAE uses packet bytes, flow mixing, and a teacher; this arm uses only reconstruction. | Fill blanks in the log. |
| **M1 teacher** | Predict the unmasked teacher's vector for the same flow | [data2vec](../notes/epa/data2vec.md), [TS-JEPA](../notes/epa/tsjepa.md) | Apply masked teacher prediction to irregular flow records with past-only attention, rather than language, images, or sampled time-series patches. | Guess the teacher's summary from a partly hidden page. |
| **M2 hybrid** | Combine M1 reconstruct and M1 teacher losses | [MMAE](../notes/traffic/mmae.md) | Use grouped NetFlow fields, not packet bytes or FlowMix. Test whether the combination beats each loss alone and an MMAE adaptation. | Recover both the missing numbers and the summary. |
| **M3-Ego** | Give M2 hybrid past flows involving either endpoint | [CMES cross-flow model](../notes/traffic/cmes.md) | Restrict selection to completed past flows touching either host. Test history selection separately from relation-aware attention. | Read the two hosts' logbooks instead of the latest entries from everyone. |
| **M4-Rel** | Add endpoint relationships to M3 attention | [CMES cross-flow model](../notes/traffic/cmes.md) | Use only anonymous, directed endpoint matches. Exclude address values, port/protocol similarity, and future flows. | Mark who contacted whom, even after names change. |
| **M2 future-hybrid**, optional | Add future-flow targets to M2 hybrid | [HEPA](../notes/epa/hepa.md), [LeNEPA](../notes/epa/lenepa.md) | Predict individual later flows selected by endpoint, using an EMA teacher. Remove the predictor after training; future flows never enter the student input. | Guess a host's next log entries from its earlier ones. |
| **M2 future-jepa** | Predict only future teacher vectors | [I-JEPA](../notes/epa/ijepa.md), teacher/predictor idea | Use the same masked history and future targets as future-hybrid, with no reconstruction or same-flow loss. This is our temporal JEPA variant, not an I-JEPA reproduction. | Predict the next entries' summaries. |
| **M5-Hier**, conditional | Summarise M4 history at several time scales | [netFound](../notes/background/papers.md#netfound), [FlowletFormer](../notes/background/papers.md#flowletformer) | Proposed flow-only summaries over 1, 10, and 60 seconds, with past-only attention and the same runtime limit. Earlier paper notes still need rechecking. | Read short summaries alongside individual entries. |
| **X1-Distill**, conditional | Teach a flow-only student using paired packet data | [ConMD](../notes/background/papers.md#conmd); [YaTC](../notes/background/papers.md#yatc) as a possible teacher | The proposed student uses only flow fields at deployment. The exact difference from ConMD needs checking once paired data and a teacher are chosen. | Learn from someone who sees the full recording, then work from its summary. |

The M1/M2 commands pretrain, then fine-tune with all scorable training labels on
one NF3 dataset. The broader comparisons below have not run.

Run M1 reconstruct and M1 teacher together, then M2 hybrid → M3-Ego → M4-Rel. M2 future-hybrid is a separate
branch after M2 hybrid. M3, M4, and M5 keep the M2 hybrid objective; they do not inherit
M2 future-hybrid. Remove additions that fail their comparisons.

Try M5 only if at least 20% of histories must be cut to 256 events and recall
over longer histories is weak. X1 needs exact permitted packet/flow pairs and
a refreshed literature check. It must beat shuffled pairing and meet the
[required improvement](Thesis.md#what-counts-as-an-improvement).

### Which past flows should the model see?

The endpoints are the hosts at either end of a flow. For target A → B,
consider two available history slots:

| Earlier flows, oldest first | Flat history | Endpoint history |
|---|---|---|
| A → C, D → B, X → Y, P → Q | X → Y, P → Q | A → C, D → B |

Both models also receive A → B. The letters select records; they are not
features. The comparison tests whether unrelated recent traffic crowds out
useful endpoint history.

## Training objectives and controls

<details>
<summary>Exact losses and model settings</summary>

Mask 30% of the [five feature groups](Architecture.md#prediction-unit-and-input-features)
using randomly shifted three-event spans; adjacent spans can join. The student sees the masked sequence.
Follow the [training/test separation](Thesis.md#evaluation-and-leakage-checks).

| Loss | Exact training target |
|---|---|
| $L_{\mathrm{raw}}$ | Selected categorical, missingness, and numeric values, using cross-entropy, binary cross-entropy (BCE), and Smooth L1 respectively. Average by group, then across groups. Never reconstruct padding. |
| $L_{\mathrm{latent}}$ | Smooth L1 against the mean of the teacher's top four layer-normalized states for the same event. Two-layer student predictor; teacher dropout off; EMA momentum rises from 0.99 to 0.9999 on a cosine schedule. |
| $L_{\mathrm{hybrid}}$ | $\lambda_{\mathrm{raw}}L_{\mathrm{raw}}+\lambda_{\mathrm{latent}}L_{\mathrm{latent}}$. |
| $L_{\mathrm{future}}$ | The later endpoint-related teacher states defined below, used by both M2 future variants. |

For both M2 future variants, [Architecture](Architecture.md#future-targets) defines the later flows
and teacher inputs. An **anchor** is the current flow. Let $z_i$ be its student
state and $H_i$ its available horizons from $h\in\{1,4,16\}$.

$$
L_{\mathrm{future}}(i)=\frac{1}{|H_i|}\sum_{h\in H_i}
\left\|\mathrm{norm}\!\left(p(z_i+e_h)\right)-
\mathrm{sg}\!\left(\mathrm{norm}(\bar z_{i,h})\right)\right\|_2^2.
$$

$e_h$ is a learned horizon embedding, $p$ is one shared two-layer predictor,
and $\bar z_{i,h}$ is the mean of the teacher's top four layer-normalized
states. $\mathrm{norm}(u)=u/(\lVert u\rVert_2+10^{-6})$. I skip unavailable
horizons and anchors with none, then average horizons within each anchor.
`sg` means stop-gradient: the teacher target is not updated by this loss.

In M2 future-hybrid I train
$L_{\mathrm{M2 future-hybrid}}=\lambda_{\mathrm{raw}}L_{\mathrm{raw}}+
\lambda_{\mathrm{latent}}L_{\mathrm{latent}}+
\lambda_{\mathrm{future}}L_{\mathrm{future}}$.

For every hybrid arm, I set
weights from 200 training-only warm-up batches by inverse median loss,
normalize them to sum to two, then freeze them. M2 future-hybrid has three weights;
eligible-anchor M2 hybrid warms separately. I count warm-up compute.
The implementation measures 200 forward-only batches with dropout off and
training RNG preserved; it does not update the student or teacher during this
step. `future-jepa` uses only $L_{\mathrm{future}}$ with weight one.

## Shared deployable model

Width is the flow-vector size; FFN is the feed-forward hidden width. Context
length includes the target. Base and Matched share the same factorized
one-hot-equivalent record encoder, including PAD, UNK, and missing values.
They differ only in attention masking.

### M0 supervised models

| Model | Role | Shape |
|---|---|---|
| M0 Base | Attention across the supplied window | Width 512, 8 post-LN blocks, 8 heads, FFN 1792, 256 events, $25M\pm5\%$ parameters |
| M0 Matched | Each record attends only to itself and earlier records | Same shape as Base |
| M0 Small | Cheap check of the causal model | Width 64, 2 post-LN blocks, 2 heads, FFN 128, 8 flows |

All three use ReLU, dropout 0.1, and no position embeddings. Both Base and
Matched windows end at the target; neither receives later completed flows.

### S0 screening model

S0 is the small trial before larger pretraining runs: width 256, 4 blocks,
8 heads, FFN 1024, 256 events, tokenizer at most 1.5M parameters, encoder
3 to 5M. SSL means self-supervised learning from unlabelled inputs.

I count trainable parameters. One search fixes
context horizon $\{1,10,60\}$ minutes, learning rate
$\{10^{-4},3\times10^{-4}\}$, weight decay $\{0.01,0.05\}$, and dropout
$\{0,0.1\}$. I use AdamW, clipping 1.0, 5% linear warm-up, cosine decay, and
fixed event exposure per update. AMP must match FP32 first.

</details>

## Screening and branch comparisons

<details>
<summary>Matched tests, stopping rules, and scale settings</summary>

If both M1 objectives fail, stop SSL. Drop M1 teacher if it repeatedly produces
nearly constant vectors or adds nothing over M1 reconstruct. First apply the
[classical model check](#classical-model-check).

Before moving to the larger experiments, run 20M-exposure S0
`{scratch, raw, latent, hybrid} x {flat, ego}`, paper controls, and relation
separately. The paired block-bootstrap lower bound of
$(EgoHybrid-EgoScratch)-(FlatHybrid-FlatScratch)$ on the fixed low-label
cross-domain aggregate must exceed zero.

Pretraining must help more with endpoint history than flat history. An
*exposure* counts each presentation of a flow, including repeats.

If M2 future-hybrid and M3-Ego pass alone, I test $L_{\mathrm{M2 future-hybrid}}$ on ego context once;
it must beat flat M2 future-hybrid and ego M2 hybrid. The separate context check uses
$W=\{3,8,20\}$, left-padding to 20, fixed anchors and updates, and only visible
earlier events. `W` includes the target. Grouping flows into sessions by time
gaps remains a training-only diagnostic, followed by at most one fixed,
isolated comparison. History controls are defined in
[Architecture](Architecture.md#context-and-endpoint-relationships).

## Comparisons

### M2 future-hybrid: distinguish useful future targets from shortcuts

Compare with M2 hybrid and M2 hybrid retrained on the same eligible anchors, so busy
endpoints do not benefit simply from having more future targets.

FLOPs count floating-point operations. Fix eligible anchors, horizons, and
comparable time/activity groups (strata) before warm-up. Match encoder,
predictor, masks, and groups; change only the third target and head. Run
fixed-exposure and fixed-FLOP versions, matching predictor parameters and
FLOPs within 5%.

- **Endpoint-incident:** the M2 future-hybrid target.
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

Report eligibility by corpus, day, activity, degree, horizon, and elapsed time;
teacher/predictor rank, variance, and nearest-neighbour cosine by horizon.
Keep M2 future-hybrid only if it beats every future control on predictive validation and
external few-label detection under the [acceptance rules](Thesis.md#what-counts-as-an-improvement).
A tie with shuffled targets rejects it.

### Other models on the same data

Using Thesis splits, I compare always-benign; median/MAD novelty over log
duration, bytes, and packets by protocol and port range; logistic regression;
Isolation Forest; OC-SVM; boosted trees with and without past aggregates;
per-flow MLP; M0 variants; scratch; frozen probes; full fine-tuning; and usable
E-GraphSAGE, Anomal-E, NEGSC, GraphIDS, and Van Langendonck implementations.

MMAE-NF adapts MMAE to flow records. CMES-25M uses the paper's context and
relation rules; CMES-Causal uses endpoint history with those rules. These
are adaptations. See [Refs](Refs.md) for sources and the unresolved GNNet entry.

<a id="classical-model-check"></a>

## Classical model check for later research

This is a later research-screening condition, not a required M0 baseline run.
M0 Matched must pass [the classical threshold](Thesis.md#what-counts-as-an-improvement)
before it is used to justify later pretraining work.
The current binary launcher [cannot yet run this test](../context.md#what-remains).

If M0 does not clearly beat classical models, allow only a restricted S0:
flat scratch, M1 reconstruct, and M1 teacher, including a matched-FLOP comparison. At least
one pretrained model must beat scratch and the best classical model under
M0 criteria on two development networks. Otherwise stop Transformer and SSL work.

## Passing checks, collapse checks, and scaling

Use the [acceptance rules](Thesis.md#what-counts-as-an-improvement) for every stage.

I sample source $d$ with $p(d)\propto n_d^\alpha$. At S0 I choose
$\alpha\in\{0,0.5,1\}$ once, cap a source at 20%, redistribute excess, then
freeze the sequence. Every 100 updates I log gradients and objective-gradient
cosines. If over 50% conflict for three epochs, I test one compute-matched
alternating schedule.

**Unresolved:** a 20% source cap needs at least five sources. We have three
development datasets, and CIC must stay held out. Set a feasible mixture
before S0; M0 does not implement this cap.

At validation I record embedding spread, covariance eigenvalues, active/effective
rank, correlation, cosine, alignment, uniformity, gradients, and source/day/entity
distributions.

I reject a checkpoint after three validations if median standard
deviation is below $10^{-3}$, active/effective rank below 10% of width, over half
of dimensions below $10^{-3}$, rank falls 50% while SSL loss improves, or a
constant vector matches every label budget.

These checks catch **collapse**: different flows getting nearly identical vectors.

| Stage | Model | Minimum flows / sources | Exposures | Next step |
|---|---:|---:|---:|---|
| S0 | 3 to 5M | 5M / unresolved mixture above | 20M | test which additions help |
| S1 | 25M | 50M / 10 | 100M | external transfer |
| S2 | 25M | 250M / 20 | 500M | check whether more data helps |
| S3 | 90M | 1B / 30 | after S2 | only if S2 passes |
| S4 | 300M | n.a. | n.a. | only if 90M beats 25M by at least 2% relative external-transfer AUPRC |

I stop after two doubled-exposure runs lower SSL loss without positive transfer
and keep the smallest passing model.

Pooled future-window targets wait until point-horizon tests pass. GraphSAGE
is a comparator; add no graph module unless it wins with the same inputs and
tests.

</details>
