# Thesis research plan

I test whether source-only SSL on completed NetFlow reduces labels on a new
network. [Model](Model.md) defines models and checks,
[Architecture](Architecture.md) has inputs, and [Refs](Refs.md) has evidence.
No results yet.

## Starting point and intended claim

`explore` is data scaffolding; the historical prototype is not deployable. M0
Base is the only FlowTransformer-style baseline. M0 Small and M0 Matched are
variants. Every other method is a control or comparator.

Masked reconstruction, EMA teachers, hierarchy, relation bias, and future
latent prediction are prior art. My conditional system claim is:

> Causal endpoint-ego history, identity-free endpoint relations, and hybrid
> semantic masked prediction can improve transferable NetFlow representations
> when labels are scarce and networks shift.

M2-F asks whether later endpoint flows beat M2-H without activity shortcuts; it
is not my contribution. Packet-to-flow distillation needs safe paired captures
and fresh evidence. A held-out family is not a zero-day.

## Hypotheses and evidence levels

| ID | Claim | I reject it when |
|---|---|---|
| H1 | Source-only, identifier-free SSL beats matched scratch/tree at `k=10` labelled groups per class on two unseen networks. | The paired lower bound is not positive, or time, entity, or port-free checks erase it. |
| H2 | H1 holds at a source-fixed threshold with at most 10 false alerts per million flows. | TPR loses or target-test data selected the threshold. |
| H3 | One checkpoint transfers to binary IDS, attack family, and application/service or device work without redesign. | It helps one task family only, or needs a new checkpoint/backbone. |
| H4 | A packet teacher improves a payload-free NetFlow student beyond same-modality controls. | Pairing, privacy, alignment, or external student gain fails. |
| H5 | M2-F improves external low-label transfer beyond passing M2-H and eligible-anchor M2-H. | A shortcut matches it, its interval includes zero, or it fails externally. |

| Claim | Minimum evidence |
|---|---|
| Pretrained encoder | SSL beats parameter-matched scratch in-domain. |
| Transferable encoder | It improves a later or independent network. |
| General-purpose representation | One checkpoint improves three tasks across two families on two unseen domains. |
| System contribution | M2-H beats both constituents and MMAE-NF; M3-Ego has a positive hybrid x ego interaction; M4-Rel adds independent anonymous-relation value; H1/H2 pass on two unseen domains. No constituent is novel alone. |
| Foundation-model claim | General-purpose, low-label, and scaling tests pass without target-test selection. |

NF3 only supports a leakage-aware benchmark claim. Under M0 parity I run the
restricted [classical model check](Model.md#classical-model-check). Failure
there or on external low-label SSL stops the work.

## Datasets and their use

| Dataset | Use | Limit |
|---|---|---|
| NF-UNSW-NB15-v3, NF-BoT-IoT-v3, NF-ToN-IoT-v3 | IDS and attack-family development. | Converted captures are one source, not independent networks. |
| NF-CSE-CIC-IDS2018-v3 | Held-out benchmark. | Inspected, so not sealed. No pretraining, tuning, adaptation, or threshold selection. A sealed claim needs a new uninspected target. |
| Private multi-site | Main unlabelled source and approved benign/calibration data. | Approval does not prove benignness beyond that use. |
| Paired private or public captures | X1-Distill source. | Capture and derivatives stay in one partition. Report pairing and privacy limits. |
| CTU-13, UGR'16, LITNET-2020, MAWI, CESNET-TLS-Year22, CIC-IoT-2022 | Optional external tests or probes. | I report unsupported fields or time order. |

## Exploration and M0

Exploration covers schema, labels, order, missingness, duplicates, fields, and
evaluation. Its [findings](../exp/explore.md) contain no model results. I fit
preprocessing on training only and reset causal state at each boundary.

I run M0 Base first, then M0 Small and M0 Matched as causal checks. SSL starts
only if the [classical model check](Model.md#classical-model-check) permits it.
[The model ladder](Model.md#experiment-sequence) states each SSL signal, run
trigger, and stop trigger.

## Evaluation and leakage checks

The main result is inductive: only source partitions pretrain. Target traffic
cannot influence it. A separate transductive arm may use target training-period
unlabelled traffic, but cannot replace the main result.

I split targets into adaptation, calibration, and final test periods. State
resets at boundaries and test state uses earlier test events only. Exact
duplicates stay in one partition. I use the applicable chronological purge,
endpoint-principal, family/campaign, and capture separation. Target-test
influence on selection, transforms, pretraining, calibration, or thresholds
invalidates the result.

Few-label support is grouped, never sampled by row:
`k={1,5,10,50,100}` independent groups per class. A group is a documented
attack episode/campaign, or a maximal contiguous same-family/entity cluster
whose consecutive gaps do not exceed five minutes.

Benign support is host-hours. I report groups, flows, and analyst minutes.
Validation and
calibration labels count against `k`. At `k={1,5,10}` source data fixes
hyperparameters, calibrator, and threshold. Missing groups are unsupported.

Before screening I fix five pretraining seeds and five support draws. Screening
uses three pretraining seeds; the final uses five. Both use every support draw
and three downstream seeds. Paired 10,000-resample block bootstraps sample days,
weeks, campaigns, hosts, or domains, never flows.
Seeds measure variability, not sample size. Cross-domain summaries weight each
usable domain equally.

| Evaluation | Report |
|---|---|
| Binary and held-out-family IDS | AUPRC, AUROC, TPR at FPR `10^-4`, `10^-3`, and `10^-2`; TPR at 1, 10, and 100 alerts per million; prevalence, precision, alerts/hour, recall, and delay. |
| Attack family | Macro one-vs-rest AUPRC, macro/weighted F1, worst-family recall, coverage, and binary measures. |
| Zero-label anomaly | Benign-only score, AUPRC where labelled, and recall at frozen budgets. Never call it supervised zero-shot. |
| Application/service and device | Frozen linear probe and full fine-tune macro F1 under stated exclusions and groups. |
| Efficiency | Parameters, FLOPs, flows, exposures, labels/minutes, accelerator-hours, checkpoint size, memory, throughput, and p50/p95/p99 completion latency. |

Every alert or fixed-FPR result reports its benign denominator and resolution.
I mark an unresolved point unsupported, never interpolate below one observed
false alert. NLL, Brier, classwise ECE, and reliability plots use calibration
data only. Source-selected calibration stays fixed for zero-label targets.

To continue, TPR at 10 alerts per million may lose at most one absolute point,
classwise ECE may worsen by at most 0.02, no supported frozen probe may lose
more than two macro-F1 points, and p95 CPU plus endpoint state must meet
[Architecture](Architecture.md#encoder-and-inference).

I freeze hardware, batch policy, and harness before the final run. No second
task family means no
general-purpose or foundation-model claim.

## Research questions and stop rules

1. Does SSL beat matched scratch and the best classical control?
2. Does M2-H beat both constituents and MMAE-NF at matched exposure and FLOPs?
3. Does M3-Ego have positive context-main and hybrid x ego interaction effects
   beyond flat, random, time/feature-matched, and CMES grouping?
4. Does M4-Rel beat no-bias and CMES-Causal while passing endpoint-renaming,
   relation-destruction, and endpoint-held-out checks?
5. Does M2-F beat both M2-H controls without corpus, day, eligibility, or
   endpoint-activity shortcuts matching it?
6. Do gains survive time, endpoint, family, and network tests at H2's budget?
7. Does data-first scaling justify its measured cost?
8. Does X1 improve the deployed NetFlow-only student beyond same-modality and
   shuffled-alignment controls?

I follow [Model](Model.md#experiment-sequence) for order and
[Model](Model.md#passing-checks-collapse-checks-and-scaling) for collapse and
scaling checks. Failed additions are removed and reported as negative
ablations. I open final-test labels once, after every choice is fixed.

## Options outside the main plan

| Option | Rule |
|---|---|
| M2-F future latent | Conditional after M2-H. Keep it only if external transfer beats every control. |
| Pooled future-window latent | Excluded until point-horizon evidence exists. |
| Packet-prefix/live scoring | Outside completion-flow scope without active snapshots. |
| Packet teacher | X1-Distill only. The deployed student stays NetFlow-only. |
| GraphSAGE | Comparator only. No graph module unless it wins with the same inputs and tests. |

## One-year schedule

| Time | Work | Decision |
|---|---|---|
| Months 1-2 | Exploration and split rules. | Freeze split rules. |
| Month 3 | M0 Base, Small, and Matched. | Stop if the classical check fails. |
| Months 4-5 | Constituent screen and M1 confirmation. | Stop SSL if neither has external low-label signal. |
| Months 6-7 | M2-H, then separate M2-F and M3-Ego branches. M3-Ego starts from M2-H even if M2-F fails. | Keep a branch only if its controls lose. |
| Month 8 | Relation work and first-year review. | State the positive or negative H1/H2 result. |
| Months 9-10 | Conditional M5 and X1. | Require at least 20% context truncation and weak long-horizon recall for M5; paired data for X1. |
| Month 11 | Scale data before model size. | Stop under Model's doubled-exposure rule. |
| Month 12 | Final replication and reporting. | Use five seeds and make the narrowest supported claim. |

## What to conclude when a test fails

| Outcome | Conclusion |
|---|---|
| Classical parity and restricted-check failure | Sequence modelling and SSL did not help with this portable NetFlow setup. |
| Both SSL constituents fail | SSL did not close the observed domain or label-scarcity gap. |
| M2-F fails | Remove it, keep M2-H, and make no JEPA claim. |
| Hybrid, ego, or relation fails | Remove the mechanism. Do not keep dead decoration. |
| External transfer or H2 fails | At most an in-domain pretrained encoder. |
| H3 probe unavailable | Remove general-purpose and foundation-model language. |
| X1 fails | Omit packet distillation without delaying the NetFlow result. |
| Scaling fails | Keep the smaller model. Do not compensate with a new architecture. |
| NF3-only study | Call it a leakage-aware benchmark study. |

I make only the strongest claim these tests support.
