# Thesis research plan

This file sets out the claims, hypotheses, datasets, evaluation, stop rules,
and first-year schedule. [Model](Model.md) describes the models and experiments.
[Architecture](Architecture.md) describes the data and inference setup.
[Refs](Refs.md) records the evidence from earlier work. The proposed system has
not yet been shown to work.

## Starting point and intended claim

`explore` is only a data-exploration scaffold. The local historical prototype
is neither a deployment model nor a foundation-model design. It remains only
as the `historical-recreation` described in
[Model](Model.md#what-this-file-covers).
[FlowTransformer](Refs.md#flowtransformer2024) is a supervised flow-sequence
baseline and [Anomal-E](Refs.md#anomale) is graph SSL; neither is an existing
Transformer foundation model for unlabeled NetFlow.

Masked reconstruction, EMA teachers, hierarchy, and relation bias already
exist. [MMAE](Refs.md#mmae2026) (March 2026) covers much of masked
teacher--student traffic learning. [CMES](Refs.md#cmescrossflow2026) (July
2026) already uses learned cross-flow relation bias. This thesis tests the
narrower combination:

> Causal endpoint-ego histories, identity-free endpoint relations and hybrid
> semantic masked prediction for transferable NetFlow representations under
> label scarcity and network shift.

M2-F future-latent prediction is a conditional JEPA-family comparison, closest
in causal next-latent form to [LeNEPA](Refs.md#lenepa). It is not the thesis
contribution. [I-JEPA](Refs.md#ijepa) and [data2vec](Refs.md#data2vec) already
establish the broader predictive-representation family. M2-F asks only whether
the objective improves external low-label transfer after M2-H while ruling out
corpus, day, target-eligibility, and endpoint-activity shortcuts.

A packet-sequence teacher with a NetFlow-only student is a possible stronger
result, but only if paired captures, safe alignment, and an updated literature
search support it. Comparisons with earlier research count only when data,
splits, labels, parameters, and compute match. Headline scores from incompatible
random-row or author-specific splits do not count. A held-out family is not a
“zero-day.”

## Hypotheses and evidence levels

| ID | Hypothesis | What would disprove it |
|---|---|---|
| H1 — low-label transfer | Source-only, identifier-free NetFlow SSL improves binary IDS over the strongest matched scratch/tree comparator at `k=10` labelled groups per class on at least two unseen networks. | The paired lower confidence bound is not positive, or the result vanishes under chronological, entity-disjoint, or port-free evaluation. |
| H2 — operational | H1 survives one threshold fixed before target testing at no more than 10 false alerts per million flows. | TPR is inferior at that alert budget or the threshold needs target-test information. |
| H3 — versatility | The same pretrained checkpoint, without task-specific pretraining or backbone redesign, transfers to binary IDS, attack-family, and application/service or device classification. | It helps only one task family or requires a different pretraining checkpoint/backbone per task. |
| H4 — conditional privileged distillation | A packet/payload teacher improves a payload-free NetFlow student on unseen networks beyond same-modality controls. | Pairing, privacy, or external student-gain checks fail. |
| H5 — conditional future latent | M2-F improves external low-label transfer beyond the M2-H model that passed earlier checks and eligible-anchor M2-H instead of learning corpus, day, eligibility, or endpoint-activity patterns. | Its controls match it, its incremental transfer confidence interval includes zero, or it fails on unseen networks. |

| Claim | Minimum evidence |
|---|---|
| Pretrained encoder | SSL beats the parameter-matched scratch model in-domain. |
| Transferable encoder | The result improves a later or independent network. |
| General-purpose representation | One checkpoint improves at least three tasks spanning at least two task families on at least two unseen domains, including frozen-probe evidence where applicable. |
| Novel system contribution | M2-H beats both constituents and MMAE-NF; M3-Ego has a positive hybrid×ego interaction; M4-Rel adds independent anonymous-relation value; H1/H2 pass on two unseen domains. No constituent is claimed as novel alone. |
| Foundation-model claim | The general-purpose criterion passes; low-label adaptation and scaling remain positive; no target-test flow, metadata, prevalence, or label affected selection. |

The four supplied NF3 datasets are benchmark inputs. Results limited to NF3
make a benchmark study, not evidence for live operation or a foundation model.
If M0 only matches the classical models, run the restricted
[classical model check](Model.md#classical-model-check). If that check or the
external low-label SSL test fails, stop. A clear negative result is better than
an inflated claim.

## Datasets and their use

| Dataset | How it is used | What it cannot show |
|---|---|---|
| NF-UNSW-NB15-v3, NF-BoT-IoT-v3, NF-ToN-IoT-v3 | Controlled IDS and attack-family development. | Different conversions of the same capture count as one source; matching schemas do not make them operational domains. |
| NF-CSE-CIC-IDS2018-v3 | Held-out public benchmark after Exploration. | Its labels and distributions have been inspected, so it is not sealed. It receives no pretraining, model selection, threshold tuning, or adaptation. A genuinely sealed claim requires a new uninspected target. |
| Private multi-site | Main unlabeled source and approved benign/calibration material. | Does not establish benignness beyond the approved use. |
| Private or public paired captures | X1-Distill packet-teacher source. | Keep one original capture and all derivatives in one partition; report pairing and privacy limits. |
| CTU-13, UGR’16, LITNET-2020, MAWI, CESNET-TLS-Year22, CIC-IoT-2022 | Optional external stress tests or task probes. | Use only when their fields and time/order support the stated split; otherwise report them as unavailable rather than forcing a claim. |

## Exploration and M0

Exploration is complete. It covers schema and labels, time order, missingness,
duplicates, field selection, and evaluation choices. The
[findings](../exp/explore.md) contain no model results.

Before training, code under `src` builds reproducible splits, fits preprocessing
on training data only, and checks causal state and context at every split. See
[Exploration](Architecture.md#exploration) and [M0](Architecture.md#m0).

M0 then runs three supervised references under the same fields and evaluation:
M0 Base (25M unrestricted FlowTransformer-style model), M0 Small (eight-flow
causal), and M0 Matched (25M causal matched to Base except mask). Base is a
PyTorch model, not the original implementation. Causality counts as useful only
if the matched comparison shows it. Any later reference to M0 means M0 Matched.
The 25M S0→S4 sequence is for SSL work that passes the earlier checks.

## Evaluation and leakage checks

The main result is inductive: pretraining uses source partitions only and sees
no target traffic, labelled or unlabelled. A separate transductive result may
use unlabelled traffic from the target's training period, but it cannot replace
the main result.

Split each target chronologically into early adaptation, later calibration,
and final test periods. Reset state at every boundary. Test state may contain
only earlier events from the same test partition. Keep exact duplicate records
in one partition. Report near-duplicate counts from Exploration, but do not use
them to define a split. Depending on the evaluation, also separate by time with
a boundary purge, endpoint principal, complete family or campaign, or dataset
or capture source. If target-test data affects model choice, transforms,
pretraining, calibration, or thresholds, the result is invalid.

Few-label support uses groups, not sampled rows:
`k={1,5,10,50,100}` independent groups per class. A group is a
documented attack episode/campaign, or a maximal contiguous
same-family/entity cluster whose consecutive gaps do not exceed five minutes;
benign supports are host-hours. Report support groups, covered flows, and
measured analyst minutes when adjudication exists.
Any target validation or calibration labels count against `k`. At
`k={1,5,10}`, choose the hyperparameters, calibrator, and alert threshold from
source data only. If a class lacks `k` independent groups, report it as
unsupported instead of resampling rows. Unsupported classes remain
unsupported.

Choose five pretraining seeds and five support draws before screening. Use the
first three pretraining seeds during screening and all five for the final run.
Use all five support draws and three downstream optimization seeds in both
phases. Use paired 10,000-resample block bootstraps over days, weeks, campaigns,
hosts, or domains—not flow rows. Average repeated optimization/seed results within each
independent block; seeds measure variability, not sample size. Cross-domain
aggregates give each usable domain equal weight and never pool rows from
different domains.

| Evaluation | Report |
|---|---|
| Binary and held-out-family IDS | AUPRC, AUROC, TPR at FPR `10^-4`, `10^-3`, and `10^-2`, and TPR at 1, 10, and 100 false alerts per million flows; report prevalence, precision, alerts/hour, campaign recall, and detection delay. H2 uses 10 alerts per million. |
| Attack-family IDS | Macro one-vs-rest AUPRC, macro/weighted F1, worst-family recall, supported-class coverage, and binary operational measures. |
| Zero-label anomaly | Benign-only score, AUPRC where labels exist, and recall at frozen alert budgets; never a supervised “zero-shot” claim. |
| Application/service and device | Frozen linear probe and full fine-tune macro F1, under the stated corpus exclusions/groups. |
| Efficiency | Parameters, training/adaptation FLOPs, flows, exposures, labelled groups/minutes, accelerator-hours, checkpoint size, memory, throughput, and p50/p95/p99 completion latency. |

For every fixed-FPR or alert-budget point, report the benign denominator and
achievable empirical resolution. Mark an operating point unsupported when the
test cannot resolve it; do not interpolate a claim below one observed false
alert.

Report NLL, Brier score, classwise ECE, and reliability plots. Thresholds and
temperature scaling use calibration data only. Keep the source-selected
threshold and calibrator fixed for zero-label target results. Run all five seed
sets and the 10,000-resample bootstrap. Add nonparametric dataset tests only
when there are enough independent domains.

To continue, a model must also stay within these limits: TPR at 10 false alerts
per million may fall by at most one absolute point; classwise ECE may worsen by
at most 0.02; no supported application or device frozen probe may lose more
than two macro-F1 points; and completion inference must meet the p95 CPU and
endpoint-state limits in [Architecture](Architecture.md#encoder-and-inference).
Choose the target CPU, accelerator, batch policy, and measurement harness
before the final run and do not change them afterward. If a second task family
is unavailable, do not make a general-purpose or foundation-model claim.

## Research questions and stop rules

1. Does SSL beat the identical scratch model and best classical model under
   fixed field, split, label, and compute budgets?
2. Does raw-plus-latent beat both constituents and MMAE-NF at matched exposure
   and FLOPs?
3. Does causal ego context yield positive main and SSL×ego interaction effects
   beyond flat, random, time/feature-matched, and CMES grouping?
4. Do anonymous directed relations add information under endpoint renaming and
   relation-destruction controls?
5. Does M2-F future-latent prediction improve external low-label results beyond
   the M2-H model that passed earlier checks and eligible-anchor M2-H, or does
   it only learn corpus, day, eligibility, and endpoint-activity patterns?
6. Do gains survive future-time, endpoint-disjoint, held-out-family, and
   cross-network tests at H2’s alert budget?
7. Does scaling the data before the model justify its measured cost?
8. With paired captures, does X1-Distill improve the deployed NetFlow-only
   student beyond same-modality and shuffled-alignment controls?

Follow the experiment sequence and collapse and scaling checks in
[Model](Model.md#experiment-sequence) and
[Model](Model.md#passing-checks-collapse-checks-and-scaling). Remove additions
that fail, and report them as negative ablations. Open final-test labels exactly
once, after every choice is fixed.

## Options outside the main plan

| Option | Use | Reason |
|---|---|---|
| `d=256` compact Transformer | S0 mechanism screen only. | The main comparison uses the 25M model; a small screening model cannot establish the final result. |
| M2-F JEPA-family future latent | Conditional branch after M2-H. | Predictive-latent learning already exists. Keep this branch only if it adds external low-label value beyond its controls. |
| Next-60-second raw aggregate forecast | M2-F control only. | It checks whether future-latent prediction does more than forecast aggregate activity. |
| Pooled future-window latent | Leave out. | Pooling and event count add another confound before point-horizon evidence exists. |
| Context length `W={3,8,20}` | One fixed source-validation ablation, with the target included in `W`. | It tests length, not a new method. Do not combine it with M2-F or replace the default context without external evidence. |
| Time-gap sessionisation | Training-only diagnostic, followed by one isolated ablation only if the pattern is stable. | A source-derived gap rule may expose burst structure, but it must not become a dataset or session identifier. |
| Conversation and source-host context | Matched controls for M3-Ego. | Test one builder at a time with the same history budget. If either matches endpoint-ego, reject the endpoint-ego contribution. Do not combine views or add a second encoder. |
| Packet-prefix/live scoring | Outside the core completion-flow thesis. | Use only when active flow-snapshot records make causal availability checkable; otherwise do not claim live detection. |
| Packet/payload teacher | X1-Distill only. | The student must remain NetFlow-only, with independent checks for data, privacy, alignment, and earlier work. |
| GraphSAGE relational extension | Comparison only. | Do not add a graph module unless it beats the fixed causal Transformer under the same setup. |
| 1.5M→6M→40M scale plan | Leave out. | Use Model's S0→S4 data-first sequence. Small models may screen ideas but cannot change the scaling result. |

## One-year schedule

| Time | Work | Decision |
|---|---|---|
| Months 1–2 | Exploration: schema/labels, time order, missingness, duplicates, field selection, and evaluation decisions. | Record benchmark limits; do not overstate the result. |
| Month 3 | M0 under `src`: splits, training-only preprocessing, causal-state and context checks, M0 Base (25M unrestricted FlowTransformer-style model), M0 Small (eight-flow causal model), and M0 Matched (25M causal model matched to Base except for the mask). Any later M0 reference means M0 Matched. | Compare all three under the same setup. Apply the [classical model check](Model.md#classical-model-check) and stop Transformer and SSL work if it fails. |
| Months 4–5 | Constituent screen and M1-R/M1-L confirmation. | Stop SSL if neither constituent has external low-label signal. |
| Months 6–7 | Confirm the hybrid, then test M2-F future latent and M3-Ego as separate branches. M3-Ego starts from M2-H whether or not M2-F survives. Keep length, time-gap, conversation, and source-host checks separate. | Keep M2-F only if it adds external low-label value. Reject ego if a matched control is enough. If both branches pass, their combination must beat each branch before continuing. |
| Month 8 | Study relationships and review the first-year results across binary, family, and available H3 probes. | Report a positive or negative H1/H2 and hybrid/ego result. |
| Months 9–10 | Conditional hierarchy and X1-Distill preparation/experiment. | M5-Hier requires observed truncation; X1-Distill requires paired data and refreshed novelty evidence. |
| Month 11 | Scale data before scaling the model. | Stop under the Model double-exposure failure rule; graph or prefix work cannot rescue a failure. |
| Month 12 | Sealed replication and reporting. | Run all five seeds, save configurations and results, report uncertainty, and make only the narrowest supported claim. |

The first year succeeds only if H1 holds on two independent held-out domains,
H2 holds at the fixed 10-alert-per-million threshold, both SSL objectives and
their combination have results, and every available task probe is reported
with its limits. A benchmark alone does not establish H3 or X1-Distill.

## What to conclude when a test fails

| Outcome | Conclusion |
|---|---|
| Classical parity and restricted check failure | Sequence modelling and SSL did not help under the tested portable NetFlow setup. |
| Both SSL constituents fail | SSL did not overcome the observed domain or label-scarcity gap. |
| M2-F future-latent prediction fails | Remove it, keep M2-H, and make no JEPA claim. |
| Hybrid, ego, or relation fails | Remove the mechanism; do not keep it as decoration. |
| Operational or external transfer fails | At most an in-domain pretrained encoder; no foundation or operational claim. |
| H3 probe unavailable | Remove general-purpose and foundation-model language. |
| X1-Distill fails | Omit packet distillation without delaying the NetFlow result. |
| Scaling fails | Keep the smaller model; do not compensate with repeated exposure or a new architecture. |
| NF3-only study | Describe a leakage-aware benchmark study only. |

The final thesis should make only the strongest claim supported by these tests,
and no stronger. Link the model evidence to [Model](Model.md), the data and
runtime rules to [Architecture](Architecture.md), and the earlier-work and
reproduction notes to [Refs](Refs.md#closest-work-and-reproduction-status).
