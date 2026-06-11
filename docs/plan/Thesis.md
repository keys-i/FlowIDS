# Evaluation plan

This is the evaluation plan for the later research ladder, not the M0 baseline.
M0 uses one NF3 dataset and a chronological 70/15/15 split; see the
[repository README](../../README.md#m0-baseline). The ladder compares
[models](Model.md) with the same label budget and false-alert limits.

Here, *scratch* has no pretraining, *frozen* keeps the encoder fixed for the
target task, and *fine-tuning* updates it with the allowed labels.

| Claim | Evidence needed |
|---|---|
| Pretrained encoder | Beats parameter-matched scratch in-domain |
| Transferable encoder | Improves a later or independent network |
| Combined method | Each addition passes its [matched comparison](Model.md#experiment-sequence), including greater benefit from pretraining with endpoint history; the thresholds below pass on two unseen networks |
| General-purpose representation | One checkpoint helps binary IDS, attack-family, and application/service or device tasks across two task families and two unseen domains |

Foundation-model claims also require few-label and scaling evidence, with no
test data used to choose the model. A held-out attack family alone does not
establish detection of new zero-days.

## Datasets and their use

| Dataset | Use | Limit |
|---|---|---|
| NF-UNSW-NB15-v3, NF-BoT-IoT-v3, NF-ToN-IoT-v3 | IDS and attack-family development | Different benchmark captures do not prove independent operational networks |
| NF-CSE-CIC-IDS2018-v3 | Held-out benchmark | Already inspected: no pretraining, tuning, adaptation, or threshold selection; a sealed claim needs a new uninspected target |
| Private multi-site traffic | Unlabelled source and approved benign/calibration data | Access needs confirmation; permission to use data does not establish that it is benign |
| Paired private or public captures | Packet-teacher comparison | A capture and its derivatives stay in one partition; report pairing and privacy limits |
| CTU-13, UGR'16, LITNET-2020, MAWI, CESNET-TLS-Year22, CIC-IoT-2022 | Optional external tests or probes | Report missing fields or unsuitable time order |

## Evaluation and leakage checks

The main result is *inductive*: pretraining uses source partitions only. A
separate transductive experiment may use unlabelled target training-period
traffic, but cannot replace the main result.

Split target traffic into adaptation, calibration, and final test periods.
Keep exact duplicates together. Apply the relevant chronological purge,
family/campaign, and capture separation. [Architecture](Architecture.md#context-and-endpoint-relationships)
defines history resets and endpoint-disjoint splitting. Test data must not choose transforms,
pretraining, calibration, thresholds, or models. Open final-test labels once,
after those choices are fixed.

<details>
<summary>Label budgets, repeated runs, and reported metrics</summary>

Few-label support is grouped rather than sampled by row:
`k={1,5,10,50,100}` independent groups per class. A group is a documented
attack episode or campaign, or a maximal same-family/entity cluster with gaps
of at most five minutes. Benign support is host-hours. Report groups, flows,
and analyst minutes. Validation and calibration labels count against `k`; at
`k={1,5,10}`, source data fixes hyperparameters, calibrator, and threshold.
An unavailable group is unsupported.

Fix five pretraining seeds and five support draws before screening. Screening
uses three pretraining seeds; the final result uses five. Both use every
support draw and three downstream seeds. Use paired 10,000-resample block
bootstraps over days, weeks, campaigns, hosts, or domains, never individual
flows. Training seeds measure run variation, not independent data. Weight
each usable domain equally in cross-domain summaries.

FPR is the fraction of benign flows incorrectly flagged.

Report AUPRC, AUROC, and TPR at FPR `10^-4`, `10^-3`, and `10^-2`. Also report TPR at 1, 10, and
100 alerts per million flows, prevalence, precision, alerts per hour, recall,
and delay. For attack families, add macro one-vs-rest AUPRC, macro/weighted
F1, worst-family recall, and coverage. For application/service and device
tasks, report frozen linear probes and full fine-tuning macro F1 under the
stated exclusions and groups. Zero-label anomaly tests use benign-only scores,
AUPRC where labels exist, and recall at fixed budgets; they are not supervised
zero-shot tests. Report
parameters, FLOPs, flows, exposures, labels/minutes, accelerator-hours,
checkpoint size, memory, throughput, and p50/p95/p99 completion latency.

Every fixed-alert or fixed-FPR result names its benign denominator and
resolution. Do not interpolate below one observed false alert. NLL, Brier,
classwise expected calibration error (ECE), and reliability plots use calibration data only; source-selected
calibration stays fixed for zero-label targets. Freeze hardware, batch policy,
and the measurement harness before the final run.

</details>

## What counts as an improvement

AUPRC summarises the trade-off between detecting attacks and alert precision.
TPR is the fraction of attacks detected. Macro scores give each supported
attack family equal weight. One AUPRC point is `0.01`.

The label-efficiency area combines scores across label budgets: use the
trapezoidal area under score versus `log10(k)`, normalized by that log span.

Before M0 is used to justify later pretraining work, the classical check must
pass: M0 Matched beats the best classical control by 1.0 macro one-vs-rest
AUPRC point at `k=10` labelled support groups per class, the paired
hierarchical-bootstrap 95% lower bound is above zero, and TPR at 10 false
alerts per million loses at most one absolute point.

Any later stage must pass all three checks:

- Gain 1.0 absolute macro-AUPRC point at `k=10` and in the fixed
  label-efficiency area, beating the best model with the same inputs.
- Improve on two independent targets, with the paired hierarchical-bootstrap
  95% lower bound above zero.
- Lose at most one macro-AUPRC point on any target and two recall points on
  any critical attack family.

To continue, TPR at 10 alerts per million may lose at most
one absolute point, classwise ECE may worsen by at most `0.02`, no supported
frozen probe may lose more than two macro-F1 points, and p95 CPU plus endpoint
state must meet [Architecture's inference target](Architecture.md#encoder-and-inference).

## If a test fails

Remove the failed addition and make the narrowest claim supported by the
result. Classical parity plus failure of the [restricted S0 check](Model.md#classical-model-check)
means sequence modelling and SSL did not help this flow setup. If transfer fails, report an in-domain pretrained encoder
only. If a task family is unavailable, omit general-purpose language. A failed
packet teacher does not delay the NetFlow result. An NF3-only study is a
leakage-aware benchmark study.
