# Thesis execution roadmap

This file owns claims, hypotheses, corpus roles, evaluation, decision rules, and
the year-one programme. [Model](Model.md) owns model/rung mechanics;
[Architecture](Architecture.md) owns data and runtime contracts; [Refs](Refs.md)
owns evidence. Nothing here asserts that the proposed system already works.

## Starting point and claim boundary

`explore` is a data-exploration and audit scaffold, not a model to retain.
The local historical prototype is ineligible as a deployment or
foundation-model design and survives only as the `historical-recreation`
defined in [Model](Model.md#scope-and-decision-boundary).
[FlowTransformer](Refs.md#flowtransformer2024) is a supervised flow-sequence
baseline and [Anomal-E](Refs.md#anomale) is graph SSL; neither is an existing
Transformer foundation model for unlabeled NetFlow.

Masked reconstruction, EMA teachers, hierarchy, and relation bias are not
individually novel. [MMAE](Refs.md#mmae2026) (March 2026) substantially
anticipates masked teacher--student traffic learning, while
[CMES](Refs.md#cmescrossflow2026) (July 2026) directly anticipates learned
cross-flow relation bias. The narrow contribution under test is:

> Causal endpoint-ego histories, identity-free endpoint relations and hybrid
> semantic masked prediction for transferable NetFlow representations under
> label scarcity and network shift.

A stronger packet-sequence-teacher → NetFlow-only-student contribution is
conditional on lawful paired captures, alignment/privacy evidence, and a
refreshed novelty search. “Better than leading research” means winning under
one data, split, label, parameter, and compute contract. Published headline
scores from incompatible random-row or author-specific splits do not count. A
held-out family is never “zero-day.”

## Falsifiable hypotheses and claim tiers

| ID | Hypothesis | Falsifier |
|---|---|---|
| H1 — low-label transfer | Source-only, identifier-free NetFlow SSL improves binary IDS over the strongest matched scratch/tree comparator at `k=10` labelled groups per class on at least two unseen networks. | The paired lower confidence bound is not positive, or the result vanishes under chronological, entity-disjoint, or port-free evaluation. |
| H2 — operational | H1 survives one frozen threshold at no more than 10 false alerts per million flows. | TPR is inferior at that alert budget or the threshold needs target-test information. |
| H3 — versatility | The same pretrained checkpoint, without task-specific pretraining or backbone redesign, transfers to binary IDS, attack-family, and application/service or device classification. | It helps only one task family or requires a different pretraining checkpoint/backbone per task. |
| H4 — conditional privileged distillation | A packet/payload teacher improves a payload-free NetFlow student on unseen networks beyond same-modality controls. | Pairing, governance, privacy, or external student-gain gates fail. |

| Claim | Minimum evidence |
|---|---|
| Pretrained encoder | SSL beats the parameter-matched scratch model in-domain. |
| Transferable encoder | The result improves a later or independent network. |
| General-purpose representation | One checkpoint improves at least three tasks spanning at least two task families on at least two unseen domains, including frozen-probe evidence where applicable. |
| Novel system contribution | M2-H beats both constituents and MMAE-NF; M3-Ego has a positive hybrid×ego interaction; M4-Rel adds independent anonymous-relation value; H1/H2 pass on two unseen domains. No constituent is claimed as novel alone. |
| Foundation-model claim | The general-purpose criterion passes; low-label adaptation and scaling remain positive; no target-test flow, metadata, prevalence, or label affected selection. |

If only the four converted NF3 benchmarks are available, and their use rights
and field semantics are verified, the study is a benchmark study: it cannot
support operational, broad generalisation, or foundation-model language. If
M0 reaches classical parity, apply only the
[restricted discriminator](Model.md#classical-parity-discriminator); if that
or external low-label SSL fails, stop. A negative result is preferable to a
false foundation-model claim.

## Corpus roles and admissibility

Quality-tier definitions and all lineage/feature/split mechanics belong to
[Architecture](Architecture.md#source-quality-and-capture-lineage). This table sets
scientific roles only.

| Corpus | Scientific role | Admissibility and reporting boundary |
|---|---|---|
| Private multi-site, Q1/Q2 | Main unlabeled source (Q1); approved benign/calibration material only (Q2). | Q0 is excluded; Q1 cannot establish benignness; Q2 does not prove other data are benign. |
| Private Q3 or public paired captures | X1-Distill packet-teacher source. | One original capture and all derivatives remain in one partition; pairing, governance, and privacy gates must pass. |
| NF-UNSW-NB15-v3, NF-BoT-IoT-v3, NF-ToN-IoT-v3 | Controlled IDS/attack-family development. | Current copies are Q0 because source metadata and use rights are incomplete. Exclude them entirely until rights and field semantics resolve; if only non-rights provenance gates remain blocked, permit quarantined benchmark diagnostics outside the ladder. Even after admission, conversions of one capture are one lineage and the three corpora are not three operational domains merely because their schemas match. |
| NF-CSE-CIC-IDS2018-v3 | Conditional sealed NF3 benchmark. | Current copy is Q0. If admitted, no pretraining, selection, threshold, or schema decision may observe it before the sealed run. It remains controlled benchmark evidence, never operational evidence. |
| CTU-13 | Conditional external IDS/attack-family stress test. | Published meter/timeouts, timezone, release-integrity, and redistribution evidence are insufficient for D0, so it is currently Q0. If those gates are resolved, keep captures/campaigns atomic; `Background` remains unknown and never clean benign or one-class fitting material. |
| UGR’16 | Conditional long-horizon operational target. | It is currently Q0 because published licence, timezone, per-exporter lineage, and release-integrity evidence are incomplete. If those gates are resolved, source-only pretraining and selection see no UGR traffic; few-label support uses only the earlier designated block and charges every label to `k`; June is never clean-fitting material without audit; July--August remains sealed with attack-free and attack-period results separated. |
| LITNET-2020 | Conditional external academic-network target. | It is currently Q0: published timeout, timezone, bidirectional-completion, licence, and release-integrity evidence are incomplete. Admit only after those facts, label provenance, and common-field compatibility pass D0; its extra 85-feature view never enters the headline result. |
| MAWI | Candidate Q1 unlabeled SSL source and drift/alert-volume observation. | Keep raw and derived data local under WIDE's research-only terms. Admission requires a bounded capture list, local hashes, one observation domain per trace, a pinned bidirectional meter/configuration, packet-to-flow accounting, and PCAP-epoch ordering. Daily endpoint identities never cross traces. Without those receipts it remains Q0; with them it may support general SSL but not label-based IDS accuracy. |
| CESNET-TLS-Year22 | Conditional month-held-out application/service diagnostic. | Hour-clipped starts, duration-derived ends, and absent row-level probe identity make it Q0 for causal ego pretraining. Until a claim-specific admission path is approved it cannot satisfy H3. For any diagnostic, exclude `TLS_SNI`, `APP`, `CATEGORY`, hostnames, DNS names, certificates, IP/MAC identifiers, timestamps, flow IDs, and capture metadata; audit `TLS_JA3` as a service shortcut. |
| CIC-IoT-2022 | Conditional device probe. | Run only if both device-instance and capture-period separation are possible; exclude MAC/IP and dataset/device identifiers. Otherwise it cannot support H3. |

## Evaluation and leakage contract

All [D0 acceptance evidence](Architecture.md#d0-acceptance-evidence) is a hard
precondition for a claim-bearing ladder result. A Q0 benchmark-only run reports
unsupported gates as blocked and never enters that ladder; `explore` query
output alone does not satisfy either standard.

The primary result is inductive: pretraining uses source partitions only, with
no unlabeled target traffic. A separately reported transductive track may use
only unlabeled target-training-period traffic and never substitutes for the
primary claim.

Within each target, allocate early adaptation, later calibration, and final
chronological test. Reset state at every boundary; test state may include only
earlier events from that same test partition. Exact-record and strict near
groups are hard in every track. The additional isolation factor is
track-specific: time with boundary purge, endpoint principals, complete
family/campaign units, or source/capture lineage. Reject unresolved leakage and
any target-test influence on selection, transforms, pretraining, calibration,
or thresholds.

Few-label support is grouped, not row sampled:
`k={1,5,10,50,100}` independent groups per supported class. A group is a
documented attack episode/campaign, or a maximal contiguous
same-family/entity cluster whose consecutive gaps do not exceed five minutes;
benign supports are host-hours. Report support groups, covered flows, and
measured analyst minutes when adjudication exists.
Any target validation or calibration labels count against `k`; at
`k={1,5,10}`, hyperparameters, calibrator, and alert threshold stay
source-selected. A class without `k` independent groups is unsupported rather
than row-resampled. Unsupported classes remain unsupported. Freeze five
pretraining seeds and five support draws before screening; use the first three
pretraining seeds for screening and all five for final confirmation, with all
five support draws and
three downstream optimization seeds in either phase. Use paired
10,000-resample block bootstraps over days, weeks, campaigns, hosts, or
domains—not flow rows. Average repeated optimization/seed results within each
independent block; seeds measure variability, not sample size. Cross-domain
aggregates give each eligible domain equal weight and never pool its flow rows
with another domain.

| Evaluation family | Required outcomes |
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
temperature scaling use calibration only. Keep a source-selected
threshold/calibrator fixed for zero-label target results. The five seed suites
and 10,000-resample bootstrap are mandatory; nonparametric dataset tests are
supplementary only when enough independent domains exist.

Promotion also requires all precommitted non-inferiority limits: TPR at 10
false alerts per million may fall by at most one absolute point; classwise ECE
may worsen by at most 0.02; no supported application/device frozen probe may
lose more than two macro-F1 points; completion inference must meet the p95 CPU
and endpoint-state limits in
[Architecture](Architecture.md#encoder-and-inference-envelope). Freeze the
target CPU, accelerator, batch policy, and measurement harness in WP0. An
unavailable second task family does not receive a pass; it removes the
general-purpose and foundation-model claims.

## Research questions and decision rules

1. Does SSL beat the identical scratch model and best classical model under
   fixed field, split, label, and compute budgets?
2. Does raw-plus-latent beat both constituents and MMAE-NF at matched exposure
   and FLOPs?
3. Does causal ego context yield positive main and SSL×ego interaction effects
   beyond flat, random, time/feature-matched, and CMES grouping?
4. Do anonymous directed relations add information under endpoint renaming and
   relation-destruction controls?
5. Do gains survive future-time, endpoint-disjoint, held-out-family, and
   cross-network tests at H2’s alert budget?
6. Does data-first, then parameter, scaling justify measured cost?
7. Conditional on paired captures, does X1-Distill improve the deployed NetFlow-only
   student beyond same-modality and shuffled-alignment controls?

Use the exact component gates, collapse screens, rungs, comparators, and scale
policy in [Model](Model.md#evidence-ladder) and
[Model](Model.md#promotion-collapse-and-scaling). Failed additions are deleted
from the promoted stack and retained as negative ablations. Open final-test
labels exactly once after freezing all choices.

## Excluded or conditional alternatives

| Candidate | Decision | Reason |
|---|---|---|
| `d=256` compact Transformer | S0 mechanism screen only. | The canonical comparison is the Model 25M backbone; compact screening cannot establish the final claim. |
| Next-60-second forecasting | M2-F control/candidate only. | It is not inherited without external downstream value beyond shuffled-future and raw-next controls. |
| Packet-prefix/live scoring | Outside the core completion-flow thesis. | Admit only when active flow-snapshot records make causal availability auditable; otherwise do not claim live detection. |
| Packet/payload teacher | X1-Distill only. | X1-Distill is privileged training with a NetFlow-only student and independent data, privacy, alignment, and novelty gates. |
| GraphSAGE relational extension | Challenger only. | No graph module enters the promoted architecture unless it beats the fixed causal Transformer under the same contract. |
| 1.5M→6M→40M scale plan | Not canonical. | Use Model’s S0→S4 data-first ladder; compact alternatives can screen mechanisms but cannot change scaling claims. |

## Work packages and one-year schedule

| Time | Work packages | Exit decision |
|---|---|---|
| Months 1–2 | WP0 evidence and target-hardware freeze; WP1 corpus/lineage audit; WP2 evaluation freeze. | Stop the claim-bearing ladder if provenance, duplicate isolation, feature semantics, causal availability, or split replay are unauditable; retain at most the quarantined Q0 benchmark track. |
| Month 3 | WP3 historical/classical controls and M0. | Apply the exact [classical-parity discriminator](Model.md#classical-parity-discriminator); stop all Transformer/SSL progression if it fails. |
| Months 4–5 | WP4 constituent screen and M1-R/M1-L confirmation. | Stop SSL if neither constituent has external low-label signal. |
| Months 6–7 | WP5 hybrid/future; WP6 ego factorial/confirmation. | Retain only hybrid evidence beyond constituents; reject ego if matched history suffices. |
| Month 8 | WP7 relation study and year-one acceptance across binary, family, and available H3 probes. | Year-one result is a clean positive/negative H1/H2 and hybrid/ego answer, not optional-extension inflation. |
| Months 9–10 | WP8 conditional hierarchy and X1-Distill preparation/experiment. | M5-Hier requires observed truncation; X1-Distill requires Q3 paired data and refreshed novelty evidence. |
| Month 11 | WP9 data-first scaling of the promoted stack. | Stop under the Model double-exposure failure rule; graph/prefix work cannot rescue failure. |
| Month 12 | WP10 sealed replication; WP11 reporting. | Five-seed final confirmation, immutable artifacts, uncertainty report, narrowest claim tier. |

Year-one acceptance requires H1 on two independent held-out domains, H2 at the
frozen 10-alert-per-million threshold, a result for both SSL constituents and
their combination, and every available task probe reported with its limits.
H3/X1-Distill are not claimed merely because a benchmark exists.

## Negative outcomes and final statement

| Outcome | Required conclusion |
|---|---|
| Classical parity and restricted discriminator failure | Sequence modelling and SSL were not justified under the tested portable NetFlow contract. |
| Both SSL constituents fail | SSL did not overcome the observed domain or label-scarcity gap. |
| Hybrid, ego, or relation fails | Remove the mechanism; do not keep it as decoration. |
| Operational or external transfer fails | At most an in-domain pretrained encoder; no foundation or operational claim. |
| H3 probe unavailable | Remove general-purpose and foundation-model language. |
| Q3/X1-Distill fails | Omit packet distillation without delaying the NetFlow result. |
| Scaling fails | Retain the smaller promoted model; do not compensate with repeated exposure or a new architecture. |
| NF3-only/provenance-limited study | Describe a leakage-aware benchmark study only. |

The final thesis claim is the highest supported tier above—never stronger—and
links component evidence to [Model](Model.md), implementation invariants to
[Architecture](Architecture.md), and novelty/reproduction evidence to
[Refs](Refs.md#prior-art-and-reproduction-boundaries).
