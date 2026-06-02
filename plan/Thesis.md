# Thesis execution roadmap

**Causal endpoint-ego histories, identity-free endpoint relations and hybrid semantic masked prediction for transferable NetFlow representations under label scarcity and network shift.**

This is an execution roadmap, not a chapter outline. [FlowTransformer](Refs.md#flowtransformer2024) is supervised and [Anomal-E](Refs.md#anomale) is graph SSL; neither is a Transformer foundation model for unlabeled NetFlow. The work does not claim novelty for masking, EMA, hierarchy, or relation bias separately. [MMAE](Refs.md#mmae2026) (March 2026) and [CMES](Refs.md#cmescrossflow2026) (July 2026) are claim collisions. The stronger optional packet-sequence-teacher → NetFlow-only-student contribution requires paired PCAP/NetFlow data and a refreshed novelty search. “Better than leading” requires identical data, split, labels, parameters, and compute. If classical models still match the Transformer after the one predeclared small SSL discriminator, or no external low-label SSL gain is established, stop. A negative result is preferable to a fraudulent foundation-model claim. With only the four NF3 datasets and no auditable pretraining provenance, report a benchmark study—not a foundation-model claim.

## Claim boundary

Use this claim hierarchy, in order:

| Claim | Minimum evidence |
|---|---|
| Pretrained | In-domain pretraining beats the same scratch model. |
| Transferable | Gain holds on a later or independent network. |
| General-purpose | Frozen representation improves two task families across two unseen networks. |
| Foundation | General-purpose evidence, positive scaling evidence, and no target-test exposure. |

The only possible novel system claim is the interaction of causal endpoint-ego context, hybrid semantic masked prediction, and identity-free directed relations, and only if the interaction gates in [Model](Model.md#evidence-ladder) pass. Do not claim the first cross-flow method, teacher method, or relation method. A held-out family is not zero-day.

No SOTA claim or direct score ranking is permitted without rerunning the comparator under the same contract. Keep prior-art, date, modality, and reproducibility evidence in [Refs](Refs.md#prior-art-and-reproduction-boundaries); keep architecture only in [Model](Model.md) and [Architecture](Architecture.md).

## Research questions and decision rules

1. Does SSL outperform the identical scratch model under the matched evaluation contract?
2. Do raw-plus-latent targets outperform raw-only, latent-only, constituent-only, and MMAE-style controls?
3. Does causal endpoint-ego history outperform flat, random, time-feature-matched/statistical-similarity, and CMES grouping contexts?
4. Do directed anonymous relations outperform no relation and CMES relations under causal context?
5. Do gains survive future-time, endpoint-disjoint, held-out-family, and cross-network evaluation?
6. Does greater data or model scale improve transfer enough to justify compute and deployment cost?
7. Does a packet teacher add value only when a semantically compatible packet corpus and audit permit the conditional distillation track?

Apply the promotion and collapse thresholds exactly as specified in [Model](Model.md#promotion-collapse-scaling), including the paired hierarchical-bootstrap lower confidence bound, no-target-loss, critical-family, TPR, calibration, and same-input-comparator gates. A module that fails its gate is retained as a negative result and not promoted. Run sealed evaluation once, after all choices are frozen.

## Data and split contract

Controlled development datasets are NF-UNSW-NB15-v3, NF-BoT-IoT-v3, and NF-ToN-IoT-v3; NF-CSE-CIC-IDS2018-v3 is sealed. Foundation pretraining may use only provenance-audited private and eligible public operational traffic, with no sealed-target lineage and no synthetic core. UGR’16 and LITNET-2020 are external only if fields are compatible. For UGR’16, pretrain on March–April, reserve May for calibration, exclude June from benign-only fitting unless audited, and seal July–August; report attack-free and attack-period results separately. A separate application/service dataset must be converted by the same documented meter; otherwise drop the general-purpose claim. Treat all conversions of one capture as one lineage.

Freeze and version: capture lineage, meter/conversion, fields, labels, taxonomy map, time zone, exclusions, endpoint derivation, split manifests, preprocessing fit partition, and pretraining exposure ledger. Do not use target-test flows, prevalence, or labels for tuning, calibration, thresholding, model choice, or pretraining.

Run these separate tracks:

- Blocked chronological development and rolling-future evaluation.
- Endpoint-disjoint evaluation.
- Held-out family with campaign/day metadata; call it held-out family, never zero-day.
- Source-to-target transfer with zero target labels, then 1/5/10/100/full labels from independent target train-time support groups per class.
- Frozen and full fine-tuning; use five support draws and report validation/calibration counts. At 1/5/10 labels, keep hyperparameters and thresholds source-selected; count any target calibration label against the budget. Mark unsupported classes rather than imputing them.
- Benign-only anomaly detection at zero labels.
- A separately labeled transductive track using only unlabeled target training-period flows, never test-period flows.

## Evaluation contract

Primary outcomes are binary AUPRC, macro one-vs-rest AUPRC, and TPR at FPR 1e-4, 1e-3, and 1e-2. Secondary outcomes are per-family precision/recall/F1, macro metrics, MCC, ROC-AUC, alerts per 100k flows, campaign recall, and detection delay. Report NLL, Brier score, classwise ECE, and reliability diagrams for calibration. Fit thresholds and temperature scaling on validation only; freeze the source-validation calibrator and threshold across zero-label targets.

Report parameters, FLOPs, unique flows, exposures, accelerator-hours, memory, throughput, and p50/p95/p99 latency. The statistical unit is day, entity, campaign, or source—not flow rows or random seeds. Use paired hierarchical block bootstrap; use dataset-level nonparametric tests only when enough independent domains exist.

## Comparator and reproduction policy

Publish three distinct tables:

1. **Author-reported literature:** no rankings or direct score comparisons.
2. **Reproductions:** MMAE only in its exact packet modality; CMES only when reproducible, with original and strict split results separated.
3. **Matched mechanisms:** MMAE-NF, CMES-25M, CMES-Causal, and the proposed model under one data/split/label/parameter/compute contract.

MMAE’s recorded facts are first 5 × 320 bytes, cross-flow support corruption, EMA/reconstruction/alignment, and a stated 8:1:1 split whose temporal/entity grouping is unspecified. CMES’s recorded facts are feature sorting/grouping, four bits yielding 16 relation types, bidirectional frozen 8B supervised training, and cross-dataset accuracy/F1 reporting. These facts motivate RQ2–RQ4; they do not license first-method claims or direct score comparisons. MMAE-NF is an adaptation, not a reproduction. The full comparator definitions remain in [Model](Model.md#comparator-contract).

## Work packages and gates

| Package | Deliverable | Proceed only if |
|---|---|---|
| WP0 Evidence freeze | MMAE/CMES claim matrix, dated sources, reproduction feasibility, terminology and claim exclusions | Collision and comparator contract are frozen. |
| WP1 Data audit | Lineage/provenance ledger, conversion parity, label/split manifest, external-field audit | Sealed lineage is excluded; operational pretraining provenance is auditable. |
| WP2 Evaluation freeze | Tracks, label-support sampler, metrics, calibration/threshold policy, statistical plan | No target-test information can affect selection. |
| WP3 Classical and M0 | Logistic/CatBoost/MLP and supervised flat causal Transformer on matched splits | If M0 does not clear classical parity, permit only the predeclared small M1 discriminator; stop unless SSL then beats both. |
| WP4 Constituents | Raw and latent studies on flat context; raw-next is a future-objective control | If both M1-R and M1-L fail, stop SSL progression; otherwise the surviving constituent may enter WP5. |
| WP5 Hybrid/future | Frozen hybrid; future target only as optional promotion | Future beats both controls externally; otherwise keep hybrid only. |
| WP6 Ego | Causal ego against flat/random/time-matched/statistical-similarity/CMES grouping | Ego gain and hybrid×ego interaction lower CI exceed zero. |
| WP7 Relations | Anonymous directed relations, renaming invariance, marginal-preserving destruction | Relations beat no-relation and CMES-relation controls. |
| WP8 Conditional hierarchy/distill | Hierarchy; packet teacher only after modality and provenance audit | Earlier gates pass; packet teacher adds matched external value. |
| WP9 Data-first scaling | Exposure/source scaling at fixed architecture before larger models | Positive transfer scaling; stop after the two failed doubled-exposure rule in [Model](Model.md#promotion-collapse-scaling). |
| WP10 Sealed evaluation | One evaluation of untouched NF-CSE-CIC-IDS2018-v3 and eligible external networks/tasks, followed by the final uncertainty report | All choices, controls, and thresholds are frozen. |
| WP11 Narrow outcome | Claim-tier decision and negative-ablation report | Claim never exceeds demonstrated evidence. |

## Precommitted negative outcomes

Report, rather than rescue, these outcomes: trees at least match M0; constituents fail; hybrid fails; MMAE-NF matches the proposal; CMES grouping matches ego; CMES relations match proposed relations; M3/M4 fail; scaling fails; external transfer fails; a second task family is absent; or provenance is absent. Each outcome narrows the conclusion to the highest supported rung in the claim hierarchy.

## Final reporting boundary

State whether evidence supports only an in-domain pretrained model, transferable representation, general-purpose representation, or a foundation-model candidate. Do not call four NF3-only results a foundation model. Link experiment decisions to [Model](Model.md#evidence-ladder), claims and prior art to [Refs](Refs.md#prior-art-and-reproduction-boundaries), and implementation detail to [Architecture](Architecture.md); do not duplicate architecture here.
