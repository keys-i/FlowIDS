# Model specification

## Claim boundary

FlowTransformer is a supervised baseline and Anomal-E is graph SSL; neither is a Transformer foundation model for unlabeled NetFlow. [MMAE](Refs.md#mmae2026) (March 2026) substantially anticipates masked EMA teacher--student traffic learning; [CMES cross-flow modeling](Refs.md#cmescrossflow2026) (July 2026) directly anticipates learned cross-flow 4-bit/16-type relation bias. Masking, EMA, hierarchy, and relation bias alone are not novel. Claims, data controls, and exclusions are governed by [Thesis](Thesis.md#claim-boundary) and [Refs](Refs.md#prior-art-and-reproduction-boundaries). The stronger optional contribution—packet-sequence teacher to NetFlow-only student distillation—remains conditional on paired data and a refreshed novelty search.

## Unit and backbone

All neural comparisons use the same deployable backbone unless model size is the isolated variable. One completed, exported, bidirectional flow is one token. A unidirectional source must be converted causally under a documented pairing rule or excluded. The record encoder uses train-only robust numeric transforms; numeric field projections; categorical PAD/UNK; missingness; and a two-layer event projection. The causal backbone is 512-wide, eight pre-LN blocks, eight heads, FFN 2048, GELU, nominal dropout 0.1, RoPE, and a learned elapsed-time projection. It consumes at most 255 prior tokens plus target (256 total). Transformer blocks are approximately 25.2M weights before tokenizer; the measured deployable target is approximately 25M and all record-encoder, embedding, and backbone parameters count toward the ±5% comparison band. If required, reduce FFN width only enough to enter that band and freeze it for every rung. EMA copies and SSL heads count toward training compute and memory but not deployed parameters; SSL heads are removed for deployment. Bidirectional models are offline-only.

Source-only tuning grid: horizon {1, 10, 60} minutes, learning rate {1e-4, 3e-4}, weight decay {0.01, 0.05}, and dropout {0, 0.1}; freeze the selected setting and permit no post-test search.

## SSL objectives and guardrails

Mask 30% of event/group units in five groups: transport+flags; service+port; directional volume; packet-size+retransmission; duration+IAT. Never mask or reconstruct identifiers, addresses, labels, capture identity, or absolute time. Use contiguous spans with mean length three events and a learned group token. `L_raw` is categorical CE, missingness BCE, and transformed-numeric Smooth L1, averaged first within each group and then across groups. Padding and absent values are not value-reconstruction targets; their missingness remains a target.

`L_latent`: the masked student predicts the stop-gradient, unmasked causal EMA teacher state for the same event: mean of the top four layer-normalized states through a two-layer predictor. Teacher dropout is off; update it after each optimizer step with momentum 0.99→0.9999. `L_future` predicts the first, fourth, and sixteenth subsequent endpoint-incident teacher latents. These targets stay inside the pretraining partition but outside encoder input; ignore absent targets. Shuffle only within source/day/horizon and include a raw-next-value control. The core hybrid is `L_raw + L_latent`. After 200 train-only warmup batches, set weights by inverse median loss, normalize them to sum to two, and freeze them; count the warmup compute. Log unweighted encoder-gradient norms and objective-gradient cosines every 100 updates. Use one compute-matched alternating schedule only after conflict occurs on >50% of sampled updates for three epochs. Promote future only after downstream external transfer exceeds both controls. Exclude generic contrastive learning except as an audited baseline.

## Comparator contract

Run logistic regression; CatBoost per-flow; CatBoost deterministic causal aggregates; MLP; supervised flat causal Transformer; corrected historical FlowTransformer; matched random initialization; frozen/full fine-tuning; reproducible E-GraphSAGE, Anomal-E, NEGSC, GraphIDS, and Van Langendonck graph-foundation work; and CMES offline upper bound. The requested `GNNet` label remains unresolved and is not silently treated as a synonym for a verified paper. Unavailable or unidentified models remain literature-only.

Controls: MMAE-Exact is the official-code packet track. MMAE-NF is a same-25M NetFlow adaptation that replaces semantic groups from a statistically matched support flow only after a semantic-validity audit; never call it a reproduction. CMES-25M is an explicit reimplementation with the same data, fields, backbone, and heads, CMES sorted/grouped cross-flow context, its four endpoint/destination-port/protocol/feature-similarity predicates encoded as 16 types, and bidirectional supervised training. CMES-Causal uses proposed ego context with CMES predicates. Production core never imports these mechanisms automatically.

## Evidence ladder

| Rung | Only change | Promotion evidence | Immediate rejection |
|---|---|---|---|
| D0 | No model | Reconstructable provenance, chronology, schema, labels, duplicate groups, and sealed splits | Stop all modeling if any remains unresolved. |
| M0 | Fixed flat causal backbone trained from labels | Establish neural reference against the best classical baseline | If classical parity holds, allow only the predeclared small M1 discriminator; do not scale. |
| M1-R | Add `L_raw` pretraining | External low-label gain over M0 and the best tree on at least two targets | Reject if gain depends on identifiers/ports, disappears chronologically, or stays in-domain. |
| M1-L | Replace raw with `L_latent`; run parallel to M1-R | Non-collapsed external low-label gain under the same budget | Reject if raw reconstruction matches it or collapse persists. |
| M2-H | Combine raw and latent | Beat M1-R, M1-L, M0, best tree, and MMAE-NF | Reject if either constituent or MMAE-NF matches it under fixed exposure or FLOPs. |
| M2-F | Conditionally add `L_future` | Beat shuffled-future and raw-next controls on external transfer | Reject if it learns source/day identity, conflicts persist, or transfer does not improve. |
| M3-Ego | Replace flat collector history with causal endpoint-ego history | Positive context main effect and positive hybrid×ego interaction | Reject if random, matched, statistical-similarity, or CMES grouping matches it. |
| M4-Rel | Add directed identity-free endpoint relation bias | Beat no-bias and CMES-Causal; pass destruction and renaming tests | Reject if endpoint identity is required, relations are unused, or endpoint-held-out performance falls. |
| M5-Hier | Conditionally add multi-resolution summaries | Run only when ≥20% of contexts truncate and long-horizon recall is deficient; beat flat M4 within runtime budget | Reject otherwise or if flat M4 matches it. |
| X1 | Optional packet-teacher distillation | Legally usable paired data and external flow-only student gain over same-modality controls | Reject if alignment/privacy fails or gain is <1 absolute AUPRC point. |
| Final | No mechanism change | Five-seed replication on untouched networks/tasks | Any failed primary gate narrows the claim. |

Ego context is the latest 128 source and 128 destination prior flows, unioned, deduplicated, ordered, horizon-filtered, capped to 255, then target. Controls: flat; random; time-and-feature-matched with no shared endpoint; target-only; CMES grouped; statistical-similarity. M4 relation bits are source-source, destination-destination, source-earlier-destination, destination-earlier-source: 16 types. Each layer/head gets a scalar causal-logit bias per type; raw IDs are never embedded. Require renaming invariance and marginal-preserving relation destruction. Hierarchy is conditional on ladder evidence. See [Architecture](Architecture.md#causal-ego-history) and [relation bias](Architecture.md#relation-bias-and-encoder).

Screen 3–5M models on 20M exposures: `{scratch, raw, latent, hybrid} × {flat, ego}`, with paper controls; screen relation separately. Estimate interaction as `(EgoHybrid − EgoScratch) − (FlatHybrid − FlatScratch)`; paired block-bootstrap lower 95% CI must exceed zero.

## Promotion, collapse, scaling

Promotion requires ≥1 absolute macro OVR AUPRC point at 10 labels per supported class, sampled from independent support groups, and on the area under the label-efficiency curve at {1, 5, 10, 100}; paired hierarchical-bootstrap lower CI >0; improvement on two networks; no target loses >1 AUPRC point; critical-family recall loses ≤2 points; TPR at FPR 1e-3 is noninferior within one point; ECE worsens by ≤0.02; and the rung beats the best applicable same-input comparator. Fixed-exposure and fixed-FLOP comparisons are separate runs. Use three pretraining seeds for screening, five for final confirmation, five support draws, and three downstream-optimization seeds.

At every validation record embedding standard-deviation quantiles, covariance eigenvalues, active/effective rank, off-diagonal correlation, mean/nearest-neighbor cosine, alignment/uniformity, head and encoder gradient norms, and source/day/entity distributions. Treat representation collapse as persistent across three validations if median standard deviation <1e-3, active/effective rank <10%, more than half dimensions have standard deviation <1e-3, rank falls ≥50% while loss improves, or the vector is indistinguishable from constant. These are preregistered rejection thresholds, not literature constants; decision provenance is in [Thesis](Thesis.md#research-questions-and-decision-rules).

| Stage | Model | Flows / sources | Exposures | Gate |
|---|---:|---:|---:|---|
| S0 | 3--5M | ≥5M / — | 20M | screen |
| S1 | 25M | ≥50M / 10 | 100M | external transfer |
| S2 | 25M | ≥250M / 20 | 500M | scale confirmation |
| S3 | 90M | ≥1B / 30 | after S2 | only if S2 passes |
| S4 | 300M | — | — | only if 90M beats 25M by ≥2% relative external AUPRC |

Do not replace source diversity with repeated epochs or model size. Stop after two doubled-exposure runs that improve SSL loss without transfer improvement.
