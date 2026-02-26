# Baseline 1 workflow — FlowTransformer++

## Goal
Build the strongest honest **vanilla non-graph temporal Transformer baseline** first.

This baseline must be strong enough that:
- beating it later actually means something
- losing to simpler models forces an honest rethink
- any future looped or graph-temporal model is compared against a real bar

---

# 1. Baseline 1 contract

## Objective
Define exactly what Baseline 1 is and what is fixed before training.

### Baseline 1 definition
FlowTransformer++ is:
- a non-graph temporal Transformer over fixed flow windows
- a **vanilla untied Transformer baseline**
- the main temporal-attention reference point before any looped or graph model

### Fixed design rules
- [ ] same dataset version used everywhere
- [ ] same prediction unit used everywhere
- [ ] same split used everywhere
- [ ] same input features used everywhere
- [ ] same evaluation protocol used everywhere
- [ ] no looped recurrence in Baseline 1
- [ ] no graph structure in Baseline 1

### Success condition
- [ ] Baseline 1 is strong, reproducible, and fair enough to act as the reference model for all later work

---

# 2. Dataset contract for Baseline 1

## Objective
Lock what the model sees and predicts.

### Checklist
- [x] Lock dataset version and schema
- [-] Write down all numeric features
- [x] Write down all categorical features
- [x] Mark which features are available at inference time
- [x] remove leakage-prone or suspicious fields
- [x] define prediction unit as fixed temporal window
- [x] freeze main targets:
  - [x] binary malicious vs benign
  - [x] coarse attack-family classification
- [x] freeze auxiliary targets if used:
  - [x] `PROTOCOL`
  - [x] `L7_PROTO`
  - [x] `DST_PORT`
  - [x] `SRC_PORT`
  - [x] `IN_BYTES`, `OUT_BYTES`
  - [x] `IN_PKTS`, `OUT_PKTS`

### Deliverables
- [x] dataset card
- [x] feature inventory
- [x] leakage note
- [x] target definition note

---

# 3. Data integrity and leakage audit

## Objective
Make sure Baseline 1 is learning signal, not contamination.

### Checklist
- [x] check timestamp consistency
- [x] check duplicate and near-duplicate rows
- [x] check impossible labels
- [x] check temporal clustering by attack day or campaign
- [x] check whether neighbouring rows leak across splits
- [x] run a simple leakage baseline on suspicious columns
- [x] confirm no single suspicious feature can produce unrealistic accuracy

### Pass condition
- [x] no obvious feature or split artifact explains the expected performance

---

# 4. Window construction

## Objective
Turn raw flows into the fixed temporal unit used by Baseline 1.

### Default
- window length: `L = 32`
- stride: `32`
- later ablate: `8 / 16 / 32`

### Checklist
- [x] build windows deterministically
- [x] preserve order inside each window
- [x] keep the window builder frozen once Baseline 1 starts
- [x] store mapping from each window back to source rows
- [x] define truncation or padding rules if needed
- [x] emit optional mask if padding exists

### Output per timestep
- [x] numeric flow features
- [x] categorical protocol / port features
- [x] optional padding mask

### Deliverables
- [x] window builder script
- [x] saved window index map
- [x] reproducibility note for `L` and stride

---

# 5. Split protocol

## Objective
Make Baseline 1 evaluation believable.

### Default
Use a **block-aware temporal split**, not random row splitting.

### Checklist
- [ ] freeze train / val / test split once
- [ ] split before model tuning
- [ ] prevent overlapping-window contamination across splits
- [ ] save exact split indices
- [ ] prepare one harsher pure-temporal stress split

### Deliverables
- [ ] split file
- [ ] split generation script
- [ ] stress-split file

---

# 6. Baseline 1 architecture

## Objective
Specify the exact FlowTransformer++ baseline.

### Core structure
- [ ] numeric projection into model dimension
- [ ] additive categorical embeddings
- [ ] positional encoding
- [ ] learnable `[CLS]` token or equivalent global token
- [ ] pre-LN Transformer encoder
- [ ] dual pooling:
  - [ ] CLS pooling
  - [ ] mean pooling
- [ ] binary classification head
- [ ] family classification head
- [ ] optional auxiliary categorical heads
- [ ] optional auxiliary numeric heteroscedastic heads

### Fixed architecture checklist
- [ ] choose `d_model`
- [ ] choose number of heads
- [ ] choose number of encoder layers
- [ ] choose FFN width
- [ ] record total parameter count
- [ ] record FLOPs or throughput estimate

### Deliverables
- [ ] architecture spec table
- [ ] parameter count
- [ ] forward-pass cost note

---

# 7. Baseline 1 representation and heads

## Objective
Make the representation choice explicit and frozen.

### Representation
- [ ] use dual pooling
- [ ] concatenate `h_cls` and `h_mean`
- [ ] freeze representation choice before ablations

### Heads
#### Required heads
- [ ] binary head
- [ ] family head

#### Optional auxiliary heads
- [ ] protocol head
- [ ] L7 protocol head
- [ ] dst-port head
- [ ] src-port head
- [ ] bytes head
- [ ] packets head

### Rule
- [ ] auxiliary heads must not become the thesis story
- [ ] if auxiliaries are used, test whether they help or hurt the main targets

---

# 8. Baseline 1 training recipe

## Objective
Use a strong training recipe without hidden confounds.

### Checklist
- [ ] choose one optimizer family
- [ ] freeze LR schedule
- [ ] use gradient clipping
- [ ] decide whether EMA is on
- [ ] fix epoch budget
- [ ] fix batch size
- [ ] fix AMP policy
- [ ] fix seed set
- [ ] log train and val curves every epoch
- [ ] save best checkpoint by validation metric
- [ ] also save final checkpoint

### Loss stack
- [ ] binary loss fixed
- [ ] family loss fixed
- [ ] auxiliary categorical loss fixed if used
- [ ] auxiliary numeric loss fixed if used
- [ ] loss weights fixed before main comparisons

### Important
- [ ] Baseline 1 should not win because of a secret recipe change
- [ ] training choices must be recorded well enough to reproduce exactly

---

# 9. Baseline 1 evaluation protocol

## Objective
Evaluate Baseline 1 in a thesis-safe way.

### Primary metrics
- [ ] binary AUROC
- [ ] binary AUPRC
- [ ] family macro-F1
- [ ] family weighted-F1

### Operational metrics
- [ ] low-FPR behaviour
- [ ] confusion matrix by family
- [ ] calibration:
  - [ ] ECE
  - [ ] Brier
- [ ] parameter count
- [ ] inference throughput
- [ ] FLOPs or latency

### Robustness checks
- [ ] mean ± std over multiple seeds
- [ ] performance on harsher temporal split
- [ ] ablation on window length
- [ ] ablation on auxiliary heads
- [ ] error analysis on rare families

### Deliverables
- [ ] final metrics table
- [ ] calibration table
- [ ] cost table
- [ ] per-family error table

---

# 10. Baseline 1 ablation matrix

## Objective
Show what actually makes FlowTransformer++ work.

### Required ablations
- [ ] categorical embeddings on vs off
- [ ] positional encoding choice
- [ ] CLS only vs mean only vs dual pool
- [ ] auxiliary heads on vs off
- [ ] loss stack simplified vs full
- [ ] smaller vs chosen model capacity
- [ ] window length ablation
- [ ] stride ablation
- [ ] EMA on vs off

### Rule
- [ ] change one main factor at a time
- [ ] keep all other settings frozen
- [ ] report seed-aware results at least for the final configuration

---

# 11. Baseline 1 decision gates

## Gate A: does Baseline 1 actually work?
- [ ] it trains stably
- [ ] it beats trivial or leakage baselines
- [ ] it reaches sensible validation performance

## Gate B: is Baseline 1 strong enough to be the main reference?
- [ ] it is reproducible across seeds
- [ ] it survives a harsher temporal split
- [ ] it has a fair and well-documented recipe

## Gate C: is Baseline 1 good enough to justify later models?
- [ ] later looped or graph models will be compared directly against this exact baseline
- [ ] no future claim is allowed to skip this comparison

---

# 12. Negative-result policy for Baseline 1

## Pre-register this now
- [ ] if Baseline 1 is unstable, fix the recipe before inventing a fancier model
- [ ] if Baseline 1 does not hold up across seeds or stress splits, report that honestly
- [ ] do not use a weaker Transformer setup just to make a future model look better

---

# 13. Recommended build order for Baseline 1

## Practical sequence
1. [ ] dataset card + feature freeze
2. [ ] leakage audit
3. [ ] window builder
4. [ ] split freeze
5. [ ] architecture freeze
6. [ ] training recipe freeze
7. [ ] first stable training run
8. [ ] multi-seed run
9. [ ] calibration + stress split
10. [ ] ablation matrix
11. [ ] final Baseline 1 report table

---

# 14. Core success condition

Baseline 1 is only ready if all of this is true:
- [ ] it is a clear vanilla temporal Transformer baseline
- [ ] it is strong enough to be annoying to beat
- [ ] it is reproducible across seeds
- [ ] it survives a harsher temporal split
- [ ] it is not winning because of leakage, unfair tuning, or undocumented tricks
- [ ] it can serve as the direct reference point for every later looped or graph-temporal model