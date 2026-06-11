# NF3 exploration

**Status:** complete on 18 August 2026. This is a data report, not a model
result.

## What these findings mean

- BoT-IoT is 99.6930% attacks. Calling every flow an attack gives 99.6930%
  accuracy and flags every benign flow. This is arithmetic, not a model result.
- Endpoint fields strongly track labels. A model could recognise the capture
  instead of learning behaviour that transfers.
- CIC has 440 duplicate groups with conflicting labels. Identical records
  cannot provide an unambiguous scored target.

These findings motivate the [input and split rules](../plan/Architecture.md).
They do not show that endpoint history or pretraining works.

## How the data was inspected

The five existing DuckDB queries inspected the four local NF3 Parquet files:
[profile](../../tools/scripts/exploration/profile.sql),
[integrity](../../tools/scripts/exploration/integrity.sql),
[targets](../../tools/scripts/exploration/targets.sql),
[shift](../../tools/scripts/exploration/shift.sql), and
[leakage](../../tools/scripts/exploration/leakage.sql).

## Data

| Dataset | Rows | Benign | Attack |
|---|---:|---:|---:|
| NF-BoT-IoT-v3 | 16,933,808 | 0.3070% | 99.6930% |
| NF-CSE-CIC-IDS2018-v3 | 20,115,529 | 87.0702% | 12.9298% |
| NF-ToN-IoT-v3 | 27,520,260 | 61.0176% | 38.9824% |
| NF-UNSW-NB15-v3 | 2,365,424 | 94.6017% | 5.3983% |
| **Total** | **66,935,021** | n/a | n/a |

All files expose 55 columns. Three columns disagree in physical type:
`L7_PROTO`, `SRC_TO_DST_SECOND_BYTES`, and `DST_TO_SRC_SECOND_BYTES` are
`DOUBLE` in BoT/UNSW and `INT64` in ToN/CIC. M0 must cast the two numeric
fields to one representation. `L7_PROTO` remains outside the primary feature
view.

## Findings

### Integrity and duplicates

No file has a missing target or timestamp, an invalid binary target, negative
traffic, end-before-start time, duration mismatch, or invalid packet bound.
Inter-arrival time (IAT) summaries describe gaps between packets within a
flow. They are internally invalid on some rows:

| Dataset | Rows with invalid IAT summaries | Share |
|---|---:|---:|
| BoT-IoT | 43,806 | 0.259% |
| CSE-CIC-IDS2018 | 862,525 | 4.288% |
| ToN-IoT | 456,179 | 1.658% |
| UNSW-NB15 | 316 | 0.013% |

M0 treats the offending IAT group as missing and keeps a missingness indicator;
it does not delete the whole flow.

Exact duplicate grouping used every non-label column:

| Dataset | Rows in duplicate groups | Removable rows | Conflicting-label groups |
|---|---:|---:|---:|
| BoT-IoT | 0 | 0 | 0 |
| CSE-CIC-IDS2018 | 1,257,554 | 628,914 | 440 |
| ToN-IoT | 2,296,177 | 1,816,137 | 0 |
| UNSW-NB15 | 29,630 | 14,815 | 0 |

Duplicates must stay in one partition. CIC's 440 conflicting groups cannot be
silently deduplicated or assigned a majority label; M0 must exclude them from
scored results or report them separately.

### Target balance

The class mix is extreme and inconsistent across datasets. Examples of rare
attack labels are UNSW `worms` (158) and `analysis` (1,226), CIC SQL injection
(440) and XSS (480), BoT `theft` (1,615), and ToN ransomware (3,971) and MITM
(6,013). Accuracy and one aggregate F1 score would hide these failures. Rare
classes without enough independent support groups remain unsupported.

### Time shift

Attack and family distributions change sharply by day. Maximum observed daily
total-variation shift reaches 0.997 for BoT, 0.745 for ToN, and 0.330 for CIC;
UNSW shifts much less. Random row splits would mix these day signatures and
inflate performance. M0 therefore uses chronological and group-separated
evaluation.

### Shortcut risk

The five-tuple's normalised association with attack-family labels is
0.968–1.000 across the four datasets; for the binary target it is
0.442–0.998. The query measures the fraction of label uncertainty explained
within the inspected data: zero means no observed association and one means
the feature determines the label in that sample. This is not held-out
classification accuracy; high-cardinality fields can make the association
look strong without predicting unseen examples.

Endpoint fingerprints are also close to deterministic in CIC and UNSW.
Destination-port association is material in every dataset. These observations
flag possible shortcuts; grouped and chronological tests are needed to
measure how much a trained model depends on them.

The primary model therefore receives no raw addresses, endpoint identity,
five-tuple, absolute time, capture/day identifiers, or `L7_PROTO`. Ports keep
the restricted representation in
[Architecture](../plan/Architecture.md#prediction-unit-and-input-features), and
a port-free result is mandatory.

## M0 decisions

1. Final M0 uses NF-UNSW-NB15-v3 only. The other NF3 files remain exploration
   evidence, not training or holdout data for this baseline.
2. Build chronological and group-separated splits under `src/m0`; never use a
   random row split.
3. Fit imputation, scaling, vocabularies, port buckets, and all other learned
   preprocessing on training data only.
4. Use the implemented M0 fields and context. The later ladder's input and
   evaluation rules remain separate plans.

Exploration is complete. M0 is implemented on NF-UNSW-NB15-v3; these results
do not establish transfer, operational generalisation, or a foundation model.
