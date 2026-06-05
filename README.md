# FlowIDS

Initial project scaffold for a deep learning model repository.

## Setup

```bash
pixi install
pixi run check
```

Run a Parquet exploration query from the repository root:

```bash
pixi run duckdb < tools/scripts/exploration/profile.sql
```

## Conversion audit

Audit one CSV/Parquet pair and write an append-only deterministic receipt:

```bash
pixi run d0 -- \
  --csv data/raw/example/data/flows.csv \
  --parquet data/publish/data/flows.parquet \
  --bag-manifest data/raw/example/manifest-sha1.txt \
  --feature-config tools/configs/nf3.json \
  --output data/evidence/flows.json
```

The command exits `1` when conversion integrity fails and `3` when conversion
passes but D0 is blocked (`2` remains a command-usage error). It compares every
ordered parsed CSV value with Parquet in bounded batches. Current NF3 receipts
remain `Q0` because required source provenance is unresolved. The audit never
materializes a split or authorizes model training.

## First Q1 source

`tools/configs/mawi.json` pins one strictly ordered MAWI samplepoint-F capture,
the local YAF/Super Mediator build, conversion fields, timeout semantics, and
the exact packet/byte equations. After obtaining the research-only source and
reproducing the configured conversion, materialize its local Parquet view with:

```bash
pixi run mawi -- \
  --config tools/configs/mawi.json \
  --archive data/raw/mawi/samplepoint-f/2006/200608241400/200608241400.dump.gz \
  --pcap data/work/mawi/samplepoint-f/2006/200608241400/200608241400.dump \
  --yaf-bin /path/to/pinned/yaf \
  --super-mediator-bin /path/to/pinned/super_mediator \
  --yaf-ipfix data/work/mawi/samplepoint-f/2006/200608241400/yaf.ipfix \
  --yaf-csv data/work/mawi/samplepoint-f/2006/200608241400/flows.csv \
  --parquet data/work/mawi/samplepoint-f/2006/200608241400/mawi-q1.parquet \
  --receipt data/work/mawi/samplepoint-f/2006/200608241400/mawi-q1.receipt.json
```

The verified local run retains 853,869 completed bidirectional flows and
rejects 60,109 capture-end-censored flows. All 11,603 meter exclusions are
accounted for. Because YAF does not export the packet time that triggers a
timeout, idle and active records use conservative causal availability bounds,
never the configured timeout as a fabricated export timestamp. Raw addresses
remain routing-only. The source and every
endpoint-bearing derivative stay local under WIDE's research-only terms; do
not upload them to Hugging Face or include them in a release.

The pinned full-trace replay produced the same hashes on two runs:

| Evidence | Result |
|---|---:|
| Accepted / capture-end censored flows | 853,869 / 60,109 |
| Parquet size | 16,213,707 bytes |
| Parquet SHA-256 | `afa985bc95583e968d2b194932b42bb4f34897af6cff59326687feec16c5a1a0` |
| Materialization wall time (two replays) | 30.17–31.28 s |
| Peak resident memory (maximum) | 708,624,384 bytes |

The receipt conserves packet and L3-octet totals separately for every observed
IP protocol. These are intake measurements on the recorded machine, not a
performance target. Q1 is a source-quality decision only: model-ladder D0
remains blocked until a frozen split, train-only fit, and causal state replay
also pass.

Freeze the local chronological split with an owner-only 32-byte routing key:

```bash
pixi run mawi-split -- \
  --config tools/configs/mawi.json \
  --source-receipt data/work/mawi/samplepoint-f/2006/200608241400/mawi-q1.receipt.json \
  --source-parquet data/work/mawi/samplepoint-f/2006/200608241400/mawi-q1.parquet \
  --secret-file /path/to/owner-only-32-byte-key \
  --train-end-ms 1156395900794 \
  --validation-end-ms 1156396200794 \
  --purge-ms 60000 \
  --sidecar data/evidence/mawi-q1-split.parquet \
  --receipt data/evidence/mawi-q1-split.receipt.json
```

The cutoffs are fixed at five and ten minutes after capture start. Partitioning
uses flow availability, not last-packet time. Most timeout-ended records share
the conservative capture-end bound, so this artifact proves local split
plumbing only; it cannot support a chronological-generalisation or IDS claim.
Both outputs contain endpoint-derived HMAC identifiers and remain local. D0
stays blocked until train-only fitting, semantic-context diagnostics and
partition-reset causal replay pass.

The full local split retained 824,972 of 853,869 flows after the fixed boundary
purges: 29,840 train, 21,615 validation and 773,517 test. The final block
contains one 743,518-row capture-end availability tie, which is why it is not a
temporal-generalisation benchmark. The 105,980,661-byte sidecar has SHA-256
`e15a0295b8aef9e3a932a47a5597d6b355ead899ded431069b7a623e8045f742`;
two byte-identical replays took 25.72--26.23 seconds with at most
1,220,460,544 bytes peak resident memory on the recorded machine.

Tagged releases run `pixi run release-check` against the generated archive and
refuse data, evidence, secrets, or model schemas containing forbidden fields.
