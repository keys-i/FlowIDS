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

Tagged releases run `pixi run release-check` against the generated archive and
refuse data, evidence, secrets, or model schemas containing forbidden fields.
