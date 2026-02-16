# Experiment 1: NF-CICIDS2018-v3 Dataset Understanding, Audit, Exploration, and Data Preparation

This experiment is the dataset understanding stage for the project. The aim is not to train models yet. The aim is to understand the structure, quality, chronology, label behavior, leakage risk, and preparation requirements of `NF-CICIDS2018-v3.csv` well enough that every later experiment starts from the same audited data foundation.

The output of this stage is a reusable dataset manifest, a feature audit, a leakage report, and a frozen data-preparation specification.

## Dataset-Specific Scope
The dataset contains the following feature families:

| Family | Headers |
| --- | --- |
| Chronology and duration | `FLOW_START_MILLISECONDS`, `FLOW_END_MILLISECONDS`, `FLOW_DURATION_MILLISECONDS`, `DURATION_IN`, `DURATION_OUT`, `SRC_TO_DST_IAT_MIN`, `SRC_TO_DST_IAT_MAX`, `SRC_TO_DST_IAT_AVG`, `SRC_TO_DST_IAT_STDDEV`, `DST_TO_SRC_IAT_MIN`, `DST_TO_SRC_IAT_MAX`, `DST_TO_SRC_IAT_AVG`, `DST_TO_SRC_IAT_STDDEV` |
| Endpoint and transport identity | `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`, `L4_DST_PORT`, `PROTOCOL`, `L7_PROTO` |
| Traffic volume and count | `IN_BYTES`, `OUT_BYTES`, `IN_PKTS`, `OUT_PKTS`, `SRC_TO_DST_SECOND_BYTES`, `DST_TO_SRC_SECOND_BYTES`, `SRC_TO_DST_AVG_THROUGHPUT`, `DST_TO_SRC_AVG_THROUGHPUT` |
| Packet size structure | `LONGEST_FLOW_PKT`, `SHORTEST_FLOW_PKT`, `MIN_IP_PKT_LEN`, `MAX_IP_PKT_LEN`, `NUM_PKTS_UP_TO_128_BYTES`, `NUM_PKTS_128_TO_256_BYTES`, `NUM_PKTS_256_TO_512_BYTES`, `NUM_PKTS_512_TO_1024_BYTES`, `NUM_PKTS_1024_TO_1514_BYTES` |
| TCP and retransmission behavior | `TCP_FLAGS`, `CLIENT_TCP_FLAGS`, `SERVER_TCP_FLAGS`, `TCP_WIN_MAX_IN`, `TCP_WIN_MAX_OUT`, `RETRANSMITTED_IN_BYTES`, `RETRANSMITTED_IN_PKTS`, `RETRANSMITTED_OUT_BYTES`, `RETRANSMITTED_OUT_PKTS` |
| Routing and protocol-specific fields | `MIN_TTL`, `MAX_TTL`, `ICMP_TYPE`, `ICMP_IPV4_TYPE`, `DNS_QUERY_ID`, `DNS_QUERY_TYPE`, `DNS_TTL_ANSWER`, `FTP_COMMAND_RET_CODE` |
| Labels | `Label`, `Attack` |

This experiment must be written around those fields rather than around a generic tabular pipeline.

## Core Question
Before modelling, can we explain:
- what each feature group represents
- which fields are trustworthy measurements vs identifiers or shortcuts
- how labels behave across time
- how protocol-specific sparse fields should be interpreted
- what the correct cleaned dataset should look like for later experiments

## Workflow
### 1. Raw Intake and Dataset Manifest
1. Load the raw CSV and verify the schema exactly once
   - Confirm that the expected columns are present
   - Confirm that `Label` and `Attack` are the only target columns
   - Confirm that `FLOW_START_MILLISECONDS` and `FLOW_END_MILLISECONDS` are usable chronology anchors
2. Build the dataset manifest keyed by `day`, derived from `FLOW_START_MILLISECONDS`
   - For each day, record:
     - row count
     - time range
     - binary label counts from `Label`
     - attack-family counts from `Attack`
     - whether timestamps are monotonic before sorting
     - whether rows need chronology repair by sorting on `FLOW_START_MILLISECONDS`

### 2. Schema and Type Audit
1. Assign a semantic type to every header
   - Audit-only identity fields:
     - `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`
   - High-risk shortcut fields requiring explicit justification before prediction use:
     - `L4_SRC_PORT`, `L4_DST_PORT`, `FLOW_START_MILLISECONDS`, `FLOW_END_MILLISECONDS`, `DNS_QUERY_ID`
   - Candidate predictive measurement fields:
     - duration, count, size, throughput, retransmission, TTL, and IAT features
   - Label fields:
     - `Label`, `Attack`
2. Confirm machine-readable parsing rules
   - Parse timestamps as integer milliseconds
   - Parse numeric traffic fields as numeric, not text
   - Parse IP addresses as strings for audit purposes only
   - Record whether ports and protocol codes will be treated as numeric codes, categorical variables, or both in later preparation
3. Check structural consistency
   - Confirm that `FLOW_END_MILLISECONDS >= FLOW_START_MILLISECONDS`
   - Compare `FLOW_DURATION_MILLISECONDS` against start-end differences
   - Check whether `DURATION_IN` and `DURATION_OUT` are consistent with flow duration
   - Check whether IAT statistics are non-negative and internally sensible

### 3. Feature Audit by Family
1. Profile every column
   - For each header, compute:
     - dtype
     - missingness rate
     - zero fraction
     - unique count
     - min, median, max
     - selected quantiles for heavy-tailed fields
2. Audit chronology and timing fields
   - Inspect:
     - `FLOW_START_MILLISECONDS`
     - `FLOW_END_MILLISECONDS`
     - `FLOW_DURATION_MILLISECONDS`
     - `DURATION_IN`
     - `DURATION_OUT`
     - all `SRC_TO_DST_IAT_*` and `DST_TO_SRC_IAT_*` fields
   - Check:
     - impossible negative values
     - extreme outliers
     - contradictions between duration and inter-arrival summaries
     - whether absolute time directly reveals scenario order and should remain audit-only
3. Audit endpoint and transport identity fields
   - Inspect:
     - `IPV4_SRC_ADDR`
     - `IPV4_DST_ADDR`
     - `L4_SRC_PORT`
     - `L4_DST_PORT`
     - `PROTOCOL`
     - `L7_PROTO`
   - Check:
     - number of unique hosts and ports
     - concentration of attacks on specific host pairs or service ports
     - whether raw endpoints create shortcut learning risk
4. Audit traffic volume and count fields
   - Inspect:
     - `IN_BYTES`, `OUT_BYTES`, `IN_PKTS`, `OUT_PKTS`
     - `SRC_TO_DST_SECOND_BYTES`, `DST_TO_SRC_SECOND_BYTES`
     - `SRC_TO_DST_AVG_THROUGHPUT`, `DST_TO_SRC_AVG_THROUGHPUT`
   - Check:
     - heavy tails
     - zero inflation
     - impossible relationships such as bytes with zero packets
     - whether log transforms are needed
5. Audit packet size and TCP behavior fields
   - Inspect:
     - `LONGEST_FLOW_PKT`, `SHORTEST_FLOW_PKT`, `MIN_IP_PKT_LEN`, `MAX_IP_PKT_LEN`
     - `NUM_PKTS_UP_TO_128_BYTES`, `NUM_PKTS_128_TO_256_BYTES`, `NUM_PKTS_256_TO_512_BYTES`, `NUM_PKTS_512_TO_1024_BYTES`, `NUM_PKTS_1024_TO_1514_BYTES`
     - `TCP_FLAGS`, `CLIENT_TCP_FLAGS`, `SERVER_TCP_FLAGS`
     - `TCP_WIN_MAX_IN`, `TCP_WIN_MAX_OUT`
     - all retransmission fields
   - Check:
     - physically implausible packet-size relationships
     - bucket counts inconsistent with total packet counts
     - TCP-only fields appearing active in non-TCP traffic
6. Audit protocol-specific sparse fields carefully
   - Inspect:
     - `ICMP_TYPE`, `ICMP_IPV4_TYPE`
     - `DNS_QUERY_ID`, `DNS_QUERY_TYPE`, `DNS_TTL_ANSWER`
     - `FTP_COMMAND_RET_CODE`
   - Check:
     - whether zero means "not applicable", "missing", or a real value
     - whether these fields are only meaningful under specific `PROTOCOL`, `L7_PROTO`, or port contexts
     - whether an applicability mask should be created during preparation
7. Flag columns needing special treatment
   - Mark columns that are:
     - constant
     - almost constant
     - extremely sparse
     - identifier-like
     - leak-prone
     - only valid under protocol-specific conditions

### 4. Label Audit and Class Structure
1. Validate the relationship between `Label` and `Attack`
   - Check that benign rows map cleanly to a benign attack label
   - Check that malicious rows are consistent with non-benign attack families
   - Record any mismatch between binary and fine-grained labels
2. Build the class-distribution report
   - Report:
     - global binary balance
     - per-attack counts
     - long-tail attack families
     - low-support classes that may be unreliable for later modelling
3. Build the temporal label report
   - Report attack-family counts by day
   - Check whether some attacks exist only in very narrow windows
   - Check whether some days are almost entirely benign or almost entirely attack-heavy
4. Define reusable label views
   - Keep `Label` as the binary target view
   - Keep `Attack` as the fine-grained family view
   - Create a thesis-level coarse family mapping and record it explicitly

### 5. Duplicate, Entity Overlap, and Leakage Audit
1. Run duplicate checks
   - Check exact full-row duplicates
   - Check duplicates after excluding obvious audit-only identifiers
   - Check near-duplicates using a stable flow signature based on:
     - day bucket from `FLOW_START_MILLISECONDS`
     - `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`
     - `L4_SRC_PORT`, `L4_DST_PORT`
     - `PROTOCOL`, `L7_PROTO`
     - `FLOW_DURATION_MILLISECONDS`
     - packet and byte totals
2. Audit host and service reuse
   - Quantify repeated source-destination pairs across days
   - Quantify port reuse across days and attack families
   - Check whether some attack labels are effectively proxies for specific hosts or ports
3. Identify leakage-prone fields
   - Treat the following as suspect until justified:
     - `IPV4_SRC_ADDR`
     - `IPV4_DST_ADDR`
     - raw absolute timestamps
     - exact ports if they act as scenario identifiers
     - `DNS_QUERY_ID` if it behaves like a transaction identifier rather than a generalizable measurement
4. Freeze the audit-only field list
   - Keep fields needed for provenance, overlap checks, and leakage inspection
   - Do not automatically allow them into later predictive views

### 6. Data Preparation Stage
1. Define the prepared dataset views

| View | Purpose | Included fields |
| --- | --- | --- |
| Raw audit view | Provenance and leakage inspection | All original fields plus row hash and derived day key |
| Clean predictive tabular view | Later baseline models | Only approved measurement features and approved encoded categorical features |
| Graph-ready flow view | Later graph construction | The same cleaned measurements plus entity references kept outside the predictive matrix |

2. Sort and standardize chronology
   - Sort all rows by `FLOW_START_MILLISECONDS`
   - Derive `flow_day` from `FLOW_START_MILLISECONDS`
   - Preserve the original row order in the audit manifest if needed
3. Apply column-level preparation rules
   - Drop from predictive use by default:
     - `IPV4_SRC_ADDR`
     - `IPV4_DST_ADDR`
     - raw `FLOW_START_MILLISECONDS`
     - raw `FLOW_END_MILLISECONDS`
     - raw `Label`
     - raw `Attack`
   - Review with caution before predictive use:
     - `L4_SRC_PORT`
     - `L4_DST_PORT`
     - `PROTOCOL`
     - `L7_PROTO`
     - `DNS_QUERY_ID`
   - Retain as candidate predictive measurements:
     - durations
     - bytes and packets
     - throughput
     - retransmissions
     - packet-size summaries
     - TTL fields
     - IAT summaries
4. Handle protocol-specific applicability explicitly
   - For ICMP-related fields, define when `ICMP_TYPE` and `ICMP_IPV4_TYPE` are meaningful
   - For DNS-related fields, define when `DNS_QUERY_ID`, `DNS_QUERY_TYPE`, and `DNS_TTL_ANSWER` are meaningful
   - For FTP-related fields, define when `FTP_COMMAND_RET_CODE` is meaningful
   - Create applicability indicators if zero is being used as a placeholder
5. Handle skew and sparsity
   - Apply `log1p` to highly skewed byte, packet, throughput, and retransmission fields where justified
   - Record sparse columns that should be kept, bucketed, or removed
   - Record whether extreme values are capped, winsorized, or left untouched
6. Prepare reproducible outputs
   - Save row-level hashes or immutable row IDs
   - Save the cleaned column list in order
   - Save every drop, transformation, derived field, and label mapping

### 7. Deliverables and Quality Gates
1. Save artefacts
   - dataset manifest by day
   - schema and type audit
   - feature-profile tables
   - label-distribution tables
   - duplicate and leakage report
   - frozen data-preparation manifest
2. Mark this experiment complete only if:
   - chronology is understood and documented
   - label consistency is verified
   - suspicious identifier-like fields are explicitly classified
   - protocol-specific sparse fields have a defensible interpretation
   - predictive, audit-only, and graph-ready views are frozen for later use

## Interpretation
- If the dataset looks easy only when raw hosts, timestamps, or exact service identifiers are present, later predictive claims will be weak.
- If attack families cluster tightly by day, host, or port, later splits must be conservative.
- If protocol-specific fields are mostly placeholder zeros, naive missing-value handling will be wrong.
- If the preparation manifest is stable and reusable, later modelling can start from a defensible baseline.

## Notes
- `FLOW_START_MILLISECONDS` should be the primary chronology key for this dataset.
- `Label` should remain the binary target and `Attack` should remain the family-level label.
- `IPV4_SRC_ADDR` and `IPV4_DST_ADDR` should be treated as audit-first fields, not default predictive features.
- `DNS_QUERY_ID`, `FTP_COMMAND_RET_CODE`, and ICMP fields require protocol-aware interpretation before they are trusted.
