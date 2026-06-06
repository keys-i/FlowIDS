SET VARIABLE parquet_file = coalesce(getvariable('parquet_file'), 'data/publish/data/NF-*-v3.parquet');

WITH flows AS (
    SELECT *, filename AS data_file
    FROM read_parquet(getvariable('parquet_file'), filename = true, union_by_name = true)
), counts AS (
    SELECT
        data_file,
        count(*) AS rows,
        count_if(FLOW_START_MILLISECONDS IS NULL OR FLOW_END_MILLISECONDS IS NULL) AS missing_timestamps,
        count_if(FLOW_END_MILLISECONDS < FLOW_START_MILLISECONDS) AS end_before_start,
        count_if(abs(FLOW_END_MILLISECONDS - FLOW_START_MILLISECONDS - FLOW_DURATION_MILLISECONDS) > 1) AS duration_mismatch,
        count_if(Label IS NULL OR nullif(trim(Attack), '') IS NULL) AS missing_target,
        count_if(Label NOT IN (0, 1)) AS invalid_binary_label,
        count_if(Label IS NOT NULL AND Attack IS NOT NULL AND ((Label = 0) <> (lower(trim(Attack)) = 'benign'))) AS inconsistent_target,
        count_if(IN_BYTES < 0 OR OUT_BYTES < 0 OR IN_PKTS < 0 OR OUT_PKTS < 0) AS negative_traffic,
        count_if(SHORTEST_FLOW_PKT > LONGEST_FLOW_PKT OR MIN_IP_PKT_LEN > MAX_IP_PKT_LEN) AS invalid_packet_bounds,
        count_if(SRC_TO_DST_IAT_MIN > SRC_TO_DST_IAT_AVG OR SRC_TO_DST_IAT_AVG > SRC_TO_DST_IAT_MAX OR DST_TO_SRC_IAT_MIN > DST_TO_SRC_IAT_AVG OR DST_TO_SRC_IAT_AVG > DST_TO_SRC_IAT_MAX OR SRC_TO_DST_IAT_STDDEV < 0 OR DST_TO_SRC_IAT_STDDEV < 0) AS invalid_iat_statistics
    FROM flows
    GROUP BY data_file
), failures AS (
    UNPIVOT counts
    ON COLUMNS(* EXCLUDE (data_file, rows))
    INTO NAME check_name VALUE violations
)
SELECT data_file, check_name, violations, round(violations::DOUBLE / rows, 8) AS violation_rate
FROM failures
ORDER BY data_file, violation_rate DESC, check_name;

WITH flows AS (
    SELECT *, filename AS data_file, coalesce(lower(nullif(trim(Attack), '')), '<NULL>') AS normalized_attack
    FROM read_parquet(getvariable('parquet_file'), filename = true, union_by_name = true)
), sources AS (
    SELECT DISTINCT data_file
    FROM flows
), records AS (
    SELECT
        * EXCLUDE (Label, Attack, normalized_attack),
        count(*) AS occurrences,
        count(DISTINCT Label) AS label_values,
        count(DISTINCT normalized_attack) AS attack_values,
        count(DISTINCT (Label, normalized_attack)) AS target_pairs
    FROM flows
    GROUP BY ALL
    HAVING count(*) > 1
), summary AS (
    SELECT
        data_file,
        count(*) AS duplicate_groups,
        sum(occurrences) AS duplicate_rows,
        sum(occurrences - 1) AS removable_duplicate_rows,
        count_if(label_values > 1 OR attack_values > 1 OR target_pairs > 1) AS label_conflict_groups,
        coalesce(sum(occurrences) FILTER (WHERE label_values > 1 OR attack_values > 1 OR target_pairs > 1), 0) AS label_conflict_rows
    FROM records
    GROUP BY data_file
)
SELECT
    sources.data_file,
    coalesce(summary.duplicate_groups, 0) AS duplicate_groups,
    coalesce(summary.duplicate_rows, 0) AS duplicate_rows,
    coalesce(summary.removable_duplicate_rows, 0) AS removable_duplicate_rows,
    coalesce(summary.label_conflict_groups, 0) AS label_conflict_groups,
    coalesce(summary.label_conflict_rows, 0) AS label_conflict_rows
FROM sources
LEFT JOIN summary USING (data_file)
ORDER BY sources.data_file;

WITH flows AS (
    SELECT filename AS data_file, FLOW_END_MILLISECONDS
    FROM read_parquet(getvariable('parquet_file'), filename = true, union_by_name = true)
), daily AS (
    SELECT data_file, epoch_ms(FLOW_END_MILLISECONDS)::DATE AS completion_day, count(*) AS flows
    FROM flows
    WHERE FLOW_END_MILLISECONDS IS NOT NULL
    GROUP BY ALL
)
SELECT data_file, completion_day, flows, sum(flows) OVER (PARTITION BY data_file ORDER BY completion_day) AS cumulative_flows
FROM daily
ORDER BY data_file, completion_day;

-- Strict, label-blind raw-field duplicate check.
-- `data_file` is only a lineage/observation-domain proxy; the result is not
-- equivalent to NF3-v1 until that one-file/one-domain assumption is verified.
-- `occurrences` retains row multiplicity rather than deduplicating it.
WITH flows AS (
    SELECT
        filename AS data_file,
        IPV4_SRC_ADDR,
        L4_SRC_PORT,
        IPV4_DST_ADDR,
        L4_DST_PORT,
        PROTOCOL,
        FLOW_START_MILLISECONDS,
        FLOW_END_MILLISECONDS,
        IN_PKTS,
        IN_BYTES,
        OUT_PKTS,
        OUT_BYTES,
        TCP_FLAGS,
        CLIENT_TCP_FLAGS,
        SERVER_TCP_FLAGS
    FROM read_parquet(getvariable('parquet_file'), filename = true, union_by_name = true)
), sources AS (
    SELECT DISTINCT data_file
    FROM flows
), near_keys AS (
    SELECT
        data_file,
        IPV4_SRC_ADDR,
        L4_SRC_PORT,
        IPV4_DST_ADDR,
        L4_DST_PORT,
        PROTOCOL,
        FLOW_START_MILLISECONDS,
        FLOW_END_MILLISECONDS,
        IN_PKTS,
        IN_BYTES,
        OUT_PKTS,
        OUT_BYTES,
        TCP_FLAGS,
        CLIENT_TCP_FLAGS,
        SERVER_TCP_FLAGS,
        count(*) AS occurrences
    FROM flows
    GROUP BY ALL
    HAVING count(*) > 1
), summary AS (
    SELECT
        data_file,
        count(*) AS raw_strict_key_groups,
        sum(occurrences) AS raw_strict_key_rows,
        sum(occurrences - 1) AS repeated_rows
    FROM near_keys
    GROUP BY data_file
)
SELECT
    sources.data_file,
    coalesce(summary.raw_strict_key_groups, 0) AS raw_strict_key_groups,
    coalesce(summary.raw_strict_key_rows, 0) AS raw_strict_key_rows,
    coalesce(summary.repeated_rows, 0) AS repeated_rows
FROM sources
LEFT JOIN summary USING (data_file)
ORDER BY sources.data_file;

-- Ordered-adjacency ±1ms lower bound for near-duplicate inspection.
-- It does not enumerate every qualifying pair. Even this lower bound exhibits
-- transitive bridges, so fuzzy connected components are rejected. DuckDB hash
-- is an exploratory tie-breaker only, not a portable ID.
WITH flows AS (
    SELECT
        filename AS data_file,
        IPV4_SRC_ADDR,
        L4_SRC_PORT,
        IPV4_DST_ADDR,
        L4_DST_PORT,
        PROTOCOL,
        IN_PKTS,
        IN_BYTES,
        OUT_PKTS,
        OUT_BYTES,
        TCP_FLAGS,
        CLIENT_TCP_FLAGS,
        SERVER_TCP_FLAGS,
        FLOW_START_MILLISECONDS,
        FLOW_END_MILLISECONDS,
        hash(
            IPV4_SRC_ADDR, L4_SRC_PORT, IPV4_DST_ADDR, L4_DST_PORT, PROTOCOL,
            IN_PKTS, IN_BYTES, OUT_PKTS, OUT_BYTES, TCP_FLAGS,
            CLIENT_TCP_FLAGS, SERVER_TCP_FLAGS
        ) AS key_hash,
        hash(
            IPV4_SRC_ADDR, L4_SRC_PORT, IPV4_DST_ADDR, L4_DST_PORT, PROTOCOL,
            IN_PKTS, IN_BYTES, OUT_PKTS, OUT_BYTES, TCP_FLAGS,
            CLIENT_TCP_FLAGS, SERVER_TCP_FLAGS, FLOW_START_MILLISECONDS,
            FLOW_END_MILLISECONDS
        ) AS row_hash
    FROM read_parquet(getvariable('parquet_file'), filename = true, union_by_name = true)
    WHERE FLOW_START_MILLISECONDS IS NOT NULL
      AND FLOW_END_MILLISECONDS IS NOT NULL
), adjacent AS (
    SELECT
        *,
        lag(FLOW_START_MILLISECONDS) OVER near_key AS prior_start,
        lag(FLOW_END_MILLISECONDS) OVER near_key AS prior_end
    FROM flows
    WINDOW near_key AS (
        PARTITION BY data_file, key_hash
        ORDER BY FLOW_START_MILLISECONDS, FLOW_END_MILLISECONDS, row_hash
    )
), chains AS (
    SELECT
        *,
        sum(CASE WHEN prior_start IS NULL
                      OR abs(FLOW_START_MILLISECONDS - prior_start) > 1
                      OR abs(FLOW_END_MILLISECONDS - prior_end) > 1
                 THEN 1 ELSE 0 END) OVER near_key AS chain_id
    FROM adjacent
    WINDOW near_key AS (
        PARTITION BY data_file, key_hash
        ORDER BY FLOW_START_MILLISECONDS, FLOW_END_MILLISECONDS, row_hash
    )
), chain_summary AS (
    SELECT
        data_file,
        key_hash,
        chain_id,
        count(*) AS chain_rows,
        max(FLOW_START_MILLISECONDS) - min(FLOW_START_MILLISECONDS) AS start_span_ms,
        max(FLOW_END_MILLISECONDS) - min(FLOW_END_MILLISECONDS) AS end_span_ms
    FROM chains
    GROUP BY data_file, key_hash, chain_id
)
SELECT
    data_file,
    count_if(chain_rows > 1) AS lower_bound_adjacency_chains,
    coalesce(sum(chain_rows) FILTER (WHERE chain_rows > 1), 0) AS lower_bound_adjacency_rows,
    count_if(chain_rows > 2 AND (start_span_ms > 1 OR end_span_ms > 1)) AS lower_bound_bridging_chains,
    coalesce(sum(chain_rows) FILTER (WHERE chain_rows > 2 AND (start_span_ms > 1 OR end_span_ms > 1)), 0) AS lower_bound_bridging_rows
FROM chain_summary
GROUP BY data_file
ORDER BY data_file;
