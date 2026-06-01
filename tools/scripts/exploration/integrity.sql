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
