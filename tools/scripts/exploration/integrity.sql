SET VARIABLE parquet_file = coalesce(getvariable('parquet_file'), 'data/v3/pqt/NF-CICIDS2018-v3.parquet');

WITH counts AS (
    SELECT
        count(*) AS rows,
        count_if(
            FLOW_START_MILLISECONDS IS NULL OR FLOW_END_MILLISECONDS IS NULL
        ) AS missing_timestamps,
        count_if(FLOW_END_MILLISECONDS < FLOW_START_MILLISECONDS) AS end_before_start,
        count_if(
            abs(
                FLOW_END_MILLISECONDS
                - FLOW_START_MILLISECONDS
                - FLOW_DURATION_MILLISECONDS
            ) > 1
        ) AS duration_mismatch,
        count_if(Label IS NULL OR nullif(trim(Attack), '') IS NULL) AS missing_target,
        count_if(Label NOT IN (0, 1)) AS invalid_binary_label,
        count_if(
            Label IS NOT NULL
            AND Attack IS NOT NULL
            AND ((Label = 0) <> (lower(trim(Attack)) = 'benign'))
        ) AS inconsistent_target,
        count_if(
            IN_BYTES < 0 OR OUT_BYTES < 0 OR IN_PKTS < 0 OR OUT_PKTS < 0
        ) AS negative_traffic,
        count_if(
            SHORTEST_FLOW_PKT > LONGEST_FLOW_PKT OR MIN_IP_PKT_LEN > MAX_IP_PKT_LEN
        ) AS invalid_packet_bounds,
        count_if(
            SRC_TO_DST_IAT_MIN > SRC_TO_DST_IAT_AVG
            OR SRC_TO_DST_IAT_AVG > SRC_TO_DST_IAT_MAX
            OR DST_TO_SRC_IAT_MIN > DST_TO_SRC_IAT_AVG
            OR DST_TO_SRC_IAT_AVG > DST_TO_SRC_IAT_MAX
            OR SRC_TO_DST_IAT_STDDEV < 0
            OR DST_TO_SRC_IAT_STDDEV < 0
        ) AS invalid_iat_statistics
    FROM read_parquet(getvariable('parquet_file'))
),
failures AS (
    UNPIVOT counts
    ON COLUMNS(* EXCLUDE (rows))
    INTO NAME check_name VALUE violations
)
SELECT
    check_name,
    violations,
    round(violations::DOUBLE / rows, 8) AS violation_rate
FROM failures
ORDER BY violation_rate DESC;
