SET VARIABLE parquet_file = coalesce(getvariable('parquet_file'), 'data/publish/data/NF-*-v3.parquet');

-- Total variation is 0 for identical target mixes and 1 for disjoint mixes.
WITH long AS (
    UNPIVOT (
        SELECT
            filename AS data_file,
            epoch_ms(FLOW_END_MILLISECONDS)::DATE AS completion_day,
            coalesce(Label::VARCHAR, '<NULL>') AS label,
            coalesce(lower(nullif(trim(Attack), '')), '<NULL>') AS attack
        FROM read_parquet(getvariable('parquet_file'), filename = true, union_by_name = true)
        WHERE FLOW_END_MILLISECONDS IS NOT NULL
    )
    ON COLUMNS(* EXCLUDE (data_file, completion_day))
    INTO NAME target VALUE class_name
), daily AS (
    SELECT data_file, target, completion_day, class_name, count(*) AS flows
    FROM long
    GROUP BY ALL
), days AS (
    SELECT data_file, target, completion_day, sum(flows) AS day_flows
    FROM daily
    GROUP BY ALL
), overall AS (
    SELECT data_file, target, class_name, sum(flows) AS class_flows
    FROM daily
    GROUP BY ALL
), totals AS (
    SELECT data_file, target, sum(flows) AS flows
    FROM daily
    GROUP BY ALL
), mix AS (
    SELECT
        days.data_file, days.target, days.completion_day, days.day_flows,
        coalesce(daily.flows, 0)::DOUBLE / days.day_flows AS day_share,
        overall.class_flows::DOUBLE / totals.flows AS overall_share
    FROM days
    JOIN totals USING (data_file, target)
    JOIN overall USING (data_file, target)
    LEFT JOIN daily USING (data_file, target, completion_day, class_name)
)
SELECT
    data_file, target, completion_day, day_flows,
    round(0.5 * sum(abs(day_share - overall_share)), 6) AS total_variation,
    round(max(abs(day_share - overall_share)), 6) AS largest_class_shift
FROM mix
GROUP BY ALL
ORDER BY data_file, target, completion_day;
