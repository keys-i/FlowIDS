SET VARIABLE parquet_file = coalesce(getvariable('parquet_file'), 'data/v3/pqt/NF-CICIDS2018-v3.parquet');

-- Total variation is 0 for identical attack mixes and 1 for disjoint mixes.
WITH daily AS (
    SELECT
        epoch_ms(FLOW_START_MILLISECONDS)::DATE AS flow_day,
        coalesce(lower(nullif(trim(Attack), '')), '<NULL>') AS attack,
        count(*) AS flows
    FROM read_parquet(getvariable('parquet_file'))
    WHERE FLOW_START_MILLISECONDS IS NOT NULL
    GROUP BY ALL
),
days AS (
    SELECT flow_day, sum(flows) AS day_flows
    FROM daily
    GROUP BY flow_day
),
overall AS (
    SELECT attack, sum(flows) AS attack_flows
    FROM daily
    GROUP BY attack
),
total AS (
    SELECT sum(flows) AS flows
    FROM daily
),
mix AS (
    SELECT
        days.flow_day,
        days.day_flows,
        coalesce(daily.flows, 0)::DOUBLE / days.day_flows AS day_share,
        overall.attack_flows::DOUBLE / total.flows AS overall_share
    FROM days
    CROSS JOIN overall
    CROSS JOIN total
    LEFT JOIN daily USING (flow_day, attack)
)
SELECT
    flow_day,
    day_flows,
    round(0.5 * sum(abs(day_share - overall_share)), 6) AS total_variation,
    round(max(abs(day_share - overall_share)), 6) AS largest_class_shift
FROM mix
GROUP BY flow_day, day_flows
ORDER BY flow_day;
