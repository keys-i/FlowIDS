SET VARIABLE parquet_file = coalesce(getvariable('parquet_file'), 'data/v3/pqt/NF-CICIDS2018-v3.parquet');

WITH long AS (
    UNPIVOT (
        SELECT
            coalesce(Label::VARCHAR, '<NULL>') AS label,
            coalesce(lower(nullif(trim(Attack), '')), '<NULL>') AS attack
        FROM read_parquet(getvariable('parquet_file'))
    )
    ON COLUMNS(*)
    INTO NAME target VALUE class_name
),
counts AS (
    SELECT target, class_name, count(*) AS flows
    FROM long
    GROUP BY ALL
),
shares AS (
    SELECT *, flows::DOUBLE / sum(flows) OVER (PARTITION BY target) AS prevalence
    FROM counts
)
SELECT
    target,
    class_name,
    flows,
    round(prevalence, 6) AS prevalence,
    round((max(flows) OVER (PARTITION BY target))::DOUBLE / flows, 2) AS majority_ratio,
    round(-log2(prevalence), 3) AS self_information_bits,
    round(
        sum(prevalence) OVER (PARTITION BY target ORDER BY flows DESC, class_name),
        6
    ) AS cumulative_prevalence
FROM shares
ORDER BY target, flows DESC, class_name;
