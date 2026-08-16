SET VARIABLE parquet_file = coalesce(getvariable('parquet_file'), 'data/publish/data/NF-*-v3.parquet');

WITH long AS (
    UNPIVOT (
        SELECT
            filename AS data_file,
            coalesce(Label::VARCHAR, '<NULL>') AS label,
            coalesce(lower(nullif(trim(Attack), '')), '<NULL>') AS attack
        FROM read_parquet(getvariable('parquet_file'), filename = true, union_by_name = true)
    )
    ON COLUMNS(* EXCLUDE (data_file))
    INTO NAME target VALUE class_name
), counts AS (
    SELECT data_file, target, class_name, count(*) AS flows
    FROM long
    GROUP BY ALL
), shares AS (
    SELECT *, flows::DOUBLE / sum(flows) OVER (PARTITION BY data_file, target) AS prevalence
    FROM counts
)
SELECT
    data_file, target, class_name, flows,
    round(prevalence, 6) AS prevalence,
    round((max(flows) OVER (PARTITION BY data_file, target))::DOUBLE / flows, 2) AS majority_ratio,
    round(-log2(prevalence), 3) AS self_information_bits,
    round(sum(prevalence) OVER (PARTITION BY data_file, target ORDER BY flows DESC, class_name), 6) AS cumulative_prevalence
FROM shares
ORDER BY data_file, target, flows DESC, class_name;
