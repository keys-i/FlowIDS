SET VARIABLE parquet_file = coalesce(getvariable('parquet_file'), 'data/publish/data/NF-*-v3.parquet');

SELECT
    file_name AS data_file,
    num_rows AS rows,
    num_row_groups AS row_groups,
    round(file_size_bytes / 1024.0 / 1024.0, 2) AS file_mib,
    created_by,
    format_version
FROM parquet_file_metadata(getvariable('parquet_file'))
ORDER BY data_file;

-- Native parquet types: this deliberately does not read a unioned relation.
WITH schema_fields AS (
    SELECT file_name AS data_file, name AS column_name, type AS parquet_type
    FROM parquet_schema(getvariable('parquet_file'))
    WHERE name IS NOT NULL AND name <> 'root'
)
SELECT data_file, count(*) AS columns
FROM schema_fields
GROUP BY data_file
ORDER BY data_file;

WITH schema_fields AS (
    SELECT file_name AS data_file, name AS column_name, type AS parquet_type
    FROM parquet_schema(getvariable('parquet_file'))
    WHERE name IS NOT NULL AND name <> 'root'
), drift AS (
    SELECT
        column_name,
        count(DISTINCT parquet_type) AS type_variants,
        string_agg(DISTINCT parquet_type, ', ' ORDER BY parquet_type) AS observed_types
    FROM schema_fields
    GROUP BY column_name
)
SELECT
    schema_fields.data_file, schema_fields.column_name, schema_fields.parquet_type,
    drift.type_variants, drift.observed_types
FROM schema_fields
JOIN drift USING (column_name)
WHERE drift.type_variants > 1
ORDER BY schema_fields.column_name, schema_fields.data_file;
