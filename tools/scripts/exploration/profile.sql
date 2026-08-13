SET VARIABLE parquet_file = coalesce(getvariable('parquet_file'), 'data/v3/pqt/NF-CICIDS2018-v3.parquet');

SELECT
    num_rows AS rows,
    num_row_groups AS row_groups,
    round(file_size_bytes / 1024.0 / 1024.0, 2) AS file_mib,
    created_by,
    format_version
FROM parquet_file_metadata(getvariable('parquet_file'));

SUMMARIZE
SELECT *
FROM read_parquet(getvariable('parquet_file'));
