SET VARIABLE parquet_file = coalesce(getvariable('parquet_file'), 'data/publish/data/NF-*-v3.parquet');

-- I(feature; target) / H(target): 0 is no observed association; 1 is in-sample determination.
-- Treat high-cardinality scores as screening signals until group/time holdout confirmation.
WITH fields AS (
    SELECT
        filename AS data_file,
        coalesce(Label::VARCHAR, '<NULL>') AS label,
        coalesce(lower(nullif(trim(Attack), '')), '<NULL>') AS attack,
        coalesce(nullif(trim(IPV4_SRC_ADDR), ''), '<NULL>') AS src_ip,
        coalesce(nullif(trim(IPV4_DST_ADDR), ''), '<NULL>') AS dst_ip,
        coalesce(L4_SRC_PORT::VARCHAR, '<NULL>') AS src_port,
        coalesce(L4_DST_PORT::VARCHAR, '<NULL>') AS dst_port,
        coalesce(PROTOCOL::VARCHAR, '<NULL>') AS protocol,
        coalesce(L7_PROTO::VARCHAR, '<NULL>') AS app_protocol,
        coalesce(DNS_QUERY_ID::VARCHAR, '<NULL>') AS dns_query_id,
        coalesce(epoch_ms(FLOW_END_MILLISECONDS)::DATE::VARCHAR, '<NULL>') AS completion_day,
        concat_ws('|', coalesce(nullif(trim(IPV4_SRC_ADDR), ''), '<NULL>'), coalesce(nullif(trim(IPV4_DST_ADDR), ''), '<NULL>')) AS endpoint_fingerprint,
        concat_ws('|', coalesce(nullif(trim(IPV4_SRC_ADDR), ''), '<NULL>'), coalesce(L4_SRC_PORT::VARCHAR, '<NULL>'), coalesce(nullif(trim(IPV4_DST_ADDR), ''), '<NULL>'), coalesce(L4_DST_PORT::VARCHAR, '<NULL>'), coalesce(PROTOCOL::VARCHAR, '<NULL>')) AS five_tuple_fingerprint
    FROM read_parquet(getvariable('parquet_file'), filename = true, union_by_name = true)
), long AS (
    UNPIVOT fields
    ON COLUMNS(* EXCLUDE (data_file, label, attack))
    INTO NAME feature VALUE value
), targets AS (
    UNPIVOT long
    ON label, attack
    INTO NAME target VALUE class_name
)
SELECT
    data_file, target, feature,
    count(DISTINCT value) AS levels,
    round(count(*)::DOUBLE / count(DISTINCT value), 1) AS rows_per_level,
    round((entropy(value) + entropy(class_name) - entropy(struct_pack(value := value, class_name := class_name))) / nullif(entropy(class_name), 0), 4) AS target_entropy_reduction
FROM targets
GROUP BY data_file, target, feature
ORDER BY data_file, target, target_entropy_reduction DESC, feature;
