SET VARIABLE parquet_file = coalesce(getvariable('parquet_file'), 'data/v3/pqt/NF-CICIDS2018-v3.parquet');

-- I(feature; Attack) / H(Attack): 0 is no observed association; 1 is in-sample determination.
-- Treat high-cardinality scores as screening signals until group/time holdout confirmation.
WITH fields AS (
    SELECT
        coalesce(lower(nullif(trim(Attack), '')), '<NULL>') AS attack,
        coalesce(nullif(trim(IPV4_SRC_ADDR), ''), '<NULL>') AS src_ip,
        coalesce(nullif(trim(IPV4_DST_ADDR), ''), '<NULL>') AS dst_ip,
        coalesce(L4_SRC_PORT::VARCHAR, '<NULL>') AS src_port,
        coalesce(L4_DST_PORT::VARCHAR, '<NULL>') AS dst_port,
        coalesce(PROTOCOL::VARCHAR, '<NULL>') AS protocol,
        coalesce(L7_PROTO::VARCHAR, '<NULL>') AS app_protocol,
        coalesce(DNS_QUERY_ID::VARCHAR, '<NULL>') AS dns_query_id,
        coalesce(epoch_ms(FLOW_START_MILLISECONDS)::DATE::VARCHAR, '<NULL>') AS flow_day
    FROM read_parquet(getvariable('parquet_file'))
),
long AS (
    UNPIVOT fields
    ON COLUMNS(* EXCLUDE (attack))
    INTO NAME feature VALUE value
)
SELECT
    feature,
    count(DISTINCT value) AS levels,
    round(count(*)::DOUBLE / count(DISTINCT value), 1) AS rows_per_level,
    round(
        (
            entropy(value)
            + entropy(attack)
            - entropy(struct_pack(value := value, attack := attack))
        ) / nullif(entropy(attack), 0),
        4
    ) AS attack_entropy_reduction
FROM long
GROUP BY feature
ORDER BY attack_entropy_reduction DESC;
