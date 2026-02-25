"""
NetFlow preprocessing with leakage-safe chronological splitting and train-only transforms.

This module provides a minimal preprocessing pipeline for NetFlow-based intrusion
detection experiments and self-supervised sequence modeling:

1) Load the raw NetFlow table.
2) Sort rows by a trusted flow timestamp.
3) Remove exact duplicate records before splitting.
4) Define the approved non-leaky feature set and drop direct identifiers,
   label text, scenario artifacts, and other obvious leakage carriers.
5) Split the data chronologically into train, validation, and test.
6) Fit all preprocessing steps on the training split only.
7) Apply median imputation, optional missingness indicators, selected `log1p`
   transforms for heavy-tailed flow features, and scaling where required.
8) Build cleaned model-ready arrays and temporally ordered windows/sequences
   for self-supervised Transformer training.

The module may also expose helpers for:
- duplicate and cross-split overlap checks,
- downstream binary and coarse attack-family label mappings,
- sequence construction for temporally local training examples,
- saving cleaned splits and fitted preprocessing artifacts.

Created By: Keys-i
ID: 49088276
"""

import re

import polars as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Feature Columns
TIMESTAMP_COL = "FLOW_START_MILLISECONDS"
CONTINUOUS_COLS = [
    # traffic volume
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    # packet size shape
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    # timing
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
    # rate
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    # tcp behaviour
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT",
    # directionality
    "DURATION_IN",
    "DURATION_OUT",
    # routing
    "MIN_TTL",
    "MAX_TTL",
    # packet size distribution
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
]
CODE_COLS = [
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
]
NUMERICAL_COLS = [*CONTINUOUS_COLS, *CODE_COLS]
CATEGORY_COLS = [
    "PROTOCOL",
    "L7_PROTO",
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_SRC_PORT",
    "L4_DST_PORT",
]
LABEL_COLS = ["Label", "Attack"]

# label, 0=benign,1=malicious
# attack family Map
ATTACK_FAMILY_PATTERNS = [
    (r"^benign$", "benign"),
    (r"^(ddos_attack-|ddos_attacks-)", "ddos"),
    (r"^dos_attacks-", "dos"),
    (r"^(brute_force_-xss|brute_force_-web|sql_injection)$", "web_attack"),
    (r"^(ftp-bruteforce|ssh-bruteforce)$", "credential_attack"),
    (r"^bot$", "bot"),
    (r"^infilteration$", "infiltration"),
]


def build_attack_maps(attacks: list[str]) -> dict:
    norm = lambda x: x.strip().lower()
    fam = lambda a: next(f for p, f in ATTACK_FAMILY_PATTERNS if re.search(p, a))

    attacks = sorted({norm(a) for a in attacks})
    attack_to_family = {a: fam(a) for a in attacks}
    families = list(
        dict.fromkeys(f for _, f in ATTACK_FAMILY_PATTERNS if f in attack_to_family.values())
    )
    family_to_id = {f: i for i, f in enumerate(families)}
    attack_to_code = {
        a: family_to_id[attack_to_family[a]] * 10 + i
        for f in families
        for i, a in enumerate(sorted(x for x in attacks if attack_to_family[x] == f))
    }

    return {
        "attack_to_family": attack_to_family,
        "family_to_id": family_to_id,
        "attack_to_code": attack_to_code,
    }


class NFDataset(Dataset):
    def __init__(self, csv_path):
        self.lf = pl.scan_parquet(csv_path)

        # split dataframes
        self.train_df = pl.DataFrame()
        self.val_df = pl.DataFrame()
        self.test_df = pl.DataFrame()

        # fitted preprocessing
        self.missing_indicator_cols = []
        self.imputed_values = {}
        self.log_features = []
        self.scaler = {"mean": {}, "std": {}}
        self.label_maps = {}

    def __len__(self):
        raise NotImplementedError

    def clean(self):
        self.lf = self.lf.select([TIMESTAMP_COL, *NUMERICAL_COLS, *CATEGORY_COLS, *LABEL_COLS])
        self.lf = self.lf.unique().sort(TIMESTAMP_COL)
        return self

    def split(self, train_ratio=0.7, val_ratio=0.15):
        if train_ratio + val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be less than 1.0")

        df = self.lf.collect(engine="streaming")

        train, tmp = train_test_split(df, train_size=train_ratio, shuffle=False)

        val_size_in_tmp = val_ratio / (1.0 - train_ratio)
        val, test = train_test_split(tmp, train_size=val_size_in_tmp, shuffle=False)

        self.train_df = pl.DataFrame(train)
        self.val_df = pl.DataFrame(val)
        self.test_df = pl.DataFrame(test)

        return self

    def fit(self):
        if self.train_df.is_empty():
            raise ValueError("train_df is not set. Run split() before fit().")

        # label mappings for train only
        attacks = self.train_df["Attack"].drop_nulls().to_list()
        self.label_maps = build_attack_maps(attacks)

        stats = lambda f, df=self.train_df: df.select(
            [f(pl.col(col)).alias(col) for col in NUMERICAL_COLS]
        ).to_dicts()[0]

        # train medians for imputation
        self.imputed_values = stats(lambda c: c.median())

        # columns missingness indicators
        missing_rates = stats(lambda c: c.is_null().mean())
        self.missing_indicator_cols = [
            col for col, rate in missing_rates.items() if rate is not None and rate > 0
        ]

        # heavy tailed continuous features
        heavy_keywords = ("bytes", "pkts", "pkt", "duration", "iat", "throughput")
        self.log_features = [
            col for col in CONTINUOUS_COLS if any(k in col.lower() for k in heavy_keywords)
        ]

        # fit scaler stats on train
        train_prepped = self.train_df.with_columns(
            [pl.col(col).fill_null(self.imputed_values[col]).alias(col) for col in NUMERICAL_COLS]
        ).with_columns([pl.col(col).log1p().alias(col) for col in self.log_features])

        fitted = lambda f: train_prepped.select(
            [f(pl.col(col)).alias(col) for col in NUMERICAL_COLS]
        ).to_dicts()[0]

        self.scaler = {
            "mean": fitted(lambda c: c.mean()),
            "std": fitted(lambda c: c.std()),
        }

        self.scaler["std"] = {
            col: (std if std not in (None, 0.0) else 1.0) for col, std in self.scaler["std"].items()
        }

        return self

    def transform(self):
        if self.train_df.is_empty() or self.val_df.is_empty() or self.test_df.is_empty():
            raise ValueError("Run split() before transform().")
        if not self.imputed_values or not self.scaler["mean"] or not self.scaler["std"]:
            raise ValueError("Run fit() before transform().")

        feature_cols = NUMERICAL_COLS + [f"{col}_missing" for col in self.missing_indicator_cols]

        # apply train fitted preprocessing to each split
        prep = lambda df: (
            # missingness indicators
            df.with_columns(
                [
                    pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_missing")
                    for col in self.missing_indicator_cols
                ]
            )
            # median imputation using train medians
            .with_columns(
                [
                    pl.col(col).fill_null(self.imputed_values[col]).alias(col)
                    for col in NUMERICAL_COLS
                ]
            )
            # log1p heavy tailed features
            .with_columns([pl.col(col).log1p().alias(col) for col in self.log_features])
            # standardize using train fitted mean/std
            .with_columns(
                [
                    ((pl.col(col) - self.scaler["mean"][col]) / self.scaler["std"][col]).alias(col)
                    for col in NUMERICAL_COLS
                ]
            )
            # keep ordering/grouping/label cols plus final model features
            .select([TIMESTAMP_COL, *CATEGORY_COLS, *LABEL_COLS, *feature_cols])
        )

        self.train_df = prep(self.train_df)
        self.val_df = prep(self.val_df)
        self.test_df = prep(self.test_df)

        return self

    def build_seq_inputs(self):
        if self.train_df.is_empty() or self.val_df.is_empty() or self.test_df.is_empty():
            raise ValueError("run transform() before building sequential inputs")

        feature_cols = NUMERICAL_COLS + [f"{col}_missing" for col in self.missing_indicator_cols]
        input_cols = [TIMESTAMP_COL, *CATEGORY_COLS, *feature_cols]

        build = lambda df: df.select(input_cols).sort([*CATEGORY_COLS, TIMESTAMP_COL])

        self.train_df = build(self.train_df)
        self.val_df = build(self.val_df)
        self.test_df = build(self.test_df)

        return self


def build_dataset(csv_path, train_ratio=0.7, val_ratio=0.15):
    dataset = NFDataset(csv_path)

    dataset.clean()
    dataset.split(train_ratio=train_ratio, val_ratio=val_ratio)
    dataset.fit()
    dataset.transform()
    dataset.build_seq_inputs()

    return dataset.train_df, dataset.val_df, dataset.test_df, dataset
