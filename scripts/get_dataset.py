#!/usr/bin/env python3
"""Download the NetFlow parquet file into the local data directory."""

from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "keys-i/netFlow-v3"
FILENAME = "v3/pqt/NF-CICIDS2018-v3.parquet"
DESTINATION = Path("data/NF-CICIDS2018-v3.parquet")

DESTINATION.parent.mkdir(parents=True, exist_ok=True)

cached_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    repo_type="dataset",
)

shutil.copy2(cached_path, DESTINATION)
print(f"downloaded: {DESTINATION.resolve()}")
