"""
Helper functions to make main flow stay clean.
"""

import tomllib
from pathlib import Path


# == config area ==
class Config:
    def __init__(self, data: dict):
        for k, v in data.items():
            setattr(self, k, self._convert(v))

    @classmethod
    def _convert(cls, val):
        if isinstance(val, dict):
            return cls(val)
        if isinstance(val, list):
            return [cls._convert(v) for v in val]
        return val

    def __repr__(self):
        return f"Config({self.__dict__})"


def load_config(path: str | Path) -> Config:
    with Path(path).open("rb") as f:
        data = tomllib.load(f)
    return Config(data)
