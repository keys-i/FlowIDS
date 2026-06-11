"""Load TOML configuration into attribute-accessible namespaces."""

from __future__ import annotations

import tomllib
from datetime import date, datetime, time
from pathlib import Path
from types import SimpleNamespace

type TomlValue = (
    bool | int | float | str | date | datetime | time | list[TomlValue] | dict[str, TomlValue]
)


class Config(SimpleNamespace):
    """Expose nested TOML tables through attributes."""

    def __init__(self, values: dict[str, TomlValue]) -> None:
        """Build a configuration namespace from TOML values."""
        super().__init__(
            {
                key: Config(value) if isinstance(value, dict) else value
                for key, value in values.items()
            }
        )

    @classmethod
    def load(cls, path: str | Path = "tools/config/m0.matched.toml") -> Config:
        """Load configuration from a TOML file."""
        with Path(path).open("rb") as file:
            return cls(tomllib.load(file))
