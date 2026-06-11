"""Load TOML configuration into recursive attribute namespaces"""

from __future__ import annotations

import tomllib
from datetime import date, datetime, time
from pathlib import Path
from types import SimpleNamespace

type TomlValue = (
    bool | int | float | str | date | datetime | time | list[TomlValue] | dict[str, TomlValue]
)


class Config(SimpleNamespace):
    """Expose one nested TOML table through attributes"""

    def __init__(self, values: dict[str, TomlValue]) -> None:
        """Build a namespace from values already parsed from TOML

        Args:
            values: Table mapping with TOML scalar, list, or nested table
                values
        """
        super().__init__(
            {
                key: Config(value) if isinstance(value, dict) else value
                for key, value in values.items()
            }
        )

    @classmethod
    def load(cls, path: str | Path = "tools/config/m0.matched.toml") -> Config:
        """Load a TOML file into recursively accessible configuration tables

        Args:
            path: TOML file to read, relative to the current working directory
                or absolute

        Returns:
            A `Config` containing the parsed root table

        Raises:
            FileNotFoundError: If `path` does not exist
            tomllib.TOMLDecodeError: If the file is not valid TOML
        """
        with Path(path).open("rb") as file:
            return cls(tomllib.load(file))
