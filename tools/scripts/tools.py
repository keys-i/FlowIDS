"""Run the repository's local development commands."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "tools" / "config"
RUFF = CONFIG / "ruff.toml"
BASEDPYRIGHT = CONFIG / "basedpyright.json"
MODEL_IGNORE = {RUFF.name, BASEDPYRIGHT.name, "slurm.toml"}

T = TypeVar("T")


def run(*command: str | Path) -> None:
    """Run a command from the repository root."""
    _ = subprocess.run(command, cwd=ROOT, check=True)


fmt = lambda: run("ruff", "format", ".", "--config", RUFF)  # noqa: E731


def clean() -> None:
    """Remove local build and Python cache files."""
    paths = {
        ROOT / name
        for name in (
            ".ruff_cache",
            ".pytest_cache",
            ".mypy_cache",
            ".basedpyright",
            "build",
            "dist",
        )
    }
    paths.update(ROOT.glob("*.egg-info"))
    for source in (ROOT / "src", ROOT / "tools"):
        paths.update(source.rglob("__pycache__"))

    removed = 0
    for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            continue
        removed += 1
    print(f"Removed {removed} cache path(s).")


def lint() -> None:
    """Run formatting, linting, and static type checks."""
    run("ruff", "format", "--check", ".", "--config", RUFF)
    run("ruff", "check", ".", "--config", RUFF)
    run("basedpyright", "--project", BASEDPYRIGHT)
    run(sys.executable, "-m", "src.model.check")


def choose(prompt: str, options: list[T]) -> T:
    """Return an interactively selected option."""
    print(f"\n{prompt}")
    for index, option in enumerate(options, 1):
        print(f"  {index}. {option}")

    while True:
        answer = input("Select: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        print(f"Enter a number from 1 to {len(options)}.")


def config_name(path: Path) -> str:
    """Return a model-menu label from a TOML configuration."""
    with path.open("rb") as file:
        description = tomllib.load(file).get("description")
    if not isinstance(description, str) or not description.strip():
        raise SystemExit(f"{path.name} needs a root description string.")
    return f"{path.stem.replace('.', ' ').title()} - {description.strip()}"


def model() -> None:
    """Choose a model configuration and action, then run it."""
    paths = sorted(path for path in CONFIG.glob("*.toml") if path.name not in MODEL_IGNORE)
    configs = {config_name(path): path for path in paths}
    if not configs:
        raise SystemExit(f"No model configs found in {CONFIG.relative_to(ROOT)}.")

    try:
        config = configs[choose("Configuration", list(configs))]
        action = choose("Action", ["train", "evaluate"])
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\nCancelled.") from None

    run(sys.executable, "-m", "src.main", action, "--config", config)


COMMANDS: dict[str, Callable[[], None]] = {
    "clean": clean,
    "fmt": fmt,
    "lint": lint,
    "model": model,
}


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} <{'|'.join(COMMANDS)}>")
    COMMANDS[sys.argv[1]]()
