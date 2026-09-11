"""Run formatting, linting, and cache cleanup"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "tools" / "config"
RUFF = CONFIG / "ruff.toml"
BASEDPYRIGHT = CONFIG / "basedpyright.json"


def run(*command: str | Path) -> None:
    """Run a command from the repository root

    Args:
        command: Executable name and arguments passed without shell expansion
    """
    _ = subprocess.run(command, cwd=ROOT, check=True)


fmt = lambda: run("ruff", "format", "src", "tools", "--config", RUFF)  # noqa: E731


def clean() -> None:
    """Remove local build and Python cache files

    Deletes generated cache directories and Python bytecode beneath source and
    tooling directories
    """
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
    """Run formatting, linting, type checks, and self-checks

    Checks data splits, metrics, and M1/M2 forward/backward passes after static checks pass
    """
    run("ruff", "format", "--check", "src", "tools", "--config", RUFF)
    run("ruff", "check", "src", "tools", "--config", RUFF)
    run("basedpyright", "--project", BASEDPYRIGHT)
    run(sys.executable, "-m", "src.data.load")
    run(sys.executable, "-m", "src.metrics")
    run(sys.executable, "-m", "tools.scripts.test_timing")
    run(sys.executable, "-m", "src.m1.network")
    run(sys.executable, "-m", "src.m2.data")
    run(sys.executable, "-m", "src.m2.network")


COMMANDS = {
    "clean": clean,
    "fmt": fmt,
    "lint": lint,
}


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} <{'|'.join(COMMANDS)}>")
    COMMANDS[sys.argv[1]]()
