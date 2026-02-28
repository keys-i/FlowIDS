#!/usr/bin/env python3
"""Convert one or more CSV files into Parquet.

Examples:
    python3 scripts/convert_parquet.py -i data/in -o data/out
    python3 scripts/convert_parquet.py -i data/some.csv -o data/some.parquet
    python3 scripts/convert_parquet.py -i data/some.csv -o data/some.parquet --overwrite
"""

from __future__ import annotations

import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import click
import polars as pl
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.traceback import install

install(show_locals=False)

DEFAULT_COMPRESSION = "zstd"
SUPPORTED_COMPRESSIONS = ("zstd", "snappy", "gzip", "brotli", "lz4", "uncompressed")

default_workers = lambda n: max(1, min(os.cpu_count() or 4, n, 32))
is_glob = lambda s: any(token in s for token in "*?[]")
is_csv_file = lambda p: p.exists() and p.is_file() and p.suffix.lower() == ".csv"
human_sz = lambda n: next(
    f"{float(n) / (1024**i):.2f} {u}"
    for i, u in enumerate(("B", "KB", "MB", "GB", "TB"))
    if float(n) < 1024 ** (i + 1) or u == "TB"
)


def expand_input(raw_input: str) -> list[Path]:
    candidate = Path(raw_input)

    if is_glob(raw_input):
        return sorted(Path(match) for match in glob.glob(raw_input, recursive=True))

    if candidate.is_dir():
        return sorted(candidate.rglob("*.csv"))

    return [candidate]


def resolve_inputs(raw_inputs: tuple[str, ...]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()

    for raw_input in raw_inputs:
        for path in expand_input(raw_input):
            if not is_csv_file(path):
                continue

            absolute_path = path.resolve()
            if absolute_path in seen:
                continue

            seen.add(absolute_path)
            resolved.append(absolute_path)

    return sorted(resolved)


def build_output_path(
    input_path: Path, output_path: Path, all_inputs: list[Path]
) -> Path:
    output_path = output_path.resolve()

    if len(all_inputs) == 1:
        if output_path.suffix.lower() == ".parquet":
            return output_path
        return output_path / input_path.with_suffix(".parquet").name

    if output_path.suffix.lower() == ".parquet":
        raise click.ClickException(
            "When converting multiple CSV files or a directory, --output must be a directory."
        )

    try:
        common_root = Path(
            os.path.commonpath([str(path.parent) for path in all_inputs])
        )
    except ValueError:
        common_root = Path.cwd()

    try:
        relative = input_path.relative_to(common_root)
    except ValueError:
        relative = Path(input_path.name)

    return output_path / relative.with_suffix(".parquet")


def convert(
    input_path: Path,
    output_path: Path,
    compression: str,
    compression_lvl: int | None,
    infer_schema_length: int,
    overwrite: bool,
) -> tuple[str, Path, Path, int, int]:
    input_size = input_path.stat().st_size

    if output_path.exists() and not overwrite:
        output_size = output_path.stat().st_size
        return ("skipped", input_path, output_path, input_size, output_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_kwargs: dict[str, Any] = {"compression": compression}
    if compression_lvl is not None:
        write_kwargs["compression_level"] = compression_lvl

    lazy_frame = pl.scan_csv(
        str(input_path),
        separator=",",
        infer_schema_length=infer_schema_length,
    )

    if hasattr(lazy_frame, "sink_parquet"):
        lazy_frame.sink_parquet(str(output_path), **write_kwargs)
    else:
        pl.read_csv(
            str(input_path),
            separator=",",
            infer_schema_length=infer_schema_length,
        ).write_parquet(str(output_path), **write_kwargs)

    output_size = output_path.stat().st_size
    return ("converted", input_path, output_path, input_size, output_size)


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Convert one or more CSV files into Parquet.",
)
@click.option(
    "-i",
    "--input",
    "inputs",
    multiple=True,
    required=True,
    help="Input CSV file, directory, or glob pattern. Can be passed multiple times.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Output parquet file or output directory.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing Parquet output files.",
)
@click.option(
    "--compression",
    type=click.Choice(SUPPORTED_COMPRESSIONS, case_sensitive=False),
    default=DEFAULT_COMPRESSION,
    show_default=True,
    help="Parquet compression codec.",
)
@click.option(
    "--compression-level",
    "compression_level",
    type=int,
    default=None,
    help="Optional Parquet compression level.",
)
@click.option(
    "--infer-schema-length",
    type=int,
    default=10_000,
    show_default=True,
    help="Number of rows Polars uses for CSV schema inference.",
)
@click.option(
    "--quiet",
    is_flag=True,
    help="Disable verbose colourful output and progress bar.",
)
def main(
    inputs: tuple[str, ...],
    output_path: Path,
    overwrite: bool,
    compression: str,
    compression_level: int | None,
    infer_schema_length: int,
    quiet: bool,
) -> None:
    if infer_schema_length <= 0:
        raise click.BadParameter(
            "must be a positive integer.",
            param_hint="--infer-schema-length",
        )

    input_paths = resolve_inputs(inputs)
    if not input_paths:
        raise click.ClickException("No CSV inputs were found.")

    console = Console(quiet=quiet)
    workers = default_workers(len(input_paths))

    if not quiet:
        console.print(f"[bold blue]Found[/bold blue] {len(input_paths)} CSV file(s)")
        console.print(f"[bold blue]Workers[/bold blue] {workers}")
        console.print(f"[bold blue]Compression[/bold blue] {compression}")
        console.print("[bold blue]Scan mode[/bold blue] recursive")
        console.print()

    converted = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                convert,
                input_path,
                build_output_path(input_path, output_path, input_paths),
                compression,
                compression_level,
                infer_schema_length,
                overwrite,
            ): input_path
            for input_path in input_paths
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=quiet,
        ) as progress:
            task_id = progress.add_task("Converting CSV files", total=len(futures))

            for future in as_completed(futures):
                try:
                    status, input_path, output_file, input_size, output_size = (
                        future.result()
                    )

                    if status == "converted":
                        converted += 1
                        if not quiet:
                            console.print(
                                f"[green]Wrote[/green] {output_file} "
                                f"[dim]({human_sz(input_size)} -> {human_sz(output_size)})[/dim]"
                            )
                    else:
                        skipped += 1
                        if not quiet:
                            console.print(f"[yellow]Skipped[/yellow] {output_file}")

                except Exception as exc:
                    failed += 1
                    if not quiet:
                        console.print(f"[red]Failed[/red] {exc}")

                progress.advance(task_id, 1)

    if quiet:
        click.echo(f"converted={converted} skipped={skipped} failed={failed}")
    else:
        console.print()
        console.print(
            f"[bold]Done.[/bold] "
            f"[green]converted={converted}[/green] "
            f"[yellow]skipped={skipped}[/yellow] "
            f"[red]failed={failed}[/red]"
        )

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
