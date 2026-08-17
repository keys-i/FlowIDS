"""Convert CSV files to Parquet.

Examples:
    pixi run python tools/scripts/convert.py -i data/in -o data/out
    pixi run python tools/scripts/convert.py -i data/file.csv -o data/file.parquet --overwrite
"""

import glob
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import click
import polars as pl
from rich.console import Console
from rich.progress import track
from rich.traceback import install

_ = install(show_locals=False)

type Compression = Literal["zstd", "snappy", "gzip", "brotli", "lz4", "uncompressed"]
COMPRESSIONS: tuple[Compression, ...] = ("zstd", "snappy", "gzip", "brotli", "lz4", "uncompressed")


def human_size(size: int) -> str:
    return next(
        f"{size / 1024**power:.2f} {unit}"
        for power, unit in enumerate(("B", "KB", "MB", "GB", "TB"))
        if size < 1024 ** (power + 1) or unit == "TB"
    )


def expand(raw: str) -> list[Path]:
    path = Path(raw)
    if glob.has_magic(raw):
        return [Path(match) for match in glob.glob(raw, recursive=True)]
    return list(path.rglob("*.csv")) if path.is_dir() else [path]


def resolve_inputs(raw_inputs: tuple[str, ...]) -> list[Path]:
    return sorted(
        {
            path.resolve()
            for raw in raw_inputs
            for path in expand(raw)
            if path.is_file() and path.suffix.lower() == ".csv"
        }
    )


def destination(source: Path, output: Path, sources: list[Path]) -> Path:
    output = output.resolve()
    if len(sources) == 1:
        return (
            output
            if output.suffix.lower() == ".parquet"
            else output / source.with_suffix(".parquet").name
        )
    if output.suffix.lower() == ".parquet":
        raise click.ClickException("--output must be a directory for multiple inputs.")
    root = Path(os.path.commonpath([path.parent for path in sources]))
    return output / source.relative_to(root).with_suffix(".parquet")


def convert(
    source: Path,
    output: Path,
    compression: Compression,
    compression_level: int | None,
    infer_schema_length: int,
    overwrite: bool,
) -> tuple[str, Path, int, int]:
    input_size = source.stat().st_size
    if output.exists() and not overwrite:
        return "skipped", output, input_size, output.stat().st_size

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".parquet", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame = pl.scan_csv(source, infer_schema_length=infer_schema_length)
        if hasattr(frame, "sink_parquet"):
            _ = frame.sink_parquet(
                temporary, compression=compression, compression_level=compression_level
            )
        else:
            frame.collect().write_parquet(
                temporary, compression=compression, compression_level=compression_level
            )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return "converted", output, input_size, output.stat().st_size


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-i",
    "--input",
    "inputs",
    multiple=True,
    required=True,
    help="CSV file, directory, or glob; repeat for more inputs.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Parquet file or output directory.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs.")
@click.option(
    "--compression",
    type=click.Choice(COMPRESSIONS, case_sensitive=False),
    default="zstd",
    show_default=True,
)
@click.option("--compression-level", type=int)
@click.option("--infer-schema-length", type=int, default=10_000, show_default=True)
@click.option("--quiet", is_flag=True, help="Print only the final counts.")
def main(
    inputs: tuple[str, ...],
    output_path: Path,
    overwrite: bool,
    compression: Compression,
    compression_level: int | None,
    infer_schema_length: int,
    quiet: bool,
) -> None:
    if infer_schema_length <= 0:
        raise click.BadParameter("must be positive", param_hint="--infer-schema-length")

    sources = resolve_inputs(inputs)
    if not sources:
        raise click.ClickException("No CSV inputs were found.")

    workers = min(os.cpu_count() or 4, len(sources), 32)
    console = Console(quiet=quiet)
    if not quiet:
        details = f"workers={workers}; compression={compression}; scan=recursive"
        console.print(f"[blue]Found[/blue] {len(sources)} CSV file(s); {details}")

    counts = {"converted": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {
            pool.submit(
                convert,
                source,
                destination(source, output_path, sources),
                compression,
                compression_level,
                infer_schema_length,
                overwrite,
            ): source
            for source in sources
        }
        for job in track(
            as_completed(jobs),
            total=len(jobs),
            description="Converting",
            console=console,
            disable=quiet,
        ):
            try:
                status, output, input_size, output_size = job.result()
                counts[status] += 1
                if not quiet:
                    verb = "Wrote" if status == "converted" else "Skipped"
                    console.print(
                        f"{verb} {output} ({human_size(input_size)} -> {human_size(output_size)})"
                    )
            except Exception as error:
                counts["failed"] += 1
                if not quiet:
                    console.print(f"[red]Failed[/red] {jobs[job]}: {error}")

    summary = " ".join(f"{name}={count}" for name, count in counts.items())
    click.echo(summary) if quiet else console.print(f"[bold]Done.[/bold] {summary}")
    raise SystemExit(bool(counts["failed"]))


if __name__ == "__main__":
    main()
