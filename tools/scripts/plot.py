"""Export result comparisons, learning curves, and error examples to PNG, SVG, and PDF"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import matplotlib
import polars as pl
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

matplotlib.use("Agg")


@dataclass
class Run:
    """One evaluated model, with optional metadata for seed comparisons"""

    path: Path
    dataset: str
    name: str
    metrics: dict[str, float | int | None]
    seed: int | None
    settings: str | None


def read_runs(root: Path) -> list[Run]:
    """Read completed runs and retain settings needed to compare seeds"""
    runs: list[Run] = []
    for path in sorted(root.rglob("metrics.json")):
        status = path.with_name("status.json")
        if status.exists() and not json.loads(status.read_text())["complete"]:
            print(f"{path.parent}: incomplete run; skipping")
            continue
        results = json.loads(path.read_text())
        if len(results) != 1:
            raise ValueError(f"{path} must contain exactly one dataset")
        dataset, variants = next(iter(results.items()))
        config_path = path.with_name("config.json")
        config = json.loads(config_path.read_text()) if config_path.exists() else None
        seed = None
        if config is not None:
            seed = config["run"].pop("seed")
            _ = config["run"].pop("output", None)
            _ = config["data"].pop("root", None)
        for variant, metrics in variants.items():
            if variant != "always_benign":
                family = (
                    "M0"
                    if variant in {"base", "small", "matched"}
                    else "M1"
                    if variant in {"reconstruct", "teacher"}
                    else "M2"
                )
                name = f"{family}-{variant}"
                if config is not None and "hours" in config["run"]:
                    name += f" ({config['run']['hours']:g}h)"
                runs.append(
                    Run(
                        path.parent,
                        dataset,
                        name,
                        metrics,
                        seed,
                        json.dumps(config, sort_keys=True) if config else None,
                    )
                )
    if not runs:
        raise ValueError(f"No completed runs beneath {root}; evaluate a model first")
    return runs


def save(figure: Figure, path: Path, report: PdfPages) -> None:
    """Save slide-ready images and append the figure to the combined report"""
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=180)
    report.savefig(figure)
    figure.clear()


def compare(runs: list[Run], output: Path, report: PdfPages) -> None:
    """Show means and sample SD across distinct seeds with matching settings"""
    rows: list[dict[str, str | float | int | None]] = []
    for dataset in sorted({run.dataset for run in runs}):
        names = sorted({run.name for run in runs if run.dataset == dataset})
        figure = Figure(figsize=(13, 4), layout="constrained")
        for panel, (metric, title) in enumerate(
            (
                ("auprc", "Average precision"),
                ("auroc", "ROC AUC"),
                ("tpr_at_fpr_1e-3", "Recall at FPR ≤ 0.001"),
            ),
            1,
        ):
            axis = figure.add_subplot(1, 3, panel)
            for index, name in enumerate(names):
                group = [run for run in runs if run.dataset == dataset and run.name == name]
                seeds = [run.seed for run in group]
                known = all(seed is not None for seed in seeds)
                if known and len(set(seeds)) != len(seeds):
                    raise ValueError(
                        f"{dataset}/{name}: duplicate seeds; choose one copy of each run"
                    )
                if len(group) > 1 and len({run.settings for run in group}) != 1:
                    raise ValueError(
                        f"{dataset}/{name}: settings differ; plot these runs separately"
                    )
                values = [
                    float(value) for run in group if (value := run.metrics[metric]) is not None
                ]
                deviation = stdev(values) if len(values) > 1 and known else None
                average = mean(values) if values else None
                rows.append(
                    {
                        "dataset": dataset,
                        "model": name,
                        "metric": metric,
                        "runs": len(values),
                        "mean": average,
                        "seed_sd": deviation,
                    }
                )
                if values:
                    _ = axis.errorbar(index, mean(values), yerr=deviation, fmt="o", capsize=4)
                else:
                    _ = axis.text(index, 0.5, "N/A", ha="center")
            _ = axis.set(title=title, xlim=(-0.5, len(names) - 0.5), ylim=(-0.03, 1.03))
            _ = axis.set_xticks(range(len(names)), names, rotation=25, ha="right")
            axis.grid(axis="y", alpha=0.2)
        _ = figure.suptitle(f"{dataset} — mean ± 1 SD across seeds; one run has no error bar")
        filename = "comparison-" + re.sub(r"[^a-zA-Z0-9_-]", "_", dataset)
        save(figure, output / filename, report)
    pl.DataFrame(rows).write_csv(output / "summary.csv")


def learning(run: Run, output: Path, report: PdfPages) -> None:
    """Plot supervised and optional pretraining loss without mixing their scales"""
    histories = [
        run.path / name
        for name in ("history.json", "pretrain_history.json")
        if (run.path / name).exists()
    ]
    if not histories:
        return
    figure = Figure(figsize=(6 * len(histories), 4), layout="constrained")
    for index, path in enumerate(histories, 1):
        entries = json.loads(path.read_text())
        axis = figure.add_subplot(1, len(histories), index)
        for key, label in (("train_loss", "Train"), ("validation_loss", "Validation")):
            _ = axis.plot(
                [row["epoch"] for row in entries], [row[key] for row in entries], label=label
            )
        _ = axis.set(
            xlabel="Epoch",
            ylabel="Loss",
            title=("Pretraining" if path.name.startswith("pretrain") else "Classification"),
        )
        _ = axis.legend()
        axis.grid(alpha=0.2)
    _ = figure.suptitle(f"{run.name} · seed {run.seed} · {run.dataset}")
    save(figure, output / "learning", report)


def predictions(run: Run, output: Path, report: PdfPages, threshold: float) -> None:
    """Plot score curves and errors at a user-chosen, fixed threshold"""
    path = run.path / "predictions.parquet"
    if not path.exists():
        print(f"{run.path}: no predictions.parquet; skipping score/error plots")
        return
    frame = pl.read_parquet(path)
    if (
        frame.is_empty()
        or frame["label"].null_count()
        or frame["probability"].null_count()
        or not frame["label"].is_in([0, 1]).all()
        or not frame["probability"].is_finite().all()
        or not frame["probability"].is_between(0, 1).all()
    ):
        raise ValueError(f"{path}: invalid labels or probabilities")
    positives = int(frame["label"].sum())
    negatives = len(frame) - positives
    if positives != run.metrics["positives"] or negatives != run.metrics["negatives"]:
        raise ValueError(f"{path}: class counts differ from metrics.json")
    curve = (
        frame.group_by("probability")
        .agg(pl.col("label").sum().alias("tp"), (1 - pl.col("label")).sum().alias("fp"))
        .sort("probability", descending=True)
        .with_columns(pl.col("tp", "fp").cum_sum())
    )
    curve = curve.with_row_index().filter(
        (pl.col("index") % max(len(curve) // 2000, 1) == 0) | (pl.col("index") == len(curve) - 1)
    )
    figure = Figure(figsize=(13, 4), layout="constrained")
    axes: list[Axes] = [figure.add_subplot(1, 3, i) for i in range(1, 4)]
    if positives and negatives:
        recall = (curve["tp"] / positives).to_list()
        precision = (curve["tp"] / (curve["tp"] + curve["fp"])).to_list()
        _ = axes[0].step([0, *recall], [1, *precision], where="pre")
        _ = axes[0].axhline(
            positives / len(frame), color="grey", linestyle="--", label="Prevalence"
        )
        _ = axes[0].legend()
        _ = axes[1].plot([0, *(curve["fp"] / negatives).to_list()], [0, *recall])
        _ = axes[1].plot([0, 1], [0, 1], "--", color="grey")
    else:
        for axis in axes[:2]:
            _ = axis.text(0.5, 0.5, "Both classes required", ha="center")
    bins = frame.with_columns((pl.col("probability") * 10).floor().clip(0, 9).alias("bin"))
    calibration = bins.group_by("bin").agg(pl.col("probability", "label").mean()).sort("bin")
    _ = axes[2].plot(calibration["probability"].to_list(), calibration["label"].to_list(), "o-")
    _ = axes[2].plot([0, 1], [0, 1], "--", color="grey")
    for axis, labels in zip(
        axes,
        (
            ("Recall", "Precision"),
            ("False-positive rate", "Recall"),
            ("Mean predicted probability", "Observed attack rate"),
        ),
        strict=True,
    ):
        _ = axis.set(xlabel=labels[0], ylabel=labels[1], xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=0.2)
    _ = figure.suptitle(
        f"{run.name} · {run.dataset} · curves thinned for display; metrics use all scores"
    )
    save(figure, output / "curves", report)

    frame = frame.with_columns(
        (pl.col("probability") >= threshold).cast(pl.Int8).alias("predicted")
    )
    counts = frame.group_by("label", "predicted").len()
    matrix = [[0, 0], [0, 0]]
    for label, predicted, count in counts.iter_rows():
        matrix[int(label)][int(predicted)] = count
    figure = Figure(figsize=(12, 4), layout="constrained")
    axis = figure.add_subplot(1, 2, 1)
    _ = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            _ = axis.text(
                column,
                row,
                f"{matrix[row][column]:,}",
                ha="center",
                va="center",
                color="black",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
    _ = axis.set_xticks([0, 1], ["Benign", "Attack"])
    _ = axis.set_yticks([0, 1], ["Benign", "Attack"])
    _ = axis.set(xlabel="Predicted", ylabel="Actual", title=f"Threshold {threshold:g}")
    errors = frame.filter(pl.col("label") != pl.col("predicted")).with_columns(
        pl.when(pl.col("label") == 0)
        .then(pl.lit("False positive"))
        .otherwise(pl.lit("False negative"))
        .alias("error"),
        (pl.col("probability") - pl.col("label")).abs().alias("confidence_wrong"),
    )
    errors.sort("confidence_wrong", descending=True).head(100).write_csv(
        output / "worst-100-errors.csv"
    )
    axis = figure.add_subplot(1, 2, 2)
    for label, title in ((0, "Benign"), (1, "Attack")):
        _ = axis.hist(
            frame.filter(pl.col("label") == label)["probability"].to_list(),
            bins=20,
            range=(0, 1),
            histtype="step",
            label=title,
        )
    _ = axis.axvline(threshold, color="grey", linestyle="--")
    _ = axis.set(xlabel="Attack probability", ylabel="Flows", title="Score distribution")
    _ = axis.legend()
    _ = figure.suptitle(f"{run.name} · seed {run.seed} · errors on {run.dataset}")
    save(figure, output / "errors", report)


def main() -> None:
    """Build a report from completed runs, without training or changing thresholds"""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("root", nargs="?", type=Path, default=Path("results"))
    _ = parser.add_argument("--output", type=Path, default=Path("results/figures"))
    _ = parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between zero and one")
    runs = read_runs(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output / "report.pdf") as report:
        compare(runs, args.output, report)
        for run in runs:
            destination = args.output / run.path.relative_to(args.root)
            destination.mkdir(parents=True, exist_ok=True)
            learning(run, destination, report)
            predictions(run, destination, report, args.threshold)
    print(f"Saved report.pdf, PNG/SVG figures, summary.csv, and error examples to {args.output}")


if __name__ == "__main__":
    main()
