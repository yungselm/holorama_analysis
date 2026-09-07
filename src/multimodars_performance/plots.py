"""Publication figures comparing the multimodars 0.3.4 and 0.7.0 benchmarks.

Reads the CSVs written by `mm_perf.run_pipeline` and produces, into
`output/plots/`:

* `per_step.png`            - box plots, 0.3.4 vs 0.7.0, one panel per block
* `total.png`               - box plots of the whole-pipeline time
* `per_step_normalized.png` - the same per block, per input point
* `total_normalized.png`    - the same for the total
* `summary.csv`, `summary_normalized.csv` - the numbers behind the figures

Boxes rather than bars: with 14 cases spanning 43k-280k mesh vertices the
spread is the result, not noise around it, and every case is overplotted on its
box so nothing hides behind a summary.

Every axis is linear and starts at zero. Blocks differ by three orders of
magnitude, so they are drawn as small multiples - one panel per block, each with
its own axis in seconds - rather than squeezed onto one shared scale.

Each pair carries a paired Wilcoxon signed-rank p-value over the cases both
versions completed: paired because the same case is timed twice, rank-based
because per-case runtimes are strongly skewed by mesh size.

Normalization is per block - divided by that case's CCTA mesh vertex count,
except `intravascular_alignment` (`from_file_singlepair`), which is IVUS work
and divides by its contour-point count. Each version divides by its own counts,
so the result is that version's real cost per point.

Run with `python src/make_plots.py` (the package is not installed;
`src/` reaches sys.path via the entry script, as with `src/main.py`).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from multimodars_performance.mm_perf import BLOCKS, OUTPUT_DIR  # noqa: E402

OLD, NEW = "0.3.4", "0.7.0"
COLORS = {OLD: "#C1443C", NEW: "#2C6FB5"}
FAIL = "#B3261E"

# `intravascular_alignment` is the only block that does not touch the CCTA mesh.
POINT_COLUMN = {block: "n_vertices" for block in BLOCKS}
POINT_COLUMN["intravascular_alignment"] = "n_points"
TOTAL_POINT_COLUMN = "n_vertices"

# A two-sided Wilcoxon over n pairs has a hard p floor (0.25 at n=3, 0.125 at
# n=4); below this, a non-significant result says nothing and is flagged.
MIN_PAIRS_FOR_TEST = 5

# Plain black-on-white, full axis box, outward ticks, no decoration.
PUBLICATION_RC = {
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "black",
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "axes.grid": False,
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.framealpha": 1.0,
    "legend.fancybox": False,
    "font.family": "sans-serif",
    "font.size": 9,
}


def load_results(output_dir: Path = OUTPUT_DIR) -> dict[str, pd.DataFrame]:
    """Load one CSV per version, indexed by case."""
    frames = {}
    for version in (OLD, NEW):
        path = output_dir / f"fusion_{version.replace('.', '_')}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No benchmark CSV for {version} at {path}")
        frames[version] = pd.read_csv(path).set_index("case")
    return frames


def _completed(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Drop cases that errored out.

    `total_seconds` is the sum of whatever blocks ran before the failure, so a
    failed case contributes a partial total that would understate that version.
    Individual blocks are unaffected - a block that ran is a valid measurement -
    so this filter applies only to the end-to-end figure.
    """
    kept = {}
    for version, frame in frames.items():
        if "error" not in frame:
            kept[version] = frame
            continue
        error = frame["error"]
        # An all-empty error column reads back as NaN, not as "".
        blank = error.isna() | (error.astype("string").str.strip() == "")
        kept[version] = frame[blank.fillna(True)]
    return kept


def _values(frame: pd.DataFrame, column: str, denominator: str | None) -> pd.Series:
    """One version's per-case values for *column*, optionally per point."""
    if column not in frame:
        return pd.Series(dtype=float)
    value = pd.to_numeric(frame[column], errors="coerce")
    if denominator is not None:
        count = pd.to_numeric(frame[denominator], errors="coerce")
        value = value / count.where(count > 0)
    return value.dropna()


def _paired(frames, column: str, denominator: str | None = None) -> pd.DataFrame:
    """Both versions' per-case values, aligned on the cases both completed.

    A column only one version produced (0.3.4 has no `discretization`) comes
    back with that version's column empty rather than as an empty frame, so the
    surviving box is still drawn.
    """
    values = {version: _values(frame, column, denominator) for version, frame in frames.items()}
    common = values[OLD].index.intersection(values[NEW].index)
    if len(common):
        return pd.DataFrame({version: series.loc[common] for version, series in values.items()})
    return pd.DataFrame({OLD: values[OLD], NEW: values[NEW]})


def _sum_blocks(frames, blocks: list[str]) -> pd.DataFrame:
    """Per-case sum over *blocks*, for cases where every one of them has a value."""
    totals = {}
    for version, frame in frames.items():
        columns = pd.concat(
            [pd.to_numeric(frame[block], errors="coerce") for block in blocks], axis=1
        )
        totals[version] = columns.dropna().sum(axis=1)
    common = totals[OLD].index.intersection(totals[NEW].index)
    return pd.DataFrame({version: series.loc[common] for version, series in totals.items()})


def _pvalue(paired: pd.DataFrame) -> tuple[float | None, int]:
    """Paired Wilcoxon signed-rank p-value over the complete pairs."""
    both = paired.dropna()
    if len(both) < 2 or np.allclose(both[OLD], both[NEW]):
        return None, len(both)
    try:
        return float(wilcoxon(both[OLD], both[NEW]).pvalue), len(both)
    except ValueError:
        return None, len(both)


def _format_p(p: float | None, n: int) -> str:
    """P label for a bracket. The n itself is printed under each box."""
    if p is None:
        return "not testable"
    text = "P < 0.001" if p < 0.001 else f"P = {p:.3f}"
    if n < MIN_PAIRS_FOR_TEST:
        # The test cannot reach significance at this n; say so rather than let
        # a non-significant result read as evidence of no difference.
        return f"{text} (underpowered)"
    return text


def _has(paired: pd.DataFrame, version: str) -> bool:
    return version in paired and not paired[version].dropna().empty


def _draw_boxes(axis, groups, scale: float = 1.0, n_cases: int | None = None,
                width: float = 0.55, gap: float = 0.34) -> None:
    """Paired box plots on a linear axis from zero, every case overplotted.

    One pair per entry in *groups*, positioned at 0, 1, 2, ... Each pair gets a
    significance bracket; each box gets an x tick label naming the version and
    its case count, reddened when cases are missing because that version
    crashed on them.
    """
    rng = np.random.default_rng(0)
    offsets = {OLD: -gap / 2, NEW: gap / 2}
    ceiling = 0.0
    ticks, labels, colors = [], [], []

    for index, (_, paired) in enumerate(groups):
        count = max((len(paired[v].dropna()) for v in (OLD, NEW) if v in paired), default=0)
        missing = 0 if n_cases is None else n_cases - count

        for version in (OLD, NEW):
            position = index + offsets[version]
            ticks.append(position)
            if not _has(paired, version):
                labels.append(f"{version}\nn/a")
                colors.append(FAIL)
                continue

            series = paired[version].dropna() * scale
            box = axis.boxplot(
                [series.to_numpy()],
                positions=[position],
                widths=width * gap,
                patch_artist=True,
                showfliers=False,  # every case is drawn individually below
                medianprops={"color": "black", "linewidth": 1.3},
                boxprops={"facecolor": COLORS[version], "edgecolor": "black", "linewidth": 0.8},
                whiskerprops={"color": "black", "linewidth": 0.8},
                capprops={"color": "black", "linewidth": 0.8},
            )
            box["boxes"][0].set_alpha(0.55)
            axis.plot(
                position + rng.uniform(-0.055, 0.055, len(series)), series.to_numpy(),
                "o", color="black", markersize=2.8, linestyle="none", zorder=5,
            )
            # Headroom above the data itself: a panel with no significance
            # bracket (0.7.0 only) would otherwise clip its topmost case.
            ceiling = max(ceiling, float(series.max()) * 1.15)
            labels.append(f"{version}\nn={len(series)}")
            colors.append(FAIL if missing else "black")

        if not (_has(paired, OLD) and _has(paired, NEW)):
            continue

        # Significance bracket above the taller of the two boxes.
        top = float(np.nanmax(paired[[OLD, NEW]].to_numpy(dtype=float))) * scale
        bar = top * 1.10
        for x in (index + offsets[OLD], index + offsets[NEW]):
            axis.plot([x, x], [top * 1.04, bar], color="black", linewidth=0.7)
        axis.plot([index + offsets[OLD], index + offsets[NEW]], [bar, bar],
                  color="black", linewidth=0.7)
        p, n = _pvalue(paired)
        axis.text(index, bar * 1.03, _format_p(p, n), ha="center", va="bottom", fontsize=7.5)
        ceiling = max(ceiling, bar * 1.30)

    axis.set_ylim(0, ceiling if ceiling > 0 else 1.0)
    axis.set_xlim(-0.5, len(groups) - 0.5)
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels, fontsize=7.5)
    for label, color in zip(axis.get_xticklabels(), colors):
        label.set_color(color)
    axis.yaxis.grid(True, which="major", color="0.88", linewidth=0.5, linestyle=":")
    axis.set_axisbelow(True)


def _wrap(text: str, width: int) -> str:
    """Wrap each explicit line to *width*, so a long caption cannot widen the canvas."""
    newline = chr(10)
    return newline.join(textwrap.fill(line, width) for line in text.split(newline))


def _legend_handles():
    return [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[version], alpha=0.55,
                      edgecolor="black", linewidth=0.8, label=f"multimodars {version}")
        for version in (OLD, NEW)
    ]


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def _grid_figure(groups, ylabel, path, footnote, n_cases, scale=1.0, ncols=5):
    """Small multiples: one linear panel per block, each with its own scale."""
    nrows = int(np.ceil(len(groups) / ncols))
    with plt.rc_context(PUBLICATION_RC):
        figure, axes = plt.subplots(nrows, ncols, figsize=(2.55 * ncols, 3.1 * nrows))
        axes = np.atleast_1d(axes).ravel()

        for axis, (name, paired) in zip(axes, groups):
            _draw_boxes(axis, [(name, paired)], scale, n_cases, gap=0.62)
            axis.set_title(name.replace("_", " "), pad=16)
            if not _has(paired, OLD):
                axis.text(0, axis.get_ylim()[1] * 0.5, "not available\nin 0.3.4",
                          ha="center", va="center", fontsize=7.5, style="italic", color=FAIL)
            missing = n_cases - max(
                (len(paired[v].dropna()) for v in (OLD, NEW) if v in paired), default=0
            )
            if missing and _has(paired, OLD):
                axis.text(0.5, 1.005, f"{missing}/{n_cases} failed in 0.3.4",
                          transform=axis.transAxes, ha="center", va="bottom",
                          fontsize=7.5, color=FAIL)
        for axis in axes[len(groups):]:
            axis.set_visible(False)

        figure.supylabel(ylabel, fontsize=10)
        figure.legend(handles=_legend_handles(), loc="lower left",
                      bbox_to_anchor=(0.005, 1.0), ncol=2, fontsize=8.5, borderaxespad=0)
        figure.tight_layout()
        figure.text(0.005, -0.02, _wrap(footnote, 165), fontsize=7.5,
                    va="top", ha="left", color="0.25")
        _save(figure, path)


def _single_figure(groups, ylabel, path, footnote, n_cases, scale=1.0, size=(5.6, 5.0)):
    with plt.rc_context(PUBLICATION_RC):
        figure, axis = plt.subplots(figsize=size)
        _draw_boxes(axis, groups, scale, n_cases, gap=0.44)
        axis.set_ylabel(ylabel, fontsize=10)

        # Group names ride on minor ticks, one level below the per-box labels,
        # so the two rows cannot overlap each other or the footnote.
        axis.set_xticks(range(len(groups)), minor=True)
        axis.set_xticklabels([label for label, _ in groups], minor=True, fontsize=8.5)
        axis.tick_params(axis="x", which="minor", length=0, pad=26)

        axis.legend(handles=_legend_handles(), loc="lower left", bbox_to_anchor=(0, 1.005),
                    ncol=2, fontsize=8.5, borderaxespad=0)
        figure.text(0, -0.20, _wrap(footnote, 74), transform=axis.transAxes,
                    fontsize=7.5, va="top", ha="left", color="0.25")
        _save(figure, path)


def _failure_report(frames: dict[str, pd.DataFrame]) -> tuple[str, int]:
    """One-line summary of which cases failed, and in what.

    The failures are as much a result as the timings, so they go on the figure
    rather than only into the CSV.
    """
    n_cases = max(len(frame) for frame in frames.values())
    parts = []
    for version, frame in frames.items():
        if "error" not in frame:
            continue
        errors = frame["error"].dropna().astype(str).str.strip()
        errors = errors[errors != ""]
        if errors.empty:
            continue
        kinds = sorted({error.split(":")[0] for error in errors})
        parts.append(f"{version} failed in {len(errors)}/{n_cases} cases ({', '.join(kinds)})")
    return "; ".join(parts) if parts else "no case failed in either version", n_cases


def _write_summary(groups, path: Path, n_cases: int, scale: float = 1.0, unit: str = "s") -> None:
    """Per-group medians, IQRs, ratio and P value, as a table to cite from."""
    rows = []
    for label, paired in groups:
        row = {"group": " ".join(label.split()), "unit": unit}
        for version in (OLD, NEW):
            series = (paired[version].dropna() * scale) if _has(paired, version) else pd.Series(dtype=float)
            row[f"{version}_n"] = len(series)
            row[f"{version}_failed"] = n_cases - len(series)
            row[f"{version}_median"] = series.median() if len(series) else np.nan
            row[f"{version}_q1"] = series.quantile(0.25) if len(series) else np.nan
            row[f"{version}_q3"] = series.quantile(0.75) if len(series) else np.nan
        if _has(paired, OLD) and _has(paired, NEW):
            row["ratio_old_over_new"] = row[f"{OLD}_median"] / row[f"{NEW}_median"]
            p, n = _pvalue(paired)
            row["p_wilcoxon"] = p
            row["n_pairs"] = n
            row["underpowered"] = n < MIN_PAIRS_FOR_TEST
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6g")
    print(f"wrote {path}")


def make_plots(output_dir: Path = OUTPUT_DIR, plot_dir: Path | None = None) -> None:
    plot_dir = plot_dir or output_dir / "plots"
    frames = load_results(output_dir)
    complete = _completed(frames)

    raw_blocks = [(block, _paired(frames, block)) for block in BLOCKS]
    norm_blocks = [(block, _paired(frames, block, POINT_COLUMN[block])) for block in BLOCKS]

    # Blocks every case got through in both versions. Summing only these gives a
    # total backed by all 14 cases, alongside the end-to-end total that only the
    # cases surviving 0.3.4's stitching can support.
    full_n = max(len(paired.dropna()) for _, paired in raw_blocks)
    shared = [block for block, paired in raw_blocks if len(paired.dropna()) == full_n]

    total_groups = [
        (f"End to end\n(all {len(BLOCKS)} blocks)", _paired(complete, "total_seconds")),
        (f"Sum of the {len(shared)} blocks\nevery case completed", _sum_blocks(frames, shared)),
    ]

    failures, n_cases = _failure_report(frames)
    method = (
        f"{n_cases} anomalous-coronary cases. Boxes: median, IQR, 1.5x IQR whiskers; each dot is "
        "one case. P: two-sided paired Wilcoxon signed-rank test over the cases both versions "
        f"completed.\nErrors: {failures}."
    )
    normalized_note = ("\nNormalized per CCTA mesh vertex, except intravascular alignment "
                       "(per IVUS contour point).")
    total_note = ("\nLeft: cases that ran end to end in both versions. Right: per-case sum over "
                  "only the blocks no case failed, so all cases contribute.")

    _grid_figure(raw_blocks, "Time (s)", plot_dir / "per_step.png", method, n_cases)
    _grid_figure(norm_blocks, "Time per point (µs)", plot_dir / "per_step_normalized.png",
                 method + normalized_note, n_cases, scale=1e6)

    _single_figure(total_groups, "Time (s)", plot_dir / "total.png",
                   method + total_note, n_cases)

    vertices = {
        version: pd.to_numeric(frame[TOTAL_POINT_COLUMN], errors="coerce")
        for version, frame in frames.items()
    }
    total_norm = [
        (label, pd.DataFrame({v: paired[v] / vertices[v] for v in (OLD, NEW) if v in paired}))
        for label, paired in total_groups
    ]
    _single_figure(total_norm, "Time per mesh vertex (µs)", plot_dir / "total_normalized.png",
                   method + total_note, n_cases, scale=1e6)

    _write_summary(raw_blocks + total_groups, plot_dir / "summary.csv", n_cases)
    _write_summary(norm_blocks + total_norm, plot_dir / "summary_normalized.csv",
                   n_cases, scale=1e6, unit="us_per_point")


if __name__ == "__main__":
    make_plots()
