"""
src/viz.py -- publication-grade figures for the POSRRI study.

Every figure is written twice: a 300 dpi PNG for review and manuscript
submission, and a vector PDF for typesetting. Design rules applied throughout:

* **Colourblind-safe.** All hues come from the Okabe-Ito palette. The
  categorical sets used here were checked for CVD separation (worst adjacent
  pair dE = 11.0 under deuteranopia for the four-port set, 11.0 for the
  three-pillar set) and lightness/chroma banding against a light surface.
* **Greyscale-legible.** Colour is never the only channel: series also differ
  in line style and marker, cells and bars carry printed values, and every
  multi-series figure has a legend.
* **Colour by job.** Categorical hues are assigned in a fixed order and never
  cycled (ports, pillars); the indicator heatmap uses a single-hue sequential
  ramp, light to dark; the implementation-gap figure uses a two-hue diverging
  scheme with a neutral zero.
* **Recessive furniture.** Light grids behind the data, no chart junk, no
  dual axes, and ``tight_layout`` on every figure.

Run ``python src/viz.py`` to regenerate all six figures from synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")                       # headless: CI, Colab, servers

import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402
import pandas as pd                         # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D         # noqa: E402
from matplotlib.patches import Patch        # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from src import structure as st  # noqa: E402

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

OI = config.OKABE_ITO

#: Four ports, fixed order, validated for CVD separation. Each also gets a
#: distinct line style and marker so the figure survives greyscale printing.
PORT_STYLE: Dict[str, dict] = {
    "CPA":                 dict(color=OI["blue"],         ls="-",  marker="o", lw=2.4, z=5),
    "Singapore":           dict(color=OI["vermillion"],   ls="--", marker="s", lw=1.6, z=4),
    "LosAngelesLongBeach": dict(color=OI["bluish_green"], ls="-.", marker="^", lw=1.6, z=3),
    "Australia":           dict(color=OI["orange"],       ls=":",  marker="D", lw=1.8, z=2),
}

#: Single-hue sequential ramp for the 0-3 rubric, light to dark, monotone in
#: lightness so it reads correctly in greyscale.
SCORE_RAMP = ["#EEF4F9", "#A8C9E2", "#3D8DC4", "#08508A"]

#: Colourbar tick labels, read from the locked rubric in config.py so the
#: figure can never disagree with the scoring definition used to produce it.
SCORE_LABELS = {k: config.rubric_label(k) for k in sorted(config.SCORE_RUBRIC)}

#: Diverging scheme for the documentation-vs-practice gap: two hues, neutral
#: midpoint, no rainbow.
GAP_POSITIVE = OI["vermillion"]     # paperwork ahead of practice
GAP_NEGATIVE = OI["blue"]           # practice ahead of paperwork
GAP_NEUTRAL = "#BFBFBF"

INK = "#1A1A1A"
INK_MUTED = "#5A5A5A"
GRID = "#D9D9D9"


# ---------------------------------------------------------------------------
# Style and I/O
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Set the shared matplotlib rcParams for every POSRRI figure."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": config.FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": [config.FIGURE_FONT_FAMILY, "Arial", "Helvetica"],
        "font.size": config.FIGURE_BASE_FONTSIZE,
        "axes.titlesize": config.FIGURE_BASE_FONTSIZE + 2,
        "axes.titleweight": "bold",
        "axes.labelsize": config.FIGURE_BASE_FONTSIZE + 1,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "xtick.labelsize": config.FIGURE_BASE_FONTSIZE - 0.5,
        "ytick.labelsize": config.FIGURE_BASE_FONTSIZE - 0.5,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": config.FIGURE_BASE_FONTSIZE - 0.5,
        "pdf.fonttype": 42,                 # embed TrueType, not Type 3
        "ps.fonttype": 42,
        "text.color": INK,
    })


def save_figure(fig: plt.Figure, stem: str,
                figure_dir: "Path | str" = config.FIGURE_DIR,
                close: bool = True) -> List[Path]:
    """Write ``fig`` as 300 dpi PNG and vector PDF; return the paths written."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    paths = []
    for ext in config.FIGURE_FORMATS:
        path = figure_dir / f"{stem}.{ext}"
        fig.savefig(path, format=ext, dpi=config.FIGURE_DPI)
        paths.append(path)
    if close:
        plt.close(fig)
    return paths


def _quietest_angle(bench: pd.DataFrame, domains: Sequence[str],
                    ports: Sequence[str]) -> float:
    """Angle (degrees) of the spoke gap with the most free radial space.

    For each gap between adjacent spokes, take the largest series value on
    either side: the gap whose maximum is lowest has the most empty radius
    between the plotted lines and the outer ring, so radial tick labels placed
    there overlap the least data.
    """
    values = bench.loc[list(domains), list(ports)].to_numpy(dtype=float)
    n = len(domains)
    step = 360.0 / n
    busiest = [max(np.nanmax(values[k]), np.nanmax(values[(k + 1) % n]))
               for k in range(n)]
    gap = int(np.argmin(busiest))
    return (gap + 0.5) * step


def _caption_below(fig, legend, text: str, gap_px: float = 12.0, **kwargs):
    """Put a caption directly under ``legend``, measured rather than guessed.

    Figure-fraction offsets guessed by hand break as soon as the figure height
    or the number of legend rows changes, which is how a multi-line caption
    ends up printed on top of a legend entry. Measuring the legend's rendered
    extent keeps the caption clear of it at any size.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = legend.get_window_extent(renderer=renderer)
    _, y = fig.transFigure.inverted().transform((bb.x0, bb.y0 - gap_px))
    return fig.text(0.5, y, text, ha="center", va="top", **kwargs)


def _repel_labels(ax, xs, ys, labels, fontsize=8, pad_px=10.0, **text_kw):
    """Place point labels so they collide with neither markers nor each other.

    Greedy placement using real text extents: for each point the candidate
    offsets are scored by how much their bounding box overlaps the markers and
    the labels already placed, and the cheapest is kept. This is what stops
    coincident domains (identical scores under both weightings) from printing
    their codes on top of one another.
    """
    fig = ax.figure
    fig.canvas.draw()                       # a renderer is needed for extents
    renderer = fig.canvas.get_renderer()

    candidates = [(8, 5), (8, -13), (-9, 5), (-9, -13),
                  (0, 12), (0, -19), (16, -4), (-17, -4)]
    points_px = [ax.transData.transform((x, y)) for x, y in zip(xs, ys)]
    marker_boxes = [(px - pad_px, py - pad_px, px + pad_px, py + pad_px)
                    for px, py in points_px]

    def overlap(a, b) -> float:
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])
        return dx * dy if dx > 0 and dy > 0 else 0.0

    placed_boxes: List[tuple] = []
    for (x, y), label, (px, py) in zip(zip(xs, ys), labels, points_px):
        ann = ax.annotate(label, (x, y), textcoords="offset points",
                          xytext=candidates[0], fontsize=fontsize, **text_kw)
        bb = ann.get_window_extent(renderer=renderer)
        w, h = bb.width, bb.height
        best, best_cost = candidates[0], None
        for dx, dy in candidates:
            # Reconstruct the box this offset would produce, from the anchor.
            x0 = px + dx if dx >= 0 else px + dx - w
            y0 = py + dy if dy >= 0 else py + dy - h + 6.0
            box = (x0, y0, x0 + w, y0 + h)
            cost = sum(overlap(box, m) for m in marker_boxes)
            cost += 2.0 * sum(overlap(box, o) for o in placed_boxes)
            if best_cost is None or cost < best_cost:
                best, best_cost, best_box = (dx, dy), cost, box
            if cost == 0.0:                 # nothing better than no overlap
                break
        ann.set_position(best)
        placed_boxes.append(best_box)
    return placed_boxes


def _domain_axis_labels() -> List[str]:
    """``D1 Legal basis`` style labels in canonical order."""
    return [f"{d}  {st.DOMAIN_SHORT[d]}" for d in st.DOMAIN_ORDER]


# ---------------------------------------------------------------------------
# (a) Radar: CPA against the benchmark ports
# ---------------------------------------------------------------------------

def fig_radar_benchmark(bench: pd.DataFrame,
                        figure_dir: "Path | str" = config.FIGURE_DIR,
                        stem: str = "fig1_radar_benchmark") -> List[Path]:
    """Radar of the 10 domain scores (0-100), CPA overlaid on comparator ports.

    ``bench`` is a domains x ports frame on the 0-100 scale, as returned by
    :func:`src.scoring.benchmark_matrix`.
    """
    apply_style()
    domains = [d for d in st.DOMAIN_ORDER if d in bench.index]
    n = len(domains)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(7.2, 6.6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ports = [p for p in st.BENCHMARK_PORTS if p in bench.columns]
    for port in ports:
        style = PORT_STYLE.get(port, dict(color=OI["grey"], ls="-", marker="o", lw=1.5, z=1))
        vals = bench.loc[domains, port].to_numpy(dtype=float)
        vals = np.concatenate([vals, vals[:1]])
        ax.plot(closed, vals, color=style["color"], linestyle=style["ls"],
                linewidth=style["lw"], marker=style["marker"], markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.6,
                label=st.PORT_LABELS.get(port, port), zorder=style["z"])
        if port == "CPA":                       # fill only the focal case
            ax.fill(closed, vals, color=style["color"], alpha=0.13, zorder=1)

    ax.set_xticks(angles)
    ax.set_xticklabels([f"{d}\n{st.DOMAIN_SHORT[d]}" for d in domains], fontsize=8)
    ax.tick_params(axis="x", pad=12)

    # The radial axis must start at 0 in the centre: a truncated radial scale
    # exaggerates differences between ports. Pin both the limits and the origin
    # explicitly rather than relying on the default.
    ax.set_rorigin(0)
    ax.set_rlim(0, 100)
    assert ax.get_ylim() == (0, 100), "radial axis is not anchored at 0"

    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7, color=INK_MUTED)
    # Park the radial tick labels on the emptiest spoke gap. The D1/D2 sector
    # (used by default) is where CPA scores highest and all four series bunch
    # together, so the numbers collided with the data and the gridlines there.
    ax.set_rlabel_position(_quietest_angle(bench, domains, ports))
    # Radial labels sit on top of the series; a surface-coloured backing keeps
    # them readable without hiding the data.
    for label in ax.get_yticklabels():
        label.set_bbox(dict(boxstyle="round,pad=0.12", facecolor="white",
                            edgecolor="none", alpha=0.85))
    ax.grid(color=GRID, linewidth=0.6)
    ax.spines["polar"].set_color(GRID)

    ax.set_title("Domain readiness, Chattogram Port against benchmark ports",
                 pad=26)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
              handlelength=2.6, columnspacing=2.4)
    fig.subplots_adjust(bottom=0.16)
    fig.text(0.5, 0.008, "Domain score, 0-100 (weighted mean of the 0-3 indicator rubric)",
             ha="center", fontsize=8, color=INK_MUTED)
    return save_figure(fig, stem, figure_dir)


# ---------------------------------------------------------------------------
# (b) Domain scores with AHP weights
# ---------------------------------------------------------------------------

def fig_domain_scores(domains: pd.DataFrame,
                      figure_dir: "Path | str" = config.FIGURE_DIR,
                      stem: str = "fig2_domain_scores_weights") -> List[Path]:
    """Horizontal bars of domain scores, grouped and coloured by pillar.

    Each bar is annotated with the domain's AHP global weight, so the reader
    can see at a glance whether a weak domain is also a heavily weighted one.
    Bars are grouped by pillar (a categorical identity encoding, fixed order)
    and sorted by score within each pillar.
    """
    apply_style()
    df = domains.copy()
    if "pillar_code" not in df.columns:
        df["pillar_code"] = df["domain_code"].map(st.DOMAIN_TO_PILLAR)
    weight_col = "domain_weight_global" if "domain_weight_global" in df.columns else "weight"

    # Group by pillar (A, B, C), best-scoring domain first within each group.
    order = []
    for pil in st.PILLARS:
        block = df[df["pillar_code"] == pil].sort_values("score_0_100", ascending=False)
        order.append(block)
    df = pd.concat(order, ignore_index=True)

    # Plot top-down: reverse so pillar A sits at the top of the axis.
    y = np.arange(len(df))[::-1]

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)

    colours = [config.PILLAR_COLOURS[p] for p in df["pillar_code"]]
    bars = ax.barh(y, df["score_0_100"], height=0.68, color=colours,
                   edgecolor="white", linewidth=0.8)

    # Values and weights are set in fixed columns to the right of the plotting
    # area rather than floating at each bar end, so they read as a small table
    # and never collide with a long bar.
    x_score, x_weight = 108.0, 128.0
    for rect, (_, row) in zip(bars, df.iterrows()):
        yc = rect.get_y() + rect.get_height() / 2
        ax.text(x_score, yc, f"{row['score_0_100']:.1f}", va="center", ha="right",
                fontsize=8.5, color=INK, fontweight="bold")
        ax.text(x_weight, yc, f"{row[weight_col]:.3f}", va="center", ha="right",
                fontsize=8, color=INK_MUTED)
    top = y.max() + 0.75
    ax.text(x_score, top, "Score", va="center", ha="right", fontsize=7.5,
            color=INK_MUTED, style="italic")
    ax.text(x_weight, top, "AHP weight", va="center", ha="right", fontsize=7.5,
            color=INK_MUTED, style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}  {st.DOMAIN_SHORT[c]}" for c in df["domain_code"]])
    ax.set_xlim(0, x_weight + 2)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(y.min() - 0.7, top + 0.5)
    ax.spines["bottom"].set_bounds(0, 100)
    ax.set_xlabel("Domain readiness score (0-100)")
    ax.set_title("POSRRI domain scores with AHP global weights")

    # Faint separators between pillar groups.
    sizes = [int((df["pillar_code"] == p).sum()) for p in st.PILLARS]
    boundary = 0
    for size in sizes[:-1]:
        boundary += size
        ax.axhline(y[boundary - 1] - 0.5, color=GRID, linewidth=0.8, zorder=0)

    handles = [Patch(facecolor=config.PILLAR_COLOURS[p], edgecolor="white",
                     label=f"Pillar {p}: {st.PILLARS[p]}") for p in st.PILLARS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.11),
              ncol=1, title=None)
    return save_figure(fig, stem, figure_dir)


# ---------------------------------------------------------------------------
# (c) Indicator heatmap
# ---------------------------------------------------------------------------

def fig_indicator_heatmap(indicators: pd.DataFrame,
                          figure_dir: "Path | str" = config.FIGURE_DIR,
                          stem: str = "fig3_indicator_heatmap") -> List[Path]:
    """Heatmap of all 50 indicator final scores (0-3), one row per domain.

    A single-hue sequential ramp encodes magnitude; every cell also prints its
    score, so the figure is fully readable in greyscale and by readers with
    any form of colour vision deficiency.
    """
    apply_style()
    df = indicators.copy()
    df["position"] = df["indicator_code"].str.split(".").str[1].astype(int)
    grid = (df.pivot_table(index="domain_code", columns="position",
                           values="final_score", aggfunc="mean")
              .reindex(st.DOMAIN_ORDER)
              .reindex(columns=range(1, st.INDICATORS_PER_DOMAIN + 1)))

    cmap = ListedColormap(SCORE_RAMP)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    im = ax.imshow(grid.to_numpy(dtype=float), cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(st.INDICATORS_PER_DOMAIN))
    ax.set_xticklabels([str(k) for k in range(1, st.INDICATORS_PER_DOMAIN + 1)])
    ax.set_xlabel("Indicator position within the domain")
    ax.set_yticks(range(len(st.DOMAIN_ORDER)))
    ax.set_yticklabels(_domain_axis_labels())
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 2px surface gap between cells.
    ax.set_xticks(np.arange(-0.5, st.INDICATORS_PER_DOMAIN, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(st.DOMAIN_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", length=0)

    # Each cell carries its real indicator code as well as its score, so no
    # cross-referencing against a separate key is needed to identify a cell.
    for i, dom in enumerate(st.DOMAIN_ORDER):
        for j in range(st.INDICATORS_PER_DOMAIN):
            val = grid.iloc[i, j]
            if pd.isna(val):
                continue
            ink = "white" if val >= 2 else INK
            ax.text(j, i - 0.19, f"{dom}.{j + 1}", ha="center", va="center",
                    fontsize=6.5, color=ink, alpha=0.85)
            ax.text(j, i + 0.13, f"{int(round(val))}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=ink)

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], pad=0.02, shrink=0.82)
    cbar.ax.set_yticklabels([SCORE_LABELS[k] for k in range(4)], fontsize=7.5)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)

    ax.set_title("Indicator-level readiness scores, grouped by domain")
    fig.text(0.02, 0.012,
             "Each cell is labelled with its indicator code and adjudicated 0-3 "
             "score. Columns are the five indicators within each domain, in order.",
             fontsize=7.5, color=INK_MUTED)
    return save_figure(fig, stem, figure_dir)


# ---------------------------------------------------------------------------
# (d) Work-as-imagined vs work-as-done
# ---------------------------------------------------------------------------

def fig_implementation_gap(indicators: pd.DataFrame,
                           figure_dir: "Path | str" = config.FIGURE_DIR,
                           stem: str = "fig4_implementation_gap",
                           show_zero_gap_rows: bool = False) -> List[Path]:
    """Diverging bars of ``doc_score - field_score``, by indicator.

    Positive bars are indicators where the documentation claims more than field
    verification found (work-as-imagined ahead of work-as-done); negative bars
    are undocumented practice. Two hues with a neutral zero line, per
    diverging-scheme practice.

    Indicators with **no gap** carry no information about the size or direction
    of the implementation shortfall, and drawing a zero-width bar for each of
    them fills most of the panel with blank rows. By default they are collapsed
    into a single counted annotation and the panel height shrinks to fit the
    indicators that do show a gap; the full 50-row data remains in
    ``outputs/tables/10_scores_indicators.csv``. Pass
    ``show_zero_gap_rows=True`` to plot all 50 rows instead.
    """
    apply_style()
    df = indicators.copy()
    if "implementation_gap" not in df.columns:
        df["implementation_gap"] = df["doc_score"] - df["field_score"]
    df["_order"] = df["indicator_code"].map({c: i for i, c in enumerate(st.INDICATORS)})
    df = df.sort_values("_order")

    n_total = len(df)
    zero_codes = df.loc[df["implementation_gap"] == 0, "indicator_code"].tolist()
    n_zero = len(zero_codes)

    plotted = df if show_zero_gap_rows else df[df["implementation_gap"] != 0].copy()
    if plotted.empty:                                   # degenerate but possible
        plotted = df.copy()

    y = np.arange(len(plotted))[::-1]
    gaps = plotted["implementation_gap"].to_numpy(dtype=float)
    colours = [GAP_POSITIVE if g > 0 else (GAP_NEGATIVE if g < 0 else GAP_NEUTRAL)
               for g in gaps]

    # Height tracks the number of rows actually drawn, so the row pitch stays
    # constant and the indicator labels stay legible whichever mode is used.
    height = float(np.clip(2.9 + 0.203 * len(plotted), 4.0, 11.0))
    fig, ax = plt.subplots(figsize=(6.8, height))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.barh(y, gaps, height=0.7, color=colours, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=INK_MUTED, linewidth=1.0)

    if show_zero_gap_rows:
        # A zero-width bar is invisible and would read as missing data.
        zero = gaps == 0
        ax.scatter(np.zeros(int(zero.sum())), y[zero], s=16, marker="o",
                   facecolor=GAP_NEUTRAL, edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(plotted["indicator_code"], fontsize=7)
    ax.set_ylim(y.min() - 0.9, y.max() + 1.9)
    lim = max(1.0, float(np.abs(gaps).max())) + 0.55
    ax.set_xlim(-lim, lim)
    ax.set_xticks(np.arange(-int(lim), int(lim) + 1))
    ax.set_xlabel("Documentary score minus field score (0-3 rubric points)")
    ax.set_title("Work-as-imagined versus work-as-done, by indicator")

    # Directional cues on the axis itself, so the reader never has to consult
    # the legend to know which side means what.
    top = y.max() + 1.35
    ax.annotate("", xy=(lim * 0.60, top), xytext=(0.12, top),
                arrowprops=dict(arrowstyle="->", color=GAP_POSITIVE, linewidth=1.3))
    ax.text(lim * 0.62, top, "Documentation ahead", color=GAP_POSITIVE,
            fontsize=8, fontweight="bold", ha="left", va="center")
    ax.annotate("", xy=(-lim * 0.60, top), xytext=(-0.12, top),
                arrowprops=dict(arrowstyle="->", color=GAP_NEGATIVE, linewidth=1.3))
    ax.text(-lim * 0.62, top, "Practice ahead", color=GAP_NEGATIVE,
            fontsize=8, fontweight="bold", ha="right", va="center")

    # Domain separators and labels, computed from the rows actually drawn.
    doms = plotted["indicator_code"].str.split(".").str[0].to_numpy()
    for k in range(1, len(doms)):
        if doms[k] != doms[k - 1]:
            ax.axhline(y[k - 1] - 0.5, color=GRID, linewidth=0.7, zorder=0)
    for dom in pd.unique(doms):
        rows = y[doms == dom]
        ax.text(-lim * 0.985, float(rows.mean()), dom, fontsize=7.5,
                fontweight="bold", color=INK_MUTED, ha="left", va="center")

    handles = [
        Patch(facecolor=GAP_POSITIVE, label="Documentation ahead of practice (gap > 0)"),
        Patch(facecolor=GAP_NEGATIVE, label="Practice ahead of documentation (gap < 0)"),
    ]
    if show_zero_gap_rows:
        handles.insert(1, Patch(facecolor=GAP_NEUTRAL, label="No gap (gap = 0)"))
    legend = ax.legend(handles=handles, loc="upper center",
                       bbox_to_anchor=(0.5, -0.055), ncol=1)

    if not show_zero_gap_rows and n_zero:
        _caption_below(fig, legend,
                       f"No gap (documentary score = field score): n = {n_zero} "
                       f"of {n_total} indicators, omitted above. "
                       f"{len(plotted)} indicators with a non-zero gap are shown.",
                       fontsize=7.5, color=INK_MUTED)
    return save_figure(fig, stem, figure_dir)


# ---------------------------------------------------------------------------
# (e) Delphi consensus
# ---------------------------------------------------------------------------

def fig_delphi_consensus(delphi_long: pd.DataFrame,
                         figure_dir: "Path | str" = config.FIGURE_DIR,
                         stem: str = "fig5_delphi_consensus",
                         round_no: "int | None" = None) -> List[Path]:
    """Median relevance with IQR whiskers for every indicator, consensus flagged.

    ``delphi_long`` is the indicator x round table from :func:`src.delphi.run_delphi`.
    Consensus is encoded by both colour and marker fill, so the flag survives
    greyscale printing.
    """
    apply_style()
    df = delphi_long.copy()
    if round_no is None:
        round_no = int(df["round"].max())
    df = df[df["round"] == round_no].copy()
    df["_order"] = df["indicator_code"].map({c: i for i, c in enumerate(st.INDICATORS)})
    df = df.sort_values("_order")

    y = np.arange(len(df))[::-1]
    med = df["median_relevance"].to_numpy(dtype=float)
    q1 = df["q1_relevance"].to_numpy(dtype=float)
    q3 = df["q3_relevance"].to_numpy(dtype=float)
    consensus = df["consensus"].to_numpy(dtype=bool)

    fig, ax = plt.subplots(figsize=(6.8, 10.2))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)

    ax.hlines(y, q1, q3, color=INK_MUTED, linewidth=1.4, alpha=0.85, zorder=2)
    for cap in (q1, q3):                         # whisker caps
        ax.vlines(cap, y - 0.28, y + 0.28, color=INK_MUTED, linewidth=1.0, zorder=2)

    ax.scatter(med[consensus], y[consensus], s=34, marker="o",
               facecolor=OI["blue"], edgecolor="white", linewidth=0.7,
               zorder=3, label="Consensus reached")
    ax.scatter(med[~consensus], y[~consensus], s=38, marker="D",
               facecolor="white", edgecolor=OI["vermillion"], linewidth=1.4,
               zorder=3, label="No consensus")

    ax.axvline(config.RELEVANCE_HIGH_MIN, color=OI["bluish_green"],
               linestyle="--", linewidth=1.1, zorder=1,
               label=f"Relevance threshold ({config.RELEVANCE_HIGH_MIN})")

    ax.set_yticks(y)
    ax.set_yticklabels(df["indicator_code"], fontsize=6.5)
    ax.set_ylim(y.min() - 0.8, y.max() + 0.8)
    ax.set_xlim(0.6, 9.4)
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("Expert relevance rating (1-9): median with interquartile range")
    ax.set_title(f"Delphi round {round_no}: relevance and consensus by indicator")

    for k in range(1, st.N_DOMAINS):
        ax.axhline(y[k * st.INDICATORS_PER_DOMAIN - 1] - 0.5,
                   color=GRID, linewidth=0.7, zorder=0)

    # Consensus and retention are two different criteria applied to the same
    # ratings, and they do not select the same indicators. The caption reports
    # both counts separately so the figure cannot be read as showing one number.
    n_total = len(df)
    n_consensus = int(consensus.sum())
    n_retained = int(df["retain"].sum()) if "retain" in df.columns else None

    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.045), ncol=1)

    caption = (
        f"CONSENSUS (plotted above): >=80% of experts rating 7-9, OR median >=7 "
        f"with IQR <=2  --  met by {n_consensus} of {n_total} indicators."
    )
    if n_retained is not None:
        caption += (
            f"\nRETENTION (a separate criterion, not plotted): "
            f"I-CVI >= {config.ICVI_RETAIN_THRESHOLD} "
            f"--  met by {n_retained} of {n_total} indicators."
        )
    _caption_below(fig, legend, caption, fontsize=7.5, color=INK_MUTED,
                   linespacing=1.6)
    return save_figure(fig, stem, figure_dir)


# ---------------------------------------------------------------------------
# (f) Sensitivity scatter
# ---------------------------------------------------------------------------

def fig_sensitivity_scatter(sens_domains: pd.DataFrame,
                            spearman_rho: float,
                            figure_dir: "Path | str" = config.FIGURE_DIR,
                            stem: str = "fig6_sensitivity_scatter",
                            p_value: "float | None" = None) -> List[Path]:
    """AHP-weighted against equally weighted domain scores, with Spearman rho.

    Points on the 1:1 line are unaffected by the weighting choice; the further
    a domain sits from the line, the more its score depends on the elicited
    weights. Points are coloured by pillar and directly labelled, so identity
    never rests on colour alone.
    """
    apply_style()
    df = sens_domains.copy()
    if "pillar_code" not in df.columns:
        df["pillar_code"] = df["domain_code"].map(st.DOMAIN_TO_PILLAR)

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.6)

    ax.plot([0, 100], [0, 100], color=INK_MUTED, linestyle="--", linewidth=1.0,
            zorder=1, label="1:1 (weighting has no effect)")

    for pil in st.PILLARS:
        block = df[df["pillar_code"] == pil]
        if block.empty:
            continue
        ax.scatter(block["score_equal"], block["score_ahp"], s=64,
                   color=config.PILLAR_COLOURS[pil], edgecolor="white",
                   linewidth=0.9, zorder=3,
                   label=f"Pillar {pil}: {st.PILLARS[pil]}")

    # Set the axis limits before labelling: the repel routine works in display
    # coordinates, so the data-to-pixel transform must already be final.
    lo, hi = 0, 100
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Domain score under equal weights (0-100)")
    ax.set_ylabel("Domain score under AHP weights (0-100)")
    ax.set_title("Sensitivity of domain scores to the weighting scheme")

    rho_text = f"Spearman $\\rho$ = {spearman_rho:.3f}"
    if p_value is not None:
        rho_text += f"\n$p$ = {p_value:.3g}"
    rho_text += f"\n$n$ = {len(df)} domains"
    ax.text(0.035, 0.965, rho_text, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor=GRID, linewidth=0.8))

    ax.legend(loc="lower right", handlelength=1.6)

    # Label last, once every other artist is in place and the layout is final:
    # tight_layout resizes the axes, which would otherwise invalidate the
    # pixel geometry the placement was computed against.
    fig.tight_layout()
    _repel_labels(ax,
                  df["score_equal"].to_numpy(dtype=float),
                  df["score_ahp"].to_numpy(dtype=float),
                  df["domain_code"].tolist(),
                  fontsize=8, color=INK, fontweight="bold")
    return save_figure(fig, stem, figure_dir)


# ---------------------------------------------------------------------------
# Convenience driver
# ---------------------------------------------------------------------------

#: Filename stems of the six manuscript figures, in order.
FIGURE_STEMS: Sequence[str] = (
    "fig1_radar_benchmark",
    "fig2_domain_scores_weights",
    "fig3_indicator_heatmap",
    "fig4_implementation_gap",
    "fig5_delphi_consensus",
    "fig6_sensitivity_scatter",
)


def make_all_figures(scoring_result: dict,
                     delphi_result: dict,
                     sensitivity_result: dict,
                     benchmark: pd.DataFrame,
                     ahp_domain_weights: "pd.DataFrame | None" = None,
                     figure_dir: "Path | str" = config.FIGURE_DIR) -> Dict[str, List[Path]]:
    """Render all six figures and return ``{stem: [paths]}``."""
    from src.scoring import benchmark_matrix

    domains = scoring_result["domains"].copy()
    if ahp_domain_weights is not None:
        domains = domains.merge(
            ahp_domain_weights[["domain_code", "domain_weight_global"]],
            on="domain_code", how="left")

    cpa = domains.set_index("domain_code")["score_0_100"]
    bench = benchmark_matrix(benchmark, cpa_domain_scores=cpa)

    out: Dict[str, List[Path]] = {}
    out[FIGURE_STEMS[0]] = fig_radar_benchmark(bench, figure_dir)
    out[FIGURE_STEMS[1]] = fig_domain_scores(domains, figure_dir)
    out[FIGURE_STEMS[2]] = fig_indicator_heatmap(scoring_result["indicators"], figure_dir)
    out[FIGURE_STEMS[3]] = fig_implementation_gap(scoring_result["indicators"], figure_dir)
    out[FIGURE_STEMS[4]] = fig_delphi_consensus(delphi_result["long"], figure_dir)
    out[FIGURE_STEMS[5]] = fig_sensitivity_scatter(
        sensitivity_result["domains"],
        sensitivity_result["spearman"]["rho"],
        figure_dir,
        p_value=sensitivity_result["spearman"]["p_value"],
    )
    return out


if __name__ == "__main__":
    from src.ahp import run_ahp
    from src.delphi import run_delphi
    from src.scoring import score_index
    from src.sensitivity import run_sensitivity

    paths = config.resolve_paths(use_synthetic=True)
    data_dir = paths["data_dir"]

    ahp_res = run_ahp(pd.read_csv(data_dir / "ahp_pairwise.csv"))
    delphi_res = run_delphi(pd.read_csv(data_dir / "delphi_ratings.csv"))
    scores = pd.read_csv(data_dir / "scores.csv")
    score_res = score_index(scores, ahp_res["weights"])
    sens_res = run_sensitivity(scores, ahp_res["weights"])
    bench = pd.read_csv(data_dir / "benchmark.csv")

    written = make_all_figures(score_res, delphi_res, sens_res, bench,
                               ahp_domain_weights=ahp_res["domain_weights"],
                               figure_dir=paths["figure_dir"])
    for stem, files in written.items():
        print(f"  {stem}: " + ", ".join(p.name for p in files))
