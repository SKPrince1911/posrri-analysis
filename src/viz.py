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
SCORE_LABELS = {0: "0 absent", 1: "1 partial", 2: "2 substantial", 3: "3 fully met"}

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
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7, color=INK_MUTED)
    ax.set_rlabel_position(180 / n)
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

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    im = ax.imshow(grid.to_numpy(dtype=float), cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(st.INDICATORS_PER_DOMAIN))
    ax.set_xticklabels([f"Indicator .{k}" for k in range(1, st.INDICATORS_PER_DOMAIN + 1)])
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

    for i, dom in enumerate(st.DOMAIN_ORDER):
        for j in range(st.INDICATORS_PER_DOMAIN):
            val = grid.iloc[i, j]
            if pd.isna(val):
                continue
            ax.text(j, i, f"{int(round(val))}", ha="center", va="center",
                    fontsize=8.5, fontweight="bold",
                    color="white" if val >= 2 else INK)

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], pad=0.02, shrink=0.82)
    cbar.ax.set_yticklabels([SCORE_LABELS[k] for k in range(4)], fontsize=7.5)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)

    ax.set_title("Indicator-level readiness scores, grouped by domain")
    fig.text(0.02, 0.015,
             "Cells show the adjudicated 0-3 score for each of the 50 indicators "
             "(D1.1 ... D10.5).",
             fontsize=7.5, color=INK_MUTED)
    return save_figure(fig, stem, figure_dir)


# ---------------------------------------------------------------------------
# (d) Work-as-imagined vs work-as-done
# ---------------------------------------------------------------------------

def fig_implementation_gap(indicators: pd.DataFrame,
                           figure_dir: "Path | str" = config.FIGURE_DIR,
                           stem: str = "fig4_implementation_gap") -> List[Path]:
    """Diverging bars of ``doc_score - field_score`` for all 50 indicators.

    Positive bars are indicators where the documentation claims more than
    field verification found (work-as-imagined ahead of work-as-done);
    negative bars are undocumented practice. The zero line is neutral grey and
    the two directions carry different hues, in line with diverging-scheme
    practice.
    """
    apply_style()
    df = indicators.copy()
    if "implementation_gap" not in df.columns:
        df["implementation_gap"] = df["doc_score"] - df["field_score"]
    df["_order"] = df["indicator_code"].map({c: i for i, c in enumerate(st.INDICATORS)})
    df = df.sort_values("_order")

    y = np.arange(len(df))[::-1]
    gaps = df["implementation_gap"].to_numpy(dtype=float)
    colours = [GAP_POSITIVE if g > 0 else (GAP_NEGATIVE if g < 0 else GAP_NEUTRAL)
               for g in gaps]

    fig, ax = plt.subplots(figsize=(6.8, 10.2))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.barh(y, gaps, height=0.7, color=colours, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=INK_MUTED, linewidth=1.0)
    # A zero-width bar is invisible, which would read as missing data; mark the
    # no-gap indicators explicitly with a neutral dot on the zero line.
    zero = gaps == 0
    ax.scatter(np.zeros(int(zero.sum())), y[zero], s=16, marker="o",
               facecolor=GAP_NEUTRAL, edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(df["indicator_code"], fontsize=6.5)
    ax.set_ylim(y.min() - 0.8, y.max() + 0.8)
    lim = max(1.0, float(np.abs(gaps).max())) + 0.4
    ax.set_xlim(-lim, lim)
    ax.set_xticks(np.arange(-int(lim), int(lim) + 1))
    ax.set_xlabel("Documentary score minus field score (0-3 rubric points)")
    ax.set_title("Work-as-imagined versus work-as-done, by indicator")

    # Domain separators keep 50 rows navigable.
    for k in range(1, st.N_DOMAINS):
        ax.axhline(y[k * st.INDICATORS_PER_DOMAIN - 1] - 0.5,
                   color=GRID, linewidth=0.7, zorder=0)
    for k, dom in enumerate(st.DOMAIN_ORDER):
        centre = y[k * st.INDICATORS_PER_DOMAIN + 2]
        ax.text(-lim * 0.985, centre, dom, fontsize=7.5, fontweight="bold",
                color=INK_MUTED, ha="left", va="center")

    handles = [
        Patch(facecolor=GAP_POSITIVE, label="Documentation ahead of practice (gap > 0)"),
        Patch(facecolor=GAP_NEUTRAL, label="No gap"),
        Patch(facecolor=GAP_NEGATIVE, label="Practice ahead of documentation (gap < 0)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.045), ncol=1)
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

    n_cons = int(consensus.sum())
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.045), ncol=1)
    fig.text(0.5, 0.004,
             f"Consensus: >=80% of experts rating 7-9, or median >=7 with IQR <=2. "
             f"{n_cons} of {len(df)} indicators met the criterion.",
             ha="center", fontsize=7.5, color=INK_MUTED)
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

    lo, hi = 0, 100
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, linestyle="--", linewidth=1.0,
            zorder=1, label="1:1 (weighting has no effect)")

    for pil in st.PILLARS:
        block = df[df["pillar_code"] == pil]
        if block.empty:
            continue
        ax.scatter(block["score_equal"], block["score_ahp"], s=64,
                   color=config.PILLAR_COLOURS[pil], edgecolor="white",
                   linewidth=0.9, zorder=3,
                   label=f"Pillar {pil}: {st.PILLARS[pil]}")

    # Domains with near-identical scores would print their labels on top of one
    # another; alternate the offset for any point close to an earlier one.
    placed: List[tuple] = []
    offsets = [(7, 4), (7, -11), (-16, 4), (-16, -11)]
    for _, row in df.iterrows():
        xy = (float(row["score_equal"]), float(row["score_ahp"]))
        clashes = sum(1 for px, py in placed
                      if abs(px - xy[0]) < 4.0 and abs(py - xy[1]) < 4.0)
        ax.annotate(row["domain_code"], xy, textcoords="offset points",
                    xytext=offsets[clashes % len(offsets)],
                    fontsize=8, color=INK, fontweight="bold")
        placed.append(xy)

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
