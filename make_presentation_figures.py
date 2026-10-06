"""Presentation figures from the study data. Run:  python make_presentation_figures.py
Reads data/raw/scores.csv and data/raw/benchmark.csv; writes PNGs to
outputs/presentation_figures. Equal weights; independent of the AHP stage.
Every label is a plain-language name; indicator codes are not shown.
"""
from pathlib import Path
import textwrap
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
HERE = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "presentation_figures"; OUT.mkdir(parents=True, exist_ok=True)
import sys
sys.path.insert(0, str(ROOT))
from src import structure as st  # noqa: E402

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8983", "#e6e5e1", "#ffffff"
BLUE = "#2a78d6"
LEVEL = {1: "#86b6ef", 2: "#2a78d6", 3: "#0d366b"}      # ordinal ramp, one hue
RUBRIC = {0: "Absent", 1: "Partial or ambiguous", 2: "Defined", 3: "Defined and verified"}

PILLARS = {"A": ("Governance", ["D1", "D2", "D9"]),
           "B": ("Operational preparedness", ["D3", "D4", "D5", "D6"]),
           "C": ("Resilience and financing", ["D7", "D8", "D10"])}
DOMAIN = {"D1": "Legal and regulatory basis", "D2": "Authority and command",
          "D3": "Notification and reporting", "D4": "Tiered response",
          "D5": "Resources and equipment", "D6": "Training and exercises",
          "D7": "Funding and liability", "D8": "Plan currency and documentation",
          "D9": "Stakeholder coordination", "D10": "Monitoring, review and learning"}
PILLAR_OF = {d: p for p, (_, ds) in PILLARS.items() for d in ds}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 220})

s = pd.read_csv(HERE / "scores.csv")
s["domain"] = s.indicator_code.str.split(".").str[0]
s["indicator_name"] = [st.INDICATOR_NAMES[c.split(".")[0]][int(c.split(".")[1]) - 1] for c in s.indicator_code]
dom = s.groupby("domain", sort=False).final_score.agg(["sum", "count"])
dom["pct"] = dom["sum"] / (dom["count"] * 3) * 100
overall = s.final_score.sum() / (len(s) * 3) * 100
pil = {p: s[s.domain.isin(ds)].final_score.sum() / (s.domain.isin(ds).sum() * 3) * 100
       for p, (_, ds) in PILLARS.items()}


def clean(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0, colors=INK2)


def title(fig, main, sub, y=None):
    h = fig.get_figheight()
    fig.text(0.04, 1 - 0.18 / h, main, fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 1 - 0.62 / h, sub, fontsize=11, color=INK2, va="top")


def note(fig, txt):
    fig.text(0.04, 0.02, textwrap.fill(txt, 150), fontsize=8.5, color=MUTED, va="bottom", linespacing=1.4)


# ---------------------------------------------------------------- Fig 1
def fig1():
    fig = plt.figure(figsize=(12, 6.2))
    title(fig, "Chattogram Port is about one-third of the way to full regulatory readiness",
          "Port Oil Spill Regulatory Readiness Index, 50 indicators scored 0 to 3, equal weights")
    fig.text(0.04, 0.66, f"{overall:.0f}%", fontsize=92, fontweight="bold", color=INK, va="top")
    fig.text(0.045, 0.33, f"{int(s.final_score.sum())} of {len(s) * 3} points", fontsize=14, color=INK2)
    fig.text(0.045, 0.27, "overall readiness", fontsize=12, color=MUTED)
    ax = fig.add_axes([0.52, 0.2, 0.42, 0.52])
    names = [PILLARS[p][0] for p in "ABC"][::-1]; vals = [pil[p] for p in "ABC"][::-1]
    ax.barh(names, [100] * 3, color="#f1f0ed", height=0.42)
    ax.barh(names, vals, color=BLUE, height=0.42)
    for n, v in zip(names, vals):
        ax.text(v + 2, n, f"{v:.0f}%", va="center", fontsize=15, fontweight="bold", color=INK)
    ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.set_xticklabels(["0", "50", "100%"])
    ax.tick_params(axis="y", labelsize=13, labelcolor=INK); clean(ax)
    fig.text(0.52, 0.76, "By pillar", fontsize=12, color=INK2)
    note(fig, "Scores rest on NOSCOP 2020, peer-reviewed legal analysis and verified public record. "
              "No indicator reaches 'verified' (3), so the practical ceiling today is 67%.")
    fig.savefig(OUT / "fig1_overall_readiness.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig 2
def fig2():
    d = dom.sort_values("pct")
    fig = plt.figure(figsize=(12, 6.8))
    title(fig, "Readiness is strongest where it only has to be written down",
          "Readiness by domain, % of maximum score. Weakest domains are those that must be funded, exercised and reviewed.")
    ax = fig.add_axes([0.30, 0.12, 0.50, 0.7])
    labs = [DOMAIN[i] for i in d.index]
    ax.barh(labs, [100] * len(d), color="#f1f0ed", height=0.5)
    ax.barh(labs, d.pct, color=BLUE, height=0.5)
    for lab, i, v in zip(labs, d.index, d.pct):
        ax.text(v + 1.5, lab, f"{v:.0f}%", va="center", fontsize=12, fontweight="bold", color=INK, zorder=5,
                bbox=dict(fc="#f1f0ed", ec="none", pad=1.5))
        ax.text(101.5, lab, PILLARS[PILLAR_OF[i]][0], va="center", fontsize=9, color=MUTED)
    ax.plot([overall, overall], [-0.5, len(d) - 0.3], color=INK2, lw=1)
    ax.text(overall + 1, len(d) - 0.42, f"overall {overall:.0f}%", fontsize=9.5, color=INK2, ha="left")
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.tick_params(axis="y", labelsize=12, labelcolor=INK); clean(ax)
    note(fig, "Each domain has five indicators scored 0 to 3. Grey text shows the pillar.")
    fig.savefig(OUT / "fig2_domain_profile.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig 3 (one per pillar)
def pips(ax, x0, y, score, w=0.052, h=0.56, gap=0.008):
    for k in (1, 2, 3):
        filled = k <= score
        ax.add_patch(FancyBboxPatch((x0 + (k - 1) * (w + gap), y - h / 2), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.006",
                                    fc=LEVEL[score] if filled else SURF,
                                    ec=LEVEL[score] if filled else "#c9c8c3", lw=1))


def fig3(p):
    pname, ds = PILLARS[p]
    sub = s[s.domain.isin(ds)]
    rows = len(sub) + len(ds) * 1.5
    fig = plt.figure(figsize=(12, 0.42 * rows + 2.2))
    top = 1 - 1.1 / fig.get_figheight()
    title(fig, f"{pname}: every indicator, in plain words",
          f"Pillar readiness {pil[p]:.0f}%. Filled blocks show the score out of 3; an empty row means the provision is absent.")
    ax = fig.add_axes([0.04, 0.5 / fig.get_figheight(), 0.92, top - 0.5 / fig.get_figheight()])
    ax.set_xlim(0, 1); ax.set_ylim(rows + 0.3, -0.3); ax.axis("off")
    y = 0.4
    for dcode in ds:
        dd = sub[sub.domain == dcode]
        ax.text(0, y, DOMAIN[dcode], fontsize=13, fontweight="bold", color=INK, va="center")
        ax.text(0.985, y, f"{dom.loc[dcode, 'pct']:.0f}%", fontsize=13, fontweight="bold", color=INK, va="center", ha="right")
        ax.plot([0, 0.985], [y + 0.5, y + 0.5], color=GRID, lw=1)
        y += 1.1
        for _, r in dd.iterrows():
            ax.text(0.015, y, r.indicator_name, fontsize=11, color=INK, va="center")
            pips(ax, 0.66, y, int(r.final_score))
            ax.text(0.86, y, RUBRIC[int(r.final_score)], fontsize=10, color=INK2, va="center")
            y += 1.0
        y += 0.4
    note(fig, "Scale: 0 absent, 1 partial or ambiguous, 2 defined, 3 defined and verified by drill, exercise or incident.")
    fig.savefig(OUT / f"fig3{p.lower()}_indicators_{pname.split()[0].lower()}.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig 4
def fig4():
    order = dom.sort_values("pct", ascending=True).index
    cnt = s.groupby(["domain", "final_score"]).size().unstack(fill_value=0).reindex(columns=[0, 1, 2, 3], fill_value=0)
    fig = plt.figure(figsize=(12, 6.6))
    title(fig, "Twelve of fifty provisions are absent, and none has been verified in practice",
          "Number of indicators at each score level, by domain")
    ax = fig.add_axes([0.33, 0.2, 0.6, 0.62])
    for i, dcode in enumerate(order):
        left = 0
        for lvl in (0, 1, 2, 3):
            n = cnt.loc[dcode, lvl]
            if n == 0: continue
            ax.barh(i, n - 0.06, left=left + 0.03, height=0.5,
                    color=SURF if lvl == 0 else LEVEL[lvl], edgecolor="#b9b8b3" if lvl == 0 else LEVEL[lvl], lw=1)
            ax.text(left + n / 2, i, str(n), ha="center", va="center", fontsize=11,
                    color=INK2 if lvl == 0 else (INK if lvl == 1 else "white"), fontweight="bold")
            left += n
    ax.set_yticks(range(len(order))); ax.set_yticklabels([DOMAIN[d] for d in order], fontsize=12, color=INK)
    ax.set_xlim(0, 5); ax.set_xticks(range(6)); ax.set_xlabel("indicators", color=INK2); clean(ax)
    tot = s.final_score.value_counts().reindex([0, 1, 2, 3], fill_value=0)
    x = 0.33
    for lvl in (0, 1, 2, 3):
        fig.patches.append(plt.Rectangle((x, 0.085), 0.016, 0.026, transform=fig.transFigure,
                           fc=SURF if lvl == 0 else LEVEL[lvl], ec="#b9b8b3" if lvl == 0 else LEVEL[lvl], lw=1))
        fig.text(x + 0.022, 0.098, f"{lvl}  {RUBRIC[lvl]} ({tot[lvl]})", fontsize=10, color=INK2, va="center")
        x += {0: 0.115, 1: 0.2, 2: 0.14}.get(lvl, 0)
    fig.savefig(OUT / "fig4_score_distribution.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig 5
def fig5():
    b = pd.read_csv(HERE / "benchmark.csv")
    w = b.pivot(index="domain_code", columns="port", values="score")
    others = [c for c in w.columns if c != "CPA"]
    w["lo"], w["hi"] = w[others].min(axis=1), w[others].max(axis=1)
    w["gap"] = w[others].mean(axis=1) - w["CPA"]
    w = w.sort_values("gap")
    fig = plt.figure(figsize=(12, 6.8))
    title(fig, "The gap to established systems is widest in funding, plans and coordination",
          "Domain score (0 to 3). Chattogram compared with Singapore, Los Angeles–Long Beach and Australia.")
    ax = fig.add_axes([0.33, 0.25, 0.6, 0.58])
    for i, (dcode, r) in enumerate(w.iterrows()):
        ax.plot([r.CPA, r.lo], [i, i], color="#d9d8d3", lw=2, zorder=1)
        ax.plot([r.lo - 0.03, r.hi + 0.03], [i, i], color="#8a8983", lw=9, solid_capstyle="round", zorder=2)
        ax.scatter(r.CPA, i, s=150, color=BLUE, edgecolor=SURF, lw=2, zorder=3)
        ax.text(r.CPA - 0.09, i, f"{r.CPA:.1f}", ha="right", va="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_yticks(range(len(w))); ax.set_yticklabels([DOMAIN[d] for d in w.index], fontsize=12, color=INK)
    ax.set_xlim(-0.05, 3.15); ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["0\nAbsent", "1\nPartial", "2\nDefined", "3\nVerified"], fontsize=10)
    ax.grid(axis="x", color=GRID, lw=1); ax.set_axisbelow(True); clean(ax)
    ax.scatter([], [], s=110, color=BLUE, label="Chattogram Port")
    ax.plot([], [], color="#8a8983", lw=8, solid_capstyle="round", label="Range of the three benchmark systems")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.16), ncol=2, frameon=False, fontsize=10.5, labelcolor=INK2)
    note(fig, "Benchmark scores are desk-based, from each system's own published plan (ITOPF 2020; USCG/CDFW 2024; AMSA 2020), "
              "and describe national or area systems rather than audits of a single port.")
    fig.savefig(OUT / "fig5_benchmark_gap.png"); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); [fig3(p) for p in "ABC"]; fig4(); fig5()
    print(f"overall {overall:.1f}%", {k: round(v, 1) for k, v in pil.items()})
    print("figures written to", OUT)
