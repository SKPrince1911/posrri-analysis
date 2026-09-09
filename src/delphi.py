"""
src/delphi.py -- Delphi consensus and content-validity analysis for POSRRI.

For every one of the 50 candidate indicators, and separately for each Delphi
round, the module computes:

* **median relevance** and the **interquartile range (IQR)** of the panel's
  1-9 relevance ratings;
* the **percentage of experts rating 7-9** (the "relevant" band);
* a **consensus flag**, defined by the study protocol as

      consensus = (>= 80% of experts rate 7-9)  OR  (median >= 7 AND IQR <= 2)

  i.e. either strong majority endorsement or a high and tightly clustered
  central tendency;
* the **item-level content validity index (I-CVI)** = proportion of experts
  rating 7-9. Items with I-CVI >= 0.78 are retained (Zamanzadeh et al. 2015;
  the 0.78 cut-off is the conventional value for panels of six or more);
* the **modified kappa** ``k* = (I-CVI - Pc) / (1 - Pc)`` where ``Pc`` is the
  probability of chance agreement, which corrects I-CVI for chance inflation
  (Zamanzadeh et al. 2015);
* the **scale-level content validity index**, averaging method,
  ``S-CVI/Ave = mean(I-CVI)``, reported against a target of >= 0.90, together
  with ``S-CVI/UA`` (universal agreement) for completeness.

Stability across rounds is quantified by **Cohen's kappa** between the round-1
and round-2 retain/drop decisions (Hohmann et al. 2025 recommend reporting
both a consensus criterion and a stability criterion for Delphi studies).

Run ``python src/delphi.py`` to execute against the synthetic dataset.
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from src import structure as st  # noqa: E402

REQUIRED_COLUMNS = [
    "expert_id", "round", "indicator_code", "relevance", "clarity", "feasibility",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_delphi_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Check columns, coerce dtypes and range-check the rating scales."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"delphi_ratings data is missing column(s) {missing}; "
            f"expected {REQUIRED_COLUMNS}"
        )

    out = df.copy()
    out["expert_id"] = out["expert_id"].astype(str).str.strip()
    out["indicator_code"] = out["indicator_code"].astype(str).str.strip()
    out["round"] = pd.to_numeric(out["round"], errors="raise").astype(int)
    for col in ("relevance", "clarity", "feasibility"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    unknown = sorted(set(out["indicator_code"]) - set(st.INDICATORS))
    if unknown:
        raise ValueError(
            f"{len(unknown)} unknown indicator code(s) in the Delphi data, "
            f"e.g. {unknown[:5]}"
        )

    for col, (lo, hi) in (("relevance", config.RELEVANCE_SCALE),
                          ("clarity", config.CLARITY_SCALE),
                          ("feasibility", config.FEASIBILITY_SCALE)):
        bad = out[col].dropna()
        bad = bad[(bad < lo) | (bad > hi)]
        if len(bad):
            raise ValueError(
                f"{len(bad)} {col} rating(s) outside the {lo}-{hi} scale"
            )

    if out["relevance"].isna().any():
        n_na = int(out["relevance"].isna().sum())
        raise ValueError(f"{n_na} missing relevance rating(s); these must be resolved "
                         f"or the rows removed before analysis")
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def chance_agreement(n_experts: int, n_agreeing: int) -> float:
    """Probability of chance agreement ``Pc`` for the modified kappa.

    ``Pc = [N! / (A! (N - A)!)] * 0.5^N`` (Zamanzadeh et al. 2015), i.e. the
    binomial probability of exactly ``A`` of ``N`` raters endorsing an item by
    chance when endorsement is a coin flip.
    """
    return comb(int(n_experts), int(n_agreeing)) * (0.5 ** int(n_experts))


def modified_kappa(icvi: float, n_experts: int, n_agreeing: int) -> float:
    """Chance-corrected content validity, ``k* = (I-CVI - Pc) / (1 - Pc)``."""
    pc = chance_agreement(n_experts, n_agreeing)
    if pc >= 1.0:
        return float("nan")
    return (icvi - pc) / (1.0 - pc)


def kappa_evaluation(kappa_star: float) -> str:
    """Conventional qualitative bands for the modified kappa."""
    if not np.isfinite(kappa_star):
        return "undefined"
    if kappa_star > 0.74:
        return "excellent"
    if kappa_star >= 0.60:
        return "good"
    if kappa_star >= 0.40:
        return "fair"
    return "poor"


def cohens_kappa(labels_a: Sequence, labels_b: Sequence) -> Dict[str, float]:
    """Cohen's kappa for two nominal labellings of the same items.

    Returns ``kappa``, observed agreement ``po``, expected agreement ``pe`` and
    ``n``. When one rater uses a single category for every item, ``pe`` can
    equal 1 and kappa is mathematically undefined; ``nan`` is returned in that
    case and ``po`` is still reported, rather than raising.
    """
    a = np.asarray(list(labels_a))
    b = np.asarray(list(labels_b))
    if a.shape != b.shape:
        raise ValueError(f"label vectors differ in length: {a.shape} vs {b.shape}")
    n = a.size
    if n == 0:
        raise ValueError("cannot compute kappa on an empty set of items")

    categories = sorted(set(a.tolist()) | set(b.tolist()), key=str)
    idx = {c: k for k, c in enumerate(categories)}
    table = np.zeros((len(categories), len(categories)), dtype=float)
    for x, y in zip(a, b):
        table[idx[x], idx[y]] += 1.0

    po = float(np.trace(table) / n)
    pe = float((table.sum(axis=0) * table.sum(axis=1)).sum() / (n * n))
    kappa = float("nan") if np.isclose(pe, 1.0) else (po - pe) / (1.0 - pe)

    return {
        "kappa": float(kappa),
        "po": po,
        "pe": pe,
        "n": int(n),
        "categories": categories,
        "table": pd.DataFrame(table, index=categories, columns=categories, dtype=int),
    }


def _round_stats(block: pd.DataFrame) -> pd.DataFrame:
    """Per-indicator relevance statistics for a single Delphi round."""
    rows = []
    for ind, sub in block.groupby("indicator_code", sort=False):
        rel = sub["relevance"].to_numpy(dtype=float)
        n = rel.size
        n_high = int((rel >= config.RELEVANCE_HIGH_MIN).sum())
        icvi = n_high / n

        q1, q3 = np.percentile(rel, [25, 75])
        iqr = float(q3 - q1)
        median = float(np.median(rel))

        rule_pct = icvi >= config.CONSENSUS_PCT_THRESHOLD
        rule_central = (median >= config.CONSENSUS_MEDIAN_MIN) and (iqr <= config.CONSENSUS_IQR_MAX)
        k_star = modified_kappa(icvi, n, n_high)

        rows.append({
            "indicator_code": ind,
            "n_experts": n,
            "median_relevance": median,
            "q1_relevance": float(q1),
            "q3_relevance": float(q3),
            "iqr_relevance": iqr,
            "mean_relevance": float(rel.mean()),
            "n_rating_7_9": n_high,
            "pct_rating_7_9": 100.0 * icvi,
            "I_CVI": icvi,
            "kappa_star": k_star,
            "kappa_evaluation": kappa_evaluation(k_star),
            "consensus_rule_pct": bool(rule_pct),
            "consensus_rule_median_iqr": bool(rule_central),
            "consensus": bool(rule_pct or rule_central),
            "retain": bool(icvi >= config.ICVI_RETAIN_THRESHOLD),
            "mean_clarity": float(sub["clarity"].mean()),
            "mean_feasibility": float(sub["feasibility"].mean()),
        })

    frame = pd.DataFrame(rows)
    frame["_order"] = frame["indicator_code"].map({c: i for i, c in enumerate(st.INDICATORS)})
    return frame.sort_values("_order").drop(columns="_order").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_delphi(df: pd.DataFrame) -> Dict[str, object]:
    """Full Delphi analysis.

    Returns a dict with:

    ``long``          per indicator x round statistics (100 rows for 2 rounds);
    ``per_indicator`` one row per indicator, round 1 and round 2 side by side,
                      plus the final retain decision (taken from the last round);
    ``retained``      the retained indicators as a tidy table;
    ``dropped``       the indicators failing the I-CVI cut-off in the last round;
    ``kappa``         Cohen's kappa between round-1 and round-2 decisions;
    ``summary``       S-CVI/Ave and S-CVI/UA per round, counts and kappa.
    """
    df = validate_delphi_frame(df)

    rounds = sorted(df["round"].unique())
    if not rounds:
        raise ValueError("no Delphi rounds present in the data")

    long_parts = []
    for rnd in rounds:
        part = _round_stats(df[df["round"] == rnd])
        part.insert(1, "round", rnd)
        long_parts.append(part)
    long = pd.concat(long_parts, ignore_index=True)

    # ---- wide, one row per indicator --------------------------------------
    keep = ["median_relevance", "iqr_relevance", "pct_rating_7_9", "I_CVI",
            "kappa_star", "kappa_evaluation", "consensus", "retain",
            "mean_clarity", "mean_feasibility"]
    wide = st.structure_frame()[
        ["indicator_code", "indicator_name", "domain_code", "domain_name",
         "pillar_code"]
    ].copy()
    for rnd in rounds:
        part = long[long["round"] == rnd].set_index("indicator_code")[keep]
        part = part.add_suffix(f"_r{rnd}")
        wide = wide.merge(part, left_on="indicator_code", right_index=True, how="left")

    last = rounds[-1]
    wide["final_retain"] = wide[f"retain_r{last}"].astype(bool)
    wide["final_consensus"] = wide[f"consensus_r{last}"].astype(bool)
    wide["final_I_CVI"] = wide[f"I_CVI_r{last}"]
    wide["decision"] = np.where(wide["final_retain"], "RETAIN", "DROP")

    # ---- stability between rounds -----------------------------------------
    if len(rounds) >= 2:
        first, second = rounds[0], rounds[-1]
        lab_a = wide[f"retain_r{first}"].map({True: "RETAIN", False: "DROP"})
        lab_b = wide[f"retain_r{second}"].map({True: "RETAIN", False: "DROP"})
        kappa = cohens_kappa(lab_a, lab_b)
        kappa["rounds_compared"] = (int(first), int(second))
        kappa["n_changed"] = int((lab_a.to_numpy() != lab_b.to_numpy()).sum())
    else:
        kappa = {"kappa": float("nan"), "po": float("nan"), "pe": float("nan"),
                 "n": len(wide), "categories": [], "table": pd.DataFrame(),
                 "rounds_compared": (rounds[0], rounds[0]), "n_changed": 0}

    # ---- scale-level content validity -------------------------------------
    summary = {
        "n_indicators": int(len(wide)),
        "n_experts": int(df["expert_id"].nunique()),
        "rounds": [int(r) for r in rounds],
        "icvi_retain_threshold": config.ICVI_RETAIN_THRESHOLD,
        "scvi_target": config.SCVI_TARGET,
        "cohens_kappa": float(kappa["kappa"]),
        "kappa_po": float(kappa["po"]),
        "kappa_pe": float(kappa["pe"]),
        "n_decisions_changed": int(kappa["n_changed"]),
    }
    for rnd in rounds:
        part = long[long["round"] == rnd]
        summary[f"scvi_ave_r{rnd}"] = float(part["I_CVI"].mean())
        summary[f"scvi_ua_r{rnd}"] = float((part["I_CVI"] >= 1.0).mean())
        summary[f"n_consensus_r{rnd}"] = int(part["consensus"].sum())
        summary[f"n_retained_r{rnd}"] = int(part["retain"].sum())
    summary["scvi_ave_retained"] = float(
        long[(long["round"] == last)
             & (long["retain"])]["I_CVI"].mean()
    )
    summary["scvi_target_met"] = bool(summary[f"scvi_ave_r{last}"] >= config.SCVI_TARGET)

    retained = wide.loc[wide["final_retain"],
                        ["indicator_code", "indicator_name", "domain_code",
                         "domain_name", "pillar_code", "final_I_CVI",
                         f"median_relevance_r{last}", f"iqr_relevance_r{last}"]
                        ].reset_index(drop=True)
    dropped = wide.loc[~wide["final_retain"],
                       ["indicator_code", "indicator_name", "domain_code",
                        "final_I_CVI", f"median_relevance_r{last}"]
                       ].reset_index(drop=True)

    return {
        "long": long,
        "per_indicator": wide,
        "retained": retained,
        "dropped": dropped,
        "kappa": kappa,
        "summary": summary,
    }


if __name__ == "__main__":
    data = pd.read_csv(config.SYNTHETIC_DATA_DIR / "delphi_ratings.csv")
    res = run_delphi(data)
    s = res["summary"]
    print("Delphi summary")
    print("-" * 60)
    for k, v in s.items():
        print(f"  {k:<26} {v}")
    print(f"\nRetained {len(res['retained'])} of {s['n_indicators']} indicators; "
          f"dropped {len(res['dropped'])}.")
    print("\nDropped indicators:")
    print(res["dropped"].to_string(index=False))
    print("\nRound-1 vs round-2 decision table:")
    print(res["kappa"]["table"].to_string())
