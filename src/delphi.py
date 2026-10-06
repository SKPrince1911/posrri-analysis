"""
src/delphi.py -- Delphi expert review of the candidate indicators.

This panel is small -- a handful of experts, not a survey panel -- so the
analysis is deliberately austere. With four or five raters, a "percentage in
agreement" is a proportion of four or five: 80% and 75% are the same two
people, and a chance-corrected statistic (Cohen's or Fleiss' kappa) estimated
on that many raters has a confidence interval wide enough to be uninformative.
Reporting either would dress up a judgement as a measurement.

What is reported instead:

* **every expert's rating for every indicator**, in full, so a reader can see
  the raw judgements rather than a summary of them;
* the **range** across experts (minimum, median, maximum) per indicator;
* the **all-experts-agree rule**: an indicator is retained when *every* expert
  rates its relevance in the agreed band (>= ``config.RELEVANCE_HIGH_MIN``).
  One dissenting expert is enough to flag an indicator for discussion. With a
  panel this size that is the only defensible decision rule: it needs no
  threshold tuned to the panel's size, and it cannot be moved by rounding.

Indicators that are not unanimous are listed separately, with the dissenting
experts named, because those are the ones the panel has to talk about.

Run ``python src/delphi.py`` to execute against the configured data directory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from src import structure as st  # noqa: E402

REQUIRED_COLUMNS = ["expert_id", "indicator_code", "relevance"]

#: Columns used when present, but not required: a single-round review has no
#: ``round`` column, and some panels do not rate clarity or feasibility.
OPTIONAL_COLUMNS = ["round", "clarity", "feasibility"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_delphi_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Check columns, coerce dtypes and range-check the rating scales."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"delphi_ratings data is missing column(s) {missing}; "
            f"required {REQUIRED_COLUMNS}, optional {OPTIONAL_COLUMNS}"
        )

    out = df.copy()
    out["expert_id"] = out["expert_id"].astype(str).str.strip()
    out["indicator_code"] = out["indicator_code"].astype(str).str.strip()
    out["relevance"] = pd.to_numeric(out["relevance"], errors="coerce")

    if "round" in out.columns:
        out["round"] = pd.to_numeric(out["round"], errors="raise").astype(int)
    for col in ("clarity", "feasibility"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    unknown = sorted(set(out["indicator_code"]) - set(st.INDICATORS))
    if unknown:
        raise ValueError(
            f"{len(unknown)} unknown indicator code(s) in the Delphi data, "
            f"e.g. {unknown[:5]}"
        )

    checks = [("relevance", config.RELEVANCE_SCALE)]
    if "clarity" in out.columns:
        checks.append(("clarity", config.CLARITY_SCALE))
    if "feasibility" in out.columns:
        checks.append(("feasibility", config.FEASIBILITY_SCALE))
    for col, (lo, hi) in checks:
        bad = out[col].dropna()
        bad = bad[(bad < lo) | (bad > hi)]
        if len(bad):
            raise ValueError(
                f"{len(bad)} {col} rating(s) outside the {lo}-{hi} scale: "
                f"{sorted(set(bad.tolist()))[:5]}"
            )

    if out["relevance"].isna().any():
        n_na = int(out["relevance"].isna().sum())
        raise ValueError(
            f"{n_na} missing relevance rating(s). With a panel this small every "
            f"rating matters to the unanimity rule, so these must be resolved "
            f"rather than dropped."
        )

    duplicated = out.duplicated(subset=[c for c in ("expert_id", "round", "indicator_code")
                                        if c in out.columns])
    if duplicated.any():
        rows = out[duplicated].head(3)[["expert_id", "indicator_code"]]
        raise ValueError(
            f"{int(duplicated.sum())} duplicate expert/indicator rating(s), "
            f"e.g.\n{rows.to_string(index=False)}"
        )
    return out


# ---------------------------------------------------------------------------
# Per-indicator ratings
# ---------------------------------------------------------------------------

def ratings_matrix(df: pd.DataFrame, value: str = "relevance") -> pd.DataFrame:
    """Indicators x experts matrix of raw ratings.

    One row per indicator in canonical order, one column per expert. This is
    the full record of what the panel said; everything else in this module is
    derived from it.
    """
    wide = (df.pivot_table(index="indicator_code", columns="expert_id",
                           values=value, aggfunc="last")
              .reindex([c for c in st.INDICATORS if c in set(df["indicator_code"])]))
    wide.columns.name = None
    wide.index.name = "indicator_code"
    return wide


def indicator_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-indicator summary: every rating, the range, and the agreement flag.

    Columns: the hierarchy, one column per expert, then ``n_experts``,
    ``min_relevance``, ``median_relevance``, ``max_relevance``,
    ``n_below_threshold``, ``dissenting_experts``, ``all_experts_agree`` and
    ``decision``.
    """
    wide = ratings_matrix(df)
    experts = list(wide.columns)
    threshold = config.RELEVANCE_HIGH_MIN

    frame = st.structure_frame()[
        ["indicator_code", "indicator_name", "domain_code", "domain_name",
         "pillar_code"]
    ].copy()
    frame = frame.merge(wide, left_on="indicator_code", right_index=True,
                        how="left")

    values = frame[experts].to_numpy(dtype=float)
    below = values < threshold

    frame["n_experts"] = np.sum(~np.isnan(values), axis=1).astype(int)
    frame["min_relevance"] = np.nanmin(values, axis=1)
    frame["median_relevance"] = np.nanmedian(values, axis=1)
    frame["max_relevance"] = np.nanmax(values, axis=1)
    frame["n_below_threshold"] = np.nansum(below, axis=1).astype(int)
    frame["dissenting_experts"] = [
        ", ".join(e for e, flag in zip(experts, row) if flag) or ""
        for row in below
    ]
    # The rule: every expert rates the indicator at or above the threshold.
    frame["all_experts_agree"] = frame["n_below_threshold"].eq(0)
    frame["decision"] = np.where(frame["all_experts_agree"], "RETAIN", "DISCUSS")

    for col in ("clarity", "feasibility"):
        if col in df.columns:
            means = df.groupby("indicator_code")[col].mean()
            frame[f"mean_{col}"] = frame["indicator_code"].map(means)

    return frame


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_delphi(df: pd.DataFrame) -> Dict[str, object]:
    """Full Delphi review.

    Returns a dict with:

    ``ratings``       indicators x experts matrix of raw relevance ratings;
    ``per_indicator`` the per-indicator table, including the agreement flag;
    ``retained``      indicators every expert rated at or above the threshold;
    ``to_discuss``    indicators with at least one dissenting expert;
    ``summary``       panel size, counts, and the rule in force.
    """
    df = validate_delphi_frame(df)

    rounds = sorted(df["round"].unique()) if "round" in df.columns else []
    if rounds:
        # Only the final round carries the panel's settled view; earlier rounds
        # are kept in the ratings matrix for the record but do not decide.
        final = df[df["round"] == rounds[-1]]
    else:
        final = df

    ratings = ratings_matrix(final)
    per_indicator = indicator_table(final)
    experts = list(ratings.columns)

    retained = per_indicator.loc[
        per_indicator["all_experts_agree"],
        ["indicator_code", "indicator_name", "domain_code", "domain_name",
         "pillar_code", "min_relevance", "median_relevance"]
    ].reset_index(drop=True)

    to_discuss = per_indicator.loc[
        ~per_indicator["all_experts_agree"],
        ["indicator_code", "indicator_name", "domain_code",
         "min_relevance", "median_relevance", "n_below_threshold",
         "dissenting_experts"]
    ].reset_index(drop=True)

    summary = {
        "n_experts": len(experts),
        "experts": experts,
        "rounds": [int(r) for r in rounds],
        "deciding_round": int(rounds[-1]) if rounds else None,
        "n_indicators": int(len(per_indicator)),
        "relevance_threshold": config.RELEVANCE_HIGH_MIN,
        "rule": (f"retain when all {len(experts)} experts rate relevance "
                 f">= {config.RELEVANCE_HIGH_MIN}"),
        "n_retained": int(per_indicator["all_experts_agree"].sum()),
        "n_to_discuss": int((~per_indicator["all_experts_agree"]).sum()),
        "min_rating_overall": float(per_indicator["min_relevance"].min()),
        "n_unanimous_at_ceiling": int(
            (per_indicator["min_relevance"] >= config.RELEVANCE_SCALE[1]).sum()),
    }

    return {
        "ratings": ratings.reset_index(),
        "per_indicator": per_indicator,
        "retained": retained,
        "to_discuss": to_discuss,
        "summary": summary,
    }


if __name__ == "__main__":
    paths = config.resolve_paths()
    data = pd.read_csv(paths["data_dir"] / config.DATA_FILES["delphi"])
    res = run_delphi(data)
    s = res["summary"]

    print("Delphi expert review")
    print("-" * 72)
    print(f"  Experts                    {s['n_experts']}  ({', '.join(s['experts'])})")
    if s["rounds"]:
        print(f"  Rounds                     {s['rounds']} "
              f"(round {s['deciding_round']} decides)")
    print(f"  Indicators reviewed        {s['n_indicators']}")
    print(f"  Rule                       {s['rule']}")
    print(f"  Retained (all agree)       {s['n_retained']}")
    print(f"  Flagged for discussion     {s['n_to_discuss']}")

    print("\nPer-indicator ratings (every expert, final round)")
    print(res["per_indicator"][
        ["indicator_name"] + list(res["ratings"].columns[1:])
        + ["min_relevance", "all_experts_agree"]
    ].to_string(index=False))

    if len(res["to_discuss"]):
        print("\nIndicators without unanimous agreement")
        print(res["to_discuss"][
            ["indicator_name", "min_relevance", "n_below_threshold",
             "dissenting_experts"]
        ].to_string(index=False))
