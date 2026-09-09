"""
src/scoring.py -- aggregation of indicator scores into the POSRRI.

Each of the 50 indicators carries an adjudicated ``final_score`` on a 0-3
rubric and an AHP ``global_weight`` (see ``src.ahp``). Aggregation is a
weighted-sum-then-normalise at each level of the hierarchy:

    achieved(g)      = sum_{i in g} w_i * s_i
    max_achievable(g) = sum_{i in g} w_i * 3
    score(g)          = 100 * achieved(g) / max_achievable(g)

Dividing by the *group's own* weight mass means a domain score is comparable
across domains regardless of how much of the global weight the domain holds:
it is a weighted mean of the 0-3 scores, rescaled to 0-100. Because the 50
global weights sum to 1, the overall POSRRI reduces to
``100 * sum(w_i * s_i) / 3``.

Run ``python src/scoring.py`` to execute against the synthetic dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from src import structure as st  # noqa: E402

REQUIRED_COLUMNS = [
    "indicator_code", "doc_score", "field_score", "final_score", "evidence_source",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_scores_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Check the scores table: columns, coverage of all 50 indicators, ranges."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"scores data is missing column(s) {missing}; expected {REQUIRED_COLUMNS}"
        )

    out = df.copy()
    out["indicator_code"] = out["indicator_code"].astype(str).str.strip()
    for col in ("doc_score", "field_score", "final_score"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    unknown = sorted(set(out["indicator_code"]) - set(st.INDICATORS))
    if unknown:
        raise ValueError(f"unknown indicator code(s) in scores: {unknown[:5]}")

    dupes = out["indicator_code"][out["indicator_code"].duplicated()].unique()
    if len(dupes):
        raise ValueError(f"duplicate rows for indicator(s) {list(dupes)[:5]}")

    absent = [c for c in st.INDICATORS if c not in set(out["indicator_code"])]
    if absent:
        raise ValueError(
            f"scores missing for {len(absent)} indicator(s), e.g. {absent[:5]}; "
            f"the index requires all {st.N_INDICATORS}"
        )

    lo, hi = config.SCORE_SCALE
    for col in ("doc_score", "field_score", "final_score"):
        bad = out[col][(out[col] < lo) | (out[col] > hi) | out[col].isna()]
        if len(bad):
            raise ValueError(f"{len(bad)} {col} value(s) outside the {lo}-{hi} rubric or missing")

    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _normalise(achieved: float, weight_mass: float) -> float:
    """Rescale a weighted score to 0-100 given the group's weight mass."""
    denom = weight_mass * config.MAX_INDICATOR_SCORE
    return float("nan") if denom == 0 else 100.0 * achieved / denom


def score_index(scores: pd.DataFrame,
                weights: pd.DataFrame,
                weight_col: str = "global_weight",
                label: str = "AHP") -> Dict[str, object]:
    """Compute indicator, domain, pillar and overall POSRRI scores.

    Parameters
    ----------
    scores
        Long table with ``indicator_code`` and the 0-3 score columns.
    weights
        Table with ``indicator_code`` and a weight column (default the AHP
        ``global_weight``). Weights are renormalised to sum to 1 defensively.
    weight_col
        Name of the weight column in ``weights``.
    label
        Name of the weighting scheme, carried into the output tables so that
        AHP and equal-weight runs can be concatenated.

    Returns a dict with ``indicators``, ``domains``, ``pillars`` and ``overall``.
    """
    scores = validate_scores_frame(scores)
    if weight_col not in weights.columns:
        raise ValueError(f"weights table has no column {weight_col!r}")

    w = weights[["indicator_code", weight_col]].copy()
    w["indicator_code"] = w["indicator_code"].astype(str).str.strip()
    absent = [c for c in st.INDICATORS if c not in set(w["indicator_code"])]
    if absent:
        raise ValueError(f"weights missing for {len(absent)} indicator(s), e.g. {absent[:5]}")
    if not np.all(w[weight_col].to_numpy() > 0):
        raise ValueError("all indicator weights must be strictly positive")
    w[weight_col] = w[weight_col] / w[weight_col].sum()

    ind = (st.structure_frame()
           .merge(scores, on="indicator_code", how="left")
           .merge(w, on="indicator_code", how="left"))

    ind["weight"] = ind[weight_col]
    ind["weighted_score"] = ind["weight"] * ind["final_score"]
    ind["max_weighted_score"] = ind["weight"] * config.MAX_INDICATOR_SCORE
    ind["score_0_100"] = 100.0 * ind["final_score"] / config.MAX_INDICATOR_SCORE
    ind["implementation_gap"] = ind["doc_score"] - ind["field_score"]
    # Share of the total index shortfall attributable to this indicator.
    shortfall = ind["weight"] * (config.MAX_INDICATOR_SCORE - ind["final_score"])
    ind["shortfall_share_pct"] = 100.0 * shortfall / shortfall.sum()
    ind["contribution_pct"] = 100.0 * ind["weighted_score"] / ind["weighted_score"].sum()
    ind["weighting"] = label

    def _aggregate(group_col: str, name_col: str) -> pd.DataFrame:
        agg = (ind.groupby(group_col, sort=False)
                  .agg(weight=("weight", "sum"),
                       achieved=("weighted_score", "sum"),
                       max_achievable=("max_weighted_score", "sum"),
                       mean_final_score=("final_score", "mean"),
                       mean_doc_score=("doc_score", "mean"),
                       mean_field_score=("field_score", "mean"),
                       n_indicators=("indicator_code", "size"))
                  .reset_index())
        agg["score_0_100"] = [
            _normalise(a, m) for a, m in zip(agg["achieved"], agg["weight"])
        ]
        agg["band"] = agg["score_0_100"].map(st.readiness_band)
        agg["weighting"] = label
        names = ind[[group_col, name_col]].drop_duplicates()
        return agg.merge(names, on=group_col, how="left")

    domains = _aggregate("domain_code", "domain_name")
    domains = domains.merge(
        ind[["domain_code", "pillar_code"]].drop_duplicates(), on="domain_code", how="left")
    domains["_order"] = domains["domain_code"].map(
        {d: i for i, d in enumerate(st.DOMAIN_ORDER)})
    domains = domains.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    domains["rank"] = domains["score_0_100"].rank(ascending=False, method="min").astype(int)

    pillars = _aggregate("pillar_code", "pillar_name")
    pillars = pillars.sort_values("pillar_code").reset_index(drop=True)

    achieved = float(ind["weighted_score"].sum())
    max_achievable = float(ind["max_weighted_score"].sum())
    posrri = _normalise(achieved, float(ind["weight"].sum()))

    overall = {
        "weighting": label,
        "POSRRI_0_100": posrri,
        "achieved_weighted_score": achieved,
        "max_achievable_weighted_score": max_achievable,
        "mean_final_score_0_3": float(ind["final_score"].mean()),
        "unweighted_POSRRI_0_100": 100.0 * float(ind["final_score"].mean()) / config.MAX_INDICATOR_SCORE,
        "band": st.readiness_band(posrri),
        "n_indicators": int(len(ind)),
        "weight_sum": float(ind["weight"].sum()),
        "weakest_domain": str(domains.loc[domains["score_0_100"].idxmin(), "domain_code"]),
        "strongest_domain": str(domains.loc[domains["score_0_100"].idxmax(), "domain_code"]),
        "mean_implementation_gap": float(ind["implementation_gap"].mean()),
    }

    cols = ["indicator_code", "indicator_name", "domain_code", "domain_name",
            "pillar_code", "pillar_name", "doc_score", "field_score",
            "final_score", "implementation_gap", "evidence_source", "weight",
            "weighted_score", "max_weighted_score", "score_0_100",
            "contribution_pct", "shortfall_share_pct", "weighting"]

    return {
        "indicators": ind[cols],
        "domains": domains,
        "pillars": pillars,
        "overall": overall,
    }


def benchmark_matrix(benchmark: pd.DataFrame,
                     cpa_domain_scores: "pd.Series | None" = None) -> pd.DataFrame:
    """Reshape the benchmark table to domains x ports on the 0-100 scale.

    ``benchmark`` holds 0-3 domain scores per port. If ``cpa_domain_scores``
    (a 0-100 series indexed by domain code, produced by :func:`score_index`)
    is supplied, it replaces the CPA column so that the comparison figure uses
    the weighted index result rather than a separately stated CPA value.
    """
    required = {"domain_code", "port", "score"}
    missing = required - set(benchmark.columns)
    if missing:
        raise ValueError(f"benchmark data is missing column(s) {sorted(missing)}")

    wide = (benchmark.pivot_table(index="domain_code", columns="port",
                                  values="score", aggfunc="mean")
                     .reindex(st.DOMAIN_ORDER))
    wide = 100.0 * wide / config.MAX_INDICATOR_SCORE

    if cpa_domain_scores is not None:
        wide["CPA"] = pd.Series(cpa_domain_scores).reindex(st.DOMAIN_ORDER).to_numpy()

    ordered = [p for p in st.BENCHMARK_PORTS if p in wide.columns]
    extra = [p for p in wide.columns if p not in ordered]
    wide = wide[ordered + extra]
    wide.index.name = "domain_code"
    return wide


if __name__ == "__main__":
    from src.ahp import run_ahp

    data_dir = config.SYNTHETIC_DATA_DIR
    ahp_res = run_ahp(pd.read_csv(data_dir / "ahp_pairwise.csv"))
    res = score_index(pd.read_csv(data_dir / "scores.csv"), ahp_res["weights"])

    print("Overall POSRRI")
    print("-" * 60)
    for k, v in res["overall"].items():
        print(f"  {k:<32} {v if not isinstance(v, float) else round(v, 4)}")
    print("\nDomain scores (AHP-weighted)")
    print(res["domains"][["domain_code", "domain_name", "weight",
                          "score_0_100", "band", "rank"]]
          .round(3).to_string(index=False))
    print("\nPillar scores")
    print(res["pillars"][["pillar_code", "pillar_name", "weight",
                          "score_0_100", "band"]].round(3).to_string(index=False))
