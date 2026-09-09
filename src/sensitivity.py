"""
src/sensitivity.py -- robustness of the POSRRI to the weighting scheme.

Composite indicators are only as credible as their weights, so the standard
robustness check is to re-run the aggregation under an alternative weighting
and ask whether the substantive conclusions -- above all the *ranking* of
domains, which is what drives policy priorities -- survive.

This module compares:

* **AHP weights**  -- the elicited, consistency-checked weights from
  ``src.ahp``; and
* **equal weights** -- every one of the 50 indicators weighted 1/50, the
  standard "no information" null.

It reports the Spearman rank correlation between the two domain rankings
(with Kendall's tau-b as a tie-robust companion), the per-domain score deltas
and rank shifts, and the change in the overall index.

Run ``python src/sensitivity.py`` to execute against the synthetic dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from src import structure as st  # noqa: E402
from src.ahp import equal_weights  # noqa: E402
from src.scoring import score_index  # noqa: E402


def run_sensitivity(scores: pd.DataFrame,
                    ahp_weights: pd.DataFrame,
                    weight_col: str = "global_weight") -> Dict[str, object]:
    """Compare AHP-weighted and equally weighted aggregation.

    Returns a dict with:

    ``domains``   10-row comparison table (scores, deltas, ranks, rank shifts);
    ``pillars``   3-row equivalent at pillar level;
    ``weights``   per-indicator AHP vs equal weight, with the ratio;
    ``overall``   both index values and their difference;
    ``spearman``  rho and p-value for the domain rankings;
    ``kendall``   tau-b and p-value (tie-robust companion);
    ``summary``   headline numbers for reporting in the paper.
    """
    ahp_res = score_index(scores, ahp_weights, weight_col=weight_col, label="AHP")
    eq_res = score_index(scores, equal_weights(), weight_col="global_weight",
                         label="Equal")

    # ---- domain-level comparison -----------------------------------------
    a = ahp_res["domains"].set_index("domain_code")
    e = eq_res["domains"].set_index("domain_code")

    domains = pd.DataFrame({
        "domain_code": st.DOMAIN_ORDER,
        "domain_name": [st.DOMAINS[d] for d in st.DOMAIN_ORDER],
        "pillar_code": [st.DOMAIN_TO_PILLAR[d] for d in st.DOMAIN_ORDER],
        "weight_ahp": a.loc[st.DOMAIN_ORDER, "weight"].to_numpy(),
        "weight_equal": e.loc[st.DOMAIN_ORDER, "weight"].to_numpy(),
        "score_ahp": a.loc[st.DOMAIN_ORDER, "score_0_100"].to_numpy(),
        "score_equal": e.loc[st.DOMAIN_ORDER, "score_0_100"].to_numpy(),
    })
    domains["delta"] = domains["score_ahp"] - domains["score_equal"]
    domains["abs_delta"] = domains["delta"].abs()
    # Rank 1 = highest readiness.
    domains["rank_ahp"] = domains["score_ahp"].rank(ascending=False, method="min").astype(int)
    domains["rank_equal"] = domains["score_equal"].rank(ascending=False, method="min").astype(int)
    domains["rank_shift"] = domains["rank_equal"] - domains["rank_ahp"]

    # ---- rank correlation -------------------------------------------------
    rho, rho_p = stats.spearmanr(domains["score_ahp"], domains["score_equal"])
    tau, tau_p = stats.kendalltau(domains["score_ahp"], domains["score_equal"])

    # ---- pillar-level comparison -----------------------------------------
    pa = ahp_res["pillars"].set_index("pillar_code")
    pe = eq_res["pillars"].set_index("pillar_code")
    order = list(st.PILLARS.keys())
    pillars = pd.DataFrame({
        "pillar_code": order,
        "pillar_name": [st.PILLARS[p] for p in order],
        "score_ahp": pa.loc[order, "score_0_100"].to_numpy(),
        "score_equal": pe.loc[order, "score_0_100"].to_numpy(),
    })
    pillars["delta"] = pillars["score_ahp"] - pillars["score_equal"]

    # ---- indicator weight comparison -------------------------------------
    w_ahp = ahp_res["indicators"][["indicator_code", "domain_code", "weight"]].copy()
    w_ahp = w_ahp.rename(columns={"weight": "weight_ahp"})
    w_ahp["weight_equal"] = 1.0 / st.N_INDICATORS
    w_ahp["weight_ratio"] = w_ahp["weight_ahp"] / w_ahp["weight_equal"]

    overall = {
        "POSRRI_ahp": ahp_res["overall"]["POSRRI_0_100"],
        "POSRRI_equal": eq_res["overall"]["POSRRI_0_100"],
        "delta": ahp_res["overall"]["POSRRI_0_100"] - eq_res["overall"]["POSRRI_0_100"],
        "band_ahp": ahp_res["overall"]["band"],
        "band_equal": eq_res["overall"]["band"],
        "band_unchanged": ahp_res["overall"]["band"] == eq_res["overall"]["band"],
    }

    summary = {
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "kendall_tau": float(tau),
        "kendall_p": float(tau_p),
        "max_abs_domain_delta": float(domains["abs_delta"].max()),
        "mean_abs_domain_delta": float(domains["abs_delta"].mean()),
        "domain_with_max_delta": str(domains.loc[domains["abs_delta"].idxmax(), "domain_code"]),
        "n_rank_changes": int((domains["rank_shift"] != 0).sum()),
        "max_rank_shift": int(domains["rank_shift"].abs().max()),
        "top3_ahp": list(domains.nsmallest(3, "rank_ahp")["domain_code"]),
        "top3_equal": list(domains.nsmallest(3, "rank_equal")["domain_code"]),
        "bottom3_ahp": list(domains.nlargest(3, "rank_ahp")["domain_code"]),
        "bottom3_equal": list(domains.nlargest(3, "rank_equal")["domain_code"]),
        "POSRRI_ahp": overall["POSRRI_ahp"],
        "POSRRI_equal": overall["POSRRI_equal"],
        "POSRRI_delta": overall["delta"],
        "conclusion_robust": bool(rho >= 0.80 and abs(overall["delta"]) < 10.0),
    }

    return {
        "domains": domains,
        "pillars": pillars,
        "weights": w_ahp,
        "overall": overall,
        "spearman": {"rho": float(rho), "p_value": float(rho_p), "n": len(domains)},
        "kendall": {"tau": float(tau), "p_value": float(tau_p), "n": len(domains)},
        "summary": summary,
        "ahp_result": ahp_res,
        "equal_result": eq_res,
    }


if __name__ == "__main__":
    from src.ahp import run_ahp

    data_dir = config.SYNTHETIC_DATA_DIR
    ahp_res = run_ahp(pd.read_csv(data_dir / "ahp_pairwise.csv"))
    res = run_sensitivity(pd.read_csv(data_dir / "scores.csv"), ahp_res["weights"])

    print("Sensitivity: AHP vs equal weights")
    print("-" * 72)
    print(res["domains"][["domain_code", "score_ahp", "score_equal", "delta",
                          "rank_ahp", "rank_equal", "rank_shift"]]
          .round(2).to_string(index=False))
    s = res["summary"]
    print(f"\nSpearman rho = {s['spearman_rho']:.3f} (p = {s['spearman_p']:.4g}), "
          f"Kendall tau = {s['kendall_tau']:.3f}")
    print(f"POSRRI: AHP {s['POSRRI_ahp']:.2f} vs equal {s['POSRRI_equal']:.2f} "
          f"(delta {s['POSRRI_delta']:+.2f})")
    print(f"Rank changes: {s['n_rank_changes']} of 10 domains, "
          f"max shift {s['max_rank_shift']}")
    print(f"Conclusion robust: {s['conclusion_robust']}")
