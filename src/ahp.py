"""
src/ahp.py -- Analytic Hierarchy Process weights for POSRRI.

Implements Saaty's (1980) AHP exactly as specified in the study protocol:

* pairwise judgements are supplied in long form and may use Saaty integers
  (``1`` .. ``9``) or fraction strings (``1/3``); reciprocals are filled
  automatically, so only the upper triangle needs to be elicited;
* priority weights are derived by the **row geometric mean** method
  (normalised), which is scale-invariant and robust to small perturbations;
* consistency is reported per matrix as
  ``lambda_max = mean((A.w)/w)``, ``CI = (lambda_max - n)/(n - 1)`` and
  ``CR = CI / RI(n)``, with ``CR >= 0.10`` flagged;
* experts are combined by **Aggregation of Individual Judgements (AIJ)**:
  the geometric mean of the judgements is taken cell-by-cell *before* the
  group priority vector is derived. AIJ (rather than aggregating individual
  priorities) is the appropriate choice when the panel is acting as a single
  synthetic decision maker, and it preserves the reciprocal property;
* the global weight of an indicator is
  ``pillar_w x domain_w_within_pillar x indicator_w_within_domain``,
  normalised to sum to 1 across all 50 indicators.

Run ``python src/ahp.py`` to execute the module against the synthetic dataset.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from src import structure as st  # noqa: E402

REQUIRED_COLUMNS = ["expert_id", "level", "group", "item_i", "item_j", "saaty_value"]

#: Label used for the pairwise matrix that compares the three pillars.
GOAL_GROUP = "GOAL"


# ---------------------------------------------------------------------------
# Parsing and matrix construction
# ---------------------------------------------------------------------------

def parse_saaty(value) -> float:
    """Parse one judgement into a positive float.

    Accepts ``3``, ``3.0``, ``"3"``, ``"1/3"``, ``"0.333"`` and unicode
    fraction-like input with surrounding whitespace. Raises ``ValueError`` for
    non-positive values or anything outside the admissible 1/9..9 window.
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        if isinstance(value, float) and np.isnan(value):
            raise ValueError("saaty_value is NaN")
        out = float(value)
    else:
        text = str(value).strip().replace("⁄", "/")
        if not text:
            raise ValueError("saaty_value is empty")
        try:
            out = float(Fraction(text)) if "/" in text else float(text)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"cannot parse saaty_value {value!r}") from exc

    if out <= 0:
        raise ValueError(f"saaty_value must be positive, got {value!r}")
    lo, hi = 1.0 / 9.0 - 1e-9, 9.0 + 1e-9
    if not (lo <= out <= hi):
        raise ValueError(
            f"saaty_value {value!r} = {out:.4f} is outside the Saaty scale "
            f"[1/9, 9]"
        )
    return out


def build_matrix(judgements: pd.DataFrame, items: Sequence[str]) -> np.ndarray:
    """Assemble a reciprocal pairwise matrix for ``items``.

    ``judgements`` needs the columns ``item_i``, ``item_j``, ``saaty_value``.
    Only ``i < j`` cells are required. If a pair is supplied more than once
    (for example both directions, or a repeat), the geometric mean of all
    evidence for that direction is used -- the ratio-scale analogue of taking
    an average, and the only combination rule that keeps ``a_ij = 1 / a_ji``.
    """
    index = {code: k for k, code in enumerate(items)}
    n = len(items)
    evidence: Dict[Tuple[int, int], List[float]] = {}

    for row in judgements.itertuples(index=False):
        i_code, j_code = str(row.item_i).strip(), str(row.item_j).strip()
        if i_code not in index or j_code not in index:
            raise KeyError(
                f"judgement references {i_code!r} vs {j_code!r}, which are not "
                f"both members of {list(items)}"
            )
        if i_code == j_code:
            continue                                   # diagonal is fixed at 1
        val = parse_saaty(row.saaty_value)
        a, b = index[i_code], index[j_code]
        if a < b:
            evidence.setdefault((a, b), []).append(val)
        else:
            evidence.setdefault((b, a), []).append(1.0 / val)

    A = np.ones((n, n), dtype=float)
    missing = []
    for i in range(n):
        for j in range(i + 1, n):
            vals = evidence.get((i, j))
            if not vals:
                missing.append((items[i], items[j]))
                continue
            v = float(np.exp(np.mean(np.log(vals))))    # geometric mean
            A[i, j] = v
            A[j, i] = 1.0 / v

    if missing:
        raise ValueError(
            f"incomplete pairwise matrix: missing {len(missing)} comparison(s), "
            f"e.g. {missing[:3]}"
        )
    return A


# ---------------------------------------------------------------------------
# Priorities and consistency
# ---------------------------------------------------------------------------

def priority_weights(A: np.ndarray) -> np.ndarray:
    """Priority vector by the normalised row geometric mean method."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {A.shape}")
    if np.any(A <= 0):
        raise ValueError("pairwise matrices must be strictly positive")
    gm = np.exp(np.mean(np.log(A), axis=1))
    return gm / gm.sum()


def consistency_ratio(A: np.ndarray, weights: "np.ndarray | None" = None) -> Dict[str, float]:
    """Saaty consistency statistics for one pairwise matrix.

    Returns ``n``, ``lambda_max``, ``CI``, ``RI``, ``CR`` and ``consistent``.
    For ``n <= 2`` a reciprocal matrix is consistent by construction and
    ``RI = 0``; CR is reported as 0.0 rather than dividing by zero.
    """
    A = np.asarray(A, dtype=float)
    w = priority_weights(A) if weights is None else np.asarray(weights, dtype=float)
    n = A.shape[0]

    aw = A @ w
    lambda_max = float(np.mean(aw / w))
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = st.random_index(n)
    cr = 0.0 if ri == 0 else ci / ri

    return {
        "n": int(n),
        "lambda_max": lambda_max,
        "CI": float(ci),
        "RI": float(ri),
        "CR": float(cr),
        "consistent": bool(cr < config.CR_THRESHOLD),
    }


def aggregate_judgements(matrices: Iterable[np.ndarray]) -> np.ndarray:
    """Aggregate Individual Judgements (AIJ): cell-wise geometric mean.

    The geometric mean is the only aggregation rule that preserves the
    reciprocal property ``a_ij = 1 / a_ji`` of the group matrix.
    """
    stack = np.stack([np.asarray(m, dtype=float) for m in matrices], axis=0)
    if stack.size == 0:
        raise ValueError("no matrices to aggregate")
    return np.exp(np.mean(np.log(stack), axis=0))


# ---------------------------------------------------------------------------
# Hierarchy plumbing
# ---------------------------------------------------------------------------

def expected_matrices() -> List[Tuple[str, str, List[str]]]:
    """The 14 matrices the POSRRI hierarchy requires: 1 + 3 + 10."""
    out: List[Tuple[str, str, List[str]]] = [
        ("pillar", GOAL_GROUP, list(st.PILLARS.keys()))
    ]
    for pil, doms in st.PILLAR_DOMAINS.items():
        out.append(("domain", pil, list(doms)))
    for dom in st.DOMAIN_ORDER:
        out.append(("indicator", dom, list(st.DOMAIN_INDICATORS[dom])))
    return out


def validate_ahp_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Check required columns exist and normalise whitespace/case."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"ahp_pairwise data is missing column(s) {missing}; "
            f"expected {REQUIRED_COLUMNS}"
        )
    out = df.copy()
    for col in ("expert_id", "level", "group", "item_i", "item_j"):
        out[col] = out[col].astype(str).str.strip()
    out["level"] = out["level"].str.lower()

    bad_levels = sorted(set(out["level"]) - set(config.AHP_LEVELS))
    if bad_levels:
        raise ValueError(
            f"unknown AHP level(s) {bad_levels}; expected {list(config.AHP_LEVELS)}"
        )
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ahp(df: pd.DataFrame) -> Dict[str, object]:
    """Derive POSRRI weights from long-form pairwise judgements.

    Returns a dict with:

    ``consistency``  per-matrix CR report (every expert plus the AIJ group
                     matrix for each of the 14 matrices);
    ``local_weights`` local priority vector of every node under its parent;
    ``weights``      the 50-row indicator weight table (local + global);
    ``domain_weights`` 10-row domain table (local within pillar + global);
    ``pillar_weights`` 3-row pillar table;
    ``group_matrices`` the aggregated matrices, for auditing;
    ``summary``      counts of inconsistent matrices and the weight checksum.
    """
    df = validate_ahp_frame(df)

    consistency_rows: List[dict] = []
    local_weights: Dict[Tuple[str, str], pd.Series] = {}
    group_matrices: Dict[Tuple[str, str], np.ndarray] = {}

    for level, group, items in expected_matrices():
        block = df[(df["level"] == level) & (df["group"] == group)]
        if block.empty:
            raise ValueError(
                f"no pairwise judgements found for level={level!r} group={group!r}; "
                f"the hierarchy needs all {len(expected_matrices())} matrices"
            )

        per_expert: List[np.ndarray] = []
        for expert, sub in block.groupby("expert_id", sort=True):
            A = build_matrix(sub, items)
            stats = consistency_ratio(A)
            per_expert.append(A)
            consistency_rows.append({
                "level": level, "group": group, "expert_id": expert,
                "n_items": stats["n"], "lambda_max": stats["lambda_max"],
                "CI": stats["CI"], "RI": stats["RI"], "CR": stats["CR"],
                "consistent": stats["consistent"],
                "flag": "" if stats["consistent"] else "CR>=0.10",
            })

        # AIJ: aggregate the judgements, then derive the group priorities.
        G = aggregate_judgements(per_expert)
        w = priority_weights(G)
        g_stats = consistency_ratio(G, w)
        consistency_rows.append({
            "level": level, "group": group, "expert_id": "AIJ_GROUP",
            "n_items": g_stats["n"], "lambda_max": g_stats["lambda_max"],
            "CI": g_stats["CI"], "RI": g_stats["RI"], "CR": g_stats["CR"],
            "consistent": g_stats["consistent"],
            "flag": "" if g_stats["consistent"] else "CR>=0.10",
        })

        group_matrices[(level, group)] = G
        local_weights[(level, group)] = pd.Series(w, index=items, name="local_weight")

    consistency = pd.DataFrame(consistency_rows)
    consistency = consistency.sort_values(
        ["level", "group", "expert_id"],
        key=lambda s: s.map({"pillar": 0, "domain": 1, "indicator": 2}).fillna(s)
        if s.name == "level" else s,
    ).reset_index(drop=True)

    # ---- pillar weights ---------------------------------------------------
    pillar_w = local_weights[("pillar", GOAL_GROUP)]
    pillars = pd.DataFrame({
        "pillar_code": pillar_w.index,
        "pillar_name": [st.PILLARS[p] for p in pillar_w.index],
        "pillar_weight": pillar_w.to_numpy(),
    })

    # ---- domain weights ---------------------------------------------------
    dom_rows = []
    for pil in st.PILLARS:
        dw = local_weights[("domain", pil)]
        for dom in dw.index:
            dom_rows.append({
                "domain_code": dom,
                "domain_name": st.DOMAINS[dom],
                "pillar_code": pil,
                "pillar_name": st.PILLARS[pil],
                "pillar_weight": float(pillar_w[pil]),
                "domain_weight_within_pillar": float(dw[dom]),
                "domain_weight_global": float(pillar_w[pil] * dw[dom]),
            })
    domains = (pd.DataFrame(dom_rows)
               .set_index("domain_code")
               .loc[st.DOMAIN_ORDER]
               .reset_index())

    # ---- indicator weights ------------------------------------------------
    ind_rows = []
    for dom in st.DOMAIN_ORDER:
        iw = local_weights[("indicator", dom)]
        pil = st.DOMAIN_TO_PILLAR[dom]
        dom_local = float(local_weights[("domain", pil)][dom])
        for ind in st.DOMAIN_INDICATORS[dom]:
            ind_rows.append({
                "indicator_code": ind,
                "indicator_name": st.INDICATOR_LABELS[ind],
                "domain_code": dom,
                "domain_name": st.DOMAINS[dom],
                "pillar_code": pil,
                "pillar_name": st.PILLARS[pil],
                "pillar_weight": float(pillar_w[pil]),
                "domain_weight_within_pillar": dom_local,
                "indicator_weight_within_domain": float(iw[ind]),
                "global_weight_raw": float(pillar_w[pil] * dom_local * iw[ind]),
            })
    weights = pd.DataFrame(ind_rows)
    total = weights["global_weight_raw"].sum()
    weights["global_weight"] = weights["global_weight_raw"] / total
    weights = weights.drop(columns="global_weight_raw")

    # Keep the domain table consistent with the normalised indicator weights.
    domains = domains.merge(
        weights.groupby("domain_code", sort=False)["global_weight"].sum()
               .rename("domain_weight_normalised"),
        on="domain_code", how="left",
    )

    n_bad = int((~consistency["consistent"]).sum())
    summary = {
        "n_matrices": len(expected_matrices()),
        "n_experts": int(df["expert_id"].nunique()),
        "n_consistency_rows": len(consistency),
        "n_inconsistent": n_bad,
        "max_CR": float(consistency["CR"].max()),
        "global_weight_sum": float(weights["global_weight"].sum()),
        "cr_threshold": config.CR_THRESHOLD,
    }

    return {
        "consistency": consistency,
        "local_weights": local_weights,
        "group_matrices": group_matrices,
        "pillar_weights": pillars,
        "domain_weights": domains,
        "weights": weights,
        "summary": summary,
    }


def equal_weights() -> pd.DataFrame:
    """Benchmark weighting: all 50 indicators weighted equally.

    Used by ``src.sensitivity`` as the null model against which the AHP
    weighting is tested.
    """
    frame = st.structure_frame().copy()
    frame["global_weight"] = 1.0 / st.N_INDICATORS
    return frame


if __name__ == "__main__":
    data = pd.read_csv(config.SYNTHETIC_DATA_DIR / "ahp_pairwise.csv")
    res = run_ahp(data)
    print("AHP summary:", res["summary"])
    print("\nPillar weights:\n", res["pillar_weights"].to_string(index=False))
    print("\nTop 8 indicators by global weight:")
    print(res["weights"].nlargest(8, "global_weight")[
        ["indicator_code", "domain_code", "global_weight"]].to_string(index=False))
