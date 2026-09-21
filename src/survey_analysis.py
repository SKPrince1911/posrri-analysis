"""
src/survey_analysis.py -- descriptive analysis of the stakeholder survey.

Three products:

* **Descriptive statistics.** n, mean, SD and median for the seven five-point
  Likert items; frequency tables (count and percent per category, with the
  original option labels restored) for the coded categorical items; response
  counts for the free-text items.
* **Cronbach's alpha** over the seven Likert confidence items
  (q1, q3, q4, q5, q8, q9, q10) as an internal-consistency check, reported with
  the corrected item-total correlations and alpha-if-item-deleted, which are
  what identify a misbehaving item.
* **Figure 7**, the stacked Likert distribution chart, rendered by
  :func:`src.viz.fig_survey_likert` so it shares the styling, the 300 dpi PNG
  plus vector PDF output and the colour rules of the other figures.

Alpha is computed on complete cases across the seven items (listwise
deletion), which is the standard definition; the number of respondents dropped
is reported rather than left implicit.

Run ``python src/survey_analysis.py`` to execute against the synthetic dataset.
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
from src import viz  # noqa: E402
from src.survey_ingest import (ALLOWED_VALUES, ITEM_LABELS,  # noqa: E402
                               LIKERT_ITEMS, ORG_GROUPS, ORG_OTHER,
                               ORG_SHORT_LABELS, OUTPUT_COLUMNS, RECODE,
                               TEXT_ITEMS, category_labels,
                               organisation_audit, read_survey)

#: The internal-consistency scale: the seven Likert confidence/adequacy items.
CONFIDENCE_SCALE = list(LIKERT_ITEMS)

#: Coded items that are ordinal categories rather than the five-point Likert.
CATEGORICAL_ITEMS = [item for item in OUTPUT_COLUMNS if item in RECODE]

#: Reported columns (everything but the identifier).
REPORTED_COLUMNS = [c for c in OUTPUT_COLUMNS if c != "respondent_id"]


# ---------------------------------------------------------------------------
# Cronbach's alpha
# ---------------------------------------------------------------------------

def cronbach_alpha(frame: pd.DataFrame) -> Dict[str, object]:
    """Cronbach's alpha for the items in ``frame`` (one column per item).

    ``alpha = k / (k - 1) * (1 - sum(item variances) / variance of the total)``
    using sample variances (ddof = 1) and complete cases only.

    Returns ``alpha``, the number of items and respondents used, the number
    excluded by listwise deletion, a per-item table (corrected item-total
    correlation and alpha with that item deleted), and a qualitative band.
    """
    numeric = frame.apply(pd.to_numeric, errors="coerce").astype(float)
    n_total = len(numeric)
    data = numeric.dropna(axis=0, how="any")
    k = data.shape[1]

    if k < 2:
        raise ValueError(f"Cronbach's alpha needs at least 2 items, got {k}")
    if len(data) < 3:
        raise ValueError(
            f"only {len(data)} complete case(s) across {k} items; "
            f"alpha is not estimable"
        )

    def _alpha(block: pd.DataFrame) -> float:
        m = block.shape[1]
        item_var = block.var(axis=0, ddof=1).sum()
        total_var = block.sum(axis=1).var(ddof=1)
        if total_var == 0:
            return float("nan")
        return (m / (m - 1)) * (1.0 - item_var / total_var)

    alpha = _alpha(data)

    rows = []
    for item in data.columns:
        others = data.drop(columns=item)
        rest_total = others.sum(axis=1)
        # Corrected item-total correlation: item against the sum of the *rest*,
        # so the item is not correlated with itself.
        if data[item].std(ddof=1) == 0 or rest_total.std(ddof=1) == 0:
            r = float("nan")
        else:
            r = float(np.corrcoef(data[item], rest_total)[0, 1])
        rows.append({
            "item": item,
            "label": ITEM_LABELS.get(item, item),
            "n": int(len(data)),
            "mean": float(data[item].mean()),
            "sd": float(data[item].std(ddof=1)),
            "item_total_r": r,
            "alpha_if_deleted": _alpha(others) if others.shape[1] >= 2 else float("nan"),
        })

    if alpha >= 0.90:
        band = "excellent"
    elif alpha >= 0.80:
        band = "good"
    elif alpha >= 0.70:
        band = "acceptable"
    elif alpha >= 0.60:
        band = "questionable"
    elif alpha >= 0.50:
        band = "poor"
    else:
        band = "unacceptable"

    return {
        "alpha": float(alpha),
        "n_items": int(k),
        "items": list(data.columns),
        "n_respondents": int(len(data)),
        "n_excluded_incomplete": int(n_total - len(data)),
        "item_statistics": pd.DataFrame(rows),
        "interpretation": band,
        "acceptable": bool(alpha >= 0.70),
    }


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def likert_descriptives(survey: pd.DataFrame) -> pd.DataFrame:
    """n, mean, SD, median (plus quartiles and % 1-2 / % 4-5) per Likert item."""
    rows = []
    for item in LIKERT_ITEMS:
        values = pd.to_numeric(survey[item], errors="coerce").dropna().astype(float)
        n = int(values.size)
        pct_low = 100.0 * float((values <= 2).mean()) if n else float("nan")
        pct_high = 100.0 * float((values >= 4).mean()) if n else float("nan")
        rows.append({
            "item": item,
            "label": ITEM_LABELS.get(item, item),
            "n": n,
            "n_missing": int(survey[item].isna().sum()),
            "mean": float(values.mean()) if n else float("nan"),
            "sd": float(values.std(ddof=1)) if n > 1 else float("nan"),
            "median": float(values.median()) if n else float("nan"),
            "q1": float(values.quantile(0.25)) if n else float("nan"),
            "q3": float(values.quantile(0.75)) if n else float("nan"),
            "min": float(values.min()) if n else float("nan"),
            "max": float(values.max()) if n else float("nan"),
            "pct_1_2": pct_low,
            "pct_4_5": pct_high,
        })
    return pd.DataFrame(rows)


def frequency_tables(survey: pd.DataFrame) -> pd.DataFrame:
    """Frequency table for every coded categorical item, in one tidy frame.

    Percentages are of *valid* responses; the missing count is carried on each
    row so the denominator is never ambiguous.
    """
    rows = []
    for item in CATEGORICAL_ITEMS:
        values = pd.to_numeric(survey[item], errors="coerce").dropna().astype(int)
        n_valid, n_missing = int(values.size), int(survey[item].isna().sum())
        labels = category_labels(item)
        codes = sorted(ALLOWED_VALUES[item])
        counts = values.value_counts().reindex(codes, fill_value=0)
        for code in codes:
            count = int(counts[code])
            rows.append({
                "item": item,
                "label": ITEM_LABELS.get(item, item),
                "code": code,
                "category": labels.get(code, str(code)),
                "count": count,
                "pct_of_valid": 100.0 * count / n_valid if n_valid else float("nan"),
                "n_valid": n_valid,
                "n_missing": n_missing,
            })
    return pd.DataFrame(rows)


def organisation_frequencies(survey: pd.DataFrame) -> pd.DataFrame:
    """Respondents per organisation group, in the form's own category order.

    Percentages are of respondents who answered the question; the blank count
    is carried on every row so the denominator is never ambiguous.
    """
    groups = survey["q12_group"]
    n_valid = int(groups.notna().sum())
    n_missing = int(groups.isna().sum())
    counts = groups.value_counts()

    rows = []
    for group in ORG_GROUPS:
        count = int(counts.get(group, 0))
        rows.append({
            "q12_group": group,
            "group_short": ORG_SHORT_LABELS.get(group, group),
            "count": count,
            "pct_of_valid": 100.0 * count / n_valid if n_valid else float("nan"),
            "n_valid": n_valid,
            "n_missing": n_missing,
            "is_residual": group == ORG_OTHER,
        })
    return pd.DataFrame(rows)


def text_response_summary(survey: pd.DataFrame) -> pd.DataFrame:
    """Response counts and the most common answers for the free-text items."""
    rows = []
    for item in TEXT_ITEMS:
        values = survey[item].astype("string").fillna("").str.strip()
        answered = values[values != ""]
        top = answered.value_counts().head(3)
        rows.append({
            "item": item,
            "label": ITEM_LABELS.get(item, item),
            "n_answered": int(answered.size),
            "n_blank": int((values == "").sum()),
            "pct_answered": 100.0 * answered.size / len(values) if len(values) else float("nan"),
            "n_distinct": int(answered.nunique()),
            "most_common": "; ".join(f"{k} ({v})" for k, v in top.items()) or "-",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_survey_analysis(survey: "pd.DataFrame | str | Path",
                        figure_dir: "Path | str | None" = None,
                        make_figure: bool = True) -> Dict[str, object]:
    """Full survey analysis.

    ``survey`` may be a loaded frame or a path to a cleaned ``survey.csv``.
    Returns ``likert``, ``frequencies``, ``text``, ``alpha``, ``figure`` and
    ``summary``.
    """
    if isinstance(survey, (str, Path)):
        survey = read_survey(survey)

    likert = likert_descriptives(survey)
    frequencies = frequency_tables(survey)
    text = text_response_summary(survey)
    organisations = organisation_frequencies(survey)
    org_audit = organisation_audit(survey)
    alpha = cronbach_alpha(survey[CONFIDENCE_SCALE])

    figures: List[Path] = []
    if make_figure:
        target = figure_dir if figure_dir is not None else config.FIGURE_DIR
        figures = list(viz.fig_survey_likert(survey, target))
        figures += list(viz.fig_survey_by_group(survey, target))

    summary = {
        "n_respondents": int(len(survey)),
        "n_items": len(REPORTED_COLUMNS),
        "n_likert_items": len(LIKERT_ITEMS),
        "n_categorical_items": len(CATEGORICAL_ITEMS),
        "n_text_items": len(TEXT_ITEMS),
        "cronbach_alpha": alpha["alpha"],
        "alpha_interpretation": alpha["interpretation"],
        "alpha_n_respondents": alpha["n_respondents"],
        "alpha_n_excluded": alpha["n_excluded_incomplete"],
        "mean_likert_overall": float(likert["mean"].mean()),
        "highest_rated_item": str(likert.loc[likert["mean"].idxmax(), "item"]),
        "lowest_rated_item": str(likert.loc[likert["mean"].idxmin(), "item"]),
        "total_missing": int(sum(survey[c].isna().sum()
                                 for c in REPORTED_COLUMNS
                                 if c not in TEXT_ITEMS)),
        "n_org_groups": int(survey["q12_group"].nunique(dropna=True)),
        "n_org_other": int((survey["q12_group"] == ORG_OTHER).sum()),
        "n_org_missing": int(survey["q12_group"].isna().sum()),
        "largest_org_group": (str(survey["q12_group"].value_counts().idxmax())
                              if survey["q12_group"].notna().any() else "n/a"),
    }

    return {
        "likert": likert,
        "frequencies": frequencies,
        "text": text,
        "organisations": organisations,
        "org_audit": org_audit,
        "alpha": alpha,
        "figure": figures,
        "summary": summary,
    }


if __name__ == "__main__":
    paths = config.resolve_paths(use_synthetic=True)
    res = run_survey_analysis(paths["data_dir"] / config.DATA_FILES["survey"],
                              figure_dir=paths["figure_dir"])

    print("Survey analysis")
    print("=" * 78)
    s = res["summary"]
    print(f"  Respondents                {s['n_respondents']}")
    print(f"  Items                      {s['n_items']} "
          f"({s['n_likert_items']} Likert, {s['n_categorical_items']} coded "
          f"categorical, {s['n_text_items']} free text)")
    print(f"  Missing coded values       {s['total_missing']}")

    print("\nLikert descriptives")
    print("-" * 78)
    print(res["likert"][["item", "label", "n", "mean", "sd", "median",
                         "pct_1_2", "pct_4_5"]]
          .round(2).to_string(index=False))

    print("\nCronbach's alpha, 7-item confidence scale")
    print("-" * 78)
    a = res["alpha"]
    print(f"  alpha = {a['alpha']:.3f}  ({a['interpretation']}; "
          f"acceptable = {a['acceptable']})")
    print(f"  {a['n_items']} items, {a['n_respondents']} complete cases "
          f"({a['n_excluded_incomplete']} excluded)")
    print(a["item_statistics"][["item", "mean", "sd", "item_total_r",
                                "alpha_if_deleted"]]
          .round(3).to_string(index=False))

    print("\nFrequency tables, coded categorical items")
    print("-" * 78)
    print(res["frequencies"][["item", "category", "count", "pct_of_valid",
                              "n_missing"]]
          .round(1).to_string(index=False))

    print("\nOrganisation type (q12): respondents per group")
    print("-" * 78)
    print(res["organisations"][["group_short", "q12_group", "count",
                                "pct_of_valid", "n_missing"]]
          .round(1).to_string(index=False))

    print("\nOrganisation type (q12): every distinct raw answer and its group")
    print("-" * 96)
    print(res["org_audit"].to_string(index=False))

    print("\nFree-text items")
    print("-" * 78)
    print(res["text"][["item", "n_answered", "n_blank", "n_distinct"]]
          .to_string(index=False))

    print("\nFigure written:")
    for path in res["figure"]:
        print(f"  {path}")
