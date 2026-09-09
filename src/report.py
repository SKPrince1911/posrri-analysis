"""
src/report.py -- assemble and export the POSRRI result tables.

The notebook and the validation script both need the same set of output
tables written to disk in the same shape, so the logic lives here once rather
than being duplicated. Two entry points:

``collect_tables``  gathers every result frame into an ordered mapping, with
                    short keys that double as CSV stems and Excel sheet names;
``export_tables``   writes each frame to CSV and all of them to a single
                    multi-sheet Excel workbook.

Excel sheet names are capped at 31 characters by the file format, so the keys
are kept short deliberately.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Mapping

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from src import structure as st  # noqa: E402

EXCEL_FILENAME = "posrri_results.xlsx"


def summary_frame(delphi_result: dict,
                  ahp_result: dict,
                  scoring_result: dict,
                  sensitivity_result: dict) -> pd.DataFrame:
    """Headline numbers as a tidy metric/value/notes table."""
    d = delphi_result["summary"]
    a = ahp_result["summary"]
    o = scoring_result["overall"]
    s = sensitivity_result["summary"]
    last = d["rounds"][-1]

    rows = [
        ("Index structure", f"{st.N_PILLARS} pillars / {st.N_DOMAINS} domains / "
                            f"{st.N_INDICATORS} indicators", ""),
        ("Delphi experts", d["n_experts"], f"rounds {d['rounds']}"),
        ("Indicators retained (I-CVI >= "
         f"{config.ICVI_RETAIN_THRESHOLD})", d[f"n_retained_r{last}"],
         f"of {d['n_indicators']} in round {last}"),
        ("Indicators meeting consensus", d[f"n_consensus_r{last}"],
         ">=80% rating 7-9, or median >=7 with IQR <=2"),
        (f"S-CVI/Ave (round {last})", round(d[f"scvi_ave_r{last}"], 4),
         f"target >= {config.SCVI_TARGET}; "
         f"{'met' if d['scvi_target_met'] else 'not met'}"),
        (f"S-CVI/UA (round {last})", round(d[f"scvi_ua_r{last}"], 4),
         "proportion of items with universal agreement"),
        ("Cohen's kappa, round 1 vs 2", round(d["cohens_kappa"], 4),
         f"{d['n_decisions_changed']} decisions changed"),
        ("AHP experts", a["n_experts"], f"{a['n_matrices']} matrices each"),
        ("AHP matrices with CR >= 0.10", a["n_inconsistent"],
         f"of {a['n_consistency_rows']} checked; max CR {a['max_CR']:.4f}"),
        ("Global weights sum", round(a["global_weight_sum"], 10),
         "across all 50 indicators"),
        ("POSRRI (AHP weights)", round(o["POSRRI_0_100"], 2), o["band"]),
        ("POSRRI (equal weights)", round(s["POSRRI_equal"], 2),
         sensitivity_result["overall"]["band_equal"]),
        ("Strongest domain", o["strongest_domain"], st.DOMAINS[o["strongest_domain"]]),
        ("Weakest domain", o["weakest_domain"], st.DOMAINS[o["weakest_domain"]]),
        ("Mean implementation gap", round(o["mean_implementation_gap"], 3),
         "documentary minus field score, 0-3 points"),
        ("Spearman rho (AHP vs equal)", round(s["spearman_rho"], 4),
         f"p = {s['spearman_p']:.3g}; {s['n_rank_changes']} domains change rank"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "notes"])


def collect_tables(delphi_result: dict,
                   ahp_result: dict,
                   scoring_result: dict,
                   sensitivity_result: dict,
                   benchmark_0_100: "pd.DataFrame | None" = None
                   ) -> "OrderedDict[str, pd.DataFrame]":
    """Gather every result frame in reporting order."""
    tables: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    tables["00_summary"] = summary_frame(
        delphi_result, ahp_result, scoring_result, sensitivity_result)
    tables["01_index_structure"] = st.structure_frame()
    tables["02_delphi_per_indicator"] = delphi_result["per_indicator"]
    tables["03_delphi_by_round"] = delphi_result["long"]
    tables["04_delphi_retained"] = delphi_result["retained"]
    tables["05_delphi_dropped"] = delphi_result["dropped"]
    tables["06_ahp_consistency"] = ahp_result["consistency"]
    tables["07_ahp_pillar_weights"] = ahp_result["pillar_weights"]
    tables["08_ahp_domain_weights"] = ahp_result["domain_weights"]
    tables["09_ahp_indicator_weights"] = ahp_result["weights"]
    tables["10_scores_indicators"] = scoring_result["indicators"]
    tables["11_scores_domains"] = scoring_result["domains"]
    tables["12_scores_pillars"] = scoring_result["pillars"]
    tables["13_sensitivity_domains"] = sensitivity_result["domains"]
    tables["14_sensitivity_pillars"] = sensitivity_result["pillars"]
    if benchmark_0_100 is not None:
        tables["15_benchmark_0_100"] = benchmark_0_100.reset_index()
    return tables


def export_tables(tables: Mapping[str, pd.DataFrame],
                  table_dir: "Path | str" = config.TABLE_DIR,
                  excel_name: str = EXCEL_FILENAME,
                  verbose: bool = False) -> Dict[str, Path]:
    """Write every table to CSV and all of them to one Excel workbook.

    Returns ``{key: csv_path}`` plus ``{"__workbook__": xlsx_path}``.
    """
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}
    for key, frame in tables.items():
        path = table_dir / f"{key}.csv"
        frame.to_csv(path, index=False)
        written[key] = path
        if verbose:
            print(f"  {path.name:<34} {len(frame):>4} rows")

    workbook = table_dir / excel_name
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for key, frame in tables.items():
            frame.to_excel(writer, sheet_name=key[:31], index=False)
    written["__workbook__"] = workbook
    if verbose:
        print(f"  {workbook.name:<34} {len(tables)} sheets")
    return written
