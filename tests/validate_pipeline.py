"""
tests/validate_pipeline.py -- end-to-end validation of the POSRRI pipeline.

Runs every stage against the study data in ``data/raw`` and asserts the
properties the analysis must satisfy for its results to be trustworthy, then
prints a PASS/FAIL checklist and exits non-zero if anything failed.

The synthetic dataset is NOT used here. It exists only to smoke-test the code
and must never stand in for the study data in a validation run: a checklist
that passes on invented numbers says nothing about the analysis.

Checks
------
1.  Index structure       3 pillars, 10 domains, 50 indicators, clean partition
2.  Study data            the four input files present in data/raw and valid
3.  AHP consistency       every matrix reports a CR, and none is at or above
                          the 0.10 acceptance threshold
5.  AHP weights           local vectors and the 50 global weights sum to 1
6.  Expert review         every indicator carries every expert's rating
7.  Agreement rule        retention is exactly unanimity at the threshold, and
                          no percentage or kappa is reported
8.  Scoring               POSRRI and every domain and pillar score lie in 0-100
9.  Sensitivity           Spearman rho returned, in [-1, 1]
10. Figures               all six figures exist as PNG and PDF, non-empty
11. Tables                every result table exports, including the workbook

Usage
-----
    python tests/validate_pipeline.py            # full run on data/raw
    python tests/validate_pipeline.py --quick    # skip figure rendering
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from src import structure as st  # noqa: E402
from src import viz  # noqa: E402
from src.ahp import (consistency_ratio, equal_weights, expected_matrices,  # noqa: E402
                     priority_weights, run_ahp)
from src.delphi import run_delphi  # noqa: E402
from src.report import collect_tables, export_tables  # noqa: E402
from src.scoring import benchmark_matrix, score_index  # noqa: E402
from src.sensitivity import run_sensitivity  # noqa: E402

TOL = 1e-9


class Checklist:
    """Collect check outcomes and render a PASS/FAIL report."""

    def __init__(self) -> None:
        self.rows: List[Tuple[str, bool, str]] = []

    def check(self, name: str, fn: Callable[[], str]) -> bool:
        """Run ``fn``; it returns a detail string or raises AssertionError."""
        try:
            detail = fn() or ""
            self.rows.append((name, True, detail))
            return True
        except AssertionError as exc:
            self.rows.append((name, False, str(exc)))
            return False
        except Exception as exc:                              # noqa: BLE001
            self.rows.append((name, False, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
            return False

    @property
    def failed(self) -> int:
        return sum(1 for _, ok, _ in self.rows if not ok)

    def report(self) -> None:
        width = max(len(n) for n, _, _ in self.rows) + 2
        print("\n" + "=" * (width + 58))
        print("POSRRI PIPELINE VALIDATION CHECKLIST".center(width + 58))
        print("=" * (width + 58))
        for name, ok, detail in self.rows:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}]  {name:<{width}} {detail}")
        print("-" * (width + 58))
        total = len(self.rows)
        passed = total - self.failed
        verdict = "ALL CHECKS PASSED" if self.failed == 0 else f"{self.failed} CHECK(S) FAILED"
        print(f"  {passed}/{total} checks passed  --  {verdict}")
        print("=" * (width + 58))


def main(quick: bool = False, data_dir: "str | None" = None) -> int:
    cl = Checklist()

    print(f"Running the POSRRI pipeline against "
          f"{config.resolve_paths(data_dir=data_dir)['data_dir']}\n")

    # ---------------------------------------------------------------- 1 ----
    def _structure() -> str:
        ok, problems = st.validate_structure()
        assert ok, f"structure invalid: {problems}"
        assert len(st.PILLARS) == 3, "expected 3 pillars"
        assert len(st.DOMAINS) == 10, "expected 10 domains"
        assert len(st.INDICATORS) == 50, "expected 50 indicators"
        for dom in st.DOMAIN_ORDER:
            assert len(st.DOMAIN_INDICATORS[dom]) == 5, f"{dom} has != 5 indicators"
        return "3 pillars / 10 domains / 50 indicators"

    cl.check("Index structure is 3 x 10 x 50", _structure)

    # ---------------------------------------------------------------- 2 ----
    paths = config.resolve_paths(data_dir=data_dir)
    data_dir = paths["data_dir"]
    needed = [config.DATA_FILES[k]
              for k in ("delphi", "ahp", "scores", "benchmark")]

    absent = [f for f in needed if not (data_dir / f).exists()]
    frames = {} if absent else {f[:-4]: pd.read_csv(data_dir / f) for f in needed}

    def _data_present() -> str:
        assert not absent, (
            f"study data missing from {data_dir}: {absent}. "
            f"The pipeline reads the real study data from this directory; put "
            f"the CSVs there (schemas in data/templates/README.md). Nothing "
            f"below can be validated without them."
        )
        for name, frame in frames.items():
            assert len(frame) > 0, f"{name}.csv is empty"
        d = frames["delphi_ratings"]
        assert d["indicator_code"].nunique() == st.N_INDICATORS, (
            f"Delphi covers {d['indicator_code'].nunique()} indicators, "
            f"expected {st.N_INDICATORS}")
        assert len(frames["scores"]) == st.N_INDICATORS, (
            f"expected {st.N_INDICATORS} scored indicators, "
            f"got {len(frames['scores'])}")
        n_ports = frames["benchmark"]["port"].nunique()
        assert len(frames["benchmark"]) == st.N_DOMAINS * n_ports, (
            f"benchmark is not {st.N_DOMAINS} domains x {n_ports} ports")
        return (f"{len(d):,} Delphi rows ({d['expert_id'].nunique()} experts), "
                f"{len(frames['ahp_pairwise']):,} AHP rows, "
                f"{len(frames['scores'])} scores, "
                f"{len(frames['benchmark'])} benchmark rows")

    cl.check(f"Study data present in {data_dir.name}/", _data_present)

    if absent:
        cl.report()
        print(f"\nValidation stopped: no study data in {data_dir}.")
        print("Nothing downstream can run. Add the study CSVs and re-run.")
        return 1

    # ---------------------------------------------------------------- 4 ----
    ahp_res = run_ahp(frames["ahp_pairwise"])
    consistency = ahp_res["consistency"]

    def _cr_reported() -> str:
        expected = len(expected_matrices())
        n_experts = frames["ahp_pairwise"]["expert_id"].nunique()
        assert len(consistency) == expected * (n_experts + 1), (
            f"expected {expected * (n_experts + 1)} consistency rows "
            f"({expected} matrices x {n_experts} experts + AIJ), got {len(consistency)}"
        )
        assert consistency["CR"].notna().all(), "some matrices report no CR"
        assert (consistency["CR"] >= 0).all(), "negative consistency ratio"
        assert consistency.groupby(["level", "group"]).ngroups == expected, (
            "not every matrix in the hierarchy was evaluated")
        # lambda_max >= n is required for a positive reciprocal matrix.
        assert (consistency["lambda_max"] >= consistency["n_items"] - 1e-8).all(), (
            "lambda_max below n, which is impossible for a reciprocal matrix")
        return f"{len(consistency)} matrices evaluated, all report a CR"

    cl.check("Every AHP matrix reports a consistency ratio", _cr_reported)

    def _cr_threshold() -> str:
        bad = consistency[consistency["CR"] >= config.CR_THRESHOLD]
        assert len(bad) == 0, (
            f"{len(bad)} matrices at or above CR {config.CR_THRESHOLD}: "
            f"{bad[['level', 'group', 'expert_id', 'CR']].head().to_dict('records')}"
        )
        return (f"max CR = {consistency['CR'].max():.4f} "
                f"(threshold {config.CR_THRESHOLD})")

    cl.check("All AHP matrices have CR < 0.10", _cr_threshold)

    # ---------------------------------------------------------------- 5 ----
    weights = ahp_res["weights"]

    def _weights_sum() -> str:
        total = float(weights["global_weight"].sum())
        assert abs(total - 1.0) < 1e-9, f"global weights sum to {total!r}, not 1.0"
        assert len(weights) == 50, f"expected 50 weight rows, got {len(weights)}"
        assert (weights["global_weight"] > 0).all(), "non-positive global weight"
        for key, vec in ahp_res["local_weights"].items():
            s = float(vec.sum())
            assert abs(s - 1.0) < 1e-9, f"local weights for {key} sum to {s}"
        pillar_sum = float(ahp_res["pillar_weights"]["pillar_weight"].sum())
        assert abs(pillar_sum - 1.0) < 1e-9, f"pillar weights sum to {pillar_sum}"
        dom_sum = float(ahp_res["domain_weights"]["domain_weight_global"].sum())
        assert abs(dom_sum - 1.0) < 1e-9, f"global domain weights sum to {dom_sum}"
        return f"sum = {total:.12f} over 50 indicators; all local vectors sum to 1"

    cl.check("Global weights sum to 1.0", _weights_sum)

    def _weight_method() -> str:
        # Spot-check the row geometric mean against a hand-computed case.
        A = np.array([[1.0, 3.0, 5.0], [1 / 3, 1.0, 2.0], [1 / 5, 0.5, 1.0]])
        w = priority_weights(A)
        gm = np.exp(np.mean(np.log(A), axis=1))
        expected = gm / gm.sum()
        assert np.allclose(w, expected, atol=1e-12), "row geometric mean mismatch"
        assert abs(w.sum() - 1.0) < TOL, "priority vector not normalised"
        # A perfectly consistent matrix must return CR = 0.
        v = np.array([0.5, 0.3, 0.2])
        C = np.outer(v, 1.0 / v)
        assert abs(consistency_ratio(C)["CR"]) < 1e-9, (
            "a perfectly consistent matrix did not return CR = 0")
        return "row geometric mean and CR = 0 on a consistent matrix verified"

    cl.check("AHP weight and CR arithmetic verified", _weight_method)

    # ---------------------------------------------------------------- 6 ----
    delphi_res = run_delphi(frames["delphi_ratings"])
    per_indicator = delphi_res["per_indicator"]
    delphi_summary = delphi_res["summary"]

    def _delphi_coverage() -> str:
        assert len(per_indicator) == st.N_INDICATORS, (
            f"expected {st.N_INDICATORS} indicator rows, got {len(per_indicator)}")
        assert set(per_indicator["indicator_code"]) == set(st.INDICATORS), (
            "the Delphi table does not cover exactly the indicator set")
        ratings = delphi_res["ratings"]
        expert_cols = [c for c in ratings.columns if c != "indicator_code"]
        assert expert_cols, "no expert columns in the ratings matrix"
        assert len(expert_cols) == delphi_summary["n_experts"], (
            "expert count disagrees between the ratings matrix and the summary")
        assert per_indicator["n_experts"].eq(len(expert_cols)).all(), (
            "some indicator was not rated by every expert")
        lo, hi = config.RELEVANCE_SCALE
        for col in ("min_relevance", "median_relevance", "max_relevance"):
            assert per_indicator[col].between(lo, hi).all(), (
                f"{col} outside the {lo}-{hi} rating scale")
        assert (per_indicator["min_relevance"]
                <= per_indicator["max_relevance"]).all(), "min exceeds max"
        return (f"{len(per_indicator)} indicators x "
                f"{len(expert_cols)} experts, every rating present")

    cl.check("Per-indicator ratings reported for every indicator", _delphi_coverage)

    def _agreement_rule() -> str:
        """Retention must be exactly unanimity at or above the threshold."""
        threshold = config.RELEVANCE_HIGH_MIN
        ratings = delphi_res["ratings"].set_index("indicator_code")
        expected = (ratings >= threshold).all(axis=1)
        got = per_indicator.set_index("indicator_code")["all_experts_agree"]
        mismatch = expected.index[expected.ne(got.reindex(expected.index))]
        assert not len(mismatch), (
            f"all_experts_agree disagrees with the rule for "
            f"{list(mismatch)[:5]}")
        assert len(delphi_res["retained"]) == int(expected.sum()), (
            "the retained list does not match the agreement flag")
        assert (len(delphi_res["retained"]) + len(delphi_res["to_discuss"])
                == len(per_indicator)), (
            "retained + to_discuss does not account for every indicator")
        # A single sub-threshold rating must break unanimity.
        n_dissent = per_indicator["n_below_threshold"]
        assert per_indicator["all_experts_agree"].eq(n_dissent.eq(0)).all(), (
            "an indicator with a dissenting rating was still marked unanimous")

        # Percentages and chance-corrected agreement must be gone.
        banned = ("cohens_kappa", "fleiss", "scvi", "icvi", "pct_rating",
                  "consensus")
        present = [k for k in delphi_summary
                   if any(b in k.lower() for b in banned)]
        assert not present, f"consensus/kappa output still reported: {present}"
        cols = [c for c in per_indicator.columns
                if any(b in c.lower() for b in banned)]
        assert not cols, f"consensus/kappa column(s) still present: {cols}"
        return (f"unanimity at >= {threshold}: "
                f"{delphi_summary['n_retained']} retained, "
                f"{delphi_summary['n_to_discuss']} for discussion")

    cl.check("All-experts-agree rule applied, no percentages or kappa",
             _agreement_rule)

    # ---------------------------------------------------------------- 8 ----
    score_res = score_index(frames["scores"], weights)
    overall = score_res["overall"]

    def _score_ranges() -> str:
        posrri = overall["POSRRI_0_100"]
        assert np.isfinite(posrri), "POSRRI is not finite"
        assert 0.0 <= posrri <= 100.0, f"POSRRI {posrri} outside 0-100"
        doms = score_res["domains"]
        assert len(doms) == 10, f"expected 10 domain rows, got {len(doms)}"
        assert doms["score_0_100"].notna().all(), "missing domain score"
        assert doms["score_0_100"].between(0.0, 100.0).all(), (
            f"domain score outside 0-100: "
            f"{doms.loc[~doms['score_0_100'].between(0, 100), 'domain_code'].tolist()}")
        pil = score_res["pillars"]
        assert len(pil) == 3, f"expected 3 pillar rows, got {len(pil)}"
        assert pil["score_0_100"].between(0.0, 100.0).all(), "pillar score outside 0-100"
        inds = score_res["indicators"]
        assert len(inds) == 50, "expected 50 indicator rows"
        assert inds["score_0_100"].between(0.0, 100.0).all(), "indicator score outside 0-100"
        return (f"POSRRI = {posrri:.2f} ({overall['band']}); domains "
                f"{doms['score_0_100'].min():.1f}-{doms['score_0_100'].max():.1f}")

    cl.check("POSRRI and all domain scores within 0-100", _score_ranges)

    def _score_arithmetic() -> str:
        # The overall index must equal 100 * sum(w * s) / 3 when weights sum to 1.
        inds = score_res["indicators"]
        manual = 100.0 * float((inds["weight"] * inds["final_score"]).sum()) / 3.0
        assert abs(manual - overall["POSRRI_0_100"]) < 1e-9, (
            f"POSRRI {overall['POSRRI_0_100']} != manual {manual}")
        # A domain in which every indicator scores 3 must score exactly 100.
        perfect = frames["scores"].copy()
        perfect["final_score"] = 3
        top = score_index(perfect, weights)
        assert abs(top["overall"]["POSRRI_0_100"] - 100.0) < 1e-9, (
            "all-3 scores did not produce POSRRI = 100")
        zero = frames["scores"].copy()
        zero["final_score"] = 0
        bottom = score_index(zero, weights)
        assert abs(bottom["overall"]["POSRRI_0_100"]) < 1e-9, (
            "all-0 scores did not produce POSRRI = 0")
        return "0-3 rubric maps to 0-100 exactly at both endpoints"

    cl.check("Score normalisation arithmetic verified", _score_arithmetic)

    # ---------------------------------------------------------------- 9 ----
    sens_res = run_sensitivity(frames["scores"], weights)

    def _sensitivity() -> str:
        rho = sens_res["spearman"]["rho"]
        assert rho is not None and np.isfinite(rho), "Spearman rho not returned"
        assert -1.0 - TOL <= rho <= 1.0 + TOL, f"rho {rho} outside [-1, 1]"
        doms = sens_res["domains"]
        assert len(doms) == 10, "sensitivity table is not 10 rows"
        assert doms["score_ahp"].between(0, 100).all(), "AHP score outside 0-100"
        assert doms["score_equal"].between(0, 100).all(), "equal-weight score outside 0-100"
        eq = equal_weights()
        assert abs(eq["global_weight"].sum() - 1.0) < TOL, "equal weights do not sum to 1"
        assert eq["global_weight"].nunique() == 1, "equal weights are not equal"
        return (f"rho = {rho:.4f} (p = {sens_res['spearman']['p_value']:.3g}); "
                f"POSRRI {sens_res['summary']['POSRRI_ahp']:.2f} vs "
                f"{sens_res['summary']['POSRRI_equal']:.2f}")

    cl.check("Sensitivity: Spearman rho computed", _sensitivity)

    # --------------------------------------------------------------- 10 ----
    figure_dir = paths["figure_dir"]
    # Built the same way in both modes: --quick skips *rendering*, it must not
    # change any exported table.
    bench = benchmark_matrix(
        frames["benchmark"],
        cpa_domain_scores=score_res["domains"].set_index("domain_code")["score_0_100"])
    if not quick:
        viz.make_all_figures(score_res, delphi_res, sens_res, frames["benchmark"],
                             ahp_domain_weights=ahp_res["domain_weights"],
                             figure_dir=figure_dir)

    def _figures() -> str:
        missing, empty = [], []
        for stem in viz.FIGURE_STEMS:
            for ext in config.FIGURE_FORMATS:
                path = figure_dir / f"{stem}.{ext}"
                if not path.exists():
                    missing.append(path.name)
                elif path.stat().st_size < 1024:
                    empty.append(path.name)
        assert not missing, f"missing figure file(s): {missing}"
        assert not empty, f"suspiciously small figure file(s): {empty}"
        assert len(viz.FIGURE_STEMS) == 6, "expected 6 figures"
        return f"6 figures x {len(config.FIGURE_FORMATS)} formats in {figure_dir}"

    cl.check("All 6 figures exist in outputs/figures", _figures)

    def _rubric_wording() -> str:
        """The 0-3 rubric is locked; figures and docs must not drift from it."""
        expected = {0: "absent", 1: "partial or ambiguous",
                    2: "defined", 3: "defined and verified"}
        assert config.SCORE_RUBRIC == expected, (
            f"config.SCORE_RUBRIC has drifted: {config.SCORE_RUBRIC}")
        for level, text in expected.items():
            assert viz.SCORE_LABELS[level] == f"{level}  {text}", (
                f"figure colourbar label for {level} is "
                f"{viz.SCORE_LABELS[level]!r}, expected {level}  {text!r}")
        # Superseded wording must not survive anywhere in the source or docs.
        banned = ("substantial", "fully met")
        offenders = []
        for path in list(_ROOT.glob("*.py")) + list(_ROOT.glob("*.md")) + \
                list(_ROOT.glob("src/*.py")) + list(_ROOT.glob("tests/*.py")) + \
                list(_ROOT.glob("synthetic/*.py")) + list(_ROOT.glob("notebooks/*.ipynb")) + \
                list(_ROOT.glob("data/templates/*.md")):
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            # This check names the banned strings itself; skip its own source.
            if path.name == "validate_pipeline.py":
                continue
            for word in banned:
                if word in text:
                    offenders.append(f"{path.relative_to(_ROOT)}:{word}")
        assert not offenders, f"superseded rubric wording still present: {offenders}"
        return "0/1/2/3 = absent / partial or ambiguous / defined / defined and verified"

    cl.check("Scoring rubric wording is locked and consistent", _rubric_wording)

    # --------------------------------------------------------------- 11 ----
    def _tables() -> str:
        tables = collect_tables(delphi_res, ahp_res, score_res, sens_res,
                                benchmark_0_100=bench)
        written = export_tables(tables, table_dir=paths["table_dir"])
        for key, path in written.items():
            assert path.exists(), f"table {key} was not written"
            assert path.stat().st_size > 0, f"table {key} is empty"
        workbook = written["__workbook__"]
        assert workbook.suffix == ".xlsx", "combined workbook is not .xlsx"
        return f"{len(tables)} tables + {workbook.name} in {paths['table_dir']}"

    cl.check("Result tables and Excel workbook export", _tables)

    # ------------------------------------------------------------ report ----
    cl.report()

    if cl.failed == 0:
        print("\nHeadline results")
        print("-" * 72)
        print(f"  POSRRI                    {overall['POSRRI_0_100']:.2f} / 100  "
              f"({overall['band']})")
        print(f"  Strongest domain          {overall['strongest_domain']}  "
              f"{st.DOMAINS[overall['strongest_domain']]}")
        print(f"  Weakest domain            {overall['weakest_domain']}  "
              f"{st.DOMAINS[overall['weakest_domain']]}")
        print(f"  Expert review             "
              f"{delphi_res['summary']['n_retained']}/"
              f"{delphi_res['summary']['n_indicators']} indicators unanimous "
              f"({delphi_res['summary']['n_experts']} experts)")
        print(f"  Max AHP consistency ratio {consistency['CR'].max():.4f}")
        print(f"  Spearman rho (AHP/equal)  {sens_res['spearman']['rho']:.3f}")

    return 0 if cl.failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the POSRRI pipeline.")
    parser.add_argument("--quick", action="store_true",
                        help="skip figure rendering (figures must already exist)")
    parser.add_argument("--data-dir", default=None,
                        help="validate a different directory (default data/raw). "
                             "Pointing this at synthetic/data is a smoke test of "
                             "the code, never a validation of the study.")
    args = parser.parse_args()
    sys.exit(main(quick=args.quick, data_dir=args.data_dir))
