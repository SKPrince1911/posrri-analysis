"""
tests/validate_pipeline.py -- end-to-end validation of the POSRRI pipeline.

Runs every stage against the synthetic dataset and asserts the properties the
analysis must satisfy for its results to be trustworthy, then prints a
PASS/FAIL checklist and exits non-zero if anything failed.

Checks
------
1.  Index structure       3 pillars, 10 domains, 50 indicators, clean partition
2.  Synthetic data        all five files present and schema-valid
3.  Determinism           regenerating from the master seed reproduces the data
4.  AHP consistency       every matrix reports a CR; consistent synthetic
                          matrices are below 0.10
5.  AHP weights           local vectors and the 50 global weights sum to 1
6.  Delphi                consensus flag and I-CVI computed for all 50
                          indicators, in every round
7.  Cohen's kappa         returns a finite value with a valid cross-tabulation
8.  Scoring               POSRRI and every domain and pillar score lie in 0-100
9.  Sensitivity           Spearman rho returned, in [-1, 1]
10. Figures               all six figures exist as PNG and PDF, non-empty
11. Tables                every result table exports, including the workbook

Usage
-----
    python tests/validate_pipeline.py            # full run
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
from src.delphi import cohens_kappa, run_delphi  # noqa: E402
from src.report import collect_tables, export_tables  # noqa: E402
from src.scoring import benchmark_matrix, score_index  # noqa: E402
from src.survey_analysis import cronbach_alpha, run_survey_analysis  # noqa: E402
from src.survey_ingest import (AGREEMENT, ALLOWED_VALUES,  # noqa: E402
                               EXPERIENCE_BANDS, ITEMS, LIKERT_ITEMS,
                               ORG_CATEGORIES, ORG_GROUPS, ORG_OTHER,
                               OUTPUT_COLUMNS, RECODE, RESPONDENT_ID_PREFIX,
                               CONSENT_YES, LIKERT_LABELS, TEXT_ITEMS,
                               _coerce_likert, _norm, load_survey_export,
                               organisation_group, read_survey)
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


def main(quick: bool = False) -> int:
    cl = Checklist()

    print("Running the POSRRI pipeline against the synthetic dataset...\n")

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
    paths = config.resolve_paths(use_synthetic=True)
    data_dir = paths["data_dir"]
    needed = ["delphi_ratings.csv", "ahp_pairwise.csv", "scores.csv",
              "survey.csv", "benchmark.csv", "survey_export.csv"]

    if any(not (data_dir / f).exists() for f in needed):
        print("Synthetic data missing; generating it...")
        from synthetic.generate_synthetic import generate_all
        generate_all(seed=config.RANDOM_SEED, outdir=data_dir, verbose=False)

    frames = {f[:-4]: pd.read_csv(data_dir / f) for f in needed}

    def _data_present() -> str:
        for name, frame in frames.items():
            assert len(frame) > 0, f"{name}.csv is empty"
        d = frames["delphi_ratings"]
        assert d["indicator_code"].nunique() == 50, "Delphi does not cover 50 indicators"
        assert d["expert_id"].nunique() == 18, "expected 18 Delphi experts"
        assert sorted(d["round"].unique()) == [1, 2], "expected 2 Delphi rounds"
        assert frames["ahp_pairwise"]["expert_id"].nunique() == 8, "expected 8 AHP experts"
        assert len(frames["scores"]) == 50, "expected 50 scored indicators"
        assert len(frames["survey"]) == 50, "expected 50 survey respondents"
        assert len(frames["survey_export"]) == 53, (
            "expected 53 raw survey rows (50 consenting + 3 to be dropped)")
        assert len(frames["benchmark"]) == 40, "expected 10 domains x 4 ports"
        return (f"{len(d):,} Delphi + {len(frames['ahp_pairwise']):,} AHP rows, "
                f"50 scores, 50 survey, 40 benchmark")

    cl.check("Synthetic data present and complete", _data_present)

    # ---------------------------------------------------------------- 3 ----
    def _determinism() -> str:
        from synthetic.generate_synthetic import generate_all
        tmp = Path(tempfile.mkdtemp(prefix="posrri_seed_"))
        try:
            regenerated = generate_all(seed=config.RANDOM_SEED, outdir=tmp, verbose=False)
            for name, frame in regenerated.items():
                original = pd.read_csv(data_dir / f"{name}.csv")
                fresh = pd.read_csv(tmp / f"{name}.csv")
                assert original.shape == fresh.shape, f"{name}: shape changed"
                pd.testing.assert_frame_equal(original, fresh, check_dtype=False,
                                              obj=name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return (f"seed {config.RANDOM_SEED} reproduces all "
                f"{len(regenerated)} files exactly")

    cl.check("Synthetic generation is deterministic", _determinism)

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

    cl.check("Consistent synthetic matrices have CR < 0.10", _cr_threshold)

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
    long = delphi_res["long"]
    per_ind = delphi_res["per_indicator"]

    def _delphi_coverage() -> str:
        rounds = sorted(long["round"].unique())
        assert len(per_ind) == 50, f"expected 50 indicator rows, got {len(per_ind)}"
        assert len(long) == 50 * len(rounds), (
            f"expected {50 * len(rounds)} indicator-round rows, got {len(long)}")
        assert set(long["indicator_code"]) == set(st.INDICATORS), (
            "Delphi output does not cover exactly the 50 indicators")
        assert long["consensus"].notna().all(), "missing consensus flag"
        assert long["consensus"].map(lambda v: isinstance(v, (bool, np.bool_))).all(), (
            "consensus flag is not boolean")
        assert long["I_CVI"].notna().all(), "missing I-CVI"
        assert ((long["I_CVI"] >= 0) & (long["I_CVI"] <= 1)).all(), (
            "I-CVI outside [0, 1]")
        assert long["median_relevance"].between(1, 9).all(), "median outside 1-9"
        assert (long["iqr_relevance"] >= 0).all(), "negative IQR"
        return (f"50 indicators x {len(rounds)} rounds; "
                f"S-CVI/Ave r{rounds[-1]} = "
                f"{delphi_res['summary'][f'scvi_ave_r{rounds[-1]}']:.3f}")

    cl.check("Delphi consensus flags and I-CVI for all 50", _delphi_coverage)

    def _consensus_rule() -> str:
        # The flag must be exactly the disjunction defined in the protocol.
        pct = long["I_CVI"] >= config.CONSENSUS_PCT_THRESHOLD
        central = ((long["median_relevance"] >= config.CONSENSUS_MEDIAN_MIN)
                   & (long["iqr_relevance"] <= config.CONSENSUS_IQR_MAX))
        expected = (pct | central).to_numpy()
        assert np.array_equal(long["consensus"].to_numpy(), expected), (
            "consensus flag does not match (>=80% in 7-9) OR (median>=7 AND IQR<=2)")
        retain = (long["I_CVI"] >= config.ICVI_RETAIN_THRESHOLD).to_numpy()
        assert np.array_equal(long["retain"].to_numpy(), retain), (
            f"retention does not match I-CVI >= {config.ICVI_RETAIN_THRESHOLD}")
        return "flag matches the protocol disjunction on all rows"

    cl.check("Consensus and retention rules implemented as specified", _consensus_rule)

    # ---------------------------------------------------------------- 7 ----
    def _kappa() -> str:
        k = delphi_res["kappa"]
        value = k["kappa"]
        assert value is not None, "Cohen's kappa returned None"
        assert isinstance(value, float), f"kappa is {type(value).__name__}, not float"
        assert np.isfinite(value), "Cohen's kappa is not finite"
        assert -1.0 - TOL <= value <= 1.0 + TOL, f"kappa {value} outside [-1, 1]"
        assert int(k["table"].to_numpy().sum()) == 50, (
            "kappa cross-tabulation does not cover all 50 indicators")
        # Identical labellings must give kappa = 1.
        labels = ["A", "B", "A", "B", "A", "A", "B", "B"]
        assert abs(cohens_kappa(labels, labels)["kappa"] - 1.0) < TOL, (
            "kappa on identical labellings is not 1.0")
        # A degenerate single-category rater must return nan, not raise.
        degenerate = cohens_kappa(["A"] * 6, ["A"] * 6)["kappa"]
        assert np.isnan(degenerate), "degenerate kappa should be nan"
        return (f"kappa = {value:.4f} (po = {k['po']:.3f}, pe = {k['pe']:.3f}, "
                f"{k['n_changed']} decisions changed)")

    cl.check("Cohen's kappa returns a value", _kappa)

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

    # ---------------------------------------------------- survey ingest ----
    # Loaded inside the first check rather than at module level, so a
    # malformed survey.csv is reported as a FAIL on the checklist instead of
    # aborting the whole run with a traceback. Later checks read from `loaded`
    # and fail cleanly if the load did not succeed.
    export_path = data_dir / "survey_export.csv"
    loaded: dict = {}

    def _survey_schema() -> str:
        ingested = load_survey_export(export_path, write=False, verbose=False)
        survey = read_survey(data_dir / "survey.csv")
        loaded["ingested"], loaded["survey"] = ingested, survey

        assert list(ingested.columns) == OUTPUT_COLUMNS, (
            f"ingest produced columns {list(ingested.columns)}, "
            f"expected {OUTPUT_COLUMNS}")
        assert list(survey.columns) == OUTPUT_COLUMNS, (
            f"survey.csv columns {list(survey.columns)}, "
            f"expected {OUTPUT_COLUMNS}")
        assert len(ITEMS) == 15, f"expected 15 source items, got {len(ITEMS)}"
        # q12 is the one item that expands into two output columns.
        assert "q12" not in OUTPUT_COLUMNS, "q12 should expand to q12_raw/q12_group"
        for column in ("q12_raw", "q12_group"):
            assert column in OUTPUT_COLUMNS, f"{column} missing from the schema"

        # The committed survey.csv must be exactly what the ingest produces:
        # if they diverge, the demonstration file was written by something
        # other than src/survey_ingest.py.
        pd.testing.assert_frame_equal(
            ingested.reset_index(drop=True),
            survey.reset_index(drop=True),
            check_dtype=False, obj="survey.csv vs ingest output")

        raw = pd.read_csv(export_path, dtype=str)
        n_dropped = len(raw) - len(ingested)
        assert n_dropped > 0, (
            "the synthetic export contains no non-consenting rows, so the "
            "consent filter is never exercised")
        return (f"respondent_id + q1..q15; {len(ingested)} rows from {len(raw)} "
                f"raw ({n_dropped} dropped on consent)")

    cl.check("Survey ingest produces the exact schema", _survey_schema)

    def _survey_ids() -> str:
        assert loaded, "survey data failed to load (see the schema check above)"
        # Pin the value, not just the format: deriving everything from the
        # constant would let a silent change to it pass unnoticed.
        assert RESPONDENT_ID_PREFIX == "SRV", (
            f"respondent id prefix is {RESPONDENT_ID_PREFIX!r}, expected 'SRV' "
            f"('AGT' is reserved for shipping-agent interviews)")
        # Both the ingest output AND the committed file: checking only the
        # freshly ingested frame would be vacuous, since the ingest generates
        # the ids itself and cannot produce a duplicate.
        for source, frame in (("ingest output", loaded["ingested"]),
                              ("survey.csv", loaded["survey"])):
            ids = frame["respondent_id"]
            dupes = ids[ids.duplicated()].tolist()
            assert not dupes, f"{source}: duplicate respondent_id {dupes[:5]}"
            assert ids.notna().all(), f"{source}: missing respondent_id"
            # Checked before the exact-sequence assertion below, which would
            # otherwise fire first and report a generic mismatch: a collision
            # with the reserved interview prefix deserves to say so.
            reserved = [i for i in ids if str(i).upper().startswith("AGT-")]
            assert not reserved, (
                f"{source}: {len(reserved)} respondent_id(s) use the reserved "
                f"AGT- shipping-agent interview prefix, e.g. {reserved[:3]}; "
                f"survey respondents must use {RESPONDENT_ID_PREFIX}-")
            expected = [f"{RESPONDENT_ID_PREFIX}-{k:03d}"
                        for k in range(1, len(ids) + 1)]
            assert ids.tolist() == expected, (
                f"{source}: respondent_id is not {RESPONDENT_ID_PREFIX}-001.."
                f"{RESPONDENT_ID_PREFIX}-{len(ids):03d} in order; first "
                f"mismatch at index "
                f"{next(i for i, (a, b) in enumerate(zip(ids, expected)) if a != b)}")
        ids = loaded["ingested"]["respondent_id"]

        # IDs must follow ascending timestamp order, not file order.
        # Reuse the ingest's own consent vocabulary and normaliser rather than
        # a simplified copy: a hardcoded "yes" here silently stopped matching
        # when the form's consent option became "Yes, I consent".
        raw = pd.read_csv(export_path, dtype=str)
        stamps = pd.to_datetime(raw.iloc[:, 0], errors="coerce", format="mixed")
        yes_keys = {_norm(v) for v in CONSENT_YES}
        consented = raw.iloc[:, 1].map(_norm).isin(yes_keys)
        assert consented.any(), (
            f"no row in the export matches a consent value from "
            f"{sorted(CONSENT_YES)!r}")
        kept = stamps[consented].sort_values()
        assert kept.is_monotonic_increasing and len(kept) == len(ids), (
            "respondent_id ordering does not follow ascending timestamp")
        assert not stamps.is_monotonic_increasing, (
            "the synthetic export is already timestamp-sorted, so the sort is "
            "never exercised")
        return (f"{len(ids)} unique ids, {RESPONDENT_ID_PREFIX}-001.."
                f"{RESPONDENT_ID_PREFIX}-{len(ids):03d}, timestamp-ordered, "
                f"no AGT- collision")

    cl.check("Survey respondent_ids are unique and ordered", _survey_ids)

    def _experience_bands() -> str:
        """The rebuilt bands, their no-"years" aliases, and old bands refused."""
        for label, code in (("Less than 5 years", 1), ("5 to 10 years", 2),
                            ("11 to 20 years", 3), ("More than 20 years", 4)):
            assert EXPERIENCE_BANDS.get(label) == code, (
                f"q13 band {label!r} maps to "
                f"{EXPERIENCE_BANDS.get(label)!r}, expected {code}")
            alias = label.replace(" years", "")
            assert EXPERIENCE_BANDS.get(alias) == code, (
                f"q13 alias {alias!r} missing or wrong")
        # The pre-rebuild bands cut at 2/5/10 rather than 5/10/20, so silently
        # accepting them would merge two different measurements.
        for old_band in ("Less than 2 years", "2-5 years", "6-10 years",
                         "More than 10 years"):
            assert old_band not in EXPERIENCE_BANDS, (
                f"{old_band!r} is a pre-rebuild band with different cut points "
                f"and must not be silently accepted")
        return "4 bands + no-'years' aliases; pre-rebuild bands refused"

    cl.check("q13 experience bands match the rebuilt form", _experience_bands)

    def _survey_codes() -> str:
        assert loaded, "survey data failed to load (see the schema check above)"
        survey = loaded["survey"]
        for item, allowed in ALLOWED_VALUES.items():
            values = survey[item].dropna()
            assert values.map(lambda v: float(v).is_integer()).all(), (
                f"{item}: non-integer coded value present")
            present = set(values.astype(int).tolist())
            unexpected = present - allowed
            assert not unexpected, (
                f"{item}: value(s) {sorted(unexpected)} outside allowed "
                f"{sorted(allowed)}")
        for item in LIKERT_ITEMS:
            values = survey[item].dropna().astype(int)
            assert values.between(1, 5).all(), (
                f"{item}: Likert value outside 1-5")
        # Items with no mapping and no Likert role must remain free text.
        for item in TEXT_ITEMS:
            assert item not in RECODE and item not in LIKERT_ITEMS, (
                f"{item} is listed as free text but is also coded")
            assert str(survey[item].dtype) in ("string", "object"), (
                f"{item} should be text, got dtype {survey[item].dtype}")
        coded = [i for i in ITEMS if i not in TEXT_ITEMS]
        return (f"{len(coded)} coded items within their allowed sets; "
                f"{len(TEXT_ITEMS)} text items left as text")

    cl.check("Survey coded values within allowed sets", _survey_codes)

    def _likert_formats() -> str:
        """The loader must accept agreement labels and bare numbers alike."""
        assert AGREEMENT == {"Strongly disagree": 1, "Disagree": 2,
                             "Neutral": 3, "Agree": 4, "Strongly agree": 5,
                             "Neither agree nor disagree": 3}, (
            f"the agreement mapping has drifted: {AGREEMENT}")
        # The rebuilt form's midpoint, and the pre-rebuild alias, are one point.
        for spelling in ("Neutral", "  NEUTRAL ",
                         "Neither agree nor disagree", "neither agree nor disagree"):
            got = _coerce_likert(pd.Series([spelling]), "q1").tolist()
            assert got == [3], f"{spelling!r} coded as {got}, expected [3]"
        assert LIKERT_LABELS[3] == "Neutral", (
            f"the midpoint displays as {LIKERT_LABELS[3]!r}, expected 'Neutral'")

        labels = [l for l in AGREEMENT if l != "Neither agree nor disagree"]
        # Exact labels, and the same labels with case and padding mangled.
        exact = _coerce_likert(pd.Series(labels), "q1")
        assert exact.tolist() == [1, 2, 3, 4, 5], (
            f"exact agreement labels coded as {exact.tolist()}")
        messy = _coerce_likert(
            pd.Series([f"  {t.upper()} " for t in labels]), "q1")
        assert messy.tolist() == [1, 2, 3, 4, 5], (
            f"case/whitespace variants coded as {messy.tolist()}")
        # "Disagree" must not be absorbed by "Strongly disagree".
        pair = _coerce_likert(pd.Series(["Disagree", "Strongly disagree"]), "q1")
        assert pair.tolist() == [2, 1], f"substring collision: {pair.tolist()}"

        # Legacy numeric responses pass through unchanged, in any form.
        for label, series in (("str", pd.Series(["1", "3", "5"])),
                              ("int", pd.Series([1, 3, 5])),
                              ("float", pd.Series([1.0, 3.0, 5.0]))):
            got = _coerce_likert(series, "q1")
            assert got.tolist() == [1, 3, 5], f"numeric {label}: {got.tolist()}"

        # Both representations mixed inside one column, plus blanks.
        tally: dict = {}
        mixed = _coerce_likert(
            pd.Series(["Agree", "2", " strongly agree ", 4, None, ""]),
            "q1", tally)
        assert mixed.tolist()[:4] == [4, 2, 5, 4], f"mixed column: {mixed.tolist()}"
        assert mixed.isna().sum() == 2, "blanks did not become missing"
        assert tally == {"text": 2, "numeric": 2}, f"tally wrong: {tally}"

        # Unrecognised values must raise and name the offender.
        for bad in ("Somewhat agree", "Strongly Agreed", "n/a", "3.5"):
            try:
                _coerce_likert(pd.Series(["Agree", bad]), "q1")
            except ValueError as exc:
                assert bad in str(exc), (
                    f"error for {bad!r} does not name the offending value: {exc}")
            else:
                raise AssertionError(f"{bad!r} was accepted on a Likert item")
        try:
            _coerce_likert(pd.Series(["7"]), "q1")
        except ValueError as exc:
            assert "7" in str(exc), f"out-of-range error does not name 7: {exc}"
        else:
            raise AssertionError("numeric 7 was accepted on a 1-5 item")

        # The committed export must exercise both paths, or this is untested.
        raw = pd.read_csv(export_path, dtype=str)
        cols = list(raw.columns)
        positions = [ITEMS.index(i) + 2 for i in LIKERT_ITEMS]
        values = pd.concat([raw[cols[p]] for p in positions]).dropna()
        numeric = values[values.str.strip().str.fullmatch(r"\d+")]
        text = values[~values.str.strip().str.fullmatch(r"\d+")]
        assert len(numeric) > 0 and len(text) > 0, (
            f"the synthetic export must contain both representations to test "
            f"them; found {len(numeric)} numeric and {len(text)} text")
        return (f"{len(text)} agreement labels + {len(numeric)} numeric in the "
                f"export; {text.nunique()} surface variants all resolved")

    cl.check("Likert accepts agreement labels and numbers", _likert_formats)

    def _org_groups() -> str:
        """Every q12 answer must land in a valid group, or be blank."""
        assert loaded, "survey data failed to load (see the schema check above)"
        survey = loaded["survey"]

        assert len(ORG_CATEGORIES) == 7, (
            f"expected 7 listed organisation categories, got {len(ORG_CATEGORIES)}")
        assert ORG_GROUPS == ORG_CATEGORIES + [ORG_OTHER], (
            "ORG_GROUPS must be the listed categories followed by Other")

        # Each listed option maps to itself, whatever its casing or padding.
        for category in ORG_CATEGORIES:
            for variant in (category, category.upper(), f"  {category.lower()} "):
                got = organisation_group(variant)
                assert got == category, (
                    f"{variant!r} grouped as {got!r}, expected {category!r}")
        # Free text falls through to Other; blanks are missing, not Other.
        for typed in ("Freelance marine surveyor", "Customs broker", "n/a"):
            assert organisation_group(typed) == ORG_OTHER, (
                f"{typed!r} should group as {ORG_OTHER!r}")
        for blank in ("", "   ", None, pd.NA):
            assert pd.isna(organisation_group(blank)), (
                f"{blank!r} should be missing, not a group")

        # Every value in the data lands somewhere valid.
        groups = survey["q12_group"]
        unexpected = set(groups.dropna()) - set(ORG_GROUPS)
        assert not unexpected, (
            f"q12_group contains {sorted(unexpected)!r}, not in {ORG_GROUPS!r}")
        # And each row's group is exactly what the raw answer implies.
        for raw, group in zip(survey["q12_raw"], groups):
            want = organisation_group(raw)
            same = (pd.isna(want) and pd.isna(group)) or want == group
            assert same, f"q12_raw {raw!r} grouped as {group!r}, expected {want!r}"

        n_other = int((groups == ORG_OTHER).sum())
        n_blank = int(groups.isna().sum())
        assert n_other > 0, (
            "no free-text 'Other' answers in the synthetic export, so the "
            "fall-through is never exercised")
        assert n_blank > 0, (
            "no blank organisation answers, so the blank-vs-Other distinction "
            "is never exercised")
        return (f"{int(groups.notna().sum())} classified into "
                f"{groups.nunique()} of {len(ORG_GROUPS)} groups "
                f"({n_other} free-text -> Other, {n_blank} blank -> missing)")

    cl.check("Every q12 answer lands in a valid group", _org_groups)

    def _survey_analysis() -> str:
        assert loaded, "survey data failed to load (see the schema check above)"
        survey = loaded["survey"]
        res = run_survey_analysis(survey, figure_dir=figure_dir,
                                  make_figure=not quick)
        alpha = res["alpha"]
        value = alpha["alpha"]
        assert isinstance(value, float) and np.isfinite(value), (
            "Cronbach's alpha is not a finite float")
        assert -1.0 <= value <= 1.0, f"alpha {value} outside [-1, 1]"
        assert alpha["n_items"] == 7, (
            f"expected a 7-item scale, got {alpha['n_items']}")
        assert len(res["likert"]) == len(LIKERT_ITEMS), (
            "descriptives missing for some Likert items")
        assert res["likert"]["mean"].between(1, 5).all(), (
            "a Likert item mean fell outside 1-5")
        assert res["likert"]["sd"].notna().all(), "a Likert SD is missing"

        # Frequency tables must account for every valid response.
        freq = res["frequencies"]
        for item, block in freq.groupby("item"):
            assert int(block["count"].sum()) == int(block["n_valid"].iloc[0]), (
                f"{item}: frequency counts do not sum to n_valid")
            assert abs(block["pct_of_valid"].sum() - 100.0) < 1e-9, (
                f"{item}: percentages do not sum to 100")

        # Alpha on 7 identical columns is exactly 1.0; a useful sanity anchor.
        same = pd.DataFrame({f"i{k}": survey["q1"].astype(float) for k in range(7)})
        assert abs(cronbach_alpha(same)["alpha"] - 1.0) < 1e-9, (
            "alpha on seven identical items is not 1.0")
        return (f"alpha = {value:.3f} ({alpha['interpretation']}), "
                f"{alpha['n_respondents']} complete cases")

    cl.check("Survey descriptives and Cronbach's alpha", _survey_analysis)

    def _survey_figures() -> str:
        missing = []
        for stem in viz.SURVEY_FIGURE_STEMS:
            for ext in config.FIGURE_FORMATS:
                path = figure_dir / f"{stem}.{ext}"
                if not path.exists() or path.stat().st_size < 1024:
                    missing.append(path.name)
        assert not missing, f"survey figure(s) missing or too small: {missing}"
        assert len(viz.SURVEY_FIGURE_STEMS) == 2, "expected 2 survey figures"
        return (f"{', '.join(viz.SURVEY_FIGURE_STEMS)} in "
                f"{len(config.FIGURE_FORMATS)} formats each")

    cl.check("Survey figures exist", _survey_figures)

    # --------------------------------------------------------------- 11 ----
    def _tables() -> str:
        survey_res = (run_survey_analysis(loaded["survey"], make_figure=False)
                      if loaded else None)
        tables = collect_tables(delphi_res, ahp_res, score_res, sens_res,
                                benchmark_0_100=bench, survey_result=survey_res)
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
        print("\nHeadline results (synthetic data)")
        print("-" * 72)
        print(f"  POSRRI                    {overall['POSRRI_0_100']:.2f} / 100  "
              f"({overall['band']})")
        print(f"  Strongest domain          {overall['strongest_domain']}  "
              f"{st.DOMAINS[overall['strongest_domain']]}")
        print(f"  Weakest domain            {overall['weakest_domain']}  "
              f"{st.DOMAINS[overall['weakest_domain']]}")
        print(f"  Delphi retained           "
              f"{len(delphi_res['retained'])}/50 indicators "
              f"(Cohen's kappa {delphi_res['summary']['cohens_kappa']:.3f})")
        print(f"  Max AHP consistency ratio {consistency['CR'].max():.4f}")
        print(f"  Spearman rho (AHP/equal)  {sens_res['spearman']['rho']:.3f}")

    return 0 if cl.failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the POSRRI pipeline.")
    parser.add_argument("--quick", action="store_true",
                        help="skip figure rendering (figures must already exist)")
    args = parser.parse_args()
    sys.exit(main(quick=args.quick))
