"""
synthetic/generate_synthetic.py -- reproducible synthetic POSRRI dataset.

Generates a complete, schema-matching demonstration dataset so that the whole
pipeline (Delphi -> AHP -> scoring -> sensitivity -> figures) can be executed,
reviewed and unit-tested without touching real participant data.

Design goals
------------
* **Deterministic.** One master seed (``config.RANDOM_SEED``) is split into
  independent ``SeedSequence`` children, one per dataset, so adding or
  reordering a dataset never changes the others.
* **Exercises the analysis logic.** Delphi relevance is generated from three
  latent indicator classes (strong / borderline / weak) so that the consensus
  rule, the I-CVI retention cut-off and the round-1 vs round-2 Cohen's kappa
  all have something to bite on.
* **Realistic AHP.** Pairwise judgements are built from a hidden "true"
  priority vector, perturbed with mild multiplicative noise and snapped to the
  Saaty scale, then rejection-sampled so every expert matrix is near-consistent
  (CR < 0.10) as a well-run elicitation would be.
* **Internally consistent.** The CPA column of the benchmark table is derived
  from the same synthetic indicator scores that feed the index, so the radar
  chart cannot contradict the results table.

Usage
-----
    python synthetic/generate_synthetic.py                 # -> synthetic/data/
    python synthetic/generate_synthetic.py --seed 7 --outdir /tmp/alt
"""

from __future__ import annotations

import argparse
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
from src.ahp import consistency_ratio, priority_weights  # noqa: E402
from src.survey_ingest import load_survey_export  # noqa: E402

# ---------------------------------------------------------------------------
# Panel sizes
# ---------------------------------------------------------------------------

N_DELPHI_EXPERTS = 18
N_AHP_EXPERTS = 8
N_SURVEY_RESPONDENTS = 50          # respondents who consent, i.e. analysed
N_SURVEY_NONCONSENT = 3            # rows the consent filter must drop
N_SURVEY_ITEMS = 15

#: The seven five-point Likert items, in questionnaire order.
LIKERT_POSITIONS = ["q1", "q3", "q4", "q5", "q8", "q9", "q10"]

#: Admissible Saaty judgements as (value, label) pairs, reciprocals included.
SAATY_CHOICES: List[tuple] = (
    [(1.0 / k, f"1/{k}") for k in range(9, 1, -1)]
    + [(float(k), str(k)) for k in range(1, 10)]
)
_SAATY_VALUES = np.array([v for v, _ in SAATY_CHOICES])
_SAATY_LABELS = [lab for _, lab in SAATY_CHOICES]


# ---------------------------------------------------------------------------
# 1. Delphi ratings
# ---------------------------------------------------------------------------

#: Latent quality classes. ``p_high`` is the probability that an expert rates
#: the indicator 7-9 in round 1 (i.e. the expected I-CVI).
_DELPHI_CLASSES = {
    "strong":     dict(p_high=0.94, clarity=4.4, feasibility=4.2),
    "borderline": dict(p_high=0.76, clarity=3.6, feasibility=3.4),
    "weak":       dict(p_high=0.48, clarity=2.9, feasibility=2.8),
}


def _assign_delphi_classes(rng: np.random.Generator) -> Dict[str, str]:
    """Give every indicator a latent quality class.

    Roughly 72% strong, 16% borderline, 12% weak: enough weak indicators to
    fail the I-CVI cut-off and enough borderline ones to change decision
    between rounds, which is what makes Cohen's kappa informative.
    """
    n = st.N_INDICATORS
    labels = np.array(["strong"] * n, dtype=object)
    idx = rng.permutation(n)
    labels[idx[:6]] = "weak"
    labels[idx[6:14]] = "borderline"
    return dict(zip(st.INDICATORS, labels))


def _round2_p_high(p1: float, rng: np.random.Generator) -> float:
    """Round-2 probability of a 7-9 rating.

    Delphi feedback polarises opinion: panels that already agree converge
    further, panels that are split drift either way. Borderline indicators
    therefore flip retain/drop between rounds, which is exactly the behaviour
    the kappa statistic is meant to quantify.
    """
    if p1 >= 0.85:
        drift = rng.normal(0.04, 0.03)
    elif p1 <= 0.60:
        drift = rng.normal(-0.06, 0.05)
    else:
        drift = rng.normal(0.0, 0.10)          # genuinely undecided
    return float(np.clip(p1 + drift, 0.05, 0.99))


def generate_delphi(rng: np.random.Generator) -> pd.DataFrame:
    """18 experts x 50 indicators x 2 rounds of relevance/clarity/feasibility."""
    experts = [f"E{i:02d}" for i in range(1, N_DELPHI_EXPERTS + 1)]
    classes = _assign_delphi_classes(rng)

    # Per-expert leniency: some raters are systematically generous.
    leniency = rng.normal(0.0, 0.35, size=len(experts))

    rows = []
    for ind in st.INDICATORS:
        params = _DELPHI_CLASSES[classes[ind]]
        p_by_round = {1: params["p_high"], 2: _round2_p_high(params["p_high"], rng)}

        for rnd in config.DELPHI_ROUNDS:
            p_high = p_by_round[rnd]
            # Round 2 dispersion is tighter (feedback narrows the IQR).
            spread = 1.0 if rnd == 1 else 0.7
            for e, expert in enumerate(experts):
                p_e = float(np.clip(p_high + 0.05 * leniency[e], 0.02, 0.99))
                if rng.random() < p_e:
                    relevance = int(np.clip(round(rng.normal(8.1, 0.8 * spread)), 7, 9))
                else:
                    relevance = int(np.clip(round(rng.normal(4.6, 1.4 * spread)), 1, 6))

                clarity = int(np.clip(
                    round(rng.normal(params["clarity"] + 0.2 * leniency[e], 0.7)), 1, 5))
                feasibility = int(np.clip(
                    round(rng.normal(params["feasibility"] + 0.2 * leniency[e], 0.8)), 1, 5))

                rows.append({
                    "expert_id": expert,
                    "round": rnd,
                    "indicator_code": ind,
                    "relevance": relevance,
                    "clarity": clarity,
                    "feasibility": feasibility,
                })

    frame = pd.DataFrame(rows)
    return frame.sort_values(["round", "expert_id", "indicator_code"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. AHP pairwise judgements
# ---------------------------------------------------------------------------

def _snap_to_saaty(ratio: float) -> tuple:
    """Snap a continuous preference ratio to the nearest Saaty value.

    Nearest is measured in log space, which is the natural metric for a ratio
    scale (3 vs 4 is the same perceptual step as 1/3 vs 1/4).
    """
    ratio = float(np.clip(ratio, 1.0 / 9.0, 9.0))
    k = int(np.argmin(np.abs(np.log(_SAATY_VALUES) - np.log(ratio))))
    return _SAATY_VALUES[k], _SAATY_LABELS[k]


def _matrix_from_labels(items: List[str], labels: Dict[tuple, float]) -> np.ndarray:
    """Rebuild the full reciprocal matrix from upper-triangular judgements."""
    n = len(items)
    A = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = labels[(items[i], items[j])]
            A[j, i] = 1.0 / A[i, j]
    return A


def _expert_judgements(items: List[str],
                       true_w: np.ndarray,
                       rng: np.random.Generator,
                       sigma: float = 0.16,
                       max_tries: int = 60) -> Dict[tuple, str]:
    """One expert's upper-triangular judgements for a single matrix.

    Judgements are ``(w_i / w_j) * exp(eps)`` snapped to the Saaty scale.
    The matrix is resampled (with shrinking noise) until CR < 0.10, mimicking
    an elicitation in which inconsistent respondents are fed back to and asked
    to revise -- standard practice in applied AHP studies.
    """
    n = len(items)
    for attempt in range(max_tries):
        s = sigma * (0.85 ** attempt)          # shrink noise if we keep failing
        num, lab = {}, {}
        for i in range(n):
            for j in range(i + 1, n):
                ratio = (true_w[i] / true_w[j]) * np.exp(rng.normal(0.0, s))
                val, text = _snap_to_saaty(ratio)
                num[(items[i], items[j])] = val
                lab[(items[i], items[j])] = text
        A = _matrix_from_labels(items, num)
        cr = consistency_ratio(A, priority_weights(A))["CR"]
        if cr < config.CR_THRESHOLD:
            return lab
    return lab  # pragma: no cover - unreachable for n <= 5 at this noise level


def _true_weights(n: int, rng: np.random.Generator) -> np.ndarray:
    """Hidden priority vector for a matrix of order ``n``.

    A moderately concentrated Dirichlet keeps every pairwise ratio inside the
    1/9..9 Saaty window, so snapping never has to clip a genuine preference.
    """
    for _ in range(200):
        w = rng.dirichlet(np.full(n, 7.0))
        if (w.max() / w.min()) <= 7.0:
            return w
    return np.full(n, 1.0 / n)  # pragma: no cover


def generate_ahp(rng: np.random.Generator) -> pd.DataFrame:
    """Pairwise judgements from 8 experts across all three hierarchy levels."""
    experts = [f"E{i:02d}" for i in range(1, N_AHP_EXPERTS + 1)]

    # One hidden "true" priority vector per matrix, shared by all experts.
    matrices = [("pillar", "GOAL", list(st.PILLARS.keys()))]
    for pil, doms in st.PILLAR_DOMAINS.items():
        matrices.append(("domain", pil, list(doms)))
    for dom in st.DOMAIN_ORDER:
        matrices.append(("indicator", dom, list(st.DOMAIN_INDICATORS[dom])))

    truth = {(lvl, grp): _true_weights(len(items), rng)
             for lvl, grp, items in matrices}

    rows = []
    for expert in experts:
        for lvl, grp, items in matrices:
            lab = _expert_judgements(items, truth[(lvl, grp)], rng)
            for (i_code, j_code), text in lab.items():
                rows.append({
                    "expert_id": expert,
                    "level": lvl,
                    "group": grp,
                    "item_i": i_code,
                    "item_j": j_code,
                    "saaty_value": text,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Indicator scores (documentary vs field)
# ---------------------------------------------------------------------------

#: Latent readiness per domain on the 0-3 rubric. The profile is deliberately
#: shaped like a developing-country port: statute and command structure are
#: comparatively strong, physical response capacity and after-action learning
#: are weak. These are illustrative values, not findings.
_DOMAIN_READINESS = {
    "D1": 2.25, "D2": 1.85, "D3": 1.95, "D4": 1.40, "D5": 1.05,
    "D6": 1.20, "D7": 1.10, "D8": 1.65, "D9": 1.50, "D10": 0.90,
}

_EVIDENCE_SOURCES = [
    "Gazette notification (statutory instrument)",
    "Port contingency plan, Annex II",
    "Key-informant interview, port authority",
    "Site inspection checklist",
    "Exercise after-action report",
    "IMO / regional convention status record",
    "Equipment inventory register",
    "Training records and certificates",
    "Inter-agency memorandum of understanding",
    "Budget allocation document",
]


def generate_scores(rng: np.random.Generator) -> pd.DataFrame:
    """Documentary, field and adjudicated scores for all 50 indicators.

    ``doc_score`` is what the paperwork claims (work-as-imagined) and
    ``field_score`` what verification found (work-as-done). The implementation
    gap is positive on average -- the gap figure would be meaningless
    otherwise -- but a minority of indicators score higher in the field than on
    paper, reflecting undocumented practice.
    """
    rows = []
    for ind in st.INDICATORS:
        base = _DOMAIN_READINESS[st.domain_of(ind)]
        latent = base + rng.normal(0.0, 0.55)

        doc = int(np.clip(round(latent + rng.normal(0.35, 0.45)), 0, 3))
        # Implementation gap: usually doc >= field, occasionally reversed.
        gap = rng.normal(0.55, 0.55) if rng.random() < 0.82 else rng.normal(-0.45, 0.35)
        field = int(np.clip(round(doc - gap), 0, 3))

        # Adjudicated score weights verified practice above documentation.
        final = int(np.clip(round((doc + 2.0 * field) / 3.0), 0, 3))

        rows.append({
            "indicator_code": ind,
            "doc_score": doc,
            "field_score": field,
            "final_score": final,
            "evidence_source": _EVIDENCE_SOURCES[int(rng.integers(len(_EVIDENCE_SOURCES)))],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Stakeholder survey
# ---------------------------------------------------------------------------

#: Question wording as it would appear in the Google Form. The ingest matches
#: items BY POSITION, so these strings exist only to make the synthetic export
#: realistic -- editing any of them must not change the ingested result.
SURVEY_HEADERS = [
    "Timestamp",
    "Do you consent to take part in this study? Responses are anonymised and "
    "used only for academic research.",
    "How confident are you that the port oil spill contingency plan would work "
    "in a real Tier 2 spill?",                                         # q1  Likert
    "Are you aware of a designated On-Scene Commander for oil spill response "
    "at this port?",                                                   # q2  Yes/Unsure/No
    "How clear are your own roles and responsibilities during a spill response?",
                                                                       # q3  Likert
    "How adequate do you consider the response equipment currently available?",
                                                                       # q4  Likert
    "How effective is coordination between agencies during a spill response?",
                                                                       # q5  Likert
    "Have you been given access to the port oil spill contingency plan?",
                                                                       # q6  Yes/Unsure/No
    "How often does your organisation take part in oil spill response exercises?",
                                                                       # q7  frequency
    "How confident are you in the notification and reporting chain for a spill?",
                                                                       # q8  Likert
    "How adequate is the funding available for oil spill preparedness?",
                                                                       # q9  Likert
    "How confident are you that lessons from incidents and exercises are acted "
    "upon?",                                                           # q10 Likert
    "In your view, what is the single greatest barrier to oil spill "
    "preparedness at this port?",                                      # q11 text
    "What one change would most improve oil spill readiness at this port?",
                                                                       # q12 text
    "How many years have you worked in port operations, shipping or "
    "environmental regulation?",                                       # q13 bands
    "Have you received formal oil spill response training (e.g. IMO OPRC Level "
    "1, 2 or 3)?",                                                     # q14 Yes/No
    "Which organisation or agency do you work for? (optional)",        # q15 text
]

_BARRIERS = [
    "Equipment is not maintained", "No dedicated budget",
    "Unclear lines of authority", "Too few trained staff",
    "Plan is out of date", "Agencies do not exercise together", "",
]
_IMPROVEMENTS = [
    "Fund a standing response unit", "Run a full-scale annual exercise",
    "Appoint a permanent on-scene commander", "Replace ageing booms",
    "Publish the plan to all terminals", "Set up an incident database", "",
]
_ORGANISATIONS = [
    "Port authority", "Department of Environment", "Coast guard",
    "Terminal operator", "Shipping agent", "Fisheries department", "",
]


def generate_survey_export(rng: np.random.Generator) -> pd.DataFrame:
    """A synthetic *raw Google Forms export*, before any cleaning.

    Deliberately messy in the ways a real export is, so that
    :func:`src.survey_ingest.load_survey_export` is exercised rather than
    merely called: it carries a Timestamp column and a Q0 consent question,
    rows in submission order rather than sorted, non-consenting rows that must
    be dropped, category labels rather than codes, "Don't know" answers that
    must become missing, blank free-text answers, and a handful of labels
    carrying the curly apostrophes and en-dashes Google Sheets substitutes.

    The seven Likert confidence items load on a shared respondent factor, so
    the Cronbach's alpha reported by ``src.survey_analysis`` is meaningful
    rather than an artefact of independent noise. The loading (0.62) and
    residual SD (0.65) are calibrated to put alpha in the low-to-mid 0.8s --
    the range a well-behaved scale actually occupies. A stronger loading gave
    alpha > 0.95, which in a real study signals redundant items rather than a
    good scale, and would have been a misleading demonstration.
    """
    n = N_SURVEY_RESPONDENTS + N_SURVEY_NONCONSENT

    # Shared respondent factor plus per-item difficulty for the Likert items.
    theta = rng.normal(0.0, 1.0, size=n)
    item_effect = rng.normal(0.0, 0.40, size=len(LIKERT_POSITIONS))

    # Submission times over roughly six weeks, then shuffled so the ingest has
    # to sort them: respondent_id must follow timestamp order, not file order.
    minutes = np.sort(rng.integers(0, 60 * 24 * 42, size=n))
    minutes = rng.permutation(minutes)
    base = np.datetime64("2026-03-02T08:00:00")

    consent = np.array(["Yes"] * n, dtype=object)
    consent[rng.choice(n, size=N_SURVEY_NONCONSENT, replace=False)] = "No"

    rows = []
    for i in range(n):
        stamp = base + np.timedelta64(int(minutes[i]), "m")
        # Google Forms' default US locale format.
        ts = pd.Timestamp(stamp).strftime("%-m/%-d/%Y %-H:%M:%S")

        likert = {}
        for k, item in enumerate(LIKERT_POSITIONS):
            val = 3.0 + 0.62 * theta[i] + item_effect[k] + rng.normal(0, 0.65)
            likert[item] = str(int(np.clip(round(val), 1, 5)))

        # Awareness tracks confidence, so the coded items are not independent
        # of the Likert block -- as in real data.
        aware = "Yes" if theta[i] > 0.25 else ("Unsure" if theta[i] > -0.7 else "No")
        access = "Yes" if theta[i] > 0.55 else ("Unsure" if theta[i] > -0.4 else "No")

        freq_roll = rng.random()
        if freq_roll < 0.10:
            # Curly apostrophe for some rows: Sheets substitutes it silently.
            frequency = "Don\u2019t know" if rng.random() < 0.5 else "Don't know"
        else:
            frequency = str(rng.choice(["Never", "Rarely", "Sometimes", "Often"],
                                       p=[0.30, 0.34, 0.26, 0.10]))

        band = str(rng.choice(["Less than 2 years", "2-5 years", "6-10 years",
                               "More than 10 years"], p=[0.16, 0.32, 0.30, 0.22]))
        if band == "2-5 years" and rng.random() < 0.35:
            band = "2\u20135 years"                  # en-dash variant

        rows.append({
            SURVEY_HEADERS[0]: ts,
            SURVEY_HEADERS[1]: consent[i],
            SURVEY_HEADERS[2]: likert["q1"],
            SURVEY_HEADERS[3]: aware,
            SURVEY_HEADERS[4]: likert["q3"],
            SURVEY_HEADERS[5]: likert["q4"],
            SURVEY_HEADERS[6]: likert["q5"],
            SURVEY_HEADERS[7]: access,
            SURVEY_HEADERS[8]: frequency,
            SURVEY_HEADERS[9]: likert["q8"],
            SURVEY_HEADERS[10]: likert["q9"],
            SURVEY_HEADERS[11]: likert["q10"],
            SURVEY_HEADERS[12]: _BARRIERS[int(rng.integers(len(_BARRIERS)))],
            SURVEY_HEADERS[13]: _IMPROVEMENTS[int(rng.integers(len(_IMPROVEMENTS)))],
            SURVEY_HEADERS[14]: band,
            SURVEY_HEADERS[15]: "Yes" if theta[i] > -0.15 else "No",
            SURVEY_HEADERS[16]: _ORGANISATIONS[int(rng.integers(len(_ORGANISATIONS)))],
        })

    return pd.DataFrame(rows, columns=SURVEY_HEADERS)


# ---------------------------------------------------------------------------
# 5. Benchmark ports
# ---------------------------------------------------------------------------

#: Illustrative comparator profiles: overall level plus per-domain dispersion.
_PORT_PROFILE = {
    "Singapore": dict(level=2.70, sd=0.16),
    "LosAngelesLongBeach": dict(level=2.62, sd=0.20),
    "Australia": dict(level=2.50, sd=0.22),
}


def generate_benchmark(scores: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Domain-level 0-3 scores for the four benchmark ports.

    CPA is not invented: it is the domain mean of the synthetic indicator
    ``final_score`` values, so the benchmark table and the index results are
    guaranteed to tell the same story.
    """
    dom = scores.assign(domain_code=scores["indicator_code"].map(st.INDICATOR_TO_DOMAIN))
    cpa = dom.groupby("domain_code")["final_score"].mean()

    rows = []
    for d in st.DOMAIN_ORDER:
        rows.append({"domain_code": d, "port": "CPA", "score": round(float(cpa[d]), 2)})
    for port, prof in _PORT_PROFILE.items():
        for d in st.DOMAIN_ORDER:
            val = float(np.clip(rng.normal(prof["level"], prof["sd"]), 0.0, 3.0))
            rows.append({"domain_code": d, "port": port, "score": round(val, 2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def generate_all(seed: int = config.RANDOM_SEED,
                 outdir: "Path | str" = config.SYNTHETIC_DATA_DIR,
                 verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Generate every dataset and write it to ``outdir``. Returns the frames."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Independent streams: changing one dataset cannot perturb the others.
    streams = np.random.SeedSequence(seed).spawn(5)
    rng_delphi, rng_ahp, rng_scores, rng_survey, rng_bench = (
        np.random.default_rng(s) for s in streams
    )

    delphi = generate_delphi(rng_delphi)
    ahp = generate_ahp(rng_ahp)
    scores = generate_scores(rng_scores)
    survey_export = generate_survey_export(rng_survey)
    benchmark = generate_benchmark(scores, rng_bench)

    frames = {
        "delphi_ratings": delphi,
        "ahp_pairwise": ahp,
        "scores": scores,
        "survey_export": survey_export,
        "benchmark": benchmark,
    }

    def _report(path: Path, frame: pd.DataFrame) -> None:
        if verbose:
            shown = path.relative_to(_ROOT) if path.is_relative_to(_ROOT) else path
            print(f"  wrote {shown}  ({len(frame):,} rows x {frame.shape[1]} cols)")

    for name, frame in frames.items():
        path = outdir / f"{name}.csv"
        frame.to_csv(path, index=False)
        _report(path, frame)

    # survey.csv is not generated directly: it is produced by running the real
    # ingest over the synthetic raw export. That keeps one source of truth for
    # the survey coding, and means the committed demonstration file is
    # schema-valid by construction rather than by a parallel implementation
    # that could drift from src/survey_ingest.py.
    survey = load_survey_export(outdir / "survey_export.csv", data_dir=outdir,
                                write=True, verbose=False)
    frames["survey"] = survey
    _report(outdir / "survey.csv", survey)

    return frames


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help=f"master random seed (default {config.RANDOM_SEED})")
    parser.add_argument("--outdir", default=str(config.SYNTHETIC_DATA_DIR),
                        help="output directory (default synthetic/data)")
    args = parser.parse_args()

    print(f"Generating synthetic POSRRI data (seed={args.seed})")
    frames = generate_all(seed=args.seed, outdir=args.outdir)

    print("\nSanity summary")
    print("-" * 60)
    d = frames["delphi_ratings"]
    print(f"Delphi      : {d['expert_id'].nunique()} experts, "
          f"{d['indicator_code'].nunique()} indicators, "
          f"rounds {sorted(int(r) for r in d['round'].unique())}")
    a = frames["ahp_pairwise"]
    print(f"AHP         : {a['expert_id'].nunique()} experts, "
          f"{a.groupby(['level', 'group']).ngroups} matrices per expert")
    s = frames["scores"]
    print(f"Scores      : mean final={s['final_score'].mean():.2f}, "
          f"mean doc-field gap={(s['doc_score'] - s['field_score']).mean():+.2f}")
    e, sv = frames["survey_export"], frames["survey"]
    print(f"Survey      : {len(e)} raw rows -> {len(sv)} consenting respondents "
          f"x {N_SURVEY_ITEMS} items "
          f"({len(e) - len(sv)} dropped on consent)")
    b = frames["benchmark"]
    print(f"Benchmark   : {b['port'].nunique()} ports x {b['domain_code'].nunique()} domains")


if __name__ == "__main__":
    _cli()
