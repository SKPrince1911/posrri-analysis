"""
synthetic/generate_synthetic.py -- reproducible synthetic POSRRI dataset.

Generates a schema-matching dataset so the pipeline code can be smoke-tested
without touching the study data.

THIS IS NOT STUDY DATA. It exists to exercise the code paths, nothing else.
Never use it to produce a reported result or to validate the analysis: the
numbers are invented. The pipeline reads the study data from ``data/raw``.

Design goals
------------
* **Deterministic.** One master seed (``config.RANDOM_SEED``) is split into
  independent ``SeedSequence`` children, one per dataset, so adding or
  reordering a dataset never changes the others.
* **Exercises the analysis logic.** Expert ratings are generated for a small
  panel with a minority of contested indicators, so the all-experts-agree rule
  has something to separate.
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

# ---------------------------------------------------------------------------
# Panel sizes
# ---------------------------------------------------------------------------

N_DELPHI_EXPERTS = 5
N_AHP_EXPERTS = 8
#: Admissible Saaty judgements as (value, label) pairs, reciprocals included.
SAATY_CHOICES: List[tuple] = (
    [(1.0 / k, f"1/{k}") for k in range(9, 1, -1)]
    + [(float(k), str(k)) for k in range(1, 10)]
)
_SAATY_VALUES = np.array([v for v, _ in SAATY_CHOICES])
_SAATY_LABELS = [lab for _, lab in SAATY_CHOICES]


# ---------------------------------------------------------------------------
# 1. Expert review ratings
# ---------------------------------------------------------------------------

def generate_delphi(rng: np.random.Generator) -> pd.DataFrame:
    """A small expert panel rating every indicator once.

    Shaped like the real review: a handful of experts, a single round, and most
    indicators unanimous with a minority drawing one or two dissenting ratings,
    so the all-experts-agree rule has something to separate.
    """
    experts = [f"E{i:02d}" for i in range(1, N_DELPHI_EXPERTS + 1)]

    # Most indicators are endorsed by everyone; some draw a dissent.
    n = st.N_INDICATORS
    contested = set(rng.choice(n, size=max(1, n // 6), replace=False).tolist())

    rows = []
    for k, ind in enumerate(st.INDICATORS):
        dissenters = set()
        if k in contested:
            n_dissent = int(rng.integers(1, 3))
            dissenters = set(rng.choice(len(experts), size=n_dissent,
                                        replace=False).tolist())
        for e, expert in enumerate(experts):
            if e in dissenters:
                relevance = int(np.clip(round(rng.normal(5.0, 1.0)), 1, 6))
            else:
                relevance = int(np.clip(round(rng.normal(8.2, 0.7)), 7, 9))
            rows.append({
                "expert_id": expert,
                "indicator_code": ind,
                "relevance": relevance,
                "clarity": int(np.clip(round(rng.normal(4.2, 0.7)), 1, 5)),
                "feasibility": int(np.clip(round(rng.normal(4.0, 0.8)), 1, 5)),
            })

    return pd.DataFrame(rows).sort_values(
        ["expert_id", "indicator_code"]).reset_index(drop=True)


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
# 4. Benchmark ports
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
    rng_delphi, rng_ahp, rng_scores, _reserved, rng_bench = (
        np.random.default_rng(s) for s in streams
    )

    delphi = generate_delphi(rng_delphi)
    ahp = generate_ahp(rng_ahp)
    scores = generate_scores(rng_scores)
    benchmark = generate_benchmark(scores, rng_bench)

    frames = {
        "delphi_ratings": delphi,
        "ahp_pairwise": ahp,
        "scores": scores,
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
    print(f"Expert review: {d['expert_id'].nunique()} experts x "
          f"{d['indicator_code'].nunique()} indicators")
    a = frames["ahp_pairwise"]
    print(f"AHP         : {a['expert_id'].nunique()} experts, "
          f"{a.groupby(['level', 'group']).ngroups} matrices per expert")
    s = frames["scores"]
    print(f"Scores      : mean final={s['final_score'].mean():.2f}, "
          f"mean doc-field gap={(s['doc_score'] - s['field_score']).mean():+.2f}")
    b = frames["benchmark"]
    print(f"Benchmark   : {b['port'].nunique()} ports x {b['domain_code'].nunique()} domains")


if __name__ == "__main__":
    _cli()
