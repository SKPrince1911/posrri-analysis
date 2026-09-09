# POSRRI — Port Oil Spill Regulatory Readiness Index

A complete, reproducible analysis pipeline for the **Port Oil Spill Regulatory
Readiness Index (POSRRI)**, applied to **Chattogram Port, Bangladesh**.
Target journal: *Marine Policy*.

The index is a three-level composite: **3 pillars → 10 domains → 50 indicators**.
Indicators are validated by a two-round **Delphi** panel, weighted by the
**Analytic Hierarchy Process (AHP)**, scored on a 0–3 evidence rubric, aggregated
to a 0–100 readiness score, and stress-tested against an equal-weight null.

Everything is deterministic (one master seed), modular, documented, and runs
unchanged in Google Colab or on a local machine.

---

## Contents

- [Quick start](#quick-start)
- [The Colab workflow](#the-colab-workflow)
- [Repository layout](#repository-layout)
- [Index structure](#index-structure)
- [Data schemas](#data-schemas)
- [Method summary](#method-summary)
- [Outputs](#outputs)
- [Validation](#validation)
- [Reproducibility and data protection](#reproducibility-and-data-protection)
- [Method references](#method-references)

---

## Quick start

```bash
git clone https://github.com/SKPrince1911/posrri-analysis.git
cd posrri-analysis
pip install -r requirements.txt

# 1. Generate the synthetic demonstration dataset (deterministic, seed 42)
python synthetic/generate_synthetic.py

# 2. Run the whole pipeline and print a PASS/FAIL checklist
python tests/validate_pipeline.py
```

Individual stages can also be run on their own; each module has a `__main__`
block that executes it against the synthetic data:

```bash
python src/structure.py      # print and self-validate the hierarchy
python src/delphi.py         # Delphi consensus, I-CVI/S-CVI, Cohen's kappa
python src/ahp.py            # AHP weights and consistency ratios
python src/scoring.py        # domain, pillar and overall POSRRI
python src/sensitivity.py    # AHP versus equal weights
python src/viz.py            # regenerate all six figures
```

For the full narrated run with tables and figures, open
`notebooks/00_run_all.ipynb`.

---

## The Colab workflow

The notebook is Colab-first. Open it, then:

1. **Open in Colab.** From GitHub, use
   `https://colab.research.google.com/github/SKPrince1911/posrri-analysis/blob/main/notebooks/00_run_all.ipynb`
   (substitute the branch name if you are working on a branch).
2. **Set `USE_SYNTHETIC` in the first code cell.**
   `True` reads the committed synthetic dataset; `False` reads your real data.
3. **Runtime → Run all.**

The configuration cell handles the rest automatically:

| It does this | Why |
|---|---|
| Detects Colab | so the same notebook still runs locally |
| `pip install`s the requirements | Colab images drift over time |
| Clones the repo to `/content/posrri-analysis`, or fast-forwards it if already there | you always run the current code |
| **Mounts Google Drive** and creates `MyDrive/POSRRI/` | Colab runtimes are recycled; Drive is not |
| Sets `DATA_DIR` and `OUTPUT_DIR` | outputs land in `MyDrive/POSRRI/outputs/`, real data is read from `MyDrive/POSRRI/data/raw/` |

If Drive cannot be mounted the notebook says so and falls back to runtime-local
paths, rather than failing.

**Using real data in Colab:** put your filled-in CSVs in
`MyDrive/POSRRI/data/raw/`, set `USE_SYNTHETIC = False`, and *Run all*. Real
data never enters the repository, so it is never at risk of being committed.

To pick up later changes to the code, just re-run the configuration cell — it
fetches and fast-forwards the clone.

---

## Repository layout

```
posrri-analysis/
├── README.md
├── requirements.txt
├── .gitignore                     excludes data/raw/ and Colab checkpoints
├── config.py                      paths, master seed, every threshold, figure style
├── data/
│   ├── templates/                 header-only CSVs + schema documentation
│   └── raw/                       real data (GIT-IGNORED — never committed)
├── synthetic/
│   ├── generate_synthetic.py      deterministic synthetic dataset generator
│   └── data/                      the generated dataset (committed)
├── src/
│   ├── structure.py               3 × 10 × 50 hierarchy, Saaty Random Index
│   ├── delphi.py                  consensus, I-CVI / S-CVI, Cohen's kappa
│   ├── ahp.py                     weights, consistency ratios, AIJ aggregation
│   ├── scoring.py                 weighted domain / pillar / overall POSRRI
│   ├── sensitivity.py             AHP versus equal weights, Spearman rho
│   ├── viz.py                     six publication-grade figures
│   └── report.py                  table assembly and CSV/Excel export
├── notebooks/
│   └── 00_run_all.ipynb           Colab-ready end-to-end run
├── outputs/
│   ├── figures/                   300 dpi PNG + vector PDF
│   └── tables/                    CSVs + posrri_results.xlsx
└── tests/
    └── validate_pipeline.py       PASS/FAIL checklist over the whole pipeline
```

`src/report.py` is the one module beyond the original specification: the
notebook and the validation script both need identical table exports, so that
logic lives in one place rather than being duplicated.

---

## Index structure

Defined once, in `src/structure.py`, and shared by every stage.

| Pillar | Name | Domains |
|---|---|---|
| **A** | Governance & legal foundation | D1, D2, D9 |
| **B** | Operational preparedness & response | D3, D4, D5, D6 |
| **C** | Resilience, financing & continuity | D7, D8, D10 |

| Code | Domain |
|---|---|
| D1 | Legal & Regulatory Basis |
| D2 | Designated Authority & Command Structure |
| D3 | Notification & Reporting |
| D4 | Tiered Response Arrangements |
| D5 | Resources & Equipment |
| D6 | Training & Exercises |
| D7 | Funding & Liability Arrangements |
| D8 | Plan Currency & Documentation |
| D9 | Stakeholder Coordination |
| D10 | Monitoring, Review & After-Action Learning |

Each domain holds exactly 5 indicators, coded `D1.1`…`D1.5` through
`D10.1`…`D10.5` — **50 in total**.

Benchmark ports: `CPA` (Chattogram, the case study), `Singapore`,
`LosAngelesLongBeach`, `Australia`.

---

## Data schemas

Header-only templates live in `data/templates/`, with full column
documentation in `data/templates/README.md`. Copy them into `data/raw/` and
fill them in. Loaders validate columns, indicator-code coverage and value
ranges, and fail with an explicit message rather than producing a silently
wrong index.

### `delphi_ratings.csv`
`expert_id, round, indicator_code, relevance, clarity, feasibility`
One row per expert × round × indicator. `relevance` 1–9 (7–9 = "relevant"),
`clarity` and `feasibility` 1–5.

### `ahp_pairwise.csv`
`expert_id, level, group, item_i, item_j, saaty_value`
One row per upper-triangular cell; reciprocals are filled automatically.
`level` ∈ {`pillar`, `domain`, `indicator`}; `group` is the parent node
(`GOAL` for the pillar matrix, `A`/`B`/`C` for domain matrices, `D1`…`D10` for
indicator matrices). `saaty_value` accepts integers `1`–`9` and fraction
strings such as `1/3`.

### `scores.csv`
`indicator_code, doc_score, field_score, final_score, evidence_source`
One row per indicator (50 rows). All three scores use the 0–3 rubric:
**0** absent · **1** partial · **2** substantial · **3** fully met and evidenced.
`doc_score` is work-as-imagined (documentary evidence), `field_score` is
work-as-done (field verification), `final_score` is the adjudicated value the
index uses.

### `survey.csv`
`respondent_id, q1 … q15` — one row per respondent, Likert items 1–5.

### `benchmark.csv`
`domain_code, port, score` — one row per domain × port (10 × 4 = 40 rows),
score on the 0–3 scale.

---

## Method summary

### Delphi (`src/delphi.py`)
Per indicator and round: median relevance, IQR, and the percentage rating 7–9.
The protocol consensus flag is

> **consensus** = (≥ 80 % of experts rate 7–9) **or** (median ≥ 7 **and** IQR ≤ 2)

Content validity follows Zamanzadeh et al. (2015): **I-CVI** is the proportion
rating 7–9, with items retained at **I-CVI ≥ 0.78**; the chance-corrected
**modified kappa** κ\* = (I-CVI − P<sub>c</sub>) / (1 − P<sub>c</sub>) is reported
alongside; **S-CVI/Ave** (mean I-CVI) is reported against a target of **≥ 0.90**,
with S-CVI/UA for completeness. Round-to-round stability is quantified with
**Cohen's kappa** on the retain/drop decisions.

### AHP (`src/ahp.py`)
Priority weights come from the **normalised row geometric mean**. For each
matrix, λ<sub>max</sub> = mean((A·w)/w), CI = (λ<sub>max</sub> − n)/(n − 1) and
**CR = CI / RI(n)** against Saaty's Random Index
(n = 3 → 0.58, 4 → 0.90, 5 → 1.12); **CR ≥ 0.10 is flagged**. Experts are combined
by **Aggregation of Individual Judgements (AIJ)** — the cell-wise geometric mean
of the judgements, taken *before* deriving the group priority vector, which is
the only rule preserving a<sub>ij</sub> = 1/a<sub>ji</sub>. The global weight of an
indicator is `pillar_w × domain_w_within_pillar × indicator_w_within_domain`,
normalised to sum to 1 across all 50.

### Scoring (`src/scoring.py`)
For any group *g* of indicators with weights *w* and 0–3 scores *s*:

> score(*g*) = 100 × Σ *w·s* ⁄ (3 × Σ *w*)

Dividing by the group's own weight mass makes domain scores comparable
regardless of how much global weight a domain carries. Since the 50 global
weights sum to 1, the overall POSRRI is 100 × Σ *w·s* / 3.

### Sensitivity (`src/sensitivity.py`)
The aggregation is re-run with every indicator weighted 1/50 (the
"no information" null). Reported: per-domain score deltas and rank shifts, the
change in the overall index, and the **Spearman rank correlation** between the
two domain rankings (Kendall's τ-b as a tie-robust companion).

---

## Outputs

All written to `outputs/` (or your Drive folder in Colab).

**Figures** — 300 dpi PNG plus vector PDF, colourblind-safe Okabe-Ito palette,
greyscale-legible (colour is never the only channel):

| File | Content |
|---|---|
| `fig1_radar_benchmark` | Radar of 10 domain scores, CPA overlaid on the benchmark ports |
| `fig2_domain_scores_weights` | Domain scores grouped by pillar, AHP weights annotated |
| `fig3_indicator_heatmap` | All 50 indicator scores (0–3), grouped by domain |
| `fig4_implementation_gap` | Diverging bars: work-as-imagined vs work-as-done |
| `fig5_delphi_consensus` | Median relevance + IQR whiskers, consensus flagged |
| `fig6_sensitivity_scatter` | AHP vs equal-weight domain scores, Spearman ρ annotated |

**Tables** — 16 CSVs plus `posrri_results.xlsx`, a single workbook with every
table as a sheet (summary, structure, Delphi per-indicator and per-round,
retained/dropped lists, AHP consistency report, pillar/domain/indicator
weights, indicator/domain/pillar scores, sensitivity, benchmark).

---

## Validation

```bash
python tests/validate_pipeline.py           # full run, renders figures
python tests/validate_pipeline.py --quick   # skip figure rendering
```

Runs the whole pipeline on synthetic data and prints a PASS/FAIL checklist over
15 checks: hierarchy shape, data completeness, generator determinism, CR
reported for every matrix and all below 0.10, global weights summing to 1.0,
Delphi consensus flags and I-CVI for all 50 indicators, the consensus rule
matching the protocol exactly, Cohen's kappa returning a finite value (and
`nan` rather than an exception in the degenerate case), POSRRI and all domain
scores within 0–100, exact 0/100 endpoints for the 0–3 rubric, Spearman ρ in
range, all six figures present, and every table exporting. Exits non-zero on
any failure.

---

## Reproducibility and data protection

* **One seed.** `config.RANDOM_SEED` is split into independent `SeedSequence`
  streams, one per synthetic dataset, so adding or reordering a dataset never
  perturbs the others. Regeneration is verified frame-for-frame by the
  validation script.
* **No magic numbers.** Every threshold — I-CVI 0.78, S-CVI target 0.90, the
  consensus rule, CR 0.10, the 0–3 rubric, the readiness bands — lives in
  `config.py`.
* **One structure.** The hierarchy is defined once in `src/structure.py` and
  cannot drift between stages.
* **`data/raw/` is git-ignored.** Real participant data — expert identities,
  interview-derived scores, survey responses — must never be committed. In
  Colab, keep it in Drive. The committed synthetic dataset exists precisely so
  that the pipeline can be demonstrated and reviewed without exposing anyone's
  data.

### Scope note

The Delphi stage is instrument development: it reports which indicators the
panel endorsed. The demonstration then scores the **full 50-indicator set** so
the weight and score tables are complete. In a live application you would carry
the retained set forward and re-elicit weights over it; the retained and
dropped lists are exported for exactly that purpose.

---

## Method references

* **Saaty, T.L. (1980)** *The Analytic Hierarchy Process.* New York: McGraw-Hill.
  — AHP, the 1–9 fundamental scale, the Random Index, and the consistency ratio.
* **Zamanzadeh, V., Ghahramanian, A., Rassouli, M., Abbaszadeh, A.,
  Alavi-Majd, H. & Nikanfar, A.-R. (2015)** 'Design and implementation content
  validity study: development of an instrument for measuring patient-centered
  communication', *Journal of Caring Sciences* 4(2), 165–178.
  — I-CVI, S-CVI/Ave, S-CVI/UA and the chance-corrected modified kappa.
* **Hohmann, E. et al. (2025)** — Delphi methodology: reporting both a
  consensus criterion and a stability criterion across rounds, which motivates
  the round-to-round Cohen's kappa reported here.
* **Okabe, M. & Ito, K. (2008)** 'Color universal design: how to make figures
  and presentations that are friendly to colorblind people'.
  — the colourblind-safe qualitative palette used in every figure.

Convention and instrument sources referenced by the indicator set include the
**IMO OPRC Convention 1990**, the **CLC/FUND** compensation regime, and IMO
OPRC Model Courses (Levels 1–3).
