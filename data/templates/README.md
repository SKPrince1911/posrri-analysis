# Data templates

Header-only CSV templates defining the exact schema every stage of the POSRRI
pipeline expects. Copy a template into `data/raw/` (git-ignored) and fill it
with real data, or run `synthetic/generate_synthetic.py` to produce a complete
matching dataset in `synthetic/data/`.

Column names and value ranges are validated on load; extra columns are ignored,
missing columns raise an error.

## `delphi_ratings.csv`
One row per expert x round x indicator.

| column | type | range / values | notes |
|---|---|---|---|
| `expert_id` | str | e.g. `E01` | stable across rounds |
| `round` | int | `1`, `2` | Delphi round |
| `indicator_code` | str | `D1.1` .. `D10.5` | must exist in `src/structure.py` |
| `relevance` | int | 1-9 | 7-9 counts as "relevant" for I-CVI |
| `clarity` | int | 1-5 | reported, not used for retention |
| `feasibility` | int | 1-5 | reported, not used for retention |

## `ahp_pairwise.csv`
One row per expert x matrix x upper-triangular cell. Reciprocals are filled
automatically, so only `i < j` cells are required.

| column | type | range / values | notes |
|---|---|---|---|
| `expert_id` | str | e.g. `E01` | |
| `level` | str | `pillar`, `domain`, `indicator` | hierarchy level being compared |
| `group` | str | `GOAL` for pillars; `A`/`B`/`C` for domains; `D1`..`D10` for indicators | the parent node |
| `item_i` | str | pillar / domain / indicator code | row element |
| `item_j` | str | pillar / domain / indicator code | column element |
| `saaty_value` | str or float | `1`..`9` or a fraction such as `1/3` | how much more important `i` is than `j` |

## `scores.csv`
One row per indicator (50 rows).

| column | type | range / values | notes |
|---|---|---|---|
| `indicator_code` | str | `D1.1` .. `D10.5` | |
| `doc_score` | int | 0-3 | work-as-imagined (documentary evidence) |
| `field_score` | int | 0-3 | work-as-done (field verification) |
| `final_score` | int/float | 0-3 | adjudicated score used by the index |
| `evidence_source` | str | free text | citation / document / interview reference |

Rubric (locked; defined once in `config.SCORE_RUBRIC`): 0 = absent, 1 = partial or ambiguous, 2 = defined, 3 = defined and verified.

## `survey.csv`
One row per consenting respondent. **Normally produced by the ingest, not typed
by hand:** run `python src/survey_ingest.py <raw Forms export>.csv`, which
writes this file to `data/raw/`.

| column | type | values |
|---|---|---|
| `respondent_id` | str | `AGT-001`, `AGT-002`, … assigned in ascending timestamp order |
| `q1, q3, q4, q5, q8, q9, q10` | int | 1-5 five-point Likert (intensity) |
| `q2, q6` | int | Yes = 2, Unsure = 1, No = 0 |
| `q7` | int | Never = 1, Rarely = 2, Sometimes = 3, Often = 4; "Don't know" → blank |
| `q13` | int | < 2 years = 1, 2-5 = 2, 6-10 = 3, > 10 = 4 |
| `q14` | int | Yes = 1, No = 0 |
| `q11, q12, q15` | str | free text, kept verbatim |

The coding tables are defined once, at the top of `src/survey_ingest.py`, and
nothing else in the pipeline recodes survey answers.

## `survey_export.csv` (raw Google Forms export)
The *input* to the ingest, not an analysis file. Column order is what matters,
because the headers are full question sentences and change whenever the form is
edited — the ingest maps items **by position**:

| position | content |
|---|---|
| 1 | `Timestamp` (added by Forms; required, and used for the respondent ordering) |
| 2 | consent question (Q0); rows not answering "Yes" are reported and dropped |
| 3-17 | the 15 items, in questionnaire order, as category labels or 1-5 integers |

A synthetic example is committed at `synthetic/data/survey_export.csv`.

## `benchmark.csv`
One row per domain x port (10 x 4 = 40 rows).

| column | type | range / values |
|---|---|---|
| `domain_code` | str | `D1` .. `D10` |
| `port` | str | `CPA`, `Singapore`, `LosAngelesLongBeach`, `Australia` |
| `score` | float | 0-3 |
