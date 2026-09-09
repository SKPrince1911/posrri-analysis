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

Rubric: 0 = absent, 1 = partial, 2 = substantial, 3 = fully met and evidenced.

## `survey.csv`
One row per respondent; `q1`..`q15` are Likert items (1-5).

## `benchmark.csv`
One row per domain x port (10 x 4 = 40 rows).

| column | type | range / values |
|---|---|---|
| `domain_code` | str | `D1` .. `D10` |
| `port` | str | `CPA`, `Singapore`, `LosAngelesLongBeach`, `Australia` |
| `score` | float | 0-3 |
