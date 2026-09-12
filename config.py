"""
config.py -- central configuration for the POSRRI analysis pipeline.

Port Oil Spill Regulatory Readiness Index (POSRRI), Chattogram Port, Bangladesh.

Everything that a user might reasonably want to change (paths, random seed,
methodological thresholds, figure styling) lives here so that the analysis
modules in ``src/`` stay free of magic numbers.

The module is import-safe from anywhere (local shell, pytest, Google Colab):
it never creates directories at import time except the output tree, which is
cheap and idempotent.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
TEMPLATE_DIR = DATA_DIR / "templates"
RAW_DATA_DIR = DATA_DIR / "raw"

SYNTHETIC_DIR = ROOT_DIR / "synthetic"
SYNTHETIC_DATA_DIR = SYNTHETIC_DIR / "data"

OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

#: Files the pipeline expects to find in whichever data directory is active.
DATA_FILES = {
    "delphi": "delphi_ratings.csv",
    "ahp": "ahp_pairwise.csv",
    "scores": "scores.csv",
    "survey": "survey.csv",
    "survey_export": "survey_export.csv",   # raw Google Forms export
    "benchmark": "benchmark.csv",
}


def resolve_paths(use_synthetic: bool = True,
                  data_dir: "str | os.PathLike | None" = None,
                  output_dir: "str | os.PathLike | None" = None) -> dict:
    """Return the input/output directories the pipeline should use.

    Parameters
    ----------
    use_synthetic
        ``True``  -> read from ``synthetic/data`` (the committed, reproducible
                     demonstration dataset).
        ``False`` -> read from ``data/raw`` (real, git-ignored participant data).
    data_dir, output_dir
        Explicit overrides. Google Colab notebooks pass a Google Drive path
        here so that inputs and outputs survive the runtime being recycled.

    Returns
    -------
    dict with keys ``data_dir``, ``output_dir``, ``figure_dir``, ``table_dir``.
    Output directories are created if they do not exist.
    """
    if data_dir is not None:
        in_dir = Path(data_dir)
    else:
        in_dir = SYNTHETIC_DATA_DIR if use_synthetic else RAW_DATA_DIR

    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    for d in (fig_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "data_dir": in_dir,
        "output_dir": out_dir,
        "figure_dir": fig_dir,
        "table_dir": tab_dir,
    }


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

#: Master seed. Every stochastic step derives its generator from this value,
#: so the whole synthetic dataset is bit-for-bit reproducible.
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Delphi parameters
# ---------------------------------------------------------------------------

#: Relevance is rated 1-9; 7-9 counts as "relevant" (Zamanzadeh et al. 2015).
RELEVANCE_SCALE = (1, 9)
RELEVANCE_HIGH_MIN = 7

#: Consensus rule: (>=80% of experts rating 7-9) OR (median >=7 AND IQR <=2).
CONSENSUS_PCT_THRESHOLD = 0.80
CONSENSUS_MEDIAN_MIN = 7.0
CONSENSUS_IQR_MAX = 2.0

#: Content validity. I-CVI retention cut-off and S-CVI/Ave reporting target.
ICVI_RETAIN_THRESHOLD = 0.78
SCVI_TARGET = 0.90

#: Rounds present in the Delphi design.
DELPHI_ROUNDS = (1, 2)

#: Auxiliary rating scales (reported, not used for retention).
CLARITY_SCALE = (1, 5)
FEASIBILITY_SCALE = (1, 5)

# ---------------------------------------------------------------------------
# AHP parameters
# ---------------------------------------------------------------------------

#: Saaty's consistency-ratio acceptance threshold.
CR_THRESHOLD = 0.10

#: Admissible Saaty judgements: 1..9 and their reciprocals.
SAATY_INTEGERS = (1, 2, 3, 4, 5, 6, 7, 8, 9)

#: AHP hierarchy levels, ordered from the goal downwards.
AHP_LEVELS = ("pillar", "domain", "indicator")

# ---------------------------------------------------------------------------
# Scoring parameters
# ---------------------------------------------------------------------------

#: The locked 0-3 scoring rubric. This is the single source of truth for the
#: wording: figures, tables and documentation all read it from here, so the
#: level descriptions cannot drift apart between the code and the manuscript.
SCORE_RUBRIC = {
    0: "absent",
    1: "partial or ambiguous",
    2: "defined",
    3: "defined and verified",
}

SCORE_SCALE = (0, 3)
MAX_INDICATOR_SCORE = 3


def rubric_label(score: int, with_number: bool = True) -> str:
    """Render one rubric level, e.g. ``"2  defined"``."""
    text = SCORE_RUBRIC[int(score)]
    return f"{int(score)}  {text}" if with_number else text

#: Interpretation bands for the 0-100 index (used in tables and figure legends).
READINESS_BANDS = (
    (0.0, 25.0, "Critical gap"),
    (25.0, 50.0, "Emerging"),
    (50.0, 75.0, "Developing"),
    (75.0, 100.0001, "Advanced"),
)

# ---------------------------------------------------------------------------
# Figure styling (publication grade)
# ---------------------------------------------------------------------------

FIGURE_DPI = 300
FIGURE_FORMATS = ("png", "pdf")          # raster for review, vector for print
FIGURE_FONT_FAMILY = "DejaVu Sans"       # metric-compatible, always available
FIGURE_BASE_FONTSIZE = 9

#: Okabe-Ito colourblind-safe qualitative palette (Okabe & Ito 2008).
#: Ordered so that adjacent entries also differ in greyscale luminance.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#999999",
}

#: Default ordered cycle for categorical series.
PALETTE = [
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["orange"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["black"],
    OKABE_ITO["grey"],
]

#: Fixed colours for the three pillars (stable across every figure).
PILLAR_COLOURS = {
    "A": OKABE_ITO["blue"],
    "B": OKABE_ITO["vermillion"],
    "C": OKABE_ITO["bluish_green"],
}

#: Marker/linestyle cycles so figures stay legible in greyscale print.
LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
