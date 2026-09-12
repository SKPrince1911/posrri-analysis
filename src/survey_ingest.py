"""
src/survey_ingest.py -- convert a raw Google Forms export into survey.csv.

A Google Forms response export (downloaded directly, or taken from the linked
Google Sheet and exported as CSV) has three properties that make it unsafe to
read straight into the analysis:

1. the first column is a ``Timestamp`` added by Forms, and the second is the
   consent question (Q0), neither of which belongs in the analysis file;
2. the headers are the full question sentences, and they change whenever anyone
   edits the wording of a question in the form -- so columns must be matched
   **by position**, never by header text;
3. answers are category labels ("Unsure", "Don't know", "6-10 years"), not the
   numeric codes the analysis needs.

``load_survey_export`` handles all three and writes the project's
``survey.csv`` schema (``respondent_id, q1 ... q15``).

Where the output goes
---------------------
The default target is ``data/raw/survey.csv``. A Forms export contains real
participant responses, so it belongs in the git-ignored raw directory -- never
in ``synthetic/data/``, which holds the committed demonstration dataset. Pass
``data_dir`` explicitly to override.

Coding is defined by the mapping tables at the top of this module. Nothing is
recoded anywhere else in the pipeline, so those tables are the whole
specification of how a category label becomes a number.

Usage
-----
    python src/survey_ingest.py path/to/forms_export.csv
    python src/survey_ingest.py export.csv --data-dir data/raw
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# The instrument: 15 analysed items plus Timestamp and the Q0 consent question
# ---------------------------------------------------------------------------

N_ITEMS = 15
ITEMS: List[str] = [f"q{k}" for k in range(1, N_ITEMS + 1)]

#: Five-point Likert confidence/adequacy items, kept as integers 1-5.
LIKERT_ITEMS = ["q1", "q3", "q4", "q5", "q8", "q9", "q10"]

#: Free-text items, kept verbatim.
TEXT_ITEMS = ["q11", "q12", "q15"]

#: Short labels for tables and figure axes. The full question sentences live in
#: the form itself; these are the abbreviations used for reporting.
ITEM_LABELS: Dict[str, str] = {
    "q1": "Plan would work in a Tier 2 spill",
    "q2": "Aware of designated On-Scene Commander",
    "q3": "Own roles and responsibilities are clear",
    "q4": "Response equipment is adequate",
    "q5": "Inter-agency coordination is effective",
    "q6": "Given access to the contingency plan",
    "q7": "Frequency of exercise participation",
    "q8": "Confidence in the notification chain",
    "q9": "Funding for preparedness is adequate",
    "q10": "Lessons from incidents are acted upon",
    "q11": "Greatest barrier (free text)",
    "q12": "Most valuable single change (free text)",
    "q13": "Years of relevant experience",
    "q14": "Formal OPRC training received",
    "q15": "Organisation (free text)",
}

#: Response labels for the shared five-point scale. The items are worded
#: "how confident / how clear / how adequate / how effective", so the anchors
#: are intensity levels rather than agreement levels.
LIKERT_LABELS = {
    1: "Not at all",
    2: "A little",
    3: "Moderately",
    4: "Considerably",
    5: "Completely",
}

# ---------------------------------------------------------------------------
# Recoding tables -- the documented, explicit specification
# ---------------------------------------------------------------------------
#
# Keys are the exact option labels as they appear in the form. Matching is
# case-insensitive and tolerant of whitespace, curly apostrophes and en-dashes
# (Google Sheets rewrites all three), but NOT of different wording: an
# unrecognised label raises rather than being silently dropped to missing.

#: Awareness/access items. "Unsure" is a real middle state, not missing data:
#: a respondent who is unsure whether a plan exists is evidence about how well
#: the plan is communicated, so it scores 1 rather than NA.
YES_UNSURE_NO = {"Yes": 2, "Unsure": 1, "No": 0}

#: Exercise frequency. "Don't know" is NOT a frequency and cannot be placed on
#: the Never..Often ordinal scale, so it becomes missing.
FREQUENCY = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4,
             "Don't know": pd.NA}

#: Binary training item.
YES_NO = {"Yes": 1, "No": 0}

#: Experience bands, ordered.
EXPERIENCE_BANDS = {"Less than 2 years": 1, "2-5 years": 2,
                    "6-10 years": 3, "More than 10 years": 4}

#: item -> mapping. Items absent from this dict are Likert or free text.
RECODE: Dict[str, Dict[str, object]] = {
    "q2": YES_UNSURE_NO,
    "q6": YES_UNSURE_NO,
    "q7": FREQUENCY,
    "q13": EXPERIENCE_BANDS,
    "q14": YES_NO,
}

#: Allowed coded values per item, used here and re-asserted by the validator.
ALLOWED_VALUES: Dict[str, set] = {
    **{item: {1, 2, 3, 4, 5} for item in LIKERT_ITEMS},
    "q2": {0, 1, 2},
    "q6": {0, 1, 2},
    "q7": {1, 2, 3, 4},
    "q13": {1, 2, 3, 4},
    "q14": {0, 1},
}

#: Consent (Q0). Only "Yes" is retained; "No" rows are reported and dropped.
CONSENT_YES = {"Yes", "I consent", "I agree"}
CONSENT_NO = {"No", "I do not consent", "I disagree"}

#: Header patterns used only to find the Timestamp column, which Forms always
#: names itself. Every analysed item is located by position, not by header.
TIMESTAMP_PATTERN = re.compile(r"time\s*stamp|timestamp|date", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _norm(value) -> str:
    """Normalise a category label for matching.

    Unifies the transformations a label picks up on its way through Forms,
    Sheets and CSV: surrounding whitespace, repeated spaces, curly apostrophes
    and en/em dashes. Case is folded last.
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def _normalised_map(mapping: Dict[str, object]) -> Dict[str, object]:
    return {_norm(k): v for k, v in mapping.items()}


# ---------------------------------------------------------------------------
# Column location
# ---------------------------------------------------------------------------

def _locate_timestamp(columns: List[str]) -> int:
    """Index of the Timestamp column, which the respondent ordering needs."""
    for k, name in enumerate(columns):
        if TIMESTAMP_PATTERN.search(str(name)):
            return k
    raise ValueError(
        "no Timestamp column found in the export. respondent_id must be "
        "assigned in ascending timestamp order, so the Timestamp column "
        "cannot be omitted. Re-export from Google Forms (or from the linked "
        f"Sheet) with the Timestamp column intact. Headers seen: {list(columns)[:4]}"
    )


def _parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse Forms timestamps, trying the locale variants Forms emits."""
    parsed = pd.to_datetime(values, errors="coerce", format="mixed", dayfirst=False)
    if parsed.isna().any():
        # Retry day-first for exports made under a non-US locale.
        retry = pd.to_datetime(values, errors="coerce", format="mixed", dayfirst=True)
        parsed = parsed.fillna(retry)
    if parsed.isna().any():
        bad = values[parsed.isna()].head(3).tolist()
        raise ValueError(
            f"{int(parsed.isna().sum())} timestamp(s) could not be parsed, "
            f"e.g. {bad}. Fix these in the export before ingesting."
        )
    return parsed


# ---------------------------------------------------------------------------
# Recoding and validation
# ---------------------------------------------------------------------------

def _recode_categorical(series: pd.Series, item: str) -> pd.Series:
    """Apply ``RECODE[item]``, raising on any unrecognised label."""
    lookup = _normalised_map(RECODE[item])
    out, offenders = [], []
    for raw in series:
        key = _norm(raw)
        if key == "":
            out.append(pd.NA)                      # genuinely blank answer
            continue
        if key not in lookup:
            offenders.append(str(raw))
            out.append(pd.NA)
            continue
        out.append(lookup[key])

    if offenders:
        unique = sorted(set(offenders))
        raise ValueError(
            f"{item}: unexpected category value(s) {unique!r}. "
            f"Expected one of {list(RECODE[item])!r}. Either fix the export or "
            f"add the label to the mapping table in src/survey_ingest.py."
        )
    return pd.Series(out, index=series.index, dtype="Int64")


def _coerce_likert(series: pd.Series, item: str) -> pd.Series:
    """Coerce a Likert column to Int64 and range-check it to 1-5."""
    numeric = pd.to_numeric(series, errors="coerce")
    unparsed = series[numeric.isna() & (series.map(_norm) != "")]
    if len(unparsed):
        raise ValueError(
            f"{item}: non-numeric value(s) on a 1-5 Likert item: "
            f"{sorted(set(unparsed.astype(str)))[:5]!r}"
        )
    rounded = numeric.round()
    if not np.allclose(numeric.dropna(), rounded.dropna()):
        raise ValueError(f"{item}: non-integer Likert value(s) present")

    out = rounded.astype("Int64")
    bad = out.dropna()
    bad = bad[~bad.isin(ALLOWED_VALUES[item])]
    if len(bad):
        raise ValueError(
            f"{item}: {len(bad)} Likert value(s) outside 1-5: "
            f"{sorted(set(bad.tolist()))!r}"
        )
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_survey_export(path: "str | Path",
                       data_dir: "str | Path | None" = None,
                       write: bool = True,
                       verbose: bool = True) -> pd.DataFrame:
    """Convert a raw Google Forms export into the project ``survey.csv`` schema.

    Parameters
    ----------
    path
        The raw export (CSV downloaded from Forms, or the linked Sheet saved
        as CSV).
    data_dir
        Directory to write ``survey.csv`` into. Defaults to
        ``config.RAW_DATA_DIR`` (``data/raw/``), which is git-ignored because a
        real export contains participant responses.
    write
        Set ``False`` to convert without writing (used by the tests).
    verbose
        Print the summary: respondents kept, rows dropped, missing counts.

    Returns
    -------
    DataFrame with columns ``respondent_id, q1 ... q15``, one row per
    consenting respondent, ordered by ascending timestamp.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"survey export not found: {path}")

    raw = pd.read_csv(path, dtype=str)
    if raw.empty:
        raise ValueError(f"survey export {path} has no data rows")

    columns = list(raw.columns)
    ts_idx = _locate_timestamp(columns)

    # Q0 is the column immediately after Timestamp -- located by position,
    # because the consent wording is a full sentence that may be edited.
    consent_idx = ts_idx + 1
    if consent_idx >= len(columns):
        raise ValueError(
            "the export has a Timestamp column but no consent question after "
            "it; expected Timestamp, consent (Q0), then the 15 items."
        )

    timestamps = _parse_timestamps(raw.iloc[:, ts_idx])
    consent_raw = raw.iloc[:, consent_idx]

    # ---- consent: check, report, drop --------------------------------------
    yes_keys = {_norm(v) for v in CONSENT_YES}
    no_keys = {_norm(v) for v in CONSENT_NO}
    normalised = consent_raw.map(_norm)

    # Report the labels exactly as they appear in the file, not normalised:
    # whoever has to fix the export needs the string they will be searching for.
    known = yes_keys | no_keys
    unknown = sorted({str(raw_value) for raw_value, key
                      in zip(consent_raw, normalised)
                      if key and key not in known})
    if unknown:
        raise ValueError(
            f"unexpected consent value(s) {unknown!r} in column "
            f"{columns[consent_idx]!r}. Expected one of "
            f"{sorted(CONSENT_YES | CONSENT_NO)!r}."
        )

    consented = normalised.isin(yes_keys)
    n_dropped = int((~consented).sum())
    dropped_rows = [
        {"row": int(i) + 2,                         # +2: header plus 1-based
         "timestamp": str(timestamps.iloc[i]),
         "consent": str(consent_raw.iloc[i]) if _norm(consent_raw.iloc[i]) else "(blank)"}
        for i in np.flatnonzero(~consented.to_numpy())
    ]

    frame = raw.loc[consented.to_numpy()].copy()
    kept_timestamps = timestamps.loc[consented.to_numpy()]
    if frame.empty:
        raise ValueError(
            f"every one of the {len(raw)} response(s) failed the consent check; "
            f"nothing left to analyse."
        )

    # The invariant the caller asked for: after dropping, every retained row
    # consented "Yes". Raised rather than `assert`, so it still holds under -O.
    if not frame.iloc[:, consent_idx].map(_norm).isin(yes_keys).all():
        raise RuntimeError(
            "internal error: a non-consenting row survived the consent filter"
        )

    # ---- order by timestamp, then assign identifiers -----------------------
    order = np.argsort(kept_timestamps.to_numpy(), kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    ordered_timestamps = kept_timestamps.to_numpy()[order]
    respondent_id = [f"AGT-{k:03d}" for k in range(1, len(frame) + 1)]

    # ---- items, BY POSITION ------------------------------------------------
    item_frame = frame.drop(columns=[columns[ts_idx], columns[consent_idx]])
    source_headers = list(item_frame.columns)
    if len(source_headers) != N_ITEMS:
        raise ValueError(
            f"after dropping Timestamp and consent, the export has "
            f"{len(source_headers)} column(s), expected exactly {N_ITEMS}. "
            f"Columns found: {source_headers}"
        )

    out = pd.DataFrame({"respondent_id": respondent_id})
    for position, item in enumerate(ITEMS):
        source = item_frame.iloc[:, position]
        if item in RECODE:
            out[item] = _recode_categorical(source, item).to_numpy()
        elif item in LIKERT_ITEMS:
            out[item] = _coerce_likert(source, item).to_numpy()
        else:                                       # free text, kept verbatim
            out[item] = source.astype("string").str.strip().to_numpy()

    # Restore nullable integer dtypes lost by the round-trip through numpy.
    for item in ITEMS:
        if item in LIKERT_ITEMS or item in RECODE:
            out[item] = out[item].astype("Int64")

    # ---- final validation --------------------------------------------------
    for item, allowed in ALLOWED_VALUES.items():
        present = set(out[item].dropna().astype(int).tolist())
        unexpected = present - allowed
        if unexpected:
            raise ValueError(
                f"{item}: coded value(s) {sorted(unexpected)!r} outside the "
                f"allowed set {sorted(allowed)!r}"
            )
    if out["respondent_id"].duplicated().any():
        raise RuntimeError("internal error: duplicate respondent_id generated")

    missing = {item: int(out[item].isna().sum()) if item not in TEXT_ITEMS
               else int((out[item].fillna("") == "").sum())
               for item in ITEMS}

    # ---- write and report --------------------------------------------------
    target_dir = Path(data_dir) if data_dir is not None else config.RAW_DATA_DIR
    target = target_dir / config.DATA_FILES["survey"]
    if write:
        target_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(target, index=False)

    if verbose:
        print(f"Survey ingest: {path}")
        print("-" * 66)
        print(f"  Rows in export              {len(raw)}")
        print(f"  Dropped (did not consent)   {n_dropped}")
        for row in dropped_rows:
            print(f"      row {row['row']}: consent = {row['consent']!r} "
                  f"({row['timestamp']})")
        print(f"  Respondents retained        {len(out)}")
        if len(out):
            print(f"  respondent_id range         {out['respondent_id'].iloc[0]} .. "
                  f"{out['respondent_id'].iloc[-1]}")
            print(f"  Timestamp range             {ordered_timestamps[0]} .. "
                  f"{ordered_timestamps[-1]}")
        total_missing = sum(missing.values())
        print(f"  Missing values (total)      {total_missing}")
        if total_missing:
            for item, count in missing.items():
                if count:
                    kind = "blank text" if item in TEXT_ITEMS else "NA"
                    print(f"      {item:<4} {count:>3}  ({kind})")
        else:
            print("      none")
        if write:
            print(f"  Written to                  {target}")
        print()

    return out


def read_survey(path: "str | Path") -> pd.DataFrame:
    """Read a cleaned ``survey.csv`` back with the correct dtypes.

    ``pd.read_csv`` alone returns the coded items as floats whenever a column
    contains a missing value, which then propagates ".0" into every table and
    figure label. This restores nullable integers for the coded items and
    strings for the free-text ones, and re-checks the schema on the way in.
    """
    path = Path(path)
    frame = pd.read_csv(path)

    expected = ["respondent_id"] + ITEMS
    if list(frame.columns) != expected:
        raise ValueError(
            f"{path} does not match the survey schema.\n"
            f"  expected: {expected}\n  found:    {list(frame.columns)}"
        )

    frame["respondent_id"] = frame["respondent_id"].astype("string")
    for item in ITEMS:
        if item in TEXT_ITEMS:
            frame[item] = frame[item].astype("string")
        else:
            frame[item] = pd.to_numeric(frame[item], errors="coerce").astype("Int64")
    return frame


def category_labels(item: str) -> Dict[int, str]:
    """``{code: label}`` for a coded item, for frequency tables and legends."""
    if item in LIKERT_ITEMS:
        return dict(LIKERT_LABELS)
    if item not in RECODE:
        raise KeyError(f"{item} is not a coded item")
    out = {}
    for label, code in RECODE[item].items():
        if code is pd.NA:                    # e.g. "Don't know" -> missing
            continue
        out[int(code)] = label
    return dict(sorted(out.items()))


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a raw Google Forms export into survey.csv.")
    parser.add_argument("path", help="raw Forms/Sheets CSV export")
    parser.add_argument("--data-dir", default=None,
                        help="output directory (default data/raw)")
    parser.add_argument("--no-write", action="store_true",
                        help="convert and report without writing the file")
    args = parser.parse_args()

    frame = load_survey_export(args.path, data_dir=args.data_dir,
                               write=not args.no_write)
    print(frame.head(8).to_string(index=False))


if __name__ == "__main__":
    _cli()
