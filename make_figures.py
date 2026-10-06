"""
make_figures.py -- render every POSRRI figure from the study data.

One command to regenerate the full figure set after the data or the indicator
set changes:

    python make_figures.py                 # read data/raw, write outputs/
    python make_figures.py --data-dir path/to/data --output-dir path/to/out
    python make_figures.py --list          # names the figures and exits

Every figure carries plain-language indicator names. Indicator codes (D1.1,
D7.4 ...) appear nowhere on an axis, a tick or a cell: a reader should not need
the codebook open to read a figure.

Figures written, each as a 300 dpi PNG and a vector PDF:

    fig1_radar_benchmark          domain readiness, CPA against the comparators
    fig2_domain_scores_weights    domain scores grouped by pillar, AHP weights
    fig3_indicator_heatmap        every indicator score, grouped by domain
    fig4_implementation_gap       documentary versus field evidence
    fig5_delphi_ratings           every expert rating, with the agreement rule
    fig6_sensitivity_scatter      AHP versus equal weights
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from src import viz  # noqa: E402
from src.ahp import run_ahp  # noqa: E402
from src.delphi import run_delphi  # noqa: E402
from src.scoring import benchmark_matrix, score_index  # noqa: E402
from src.sensitivity import run_sensitivity  # noqa: E402


def _require(path: Path, what: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}.\n"
            f"The pipeline reads the study data from {path.parent}. Put the "
            f"study CSVs there (see data/templates/README.md for the schemas)."
        )
    return pd.read_csv(path)


def make_figures(data_dir: "Path | str | None" = None,
                 output_dir: "Path | str | None" = None,
                 verbose: bool = True) -> Dict[str, List[Path]]:
    """Render the full figure set. Returns ``{stem: [paths written]}``."""
    paths = config.resolve_paths(data_dir=data_dir, output_dir=output_dir)
    data_dir = Path(paths["data_dir"])
    figure_dir = paths["figure_dir"]

    delphi_raw = _require(data_dir / config.DATA_FILES["delphi"], "delphi_ratings.csv")
    ahp_raw = _require(data_dir / config.DATA_FILES["ahp"], "ahp_pairwise.csv")
    scores_raw = _require(data_dir / config.DATA_FILES["scores"], "scores.csv")
    benchmark_raw = _require(data_dir / config.DATA_FILES["benchmark"], "benchmark.csv")

    delphi_res = run_delphi(delphi_raw)
    ahp_res = run_ahp(ahp_raw)
    score_res = score_index(scores_raw, ahp_res["weights"])
    sens_res = run_sensitivity(scores_raw, ahp_res["weights"])

    written = viz.make_all_figures(
        scoring_result=score_res,
        delphi_result=delphi_res,
        sensitivity_result=sens_res,
        benchmark=benchmark_raw,
        ahp_domain_weights=ahp_res["domain_weights"],
        figure_dir=figure_dir,
    )

    if verbose:
        print(f"Data   : {data_dir}")
        print(f"Figures: {figure_dir}")
        print("-" * 68)
        for stem, files in written.items():
            print(f"  {stem:<30} {', '.join(p.suffix.lstrip('.') for p in files)}")
        print(f"\n{len(written)} figures, "
              f"{sum(len(v) for v in written.values())} files.")
    return written


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None,
                        help="directory holding the study CSVs (default data/raw)")
    parser.add_argument("--output-dir", default=None,
                        help="directory for outputs/figures (default outputs/)")
    parser.add_argument("--list", action="store_true",
                        help="list the figures that would be written, and exit")
    args = parser.parse_args()

    if args.list:
        for stem in viz.FIGURE_STEMS:
            print(stem)
        return

    try:
        make_figures(data_dir=args.data_dir, output_dir=args.output_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    _cli()
