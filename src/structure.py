"""
src/structure.py -- the fixed POSRRI index hierarchy.

The Port Oil Spill Regulatory Readiness Index is a three-level construct:

    3 pillars  ->  10 domains  ->  50 indicators   (5 indicators per domain)

This module is the single source of truth for that hierarchy. Every other
module derives its row order, grouping and validation from here, so the
structure can never silently drift between the Delphi, AHP, scoring and
figure stages.

Run ``python src/structure.py`` to print a summary and self-validate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Level 1 -- pillars
# ---------------------------------------------------------------------------

PILLARS: Dict[str, str] = {
    "A": "Governance & legal foundation",
    "B": "Operational preparedness & response",
    "C": "Resilience, financing & continuity",
}

#: Which domains sit under which pillar (order is meaningful and fixed).
PILLAR_DOMAINS: Dict[str, List[str]] = {
    "A": ["D1", "D2", "D9"],
    "B": ["D3", "D4", "D5", "D6"],
    "C": ["D7", "D8", "D10"],
}

# ---------------------------------------------------------------------------
# Level 2 -- domains
# ---------------------------------------------------------------------------

DOMAINS: Dict[str, str] = {
    "D1": "Legal & Regulatory Basis",
    "D2": "Designated Authority & Command Structure",
    "D3": "Notification & Reporting",
    "D4": "Tiered Response Arrangements",
    "D5": "Resources & Equipment",
    "D6": "Training & Exercises",
    "D7": "Funding & Liability Arrangements",
    "D8": "Plan Currency & Documentation",
    "D9": "Stakeholder Coordination",
    "D10": "Monitoring, Review & After-Action Learning",
}

#: Compact domain labels for figure axes, where the full names do not fit.
DOMAIN_SHORT: Dict[str, str] = {
    "D1": "Legal basis",
    "D2": "Authority & command",
    "D3": "Notification",
    "D4": "Tiered response",
    "D5": "Resources & equipment",
    "D6": "Training & exercises",
    "D7": "Funding & liability",
    "D8": "Plan currency",
    "D9": "Coordination",
    "D10": "Monitoring & learning",
}

#: Canonical domain order D1..D10 (not the pillar-grouped order).
DOMAIN_ORDER: List[str] = [f"D{i}" for i in range(1, 11)]

# ---------------------------------------------------------------------------
# Level 3 -- indicators (5 per domain, 50 in total)
# ---------------------------------------------------------------------------

INDICATOR_NAMES: Dict[str, List[str]] = {
    "D1": [
        "Explicit legal mandate for port oil spill preparedness and response",
        "Formal alignment of the port with NOSCOP",
        "Enforcement provisions actually applied",
        "Legal basis reflects OPRC and MARPOL",
        "Liability and compensation enforceable",
    ],
    "D2": [
        "Single competent authority designated",
        "Named On-Scene Commander (Harbour officer)",
        "Incident management system with defined roles",
        "Activation thresholds and escalation triggers",
        "Unified-command interface with Coast Guard, Navy and DoE",
    ],
    "D3": [
        "Defined first point of notification",
        "Time-bound notification chain",
        "Standard reporting format",
        "Redundant communications (VHF) mandated",
        "Reporting duties on vessels, terminals and facilities",
    ],
    "D4": [
        "Tier 1/2/3 defined with thresholds",
        "Port responsibilities defined for each tier",
        "Escalation route to national and regional resources",
        "Response options chosen by net environmental benefit analysis",
        "In-house Tier 1 response capability",
    ],
    "D5": [
        "Current documented equipment inventory",
        "Equipment mobilisable within set times",
        "Reception facilities for oily waste",
        "Mutual-aid and contractor arrangements",
        "Recovered-oil storage and waste management",
    ],
    "D6": [
        "Recurring training programme",
        "Scheduled exercise and drill cadence",
        "Multi-agency exercise participation",
        "Lessons from incidents built into plans",
        "Responder health and safety competencies",
    ],
    "D7": [
        "Funding beyond initial response costs",
        "Dedicated response fund",
        "Cost recovery from the polluter",
        "Applicable CLC/Bunker liability regime",
        "Financial security or insurance required of port users",
    ],
    "D8": [
        "Current, complete port contingency plan",
        "Mandatory periodic review of the plan",
        "Documented risk assessment",
        "Environmental sensitivity mapping",
        "Version control and controlled distribution of the plan",
    ],
    "D9": [
        "Inter-agency roles formally defined",
        "Terminal operators and agents integrated",
        "Regional (SACEP) cooperation put into operation",
        "Community engagement and awareness",
        "Regular joint planning meetings",
    ],
    "D10": [
        "Structured after-action review",
        "After-action reports within a set timeframe",
        "Mechanism that updates plans from lessons",
        "Records of drills, incidents and performance",
        "Periodic readiness self-assessment (RETOS or similar)",
    ],
}

#: Ordered list of the 50 indicator codes, D1.1 .. D10.5.
INDICATORS: List[str] = [
    f"{d}.{k}" for d in DOMAIN_ORDER for k in range(1, 6)
]

#: indicator_code -> short descriptive name.
INDICATOR_LABELS: Dict[str, str] = {
    f"{d}.{k}": INDICATOR_NAMES[d][k - 1]
    for d in DOMAIN_ORDER
    for k in range(1, 6)
}

N_PILLARS = len(PILLARS)
N_DOMAINS = len(DOMAINS)
N_INDICATORS = len(INDICATORS)
INDICATORS_PER_DOMAIN = 5

# ---------------------------------------------------------------------------
# Saaty's Random Index (Saaty 1980), keyed by matrix order n
# ---------------------------------------------------------------------------

RI: Dict[int, float] = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}

# ---------------------------------------------------------------------------
# Benchmark ports
# ---------------------------------------------------------------------------

#: CPA = Chattogram Port Authority (the case study); the rest are comparators.
BENCHMARK_PORTS: List[str] = [
    "CPA",
    "Singapore",
    "LosAngelesLongBeach",
    "Australia",
]

#: Display labels used in figures and tables.
PORT_LABELS: Dict[str, str] = {
    "CPA": "Chattogram (CPA)",
    "Singapore": "Singapore",
    "LosAngelesLongBeach": "Los Angeles / Long Beach",
    "Australia": "Australia (national)",
}

# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

#: domain_code -> pillar_code
DOMAIN_TO_PILLAR: Dict[str, str] = {
    dom: pil for pil, doms in PILLAR_DOMAINS.items() for dom in doms
}

#: indicator_code -> domain_code
INDICATOR_TO_DOMAIN: Dict[str, str] = {
    ind: ind.split(".")[0] for ind in INDICATORS
}

#: indicator_code -> pillar_code
INDICATOR_TO_PILLAR: Dict[str, str] = {
    ind: DOMAIN_TO_PILLAR[dom] for ind, dom in INDICATOR_TO_DOMAIN.items()
}

#: domain_code -> ordered list of its 5 indicator codes
DOMAIN_INDICATORS: Dict[str, List[str]] = {
    dom: [f"{dom}.{k}" for k in range(1, 6)] for dom in DOMAIN_ORDER
}

#: pillar_code -> ordered list of all indicator codes beneath it
PILLAR_INDICATORS: Dict[str, List[str]] = {
    pil: [ind for dom in doms for ind in DOMAIN_INDICATORS[dom]]
    for pil, doms in PILLAR_DOMAINS.items()
}


def domain_of(indicator_code: str) -> str:
    """Return the domain code (e.g. ``'D3'``) owning ``indicator_code``."""
    try:
        return INDICATOR_TO_DOMAIN[indicator_code]
    except KeyError as exc:
        raise KeyError(f"Unknown indicator code: {indicator_code!r}") from exc


def pillar_of(code: str) -> str:
    """Return the pillar code for a domain code or an indicator code."""
    if code in DOMAIN_TO_PILLAR:
        return DOMAIN_TO_PILLAR[code]
    if code in INDICATOR_TO_PILLAR:
        return INDICATOR_TO_PILLAR[code]
    raise KeyError(f"Unknown domain or indicator code: {code!r}")


def random_index(n: int) -> float:
    """Saaty Random Index for a pairwise matrix of order ``n``."""
    if n not in RI:
        raise KeyError(
            f"No Random Index tabulated for n={n}; POSRRI matrices are of "
            f"order {sorted(RI)}."
        )
    return RI[n]


def readiness_band(score_0_100: float) -> str:
    """Map a 0-100 index score onto its interpretation band."""
    for low, high, label in config.READINESS_BANDS:
        if low <= score_0_100 < high:
            return label
    return "Out of range"


def structure_frame() -> pd.DataFrame:
    """Return the full hierarchy as a tidy 50-row DataFrame.

    Columns: pillar_code, pillar_name, domain_code, domain_name,
    indicator_code, indicator_name.
    """
    rows = []
    for pil, doms in PILLAR_DOMAINS.items():
        for dom in doms:
            for ind in DOMAIN_INDICATORS[dom]:
                rows.append({
                    "pillar_code": pil,
                    "pillar_name": PILLARS[pil],
                    "domain_code": dom,
                    "domain_name": DOMAINS[dom],
                    "indicator_code": ind,
                    "indicator_name": INDICATOR_LABELS[ind],
                })
    frame = pd.DataFrame(rows)
    # Present in canonical indicator order D1.1 .. D10.5.
    frame["_order"] = frame["indicator_code"].map(
        {c: i for i, c in enumerate(INDICATORS)}
    )
    return (frame.sort_values("_order")
                 .drop(columns="_order")
                 .reset_index(drop=True))


def validate_structure() -> Tuple[bool, List[str]]:
    """Check the hierarchy is internally consistent.

    Returns ``(ok, problems)``; ``problems`` is empty when ``ok`` is True.
    """
    problems: List[str] = []

    if len(PILLARS) != 3:
        problems.append(f"expected 3 pillars, found {len(PILLARS)}")
    if len(DOMAINS) != 10:
        problems.append(f"expected 10 domains, found {len(DOMAINS)}")
    if len(INDICATORS) != 50:
        problems.append(f"expected 50 indicators, found {len(INDICATORS)}")

    assigned = [d for doms in PILLAR_DOMAINS.values() for d in doms]
    if sorted(assigned) != sorted(DOMAINS):
        problems.append("PILLAR_DOMAINS does not partition the 10 domains")
    if len(assigned) != len(set(assigned)):
        problems.append("a domain is assigned to more than one pillar")

    for dom in DOMAIN_ORDER:
        n = len(INDICATOR_NAMES.get(dom, []))
        if n != INDICATORS_PER_DOMAIN:
            problems.append(f"{dom} has {n} indicators, expected 5")

    if len(set(INDICATORS)) != len(INDICATORS):
        problems.append("duplicate indicator codes")

    expected_ri_orders = {3, 4, 5}
    missing = expected_ri_orders - set(RI)
    if missing:
        problems.append(f"Random Index missing orders {sorted(missing)}")

    if BENCHMARK_PORTS[0] != "CPA":
        problems.append("CPA must be the first benchmark port (case study)")

    return (not problems), problems


if __name__ == "__main__":
    ok, issues = validate_structure()
    print("POSRRI index structure")
    print("=" * 62)
    for pil, doms in PILLAR_DOMAINS.items():
        print(f"\nPillar {pil}: {PILLARS[pil]}")
        for dom in doms:
            print(f"  {dom:<4} {DOMAINS[dom]}")
            for ind in DOMAIN_INDICATORS[dom]:
                print(f"       {ind:<7} {INDICATOR_LABELS[ind]}")
    print("=" * 62)
    print(f"{N_PILLARS} pillars | {N_DOMAINS} domains | {N_INDICATORS} indicators")
    print(f"Benchmark ports: {', '.join(BENCHMARK_PORTS)}")
    print(f"Structure valid: {ok}" + ("" if ok else f" -- {issues}"))
