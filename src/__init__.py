"""POSRRI analysis package.

Modules
-------
structure   : the fixed 3 x 10 x 50 index hierarchy and Saaty's Random Index.
delphi      : Delphi consensus, content validity (I-CVI / S-CVI), Cohen's kappa.
ahp         : Analytic Hierarchy Process weights, consistency ratios, AIJ.
scoring     : weighted domain / pillar / overall POSRRI scores on 0-100.
sensitivity : AHP versus equal-weight robustness checks.
viz         : publication-grade matplotlib figures.
"""

__all__ = ["structure", "delphi", "ahp", "scoring", "sensitivity", "viz"]
