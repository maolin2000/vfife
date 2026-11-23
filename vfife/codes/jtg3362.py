"""JTG 3362-2018 (and JTG D62-2004, which has the same clauses below):
Specifications for Design of Highway Reinforced Concrete and Prestressed
Concrete Bridges and Culverts.

CHECKED AGAINST MIDAS CIVIL, not against the code's text, which could not
be read while this was written: Midas' tendon loss table of
a rigid frame's closure tendon, stressed and held one day, gives the
relaxation loss this module gives to four figures at every element end
(4.1127 MPa at 1107.16 MPa, 5.94 at 1171.93, 6.17 at 1179.55). That checks
the formula and the start of the time table (half the final loss at two
days, reached linearly). The later rows of the table are still to be
checked the same way against a tendon held longer.
"""
from __future__ import annotations

from typing import Optional

#: the ratio of the relaxation loss at a time to its final value, against
#: days after stressing (the code's appendix table); linear between rows,
#: from zero at stressing to the first row, and 1 after the last
RELAXATION_RATIO = ((2.0, 0.50), (10.0, 0.61), (20.0, 0.74), (30.0, 0.87), (40.0, 1.00))


def relaxation_final(sigma_pe: float, fpk: float, zeta: float, psi: float = 1.0) -> float:
    """The final relaxation loss of a wire or strand as a FRACTION of its
    stress after anchoring, sigma_l5 / sigma_pe = psi zeta (0.52 sigma_pe /
    fpk - 0.26): psi 1.0 for one stressing, 0.9 for over-stressing; zeta 1.0
    for ordinary and 0.3 for low relaxation. Nothing at or below half fpk."""
    ratio = float(sigma_pe) / float(fpk)
    if ratio <= 0.5:
        return 0.0
    return float(psi) * float(zeta) * (0.52 * ratio - 0.26)


def relaxation_curve(sigma_pe: float, fpk: float, zeta: float, psi: float = 1.0) -> Optional[dict]:
    """The solver's `relaxation` spec for a tendon anchored at sigma_pe: a
    `curve` of the loss fraction against hours, read linearly in t. None
    when it does not relax."""
    return relaxation_curve_of(relaxation_final(sigma_pe, fpk, zeta, psi))


def relaxation_curve_of(final: float) -> Optional[dict]:
    """The `curve` spec of a final loss fraction developing as the table says."""
    if final <= 0.0:
        return None
    return {"model": "curve", "interpolation": "linear",
            "curve": [[days * 24.0, round(final * r, 8)] for days, r in RELAXATION_RATIO]}
