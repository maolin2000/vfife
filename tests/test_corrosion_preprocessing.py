"""Corrosion is a preprocessor, not a material (decision D2).

Four laws take one number -- the strand's mass loss -- and produce a different
strand: less area (hemispherical pitting, R = 9), less modulus, less strength,
and much less ultimate strain. Three of them change numbers a case file already
carries; the fourth feeds `steel02.rupture_strain`. NOTHING in `vfife/` knows
the word corrosion, and these tests are what keeps it that way -- they live
entirely on `benchmarks/generate_corroded_pc_beam.py`.

The load-bearing test is `test_zero_corrosion_returns_the_control_beams_own
_hardening_ratio`: the hardening ratio is derived, not chosen, and at rho = 0
it has to come back as the 0.008 that case_31 was written with months before
any of this existed. If the four laws were wired to the wrong numbers, that
is where it would show.
"""
import pathlib
import re
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from generate_corroded_pc_beam import (            # noqa: E402
    AREA_STRAND, CASE_31, D_OUTER, EPS_U_STRAND, E_STRAND, FY_STRAND, LI_BEAMS,
    corroded_strand, pit_area, write_corroded_case)

PRESTRESS_0 = 7.786e+08


def test_zero_corrosion_is_the_identity():
    c = corroded_strand(0.0, PRESTRESS_0)
    assert c["area"] == pytest.approx(AREA_STRAND)
    assert c["area_ratio"] == pytest.approx(1.0)
    assert c["E"] == pytest.approx(E_STRAND)
    assert c["Fy"] == pytest.approx(FY_STRAND)
    assert c["eps_u"] == pytest.approx(EPS_U_STRAND)
    assert c["prestress"] == pytest.approx(PRESTRESS_0)
    assert not c["brittle"]


def test_zero_corrosion_returns_the_control_beams_own_hardening_ratio():
    """`b` is derived -- whatever makes the strand reach its ultimate strength
    at its ultimate strain -- and case_31 was written with 0.008 by hand."""
    text = CASE_31.read_text(encoding="utf-8")
    m = re.search(r"strand:.*?\bb:\s*([0-9.eE+-]+)", text)
    assert m, "case_31 no longer declares the strand's hardening ratio"
    assert corroded_strand(0.0)["b"] == pytest.approx(float(m.group(1)), rel=0.05)


def test_the_prestrain_is_invariant_because_the_tendon_is_anchored():
    """Corrosion does not change the tendon's LENGTH, so its strain stays put
    and the stress follows the modulus down. That is what propagates the
    control beam's one calibrated number into every corroded beam without a
    second calibration."""
    for _nm, rho, _m in LI_BEAMS:
        c = corroded_strand(rho, PRESTRESS_0)
        assert c["prestress"] / c["E"] == pytest.approx(PRESTRESS_0 / E_STRAND)
        # and the FORCE does fall, because both area and modulus fall
        assert c["prestress"] * c["area"] < PRESTRESS_0 * AREA_STRAND + 1e-6


@pytest.mark.parametrize("key", ["area", "E", "Fy", "Fu", "eps_u"])
def test_every_law_is_monotone_in_the_mass_loss(key):
    v = [corroded_strand(r)[key] for r in np.linspace(0.0, 0.12, 40)]
    assert all(b <= a + 1e-18 for a, b in zip(v, v[1:])), f"{key} is not monotone"


def test_pitting_removes_less_area_than_uniform_corrosion_would():
    """The core wire is not attacked and the pit is local, so the area left is
    always MORE than (1 - rho). A model that lost rho of the area would be
    describing uniform corrosion, which is the thing R = 9 exists to deny."""
    for _nm, rho, _m in LI_BEAMS[1:]:
        assert corroded_strand(rho)["area_ratio"] > 1.0 - rho


def test_the_pit_deepens_faster_than_the_mass_is_lost():
    """R = 9: the maximum depth is nine times the average, so a 1% mass loss
    on a 5 mm wire is already a 0.11 mm notch."""
    c = corroded_strand(0.01)
    assert c["pit_depth"] == pytest.approx(9.0 * 0.5 * D_OUTER
                                           * (1 - np.sqrt(1 - 0.01)))
    assert c["pit_depth"] > 0.1e-3


def test_the_law_goes_brittle_between_the_last_two_beams():
    """Past rho = 7.46% the ultimate strain falls BELOW the elastic limit: the
    strand can no longer reach its own proof stress before it breaks. Li's
    beams switch from concrete crushing to wire fracture at about 8%, and that
    the two coincide is the whole reason this law is worth testing against
    them. A5 must still be ductile and A6 must not."""
    assert not corroded_strand(0.0615)["brittle"]      # A5
    assert corroded_strand(0.0747)["brittle"]          # A6
    assert corroded_strand(0.0747)["b"] == 0.0, "a brittle strand has no hardening"


def test_a_pit_past_the_wire_is_refused_rather_than_extrapolated():
    with pytest.raises(ValueError, match="Figure 5"):
        pit_area(0.40)


def test_a_corroded_case_differs_from_the_control_ONLY_in_its_strand(tmp_path):
    """The corroded cases are derived from case_31's own file so that geometry,
    mesh, load steps and monitors cannot drift. This asserts that: the lines
    that differ are the name, the output path, the mesh path, and the three
    that describe the strand."""
    out, _c = write_corroded_case(0.0452, "c31_test", tmp_path)
    a = CASE_31.read_text(encoding="utf-8").splitlines()
    b = [ln for ln in out.read_text(encoding="utf-8").splitlines()
         if not ln.startswith("# GENERATED")]
    import difflib
    changed = [ln[1:] for ln in difflib.unified_diff(a, b, n=0, lineterm="")
               if ln.startswith("-") and not ln.startswith("---")]
    assert changed, "the generator changed nothing at all"
    for ln in changed:
        assert ("strand" in ln or "prestress:" in ln or ln.startswith("name:")
                or "path:" in ln), f"an unexpected line changed:\n  {ln}"
