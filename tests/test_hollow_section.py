"""Voids in a `patches` fibre section, and the hollow-core slab they were
added for.

A hollow-core slab is a rectangle with two circles taken out of it. The
patches are quadrilaterals, so the circles are `holes`: a cell whose
samples all fall inside one is dropped, and a cell the circle cuts keeps
the fraction of its area outside. Chen's slab is the first MEASURED
destructive test in this line of work (cases/case_25), and its section is
the check that the arithmetic is right.
"""
import pathlib
import sys

import numpy as np
import pytest

from vfife.api import _with_element, prepare_case
from vfife.core.structural import make_structural_solver
from vfife.elements.fiber_section import patch_fibers

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))
ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_a_hole_takes_its_area_out_of_the_patch():
    """A unit square with a centred hole of r = 0.25: the exact net area is
    1 - pi/16 = 0.804603. Dropping whole cells would cost a percent or two;
    weighting the cut ones by their sampled fraction keeps it to 0.3%, and
    no fibre is left inside the hole."""
    for n, tol in ((10, 4e-3), (20, 4e-3), (40, 2e-3)):
        y, z, a, cells = patch_fibers([{"corners": [[0, 0], [1, 0], [1, 1], [0, 1]],
                                        "ny": n, "nz": n, "holes": [[0.5, 0.5, 0.25]]}])
        assert a.sum() == pytest.approx(1.0 - np.pi / 16, rel=tol)
        assert len(cells) == len(a)                       # the sweep paints one cell per fibre
        r = np.hypot(y - 0.5, z - 0.5)
        assert r.min() > 0.25 - 1.0 / n                   # nothing sits well inside the hole
    # two holes, and a hole that swallows the patch
    y, z, a, _ = patch_fibers([{"corners": [[0, 0], [2, 0], [2, 1], [0, 1]], "ny": 40, "nz": 20,
                                "holes": [[0.5, 0.5, 0.25], [1.5, 0.5, 0.25]]}])
    assert a.sum() == pytest.approx(2.0 - 2 * np.pi / 16, rel=3e-3)
    y, z, a, _ = patch_fibers([{"corners": [[0, 0], [1, 0], [1, 1], [0, 1]], "ny": 6, "nz": 6,
                                "holes": [[0.5, 0.5, 5.0]]}])
    assert len(a) == 0


def test_the_slab_section_is_chen_s():
    """generate_chen_slab: 99 x 55 cm with two R19 voids. Chen meshed the
    same section into 3046 fibres of 1e-4 m2, so his net area is 0.3046 m2
    -- the one number that pins the shoulder height the figure does not
    dimension."""
    from generate_chen_slab import section_properties, hand_checks, SPAN
    p = section_properties()
    assert p["A"] == pytest.approx(0.3046, rel=5e-3)
    assert 0.25 < p["v_c"] < 0.27                 # the voids make it nearly symmetric
    h = hand_checks()
    # the stress-block ultimate, both beams, against what they failed at
    for arm, measured in ((SPAN / 2 - 2.00, 245.6e3), (SPAN / 2 - 3.00, 315.0e3)):
        assert h["M_u"] / arm == pytest.approx(measured, rel=0.05)


def test_the_slab_case_carries_nine_strands_with_four_bonded_lengths():
    """The `有效长度` table: three strands run the full 12.96 m and six stop
    short, which is a tendon's `extent`. Where a strand is not bonded its
    fibre has no area, so the section at the end of the slab is weaker than
    the section at midspan -- the point of debonding."""
    s = make_structural_solver(*_prep("cases/case_25_chen_hollow_slab.yaml"))
    assert len(s.tendons) == 9
    lengths = sorted(round(float(t["elements"].size and
                                 (s.mesh.nodes[s.mesh.elements[t["elements"]].max()][0]
                                  - s.mesh.nodes[s.mesh.elements[t["elements"]].min()][0])), 1)
                     for t in s.tendons)
    assert lengths[0] < 9.0 and lengths[-1] > 12.5
    n_end = int((s.fiber_area[0, [t["column"] for t in s.tendons]] > 0).sum())
    n_mid = int((s.fiber_area[s.nel // 2, [t["column"] for t in s.tendons]] > 0).sum())
    assert n_end == 3 and n_mid == 9


def _prep(case):
    prep = prepare_case(str(ROOT / case))
    return _with_element(prep.case, prep.element, prep.case_path), prep.mesh, prep.material


def test_the_section_hardens_to_its_ultimate_and_then_cliffs(tmp_path):
    """The section's moment-curvature, which is where the case file's two
    modelling choices were settled: with epscu = 0.008 it hardens past
    kappa = 0.012 (1116 kN.m), and with the code's 0.0035 for unconfined
    concrete it loses a third of its moment there -- and a section that
    cliffs makes the BEAM localise into one element at 70 mm, where the
    test was still hardening at 155."""
    p, mesh, mat = _prep("cases/case_25_chen_hollow_slab.yaml")
    s = make_structural_solver(p, mesh, mat)
    e = s.nel // 2
    y, a, pre = s.fiber_y[e], s.fiber_area[e], s.fiber_prestrain[e]
    lay = s._layouts[0]
    mats = [s.materials[g] for g in lay.conc_groups + lay.rebar_groups]
    idx = [s.group_indices[g] for g in lay.conc_groups + lay.rebar_groups]

    def moment(kappa):
        def resultants(eps0):
            st = [np.zeros((len(i), getattr(m, "n_state", 0) or 0)) for m, i in zip(mats, idx)]
            for f in np.linspace(0.05, 1.0, 20):        # monotonic: concrete02 is path dependent
                sig = np.zeros(len(a))
                for m, i, stt in zip(mats, idx, st):
                    s6 = np.zeros((len(i), 6)); s6[:, 0] = f * (eps0 - y[i] * kappa) + pre[i]
                    sig[i] = m.update(s6, stt, 1.0)[:, 0]
            return float((sig * a).sum()), float((-sig * y * a).sum())
        lo, hi = -0.02, 0.02
        for _ in range(40):                              # bisect for N = 0
            mid = 0.5 * (lo + hi)
            if resultants(mid)[0] > 0: hi = mid
            else: lo = mid
        return abs(resultants(mid)[1])

    m4, m8, m12 = (moment(k) for k in (0.004, 0.008, 0.012))
    assert m4 == pytest.approx(949e3, rel=0.05)
    assert m8 > m4 and m12 > m8                          # hardening, no cliff
    assert m12 == pytest.approx(1116e3, rel=0.05)
    # the ultimate over the 1# beam's shear span, against the measured 245.6 kN
    P = (m12 - 133e3) / 4.30
    assert 0.85 < P / 245.6e3 < 1.0
