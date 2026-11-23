"""Tube and box fibre sections.

WHY THEY EXIST: a steel spatial structure is built of tubes, and Cai et al.
(2023) model both their tube members (48.3 x 3.2 CHS, 25 x 50 x 2.5 RHS,
40 x 40 x 3 SHS) and their Kiwitt-8 dome (114 x 3 CHS) with the steel in a
thin wall. Discretising that as a solid circle puts fibres where there is no
steel, which changes the two numbers a fibre section exists to produce: when
the extreme fibre yields, and how much section is left once it has.

WHAT IS ASSERTED is that the discretisation reproduces the section's own
integrals -- area and both second moments -- because everything downstream is
built on those, and for a ring they can be made EXACT rather than convergent.
"""

import pathlib
import tempfile

import numpy as np
import pytest

from vfife.elements.fiber_section import box_fibers, rect_fibers, tube_fibers


def moments(y, z, a):
    return a.sum(), (a * z ** 2).sum(), (a * y ** 2).sum()


def chs(d, t):
    """Exact A, I of a circular hollow section."""
    di = d - 2 * t
    return np.pi / 4 * (d ** 2 - di ** 2), np.pi / 64 * (d ** 4 - di ** 4)


def rhs(b, h, t):
    """Exact A, Iy, Iz of a rectangular hollow section (sharp corners)."""
    bi, hi = b - 2 * t, h - 2 * t
    return (b * h - bi * hi,
            (b * h ** 3 - bi * hi ** 3) / 12.0,
            (h * b ** 3 - hi * bi ** 3) / 12.0)


@pytest.mark.parametrize("nt", [3, 8, 20, 40])
@pytest.mark.parametrize("nr", [1, 3])
def test_a_tube_reproduces_its_own_area_and_second_moment_exactly(nt, nr):
    """EXACTLY, at any number of fibres -- which is not what a lumped
    discretisation usually gives, and is the whole reason the fibres sit at
    the RMS radius of their wedge rather than at its centroid."""
    d, t = 0.114, 0.003
    A_ex, I_ex = chs(d, t)
    A, Iy, Iz = moments(*tube_fibers(d, t, nt, nr))
    assert A == pytest.approx(A_ex, rel=1e-14)
    assert Iy == pytest.approx(I_ex, rel=1e-13)
    assert Iz == pytest.approx(I_ex, rel=1e-13)


def test_a_tube_is_still_centred_on_its_own_axis():
    """The first moments have to vanish too, or the section's neutral axis is
    not where the element thinks it is. They do, by symmetry, at any radius --
    which is why the second moment could be bought without paying for it."""
    y, z, a = tube_fibers(0.114, 0.003, 20)
    assert abs((a * y).sum()) < 1e-18
    assert abs((a * z).sum()) < 1e-18


def test_the_obvious_placement_would_have_been_wrong_and_by_how_much():
    """The measurement behind the choice: at the area centroid of each wedge
    the second moment comes out LOW, because a point has no second moment of
    its own and a wedge does. It is 5% at 8 fibres and still 0.2% at 40."""
    d, t = 0.114, 0.003
    _A, I_ex = chs(d, t)
    r0, r1 = d / 2 - t, d / 2
    for nt, expected in ((8, 0.051), (20, 0.0084), (40, 0.0023)):
        dth = 2 * np.pi / nt
        r_c = (2 / 3) * (r1 ** 3 - r0 ** 3) / (r1 ** 2 - r0 ** 2) * np.sinc(dth / (2 * np.pi))
        area = 0.5 * dth * (r1 ** 2 - r0 ** 2)
        th = dth * (np.arange(nt) + 0.5)
        I_centroid = (area * (r_c * np.sin(th)) ** 2).sum()
        assert 1.0 - I_centroid / I_ex == pytest.approx(expected, abs=0.001), nt


def test_a_box_is_exact_in_area_and_converges_in_the_second_moment():
    """A box's cells are not congruent, so the trick a ring allows is not
    available: the fibres sit at cell centroids, the area is exact and the
    second moments converge from below as the walls are refined."""
    b, h, t = 0.025, 0.050, 0.0025
    A_ex, Iy_ex, Iz_ex = rhs(b, h, t)
    errs = []
    for n in (2, 4, 8, 16):
        A, Iy, Iz = moments(*box_fibers(b, h, t, n, 2 * n))
        assert A == pytest.approx(A_ex, rel=1e-14)
        errs.append(1.0 - Iy / Iy_ex)
    assert all(e > 0 for e in errs), "the centroid grid is low, not high"
    assert errs == sorted(errs, reverse=True), errs
    assert errs[-1] < 0.002


def test_the_four_sections_of_the_paper_come_out_right():
    """Cai et al. Table 2 / Fig. 18: the sections the tube-member cases need."""
    A, I = chs(0.0483, 0.0032)                       # Tubes C and D
    got = moments(*tube_fibers(0.0483, 0.0032, 20))
    assert got[0] == pytest.approx(A, rel=1e-14) and got[1] == pytest.approx(I, rel=1e-13)
    assert A * 1e6 == pytest.approx(453.4, abs=0.1)

    A, Iy, Iz = rhs(0.025, 0.050, 0.0025)            # Tube A
    got = moments(*box_fibers(0.025, 0.050, 0.0025, 4, 8))
    assert got[0] == pytest.approx(A, rel=1e-14)
    assert A * 1e6 == pytest.approx(350.0, abs=0.1)

    A, Iy, _ = rhs(0.040, 0.040, 0.003)              # Tube B
    got = moments(*box_fibers(0.040, 0.040, 0.003, 6, 6))
    assert got[0] == pytest.approx(A, rel=1e-14)
    assert A * 1e6 == pytest.approx(444.0, abs=0.1)

    A, I = chs(0.114, 0.003)                         # the dome's members
    got = moments(*tube_fibers(0.114, 0.003, 20))
    assert got[0] == pytest.approx(A, rel=1e-14) and got[1] == pytest.approx(I, rel=1e-13)


def test_the_solid_grid_was_left_alone():
    """`rect_fibers` is untouched on purpose: a solid section is refined in
    both directions anyway, and moving it would move every verification number
    in the library for a discretisation error the analyst already controls."""
    y, z, a = rect_fibers(0.2, 0.4, 2, 2)
    assert a == pytest.approx([0.02] * 4, rel=1e-14)
    assert sorted(set(np.round(y, 6))) == [-0.05, 0.05]
    assert (a * z ** 2).sum() < 0.2 * 0.4 ** 3 / 12, "a 2x2 grid is low, as always"


@pytest.mark.parametrize("bad, match", [
    (dict(diameter=0.05, thickness=0.025, nt=8), "solid bar"),
    (dict(diameter=0.05, thickness=0.03, nt=8), "solid bar"),
    (dict(diameter=0.05, thickness=0.003, nt=2), "nt >= 3"),
    (dict(diameter=-1.0, thickness=0.003, nt=8), "must be > 0"),
])
def test_a_tube_that_is_not_one_is_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        tube_fibers(**bad)


def test_a_box_whose_walls_meet_is_refused():
    with pytest.raises(ValueError, match="closes a"):
        box_fibers(0.02, 0.02, 0.01, 4, 4)


# --------------------------------------------------------------------------- #
#  through the input model
# --------------------------------------------------------------------------- #

def _case(td, section, name="hollow"):
    msh = pathlib.Path(td) / f"{name}.msh"
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n3\n"
                   "1 0 0 0\n2 0.5 0 0\n3 1.0 0 0\n$EndNodes\n$Elements\n2\n"
                   "1 1 2 1 1 1 2\n2 1 2 1 1 2 3\n$EndElements\n", encoding="utf-8")
    yp = pathlib.Path(td) / f"{name}.yaml"
    yp.write_text(f"""
name: {name}
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {str(msh).replace(chr(92), "/")}}}
materials:
  s: {{type: steel_cyclic, grade: Q235B, E: 2.06e11, rho: 7850.0}}
sections:
  m: {{type: fiber, material: s, {section}}}
assign: [{{region: all, section: m, orientation: [0, 0, 1]}}]
regions: {{base: {{box: [-0.01, 0.01, -1, 1, -1, 1]}}}}
constraints: [{{region: base, dofs: [1,1,1,1,1,1]}}]
solver: {{backend: numpy, mode: dynamic, total_time: 1.0e-4}}
output: {{monitors: {{}}}}
""", encoding="utf-8")
    return str(yp)


def _solver(yaml_path):
    from vfife.api import prepare_case, _with_element
    from vfife.core.structural import make_structural_solver

    prep = prepare_case(yaml_path)
    p = _with_element(prep.case, prep.element, prep.case_path)
    return make_structural_solver(p, prep.mesh, prep.material)


def test_a_case_can_ask_for_a_tube_and_gets_the_ring():
    with tempfile.TemporaryDirectory() as td:
        s = _solver(_case(td, "shape: tube, diameter: 0.114, t: 0.003, nt: 20"))
        A, I = chs(0.114, 0.003)
        assert s.n_fibers == 20
        assert float(s.total_area[0]) == pytest.approx(A, rel=1e-14)
        assert float(s.Iy[0]) == pytest.approx(I, rel=1e-13)
        assert float(s.Iz[0]) == pytest.approx(I, rel=1e-13)


def test_a_case_can_ask_for_a_box():
    with tempfile.TemporaryDirectory() as td:
        s = _solver(_case(td, "shape: box, width: 0.025, height: 0.050, "
                              "t: 0.0025, ny: 4, nz: 8", name="box"))
        assert s.n_fibers == 24
        assert float(s.total_area[0]) == pytest.approx(rhs(0.025, 0.050, 0.0025)[0], rel=1e-14)


@pytest.mark.parametrize("section, match", [
    ("shape: tube, diameter: 0.114", "diameter .outer. and t"),
    ("shape: box, width: 0.025, t: 0.0025", "width, height"),
    ("shape: I, width: 0.1, height: 0.2", "shape: rect"),
])
def test_a_hollow_section_missing_its_wall_is_refused(section, match):
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError, match=match):
            _solver(_case(td, section, name="bad"))


def test_bars_in_a_hollow_section_are_refused():
    """Not an oversight: a steel tube has no reinforcement, and a
    concrete-filled one needs a confinement model this library does not have.
    Silently ignoring the bars would be the expensive way to find that out."""
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError, match="bars in a hollow"):
            _solver(_case(td, "shape: tube, diameter: 0.114, t: 0.003, nt: 20, "
                              "bars: [{y: 0.0, z: 0.05, section: b}]", name="bars"))
