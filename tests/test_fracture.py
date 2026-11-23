"""The three layers of the fracture framework.

What is asserted here is the SEPARATION, not any particular criterion: that a
material says only whether its own point is finished, that an element family
says only how its points add up, and that neither of them knows what the other
is. If those two ever start needing to know about each other, a fracture
criterion becomes a thing you write once per material times once per element,
which is the failure mode this file exists to prevent.
"""

import pathlib
import tempfile

import numpy as np
import pytest

from vfife.core import fracture
from vfife.materials.base import create
from vfife.testing.models import MATERIAL_PARAMS


# --------------------------------------------------------------------------- #
#  layer 1: the material says one thing
# --------------------------------------------------------------------------- #

def test_a_material_with_no_notion_of_failure_says_so():
    """None, not zeros. Asking a linear elastic law about fracture is a
    modelling error, and it has to be distinguishable from a law that has a
    criterion which happens not to have fired."""
    assert create("linear_elastic", E=2e11, nu=0.3, rho=7850).failure(np.zeros((3, 0))) is None
    assert create("rebar", **MATERIAL_PARAMS["rebar"]).failure(np.zeros((3, 3))) is None


def test_the_rebar_latch_is_its_own_rupture_strain():
    m = create("rebar", **dict(MATERIAL_PARAMS["rebar"], rupture_strain=0.05))
    st = m.init_state(4)
    st[:, 1] = [0.0, 0.049, 0.05, 0.2]
    assert m.failure(st).tolist() == [0.0, 0.0, 1.0, 1.0]


def test_the_rebar_latch_agrees_with_the_stress_it_zeroes():
    """The bar law has zeroed a ruptured point's stress since it was written;
    `failure` must be the same event, not a second opinion about it.

    ONE DIRECTION ONLY, and the first version of this test got it wrong: a
    zero stress does not imply failure, because an unloaded elastic fibre
    reports zero too (at k = 0 the strain is zero and so is the stress). That
    is the whole reason the element cannot infer fracture from the stress and
    needs the material to say it.
    """
    m = create("rebar", **dict(MATERIAL_PARAMS["rebar"], rupture_strain=2.0e-3))
    st = m.init_state(1)
    eps = np.zeros((1, 6))
    ruptured_at = None
    for k in range(200):
        eps[0, 0] = 1.0e-4 * k
        stress = m.update(eps, st, 1.0)[0, 0]
        failed = bool(m.failure(st)[0] > 0.5)
        if failed:
            assert stress == 0.0, k
            if ruptured_at is None:
                ruptured_at = k
    assert ruptured_at is not None
    assert not bool(m.failure(m.init_state(1))[0]), "a fresh bar is not broken"


def test_cdp_reports_the_monotone_halves_not_the_recovering_one():
    """`d` falls when a crack closes under compression -- that is the model
    working. A failure fraction may not fall, so it is built from dt and dc."""
    m = create("cdp", **MATERIAL_PARAMS["cdp"])
    st = m.init_state(1)
    eps = np.zeros((1, 6))
    d_seen, f_seen = [], []
    path = np.r_[np.linspace(0, 3.0e-4, 40), np.linspace(3.0e-4, -1.0e-3, 60)]
    for e in path:
        eps[0, 0] = e
        m.update(eps, st, 1.0)
        d_seen.append(float(m.damage_measure(st)[0]))
        f_seen.append(float(m.failure(st)[0]))
    d_seen, f_seen = np.array(d_seen), np.array(f_seen)
    assert np.diff(d_seen).min() < -1e-6, "d is expected to recover on this path"
    assert np.diff(f_seen).min() >= -1e-12, "the failure fraction must not fall"


# --------------------------------------------------------------------------- #
#  layer 2: the element says how its points add up
# --------------------------------------------------------------------------- #

def test_a_solid_counts_its_gauss_points():
    layout = fracture.solid_layout(8)
    f = np.zeros((2, 8))
    f[0, :4] = 1.0
    f[1, :] = 1.0
    assert fracture.fraction(f, layout).tolist() == [0.5, 1.0]


def test_a_fibre_is_broken_when_it_is_broken_ANYWHERE_along_the_member():
    """Cai et al. eq. (20): fibre i fails when the strain at any of its
    integration points passes the fracture strain -- not when the average
    does. A fibre broken at one station is a fibre that carries nothing."""
    layout = fracture.fiber_layout(n_fibers=4, n_stations=3)
    f = np.zeros((1, 12))
    f[0, 1] = 1.0                      # fibre 0, station 1 only
    assert fracture.fraction(f, layout)[0] == pytest.approx(0.25)
    mean_layout = fracture.Layout("mean", 12, 4, within_group="mean")
    assert fracture.fraction(f, mean_layout)[0] == pytest.approx(1 / 12)


def test_the_two_layouts_are_the_same_arithmetic_on_a_different_shape():
    """Twenty fibres at one station and twenty Gauss points are the same
    count; the layouts differ only in how the points are grouped."""
    a = fracture.fraction(np.eye(20)[:1] * 0 + np.r_[np.ones(5), np.zeros(15)],
                          fracture.solid_layout(20))
    b = fracture.fraction(np.r_[np.ones(5), np.zeros(15)][None, :],
                          fracture.fiber_layout(20, 1))
    assert a[0] == pytest.approx(0.25) and b[0] == pytest.approx(0.25)


def test_a_layout_that_does_not_divide_is_refused():
    with pytest.raises(ValueError, match="divide"):
        fracture.Layout("bad", 7, 2)


def test_a_failure_fraction_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fracture.fraction(np.full((1, 4), 1.5), fracture.solid_layout(4))


def test_the_reduction_is_the_same_number_the_obvious_one_liner_gives():
    """`fraction` does not write `grouped.max(axis=2)`, because numpy's
    reduction over a TWO-LONG trailing axis is one call per group: measured at
    120 us against 5 us for the same answer as a running `np.maximum`, which
    was 90% of what asking about fracture cost per step.

    Bit-identical, not close: this is the same arithmetic in a different loop
    order, and a tolerance here would hide a change that is not."""
    rng = np.random.default_rng(0)
    for n_fibers, n_stations in ((68, 2), (20, 3), (9, 2), (5, 4), (8, 1)):
        f = rng.random((7, n_fibers * n_stations))
        g = f.reshape(7, n_fibers, n_stations)
        assert np.array_equal(
            fracture.fraction(f, fracture.fiber_layout(n_fibers, n_stations)),
            g.max(axis=2).mean(axis=1))
        mean_layout = fracture.Layout("m", n_fibers * n_stations, n_fibers,
                                      within_group="mean")
        assert np.array_equal(fracture.fraction(f, mean_layout),
                              g.mean(axis=2).mean(axis=1))


def test_the_fraction_is_over_the_section_even_when_fewer_fibres_are_watched():
    """An RC section is 64 concrete fibres and 4 bars, and concrete01 declares
    no criterion -- those 64 columns are structurally zero forever. Watching
    only the 4 has to leave FAC meaning the same thing: the fraction of the
    SECTION that is gone, so one ruptured bar is 1/68 and not 1/4."""
    watched = fracture.fiber_layout(n_fibers=68, n_stations=2, tracked=4)
    assert watched.points_per_element == 8 and watched.over == 68
    f = np.zeros((1, 8))
    f[0, 0] = 1.0                                   # one bar, one station
    assert fracture.fraction(f, watched)[0] == pytest.approx(1 / 68)
    full = np.zeros((1, 136))
    full[0, 0] = 1.0
    assert (fracture.fraction(f, watched)
            == fracture.fraction(full, fracture.fiber_layout(68, 2))).all()


def test_a_denominator_smaller_than_what_is_tracked_is_refused():
    with pytest.raises(ValueError, match="cannot be taken over"):
        fracture.Layout("bad", 8, 4, denominator=3)


# --------------------------------------------------------------------------- #
#  the latch
# --------------------------------------------------------------------------- #

def test_what_is_dead_stays_dead():
    latch = fracture.Latch(2, fracture.solid_layout(4))
    latch.update(np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]))
    assert latch.fac.tolist() == [0.5, 0.0]
    latch.update(np.zeros((2, 4)))                    # the measure falls back
    assert latch.fac.tolist() == [0.5, 0.0], "the latch let a point recover"


def test_an_element_fractures_once_and_is_reported_once():
    latch = fracture.Latch(3, fracture.solid_layout(2))
    assert latch.update(np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])).tolist() == [0]
    assert latch.update(np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])).tolist() == [1]
    assert latch.update(np.ones((3, 2))).tolist() == [2]
    assert latch.n_fractured == 3


def test_the_kinematic_half_needs_no_material_at_all():
    """Buckling, a collapsed volume, a rotation past a limit: element-level
    criteria that no constitutive law can provide. The solid erosion of
    v0.1 already had one, and it is why `update` takes a second argument."""
    latch = fracture.Latch(2, fracture.solid_layout(4))
    newly = latch.update(np.zeros((2, 4)), kinematic=np.array([False, True]))
    assert newly.tolist() == [1]
    assert latch.fac.tolist() == [0.0, 0.0], "a kinematic failure is not a material one"


def test_a_partial_threshold_is_allowed_and_defaults_to_whole():
    assert fracture.Latch(1, fracture.solid_layout(4)).threshold == 1.0
    latch = fracture.Latch(1, fracture.solid_layout(4), threshold=0.5)
    assert latch.update(np.array([[1.0, 1.0, 0.0, 0.0]])).tolist() == [0]
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        fracture.Latch(1, fracture.solid_layout(4), threshold=0.0)


def test_the_material_and_the_element_never_see_each_other():
    """The point of the whole file, as one assertion: a material's failure
    array feeds any layout whose point count matches, and neither knows what
    the other is. Twenty rebar fibres in a section and twenty rebar points in
    a solid block are the same call."""
    m = create("rebar", **dict(MATERIAL_PARAMS["rebar"], rupture_strain=0.01))
    st = m.init_state(20)
    st[:5, 1] = 0.02
    f = m.failure(st)[None, :]
    assert fracture.fraction(f, fracture.fiber_layout(20, 1))[0] == pytest.approx(0.25)
    assert fracture.fraction(f, fracture.solid_layout(20))[0] == pytest.approx(0.25)
    # 5 fibres of 4 stations: points 0-3 are fibre 0 and point 4 is fibre 1,
    # so two of five fibres are gone -- the same array, read as a different
    # section, is a different fraction, and that is the element's business
    assert fracture.fraction(f, fracture.fiber_layout(5, 4))[0] == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
#  layer 3: the model, end to end on the fibre beam
# --------------------------------------------------------------------------- #

def _bar_case(td, n_elem=2, rupture=2.0e-3, threshold=1.0, bars=False, name="frac"):
    """A straight bar of `n_elem` fibre members, fracture on."""
    L = 1.0
    msh = pathlib.Path(td) / f"{name}.msh"
    nodes = "\n".join(f"{i+1} {i*L/n_elem:.6f} 0 0" for i in range(n_elem + 1))
    elems = "\n".join(f"{i+1} 1 2 1 1 {i+1} {i+2}" for i in range(n_elem))
    msh.write_text(f"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n{n_elem+1}\n"
                   f"{nodes}\n$EndNodes\n$Elements\n{n_elem}\n{elems}\n$EndElements\n", encoding="utf-8")
    rebar = (", bars: [{y: 0.02, z: 0.02, section: b}, {y: -0.02, z: -0.02, section: b}]"
             if bars else "")
    bar_sec = ("  b: {type: bar, material: steel, shape: general, area: 2.0e-4}\n"
               if bars else "")
    fibre_mat = "conc" if bars else "steel"
    yp = pathlib.Path(td) / f"{name}.yaml"
    yp.write_text(f"""
name: {name}
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {str(msh).replace(chr(92), "/")}}}
materials:
  steel: {{type: rebar, E: 200.0e9, yield_strength: 4.0e8, hardening: 2.0e9,
           rho: 7850.0, rupture_strain: {rupture}}}
  conc: {{type: concrete01, fc: 3.0e7, eps_c0: 0.002, fcu: 6.0e6, eps_cu: 0.006,
          E: 3.0e10, rho: 2400.0}}
sections:
{bar_sec}  rc: {{type: fiber, material: {fibre_mat}, shape: rect, width: 0.05,
        height: 0.05, ny: 3, nz: 3, fracture: {{threshold: {threshold}}}{rebar}}}
assign: [{{region: all, section: rc}}]
regions: {{base: {{box: [-0.01,0.01,-0.01,0.01,-0.01,0.01]}}}}
constraints: [{{region: base, dofs: [1,1,1,1,1,1]}}]
solver: {{backend: numpy, mode: dynamic, n_steps: 10, total_time: 1.0e-3}}
output: {{monitors: {{}}}}
""", encoding="utf-8")
    return str(yp)


def _solver(yaml_path):
    from vfife.api import prepare_case, _with_element
    from vfife.core.structural import make_structural_solver

    prep = prepare_case(yaml_path)
    assert prep.preflight["status"] == "PASS", prep.preflight
    p = _with_element(prep.case, prep.element, prep.case_path)
    return make_structural_solver(p, prep.mesh, prep.material)


def _stretch(solver, per_element):
    """Displace every node along x so element e carries strain per_element[e]."""
    q = np.zeros(solver.ndof)
    x = 0.0
    for e, eps in enumerate(per_element):
        x += solver.length[e] * eps
        q[6 * (e + 1)] = x
    solver.q = q


def test_a_ruptured_member_stops_carrying_load_and_keeps_its_mass():
    """The whole pipeline in one test: the bar law latches, the section counts
    its fibres, the element stops contributing, and THE MASS IS STILL THERE.

    The mass is the assertion that says why this is deactivation and not the
    element deletion the solid path does: a member that has broken off is
    still weight sitting on the structure below it, and a deleted one is not
    (core/fracture.py)."""
    with tempfile.TemporaryDirectory() as td:
        s = _solver(_bar_case(td, n_elem=2, rupture=2.0e-3))
        mass_before = float(s.mass.sum())

        _stretch(s, [1.0e-3, 1.0e-3])          # elastic, nothing broken
        f0, _ = s.internal_force()
        assert s.fracture.n_fractured == 0
        assert abs(f0).max() > 1.0
        assert s.fracture.fac.tolist() == [0.0, 0.0]

        _stretch(s, [1.0e-3, 5.0e-3])          # element 1 past its rupture strain
        s.internal_force()
        assert s.fracture.latch.fractured.tolist() == [False, True]
        assert s.fracture.fac.tolist() == [0.0, 1.0]

        # and now it carries nothing: with the far element dead, the only
        # internal force left is the one the surviving element applies
        f1, actions = s.internal_force()
        assert actions[1].tolist() == [0.0] * 6
        assert abs(f1[12:]).max() == 0.0, "a fractured member still pushed its far node"
        assert abs(f1[:12]).max() > 1.0, "the surviving member stopped carrying too"
        assert float(s.mass.sum()) == mass_before


def test_the_latch_holds_when_the_load_comes_off():
    """The member is not repaired by unloading it -- which is the difference
    between a fracture and a damage variable, and the reason `Latch` exists
    at all rather than a test evaluated each step."""
    with tempfile.TemporaryDirectory() as td:
        s = _solver(_bar_case(td, n_elem=1, rupture=2.0e-3, name="latch"))
        _stretch(s, [5.0e-3])
        s.internal_force()
        assert s.fracture.n_fractured == 1
        _stretch(s, [0.0])
        f, _ = s.internal_force()
        assert s.fracture.n_fractured == 1
        assert abs(f).max() == 0.0


def test_a_section_that_can_never_fracture_is_refused():
    """`fracture:` on a section of concrete01 fibres, whose law declares no
    failure at all: the run would go to the end and report a structure that
    survived, which is the most expensive kind of wrong answer available."""
    with tempfile.TemporaryDirectory() as td:
        yp = _bar_case(td, n_elem=1, name="silent")
        text = pathlib.Path(yp).read_text(encoding="utf-8").replace(
            "material: steel, shape: rect", "material: conc, shape: rect")
        pathlib.Path(yp).write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="no fibre material declares"):
            _solver(yp)


def test_only_the_bars_of_an_rc_section_are_watched():
    """The same claim on a real solver: the section has concrete fibres and
    two bars, and what is gathered every step is the bars."""
    with tempfile.TemporaryDirectory() as td:
        s = _solver(_bar_case(td, n_elem=2, bars=True, threshold=0.15, name="watch"))
        assert s.n_fibers == 11                       # 9 concrete + 2 bars
        assert s._tracked_failure.shape == (2, 2, len(s.gauss_s))
        assert [g for g, _s, _n in s._failing_groups] == [1, 2]
        assert s.fracture.layout.over == s.n_fibers
        assert s._point_failure().shape == (2, 2 * len(s.gauss_s))
        assert s.fracture.fac.tolist() == [0.0, 0.0]


def test_the_summary_names_the_laws_that_cannot_fail():
    """A concrete section with steel bars CAN fracture -- when every bar has
    gone -- but its concrete never contributes to the count, so a threshold of
    1.0 is unreachable. Saying which laws are silent is the difference between
    that being a modelling decision and a surprise."""
    with tempfile.TemporaryDirectory() as td:
        s = _solver(_bar_case(td, n_elem=1, bars=True, threshold=0.15, name="rc"))
        line = [l for l in s.summary().splitlines() if "fracture" in l][0]
        assert "Concrete01" in line and "threshold 0.15" in line


def test_the_fused_kernel_honours_the_same_deactivation_the_reference_does():
    """The kernel is a SECOND implementation of the element loop, and v0.1
    has already shown once what happens when a second implementation quietly
    stops honouring a flag (the frame kernel was handed a numpy pin and the
    line half discarded it, so the fix looked like it had not worked).

    The mask is set by hand here, which keeps this test about the kernel and
    nothing else. A section that reaches FAC = 1 on its own, on the kernel
    path, is asserted in tests/test_steel_cyclic.py -- when v0.1 was written
    no law the kernel could inline had a fracture criterion at all, and
    `steel_cyclic` is the one that closed that gap.
    """
    with tempfile.TemporaryDirectory() as td:
        # bars of steel02, concrete01 fibres: the section the kernel is for.
        # No `fracture:` block -- this asks about deactivation, not the latch.
        yp = _bar_case(td, n_elem=2, bars=True, name="k")
        text = pathlib.Path(yp).read_text(encoding="utf-8")
        text = text.replace(", fracture: {threshold: 1.0}", "")
        text = text.replace("material: steel, shape: general",
                            "material: s2, shape: general")
        text = text.replace("materials:\n", "materials:\n  s2: {type: steel02, "
                            "E: 200.0e9, Fy: 4.0e8, b: 0.01, rho: 7850.0}\n")
        pathlib.Path(yp).write_text(text, encoding="utf-8")

        fast = _solver(yp)
        if fast._kernel is None:
            pytest.skip(f"no fused kernel here: {fast.kernel_status}")
        assert fast.fracture is None
        slow = _solver(yp)
        slow._kernel = None                      # the NumPy reference path

        for s in (fast, slow):
            _stretch(s, [1.0e-3, 1.5e-3])
        both = [s.internal_force()[0].copy() for s in (fast, slow)]
        assert np.allclose(*both, rtol=1e-10, atol=1e-6)
        assert abs(both[0][12:]).max() > 1.0

        for s in (fast, slow):
            s._active[1] = 0
        f_fast, a_fast = fast.internal_force()
        f_slow, a_slow = slow.internal_force()
        assert np.allclose(f_fast, f_slow, rtol=1e-10, atol=1e-6)
        assert a_fast[1].tolist() == a_slow[1].tolist() == [0.0] * 6
        assert abs(f_fast[12:]).max() == 0.0, "the kernel kept the dead member"
        assert abs(fast.end_actions[1]).max() == 0.0
        assert abs(f_fast[:12]).max() > 1.0
