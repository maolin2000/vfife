"""Fibre sections, cracked shear, temperature and tendons: the fifth review's
findings, one test each.

1. a step-level temperature kept acting after its step (fibre solver)
2. fibre creep was read at the END station of every Lobatto rule
3. the `fiber` strain monitor counted a shear law's rotation as curvature
4. `interpolation: quadratic` was one least-squares parabola; the curve as
   placed was not checked against the section
5. `elastic_shortening: compensate` assumed the origin is the centroid
7. misspelt `friction` keys were dropped
8. a bar's `area` / `material` were dropped
9. a patch's unknown keys were dropped
10. `axis` with a scalar `gradient` was dropped
11. fibre-section keys the shape does not read were accepted
12. the anchor-set reflection stopped at an element midpoint (over-drew)
13. a heated section with laws lacking `thermal_expansion` said nothing
14. a temperature on a bare truss2 was told about traction and pressure
15. `info` counted the bars twice and always said Euler-Bernoulli
"""
import copy
import pathlib

import numpy as np
import pytest

from vfife.api import _with_element, prepare_case
from vfife.cli import make_solver

ELASTIC = {"type": "linear_elastic", "E": 3.0e10, "nu": 0.2, "rho": 2500.0}
STRAND = {"type": "linear_elastic", "E": 1.95e11, "nu": 0.3, "rho": 7850.0}
RECT = {"type": "fiber", "material": "c", "shape": "rect", "width": 0.3, "height": 0.6, "ny": 2, "nz": 6}
T_SECTION = {"type": "fiber", "material": "c", "shape": "patches", "patches": [
    {"corners": [[-0.6, 0.0], [0.6, 0.0], [0.6, 0.2], [-0.6, 0.2]], "ny": 6, "nz": 4},
    {"corners": [[-0.1, -0.8], [0.1, -0.8], [0.1, 0.0], [-0.1, 0.0]], "ny": 1, "nz": 16}]}


def _mesh(td, L=10.0, n=10):
    p = pathlib.Path(td) / f"beam_{n}.msh"
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "4",
             '0 2 "left"', '0 3 "right"', '0 4 "mid"', '1 1 "beam"', "$EndPhysicalNames",
             "$Nodes", str(n + 1)]
    lines += [f"{i + 1} {L * i / n:.12g} 0 0" for i in range(n + 1)]
    lines += ["$EndNodes", "$Elements", str(n + 3), "1 15 2 2 2 1", f"2 15 2 3 3 {n + 1}",
              f"3 15 2 4 4 {n // 2 + 1}"]
    lines += [f"{e + 4} 1 2 1 1 {e + 1} {e + 2}" for e in range(n)]
    p.write_text("\n".join(lines + ["$EndElements"]) + "\n")
    return str(p)


def _case(td, section, materials=None, L=10.0, n=10, loads=None, constraints=None, steps=None,
          solver=None, monitors=None, extra_sections=None):
    secs = {"s": section, "strand": {"type": "bar", "material": "p", "shape": "general", "area": 2.0e-3}}
    secs.update(extra_sections or {})
    return {
        "name": "review5_fibre", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": _mesh(td, L, n)},
        "materials": materials or {"c": ELASTIC, "p": STRAND},
        "sections": secs,
        "assign": [{"region": "beam", "section": "s", "orientation": [0, 1, 0]}],
        "constraints": constraints if constraints is not None else [
            {"region": "left", "dofs": [1, 1, 1, 1, 1, 1]}],
        "loads": loads or [], "steps": steps or [{"name": "s1", "load_steps": 1}],
        "solver": {"mode": "static", "backend": "numpy", **(solver or {})},
        "output": {"path": str(pathlib.Path(td) / "o.h5"), "fields": ["displacement"],
                   "monitors": monitors or {}},
    }


def _build(case):
    prep = prepare_case(copy.deepcopy(case))
    return make_solver(_with_element(prep.case, prep.element, prep.case_path), prep.mesh, prep.material)


def _run(case):
    s = _build(case)
    if case["solver"]["mode"] == "static":
        from vfife import implicit
        implicit.run(s, output_every=10 ** 9, progress_every=10 ** 9)
    else:
        s.run(on_progress=lambda *a, **k: None)
    return s


# 1 ------------------------------------------------------------------------
def test_a_step_level_temperature_ends_with_its_step(tmp_path):
    """Step loads are not inherited: a gradient stated in step 2 must not act in
    step 3. The fibre solver returned early from `_refresh_imposed` and kept the
    field -- the tip stayed at -12.1 mm instead of going back to -2.1 mm."""
    mats = {"c": dict(ELASTIC, thermal_expansion=1e-5), "p": STRAND}
    heat = {"region": "beam", "temperature": {"gradient": [0, 20.0]}}
    case = _case(tmp_path, RECT, mats, loads=[{"region": "right", "vector": [0, 0, -1000.0]}],
                 steps=[{"name": "a", "load_steps": 1}, {"name": "heat", "load_steps": 1, "loads": [heat]},
                        {"name": "after", "load_steps": 1}],
                 monitors={"w": {"kind": "displacement", "region": "right", "dof": "z"}})
    w = _run(case).monitors["w"].values
    assert w[-2] == pytest.approx(w[0] - 1e-5 * 20.0 * 100 / 2, rel=2e-3)    # the bow, while heated
    assert w[-1] == pytest.approx(w[0], rel=1e-6)                             # and gone after


# 2 ------------------------------------------------------------------------
def test_fibre_creep_is_read_at_the_middle_station(tmp_path):
    """`argmin(|s|)` with s in [0, 1] is the element END for every Lobatto rule:
    a cantilever's creep came out 20% high. Linear creep with phi = 2 under a
    held tip load: tip = 3 x elastic."""
    curve = [[0, 0.0], [1000, 2.0], [3000, 6.0]]
    conc = {"type": "concrete02", "fpc": 3.0e7, "epsc0": 0.002, "fcu": 6.0e6, "epscu": 0.004,
            "ft": 1.0e8, "Ets": 1.5e9, "rat": 0.1, "rho": 2400.0, "creep": {"model": "curve", "curve": curve}}
    for rule, s_mid in (("gauss2", 0.211324865405187), ("lobatto3", 0.5), ("lobatto4", 0.5 * (1 - 1 / np.sqrt(5))),
                        ("lobatto5", 0.5)):
        sec = {"type": "fiber", "material": "c", "shape": "rect", "width": 0.2, "height": 0.2,
               "ny": 1, "nz": 2, "integration": rule}
        case = _case(tmp_path, sec, {"c": conc, "p": STRAND}, L=4.0, n=2,
                     loads=[{"region": "right", "vector": [0, 0, -2.0e3]}],
                     steps=[{"name": "load", "age": 28.0, "load_steps": 1},
                            {"name": "creep", "to_age": 1028.0, "load_steps": 12}],
                     solver={"mode": "relaxation", "age": 28.0, "tolerance": 1e-6, "max_steps_per_load": 100000})
        s = _build(case)
        assert s.gauss_s[s._creep_station] == pytest.approx(s_mid)
        if rule == "lobatto3":
            s.run(on_progress=lambda *a, **k: None)
            closed = -3 * 2.0e3 * 4.0 ** 3 / (3 * 3.0e10 * 1e-4)
            assert s.q[6 * 2 + 2] == pytest.approx(closed, rel=0.04)       # was +19.6%


# 3 ------------------------------------------------------------------------
def test_the_strain_monitor_takes_the_shear_rotation_out(tmp_path):
    """A member with a cracking shear law bends with its end rotations less
    gamma; `section_strains` (what a `fiber` strain monitor reads) left gamma
    in and read 6x the fibre's strain."""
    sec = dict(RECT, nz=12, shear={"direction": "z", "stiffness": 1.0e9, "cracked_stiffness": 1.0e8})
    s = _build(_case(tmp_path, sec, L=2.0, n=4, solver={"mode": "relaxation"}))
    rng = np.random.default_rng(3)
    s.q[:] = rng.normal(0.0, 1e-4, s.ndof)
    s._sl_gamma[:] = rng.normal(0.0, 2e-4, s.nel)
    s._internal_force_true()
    strains = s.section_strains()
    for e in range(s.nel):
        for gi in range(len(s.gauss_s)):
            eps0, ky, kz = strains[e, gi]
            plane = eps0 - s.fiber_y[e] * kz + s.fiber_z[e] * ky
            assert np.allclose(plane * 3.0e10, s.fiber_stress_rows()[e, gi], rtol=1e-9, atol=1e-6)


# 4 ------------------------------------------------------------------------
def _tendon(profile, **kw):
    return {"section": "strand", "profile": profile, "interpolation": "quadratic", "prestress": 1.0e9, **kw}


def test_a_quadratic_tendon_is_a_parabola_per_three_points(tmp_path):
    """One least-squares parabola put a five-point draped tendon straight at
    z = -0.08. Now 0-1-2 and 2-3-4 are each a parabola through their points."""
    from vfife.core.fiber_beam_solver import _piecewise_parabola
    prof = [[0, 0, 0.0], [5, 0, -0.4], [10, 0, 0.3], [15, 0, -0.4], [20, 0, 0.0]]
    sp = np.array([p[0] for p in prof], float); zv = np.array([p[2] for p in prof], float)
    assert np.allclose(_piecewise_parabola(sp, zv, sp), zv, atol=1e-15)
    sec = dict(RECT, width=0.4, height=1.0, ny=2, nz=10, tendons=[_tendon(prof)])
    s = _build(_case(tmp_path, sec, L=20.0, n=20))
    z = s.tendons[0]["z"]
    xm = np.arange(0.5, 20.0, 1.0)
    first = xm < 10
    a = np.polyfit(sp[:3], zv[:3], 2); b = np.polyfit(sp[2:], zv[2:], 2)
    assert np.allclose(z[first], np.polyval(a, xm[first]), atol=1e-12)
    assert np.allclose(z[~first], np.polyval(b, xm[~first]), atol=1e-12)


def test_an_even_quadratic_profile_and_a_curve_leaving_the_section_are_refused(tmp_path):
    sec = dict(RECT, width=0.4, height=1.0, tendons=[_tendon([[0, 0, 0], [5, 0, -0.3], [15, 0, -0.3], [20, 0, 0]])])
    with pytest.raises(Exception, match="an odd number of them"):
        _build(_case(tmp_path, sec, L=20.0, n=20))
    # every listed point is inside; the vertex between them is 0.37 m below the soffit
    sec = dict(RECT, width=0.4, height=1.0, tendons=[_tendon([[0, 0, 0.3], [4, 0, -0.45], [20, 0, 0.3]])])
    with pytest.raises(ValueError, match="outside the section"):
        _build(_case(tmp_path, sec, L=20.0, n=20))


# 5 ------------------------------------------------------------------------
@pytest.mark.parametrize("ez", [-0.6, 0.0, -0.1])
def test_compensated_elastic_shortening_on_a_t_section_off_its_centroid(tmp_path, ez):
    """The T's centroid is 0.1 m below its origin. N/A + M z/I about the origin
    gave 1227 MPa for 1200 at e = -0.6; the transformed section's 3x3 stiffness
    about the origin gives the stated stress back."""
    sec = dict(T_SECTION, tendons=[{"section": "strand", "profile": [[0, 0, ez], [10, 0, ez]],
                                    "prestress": 1.2e9, "elastic_shortening": "compensate"}])
    cons = [{"region": "beam", "dofs": [0, 1, 0, 1, 0, 1]}, {"region": "left", "dofs": [1, 1, 1, 1, 0, 1]},
            {"region": "right", "dofs": [0, 1, 1, 1, 0, 1]}]
    s = _run(_case(tmp_path, sec, constraints=cons, loads=[{"region": "mid", "vector": [0, 0, -1.0e3]}],
                   solver={"tolerance": 1e-7}))
    stress = s.fiber_stress_rows()[:, :, s.tendons[0]["column"]]
    assert stress.mean() == pytest.approx(1.2e9, rel=2e-4)


# 7 8 9 11 -----------------------------------------------------------------
@pytest.mark.parametrize("change, match", [
    (lambda s: s.update(tendons=[{"section": "strand", "profile": [[0, 0, -0.2], [10, 0, -0.2]], "jack": 1.2e9,
                                  "friction": {"mu": 0.2, "wobble": 0.003}}]), "does not read `wobble`"),
    (lambda s: s.update(tendons=[{"section": "strand", "profile": [[0, 0, -0.2], [10, 0, -0.2]], "jack": 1.2e9,
                                  "friction": {"mu": 0.2, "anchorage_set": 0.006}}]), "anchorage_set"),
    (lambda s: s.update(bars=[{"y": 0, "z": -0.25, "section": "strand", "area": 5e-3}]), "does not read `area`"),
    (lambda s: s.update(bars=[{"y": 0, "z": -0.25, "section": "strand", "material": "c"}]), "does not read `material`"),
    (lambda s: s.update(nt=20), "does not read `nt`"),
    (lambda s: s.update(t=0.05), "does not read `t`"),
    (lambda s: s.update(diameter=0.4), "does not read `diameter`"),
    (lambda s: s.update(cover=0.05), "without a core_material"),
])
def test_keys_a_fibre_section_does_not_read_are_refused_by_name(tmp_path, change, match):
    sec = dict(RECT)
    change(sec)
    with pytest.raises(Exception, match=match):
        _build(_case(tmp_path, sec))


def test_a_patch_and_a_tube_refuse_what_they_do_not_read(tmp_path):
    patch = {"type": "fiber", "material": "c", "shape": "patches", "patches": [
        {"corners": [[-0.15, -0.3], [0.15, -0.3], [0.15, 0.3], [-0.15, 0.3]], "nx": 2, "nz": 12}]}
    with pytest.raises(Exception, match="does not read `nx`"):
        _build(_case(tmp_path, patch))
    tube = {"type": "fiber", "material": "c", "shape": "tube", "diameter": 0.5, "t": 0.05, "width": 0.3}
    with pytest.raises(Exception, match="does not read `width`"):
        _build(_case(tmp_path, tube))


# 10 -----------------------------------------------------------------------
def test_axis_names_the_direction_of_a_scalar_gradient(tmp_path):
    from vfife.io.schema import TemperatureSpec
    assert TemperatureSpec(gradient=20.0, axis="z").gradient == [0.0, 20.0]
    assert TemperatureSpec(gradient=20.0).gradient == [20.0, 0.0]
    again = TemperatureSpec(**TemperatureSpec(gradient=20.0, axis="z").model_dump())
    assert again.gradient == [0.0, 20.0]                      # re-reads as itself
    with pytest.raises(ValueError, match="already names both axes"):
        TemperatureSpec(gradient=[1.0, 2.0], axis="z")
    mats = {"c": dict(ELASTIC, thermal_expansion=1e-5), "p": STRAND}
    s = _run(_case(tmp_path, RECT, mats, loads=[{"region": "beam", "temperature": {"gradient": 20.0, "axis": "z"},
                                                 }, {"region": "right", "vector": [0, 0, -1.0]}]))
    assert abs(s.q[6 * 10 + 1]) < 1e-9 and s.q[6 * 10 + 2] == pytest.approx(-0.01, rel=2e-3)


# 12 -----------------------------------------------------------------------
@pytest.mark.parametrize("n", [10, 20, 40])
def test_the_anchor_set_draws_in_exactly_what_it_states(n):
    """It stopped at the first element midpoint past the target: 6 mm drew in
    7.0 / 6.66 / 6.14 mm on 10 / 20 / 40 elements of the manual's 20 m tendon."""
    from vfife.elements.truss import tendon_profile
    L, f, E, d = 20.0, 0.45, 1.95e11, 0.006
    x = np.linspace(0, L, n + 1); z = -4 * f * x * (L - x) / L ** 2
    pts = np.column_stack([x, 0 * x, z]); conn = np.column_stack([np.arange(n), np.arange(1, n + 1)])
    chord = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s0 = tendon_profile(pts, conn, 1.3e9, 0.17, 0.0015, "start", None, E)
    s1 = tendon_profile(pts, conn, 1.3e9, 0.17, 0.0015, "start", d, E)
    assert float(np.sum((s0 - s1) * chord) / E) == pytest.approx(d, rel=1e-9)
    assert np.all(s1 <= s0 + 1e-6)


# 13 -----------------------------------------------------------------------
def test_preflight_warns_when_some_heated_fibres_have_no_thermal_expansion(tmp_path):
    mats = {"c": dict(ELASTIC, thermal_expansion=1e-5),
            "r": {"type": "rebar", "E": 2e11, "rho": 7850, "yield_strength": 5e8}, "p": STRAND}
    sec = dict(RECT, bars=[{"y": 0, "z": -0.25, "section": "b", "count": 4}])
    extra = {"b": {"type": "bar", "material": "r", "shape": "circle", "diameter": 0.025}}
    heated = _case(tmp_path, sec, mats, extra_sections=extra,
                   steps=[{"name": "a", "load_steps": 1, "loads": [{"region": "beam", "temperature": {"uniform": 30.0}}]}])
    codes = [i["code"] for i in prepare_case(heated).preflight["issues"]]
    assert "THERMAL_EXPANSION_PARTIAL" in codes
    mats["r"]["thermal_expansion"] = 0.0                        # an explicit zero is a choice
    codes = [i["code"] for i in prepare_case(_case(tmp_path, sec, mats, extra_sections=extra,
             loads=[{"region": "beam", "temperature": {"uniform": 30.0}}])).preflight["issues"]]
    assert "THERMAL_EXPANSION_PARTIAL" not in codes


# 14 -----------------------------------------------------------------------
def test_a_bare_truss_names_the_temperature_it_does_not_take(tmp_path):
    case = _case(tmp_path, {"type": "bar", "material": "p", "shape": "general", "area": 1e-3},
                 {"p": dict(STRAND, thermal_expansion=1.2e-5)}, L=4.0, n=4,
                 constraints=[{"region": "left", "dofs": [1, 1, 1]}, {"region": "right", "dofs": [1, 1, 1]}],
                 loads=[{"region": "beam", "temperature": {"uniform": 50.0}}], solver={"mode": "relaxation"})
    case["assign"] = [{"region": "all", "section": "s"}]
    with pytest.raises(ValueError, match="does not take a `temperature` load"):
        _run(case)


# 15 -----------------------------------------------------------------------
def test_info_counts_concrete_bars_and_tendons_once_and_names_the_theory(tmp_path):
    sec = dict(RECT, bars=[{"y": 0, "z": -0.25, "section": "strand", "count": 2}],
               tendons=[{"section": "strand", "profile": [[0, 0, -0.2], [10, 0, -0.2]], "prestress": 1e9}])
    text = _build(_case(tmp_path, sec)).summary()
    assert "12 fibers + 1 rebar + 1 tendon," in text and "Euler-Bernoulli," in text   # singular: v1.0
    text = _build(_case(tmp_path, dict(RECT, shear_stiffness=1e9))).summary()
    assert "Timoshenko" in text


# =========================================================================
# 1.0 re-review: the v1.0 fixes above, reviewed again
# =========================================================================
def _two_region_mesh(td, n=10, L=10.0):
    """A line of n elements: region `a` the first half, `b` the second, `beam` all."""
    p = pathlib.Path(td) / "two_regions.msh"
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "5",
             '0 2 "left"', '0 3 "right"', '1 1 "a"', '1 5 "b"', '1 6 "beam"', "$EndPhysicalNames",
             "$Nodes", str(n + 1)] + [f"{i + 1} {L * i / n:.12g} 0 0" for i in range(n + 1)]
    els = ["1 15 2 2 2 1", f"2 15 2 3 3 {n + 1}"]
    for e in range(n):
        els.append(f"{len(els) + 1} 1 2 {1 if e < n // 2 else 5} 1 {e + 1} {e + 2}")
    p.write_text("\n".join(lines + ["$EndNodes", "$Elements", str(len(els))] + els + ["$EndElements"]) + "\n")
    return str(p)


def _creep_tip(tmp_path, rule, fixed):
    curve = [[0, 0.0], [1000, 2.0], [3000, 6.0]]
    conc = {"type": "concrete02", "fpc": 3.0e7, "epsc0": 0.002, "fcu": 6.0e6, "epscu": 0.004,
            "ft": 1.0e8, "Ets": 1.5e9, "rat": 0.1, "rho": 2400.0, "creep": {"model": "curve", "curve": curve}}
    sec = {"type": "fiber", "material": "c", "shape": "rect", "width": 0.2, "height": 0.2,
           "ny": 1, "nz": 2, "integration": rule}
    tip = "right" if fixed == "left" else "left"
    case = _case(tmp_path, sec, {"c": conc, "p": STRAND}, L=4.0, n=4,
                 loads=[{"region": tip, "vector": [0, 0, -2.0e3]}],
                 constraints=[{"region": fixed, "dofs": [1, 1, 1, 1, 1, 1]}],
                 steps=[{"name": "load", "age": 28.0, "load_steps": 1},
                        {"name": "creep", "to_age": 1028.0, "load_steps": 12}],
                 solver={"mode": "relaxation", "age": 28.0, "tolerance": 1e-6, "max_steps_per_load": 100000})
    s = _build(case)
    s.run(on_progress=lambda *a, **k: None)
    closed = -3 * 2.0e3 * 4.0 ** 3 / (3 * 3.0e10 * 1e-4)
    return s, s.q[6 * (4 if fixed == "left" else 0) + 2] / closed


@pytest.mark.parametrize("rule", ["gauss2", "lobatto4"])
def test_149_an_even_rule_reads_creep_at_the_middle_whichever_end_is_fixed(tmp_path, rule):
    """gauss2 (the default) and lobatto4 have no middle station; v1.0 read
    the first of the two straddling it, 0.21 L from each element's node 0, so
    the same cantilever crept +6.7% fixed at x = 0 and -5.9% fixed at x = L.
    The mean of the two middle stations is the middle."""
    s, left = _creep_tip(tmp_path, rule, "left")
    assert len(s._creep_stations) == 2
    _s, right = _creep_tip(tmp_path, rule, "right")
    assert left == pytest.approx(right, rel=1e-9)
    assert left == pytest.approx(1.0, abs=0.01)             # was 1.067 / 0.941 (gauss2)


def test_149_compensate_on_a_planar_section_solves_around_the_empty_axis(tmp_path):
    """Every fibre and the tendon at y = 0 (rect ny: 1): no stiffness about z.
    np.linalg.solve raised "Singular matrix" (before v1.0: NaN)."""
    sec = dict(RECT, ny=1, tendons=[{"section": "strand", "profile": [[0, 0, -0.2], [10, 0, -0.2]],
                                     "prestress": 1.2e9, "elastic_shortening": "compensate"}])
    cons = [{"region": "beam", "dofs": [0, 1, 0, 1, 0, 1]}, {"region": "left", "dofs": [1, 1, 1, 1, 0, 1]},
            {"region": "right", "dofs": [0, 1, 1, 1, 0, 1]}]
    s = _run(_case(tmp_path, sec, constraints=cons, loads=[{"region": "mid", "vector": [0, 0, -1.0e3]}],
                   solver={"tolerance": 1e-8}))
    assert np.isfinite(s.fiber_prestrain).all()
    stress = s.fiber_stress_rows()[:, :, s.tendons[0]["column"]]
    assert stress.mean() == pytest.approx(1.2e9, rel=5e-4)


def _two_section_case(tmp_path, heated_region, step_level=False):
    mats = {"c": dict(ELASTIC, thermal_expansion=1e-5), "p": STRAND,
            "r": {"type": "rebar", "E": 2e11, "rho": 7850, "yield_strength": 5e8}}
    heat = [{"region": heated_region, "temperature": {"uniform": 30.0}}]
    case = _case(tmp_path, RECT, mats, loads=[] if step_level else heat,
                 steps=[{"name": "a", "load_steps": 1, **({"loads": heat} if step_level else {})}],
                 extra_sections={"s2": dict(RECT, bars=[{"y": 0, "z": -0.25, "section": "b", "count": 2}]),
                                 "b": {"type": "bar", "material": "r", "shape": "circle", "diameter": 0.02}})
    case["mesh"]["path"] = _two_region_mesh(tmp_path)
    case["assign"] = [{"region": "a", "section": "s", "orientation": [0, 1, 0]},
                      {"region": "b", "section": "s2", "orientation": [0, 1, 0]}]
    return [i for i in prepare_case(case).preflight["issues"] if i["code"] == "THERMAL_EXPANSION_PARTIAL"]


@pytest.mark.parametrize("region, step_level, warned", [
    ("a", False, False), ("a", True, False), ("b", False, True), ("b", True, True), ("all", False, True)])
def test_149_the_thermal_expansion_warning_is_for_sections_a_temperature_reaches(tmp_path, region, step_level, warned):
    """Section b's bar has no coefficient; only a temperature on b (or on all)
    reaches it. v1.0 warned whenever any temperature acted anywhere."""
    issues = _two_section_case(tmp_path, region, step_level)
    assert bool(issues) == warned
    if warned:
        assert issues[0]["details"]["section"] == "b"


@pytest.mark.parametrize("section, match", [
    (dict(T_SECTION, bars=[{"y": 0.4, "z": -0.5, "section": "strand"}]), "bars\\[0\\].*union of its patches"),
    (dict(T_SECTION, tendons=[{"section": "strand", "profile": [[0, 0.4, -0.5], [10, 0.4, -0.5]],
                               "prestress": 1e9}]), "tendons\\[0\\].*union of its patches"),
    ({"type": "fiber", "material": "c", "shape": "tube", "diameter": 0.6, "t": 0.1,
      "tendons": [{"section": "strand", "profile": [[0, 0.25, 0.25], [10, 0.25, 0.25]], "prestress": 1e9}]},
     "outer circle"),
])
def test_149_steel_beside_a_web_or_in_a_tubes_corner_is_outside_the_section(tmp_path, section, match):
    """The bounding rectangle let a bar or a tendon sit in the empty corner
    beside a T's web, or outside a tube in its square."""
    with pytest.raises(Exception, match=match):
        _build(_case(tmp_path, section))


def test_149_voids_count_as_inside_and_the_message_speaks_the_profiles_units(tmp_path):
    tube = {"type": "fiber", "material": "c", "shape": "tube", "diameter": 0.6, "t": 0.1,
            "tendons": [{"section": "strand", "profile": [[0, 0, 0], [10, 0, 0]], "prestress": 1e9}]}
    _build(_case(tmp_path, tube))                        # an external tendon in the bore
    _build(_case(tmp_path, dict(T_SECTION, tendons=[{"section": "strand", "prestress": 1e9,
                                                     "profile": [[0, 0.1, -0.5], [10, 0.1, -0.5]]}])))
    sec = dict(RECT, width=0.4, height=1.0, tendons=[_tendon([[0, 0, 0.3], [0.2, 0, -0.45], [1, 0, 0.3]],
                                                             normalised=True)])
    with pytest.raises(ValueError, match=r"at s = 0\.\d+ \(normalised\), outside the section"):
        _build(_case(tmp_path, sec, L=20.0, n=20))


def test_149_info_names_the_theory_per_section_and_counts_in_the_singular(tmp_path):
    case = _case(tmp_path, dict(RECT, shear={"direction": "z", "stiffness": 1.0e9, "cracked_stiffness": 1.0e8}),
                 solver={"mode": "relaxation"},
                 extra_sections={"s2": dict(RECT, tendons=[{"section": "strand", "prestress": 1e9,
                                                            "profile": [[0, 0, -0.2], [5, 0, -0.2]]}])})
    case["mesh"]["path"] = _two_region_mesh(tmp_path)
    case["assign"] = [{"region": "a", "section": "s", "orientation": [0, 1, 0]},
                      {"region": "b", "section": "s2", "orientation": [0, 1, 0]}]
    text = _build(case).summary()
    assert "theory per section" in text
    lines = [ln for ln in text.splitlines() if ln.strip().startswith("section")]
    assert lines[0].endswith("cracking shear law in series") and lines[1].endswith(", Euler-Bernoulli")
    assert "+ 1 tendon on" in lines[1]


def test_149_too_much_draw_in_from_both_ends_reports_a_positive_capacity():
    """`jack: both`: the profile rises again past the middle, and the old
    whole-profile area against the far end said "recovers only -3.60 mm"."""
    from vfife.elements.truss import tendon_profile
    n, L, f = 20, 20.0, 0.45
    x = np.linspace(0, L, n + 1); z = -4 * f * x * (L - x) / L ** 2
    pts = np.column_stack([x, 0 * x, z]); conn = np.column_stack([np.arange(n), np.arange(1, n + 1)])
    with pytest.raises(ValueError, match=r"recovers at most (\d+\.\d+) mm") as err:
        tendon_profile(pts, conn, 1.3e9, 0.17, 0.0015, "both", 0.006, 1.95e11)
    assert "-" not in err.value.args[0].split("at most")[1].split("mm")[0]
