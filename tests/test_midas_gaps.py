"""What nine real bridge models found in the Midas translation,
each on a model small enough to have a closed form:

* a section's properties as the file states them (`*SECT-PSCVALUE`, a VALUE
  section, a tapered one over a `*TS-GROUP`, a solid rectangle);
* a stub is a joint, not a softened beam, and only a stub is one -- a
  metre of a box girder is left as it is;
* a master and its slaves (`*RIGIDLINK`) are the solver's rigid links (1.21);
* a force and a moment on one node are two loads;
* an elastic link rigid in some directions, and the same link written in
  two boundary groups that follow one another."""
import json
import math

import numpy as np
import pytest
import yaml

from vfife.io import midas, midas_case
from test_midas import MCT

E, NU, W = 2.06e11, 0.3, 76980.0                 # N/m2, -, N/m3
MAT = f"*MATERIAL\n    1, STEEL, Q345 , 60, 0, , C, NO, 0.02, 2, {E:.4e}, {NU}, 1.2e-05, {W}, 0\n"
HEAD = "*VERSION\n   9.1.0\n*UNIT\n   N , M, KJ, C\n"
PIPE = "    2, DBUSER    , pipe , CC, 0, 0, 0, 0, 0, 0, YES, NO, P  , 2, 1.3, 0.022, 0, 0, 0, 0, 0, 0, 0, 0\n"
BOX = (0.93, 0.5, 0.5, 0.2, 0.5, 17.5)           # AREA, ASy, ASz, Ixx, Iyy, Izz of a steel box girder


def _model(text, tmp_path, name="g", **kw):
    p = tmp_path / f"{name}.mct"
    p.write_bytes(text.encode("gbk"))
    model = midas.read(str(p))
    kw.setdefault("shear", False)                    # the closed forms here are Euler's; the shear has a test of its own
    # the tendons as the solver reads them, to look into (the short form, the
    # case's own `tendons:`, is the same model: test_the_short_form_is_the_long_form)
    kw.setdefault("expanded", True)
    report = midas_case.convert(model, str(tmp_path), name=name, **kw)
    case = yaml.safe_load((tmp_path / f"{name}.yaml").read_text(encoding="utf-8"))
    case["mesh"]["path"] = str(tmp_path / f"{name}.msh")
    case["output"]["path"] = str(tmp_path / f"{name}.h5")
    rows = json.loads((tmp_path / f"{name}.map.json").read_text(encoding="utf-8"))["nodes"]
    return model, report, case, {int(n): r for n, r in rows.items()}


def _run(case):
    from vfife.api import run_case
    got = {}
    out = run_case(case, on_ready=lambda s, _p: got.__setitem__("s", s))
    assert out.status == "CONVERGED", out.message
    s = got["s"]
    return s.dofs.view(s.q), s.dofs.view(s.internal_force()[0])


def _girder(n=8, extra=""):
    nodes = "".join(f"   {i + 1}, {i}, 0, 0\n" for i in range(n + 1))
    elems = "".join(f"   {i + 1}, BEAM  ,    1,     1,   {i + 1},   {i + 2},     0,     0\n" for i in range(n))
    return (HEAD + "*NODE\n" + nodes + "*ELEMENT\n" + elems + MAT
            + "*SECT-PSCVALUE\n   SECT=1, PSC, box , CC, 0, 0, 0, 0, 0, 0, YES, NO, PSC\n"
            + "   " + ", ".join(str(v) for v in BOX) + "\n   0, 0, 0, 0\n   0, 0, 0, 0\n"
            + "*STLDCASE\n   自重 , D,\n"
            + f"*CONSTRAINT\n   1, 111100, \n   {n + 1}, 011000, \n"
            + extra + "*USE-STLD, 自重\n*SELFWEIGHT\n0, 0, -1, \n*ENDDATA\n")


def test_a_girder_of_short_stiff_segments_is_left_as_it_is_and_weighs_what_it_weighs(tmp_path):
    """A metre of a steel box girder has 12 E I / L^3 = 4e13 N/m sideways:
    over the limit, and not a stub. Softened by 8 (as it was), its weight
    went with its area: the deflection came out right and the reactions an
    eighth -- and with the weight kept, the deflection 8 times too large."""
    model, report, case, row = _model(_girder(), tmp_path)
    assert model.sections[1].props == BOX
    assert report["stubs_made_joints"] == 0
    assert report["members_over_the_limit_left_as_they_are"] == 8
    assert any("lengths of beam, not stubs" in n for n in report["notes"])
    (sec,) = [s for s in case["sections"].values() if s["type"] == "beam"]
    assert (sec["area"], sec["j"], sec["iy"], sec["iz"]) == pytest.approx((BOX[0], BOX[3], BOX[4], BOX[5]))
    q, f = _run(case)
    w, span = BOX[0] * W, 8.0
    assert q[row[5], 2] == pytest.approx(-5 * w * span ** 4 / (384 * E * BOX[4]), rel=0.03)
    # what the supports carry: the weight, less the half segment that hangs on each of them
    assert f[row[1], 2] + f[row[9], 2] == pytest.approx(w * span * 7 / 8, rel=1e-3)


def test_a_stub_is_a_joint_that_carries_the_moment(tmp_path):
    """5 mm of the pipe between two lengths of it. As a beam with its
    section divided until 12EI/L^3 was under the limit, its EI/L was 20
    N m/rad: a hinge (two spans of a real curved bridge met in a 2 mm stub
    and came apart). A connector, it is the joint it was drawn as."""
    text = (HEAD + "*NODE\n   1, 0, 0, 0\n   2, 5, 0, 0\n   3, 5.005, 0, 0\n   4, 10.005, 0, 0\n"
            "*ELEMENT\n   1, BEAM  ,    1,     2,     1,     2,     0,     0\n"
            "   2, BEAM  ,    1,     2,     2,     3,     0,     0\n   3, BEAM  ,    1,     2,     3,     4,     0,     0\n"
            + MAT + "*SECTION\n" + PIPE + "*STLDCASE\n   P , D,\n*CONSTRAINT\n   1, 111111, \n"
            "*USE-STLD, P\n*CONLOAD\n   4, 0, 0, -1.0e5, 0, 0, 0, ,\n*ENDDATA\n")
    _m, report, case, row = _model(text, tmp_path)
    assert report["stubs_made_joints"] == 1 and report["members_over_the_limit_left_as_they_are"] == 0
    assert report["members"] == 2 and report["connectors"] == 1
    (k,) = [s["stiffness"] for s in case["sections"].values() if s["type"] == "connector"]
    i, _j, _g = _pipe()
    r = math.sqrt(i / (math.pi / 4 * (1.3 ** 2 - 1.256 ** 2)))
    assert k[:3] == [1.0e13] * 3 and k[3:] == pytest.approx([E * i / (0.05 * r)] * 3)
    q, _f = _run(case)
    assert q[row[4], 2] == pytest.approx(-1.0e5 * 10.005 ** 3 / (3 * E * i), rel=0.01)


def test_a_stub_is_short_against_its_own_section(tmp_path):
    """The same 5 mm, of a pipe a hundred times smaller, is a length of beam."""
    text = (HEAD + "*NODE\n   1, 0, 0, 0\n   2, 5, 0, 0\n   3, 5.005, 0, 0\n"
            "*ELEMENT\n   1, BEAM  ,    1,     2,     1,     2,     0,     0\n"
            "   2, BEAM  ,    1,     2,     2,     3,     0,     0\n" + MAT + "*SECTION\n"
            + PIPE.replace("1.3, 0.022", "0.013, 0.0022")
            + "*CONSTRAINT\n   1, 111111, \n*ENDDATA\n")
    _model_, report, _case, _row = _model(text, tmp_path)
    assert report["stubs_made_joints"] == 0 and report["members"] == 2


def _arm(extra_nodes="", links="", loads=""):
    """A cantilever along x, 4 m, of the pipe; whatever hangs on its tip."""
    return (HEAD + "*NODE\n   1, 0, 0, 0\n   2, 4, 0, 0\n" + extra_nodes
            + "*ELEMENT\n   1, BEAM  ,    1,     2,     1,     2,     0,     0\n" + MAT + "*SECTION\n" + PIPE
            + "*STLDCASE\n   P , D,\n*CONSTRAINT\n   1, 111111, \n" + links + "*USE-STLD, P\n*CONLOAD\n" + loads + "*ENDDATA\n")


def _pipe():
    i = math.pi / 64 * (1.3 ** 4 - 1.256 ** 4)
    return i, 2 * i, E / (2 * (1 + NU))


def test_a_master_and_its_slaves_are_rigid_links(tmp_path):
    """Two slaves tied in all six on arms across the tip (stiff beams before
    1.21), each reached by a bar from the tip that the link moves rigidly
    (no strain: it carries nothing). One tied in all but x above it, which
    no member reaches, has nothing to hold it along x: left out, and
    counted (as is a slave no member reaches at all: the mesh has no node
    that no member uses)."""
    text = _arm("   10, 4, 2, 0\n   11, 4, -2, 0\n   12, 4, 0, 1\n",
                "*RIGIDLINK\n   2, 111111, 10 11, \n   2, 011111, 12, \n", "   10, 0, 0, -1000, 0, 0, 0, ,\n")
    text = text.replace("   1, BEAM  ,    1,     2,     1,     2,     0,     0\n",
                        "   1, BEAM  ,    1,     2,     1,     2,     0,     0\n"
                        "   2, TRUSS ,    1,     2,     2,    10,     0,     0\n"
                        "   3, TRUSS ,    1,     2,     2,    11,     0,     0\n")
    model, report, case, row = _model(text, tmp_path)
    assert len(model.rigid_links) == 2
    assert report["rigid_arms"] == 0 and report["rigid_link_slaves"] == 2
    assert report["left_out"] == {"*RIGIDLINK slave that no member reaches": 1}
    assert [lk["dofs"] for lk in case["rigid_links"]] == [[1] * 6]
    q, _f = _run(case)
    i, j, g = _pipe()
    bend = 1000 * 4 ** 3 / (3 * E * i)
    twist = 1000 * 2 * 4 / (g * j)
    assert q[row[2], 2] == pytest.approx(-bend, rel=0.01)
    assert q[row[10], 2] == pytest.approx(-bend - 2 * twist, rel=0.01)       # the arm turns with the tip
    assert q[row[11], 2] == pytest.approx(-bend + 2 * twist, rel=0.01)


def test_a_force_and_a_moment_on_one_node_are_two_loads(tmp_path):
    _model_, _report, case, row = _model(_arm(loads="   2, 0, 0, -1000, 0, 500, 0, ,\n"), tmp_path)
    kinds = sorted(("vector" if "vector" in ld else "moment") for ld in case["loads"])
    assert kinds == ["moment", "vector"]
    assert len({ld["region"] for ld in case["loads"]}) == 1
    q, _f = _run(case)
    i, _j, _g = _pipe()
    # a positive moment about y turns the x axis towards -z
    assert q[row[2], 2] == pytest.approx(-1000 * 4 ** 3 / (3 * E * i) - 500 * 4 ** 2 / (2 * E * i), rel=0.01)


def test_a_link_rigid_in_some_directions_and_one_rigid_in_all(tmp_path):
    links = ("*ELASTICLINK\n"
             "   1, 2, 10, GEN  ,     0, YES, YES, YES, YES, YES, YES, 0, 0, 0, 0, 0, 0, NO, 0.5, 0.5, \n"
             "   2, 2, 11, GEN  ,     0, YES, NO, NO, NO, NO, NO, 0, 2e6, 3e6, 4e6, 5e6, 6e6, NO, 0.5, 0.5, \n")
    model, report, case, _row = _model(_arm("   10, 4, 2, 0\n   11, 4, 0, -1\n   12, 4, 0, -2\n", links
                                            + "*CONSTRAINT\n   11, 111111, \n", ""), tmp_path)
    assert model.links[0].rigid == (True,) * 6 and model.links[1].rigid == (True,) + (False,) * 5
    assert report["rigid_arms"] == 1 and report["connectors"] == 1
    (k,) = [s for s in case["sections"].values() if s["type"] == "connector"]
    # along -Z: the link's own x is the global Z
    assert k["stiffness"][2] == pytest.approx(1.0e10)
    assert sorted(k["stiffness"][:2]) == pytest.approx([2e6, 3e6])
    assert any("rigid in some directions" in n for n in report["notes"])


def _two_groups(second):
    text = MCT.replace("    11, 20000, 0, 0\n", "    11, 10000, 0, -500\n")
    text = text.replace("   1, 111111, 拱脚固结\n", "   1, 111111, 拱脚固结\n   11, 111000, 拱脚固结\n")
    first = "     1, 2, 11, GEN  ,     0, NO, NO, NO, NO, NO, NO, 0.1, 1e+06, 1e+06, 0.01, 0.01, 0.02, NO, 0.5, 0.5, 扣地锚\n"
    text = text.replace("     1, 3, 11, GEN  ,     0, NO, NO, NO, NO, NO, NO, 0.1, 1e+06, 1e+06, 0.01, 0.01, 0.02, NO, 0.5, 0.5, 扣地锚\n",
                        first + first.replace("     1, 2", "     3, 2").replace("扣地锚", "成桥").replace("0.1, 1e+06", second))
    text = text.replace("     2, 3, 11, RIGID,     0, NO, 0.5, 0.5, 扣地锚\n", "")
    text = text.replace("   扣地锚\n*LOAD-GROUP", "   扣地锚\n   成桥\n*LOAD-GROUP")
    return text.replace("   DBNDR=扣地锚\n", "   DBNDR=扣地锚\n   ABNDR=成桥, DEFORMED\n")


def test_the_same_link_in_the_boundary_group_that_follows_is_one_link_kept(tmp_path):
    """A closure's bearings are written in the closure's boundary group and
    again in the finished bridge's, which comes on as the first goes off.
    The first was kept and removed with its group: the bridge lost its
    bearings at the stage that should have kept them."""
    _model_, report, case, _row = _model(_two_groups("0.1, 1e+06"), tmp_path)
    assert report["connectors"] == 1 and not any("link" in k for k in report["left_out"])
    (a,) = [a for a in case["assign"] if case["sections"][a["section"]]["type"] == "connector"]
    s1, s2, s3 = case["steps"]
    assert a["region"] in s2["activate"] and a["region"] not in s3.get("deactivate", [])
    # a different link there is not followed: said, and the first kept and removed as it was written
    _model_, report, case, _row = _model(_two_groups("0.2, 1e+06"), tmp_path, name="other")
    assert report["connectors"] == 1
    assert report["left_out"]["a different link on the two nodes of an earlier one (a bearing that changes; the first is kept)"] == 1
    (a,) = [a for a in case["assign"] if case["sections"][a["section"]]["type"] == "connector"]
    assert a["region"] in case["steps"][2]["deactivate"]


# ---- sections the file states -------------------------------

def _sections(body, extra=""):
    text = (HEAD + "*NODE\n   1, 0, 0, 0\n   2, 2, 0, 0\n   3, 4, 0, 0\n   4, 8, 0, 0\n"
            "*ELEMENT\n   1, BEAM  ,    1,     1,     1,     2,     0,     0\n"
            "   2, BEAM  ,    1,     1,     2,     3,     0,     0\n   3, BEAM  ,    1,     1,     3,     4,     0,     0\n"
            + MAT + body + "*CONSTRAINT\n   1, 111111, \n" + extra + "*ENDDATA\n")
    return text


def test_a_value_section_and_a_solid_rectangle(tmp_path):
    body = ("*SECTION\n    1, VALUE     , v , CC, 0, 0, 0, 0, 0, 0, YES, NO, SB , BUILT, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0\n"
            "       0.31, 0.2, 0.2, 0.011, 0.022, 0.033, 0, 0\n       0.5, 0.25, 0.5, 0.25, 0, 0, 0, 0\n")
    model, _report, case, _row = _model(_sections(body), tmp_path)
    assert model.sections[1].props == (0.31, 0.2, 0.2, 0.011, 0.022, 0.033)
    sec = case["sections"]["s1_m1"]
    assert (sec["area"], sec["j"], sec["iy"], sec["iz"]) == pytest.approx((0.31, 0.011, 0.022, 0.033))
    r = midas_case.section_properties(midas.Section(name="r", kind="DBUSER", shape="SB", dims=(1.0, 0.5)), 1.0)
    assert r["area"] == pytest.approx(0.5) and r["iy"] == pytest.approx(0.5 / 12) and r["iz"] == pytest.approx(0.125 / 12)
    assert r["j"] == pytest.approx(0.229 * 1.0 * 0.5 ** 3, rel=0.01)                 # Roark: b/a = 2


def test_a_tapered_section_runs_over_its_group(tmp_path):
    """Its two ends as the file states them; each element of the run is a
    prismatic member with the properties at its own mid-length -- at 1/8,
    3/8 and 3/4 of the way here -- and on its own, at the middle."""
    body = ("*SECT-PSCVALUE\n   SECT=1, TAPERED, t , CC, 0, 0, 0, 0, 0, 0, YES, NO, PSC\n"
            "   1.0, 0.5, 0.5, 0.1, 1.0, 2.0\n   0, 0, 0, 0\n   0, 0, 0, 0\n"
            "   3.0, 0.5, 0.5, 0.3, 5.0, 4.0\n   0, 0, 0, 0\n   0, 0, 0, 0\n")
    model, _report, case, _row = _model(_sections(body, "*TS-GROUP\n   run, 1to3, LINEAR, , , LINEAR, , , 0\n"), tmp_path)
    assert model.sections[1].props_j == (3.0, 0.5, 0.5, 0.3, 5.0, 4.0) and model.taper_groups == [("run", [1, 2, 3])]
    got = sorted((s["area"], s["iy"]) for s in case["sections"].values() if s["type"] == "beam")
    assert np.array(got) == pytest.approx(np.array([(1.25, 1.5), (1.75, 2.5), (2.5, 4.0)]))
    _model_, _report, case, _row = _model(_sections(body), tmp_path, name="alone")
    (sec,) = [s for s in case["sections"].values() if s["type"] == "beam"]
    assert (sec["area"], sec["iy"], sec["iz"], sec["j"]) == pytest.approx((2.0, 3.0, 3.0, 0.2))


# ---- found in review ---------------------------------------------------------

def _staged_links(rows, stages):
    """The cantilever of test_midas in stages, a prop under node 2, the given
    link rows on it and the given boundary groups' (on, off) stages."""
    text = MCT.replace("    11, 20000, 0, 0\n", "    11, 10000, 0, -500\n")
    text = text.replace("   1, 111111, 拱脚固结\n", "   1, 111111, 拱脚固结\n   11, 111000, 拱脚固结\n")
    row = "     {i}, 2, 11, GEN  ,     0, NO, NO, NO, NO, NO, NO, 0.1, 1e+06, 1e+06, 0.01, 0.01, 0.02, NO, 0.5, 0.5, {g}\n"
    a = text.index("*ELASTICLINK"); b = text.index("*USE-STLD")
    text = text[:a] + "*ELASTICLINK\n" + "".join(row.format(i=i + 1, g=g) for i, g in enumerate(rows)) + text[b:]
    text = text.replace("   扣地锚\n*LOAD-GROUP", "   扣地锚\n" + "".join(f"   {g}\n" for g in stages) + "*LOAD-GROUP")
    names = ["安装S1", "安装S2", "松索"]
    for g, (on, off) in stages.items():
        text = text.replace(f"   NAME={names[on]}, ", f"   NAME={names[on]}, ", 1)
        head = text.index(f"   NAME={names[on]}, "); line = text.index("\n", head) + 1
        text = text[:line] + f"   ABNDR={g}, DEFORMED\n" + text[line:]
        if off is not None:
            head = text.index(f"   NAME={names[off]}, "); line = text.index("\n", head) + 1
            text = text[:line] + f"   DBNDR={g}\n" + text[line:]
    return text


def test_two_links_are_one_only_where_one_takes_over_from_the_other(tmp_path):
    def life(rows, stages, name):
        _m, report, case, _row = _model(_staged_links(rows, stages), tmp_path, name=name)
        (a,) = [a for a in case["assign"] if case["sections"][a["section"]]["type"] == "connector"]
        on = [k for k, s in enumerate(case["steps"]) if a["region"] in s.get("activate", [])]
        off = [k for k, s in enumerate(case["steps"]) if a["region"] in s.get("deactivate", [])]
        return on, off, {k: v for k, v in report["left_out"].items() if "link" in k}
    # the later one written first: still one link, from the first's on
    assert life(["后", "先"], {"先": (1, 2), "后": (2, None)}, "rev") == ([1], [], {})
    # on together: Midas has two springs there, and it is said
    on, off, left = life(["先", "后"], {"先": (1, None), "后": (2, None)}, "both")
    assert (on, off) == ([1], []) and list(left) == [
        "identical link on the two nodes of another while that one is on (one kept; Midas adds the two)"]
    # a gap between the two: the one written first alone, as it was written
    on, off, left = life(["后", "先"], {"先": (0, 1), "后": (2, None)}, "gap")
    assert (on, off) == ([2], []) and list(left) == ["link on the same two nodes as another, built again after a gap (the first is kept)"]


def test_rigid_between_two_points_at_one_place_is_six_stiff_springs(tmp_path):
    """In kN and mm, to see the units of `link_rigid`; a rotation's flag too."""
    links = ("     1, 3, 11, GEN  ,     0, YES, YES, YES, YES, YES, YES, 0, 0, 0, 0, 0, 0, NO, 0.5, 0.5, 拱脚固结\n")
    a = MCT.index("     1, 3, 11, GEN"); b = MCT.index("*USE-STLD")
    text = MCT[:a] + links + "*RIGIDLINK\n   3, 111111, 12, 拱脚固结\n" + MCT[b:]
    text = text.replace("    11, 20000, 0, 0\n", "    11, 20000, 0, 0\n    12, 20000, 0, 0\n")
    text = text.replace("   1, 111111, 拱脚固结\n", "   1, 111111, 拱脚固结\n   11, 111111, 拱脚固结\n   12, 111111, 拱脚固结\n")
    _m, report, case, _row = _model(text, tmp_path)
    # the *RIGIDLINK beside it ties a node nothing else reaches (it was six
    # stiff springs too, before 1.21): a rigid link is no member, and its slave
    # is left out with the support on it, counted
    assert report["connectors"] == 1 and report["rigid_arms"] == 0 and report["rigid_link_slaves"] == 0
    assert report["left_out"]["*RIGIDLINK slave that no member reaches"] == 1
    assert report["left_out"]["support on a node no member reaches"] == 1
    ks = [s["stiffness"] for s in case["sections"].values() if s["type"] == "connector"]
    assert ks == [pytest.approx([1.0e10] * 6)]
    part = links.replace("YES, YES, YES, YES, YES, YES, 0, 0, 0, 0, 0, 0", "NO, NO, NO, NO, YES, NO, 1, 2, 3, 4, 0, 6")
    _m, _report, case, _row = _model(text.replace(links, part), tmp_path, name="part")
    ks = sorted(tuple(s["stiffness"]) for s in case["sections"].values() if s["type"] == "connector")
    assert ks[0] == pytest.approx((1e6, 2e6, 3e6, 4.0, 1.0e10, 6.0))             # kN/mm -> N/m; kN mm/rad is N m/rad


def test_a_stiff_member_without_an_area_is_not_a_stub(tmp_path):
    body = ("*SECTION\n    1, VALUE     , v , CC, 0, 0, 0, 0, 0, 0, YES, NO, SB , BUILT, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0\n"
            "       0, 0, 0, 0.011, 22, 33, 0, 0\n")
    _m, report, _case, _row = _model(_sections(body), tmp_path)
    assert report["stubs_made_joints"] == 0 and report["members_over_the_limit_left_as_they_are"] == 2          # the two 2 m ones


def test_a_tapered_section_of_sizes_takes_its_sizes_along_it(tmp_path):
    body = ("*SECTION\n    1, TAPERED   , t , CC, 0, 0, 0, 0, 0, 0, 0, 0, YES, NO, SB , 1, 1, USER, \n"
            "       1.0, 0.5, 0, 0, 0, 0, 0, 0, 2.0, 0.5, 0, 0, 0, 0, 0, 0\n")
    model, _report, case, _row = _model(_sections(body), tmp_path)
    sec = model.sections[1]
    assert (sec.shape, sec.db, sec.dims[:2], sec.dims_j[:2]) == ("SB", "USER", (1.0, 0.5), (2.0, 0.5))
    (s,) = [s for s in case["sections"].values() if s["type"] == "beam"]
    assert s["area"] == pytest.approx(0.75) and s["iy"] == pytest.approx(0.5 * 1.5 ** 3 / 12)   # the depth at the middle, cubed


def test_what_stands_on_a_stub_still_weighs_on_its_support(tmp_path):
    text = (HEAD + "*NODE\n   1, 0, 0, 0\n   2, 0.005, 0, 0\n   3, 5.005, 0, 0\n"
            "*ELEMENT\n   1, BEAM  ,    1,     2,     1,     2,     0,     0\n"
            "   2, BEAM  ,    1,     2,     2,     3,     0,     0\n" + MAT + "*SECTION\n" + PIPE
            + "*STLDCASE\n   自重 , D,\n*CONSTRAINT\n   1, 111111, \n*USE-STLD, 自重\n*SELFWEIGHT\n0, 0, -1, \n*ENDDATA\n")
    _m, report, case, row = _model(text, tmp_path)
    assert report["stubs_made_joints"] == 1
    _q, f = _run(case)
    area = math.pi / 4 * (1.3 ** 2 - 1.256 ** 2)
    assert f[row[1], 2] == pytest.approx(area * W * 5.0, rel=1e-3)                      # the beam's; the stub's own 5 mm is not


# ---- point springs, and a load case of the finished bridge --

def test_a_point_spring_is_a_link_to_a_ground_node_of_its_own(tmp_path):
    k = 3.0e7
    spring = f"*SPRING\n   2, LINEAR, NO, NO, NO, NO, NO, NO, 0, 0, {k}, 0, 0, 0, NO, 0, 0, 0, 0, 0, 0, , 0, 0, 0, 0, 0\n"
    model, report, case, row = _model(_arm(links=spring, loads="   2, 0, 0, -1000, 0, 0, 0, ,\n"), tmp_path)
    assert model.springs == [([2], (False,) * 6, (0.0, 0.0, k, 0.0, 0.0, 0.0), "")]
    assert report["connectors"] == 1 and report["nodes"] == 3 and report["links_made_to_turn_with_their_ends"] == 0
    q, _f = _run(case)
    i, _j, _g = _pipe()
    assert q[row[2], 2] == pytest.approx(-1000 / (k + 3 * E * i / 4 ** 3), rel=0.01)
    # a direction the row fixes is `link_rigid`; a spring of nothing is counted
    rows = spring.replace("NO, NO, NO, NO, NO, NO, 0, 0", "YES, NO, NO, NO, NO, NO, 0, 0") + \
        "   1, LINEAR, NO, NO, NO, NO, NO, NO, 0, 0, 0, 0, 0, 0, NO, 0, 0, 0, 0, 0, 0, , 0, 0, 0, 0, 0\n"
    _m, report, case, _row = _model(_arm(links=rows, loads=""), tmp_path, name="fx")
    (c,) = [s for s in case["sections"].values() if s["type"] == "connector"]
    assert c["stiffness"] == pytest.approx([1.0e10, 0.0, k, 0.0, 0.0, 0.0])
    assert report["left_out"] == {"point spring of no stiffness": 1}


def test_a_load_case_of_the_finished_bridge_is_one_more_step(tmp_path):
    text = MCT.replace("   平台 , CS, \n", "   平台 , CS, \n   风 , W, \n")
    text = text.replace("*STAGE    ; Define", "*USE-STLD, 风\n*CONLOAD\n 2, 0, 7, 0, 0, 0, 0, ,\n*STAGE    ; Define")
    _m, report, case, _row = _model(text, tmp_path, cases=("风",))
    assert report["steps"] == 4 and case["steps"][3]["name"] == "风"
    (ld,) = case["steps"][3]["loads"]
    assert ld["vector"] == [0.0, 7000.0, 0.0] and ld["keep"] is True
    _m, report, case, _row = _model(text, tmp_path, name="without")
    assert report["steps"] == 3 and report["left_out"]["load whose load group no stage activates"] == 1
    with pytest.raises(ValueError, match="no load case '雪'"):
        _model(text, tmp_path, name="none", cases=("雪",))


def _stack(first="2, 10", extra=""):
    """The cantilever's tip (node 2) and node 10 at one place, made one by a
    RIGID link; a bearing from node 10 down to node 11, which is fixed: it
    carries along its own axis (the vertical) and nothing else."""
    links = (f"*ELASTICLINK\n   1, {first}, RIGID,     0, NO, 0.5, 0.5, \n"
             "   2, 10, 11, GEN  ,     0, NO, NO, NO, NO, NO, NO, 1e12, 0, 0, 0, 0, 0, NO, 0.5, 0.5, \n")
    return _arm("   10, 4, 0, 0\n   11, 4, 0, -0.3\n", links + "*CONSTRAINT\n   11, 111111, \n" + extra,
                "   2, 0, 0, -1000, 0, 0, 0, ,\n")


@pytest.mark.parametrize("first", ["2, 10", "10, 2"])
def test_a_bearing_under_a_node_set_a_millimetre_aside_is_still_vertical(tmp_path, first):
    """Found in review: its axes were taken from the moved node, it leaned,
    and it held 4e8 N/m in the direction it slides in; and the node above
    it, no longer on a rigid ARM, was given a rotation spring."""
    from vfife.api import run_case
    _m, report, case, row = _model(_stack(first), tmp_path)
    assert report["connectors"] == 2 and report["links_made_to_turn_with_their_ends"] == 0
    ks = sorted((s["stiffness"] for s in case["sections"].values() if s["type"] == "connector"), key=lambda k: k[0])
    assert ks[0] == [0.0, 0.0, 1.0e12, 0.0, 0.0, 0.0] and ks[1] == pytest.approx([1.0e10] * 6)
    got = {}
    out = run_case(case, on_ready=lambda s_, _p: got.__setitem__("s", s_))
    assert out.status == "CONVERGED", out.message
    X = got["s"].mesh.nodes
    assert X[row[2]] == pytest.approx([4.0, 0.0, 0.0], abs=1e-12)                 # the beam's node stays; node 10 moves
    assert X[row[10]] == pytest.approx([4.001, 0.0, 0.0], abs=1e-12)
    q = got["s"].dofs.view(got["s"].q)
    assert abs(q[row[2], 2]) < 1000 / 1.0e10 * 1.5                               # it stands on the bearing


def test_a_spring_on_a_node_that_was_set_aside_gets_a_place_of_its_own(tmp_path):
    spring = "*SPRING\n   10, LINEAR, NO, NO, NO, NO, NO, NO, 0, 3e7, 0, 0, 0, 0, NO, 0, 0, 0, 0, 0, 0, , 0, 0, 0, 0, 0\n"
    _m, report, case, row = _model(_stack(extra=spring), tmp_path)
    assert report["connectors"] == 3
    (k,) = [s["stiffness"] for s in case["sections"].values() if s["type"] == "connector" and s["stiffness"][1] == 3e7]
    assert k == [0.0, 3e7, 0.0, 0.0, 0.0, 0.0]                                    # no rotation spring to the ground
    _run(case)                                                                     # (three nodes at one place: no two left there)


def test_a_millimetre_in_a_model_written_in_millimetres(tmp_path):
    a = MCT.index("     2, 3, 11, RIGID")
    text = MCT[:a] + MCT[MCT.index("*USE-STLD"):]                                 # the GEN link 3-11 alone: at one place
    text = text.replace("   1, 111111, 拱脚固结\n", "   1, 111111, 拱脚固结\n   11, 111111, 拱脚固结\n")
    _m, report, _case, _row = _model(text, tmp_path)
    assert any("1 mm along X" in n for n in report["notes"])
    msh = (tmp_path / "g.msh").read_text(encoding="utf-8")
    rows = [ln.split() for ln in msh[msh.index("$Nodes"):msh.index("$EndNodes")].splitlines()[2:]]
    at = sorted(tuple(float(c) for c in r[1:]) for r in rows if abs(float(r[1]) - 20.0) < 0.01)
    assert np.array(at) == pytest.approx(np.array([(20.0, 0.0, 0.0), (20.001, 0.0, 0.0)]), abs=1e-12)


def test_a_case_a_stage_already_applies_is_not_taken_out_of_it(tmp_path):
    p = tmp_path / "m.mct"
    p.write_bytes(MCT.encode("gbk"))
    model = midas.read(str(p))
    with pytest.raises(ValueError, match="already applied by a stage"):
        midas_case.convert(model, str(tmp_path), name="m", cases=("平台",))
    with pytest.raises(ValueError, match="no nodal or member load"):
        midas_case.convert(model, str(tmp_path), name="m", cases=("K1",))
    with pytest.raises(ValueError, match="each once"):
        midas_case.convert(model, str(tmp_path), name="m", cases="平台")


def test_a_spring_row_that_is_not_read_is_counted(tmp_path):
    rows = ("*SPRING\n   2, COMP, NO, NO, NO, NO, NO, NO, 0, 0, 1, 0, 0, 0, NO, 0, 0, 0, 0, 0, 0, , 0, 0, 0, 0, 0\n"
            "   2, LINEAR, 0, 0, 5, 0, 0, 0, \n")
    _m, report, _case, _row = _model(_arm(links=rows), tmp_path)
    assert report["connectors"] == 0
    assert report["left_out"] == {"point spring that is not linear, or whose row is not in the form this reads": 2}


def test_a_part_on_pins_in_one_line_is_held_from_spinning(tmp_path):
    """A side span cast on falsework: beams in one line, pinned. Nothing
    holds its twist, and the Newton iteration stopped on it (a real
    continuous girder, at the stage that casts its side spans)."""
    pinned = _girder().replace("   1, 111100, \n", "   1, 111000, \n")
    _m, report, case, row = _model(pinned, tmp_path)
    assert report["parts_held_from_spinning"] == 1
    assert [c["dofs"] for c in case["constraints"]][-1] == [0, 0, 0, 1, 0, 0]          # about X, its own line
    q, _f = _run(case)
    assert q[row[5], 2] == pytest.approx(-5 * BOX[0] * W * 8.0 ** 4 / (384 * E * BOX[4]), rel=0.03)
    # a rotation held somewhere: left as it is
    _m, report, _case, _row = _model(_girder(), tmp_path, name="held")
    assert report["parts_held_from_spinning"] == 0
    # a branch hanging off it spins with it; one that stands on a support of its own holds it
    bent = pinned.replace("   9, 8, 0, 0\n", "   9, 8, 0, 0\n   20, 4, 3, 0\n").replace(
        "*MATERIAL", "   20, BEAM  ,    1,     1,   5,   20,     0,     0\n*MATERIAL")
    _m, report, _case, _row = _model(bent, tmp_path, name="bent")
    assert report["parts_held_from_spinning"] == 1
    _m, report, case, _row = _model(bent.replace("   9, 011000, \n", "   9, 011000, \n   20, 001000, \n"), tmp_path, name="stood")
    assert report["parts_held_from_spinning"] == 0
    _run(case)


def _staged_girder(nodes, elems, groups, supports, stages):
    """nodes {id: (x, y, z)}, elems {id: (a, b)}, groups {name: [elements]}, supports [(nodes, flags, group)],
    stages [(element groups on, boundary groups on, boundary groups off)]; self weight from the first."""
    t = HEAD + "*NODE\n" + "".join(f"   {i}, {x}, {y}, {z}\n" for i, (x, y, z) in nodes.items())
    t += "*ELEMENT\n" + "".join(f"   {i}, BEAM  ,    1,     1,   {a},   {b},     0,     0\n" for i, (a, b) in elems.items())
    t += "*GROUP\n" + "".join(f"   {g}, , {' '.join(map(str, es))}, 0\n" for g, es in groups.items())
    t += "*BNDR-GROUP\n" + "".join(f"   {g}\n" for g in sorted({g for _n, _f, g in supports}))
    t += "*LOAD-GROUP\n   自重\n" + MAT + "*SECT-PSCVALUE\n   SECT=1, PSC, box , CC, 0, 0, 0, 0, 0, 0, YES, NO, PSC\n   "
    t += ", ".join(str(v) for v in BOX) + "\n   0, 0, 0, 0\n   0, 0, 0, 0\n*STLDCASE\n   自重 , CS,\n"
    t += "*CONSTRAINT\n" + "".join(f"   {n}, {fl}, {g}\n" for n, fl, g in supports)
    t += "*USE-STLD, 自重\n*SELFWEIGHT\n0, 0, -1, 自重\n*STAGE\n"
    for i, (ae, ab, db) in enumerate(stages):
        t += f"   NAME=S{i}, 10, YES, NO, NO, 5\n"
        t += ("   AELEM=" + ", ".join(f"{g}, 0" for g in ae) + "\n") if ae else ""
        t += ("   ABNDR=" + ", ".join(f"{g}, DEFORMED" for g in ab) + "\n") if ab else ""
        t += ("   DBNDR=" + ", ".join(db) + "\n") if db else ""
        t += "   ALOAD=自重, FIRST\n" if i == 0 else ""
    return t + "*STAGE-CTRL\n  YES, , EXTERNAL, YES\n*ENDDATA\n"


def test_the_hold_against_spinning_is_each_stages_own(tmp_path):
    """Found in review. A main part clamped at node 1; a side span in one
    line, on pins that come on a stage BEFORE it is built and are shared
    by a falsework group that goes off while it still stands alone; then
    the closure joins the two, and the hold is gone."""
    nodes = {1: (0, 0, 0), 2: (2, 0, 0), 3: (4, 0, 0), 11: (5, 0, 0), 12: (6, 0, 0), 13: (7, 0, 0), 14: (8, 0, 0), 15: (9, 0, 0)}
    elems = {1: (1, 2), 2: (2, 3), 3: (11, 12), 4: (12, 13), 5: (13, 14), 6: (14, 15), 7: (3, 11)}
    text = _staged_girder(nodes, elems, {"main": [1, 2], "side": [3, 4, 5, 6], "closure": [7]},
                          [("1", "111111", "P"), ("11 15", "111000", "B"), ("11to15", "111000", "F")],
                          [(["main"], ["P", "B"], []), (["side"], ["F"], []), ([], [], ["F"]), (["closure"], [], [])])
    _m, report, case, _row = _model(text, tmp_path)
    assert report["parts_held_from_spinning"] == 1
    twist = [[c["dofs"] for c in s.get("constraints", []) if c["dofs"][:3] == [0, 0, 0]] for s in case["steps"]]
    assert twist == [[], [[0, 0, 0, 1, 0, 0]], [[0, 0, 0, 1, 0, 0]], []]             # while it stands alone, and not kept
    assert not any(c.get("keep") for s in case["steps"] for c in s.get("constraints", []) if c["dofs"][:3] == [0, 0, 0])
    _run(case)


def test_the_hold_goes_when_something_else_holds_the_twist(tmp_path):
    """A girder on two pins, then a cross beam from its middle out to a
    support: kept, the hold took 3% of the cross beam's reaction."""
    nodes = {i + 1: (i, 0, 0) for i in range(9)}
    nodes.update({20: (4, 2, 0), 21: (4, 4, 0)})
    elems = {i + 1: (i + 1, i + 2) for i in range(8)}
    elems.update({20: (5, 20), 21: (20, 21)})
    text = _staged_girder(nodes, elems, {"girder": list(range(1, 9)), "cross": [20, 21]},
                          [("1 9", "111000", "P"), ("21", "001000", "Q")],
                          [(["girder"], ["P"], []), (["cross"], ["Q"], [])])
    _m, report, case, row = _model(text, tmp_path)
    assert [len(s.get("constraints", [])) for s in case["steps"]] == [1, 1]           # the hold, then Q's support alone
    _q, f = _run(case)
    w = BOX[0] * W
    assert f[row[21], 2] == pytest.approx(w * 4 / 2 - w, rel=1e-3)                    # a simply supported cross beam's end
    assert abs(f[row[1], 3]) < 1.0                                                    # no twisting moment left at the pin


def test_an_h_with_two_different_flanges_and_the_steel_of_jtg_d64():
    """A real truss's floor beams have a wider bottom flange (the 5th and
    6th sizes): 10% of their area was missing, and with the steel at 76.98
    instead of the 78.5 kN/m3 Midas has for JTG D64 the girder weighed 3.5%
    less than in Midas' reactions."""
    h = midas_case.section_properties(midas.Section(name="h", kind="DBUSER", shape="H",
                                                    dims=(1.05, 0.52, 0.028, 0.032, 0.72, 0.032, 0, 0)), 1.0)
    hw = 1.05 - 0.064
    area = 0.52 * 0.032 + 0.72 * 0.032 + hw * 0.028
    assert h["area"] == pytest.approx(area)
    zc = (0.52 * 0.032 * (1.05 - 0.016) + hw * 0.028 * (0.032 + hw / 2) + 0.72 * 0.032 * 0.016) / area
    iy = (0.52 * 0.032 ** 3 / 12 + 0.52 * 0.032 * (1.05 - 0.016 - zc) ** 2 + 0.028 * hw ** 3 / 12
          + hw * 0.028 * (0.032 + hw / 2 - zc) ** 2 + 0.72 * 0.032 ** 3 / 12 + 0.72 * 0.032 * (0.016 - zc) ** 2)
    assert h["iy"] == pytest.approx(iy) and zc < 1.05 / 2
    assert h["iz"] == pytest.approx((0.032 * 0.52 ** 3 + 0.032 * 0.72 ** 3 + hw * 0.028 ** 3) / 12)
    same = midas_case.section_properties(midas.Section(name="h", kind="DBUSER", shape="H", dims=(0.4, 0.3, 0.01, 0.02, 0.3, 0.02)), 1.0)
    plain = midas_case.section_properties(midas.Section(name="h", kind="DBUSER", shape="H", dims=(0.4, 0.3, 0.01, 0.02)), 1.0)
    assert same == pytest.approx(plain)
    d64 = midas.Material(name="Q345qd", kind="STEEL", E=2.06e11, standard="JTG D64-2015(S) Q345")
    gb = midas.Material(name="Q345", kind="STEEL", E=2.06e11, standard="GB12(S) Q345")
    assert midas_case._weight(d64, 1.0, 1.0)[0] == 78.5e3 and midas_case._weight(gb, 1.0, 1.0)[0] == 76.98e3


def test_a_rigid_arm_is_rigid_whatever_its_length(tmp_path):
    """A pile cap's node and its two piles, 16 m to either side: the arm was
    one section for every length, softer across than the pile is along,
    and the cap's load did not reach the piles as a cap's does. A load at
    a quarter of the cap: three quarters on the near pile, one on the far
    (the one section gave 0.84 and 0.29)."""
    conc = MAT.replace(f"{E:.4e}", "3.0000e+10")
    solid = "    2, DBUSER    , pile , CC, 0, 0, 0, 0, 0, 0, YES, NO, SR , 2, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0\n"
    text = (HEAD + "*NODE\n   1, -16, 0, -20\n   2, -16, 0, 0\n   3, 16, 0, -20\n   4, 16, 0, 0\n   5, 0, 0, 0\n   6, -8, 0, 0\n"
            "*ELEMENT\n   1, BEAM  ,    1,     2,     1,     2,     0,     0\n   2, BEAM  ,    1,     2,     3,     4,     0,     0\n"
            + conc + "*SECTION\n" + solid + "*STLDCASE\n   P , D,\n*CONSTRAINT\n   1 3, 111111, \n"
            "*ELASTICLINK\n   1, 5, 2, RIGID,     0, NO, 0.5, 0.5, \n   2, 5, 4, RIGID,     0, NO, 0.5, 0.5, \n"
            "   3, 5, 6, RIGID,     0, NO, 0.5, 0.5, \n"
            "*USE-STLD, P\n*CONLOAD\n   6, 0, 0, -1.0e7, 0, 0, 0, ,\n*ENDDATA\n")
    _m, report, case, row = _model(text, tmp_path)
    assert report["rigid_arms"] == 3
    arms = [s for n, s in case["sections"].items() if n.startswith("rigid_arm")]
    assert len(arms) == 2                                                          # 16 m and 8 m: a section each
    e_arm = case["materials"]["rigid"]["E"]
    for s, reach in zip(sorted(arms, key=lambda s: s["area"]), (8.0, 16.0)):
        assert e_arm * s["area"] / reach == pytest.approx(5.0e12) and 12 * e_arm * s["iy"] / reach ** 3 == pytest.approx(5.0e12)
    q, _f = _run(case)
    k = 3.0e10 * math.pi / 4 * 1.5 ** 2 / 20.0                                     # a pile, along
    assert q[row[2], 2] == pytest.approx(-0.75e7 / k, rel=0.02) and q[row[4], 2] == pytest.approx(-0.25e7 / k, rel=0.02)


def test_a_tapered_h_whose_bottom_flange_is_written_at_one_end_only():
    sec = midas.Section(name="t", kind="TAPERED", shape="H", db="USER",
                        dims=(1.0, 0.5, 0.02, 0.03, 0, 0, 0, 0), dims_j=(2.0, 0.5, 0.02, 0.03, 0.8, 0.04, 0, 0))
    mid = midas_case.section_properties(sec, 1.0)
    want = midas_case.section_properties(midas.Section(name="m", kind="DBUSER", shape="H",
                                                       dims=(1.5, 0.5, 0.02, 0.03, 0.65, 0.035)), 1.0)
    assert mid["area"] == pytest.approx(want["area"]) and mid["iy"] == pytest.approx(want["iy"])


# ---- a section set off its nodes ----------------------------

def _top_aligned(offset="CT", load="   2, 1.0e6, 0, 0, 0, 0, 0, ,\n"):
    rect = f"    1, DBUSER    , deck , {offset}, 0, 0, 0, 0, 0, 0, YES, NO, SB , 2, 1.0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0\n"
    return (HEAD + "*NODE\n   1, 0, 0, 0\n   2, 4, 0, 0\n   3, 8, 0, 0\n"
            "*ELEMENT\n   1, BEAM  ,    1,     1,     1,     2,     0,     0\n   2, BEAM  ,    1,     1,     2,     3,     0,     0\n"
            + MAT + "*SECTION\n" + rect + "*STLDCASE\n   P , D,\n*CONSTRAINT\n   1, 111111, \n*USE-STLD, P\n*CONLOAD\n"
            + load.replace("   2,", "   3,") + "*ENDDATA\n")


def test_a_member_drawn_by_its_top_is_on_its_centroids_line(tmp_path):
    """A girder's nodes are at the top of its section (offset CT). A pull
    along the nodes' line is then a pull and a moment on the member, and
    the node's own displacement has the section's turn times the depth in
    it: on a real rigid frame the lengthwise displacement was 0.7 to 0.85
    of Midas' with the member left on its nodes' line. It is an end
    offset (1.21): the member stays on its nodes, no node or arm is added."""
    _m, report, case, row = _model(_top_aligned(), tmp_path)
    assert report["members_set_off_their_nodes"] == 2 and report["rigid_arms"] == 0 and report["nodes"] == 3
    assert not any("offset" in k for k in report["left_out"])
    (beam,) = [a for a in case["assign"] if "end_offsets" in a]
    assert beam["end_offsets"] == {"per": "member", "i": [0.0, 0.0, -0.5], "j": [0.0, 0.0, -0.5], "axial": "full"}
    q, _f = _run(case)
    P, L, e, A, I = 1.0e6, 8.0, 0.5, 0.5, 0.5 / 12
    assert q[row[3], 0] == pytest.approx(P * L / (E * A) + P * e * e * L / (E * I), rel=0.01)
    assert q[row[3], 2] == pytest.approx(-P * e * L ** 2 / (2 * E * I), rel=0.01)       # pulled above the centroid: the tip dips
    # by its bottom, the other way; set off sideways as well, only the height is followed, and said
    _m, _report, case, row = _model(_top_aligned("CB"), tmp_path, name="cb")
    q, _f = _run(case)
    assert q[row[3], 2] == pytest.approx(P * e * L ** 2 / (2 * E * I), rel=0.01)
    _m, report, _case, _row = _model(_top_aligned("LT"), tmp_path, name="lt")
    assert report["left_out"] == {"section set off sideways (LT): only its height is followed": 2}
    # on its centroid (CC) nothing is added
    _m, report, case, row = _model(_top_aligned("CC"), tmp_path, name="cc")
    assert report["members_set_off_their_nodes"] == 0 and report["nodes"] == 3
    assert not any("end_offsets" in a for a in case["assign"])
    q, _f = _run(case)
    assert q[row[3], 0] == pytest.approx(P * L / (E * A), rel=0.01)


def _box_cantilever(offset, nodes=None, extra_nodes="", extra=""):
    nodes = nodes or [(i, 0.0, 0.0) for i in range(9)]
    text = _girder().replace(", CC, ", f", {offset}, ").replace("   0, 0, 0, 0\n   0, 0, 0, 0\n", "   2, 2, 0.8, 1.2\n   0, 0, 0, 0\n")
    a = text.index("*NODE\n"); b = text.index("*ELEMENT")
    text = text[:a] + "*NODE\n" + "".join(f"   {i + 1}, {x}, {y}, {z}\n" for i, (x, y, z) in enumerate(nodes)) + extra_nodes + text[b:]
    return text.replace("   1, 111100, \n   9, 011000, \n", "   1, 111111, \n" + extra)


def test_a_stiff_girder_drawn_by_its_top_bends_as_one_on_its_centroid(tmp_path):
    """Found in review, on a section the size of a girder's: with the arm of
    the common stiffness a cantilever clamped at its top node deflected 12%
    too much, and on a vertical curve -- two members' own up a hair apart at
    every joint, so two centroid nodes and two arms in series -- 79%."""
    _m, report, case, row = _model(_box_cantilever("CC"), tmp_path, name="cc")
    q, _f = _run(case)
    want = q[row[9], 2]
    _m, report, case, row = _model(_box_cantilever("CT"), tmp_path, name="ct")
    assert report["members_set_off_their_nodes"] == 8 and report["nodes"] == 9
    q, _f = _run(case)
    assert q[row[9], 2] == pytest.approx(want, rel=0.005)
    curve = [(i, 0.0, -i * i / 10000.0) for i in range(9)]                          # a vertical curve of 5000 m
    _m, report, case, row = _model(_box_cantilever("CT", curve), tmp_path, name="curve")
    assert report["nodes"] == 9
    q, _f = _run(case)
    assert q[row[9], 2] == pytest.approx(want, rel=0.01)


def test_a_centroid_that_is_a_node_of_the_model(tmp_path):
    """A pier's top drawn at the girder's centroid, under its node. The girder
    is on its own nodes (1.21), so the pier holds it only where the model
    joins the two -- as in Midas. With the centroid nodes it was merged with
    the pier whether the model joined them or not."""
    pier = "   20, BEAM  ,    1,     2,   20,   21,     0,     0\n"

    def text(link=""):
        t = _box_cantilever("CT", [(4.0 * i, 0.0, 0.0) for i in range(9)],
                            extra_nodes="   20, 32, 0, -0.8\n   21, 32, 0, -5\n", extra="   21, 111111, \n" + link)
        return t.replace("*MATERIAL", pier + "*MATERIAL").replace("*SECT-PSCVALUE", "*SECTION\n" + PIPE + "*SECT-PSCVALUE")
    _m, report, case, row = _model(text(), tmp_path, name="apart")
    assert report["nodes"] == 9 + 2 and report["rigid_arms"] == 0
    q_free, _f = _run(case)
    _m, _r, case, row_a = _model(_box_cantilever("CT", [(4.0 * i, 0.0, 0.0) for i in range(9)]), tmp_path, name="alone")
    q_alone, _f = _run(case)
    assert q_free[row[9], 2] == pytest.approx(q_alone[row_a[9], 2], rel=1e-6)       # the pier is not in it
    _m, report, case, row = _model(text("*ELASTICLINK\n   1, 9, 20, RIGID,     0, NO, 0.5, 0.5, \n"), tmp_path, name="joined")
    assert report["rigid_arms"] == 1
    q, _f = _run(case)
    assert abs(q[row[9], 2]) < 0.05 * abs(q_free[row[9], 2])                         # propped by the pier


def test_a_stub_at_a_free_end_and_one_over_a_bearing(tmp_path):
    """Found in review: made to wait like an elastic link, a stub at a free
    end was never built and its load was lost; and its node, on no beam any
    more, had the bearing under it turned into a moment joint."""
    stub = ("   3, 4.005, 0, 0\n", "   2, BEAM  ,    1,     2,     2,     3,     0,     0\n")
    text = _arm(stub[0], loads="   3, 0, 0, -1.0e5, 0, 0, 0, ,\n").replace("*MATERIAL", stub[1] + "*MATERIAL")
    _m, report, case, row = _model(text, tmp_path)
    assert report["stubs_made_joints"] == 1 and report["left_out"] == {}
    assert 0.0 < report["stub_weight_left_out_N"] < 50.0
    q, _f = _run(case)
    i, _j, _g = _pipe()
    assert q[row[3], 2] == pytest.approx(-1.0e5 * 4.005 ** 3 / (3 * E * i), rel=0.01)
    # a bearing (a spring upward only) under the stub's far node: a propped cantilever, and no rotation spring
    spring = "*SPRING\n   3, LINEAR, NO, NO, NO, NO, NO, NO, 0, 0, 1e12, 0, 0, 0, NO, 0, 0, 0, 0, 0, 0, , 0, 0, 0, 0, 0\n"
    text = _arm(stub[0], links=spring, loads="   2, 0, 0, 0, 0, 1.0e6, 0, ,\n").replace("*MATERIAL", stub[1] + "*MATERIAL")
    _m, report, case, row = _model(text, tmp_path, name="prop")
    assert report["links_made_to_turn_with_their_ends"] == 0
    (k,) = [s_["stiffness"] for s_ in case["sections"].values() if s_["type"] == "connector" and s_["stiffness"][0] == 0.0]
    assert k == [0.0, 0.0, 1e12, 0.0, 0.0, 0.0]
    q, _f = _run(case)
    assert q[row[2], 4] == pytest.approx(1.0e6 * 4.0 / (4 * E * i), rel=0.02)          # M L / 4EI at the propped end


def test_a_guided_support_in_a_nodes_own_axes(tmp_path):
    """A curved bridge's bearings: held across the bridge, free along it,
    in axes turned about Z at each node. Read as global they held the wrong
    direction (a real curved girder moved 2.6 times Midas' lengthwise)."""
    axis = "*LOCALAXIS\n   2, 1, 0, 0, 30\n"
    text = _arm(links=axis + "*CONSTRAINT\n   2, 100000, \n", loads="   2, 0, 1.0e5, 0, 0, 0, 0, ,\n")
    model, report, case, row = _model(text, tmp_path)
    assert model.local_axes == {2: (0.0, 0.0, 30.0)}
    assert report["connectors"] == 1 and any("own axes" in n for n in report["notes"])
    q, _f = _run(case)
    i, _j, _g = _pipe()
    ka, kb = E * math.pi / 4 * (1.3 ** 2 - 1.256 ** 2) / 4.0, 3 * E * i / 4.0 ** 3
    d = (math.cos(math.radians(30)), math.sin(math.radians(30)))
    lam = (d[1] * 1.0e5 / kb) / (d[0] ** 2 / ka + d[1] ** 2 / kb + 1.0 / 1.0e10)        # the support's force, along d (it is `link_rigid` stiff)
    want = (-lam * d[0] / ka, (1.0e5 - lam * d[1]) / kb)
    assert q[row[2], 0] == pytest.approx(want[0], rel=0.02) and q[row[2], 1] == pytest.approx(want[1], rel=0.02)
    assert q[row[2], 0] * d[0] + q[row[2], 1] * d[1] == pytest.approx(lam / 1.0e10, rel=0.02)   # along the held direction: the link's give
    # held in both x and y it is a support as it was, in any axes
    _m, report, case, _row = _model(_arm(links=axis + "*CONSTRAINT\n   2, 110000, \n"), tmp_path, name="both")
    assert report["connectors"] == 0 and [c["dofs"] for c in case["constraints"]][-1] == [1, 1, 0, 0, 0, 0]


def test_a_beam_gives_way_in_shear_where_the_file_says_how_much(tmp_path):
    """Midas' beam is a Timoshenko beam with the section's ASy, ASz. A deep
    girder on short spans deflects visibly more for it, and on a soft middle
    support carries more to that support: a real curved bridge's pier bearings
    were 18% under Midas' without it."""
    _m, _report, case, row = _model(_girder(), tmp_path, shear=True)
    (sec,) = [s_ for s_ in case["sections"].values() if s_["type"] == "beam"]
    assert sec["theory"] == "timoshenko" and (sec["ay"], sec["az"]) == (BOX[1], BOX[2])
    q, _f = _run(case)
    w, span, g = BOX[0] * W, 8.0, E / (2 * (1 + NU))
    bending, shearing = 5 * w * span ** 4 / (384 * E * BOX[4]), w * span ** 2 / (8 * g * BOX[2])
    assert shearing > 0.3 * bending
    assert q[row[5], 2] == pytest.approx(-(bending + shearing), rel=0.03)
    # a user shape: a pipe's is half its area; the default is with shear
    p = tmp_path / "pipe.mct"
    p.write_bytes(_arm(loads="   2, 0, 0, -1000, 0, 0, 0, ,\n").encode("gbk"))
    midas_case.convert(midas.read(str(p)), str(tmp_path), name="pipe")
    (sec,) = [s_ for s_ in yaml.safe_load((tmp_path / "pipe.yaml").read_text(encoding="utf-8"))["sections"].values() if s_["type"] == "beam"]
    assert sec["theory"] == "timoshenko" and sec["az"] == pytest.approx(0.5 * sec["area"])


def test_the_shear_area_is_the_one_of_the_direction_and_a_fixed_piers_arm_is_stiff(tmp_path):
    """ASy is for the shear across (local y), ASz for the shear in the depth:
    under its weight a girder gives way in ASz. And a rigid link between a
    pier and a girder -- its ends carry more in turning than an arm of the
    common stiffness -- has a section of its own (found on a real rigid
    frame; capped, because an arm of a few centimetres sized without one
    stalled the iteration)."""
    text = _girder().replace("0.93, 0.5, 0.5, ", "0.93, 0.3, 0.6, ")
    _m, _report, case, row = _model(text, tmp_path, shear=True)
    q, _f = _run(case)
    w, span, g = BOX[0] * W, 8.0, E / (2 * (1 + NU))
    assert q[row[5], 2] == pytest.approx(-(5 * w * span ** 4 / (384 * E * BOX[4]) + w * span ** 2 / (8 * g * 0.6)), rel=0.02)
    link = "*ELASTICLINK\n   1, 9, 20, RIGID,     0, NO, 0.5, 0.5, \n"
    text = _box_cantilever("CC", [(4.0 * i, 0.0, 0.0) for i in range(9)], extra_nodes="   20, 32, 0, -0.5\n   21, 32, 0, -5\n",
                           extra="   21, 111111, \n" + link)
    text = text.replace("*MATERIAL", "   20, BEAM  ,    1,     2,   20,   21,     0,     0\n*MATERIAL").replace(
        "*SECT-PSCVALUE", "*SECTION\n" + PIPE + "*SECT-PSCVALUE")
    _m, report, case, _row = _model(text, tmp_path, name="pier")
    (own,) = [v for k, v in case["sections"].items() if k.startswith("rigid_arm_own_")]
    e_arm = case["materials"]["rigid"]["E"]
    turning = 4 * E * BOX[5] / 4.0                                                 # the girder's 4EI/L (its stiffer way)
    assert e_arm * own["iy"] / 0.5 == pytest.approx(min(30 * turning, 10 * 1.0e13 * 0.5 ** 2 / 12), rel=1e-3)
    _run(case)


def test_a_concrete_grade_has_the_poisson_ratio_of_midas_database():
    c = midas.Material(name="C55", kind="CONC", E=3.55e10, standard="JTG04(RC) C55")
    s_ = midas.Material(name="Q345", kind="STEEL", E=2.06e11, standard="JTG D64-2015(S) Q345")
    assert midas_case._poisson(c) == 0.2 and midas_case._poisson(s_) == 0.3


# ---- tendons ------------------------------------------------

STRAND = "    2, STEEL, strand , 0, 0, , C, NO, 0.02, 2, 1.9500e+11, 0.3, 1.2e-05, 78500, 0\n"


def _tendon(slip=0.0, mu=0.0, k=0.0, jack="BEGIN", ref="straight", z=-0.3, force=1.0e7, group="PSG", count=0, zs=None,
            how="FORCE"):
    """A tendon 0.3 m under the centroid of the girder of `_girder` (8 m,
    pinned and on a roller), along all of it."""
    head = f"   T, INTERNAL, 2, 0.002, 0.05, 2, 0, {mu}, {k}, 1.86e9, 1.57e9, POST, {slip}, {slip}, YES, 0, NO, 0.3, NO, 1.86e9, 0, YES, 0, 0, 0\n"
    if ref == "straight":
        line = "        STRAIGHT, 0, 0, 0, X, 0, 0\n        0, YES, Y, 0\n"
        zs = zs or f"        Z=0, {z}, NO, 0, 0, NONE, , , , \n        Z=8, {z}, NO, 0, 0, NONE, , , , \n"
    else:
        line = f"        ELEMENT, END-I, 1, I-J\n        0, YES, 0, {z}\n"
        zs = "        Z=0, 0, NO, 0, 0, NONE, , , , \n        Z=8, 0, NO, 0, 0, NONE, , , , \n"
    second = f"        , AUTO1, , , YES, {count}, 0, 0\n" if count else "        , AUTO1, , , NO, , 0, 0\n"
    return ("*TDN-PROPERTY\n" + head + "*TDN-PROFILE\n   NAME=P1, T, 1to8, 0, 0, ROUND, 2D\n" + second
            + line + "        Y=0, 0, NO, 0, 0, NONE, , , , \n        Y=8, 0, NO, 0, 0, NONE, , , , \n" + zs
            + f"*USE-STLD, PS\n*TDN-PRESTRESS\n P1, {how}, {jack}, {force}, {force}, 0, {group}\n")


def _girder_with(tendon):
    text = _girder().replace(MAT, MAT + STRAND).replace("*STLDCASE\n   自重 , D,\n", "*STLDCASE\n   自重 , D,\n   PS , PS,\n")
    return text.replace("*USE-STLD, 自重\n*SELFWEIGHT\n0, 0, -1, \n", tendon)


def test_a_straight_tendon_bonded_under_the_centroid(tmp_path):
    """Jacked to P: the girder shortens by PL/EA and bows up by P e L^2 / 8EI
    (the tendon has no stiffness while it is jacked), and the bar keeps P."""
    _m, report, case, row = _model(_girder_with(_tendon()), tmp_path)
    assert report["tendons"] == 1 and report["tendon_bars"] == 8 and report["jacked_cables"] == 0
    mp = json.loads((tmp_path / "g.map.json").read_text(encoding="utf-8"))["tendons"]["P1"]
    assert [round(s_[2]) for s_ in mp["segments"]] == [1.0e7] * 8
    bars = [a for a in case["assign"] if case["sections"][a["section"]]["type"] == "bar"]
    assert len(bars) == 8 and all(b_["jack"] == pytest.approx(1.0e7) for b_ in bars)     # one region a bar
    assert case["materials"][case["sections"][bars[0]["section"]]["material"]]["rho"] < 1.0e-3     # weightless
    q, _f = _run(case)
    P, e, L = 1.0e7, 0.3, 8.0
    assert q[row[5], 2] == pytest.approx(P * e * L ** 2 / (8 * E * BOX[4]), rel=0.02)
    assert q[row[9], 0] - q[row[1], 0] == pytest.approx(-P * L / (E * BOX[0]), rel=0.02)
    # on an element's line, set off by the same 0.3 m: the same
    _m, report, case, row = _model(_girder_with(_tendon(ref="element", z=-0.3)), tmp_path, name="el")
    q2, _f = _run(case)
    assert q2[row[5], 2] == pytest.approx(q[row[5], 2], rel=1e-6)


def test_a_tendons_friction_and_draw_in():
    """k per metre along it, and 0.1 mm of draw-in taken up over l =
    sqrt(slip EA / p) with p the friction's slope: within l the force is the
    friction curve mirrored about its value at l."""
    s = np.linspace(0.0, 8.0, 801)
    P0, kw, ea, slip = 1.0e7, 0.0015, 1.95e11 * 0.002, 1.0e-4
    f = P0 * np.exp(-kw * s)
    out = midas_case._drawn_in(s, f, slip, ea)
    lost = np.sum(0.5 * ((f - out)[1:] + (f - out)[:-1]) * np.diff(s))
    assert lost == pytest.approx(slip * ea, rel=1e-3)
    ell = math.sqrt(slip * ea / (P0 * kw))
    assert out[0] == pytest.approx(2 * P0 * math.exp(-kw * ell) - P0, rel=1e-3)
    assert np.allclose(out[s > ell + 0.02], f[s > ell + 0.02])


def test_a_tendon_jacked_with_friction_from_one_end_or_both(tmp_path):
    _m, _report, _case, _row = _model(_girder_with(_tendon(k=0.0015)), tmp_path)
    seg = json.loads((tmp_path / "g.map.json").read_text(encoding="utf-8"))["tendons"]["P1"]["segments"]
    assert [s_[2] for s_ in seg] == pytest.approx([1.0e7 * math.exp(-0.0015 * (i + 0.5)) for i in range(8)], rel=1e-4)
    _m, _report, _case, _row = _model(_girder_with(_tendon(k=0.0015, jack="BOTH")), tmp_path, name="both")
    seg = json.loads((tmp_path / "both.map.json").read_text(encoding="utf-8"))["tendons"]["P1"]["segments"]
    assert [s_[2] for s_ in seg] == pytest.approx([1.0e7 * math.exp(-0.0015 * min(i + 0.5, 7.5 - i)) for i in range(8)], rel=1e-4)


def test_a_short_tendon_jacked_at_both_ends_draws_in_on_both_jacks_force(tmp_path):
    """Jacked at both ends with a draw-in that is not taken up by the middle
    (JTG D62's l_f > l/2): each half is the friction curve mirrored about
    the middle, dropped evenly by what is left of slip EA; the middle does
    not move (v1.16: each end drawn in on its own curve over the
    whole tendon, and the larger taken, kept too much)."""
    slip, P0, kw, ea, L = 2.0e-3, 1.0e7, 0.0015, 1.95e11 * 0.002, 8.0
    _m, _report, _case, _row = _model(_girder_with(_tendon(k=kw, jack="BOTH", slip=slip)), tmp_path, name="short")
    seg = json.loads((tmp_path / "short.map.json").read_text(encoding="utf-8"))["tendons"]["P1"]["segments"]
    f = lambda x: P0 * math.exp(-kw * x)                                             # noqa: E731
    half = L / 2
    taken = 2.0 * (P0 * (1.0 - math.exp(-kw * half)) / kw - half * f(half))         # the mirror about the middle
    drop = (slip * ea - taken) / half
    want = [2.0 * f(half) - f(min(x, L - x)) - drop for x in np.arange(0.5, L, 1.0)]
    assert [s_[2] for s_ in seg] == pytest.approx(want, rel=1e-4)
    assert drop > 0.0 and max(s_[2] for s_ in seg) < f(half) - 0.5 * drop


@pytest.mark.parametrize("slip_b, slip_e, bend", [(6e-3, 6e-3, 0.0), (6e-3, 0.0, 0.0), (6e-3, 6e-3, 0.3),
                                                   (2e-3, 6e-3, 0.3), (1e-4, 1e-4, 0.3)])
def test_both_ends_draw_in_takes_each_slip_and_leaves_no_step(slip_b, slip_e, bend):
    """Unequal slips, or the friction all on one side (a turn of `bend`
    radians at 7 m of a 10 m tendon): each end loses its slip EA, the force
    is continuous, nowhere above the force as jacked, and with small slips
    it is each end's own mirrored curve (review)."""
    s = np.linspace(0.0, 10.0, 1001)
    P0, mu, kw, ea = 1.0e7, 0.2, 0.0015, 1.95e11 * 0.002
    th = np.where(s > 7.0, bend, 0.0)
    ahead = P0 * np.exp(-(mu * th + kw * s))
    back = P0 * np.exp(-(mu * (bend - th) + kw * (10.0 - s)))
    f = np.maximum(ahead, back)
    out = midas_case._drawn_in_both(s, f, slip_b, slip_e, ea)
    loss = f - out
    assert (loss >= -1e-6).all()
    assert (np.abs(np.diff(out)) <= np.abs(np.diff(f)) + 1e-6 * P0).all()             # no step anywhere
    area = np.sum(0.5 * (loss[1:] + loss[:-1]) * np.diff(s))
    assert area == pytest.approx((slip_b + slip_e) * ea, rel=2e-3)
    if slip_b < 1e-3 and slip_e < 1e-3:                                             # apart: each its own curve
        k = int(np.argmin(f))
        left = midas_case._drawn_in(s[: k + 1], f[: k + 1], slip_b, ea)
        assert out[: k // 2] == pytest.approx(left[: k // 2], rel=1e-6)


def test_both_ends_draw_in_on_a_straight_middle_or_with_no_friction():
    """A straight middle with no wobble, where the two draw-ins' lines are
    level with each other: the symmetric tendon is each half's own curve;
    with no friction at all the whole tendon drops evenly by both slips
    (review, v1.16: a tie given to the far end took half a slip)."""
    s = np.linspace(0.0, 12.0, 1201)
    P0, mu, ea, slip = 1.0e7, 0.2, 1.95e11 * 0.002, 6.0e-3
    th_a = np.clip(s, 0.0, 2.0) * 0.1 + np.clip(s - 10.0, 0.0, 2.0) * 0.1
    th_b = th_a[-1] - th_a
    f = np.maximum(P0 * np.exp(-mu * th_a), P0 * np.exp(-mu * th_b))
    out = midas_case._drawn_in_both(s, f, slip, slip, ea)
    k = 600
    half = midas_case._drawn_in(s[: k + 1], f[: k + 1], slip, ea)
    assert out[: k + 1] == pytest.approx(half, rel=1e-5)
    flat = np.full_like(s, P0)
    out = midas_case._drawn_in_both(s, flat, 2.0e-3, 6.0e-3, ea)
    assert out == pytest.approx(P0 - 8.0e-3 * ea / 12.0, rel=1e-6)


def test_a_rounded_corner_is_inside_it():
    r = midas_case._rounded([(0, 0, 0), (5, -1, 10), (10, 0, 0)])
    sag = 10.0 * (1.0 / math.cos(math.atan(0.2)) - 1.0)                              # R (sec(a/2) - 1)
    assert min(v for _x, v in r) == pytest.approx(-1.0 + sag, rel=1e-6)
    assert all(r[i][0] < r[i + 1][0] for i in range(len(r) - 1))


def test_a_tendon_is_stressed_when_its_group_is_on_and_its_members_are_built(tmp_path):
    nodes = {i + 1: (float(i), 0.0, 0.0) for i in range(9)}
    elems = {i + 1: (i + 1, i + 2) for i in range(8)}
    text = _staged_girder(nodes, elems, {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}, [("1", "111111", "P")],
                          [(["a"], ["P"], []), (["b"], [], [])])
    text = text.replace(MAT, MAT + STRAND).replace("*STLDCASE\n   自重 , CS,\n", "*STLDCASE\n   自重 , CS,\n   PS , PS,\n")
    text = text.replace("*LOAD-GROUP\n   自重\n", "*LOAD-GROUP\n   自重\n   PSG\n").replace("   ALOAD=自重, FIRST\n", "   ALOAD=自重, FIRST, PSG, FIRST\n")
    text = text.replace("*STAGE\n", _tendon() + "*STAGE\n")
    _m, report, case, _row = _model(text, tmp_path)
    assert report["tendons"] == 1
    mp = json.loads((tmp_path / "g.map.json").read_text(encoding="utf-8"))["tendons"]["P1"]
    assert mp["stage"] == 1                                                           # with b, not with its load group
    bars = {a["region"] for a in case["assign"] if case["sections"][a["section"]]["type"] == "bar"}
    assert len(bars) == 8 and bars <= set(case["steps"][1]["activate"])
    _run(case)
    # a tendon no prestress stresses, or of a form not read, is counted
    _m, report, _case, _row = _model(_girder_with(_tendon().replace(" P1, FORCE", " P9, FORCE")), tmp_path, name="none")
    assert report["tendons"] == 0 and report["left_out"]["tendon that no prestress load stresses"] == 1


def test_a_tendon_relaxes_as_its_code_says_over_the_stages_time(tmp_path):
    """bRELAX with the code's method: each bar of the tendon gets the curve
    of its own stress after anchoring (vfife.codes.jtg3362), and the stages'
    10 days run after each as steps of time -- the tendon, stressed in the
    second stage at 0.72 fpk, has lost 0.61 of psi zeta (0.52 x 0.72 - 0.26)
    by the end of it."""
    from vfife.api import run_case
    from vfife.codes import jtg3362
    nodes = {i + 1: (float(i), 0.0, 0.0) for i in range(9)}
    elems = {i + 1: (i + 1, i + 2) for i in range(8)}
    text = _staged_girder(nodes, elems, {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}, [("1", "111111", "P")],
                          [(["a"], ["P"], []), (["b"], [], [])])
    text = text.replace(MAT, MAT + STRAND).replace("*STLDCASE\n   自重 , CS,\n", "*STLDCASE\n   自重 , CS,\n   PS , PS,\n")
    text = text.replace("*LOAD-GROUP\n   自重\n", "*LOAD-GROUP\n   自重\n   PSG\n").replace("   ALOAD=自重, FIRST\n", "   ALOAD=自重, FIRST, PSG, FIRST\n")
    sigma = 0.72 * 1.86e9
    text = text.replace("*STAGE\n", _tendon(how="STRESS", force=sigma) + "*STAGE\n")
    _m, report, case, _row = _model(text, tmp_path)
    assert any("duration runs after it" in n for n in report["notes"])
    names = [st["name"] for st in case["steps"]]
    assert names == ["S0", "S0 (10 d)", "S1", "S1 (10 d)"] and case["steps"][3]["to_age"] == pytest.approx(48.0)
    mp = json.loads((tmp_path / "g.map.json").read_text(encoding="utf-8"))
    assert mp["stage_steps"] == [1, 3]
    final = jtg3362.relaxation_final(sigma, 1.86e9, 0.3)
    assert final == pytest.approx(0.3 * (0.52 * 0.72 - 0.26))
    (spec,) = {json.dumps(m["relaxation"]) for m in case["materials"].values() if m.get("relaxation")}
    assert json.loads(spec)["curve"][-1][1] == pytest.approx(final, abs=1e-4)

    def bar_forces(c):
        got = {}
        out = run_case(c, on_ready=lambda s, _p: got.__setitem__("s", s))
        assert out.status == "CONVERGED", out.message
        s = got["s"]
        return np.asarray(s.end_actions)[np.asarray(s.member_kind) == "truss2", 6]
    N = bar_forces(case)
    # the same tendon not relaxing: no steps of time, and the forces the jack left
    _m, report0, case0, _row = _model(text.replace(", 0, YES, 0, 0, 0\n", ", 0, NO, 0, 0, 0\n"), tmp_path, name="g0")
    assert len(case0["steps"]) == 2 and not any(m.get("relaxation") for m in case0["materials"].values())
    N0 = bar_forces(case0)
    assert N0 == pytest.approx(np.full(len(N0), sigma * 0.002), rel=3e-3)
    # each bar loses 0.61 of the final loss, less the little the girder and
    # (at the anchors) the arms give back
    assert N / N0 == pytest.approx(np.full(len(N), 1.0 - 0.61 * final), rel=1e-3)
    assert (N / N0)[3] == pytest.approx(1.0 - 0.61 * final, rel=1e-4)


def test_a_tendon_stressed_after_its_girder_has_deflected_keeps_its_eccentricity(tmp_path):
    """Found in review: its nodes were born at their design places under a
    cantilever that had already gone down 0.1 m, and the tendon's effect came
    out 26% too large. Its nodes and arms are now built with the girder."""
    nodes = {i + 1: (float(i), 0.0, 0.0) for i in range(9)}
    elems = {i + 1: (i + 1, i + 2) for i in range(8)}

    def tip(with_tendon, name):
        text = _staged_girder(nodes, elems, {"a": list(range(1, 9))}, [("1", "111111", "P")], [(["a"], ["P"], []), ([], [], [])])
        text = text.replace(MAT, MAT + STRAND).replace("*STLDCASE\n   自重 , CS,\n", "*STLDCASE\n   自重 , CS,\n   PS , PS,\n")
        text = text.replace("*LOAD-GROUP\n   自重\n", "*LOAD-GROUP\n   自重\n   PSG\n")
        text = text.replace("   NAME=S1, 10, YES, NO, NO, 5\n", "   NAME=S1, 10, YES, NO, NO, 5\n   ALOAD=PSG, FIRST\n")
        if with_tendon:
            text = text.replace("*STAGE\n", _tendon(force=2.0e7) + "*STAGE\n")
        _m, report, case, row = _model(text, tmp_path, name=name)
        if with_tendon:
            assert json.loads((tmp_path / f"{name}.map.json").read_text(encoding="utf-8"))["tendons"]["P1"]["stage"] == 1
        q, _f = _run(case)
        return q[row[9], 2]
    # under the centroid of a cantilever it shortens the bottom: the tip goes down by P e L^2 / 2EI
    lift = tip(True, "with") - tip(False, "without")
    assert lift == pytest.approx(-2.0e7 * 0.3 * 8.0 ** 2 / (2 * E * BOX[4]), rel=0.02)


def test_a_typical_tendon_is_count_tendons_of_its_force_or_stress(tmp_path):
    for how, force in (("FORCE", 5.0e6), ("STRESS", 5.0e6 / 0.002)):
        _m, report, _case, _row = _model(_girder_with(_tendon(count=2, force=force, how=how)), tmp_path, name=how)
        t = json.loads((tmp_path / f"{how}.map.json").read_text(encoding="utf-8"))["tendons"]["P1"]
        assert t["count"] == 2 and t["area"] == pytest.approx(0.004) and t["jacking"][0] == pytest.approx(1.0e7)


def test_a_curved_tendon_loses_to_friction_by_its_turning(tmp_path):
    """Drawn with a corner rounded to its radius in the file: the friction
    takes P (1 - exp(-mu theta)) by the far end, theta the corner's turn."""
    zs = ("        Z=0, -0.1, NO, 0, 0, NONE, , , , \n        Z=4, -0.5, NO, 0, 6, NONE, , , , \n"
          "        Z=8, -0.1, NO, 0, 0, NONE, , , , \n")
    _m, report, case, _row = _model(_girder_with(_tendon(mu=0.2, zs=zs)), tmp_path)
    seg = json.loads((tmp_path / "g.map.json").read_text(encoding="utf-8"))["tendons"]["P1"]["segments"]
    theta = 2 * math.atan(0.4 / 4.0)
    assert seg[0][2] == pytest.approx(1.0e7, rel=1e-3) and seg[-1][2] == pytest.approx(1.0e7 * math.exp(-0.2 * theta), rel=2e-3)
    assert seg[3][2] > seg[4][2]                                                      # the corner is between them
    _run(case)


def test_a_stages_twist_hold_holds_through_its_time_too(tmp_path):
    """A girder on pins in one line, its tendon relaxing: the stage's hold on
    its twist is the time step's too, and a torque does not spin it there
    (review)."""
    from vfife.api import run_case
    nodes = {i + 1: (float(i), 0.0, 0.0) for i in range(9)}
    elems = {i + 1: (i + 1, i + 2) for i in range(8)}
    text = _staged_girder(nodes, elems, {"a": [1, 2, 3, 4, 5, 6, 7, 8]}, [("1", "111000", "P"), ("9", "011000", "P")],
                          [(["a"], ["P"], []), ([], [], [])])
    text = text.replace(MAT, MAT + STRAND).replace("*STLDCASE\n   自重 , CS,\n", "*STLDCASE\n   自重 , CS,\n   PS , PS,\n")
    text = text.replace("*LOAD-GROUP\n   自重\n", "*LOAD-GROUP\n   自重\n   PSG\n").replace("   ALOAD=自重, FIRST\n", "   ALOAD=自重, FIRST, PSG, FIRST\n")
    text = text.replace("*STAGE\n", _tendon(how="STRESS", force=0.72 * 1.86e9) + "*STAGE\n")
    _m, _report, case, row = _model(text, tmp_path)
    for k in range(0, len(case["steps"]), 2):
        assert case["steps"][k + 1]["to_age"] and case["steps"][k + 1]["constraints"] == case["steps"][k]["constraints"]
    node = [r for r in case["constraints"] if r["dofs"][:3] == [0, 1, 1]][0]["region"]
    case["loads"].append({"region": node, "moment": [1.0e3, 0, 0]})
    assert run_case(case).status == "CONVERGED"


def test_the_short_form_is_the_long_form(tmp_path):
    """The converter writes the case's own `tendons:`; read,
    they expand to what `expanded=True` writes out, and the two run to the
    same state, bit for bit -- staged, with the tendon in the second stage."""
    nodes = {i + 1: (float(i), 0.0, 0.0) for i in range(9)}
    elems = {i + 1: (i + 1, i + 2) for i in range(8)}
    text = _staged_girder(nodes, elems, {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}, [("1", "111111", "P")],
                          [(["a"], ["P"], []), (["b"], [], [])])
    text = text.replace(MAT, MAT + STRAND).replace("*STLDCASE\n   自重 , CS,\n", "*STLDCASE\n   自重 , CS,\n   PS , PS,\n")
    text = text.replace("*LOAD-GROUP\n   自重\n", "*LOAD-GROUP\n   自重\n   PSG\n").replace("   ALOAD=自重, FIRST\n", "   ALOAD=自重, FIRST, PSG, FIRST\n")
    text = text.replace("*STAGE\n", _tendon(k=0.0015, slip=0.004, jack="BOTH", z=-0.25) + "*STAGE\n")
    _m, rep_s, short, _row = _model(text, tmp_path, name="short", expanded=False)
    _m, rep_l, long_, _row = _model(text, tmp_path, name="long", expanded=True)
    assert "tendons" in short and "tendons" not in long_
    assert len(short["tendons"]) == 1 and rep_s["tendon_bars"] == rep_l["tendon_bars"] == 8
    qs, _f = _run(short)
    ql, _f = _run(long_)
    assert np.array_equal(qs, ql)


def test_two_tendons_short_say_what_they_share_once(tmp_path):
    """Two tendons: the short form writes what both have once
    (`tendon_defaults`) and the profiles in a CSV file, and
    still runs to the long form's state, bit for bit."""
    nodes = {i + 1: (float(i), 0.0, 0.0) for i in range(9)}
    elems = {i + 1: (i + 1, i + 2) for i in range(8)}
    text = _staged_girder(nodes, elems, {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}, [("1", "111111", "P")],
                          [(["a"], ["P"], []), (["b"], [], [])])
    text = text.replace(MAT, MAT + STRAND).replace("*STLDCASE\n   自重 , CS,\n", "*STLDCASE\n   自重 , CS,\n   PS , PS,\n")
    text = text.replace("*LOAD-GROUP\n   自重\n", "*LOAD-GROUP\n   自重\n   PSG\n").replace("   ALOAD=自重, FIRST\n", "   ALOAD=自重, FIRST, PSG, FIRST\n")
    t1 = _tendon(k=0.0015, slip=0.004, jack="BOTH", z=-0.25)
    t2 = _tendon(k=0.0015, slip=0.004, jack="BOTH", z=-0.35)
    a, b = t1.index("*TDN-PROFILE\n") + len("*TDN-PROFILE\n"), t1.index("*USE-STLD, PS")
    second = t2[a:b].replace("NAME=P1", "NAME=P2")
    jack = t1[t1.index(" P1, "):]
    both = t1[:b] + second + t1[b:] + jack.replace(" P1, ", " P2, ")
    text = text.replace("*STAGE\n", both + "*STAGE\n")
    _m, rep_s, short, _row = _model(text, tmp_path, name="short", expanded=False)
    _m, rep_l, long_, _row = _model(text, tmp_path, name="long", expanded=True)
    assert rep_s["tendons"] == rep_l["tendons"] == 2
    shared = short["tendon_defaults"]
    assert shared["profile_z"] == {"table": "short_tendon_z.csv"} and (tmp_path / "short_tendon_z.csv").is_file()
    assert {"section", "count", "friction", "draw_in"} <= set(shared) and shared["stress"]["ends"] == "both"
    assert all("profile_z" not in t and "friction" not in t for t in short["tendons"])
    for key in ("profile_y", "profile_z"):              # (run from a dict: the tables' paths from here)
        if key in shared:
            shared[key]["table"] = str(tmp_path / shared[key]["table"])
    qs, _f = _run(short)
    ql, _f = _run(long_)
    assert np.array_equal(qs, ql)


def test_a_girder_set_off_its_nodes_is_written_with_end_offsets(tmp_path):
    """(1.21) No centroid nodes and no arms: each member's end offsets, the
    short and the long form alike."""
    text = _box_cantilever("CT")
    _m, rep_s, short, _row = _model(text, tmp_path, name="short", expanded=False)
    _m, rep_l, long_, _row = _model(text, tmp_path, name="long", expanded=True)
    assert rep_s["rigid_arms"] == rep_l["rigid_arms"] == 0 and "offset_arms" not in short
    assert not any(k.startswith("offset_arm") for k in short["sections"])
    offs = [a["end_offsets"] for a in short["assign"] if "end_offsets" in a]
    assert len(offs) == 1 and offs[0]["per"] == "member" and offs[0]["i"] == offs[0]["j"]
    assert offs[0]["i"][:2] == [0.0, 0.0] and offs[0]["i"][2] < 0.0
    qs, _f = _run(short)
    ql, _f = _run(long_)
    assert np.array_equal(qs, ql)


def test_beam_end_offsets_are_rigid_zones(tmp_path):
    """Midas' *OFFSET (1.21): a rigid zone at the support end of a 4 m
    cantilever is a cantilever of 3 m; one at the tip carries the load over
    its lever; TYPE=GLOBAL gives the zone as a vector from the node."""
    i, _j, _g = _pipe()
    P = 1.0e5
    load = "   2, 0, 0, -1.0e5, 0, 0, 0, ,\n"
    model, report, case, row = _model(_arm(links="*OFFSET\n   1, ELEMENT, 1, 1, 0, 0, \n", loads=load), tmp_path)
    assert model.offsets == {1: ("ELEMENT", (1.0, 1.0, 0.0, 0.0), "")}
    assert report["members_with_end_zones"] == 1 and report["end_zone_weight_N"] > 0.0
    assert report["left_out"] == {}
    (beam,) = [a for a in case["assign"] if "end_offsets" in a]
    assert beam["end_offsets"] == {"per": "member", "i": [1.0, 0.0, 0.0], "j": [0.0, 0.0, 0.0], "axial": "full"}
    q, _f = _run(case)
    assert q[row[2], 2] == pytest.approx(-P * 3.0 ** 3 / (3 * E * i), rel=1e-3)
    _m, _r, case, row = _model(_arm(links="*OFFSET\n   1, GLOBAL, 1, 0, 0, 0, 0, 0, \n", loads=load), tmp_path, name="glob")
    q2, _f = _run(case)
    assert q2[row[2], 2] == pytest.approx(q[row[2], 2], rel=1e-9)
    _m, _r, case, row = _model(_arm(links="*OFFSET\n   1, ELEMENT, 0, 0, 1, 1, \n", loads=load), tmp_path, name="tip")
    q3, _f = _run(case)
    Lc, a = 3.0, 1.0                    # the elastic end takes P and P a; the node is a further a on, turned
    assert q3[row[2], 2] == pytest.approx(-P / (E * i) * (Lc ** 3 / 3 + a * Lc ** 2 + a * a * Lc), rel=1e-3)
    # two different zones for the two bending planes: the one for My, and said
    _m, report, case, _row = _model(_arm(links="*OFFSET\n   1, ELEMENT, 0.5, 1, 0, 0, \n", loads=load), tmp_path, name="two")
    assert report["left_out"] == {"beam end offset with different zones for the two bending planes "
                                  "(the one for My is taken)": 1}
    (beam,) = [a for a in case["assign"] if "end_offsets" in a]
    assert beam["end_offsets"]["i"] == [1.0, 0.0, 0.0]
    # longer than the member: left out
    _m, report, case, _row = _model(_arm(links="*OFFSET\n   1, ELEMENT, 3, 3, 2, 2, \n", loads=load), tmp_path, name="long")
    assert report["left_out"] == {"beam end offset longer than its member (left out)": 1}
    assert not any("end_offsets" in a for a in case["assign"])


def test_a_section_set_off_and_an_end_zone_together(tmp_path):
    """A girder drawn by its top (CT) with a rigid zone at its support: the
    two offsets add, the zone along the member and the depth across it."""
    text = _top_aligned(load="   2, 0, 0, -1.0e5, 0, 0, 0, ,\n").replace("*STLDCASE", "*OFFSET\n   1, ELEMENT, 1, 1, 0, 0, \n*STLDCASE")
    _m, report, case, row = _model(text, tmp_path)
    offs = sorted((tuple(a["end_offsets"]["i"]), tuple(a["end_offsets"]["j"])) for a in case["assign"] if "end_offsets" in a)
    assert offs == [((0.0, 0.0, -0.5), (0.0, 0.0, -0.5)), ((1.0, 0.0, -0.5), (0.0, 0.0, -0.5))]
    q, _f = _run(case)
    I_ = 0.5 / 12
    assert q[row[3], 2] == pytest.approx(-1.0e5 * 7.0 ** 3 / (3 * E * I_), rel=0.01)


def test_a_beam_end_offset_in_a_staged_model_is_its_boundary_groups(tmp_path):
    """*OFFSET is a boundary condition: in a model with stages it is there
    when a stage activates its boundary group, as a support is -- one in no
    group, or in one no stage activates, is left out, and said."""
    nodes = {1: (0, 0, 0), 2: (4, 0, 0), 3: (8, 0, 0)}
    elems = {1: (1, 2), 2: (2, 3)}

    def text(group, on):
        t = _staged_girder(nodes, elems, {"g": [1, 2]}, [("1", "111111", "P")], [(["g"], on, [])])
        t = t.replace("*BNDR-GROUP\n   P\n", "*BNDR-GROUP\n   P\n   Z\n")
        return t.replace("*STAGE\n", f"*OFFSET\n   1, ELEMENT, 1, 1, 0, 0, {group}\n*STAGE\n")
    _m, report, case, _row = _model(text("", ["P"]), tmp_path, name="none")
    assert report["left_out"] == {"beam end offset whose boundary group no stage activates (left out)": 1}
    assert not any("end_offsets" in a for a in case["assign"])
    _m, report, case, _row = _model(text("Z", ["P"]), tmp_path, name="off")
    assert report["left_out"] == {"beam end offset whose boundary group no stage activates (left out)": 1}
    _m, report, case, _row = _model(text("Z", ["P", "Z"]), tmp_path, name="on")
    assert report["left_out"] == {} and report["members_with_end_zones"] == 1
    assert [a["end_offsets"]["i"] for a in case["assign"] if "end_offsets" in a] == [[1.0, 0.0, 0.0]]
    _run(case)


def test_offsets_the_reader_and_converter_do_not_take(tmp_path):
    """A GLOBAL offset that leaves no length is left out and counted; a row
    whose values are not numbers is not read (found in review)."""
    load = "   2, 0, 0, -1.0e5, 0, 0, 0, ,\n"
    _m, report, case, _row = _model(_arm(links="*OFFSET\n   1, GLOBAL, 2, 0, 0, -2.5, 0, 0, \n", loads=load), tmp_path)
    assert report["left_out"] == {"beam end offset longer than its member (left out)": 1}
    assert not any("end_offsets" in a for a in case["assign"])
    model, report, case, _row = _model(_arm(links="*OFFSET\n   1, ELEMENT, 0.4, 0.4, G, \n", loads=load), tmp_path, name="odd")
    assert model.offsets == {} and model.other == {"OFFSET (not read)": 1}
    assert report["left_out"] == {"beam end offset whose row is not in the form this reads": 1}


# ---- 1.21: what the file says that was read and not followed ----------------

def _solver(case):
    from vfife.api import run_case
    got = {}
    out = run_case(case, on_ready=lambda s, _p: got.__setitem__("s", s))
    assert out.status == "CONVERGED", out.message
    return got["s"]


def _cable(row, extra=""):
    """`_arm` and, beside it, a cable from node 3 to node 4 (10 m), both held."""
    return _arm("   3, 0, 5, 0\n   4, 10, 5, 0\n",
                links="*CONSTRAINT\n   3 4, 111111, \n" + extra).replace(
        "*MATERIAL", "   2, TENSTR,    1,     2,     3,     4,     0, " + row + "\n*MATERIAL")


def test_a_cable_s_own_unstressed_length_or_tension(tmp_path):
    """A Midas cable (TENSTR iSUB 3) says its unstressed length (EXVAL2 3) or
    its tension in the drawn geometry (EXVAL2 1) on its own row: tension only,
    and held between two supports it pulls with EA (L - Lu) / Lu, or T."""
    i, _j, _g = _pipe()
    A = math.pi / 4 * (1.3 ** 2 - 1.256 ** 2)
    model, report, case, row = _model(_cable("3, 9.99, 3"), tmp_path)
    assert (model.elements[2].sub, model.elements[2].exval, model.elements[2].exval2) == (3, 9.99, 3)
    (bar,) = [a for a in case["assign"] if case["sections"][a["section"]]["type"] == "bar"]
    assert bar["tension_only"] is True and bar["unstressed_lengths"] == [pytest.approx(9.99)]
    assert not any("TENSTR" in k for k in report["left_out"])
    s = _solver(case)
    assert float(s.last_reaction_force[row[3], 0]) == pytest.approx(-E * A * 0.01 / 9.99, rel=1e-6)
    _m, report, case, row = _model(_cable("3, 1.0e5, 1"), tmp_path, name="t")
    (bar,) = [a for a in case["assign"] if case["sections"][a["section"]]["type"] == "bar"]
    assert bar["prestress"] == pytest.approx(1.0e5 / A) and bar["tension_only"] is True
    s = _solver(case)
    assert float(s.last_reaction_force[row[3], 0]) == pytest.approx(-1.0e5, rel=1e-6)
    # a pretension load as well: the cable's own state is taken, and said
    text = _cable("3, 9.99, 3").replace("*USE-STLD, P\n", "*USE-STLD, P\n*PRETENSION\n   2, 5.0e4, \n")
    _m, report, _case, _row = _model(text, tmp_path, name="both")
    assert report["left_out"] == {"pretension load on a cable with its own length or tension (the cable's own taken)": 1}
    # a hook's gap is not followed (a tension-only bar), and said
    _m, report, case, _row = _model(_cable("2, 0.01, 0"), tmp_path, name="hook")
    assert report["left_out"] == {"hook's gap (TENSTR iSUB 2): a tension-only bar without it": 1}


def test_section_scale_factors(tmp_path):
    """`*SECT-SCALE`: the bending constant times its factor (a cantilever
    deflects twice as much at half Iyy), the weight times its own factor
    whatever the area's."""
    load = "   2, 0, 0, -1.0e5, 0, 0, 0, ,\n"
    _m, _r, case, row = _model(_arm(loads=load), tmp_path)
    q0 = _run(case)[0][row[2], 2]
    scale = "*SECT-SCALE\n    2, 1, 1, 1, 1, 0.5, 1, 1, , 1, NO\n"
    model, report, case, row = _model(_arm(links=scale, loads=load), tmp_path, name="half")
    assert model.section_scales == {2: ((1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0), "", 1, False)}
    assert _run(case)[0][row[2], 2] == pytest.approx(2.0 * q0, rel=1e-6)
    # weight: twice, with the area unchanged
    weigh = lambda t: t.replace("*USE-STLD, P\n*CONLOAD\n", "*USE-STLD, P\n*SELFWEIGHT\n0, 0, -1, \n*CONLOAD\n")
    _m, _r, case, row = _model(weigh(_arm()), tmp_path, name="w1")
    r1 = float(_solver(case).last_reaction_force[row[1], 2])
    scale = "*SECT-SCALE\n    2, 1, 1, 1, 1, 1, 1, 2, , 1, NO\n"
    _m, report, case, row = _model(weigh(_arm(links=scale)), tmp_path, name="w2")
    assert float(_solver(case).last_reaction_force[row[1], 2]) == pytest.approx(2.0 * r1, rel=1e-9)
    assert any(k.endswith("w2") for k in case["materials"])
    # different factors at the J end: the I end's, and said
    scale = "*SECT-SCALE\n    2, 1, 1, 1, 1, 0.5, 1, 1, , 1, YES\n      1, 1, 1, 1, 0.25, 1, 1\n"
    model, report, _case, _row = _model(_arm(links=scale), tmp_path, name="ij")
    assert model.section_scales[2][0][4] == 0.5
    assert report["left_out"] == {"section scale factors different at the J end (the I end's taken)": 1}
    # a later part of a composite section has a row of its own: not read, and said; the first part's kept
    scale = "*SECT-SCALE\n    2, 1, 1, 1, 1, 0.5, 1, 1, , 1, NO\n    2, 1, 1, 1, 1, 0.8, 1, 1, , 2, NO\n"
    model, report, case, row = _model(_arm(links=scale, loads=load), tmp_path, name="parts")
    assert model.section_scales[2][0][4] == 0.5
    assert report["left_out"] == {"section scale factors of a later part of a composite section (left out)": 1}
    assert _run(case)[0][row[2], 2] == pytest.approx(2.0 * q0, rel=1e-6)


def _loaded(n2, rows):
    """A member from node 1 to node 2, both held, with member loads `rows`."""
    return (HEAD + "*NODE\n   1, 0, 0, 0\n   2, " + n2 + "\n*ELEMENT\n   1, BEAM  ,    1,     2,     1,     2,     0,     0\n"
            + MAT + "*SECTION\n" + PIPE + "*STLDCASE\n   P , D,\n*CONSTRAINT\n   1 2, 111111, \n*USE-STLD, P\n*BEAMLOAD\n"
            + rows + "*ENDDATA\n")


def _beamload(direction, w1, w2, proj="NO"):
    return f"   1, BEAM   , UNILOAD, {direction}, {proj} , NO, LY, , , , 0, {w1}, 1, {w2}, 0, 0, 0, 0, , NO, 0, 0, NO, \n"


def test_member_loads_varying_projected_and_in_the_member_s_axes(tmp_path):
    """A member load from w1 to w2 reaches the two ends as (2 w1 + w2) L / 6
    and (w1 + 2 w2) L / 6; a projected one is per unit of the length normal
    to it; one in the member's own axes (LX, LY, LZ) along them."""
    _m, report, case, row = _model(_loaded("3, 0, 4", _beamload("GZ", -10.0, -20.0)), tmp_path)
    s = _solver(case)
    R = s.last_reaction_force
    assert float(R[row[1], 2]) == pytest.approx(5.0 * 40.0 / 6.0) and float(R[row[2], 2]) == pytest.approx(5.0 * 50.0 / 6.0)
    _m, report, case, row = _model(_loaded("3, 0, 4", _beamload("GZ", -10.0, -20.0, "YES")), tmp_path, name="proj")
    R = _solver(case).last_reaction_force
    assert float(R[row[1], 2] + R[row[2], 2]) == pytest.approx(3.0 * 15.0)
    for direction, axis, sign in (("LZ", 2, 1.0), ("LY", 1, 1.0), ("LX", 0, 1.0)):
        _m, report, case, row = _model(_loaded("4, 0, 0", _beamload(direction, -10.0, -10.0)), tmp_path, name=direction)
        assert report["left_out"] == {}
        R = _solver(case).last_reaction_force
        assert float(R[row[1], axis] + R[row[2], axis]) == pytest.approx(sign * 40.0)


def test_a_rigid_link_comes_and_goes_with_its_boundary_group(tmp_path):
    """A *RIGIDLINK of the group 扣地锚 (on in the second stage, off in the
    third) is made in the second step and released in the third, by name;
    one on a slave its support holds in a tied direction is left out."""
    a = MCT.index("*USE-STLD, 自重")
    text = MCT[:a] + "*RIGIDLINK\n   1, 111000, 3, 扣地锚\n   2, 111000, 4, 扣地锚\n" + MCT[a:]
    _m, report, case, _row = _model(text, tmp_path)
    assert report["rigid_link_slaves"] == 1
    assert report["left_out"]["*RIGIDLINK slave held by a support in a tied direction (held or tied, not both)"] == 1
    assert "rigid_links" not in case
    (link,) = case["steps"][1]["rigid_links"]
    assert link["dofs"] == [1, 1, 1, 0, 0, 0]
    assert link["name"] in case["steps"][2]["release"]
