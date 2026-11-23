"""A Midas stage model as a vfife case: the small model of
tests/test_midas.py -- a two-member cantilever hung from a stay, erected in
three stages -- translated, and run as it was written.

Also here, because this is what found it: a step activates the members of a
GROUP OF LINES, not every member between two of the group's points."""
import json
import math

import numpy as np
import pytest
import yaml

from vfife.io import midas, midas_case
from test_midas import MCT


@pytest.fixture
def model(tmp_path):
    p = tmp_path / "model.mct"
    p.write_bytes(MCT.encode("gbk"))
    return midas.read(str(p))


def _sec(shape, dims, kind="DBUSER", db="", lines=()):
    return midas.Section(name=shape, kind=kind, shape=shape, dims=tuple(dims), db=db, lines=list(lines))


def test_the_section_properties_are_the_closed_forms():
    mm = 1.0e-3
    p = midas_case.section_properties(_sec("P", (1300.0, 22.0)), mm)
    assert p["area"] == pytest.approx(math.pi / 4 * (1.3 ** 2 - 1.256 ** 2))
    assert p["iy"] == p["iz"] == pytest.approx(math.pi / 64 * (1.3 ** 4 - 1.256 ** 4)) and p["j"] == 2 * p["iy"]
    r = midas_case.section_properties(_sec("SR", (42.2,)), mm)
    assert r["area"] == pytest.approx(math.pi / 4 * 0.0422 ** 2)
    b = midas_case.section_properties(_sec("B", (400.0, 300.0, 12.0, 16.0)), mm)       # H, B, tw, tf
    assert b["area"] == pytest.approx(0.4 * 0.3 - 0.368 * 0.276)
    assert b["iy"] == pytest.approx((0.3 * 0.4 ** 3 - 0.276 * 0.368 ** 3) / 12)        # the depth's
    assert b["iz"] == pytest.approx((0.4 * 0.3 ** 3 - 0.368 * 0.276 ** 3) / 12) and b["iy"] > b["iz"]
    h = midas_case.section_properties(_sec("H", (428.0, 407.0, 20.0, 35.0)), mm)
    assert h["area"] == pytest.approx(2 * 0.407 * 0.035 + 0.358 * 0.020)
    # ...and the same from a database name, said to be so
    d = midas_case.section_properties(_sec("H", (), db="GB-YB05, HW 428x407x20/35"), mm)
    assert d["area"] == pytest.approx(h["area"]) and "database name" in d["note"]
    # two channels: twice one, and stiffer sideways the further apart
    near = midas_case.section_properties(_sec("2C", (250.0, 78.0, 7.0, 12.0, 0.0)), mm)
    far = midas_case.section_properties(_sec("2C", (250.0, 78.0, 7.0, 12.0, 414.0)), mm)
    assert near["area"] == far["area"] == pytest.approx(2 * (0.226 * 0.007 + 2 * 0.078 * 0.012))
    assert near["iy"] == far["iy"] and far["iz"] > 10 * near["iz"]
    # a concrete-filled pipe is its pipe, and says so; what is not computed is refused by name
    c = midas_case.section_properties(_sec("EPC", (), kind="SRC", lines=["1, 2, 3, 711, 20, 0"]), mm)
    assert c["area"] == pytest.approx(math.pi / 4 * (0.711 ** 2 - 0.671 ** 2)) and "without its fill" in c["note"]
    with pytest.raises(ValueError, match="does not compute"):
        midas_case.section_properties(_sec("TUB", (1.0,)), mm)


def test_the_local_axes_are_midas_own():
    y = midas_case._local_y((0, 0, 0), (10, 0, 0), 0.0)            # along X: z up, y = +Y
    assert np.allclose(y, [0, 1, 0])
    assert np.allclose(midas_case._local_y((0, 0, 0), (10, 0, 0), 90.0), [0, 0, 1])
    assert np.allclose(midas_case._local_y((0, 0, 0), (0, 0, 5), 0.0), [0, -1, 0])     # a column: z along X, y = z x x
    y = np.array(midas_case._local_y((0, 0, 0), (3, 0, 4), 0.0))   # inclined in XZ: still +Y
    assert np.allclose(y, [0, 1, 0])


def test_the_model_becomes_a_case_and_runs(model, tmp_path):
    from vfife.api import run_case
    report = midas_case.convert(model, str(tmp_path), name="hung")
    assert (report["members"], report["steps"], report["jacked_cables"]) == (3, 3, 1)
    assert report["left_out"]["PLATE element"] == 1 and report["left_out"]["temperature load"] == 1
    assert any("Strand1860" in n for n in report["notes"])          # a grade with no weight in the file
    case = yaml.safe_load((tmp_path / "hung.yaml").read_text(encoding="utf-8"))
    # SI: 206 kN/mm2 is 2.06e11 Pa; 7.904e-8 kN/mm3 over g
    steel = case["materials"]["m1"]
    assert steel["E"] == pytest.approx(2.06e11) and steel["rho"] == pytest.approx(7.904e4 / 9.80665)
    s1, s2, s3 = case["steps"]
    assert [s["name"] for s in case["steps"]] == ["安装S1", "安装S2", "松索"]
    assert "activate" not in s1 and len(s2["activate"]) == 2 and s3["deactivate"] and s3["release"]
    # the stay: jacked to its 2104.2 kN in the step that builds it
    stay = next(a for a in case["assign"] if "jack" in a)
    assert stay["jack"] == pytest.approx(2104.2e3) and "prestress" not in stay
    # the platform: 5 kN a node and 1.1 kN/m over two 10 m members, on at S2 and off again at the last
    on = {tuple(l["vector"]) for l in s2["loads"]}
    off = {tuple(l["vector"]) for l in s3["loads"]}
    assert on == {(0.0, 0.0, -16000.0), (0.0, 0.0, -10500.0), (0.0, 0.0, -5500.0)}
    assert off == {(0.0, 0.0, 16000.0), (0.0, 0.0, 10500.0), (0.0, 0.0, 5500.0)}
    assert all(l["keep"] for l in s2["loads"] + s3["loads"])
    mp = json.loads((tmp_path / "hung.map.json").read_text(encoding="utf-8"))
    assert set(mp["elements"]) == {"1", "2", "10"} and mp["stages"] == ["安装S1", "安装S2", "松索"]
    case["output"]["path"] = str(tmp_path / "hung.h5")
    case["mesh"]["path"] = str(tmp_path / "hung.msh")
    case["output"]["monitors"] = {"tip": {"kind": "displacement", "region": "N5", "dof": "z"},
                                  "R": {"kind": "reaction", "region": "N4", "dof": "z"}}
    out = run_case(case)
    assert out.status == "CONVERGED", out.message
    # at the end the stay and the platform are gone: the foot carries the two members' weight
    A1 = math.pi / 4 * (1.3 ** 2 - 1.256 ** 2); A2 = 0.4 * 0.3 - 0.368 * 0.276
    W = 7.904e4 * 10.0 * (A1 + A2)
    assert float(np.asarray(out.monitors["R"])[-1]) == pytest.approx(W, rel=1e-6)


def test_the_first_stages_alone(model, tmp_path):
    report = midas_case.convert(model, str(tmp_path), name="first", stages=1)
    assert report["steps"] == 1 and report["members"] == 1
    assert report["left_out"]["member no stage activates"] == 2


def test_a_stiff_stub_and_a_rigid_material_are_brought_down_and_counted(model, tmp_path):
    import dataclasses
    nodes = dict(model.nodes); nodes[2] = (5.0, 0.0, 0.0)                 # member 1 is now 5 mm long
    mats = dict(model.materials)
    mats[3] = dataclasses.replace(mats[1], E=2.0e7)                       # "rigid": 1e5 times steel, in kN/mm2
    elements = dict(model.elements); elements[2] = dataclasses.replace(elements[2], material=3)
    stub = dataclasses.replace(model, nodes=nodes, materials=mats, elements=elements)
    report = midas_case.convert(stub, str(tmp_path), name="stub")
    assert report["stubs_made_joints"] == 1
    assert any("rigid = 100 x ordinary" in n for n in report["notes"])
    case = yaml.safe_load((tmp_path / "stub.yaml").read_text(encoding="utf-8"))
    for sec in case["sections"].values():
        if sec["type"] == "beam":
            assert case["materials"][sec["material"]]["E"] <= 100 * 2.06e11 * 1.0001
    # the 5 mm stub is a joint: the limit in each direction, and stiff in turning (1.15)
    joint = [sec["stiffness"] for sec in case["sections"].values() if sec["type"] == "connector" and sec["stiffness"][0] == 1.0e13]
    assert len(joint) == 1 and joint[0][:3] == [1.0e13] * 3 and min(joint[0][3:]) > 1.0e10


def test_init_writes_the_case_of_a_midas_model(model, tmp_path, capsys):
    from vfife import cli
    out = tmp_path / "bridge.yaml"
    assert cli.main(["init", str(out), "--mesh", model.path]) == 0
    text = capsys.readouterr().out
    assert "left out" in text and "PLATE element" in text and "members" in text
    assert out.exists() and (tmp_path / "bridge.msh").exists() and (tmp_path / "bridge.map.json").exists()
    assert cli.main(["init", str(out), "--mesh", model.path]) != 0       # it does not write over a case
    capsys.readouterr()


def test_a_step_activates_the_members_of_a_group_of_lines(tmp_path):
    """A triangle: two chords A-B, B-C in one group, the brace A-C in
    another, built a step later. By the points alone the chords' group held
    the brace too (both its nodes are chord nodes) and the second step was
    refused: 'activates member 2, which is already built'."""
    from vfife.api import run_case
    case = {
        "version": 1, "name": "tri", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "lines", "points": {"A": [0, 0, 0], "B": [2, 0, 2], "C": [4, 0, 0]},
                 "lines": {"ab": {"through": ["A", "B"], "size": 9.0}, "bc": {"through": ["B", "C"], "size": 9.0},
                           "ac": {"through": ["A", "C"], "size": 9.0}},
                 "groups": {"chords": ["ab", "bc"], "brace": ["ac"]}},
        "materials": {"steel": {"type": "linear_elastic", "E": 2.06e11, "nu": 0.3, "rho": 7850.0}},
        "sections": {"s": {"type": "beam", "material": "steel", "shape": "general",
                           "area": 0.01, "iy": 1e-4, "iz": 1e-4, "j": 2e-4}},
        "assign": [{"region": "chords", "section": "s", "orientation": [0, 1, 0]},
                   {"region": "brace", "section": "s", "orientation": [0, 1, 0]}],
        "constraints": [{"region": "A", "dofs": [1, 1, 1, 1, 1, 1]}, {"region": "C", "dofs": [0, 1, 1, 1, 0, 1]}],
        "loads": [{"region": "all", "gravity": [0, 0, -9.80665]}],
        "steps": [{"name": "chords", "activate": ["chords"]}, {"name": "brace", "activate": ["brace"]}],
        "solver": {"mode": "static", "backend": "numpy"},
        "output": {"path": str(tmp_path / "tri.h5")},
    }
    out = run_case(case)
    assert out.status == "CONVERGED", out.message


# ---- found in review: the stage bookkeeping ----------------

def _convert(text, tmp_path, name="m"):
    p = tmp_path / f"{name}.mct"
    p.write_bytes(text.encode("gbk"))
    report = midas_case.convert(midas.read(str(p)), str(tmp_path), name=name)
    return report, yaml.safe_load((tmp_path / f"{name}.yaml").read_text(encoding="utf-8"))


def test_a_pretension_and_a_self_weight_follow_their_load_groups(tmp_path):
    # the stay's pretension group is never activated: the stay is not stressed
    text = MCT.replace("   ALOAD=K1, FIRST, \n         平台, FIRST", "   ALOAD=平台, FIRST")
    report, case = _convert(text, tmp_path)
    assert not any("prestress" in a or "jack" in a for a in case["assign"]) and report["jacked_cables"] == 0
    assert report["left_out"]["pretension whose load group no stage activates"] == 1
    # self weight that comes on in the second stage is said to be on throughout
    text = MCT.replace("   ALOAD=自重, FIRST\n", "").replace("   ALOAD=K1, FIRST, ", "   ALOAD=自重, FIRST, K1, FIRST, ")
    report, case = _convert(text, tmp_path, "w")
    assert case["loads"][0]["gravity"] == pytest.approx([0.0, 0.0, -9.80665])
    assert any("self weight that comes on after the first stage" in k for k in report["left_out"])


def test_two_members_on_the_same_nodes_are_not_merged_in_silence(tmp_path):
    text = MCT.replace("     2, BEAM  ,    1,     2,     2,     3,    90,     0\n",
                       "     2, BEAM  ,    1,     2,     2,     3,    90,     0\n"
                       "     3, BEAM  ,    1,     1,     3,     2,     0,     0\n")
    text = text.replace("   节段2      , 2to3, 2 \\\n        20, 0", "   节段2      , 2to3, 2 3 20, 0")
    report, case = _convert(text, tmp_path)
    assert report["members"] == 3
    assert report["left_out"]["member on the same two nodes as another (the first is kept)"] == 1
    mp = json.loads((tmp_path / "m.map.json").read_text(encoding="utf-8"))
    assert "3" not in mp["elements"]


def test_a_release_leaves_what_another_group_still_holds(tmp_path):
    """The foot is fixed by the permanent group and held in z, too, by the
    temporary one the last stage removes. A step's `release` frees the node
    of everything; the permanent fixity is given again in that step."""
    from vfife.api import run_case
    text = MCT.replace("   4, 111000, 扣地锚\n", "   4, 111000, 扣地锚\n   1, 001000, 扣地锚\n")
    report, case = _convert(text, tmp_path)
    last = case["steps"][-1]
    mp = json.loads((tmp_path / "m.map.json").read_text(encoding="utf-8"))
    assert len(last["release"]) == 2
    (again,) = last["constraints"]
    assert again["dofs"] == [1, 1, 1, 1, 1, 1] and again["keep"]
    assert mp["regions"][again["region"]]["kind"].startswith("still held after the release")
    case["output"]["path"] = str(tmp_path / "m.h5"); case["mesh"]["path"] = str(tmp_path / "m.msh")
    out = run_case(case)
    assert out.status == "CONVERGED", out.message              # it ended MAX_STEPS with the foot freed


def test_a_load_waits_for_the_member_it_is_on(tmp_path):
    # the platform's group comes on in the FIRST stage; member 2 and node 3 are built in the second
    text = MCT.replace("   ALOAD=自重, FIRST\n", "   ALOAD=自重, FIRST, 平台, FIRST\n")
    text = text.replace("   ALOAD=K1, FIRST, \n         平台, FIRST", "   ALOAD=K1, FIRST")
    report, case = _convert(text, tmp_path)
    s1, s2, s3 = case["steps"]
    first = sorted(l["vector"][2] for l in s1["loads"])       # member 1 alone: 5.5 kN at each end, 5 kN on node 2
    assert first == pytest.approx([-10500.0, -5500.0])
    second = sorted(l["vector"][2] for l in s2["loads"])      # member 2 and node 3, when they are built
    assert second == pytest.approx([-10500.0, -5500.0])
    assert sum(l["vector"][2] for l in s3["loads"]) == pytest.approx(32000.0)        # and all of it off at the end


def test_an_element_without_its_section_is_said_by_name(model, tmp_path):
    import dataclasses
    elements = dict(model.elements); elements[2] = dataclasses.replace(elements[2], section=9)
    with pytest.raises(ValueError, match="element 2 has section 9"):
        midas_case.convert(dataclasses.replace(model, elements=elements), str(tmp_path))


# ---- elastic links ----------------------------------------

def _linked(first_node, group):
    """The model with its elastic link made a prop: from `first_node` down to
    node 11, moved 0.5 m below it and held by the permanent group."""
    x = {2: 10000, 3: 20000}[first_node]
    text = MCT.replace("    11, 20000, 0, 0\n", f"    11, {x}, 0, -500\n")
    text = text.replace("   1, 111111, 拱脚固结\n", "   1, 111111, 拱脚固结\n   11, 111000, 拱脚固结\n")
    text = text.replace("     1, 3, 11, GEN  ,     0, NO, NO, NO, NO, NO, NO, 0.1, 1e+06, 1e+06, 0.01, 0.01, 0.02, NO, 0.5, 0.5, 扣地锚",
                        f"     1, {first_node}, 11, GEN  ,     0, NO, NO, NO, NO, NO, NO, 0.1, 1e+06, 1e+06, 0.01, 0.01, 0.02, NO, 0.5, 0.5, {group}")
    return text.replace("     2, 3, 11, RIGID,     0, NO, 0.5, 0.5, 扣地锚\n", "")


def test_a_link_is_written_in_global_axes_and_built_with_its_boundary_group(tmp_path):
    from vfife.api import run_case
    report, case = _convert(_linked(2, "扣地锚"), tmp_path)
    assert report["connectors"] == 1 and report["links_made_to_turn_with_their_ends"] == 1
    (k,) = [s for s in case["sections"].values() if s["type"] == "connector"]
    # along -Z: its soft axial spring is the global Z one; 0.1 kN/mm is 1e5 N/m, 1e6 kN/mm 1e12
    assert k["stiffness"][:3] == pytest.approx([1e12, 1e12, 1e5])
    assert k["stiffness"][3:] == [1e9, 1e9, 1e9]              # node 11 is on no beam: it turns with the link
    (a,) = [a for a in case["assign"] if case["sections"][a["section"]]["type"] == "connector"]
    assert a["coordinate_system"] == "global" and "orientation" not in a
    s1, s2, s3 = case["steps"]
    assert a["region"] in s2["activate"] and a["region"] in s3["deactivate"]      # on with 扣地锚, off with it
    case["output"]["path"] = str(tmp_path / "m.h5"); case["mesh"]["path"] = str(tmp_path / "m.msh")
    out = run_case(case)
    assert out.status == "CONVERGED", out.message


def test_a_link_waits_for_the_member_at_its_far_end(tmp_path):
    # its boundary group is on from the first stage; node 3 is built in the second
    report, case = _convert(_linked(3, "拱脚固结"), tmp_path)
    (a,) = [a for a in case["assign"] if case["sections"][a["section"]]["type"] == "connector"]
    s1, s2, s3 = case["steps"]
    assert "activate" not in s1 and a["region"] in s2["activate"]
    assert not any("elastic link" in k for k in report["left_out"])


def test_a_link_to_a_node_nothing_ever_holds_is_left_out(model, tmp_path):
    report = midas_case.convert(model, str(tmp_path), name="dangle")     # node 11 is on no member and no support
    assert report["connectors"] == 0
    assert report["left_out"]["elastic link whose two ends are never both held while its boundary group is on (left out)"] == 1


def test_a_skew_link_gets_the_full_matrix():
    K = midas_case._to_global((1e5, 1e12, 1e12, 1.0, 2.0, 3.0), (0, 0, 0), (1, 1, 0), 0.0)
    K = np.array(K)
    assert K.shape == (6, 6) and np.allclose(K, K.T)
    x = np.array([1, 1, 0]) / np.sqrt(2)
    assert x @ K[:3, :3] @ x == pytest.approx(1e5) and np.trace(K[:3, :3]) == pytest.approx(2e12 + 1e5)


def test_a_bearing_on_a_fixed_support_or_a_rigid_arm_stays_a_hinge(tmp_path):
    """Found in review: every link with an end on no beam had its rotational
    stiffness raised -- a girder's bearing on a fully fixed support node
    became a moment connection to the ground."""
    text = _linked(2, "扣地锚").replace("   11, 111000, 拱脚固结\n", "   11, 111111, 拱脚固结\n")
    report, case = _convert(text, tmp_path)
    (k,) = [s for s in case["sections"].values() if s["type"] == "connector"]
    assert report["links_made_to_turn_with_their_ends"] == 0
    assert k["stiffness"][3:] == pytest.approx([0.02, 0.01, 0.01])        # 0.01, 0.01, 0.02 kN mm/rad are N m/rad; in global axes


def test_a_boundary_group_that_comes_back_is_held_again(tmp_path):
    """Found on the real arch: its feet are fixed, hinged for the erection,
    and fixed again when the hinge is closed. The second activation was
    counted and dropped, the hinge's release left the feet free, and the
    run ended at a limit point in that stage."""
    text = MCT.replace("*STAGE-CTRL", "   NAME=封铰, 0, YES, NO, NO, 5\n   ABNDR=扣地锚, DEFORMED\n*STAGE-CTRL")
    report, case = _convert(text, tmp_path)
    # (the group's elastic link is built once and removed once, and that is said)
    assert [k for k in report["left_out"] if "activated again" in k] == [
        "link of a boundary group that is activated again after its removal (built once, removed once)"]
    s2, s3, s4 = case["steps"][1:]
    (on,) = s2["constraints"]; (back,) = s4["constraints"]
    assert on["region"] == back["region"] and back["dofs"] == [1, 1, 1, 0, 0, 0] and back["keep"]
    assert s3["release"] == [on["region"]]


def test_of_a_cable_over_a_saddle_one_element_is_jacked(tmp_path):
    """The stay as two elements meeting at a saddle node no beam reaches, both
    given the same pretension: the one that reaches the member just built is
    jacked, the other is cut to the length of its force."""
    text = MCT.replace("    11, 20000, 0, 0\n", "    11, 10000, 0, 7500\n")
    text = text.replace("    10, TRUSS ,    2,     3,     4,     3,     0,     0\n",
                        "    10, TRUSS ,    2,     3,     4,    11,     0,     0\n"
                        "    12, TRUSS ,    2,     3,    11,     3,     0,     0\n")
    text = text.replace("   扣索1      , , 10, 0", "   扣索1      , , 10 12, 0")
    text = text.replace(" 10, 2104.2, K1\n", " 10, 2104.2, K1\n 12, 2104.2, K1\n")
    text = text.replace("     1, 3, 11, GEN  ,     0, NO, NO, NO, NO, NO, NO, 0.1, 1e+06, 1e+06, 0.01, 0.01, 0.02, NO, 0.5, 0.5, 扣地锚\n", "")
    text = text.replace("     2, 3, 11, RIGID,     0, NO, 0.5, 0.5, 扣地锚\n", "")
    report, case = _convert(text, tmp_path)
    assert (report["jacked_cables"], report["cables_cut_to_length"]) == (1, 1)
    mp = json.loads((tmp_path / "m.map.json").read_text(encoding="utf-8"))
    by_region = {a["region"]: a for a in case["assign"]}
    assert "jack" in by_region[mp["elements"]["12"][0]]          # to node 3, on the segment built in that stage
    assert "prestress" in by_region[mp["elements"]["10"][0]]     # to the anchor


def test_two_stays_from_one_anchor_are_two_cables_and_a_compression_is_no_jack(tmp_path):
    """Found in review: runs were joined through a supported node -- two stays
    from one anchor became one jack and one cut length -- and a negative
    pretension lost its sign."""
    text = MCT.replace("    10, TRUSS ,    2,     3,     4,     3,     0,     0\n",
                       "    10, TRUSS ,    2,     3,     4,     3,     0,     0\n"
                       "    12, TRUSS ,    2,     3,     4,     2,     0,     0\n")
    text = text.replace("   扣索1      , , 10, 0", "   扣索1      , , 10 12, 0")
    text = text.replace(" 10, 2104.2, K1\n", " 10, 2104.2, K1\n 12, 2104.2, K1\n")
    report, case = _convert(text, tmp_path)
    assert (report["jacked_cables"], report["cables_cut_to_length"]) == (2, 0)
    report, case = _convert(text.replace(" 12, 2104.2, K1\n", " 12, -2104.2, K1\n"), tmp_path, "neg")
    assert (report["jacked_cables"], report["cables_cut_to_length"]) == (1, 1)
    (cut,) = [a for a in case["assign"] if "prestress" in a]
    assert cut["prestress"] < 0.0


def test_a_plates_weight_goes_on_its_nodes(tmp_path):
    """Found on the real arch: the gusset plates at a foot were 4% of the
    first segment's moment about its hinge, and without them the rib rose
    23 mm where Midas has it still."""
    text = MCT.replace("    20, PLATE ,    1,     1,     1,     2,     3,     0,     1,     0",
                       "    20, PLATE ,    1,     1,     1,     2,     4,     0,     1,     0")
    text = text.replace("*STLDCASE", "*THICKNESS    ; Thickness\n     1, VALUE, 1, YES, 20, 0, YES, 1, 8\n*STLDCASE")
    model = midas.read(_write_mct(tmp_path, text))
    assert model.thicknesses == {1: 20.0}
    report = midas_case.convert(model, str(tmp_path), name="pl")
    W = 7.904e4 * (0.5 * 10.0 * 15.0) * 0.020                 # a 10 m by 15 m triangle, 20 mm, 79.04 kN/m3
    assert report["plate_weight_on_nodes_N"] == pytest.approx(W, rel=1e-9)
    assert report["left_out"] == {**report["left_out"], "PLATE element's stiffness (its weight is on its nodes)": 1}
    case = yaml.safe_load((tmp_path / "pl.yaml").read_text(encoding="utf-8"))
    down = [l["vector"][2] for st in case["steps"] for l in st.get("loads", []) if l["vector"][2] < 0] + \
           [l["vector"][2] for l in case["loads"] if "vector" in l]
    assert any(v == pytest.approx(-W / 3, rel=1e-9) for v in down)


def _write_mct(tmp_path, text):
    p = tmp_path / "plate.mct"
    p.write_bytes(text.encode("gbk"))
    return str(p)


def test_a_support_that_comes_later_holds_where_the_stage_says(tmp_path):
    """DEFORMED: where the node stands -- a plain constraint (1.14 holds a
    node that was free where it finds it). ORIGINAL: at its coordinates --
    the constraint brings it there, `value: 0`."""
    report, case = _convert(MCT, tmp_path)
    late = [c for st in case["steps"] for c in st.get("constraints") or []]
    assert late and all("value" not in c and c["keep"] for c in late)
    assert report["supports_brought_to_their_original_position"] == 0
    report, case = _convert(MCT.replace("ABNDR=扣地锚, DEFORMED", "ABNDR=扣地锚, ORIGINAL"), tmp_path, "o")
    late = [c for st in case["steps"] for c in st.get("constraints") or []]
    assert late and all(c["value"] == 0.0 and c["keep"] for c in late)
    assert report["supports_brought_to_their_original_position"] >= 1
    # the first stage's supports are the model's own, wherever the stage says
    assert all("value" not in c for c in case["constraints"])


def test_the_map_gives_a_nodes_row_in_the_results_when_some_nodes_are_not_used(model, tmp_path):
    """Cut at its first stage, the model's later nodes are in the mesh file
    and in no element; the reader leaves them out and the rows shift. The
    map used to give the place in the file."""
    from vfife.api import run_case
    report = midas_case.convert(model, str(tmp_path), name="cut", stages=1)
    assert report["nodes_no_member_reaches"] >= 1
    mp = json.loads((tmp_path / "cut.map.json").read_text(encoding="utf-8"))
    assert len(mp["nodes"]) == report["nodes"] < len(model.nodes)
    case = yaml.safe_load((tmp_path / "cut.yaml").read_text(encoding="utf-8"))
    case["mesh"]["path"] = str(tmp_path / "cut.msh"); case["output"]["path"] = str(tmp_path / "cut.h5")
    got = {}
    out = run_case(case, on_ready=lambda s, _p: got.__setitem__("s", s))
    assert out.status == "CONVERGED", out.message
    X = got["s"].mesh.nodes
    assert len(X) == report["nodes"]
    f, L = model.si()
    for n, r in mp["nodes"].items():
        assert X[r] == pytest.approx([c * L for c in model.nodes[int(n)]], abs=1e-9)
    # the whole model: one node only the left-out plate reaches, and the rows are 0..n-1 of the rest
    report = midas_case.convert(model, str(tmp_path), name="whole")
    mp = json.loads((tmp_path / "whole.map.json").read_text(encoding="utf-8"))
    assert report["nodes"] + report["nodes_no_member_reaches"] == len(model.nodes)
    assert sorted(mp["nodes"].values()) == list(range(report["nodes"]))

