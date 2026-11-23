"""The `offset_arms:` rule: each arm's section from what
the members set off at its centroid carry, and what cannot be one, refused."""
import pytest

from vfife.io import gmsh
from vfife.prep import offsets


def _mesh(tmp_path):
    # a girder of two members on its centroid line (nodes 4-5-6), drawn 0.5 above (1-2-3), and a pier to node 5
    nodes = [(0, 0, 0.5), (4, 0, 0.5), (8, 0, 0.5), (0, 0, 0), (4, 0, 0), (8, 0, 0), (4, 0, -5)]
    lines = [("girder", 4, 5), ("girder", 5, 6), ("a1", 1, 4), ("a2", 2, 5), ("a3", 3, 6), ("pier", 7, 5)]
    names = ["girder", "a1", "a2", "a3", "pier"]
    msh = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names))]
    msh += [f'1 {i + 1} "{n}"' for i, n in enumerate(names)]
    msh += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    msh += [f"{i + 1} {x!r} {y!r} {z!r}" for i, (x, y, z) in enumerate(nodes)]
    msh += ["$EndNodes", "$Elements", str(len(lines))]
    msh += [f"{i + 1} 1 2 {names.index(g) + 1} {names.index(g) + 1} {a} {b}" for i, (g, a, b) in enumerate(lines)]
    msh += ["$EndElements", ""]
    p = tmp_path / "m.msh"
    p.write_text("\n".join(msh), encoding="utf-8")
    return gmsh.read(str(p), verify=False)[0]


def _case(**over):
    d = {"materials": {"c": {"type": "linear_elastic", "E": 3.0e10, "nu": 0.2},
                       "rigid": {"type": "linear_elastic", "E": 3.0e12, "nu": 0.3}},
         "sections": {"g": {"type": "beam", "material": "c", "shape": "general", "area": 0.5, "iy": 0.04, "iz": 0.1, "j": 0.05},
                      "p": {"type": "beam", "material": "c", "shape": "general", "area": 2.0, "iy": 1.0, "iz": 1.0, "j": 1.0}},
         "assign": [{"region": "girder", "section": "g", "orientation": [0, 0, 1]},
                    {"region": "pier", "section": "p", "orientation": [1, 0, 0]}],
         "offset_arms": {"material": "rigid", "factor": 30.0, "limit": 1.0e13, "regions": ["a1", "a2", "a3"]}}
    d.update(over)
    return d


def test_an_arm_carries_what_the_members_set_off_carry_not_the_pier(tmp_path):
    mesh = _mesh(tmp_path)
    got = {r: (s, o) for r, s, o in offsets.arm_sections(_case(), mesh)}
    E, A, I, L, r = 3.0e10, 0.5, 0.04, 4.0, 0.5
    along, turning = max(E * A / L, 12 * E * I / L ** 3), 4 * E * I / L
    i_arm = 30.0 * max(turning * r, along * r ** 3 / 12.0) / 3.0e12
    for region in ("a1", "a2", "a3"):          # the middle one has the two girder members (the same) and the pier (not set off)
        sec, orient = got[region]
        assert sec["area"] == pytest.approx(30.0 * along * r / 3.0e12, rel=1e-12)
        assert sec["iy"] == pytest.approx(i_arm, rel=1e-12) and sec["j"] == pytest.approx(2 * i_arm, rel=1e-12)
        assert orient == [1.0, 0.0, 0.0]        # a vertical arm
    d = offsets.expand(_case(), mesh)
    assert "offset_arms" not in d and [a["region"] for a in d["assign"][2:]] == ["a1", "a2", "a3"]
    assert [k for k in d["sections"] if k.startswith("offset_arm_")] == ["offset_arm_1"]   # three alike: one section


@pytest.mark.parametrize("over, says", [
    ({"regions": ["girder"]}, "has 2 members"),
    ({"regions": ["nowhere"]}, "not a group of members"),
    ({"regions": ["a1", "a1"]}, "listed twice"),
    ({"material": "steel"}, "not in `materials`"),
])
def test_what_cannot_be_an_arm_is_said(tmp_path, over, says):
    d = _case()
    d["offset_arms"] = dict(d["offset_arms"], **over)
    with pytest.raises(ValueError, match=says):
        offsets.expand(d, _mesh(tmp_path))


def test_an_arm_with_no_member_set_off_is_refused(tmp_path):
    d = _case(assign=[{"region": "pier", "section": "p", "orientation": [1, 0, 0]}])
    with pytest.raises(ValueError, match="no member set off"):
        offsets.expand(d, _mesh(tmp_path))
    d = _case(sections=dict(_case()["sections"], offset_arm_1={"type": "beam"}))
    with pytest.raises(ValueError, match="already has 'offset_arm_1'"):
        offsets.expand(d, _mesh(tmp_path))


def test_a_beam_whose_stiffness_is_not_written_or_placed_is_said(tmp_path):
    mesh = _mesh(tmp_path)
    d = _case(regions={"deck": {"box": [[0, 0, 0], [8, 0, 0]]}},
              assign=[{"region": "deck", "section": "g", "orientation": [0, 0, 1]}])
    with pytest.raises(ValueError, match="cannot be placed"):
        offsets.expand(d, mesh)
    d = _case(sections=dict(_case()["sections"], g={"type": "beam", "material": "c", "shape": "rectangle",
                                                     "width": 1.0, "height": 0.5}))
    with pytest.raises(ValueError, match="not written out"):
        offsets.expand(d, mesh)
    # a group of the mesh's groups is placed
    d = _case(mesh={"groups": {"deck": ["girder"]}},
              assign=[{"region": "deck", "section": "g", "orientation": [0, 0, 1]},
                      {"region": "pier", "section": "p", "orientation": [1, 0, 0]}])
    assert offsets.arm_sections(d, mesh) == offsets.arm_sections(_case(), mesh)


def test_the_limit_is_in_the_cases_units(tmp_path):
    mesh = _mesh(tmp_path)
    d = _case(units={"length": "mm", "force": "N"})
    d["offset_arms"] = {k: v for k, v in d["offset_arms"].items() if k != "limit"}
    d["materials"]["c"]["E"] = 3.0e20                 # a girder so stiff that the limit (1e10 N/mm) holds it
    got = offsets.arm_sections(d, mesh)[0][1]
    assert got["area"] == pytest.approx(30.0 * 1.0e10 * 0.5 / 3.0e12)


def test_an_arm_the_rule_cannot_give_keeps_its_own_section(tmp_path):
    """The converter's fall-back: an arm left out of the rule keeps its
    section, renamed so that the rule's names stay its own."""
    import importlib
    convert = importlib.import_module("vfife.io.midas_case.convert")
    mesh = _mesh(tmp_path)
    d = _case()
    arms = d.pop("offset_arms")
    made = offsets.arm_sections(dict(d, offset_arms=arms), mesh)
    d["sections"] = dict(d["sections"], **{f"offset_arm_{i + 1}": dict(s) for i, (_r, s, _o) in enumerate(made)})
    d["sections"]["offset_arm_3"]["area"] *= 1.001          # not what the rule gives
    d["assign"] = d["assign"] + [{"region": r, "section": f"offset_arm_{i + 1}", "orientation": o}
                                 for i, (r, _s, o) in enumerate(made)]
    case, ruled, own = convert._rule_offset_arms(d, mesh, 30.0, 1.0e13)
    assert (ruled, own) == (2, 1) and case["offset_arms"]["regions"] == ["a1", "a2"]
    assert "offset_arm_own_1" in case["sections"] and not any(k.startswith("offset_arm_") and k[11:].isdigit()
                                                              for k in case["sections"])
    x = offsets.expand(case, mesh)                          # and it reads
    assert {a["region"]: a["section"] for a in x["assign"]}["a3"] == "offset_arm_own_1"
