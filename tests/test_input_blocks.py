"""The high-level blocks that need no mesh: `include:`,
`table:` and `groups:`, each a rewriting of the case into what it says --
the same case, read either way, runs to the same state bit for bit."""
import numpy as np
import pytest
import yaml

from vfife.prep import blocks             # a submodule: imported, not reached as prep.blocks
from vfife.core.diagnostics import UserError
from vfife.io import schema

from test_tendon_block import N_EL, P, _case, _final_state, _staged, _staged_msh, _tendon, _write


def _load(p):
    return schema.load(str(p))[0]


def _same(a, b):
    """Two cases read the same: their internal forms, the mesh path aside."""
    da, db = a.as_dict(), b.as_dict()
    da["mesh"].pop("path", None)
    db["mesh"].pop("path", None)
    return da == db


# ---------------------------------------------------------------- include

def test_an_included_library_is_the_same_case(tmp_path):
    d = _case(tmp_path, [_tendon()])
    lib = {"materials": d.pop("materials"), "sections": d.pop("sections")}
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "steel.yaml").write_text(yaml.safe_dump(lib), encoding="utf-8")
    d["include"] = ["lib/steel.yaml"]
    full = _case(tmp_path, [_tendon()])
    assert _same(_load(_write(tmp_path, d, "inc.yaml")), _load(_write(tmp_path, full, "full.yaml")))


def test_a_name_in_both_is_refused_unless_the_case_overrides(tmp_path):
    d = _case(tmp_path, [_tendon()])
    lib = {"materials": {"c50": dict(d["materials"]["c50"], E=1.0)}}
    (tmp_path / "lib.yaml").write_text(yaml.safe_dump(lib), encoding="utf-8")
    d["include"] = ["lib.yaml"]
    with pytest.raises(UserError, match="materials 'c50' is in the case and in"):
        _load(_write(tmp_path, d, "a.yaml"))
    d["include"] = [{"path": "lib.yaml", "override": True}]
    _load(_write(tmp_path, d, "b.yaml"))
    assert blocks.include(d, str(tmp_path))["materials"]["c50"]["E"] == 3.45e10   # the case's c50 wins


def test_an_included_file_includes_and_holds_only_definitions(tmp_path):
    d = _case(tmp_path, [_tendon()])
    mats = d.pop("materials")
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"include": ["b.yaml"], "materials": {"c50": mats["c50"]}}),
                                     encoding="utf-8")
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"materials": {"strand": mats["strand"]}}), encoding="utf-8")
    d["include"] = ["a.yaml"]
    _load(_write(tmp_path, d, "ok.yaml"))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"include": ["a.yaml"]}), encoding="utf-8")
    with pytest.raises(UserError, match="includes itself"):
        _load(_write(tmp_path, d, "loop.yaml"))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"steps": [{"name": "x"}]}), encoding="utf-8")
    with pytest.raises(UserError, match="an included file holds only"):
        _load(_write(tmp_path, d, "steps.yaml"))


# ----------------------------------------------------------------- tables

def test_a_profile_from_a_table_is_the_profile(tmp_path):
    t = _tendon()
    (tmp_path / "drape.csv").write_text("x,z,radius\n" + "".join(f"{x!r},{z!r},{r!r}\n" for x, z, r in t["profile_z"]),
                                        encoding="utf-8")
    d = _case(tmp_path, [_tendon(profile_z={"table": "drape.csv"})])
    assert _same(_load(_write(tmp_path, d, "t.yaml")), _load(_write(tmp_path, _case(tmp_path, [t]), "i.yaml")))
    # one file for several tendons, by a name column
    (tmp_path / "all.csv").write_text("name,x,z,radius\nT9,0,0,0\nT9,30,0,0\n" +
                                      "".join(f"T1,{x!r},{z!r},{r!r}\n" for x, z, r in t["profile_z"]), encoding="utf-8")
    d = _case(tmp_path, [_tendon(profile_z={"table": "all.csv", "name": "T1"})])
    assert _same(_load(_write(tmp_path, d, "n.yaml")), _load(_write(tmp_path, _case(tmp_path, [t]), "i.yaml")))


def test_nodal_loads_from_a_table_at_named_regions_or_places(tmp_path):
    d = _case(tmp_path, [])
    d.pop("tendons")
    (tmp_path / "sdl.csv").write_text("x,y,z,fz\n10,0,0,-1000\n20,0,0,-2000\n", encoding="utf-8")
    d["steps"][1]["loads"] = [{"table": "sdl.csv", "keep": True}]
    low = blocks.tables(d, str(tmp_path))
    assert low["regions"]["sdl#1"] == {"nearest": [10.0, 0.0, 0.0]}
    assert low["steps"][1]["loads"] == [
        {"region": "sdl#1", "per_node": True, "vector": [0.0, 0.0, -1000.0], "keep": True},
        {"region": "sdl#2", "per_node": True, "vector": [0.0, 0.0, -2000.0], "keep": True}]
    # read and run: the same as the two loads written out
    written = _case(tmp_path, [])
    written.pop("tendons")
    written["regions"] = low["regions"]
    written["steps"][1]["loads"] = low["steps"][1]["loads"]
    a, b = _write(tmp_path, d, "tab.yaml"), _write(tmp_path, written, "out.yaml")
    assert _same(_load(a), _load(b))
    assert np.array_equal(_final_state(str(a)), _final_state(str(b)))


def test_a_table_that_cannot_be_read_says_where(tmp_path):
    d = _case(tmp_path, [])
    d.pop("tendons")
    (tmp_path / "bad.csv").write_text("x,y,z,fz\n10,0,0,heavy\n", encoding="utf-8")
    d["loads"] = d["loads"] + [{"table": "bad.csv"}]
    with pytest.raises(UserError, match=r"loads\[1\]: fz: 'heavy' is not a number"):
        _load(_write(tmp_path, d))
    d["loads"][-1] = {"table": "missing.csv"}
    with pytest.raises(UserError, match="no such file"):
        _load(_write(tmp_path, d))


# ----------------------------------------------------------------- groups

def _fixed(d):
    """Two cantilevers built first stand only on clamped ends."""
    d["constraints"] = [{"region": "left", "dofs": [1, 1, 1, 1, 1, 1]}, {"region": "right", "dofs": [1, 1, 1, 1, 1, 1]}]
    return d


def _grouped(tmp_path):
    d = _fixed(_staged(tmp_path, _tendon(stress={"step": "closure", "ends": "both", "force": [P, P]})))
    d["groups"] = {"cantilevers": ["L", "R"], "deck": ["cantilevers", "C"]}
    d["assign"] = [{"region": "deck", "section": "box", "orientation": [0, 0, 1]}]
    d["steps"] = [{"name": "cantilevers", "activate": ["cantilevers"]}, {"name": "closure", "activate": ["C"]}]
    return d


def test_a_group_builds_its_regions_and_takes_their_section(tmp_path):
    a = _load(_write(tmp_path, _grouped(tmp_path), "g.yaml"))
    b = _load(_write(tmp_path, _fixed(_staged(tmp_path, _tendon(stress={"step": "closure", "ends": "both",
                                                                        "force": [P, P]}))), "s.yaml"))
    sa = [st for st in a.steps]
    assert sa[0].activate[:2] == ["L", "R"]
    # the same members, built in the same steps: the same state
    assert np.array_equal(_final_state(str(tmp_path / "g.yaml")), _final_state(str(tmp_path / "s.yaml")))


def test_a_group_is_a_mesh_group_of_its_parts_members_and_nodes(tmp_path):
    """Not a union of nodes: a member joining two of its parts is not in it
    (found in review: a line load on two columns loaded the beam between)."""
    from vfife.api import prepare_case
    d = _fixed(_staged(tmp_path, _tendon()))
    d.pop("tendons")
    d["groups"] = {"ends": ["L", "R"]}
    a = prepare_case(str(_write(tmp_path, d)))
    er = a.mesh.element_regions["ends"]["truss2"]
    assert er.tolist() == list(range(0, 5)) + list(range(10, 15))       # not C's five between
    assert a.mesh.regions["ends"].tolist() == list(range(0, 6)) + list(range(10, 16))
    assert a.case.as_dict()["mesh"]["groups"] == {"ends": ["L", "R"]}
    # a line load on it is on its ten members only: as on the two written apart
    d1 = dict(d, loads=[{"region": "ends", "line_load": [0, 0, -1.0e4]}])
    d2 = _fixed(_staged(tmp_path, _tendon()))
    d2.pop("tendons")
    d2["loads"] = [{"region": "L", "line_load": [0, 0, -1.0e4]}, {"region": "R", "line_load": [0, 0, -1.0e4]}]
    q1 = _final_state(str(_write(tmp_path, d1, "g.yaml")))
    q2 = _final_state(str(_write(tmp_path, d2, "s.yaml")))
    assert np.allclose(q1, q2, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("groups, says", [
    ({"a": ["a"]}, "contains itself"),
    ({"a": ["b"], "b": ["a"]}, "contains itself"),
    ({"a": []}, "a list of region"),
])
def test_what_cannot_be_a_group_is_said(tmp_path, groups, says):
    d = _case(tmp_path, [])
    d.pop("tendons")
    d["groups"] = groups
    with pytest.raises(UserError, match=says):
        _load(_write(tmp_path, d))


@pytest.mark.parametrize("name", ["girder", "left", "all"])
def test_a_group_named_as_a_mesh_group_is_refused(tmp_path, name):
    d = _case(tmp_path, [])
    d.pop("tendons")
    d["groups"] = {name: ["right"]}
    from vfife.api import prepare_case
    with pytest.raises(UserError, match=repr(name)):
        prepare_case(str(_write(tmp_path, d)))


def test_a_group_of_a_case_region_is_refused(tmp_path):
    d = _case(tmp_path, [])
    d.pop("tendons")
    d["regions"] = {"mid": {"box": [10, 20, -1, 1, -1, 1]}}
    d["groups"] = {"g": ["mid", "left"]}
    with pytest.raises(UserError, match="a region of the case"):
        _load(_write(tmp_path, d))


def test_expand_writes_a_grouped_case_without_its_groups(tmp_path):
    p = _write(tmp_path, _grouped(tmp_path), "g.yaml")
    from vfife import cli
    out = tmp_path / "sub" / "flat.yaml"
    out.parent.mkdir()
    assert cli.main(["expand", str(p), "-o", str(out)]) == 0
    flat = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert "groups" not in flat and "tendons" not in flat
    assert np.array_equal(_final_state(str(p)), _final_state(str(out)))


def test_nodal_masses_from_a_table(tmp_path):
    d = _case(tmp_path, [])
    d.pop("tendons")
    (tmp_path / "m.csv").write_text("region,mass\nleft,1000\nright,2000\n", encoding="utf-8")
    d["nodal_masses"] = [{"table": "m.csv", "weight": False}]
    low = blocks.tables(d, str(tmp_path))
    assert low["nodal_masses"] == [{"region": "left", "per_node": True, "mass": 1000.0, "weight": False},
                                   {"region": "right", "per_node": True, "mass": 2000.0, "weight": False}]
    _load(_write(tmp_path, d))


def test_a_table_column_that_is_not_read_is_refused(tmp_path):
    d = _case(tmp_path, [])
    d.pop("tendons")
    (tmp_path / "t.csv").write_text("x,y,z,f_z\n10,0,0,-1000\n", encoding="utf-8")
    d["loads"] = d["loads"] + [{"table": "t.csv"}]
    with pytest.raises(UserError, match="the column\\(s\\) f_z"):
        _load(_write(tmp_path, d))


def test_one_table_in_two_steps(tmp_path):
    d = _case(tmp_path, [])
    d.pop("tendons")
    (tmp_path / "sdl.csv").write_text("x,y,z,fz\n10,0,0,-1000\n", encoding="utf-8")
    d["steps"][0]["loads"] = [{"table": "sdl.csv"}]
    d["steps"][1]["loads"] = [{"table": "sdl.csv"}]
    _load(_write(tmp_path, d))


def test_two_includes_of_one_library_are_one(tmp_path):
    d = _case(tmp_path, [])
    d.pop("tendons")
    mats = d.pop("materials")
    (tmp_path / "common.yaml").write_text(yaml.safe_dump({"materials": mats}), encoding="utf-8")
    for n in ("a", "b"):
        (tmp_path / f"{n}.yaml").write_text(yaml.safe_dump({"include": ["common.yaml"]}), encoding="utf-8")
    d["include"] = ["a.yaml", "b.yaml"]
    _load(_write(tmp_path, d))


# -------------------------------------------------------- tendon defaults

def test_what_every_tendon_has_is_written_once(tmp_path):
    t1, t2 = _tendon(), _tendon(name="T2", profile_z=[[0.0, 0.0, 0.0], [15.0, -0.6, 40.0], [30.0, 0.0, 0.0]],
                                stress={"step": "stress", "ends": "begin", "force": [P, P]})
    rows = "".join(f"{t['name']},{x!r},{z!r},{r!r}\n" for t in (t1, t2) for x, z, r in t["profile_z"])
    (tmp_path / "tz.csv").write_text("name,x,z,radius\n" + rows, encoding="utf-8")
    shared = {k: t1[k] for k in ("nodes", "section", "count", "line", "friction", "draw_in")}
    shared["stress"] = {"step": "stress", "ends": "both", "force": [P, P]}
    shared["profile_z"] = {"table": "tz.csv"}                # each tendon's rows, by its name
    d = _case(tmp_path, [{"name": "T1"}, {"name": "T2", "stress": {"ends": "begin"}}])
    d["tendon_defaults"] = shared
    assert _same(_load(_write(tmp_path, d, "s.yaml")), _load(_write(tmp_path, _case(tmp_path, [t1, t2]), "l.yaml")))
    # a tendon's own null is its own value; a key a tendon does not have is refused
    t3 = dict(t1, relaxation=None)
    d = _case(tmp_path, [{"name": "T1", "relaxation": None}])
    d["tendon_defaults"] = dict(shared, relaxation={"code": "jtg3362", "fpk": 1.86e9, "zeta": 0.3})
    assert _same(_load(_write(tmp_path, d, "n.yaml")), _load(_write(tmp_path, _case(tmp_path, [t3]), "l1.yaml")))
    d["tendon_defaults"] = dict(shared, name="T9")
    with pytest.raises(UserError, match="tendon_defaults"):
        _load(_write(tmp_path, d, "bad.yaml"))
    # in an included file, its table is the included file's
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "tz.csv").write_text("name,x,z,radius\n" + rows, encoding="utf-8")
    (tmp_path / "lib" / "t.yaml").write_text(yaml.safe_dump({"tendon_defaults": shared}), encoding="utf-8")
    (tmp_path / "tz.csv").unlink()
    d = _case(tmp_path, [{"name": "T1"}, {"name": "T2", "stress": {"ends": "begin"}}])
    d["include"] = ["lib/t.yaml"]
    assert _same(_load(_write(tmp_path, d, "i.yaml")), _load(_write(tmp_path, _case(tmp_path, [t1, t2]), "l.yaml")))


def test_a_tendons_own_stress_or_force_replaces_the_shared_one(tmp_path):
    t1 = _tendon(stress={"step": "stress", "ends": "both", "stress": [1.2e9, 1.2e9]})
    rows = "".join(f"{x!r},{z!r},{r!r},#1\n" for x, z, r in t1["profile_z"])
    (tmp_path / "tz.csv").write_text("x,z,radius,name\n" + rows, encoding="utf-8")
    shared = {k: v for k, v in _tendon().items() if k != "name"}
    shared["profile_z"] = {"table": "tz.csv"}                # a name starting with `#`, in the last column
    d = _case(tmp_path, [{"name": "#1", "stress": {"stress": [1.2e9, 1.2e9]}}])
    d["tendon_defaults"] = shared
    assert _same(_load(_write(tmp_path, d, "s.yaml")),
                 _load(_write(tmp_path, _case(tmp_path, [dict(t1, name="#1")]), "l.yaml")))
    # a table of one other name is read whole, as before v1.20
    d = _case(tmp_path, [_tendon(profile_z={"table": "tz.csv"}, stress=t1["stress"])])
    assert _same(_load(_write(tmp_path, d, "o.yaml")), _load(_write(tmp_path, _case(tmp_path, [t1]), "l1.yaml")))
