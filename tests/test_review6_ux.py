"""Two usability features: the case file's JSON Schema for
an editor (`vfife schema`, the shipped docs/vfife.schema.json, the modeline
`vfife init` writes), and each line-member group's local axes in `vfife
info`, with a note when a horizontal member's local z is not vertical.
"""
import glob
import json
import math
import pathlib

import numpy as np
import pytest
import yaml

from vfife import api, reference
from vfife.cli import main
from vfife.core import member_axes
from vfife.io.model import is_input_form
from vfife.io.schema import coerce_numbers

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "benchmarks" / "assets"
MODELINE = "# yaml-language-server: $schema=vfife.schema.json"


# ------------------------------------------------------------------ schema
def test_the_schema_command_writes_a_schema_every_shipped_case_validates_under(tmp_path, capsys,
                                                                                monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["schema"]) == 0
    out = capsys.readouterr().out
    target = tmp_path / "vfife.schema.json"
    assert str(target) in out and MODELINE in out and "Red Hat" in out
    s = json.loads(target.read_text(encoding="utf-8"))
    assert s["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert s["title"] == "vfife case file, format 1"
    assert s == api.case_schema()                   # one schema, two doors
    # -o into a folder, the modeline relative to where it was run
    assert main(["schema", "-o", "sub"]) == 0
    assert "$schema=sub/vfife.schema.json" in capsys.readouterr().out
    assert (tmp_path / "sub" / "vfife.schema.json").read_text(encoding="utf-8") == \
        target.read_text(encoding="utf-8")

    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(s)
    v = jsonschema.Draft202012Validator(s)
    n = 0
    for f in sorted(glob.glob(str(ROOT / "cases" / "*.yaml"))):
        d = yaml.safe_load(open(f, encoding="utf-8"))
        if not (isinstance(d, dict) and is_input_form(d)):
            continue
        errs = [(e.json_path, e.message[:200]) for e in v.iter_errors(coerce_numbers(d))]
        assert not errs, (f, errs[0])
        n += 1
    assert n > 50
    # a misspelt key, a value outside an enum, a material type nobody has
    bad = coerce_numbers(yaml.safe_load((ROOT / "cases" / "case_01_beam_cantilever.yaml")
                                        .read_text(encoding="utf-8")))
    bad["solver"]["mdoe"] = "dynamic"
    assert any("mdoe" in e.message for e in v.iter_errors(bad))
    bad["solver"].pop("mdoe"); bad["solver"]["mode"] = "dynamc"
    assert any("dynamc" in e.message for e in v.iter_errors(bad))
    bad["solver"]["mode"] = "dynamic"
    name = next(iter(bad["materials"]))
    bad["materials"][name]["type"] = "stell02"
    assert any("stell02" in e.message for e in v.iter_errors(bad))


def test_every_field_of_the_schema_carries_its_description():
    s = api.case_schema()
    missing = [f"{k}.{p}" for k, d in s["$defs"].items()
               for p, v in (d.get("properties") or {}).items() if not v.get("description")]
    missing += [p for p, v in s["properties"].items() if not v.get("description")]
    assert not missing, missing[:10]
    d = s["$defs"]
    # the field comments, Chinese line then English (the YAML reference's text)
    orient = d["AssignSpec"]["properties"]["orientation"]["description"]
    assert "局部 y" in orient and "local y axis" in orient
    assert "relaxation" in d["SolverSpec"]["properties"]["mode"]["description"]
    assert set(d["SolverSpec"]["properties"]["mode"]["enum"]) == {"dynamic", "relaxation", "static"}
    assert d["SectionSpec"]["properties"]["height"]["description"]
    # the dict-typed sub-blocks have their fields
    assert "BondSpec" in d and "FailureSpec" in d and "FractureSpec" in d
    # a material's type from the catalogue, with its parameters beside each
    mt = s["properties"]["materials"]["additionalProperties"]["properties"]["type"]
    assert {"steel02", "concrete02", "linear_elastic"} <= set(mt["enum"])
    assert len(mt["enumDescriptions"]) == len(mt["enum"])


def test_reference_check_covers_the_shipped_schema(tmp_path, monkeypatch):
    shipped = ROOT / "docs" / "vfife.schema.json"
    assert shipped.read_text(encoding="utf-8") == reference.schema_text(), \
        "docs/vfife.schema.json is stale: run `python -m vfife reference`"
    # --check fails on a stale copy and plain `reference` rewrites it
    stale = tmp_path / "docs"
    stale.mkdir()
    (stale / "YAML参考.md").write_text(reference.generate(), encoding="utf-8")
    (stale / "vfife.schema.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(reference, "ROOT", tmp_path)
    assert main(["reference", "--check"]) == 1
    assert main(["reference"]) == 0
    assert (stale / "vfife.schema.json").read_text(encoding="utf-8") == reference.schema_text()
    assert main(["reference", "--check"]) == 0


def test_init_writes_the_modeline_and_the_schema_beside_the_case(tmp_path, capsys):
    msh = str(ASSETS / "nafems_le10.msh")
    flags = ["--non-interactive", "--fixed-region", "upper", "--load-region", "midplane"]
    case = tmp_path / "c.yaml"
    assert main(["init", str(case), "--mesh", msh] + flags) == 0
    text = case.read_text(encoding="utf-8")
    assert text.splitlines()[0] == MODELINE
    schema = tmp_path / "vfife.schema.json"
    assert schema.read_text(encoding="utf-8") == reference.schema_text()
    assert "vfife.schema.json" in capsys.readouterr().out
    jsonschema = pytest.importorskip("jsonschema")
    assert not list(jsonschema.Draft202012Validator(json.loads(schema.read_text(encoding="utf-8")))
                    .iter_errors(coerce_numbers(yaml.safe_load(text))))
    # one already there is kept, with a word when it is not this version's
    schema.write_text("{}\n", encoding="utf-8")
    assert main(["init", str(tmp_path / "d.yaml"), "--mesh", msh] + flags) == 0
    assert schema.read_text(encoding="utf-8") == "{}\n"
    assert "`vfife schema`" in capsys.readouterr().out
    # --overwrite still governs the case only
    assert main(["init", str(case), "--mesh", msh] + flags) == 1
    assert main(["init", str(case), "--mesh", msh, "--overwrite"] + flags) == 0


# -------------------------------------------------------------------- axes
def test_vector_text_and_one_axis_over_a_group():
    assert member_axes.vector_text([0, 0, 2]) == "+Z"
    assert member_axes.vector_text([0, -1, 1e-4]) == "-Y"
    assert member_axes.vector_text([1, 0, 1]) == "[0.71, 0, 0.71]"
    same = np.array([[1.0, 0, 0]] * 3)
    assert member_axes._axis_text("x", same) == "local x = +X"
    fan = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a)), 0] for a in (0, 10, 20)])
    assert member_axes._axis_text("x", fan) == "local x within 10 deg of [0.98, 0.17, 0]"
    ring = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a)), 0] for a in (0, 60, 120)])
    assert member_axes._axis_text("x", ring) == "local x varies"
    assert list(member_axes.vertical({"loads": [{"gravity": [0, -9.81, 0]}]})) == [0, 1, 0]
    assert list(member_axes.vertical({"steps": [{"loads": [{"gravity": [0, 0, -9.81]}]}]})) == [0, 0, 1]
    assert list(member_axes.vertical({})) == [0, 0, 1]


_ARC = [(-10 * math.sin(math.radians(p)), -10 + 10 * math.cos(math.radians(p)), 0.0)
        for p in (10, 20, 30, 40)]


def _frame(tmp_path, loads="[{region: tip, vector: [0, 0, -1000]}]", girder=""):
    """A girder along X (a rect beam, 0.3 wide and 0.6 high), a pier along
    Z, a skew brace, a tie (bar) and a curved arc (A/Iy/Iz sections), each
    its own physical group."""
    nodes = [(i, 0.0, 0.0) for i in range(5)] + [(2.0, 0.0, -3.0), (2.0, 0.0, -1.5),
                                                  (5.0, 1.0, 0.0)] + _ARC
    els = ([(1, i, i + 1) for i in range(1, 5)] + [(2, 6, 7), (2, 7, 3), (3, 5, 8), (4, 1, 6)]
           + [(5, a, b) for a, b in ((1, 9), (9, 10), (10, 11), (11, 12))])
    msh = tmp_path / "frame.msh"
    msh.write_text(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n5\n"
        '1 1 "girder"\n1 2 "pier"\n1 3 "brace"\n1 4 "tie"\n1 5 "arc"\n$EndPhysicalNames\n'
        f"$Nodes\n{len(nodes)}\n" + "".join(f"{i + 1} {x} {y} {z}\n" for i, (x, y, z) in enumerate(nodes))
        + f"$EndNodes\n$Elements\n{len(els)}\n"
        + "".join(f"{k + 1} 1 2 {g} {g} {a} {b}\n" for k, (g, a, b) in enumerate(els))
        + "$EndElements\n", encoding="utf-8")
    beam = "{type: beam, material: steel, shape: general, area: 0.01, iy: 1.0e-4, iz: 2.0e-4, j: 1.0e-4}"
    case = tmp_path / "frame.yaml"
    case.write_text(f"""
name: frame
mesh: {{type: gmsh, path: frame.msh}}
materials:
  steel: {{type: linear_elastic, E: 2.0e+11, nu: 0.3, rho: 7850}}
sections:
  b: {beam}
  r: {{type: beam, material: steel, shape: rect, width: 0.3, height: 0.6}}
  t: {{type: bar, material: steel, shape: general, area: 0.001}}
assign:
  - {{region: girder, section: r{girder}}}
  - {{region: pier, section: b, orientation: [1, 0, 0]}}
  - {{region: brace, section: b, orientation: [-1, 1, 0]}}
  - {{region: tie, section: t}}
  - {{region: arc, section: b}}
regions:
  base: {{box: [1.9, 2.1, -0.1, 0.1, -3.1, -2.9]}}
  tip:  {{box: [3.9, 4.1, -0.1, 0.1, -0.1, 0.1]}}
constraints: [{{region: base, dofs: [1, 1, 1, 1, 1, 1]}}]
loads: {loads}
solver: {{backend: numpy, mode: relaxation, n_steps: 10}}
output: {{path: {tmp_path / 'o.h5'}}}
""", encoding="utf-8")
    return str(case)


def _axes(text):
    return [l.strip() for l in text.splitlines() if l.strip().startswith(("axes", "note"))]


def test_info_prints_each_groups_local_axes(tmp_path, capsys):
    assert main(["info", _frame(tmp_path)]) == 0
    lines = _axes(capsys.readouterr().out)
    text = "\n".join(lines)
    # a beam along X with the default orientation (+Z is local y)
    assert "axes          girder (beam2, 4 members): local x = +X, local y = +Z, local z = -Y" in text
    # a pier along Z, local y along X
    assert "axes          pier (beam2, 2 members): local x = +Z, local y = +X, local z = +Y" in text
    # a skew member: rounded vectors, its local z vertical so no note
    assert ("axes          brace (beam2, 1 member): local x = [0.71, 0.71, 0], "
            "local y = [-0.71, 0.71, 0], local z = +Z") in text
    # a bar has a local x only
    assert "axes          tie (truss2, 1 member): local x = [0.55, 0, -0.83]" in text
    # a curved group says how far its members turn
    arc = next(l for l in lines if l.startswith("axes          arc"))
    assert "local x within 15 deg of [-0.94, -0.34, 0]" in arc and "local y = +Z" in arc
    assert "local z within 15 deg of" in arc
    # the note: a rect section whose height lies flat -- not the arc, whose
    # A/Iy/Iz section has no height, though its local z is horizontal too
    notes = [l for l in lines if l.startswith("note")]
    assert notes == ["note          girder: section height 0.6 m lies along local z = -Y "
                     "(horizontal); if the depth should be vertical, set orientation so local "
                     "z is up (e.g. orientation: [0, 1, 0])"]


def test_the_orientation_the_note_suggests_puts_the_height_vertical(tmp_path, capsys):
    assert main(["info", _frame(tmp_path, girder=", orientation: [0, 1, 0]")]) == 0
    lines = _axes(capsys.readouterr().out)
    assert "axes          girder (beam2, 4 members): local x = +X, local y = +Y, local z = +Z" in lines
    assert not [l for l in lines if l.startswith("note")]


def test_the_vertical_is_against_gravity(tmp_path, capsys):
    # gravity along -Y: the rect girder's local z (-Y) is now vertical
    case = _frame(tmp_path, loads="[{region: all, gravity: [0, -9.81, 0]}]")
    assert main(["info", case]) == 0
    lines = _axes(capsys.readouterr().out)
    assert any(l.startswith("axes          girder") for l in lines)
    assert not [l for l in lines if l.startswith("note")]


def test_a_patches_girder_gets_no_note_and_identical_groups_merge(capsys):
    # the box girder is drawn in patches with y up: its author chose the
    # axes, so the default orientation is no news
    assert main(["info", str(ROOT / "cases" / "case_24d_hzb_girder_piers.yaml")]) == 0
    lines = _axes(capsys.readouterr().out)
    assert lines[0] == ("axes          girder (beam2_fiber, 42 members): local x = +X, "
                        "local y = +Z, local z = -Y")
    piers = next(l for l in lines if "pier1_seg1" in l)
    assert "12 groups" in piers and "local x = -Z, local y = +X, local z = -Y" in piers
    assert not [l for l in lines if l.startswith("note")]


def test_a_rect_fibre_section_lying_flat_gets_the_note(tmp_path, capsys):
    # the manual's chapter-3 girder (a rect fibre section, height 1.2) with
    # and without its `orientation: [0, 1, 0]`
    import sys
    sys.path.insert(0, str(ROOT / "tests"))
    from test_review6_docs import _pc_beam
    case = pathlib.Path(_pc_beam(tmp_path))
    text = case.read_text(encoding="utf-8")
    assert "orientation: [0, 1, 0]" in text
    assert main(["info", str(case)]) == 0
    assert not [l for l in _axes(capsys.readouterr().out) if l.startswith("note")]
    case.write_text(text.replace(", orientation: [0, 1, 0]", ""), encoding="utf-8")
    assert main(["info", str(case)]) == 0
    notes = [l for l in _axes(capsys.readouterr().out) if l.startswith("note")]
    assert notes == ["note          beam: section height 1.2 m lies along local z = -Y (horizontal); "
                     "if the depth should be vertical, set orientation so local z is up "
                     "(e.g. orientation: [0, 1, 0])"]


def test_a_shell_line_model_prints_its_line_members_axes():
    from vfife.core.mesh import Mesh
    from vfife.core.shell_line_solver import ShellLineSolver
    from vfife.materials import create
    nodes = np.array([[0.0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]])
    blocks = {"shell4": np.array([[0, 1, 2, 3]], np.int32), "truss2": np.array([[0, 1]], np.int32)}
    mesh = Mesh(nodes, blocks["shell4"], {"all": np.arange(4), "beam": np.array([0, 1])}, blocks)
    p = {"element": {"type": "shell_line", "formulation": "selective", "thickness": 0.1,
                     "shear_correction": 5 / 6, "drilling_factor": 1.0, "hourglass_coeff": 0.1,
                     "warpage_tolerance": 0.05},
         "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 10,
                    "damping": {"type": "kinetic"}, "tolerance": 1e-5, "kinetic_tolerance": 1e-10},
         "constraints": [], "loads": [], "regions": {}, "output": {"monitors": {}},
         "members": [{"region": "beam", "type": "beam2", "area": .01, "iy": 1e-5, "iz": 2e-5,
                      "j": 3e-5, "orientation": [0, 1, 0], "releases": []}]}
    s = ShellLineSolver(p, mesh, create("linear_elastic", E=210e9, nu=0.3, rho=7850))
    assert ("axes          beam (beam2, 1 member): local x = +X, local y = +Y, local z = +Z"
            in s.summary())
    assert "note" not in s.summary()
