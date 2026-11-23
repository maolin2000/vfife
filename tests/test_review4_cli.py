"""The fourth review's findings on the user-facing layer -- CLI, Python API,
init, formfind, view, sweep (section five). One test per
finding, each small enough to run in seconds.
"""
import csv
import glob
import json
import pathlib
import shutil
import sys

import h5py
import numpy as np
import pytest
import yaml

from vfife import api, formfind
from vfife.cli import main
from vfife.core.diagnostics import UserError

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "benchmarks" / "assets"
CANTILEVER = ROOT / "tests" / "data" / "result_format_2" / "cantilever.yaml"
sys.path.insert(0, str(ROOT / "benchmarks"))


def _case01(td, **output):
    """case_01 with its mesh by an absolute path and its result in `td`."""
    doc = yaml.safe_load((ROOT / "cases" / "case_01_beam_cantilever.yaml").read_text(encoding="utf-8"))
    doc["mesh"]["path"] = str(ASSETS / "beam_cantilever_v22.msh")
    doc["output"]["path"] = str(pathlib.Path(td) / "c01.h5")
    doc["output"].update(output)
    p = pathlib.Path(td) / "c01.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return p


def _bridge(td, n=4, **solver):
    """case_23's suspension bridge, small, dead load only, mesh beside it."""
    from generate_suspension_bridge import build
    td = pathlib.Path(td)
    info = build(str(td / "bridge.msh"), n=n)
    doc = yaml.safe_load((ROOT / "cases" / "case_23_suspension_bridge.yaml").read_text(encoding="utf-8"))
    doc["mesh"]["path"] = "bridge.msh"                          # relative, on purpose
    doc["output"]["path"] = str(td / "out.h5")
    doc["loads"][0]["vector"] = [0.0, 0.0, -info["deck_interior_load"] * (n - 1)]
    doc["steps"] = doc["steps"][:1]
    for a in doc["assign"]:
        a.pop("unstressed_lengths", None)
    if solver.get("mode") == "dynamic":
        doc["solver"] = {"backend": "numpy", **solver}
    else:
        doc["solver"].update(solver)
    p = td / "bridge.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return p


# --------------------------------------------------------------- 2: schema
def test_case_schema_is_the_input_form_every_shipped_case_is_valid_under():
    # jsonschema is not a vfife dependency (the Windows check_all had none,
    # v1.0): without it the same model the schema is made from
    # validates the cases, and the schema itself is still checked above
    try:
        import jsonschema
    except ImportError:
        jsonschema = None
    from vfife.io.model import is_input_form
    from vfife.io.schema import coerce_numbers
    s = api.case_schema()
    props = set(s["properties"])
    assert {"materials", "sections", "assign", "mesh", "solver", "output", "loads", "steps"} <= props
    assert not props & {"element", "material", "members", "blocks"}      # the lowered form
    json.dumps(s)                                           # it is plain JSON
    if jsonschema is not None:
        v = jsonschema.Draft202012Validator(s)
        errors = lambda d: [(e.json_path, e.message[:200]) for e in v.iter_errors(d)]
    else:
        model = api._input_case_model()

        def errors(d):
            try:
                model.model_validate(d)
            except Exception as e:                          # pydantic.ValidationError
                return [str(e)[:300]]
            return []
    n = 0
    for f in sorted(glob.glob(str(ROOT / "cases" / "*.yaml"))):
        d = yaml.safe_load(open(f, encoding="utf-8"))
        if not (isinstance(d, dict) and is_input_form(d)):
            continue
        errs = errors(coerce_numbers(d))                     # `2e11` is a string to YAML 1.1
        assert not errs, (f, errs[0])
        n += 1
    assert n > 50
    # the old form every loader refuses is refused by the schema too
    assert errors({"name": "x", "mesh": {"type": "box"}, "solver": {},
                   "sections": {}, "material": {"type": "linear_elastic"}})


# ------------------------------------------------------ 3, 4, 13, 14: init
def test_init_refuses_a_shell_mesh_and_a_line_mesh_saying_what_to_write(tmp_path, capsys):
    for msh, word in (("clamped_square_shell4_v22.msh", "`shell` section"),
                      ("beam_cantilever_v22.msh", "`beam`")):
        rc = main(["init", str(tmp_path / "c.yaml"), "--mesh", str(ASSETS / msh),
                   "--non-interactive", "--fix", "x", "111", "--force", "y", "0", "0", "-1"])
        err = capsys.readouterr().err
        assert rc == 1 and "writes a solid section" in err and word in err
        assert not (tmp_path / "c.yaml").exists()
    # and the case it used to write is refused before the run, not by a crash in it
    from vfife.scaffold import generic_gmsh_case
    case = tmp_path / "shell.yaml"
    case.write_text(generic_gmsh_case("shell", str(ASSETS / "clamped_square_shell4_v22.msh"),
                                      "clamped", "plate", [0.0, 0.0, -1.0e3]), encoding="utf-8")
    assert main(["check", str(case), "-o", str(tmp_path / "r.json")]) == 1
    assert "contains only shell4 elements" in capsys.readouterr().err


def test_init_does_not_overwrite_without_asking_and_never_its_own_mesh(tmp_path, capsys):
    msh = tmp_path / "le10.msh"
    shutil.copy(ASSETS / "nafems_le10.msh", msh)
    before = msh.read_bytes()
    flags = ["--non-interactive", "--fixed-region", "upper", "--load-region", "midplane"]
    assert main(["init", str(msh), "--mesh", str(msh), "--overwrite"] + flags) == 1
    assert "the mesh file itself" in capsys.readouterr().err and msh.read_bytes() == before
    case = tmp_path / "le10.yaml"
    case.write_text("# mine\n", encoding="utf-8")
    assert main(["init", str(case), "--mesh", str(msh)] + flags) == 1
    assert "--overwrite" in capsys.readouterr().err and case.read_text(encoding="utf-8") == "# mine\n"
    assert main(["init", str(case), "--mesh", str(msh), "--overwrite"] + flags) == 0
    assert "sections" in yaml.safe_load(case.read_text(encoding="utf-8"))


def test_mesh_reports_the_line_block_and_init_warns_it_is_not_assigned(tmp_path, capsys):
    msh = str(ASSETS / "coupled_hex_truss_v22.msh")
    assert main(["mesh", msh]) == 0
    out = capsys.readouterr().out
    assert "also in file  truss2 (1)" in out
    assert main(["mesh", msh, "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["node_set_blocks"] == {"truss2": 1}
    assert main(["init", str(tmp_path / "c.yaml"), "--mesh", msh, "--non-interactive",
                 "--fixed-region", "fixed", "--load-region", "loaded"]) == 0
    assert "1 Line2 element(s)" in capsys.readouterr().err


def test_interactive_init_asks_only_what_the_options_left_open(tmp_path, monkeypatch, capsys):
    asked = []

    def answer(prompt=""):
        asked.append(prompt)
        return "fixed" if "support" in prompt else ""

    monkeypatch.setattr("builtins.input", answer)
    msh = str(ASSETS / "coupled_hex_truss_v22.msh")
    out = tmp_path / "c.yaml"
    # --force is the load: no load prompt, no -1000 N on top of it
    assert main(["init", str(out), "--mesh", msh, "--force", "loaded", "0", "0", "-5"]) == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["loads"] == [{"region": "loaded", "vector": [0.0, 0.0, -5.0]}]
    assert len(asked) == 1 and "support" in asked[0]
    # and the whole-volume group `solid` is not offered as a boundary
    assert "solid" not in asked[0] and "fixed" in asked[0]
    # --fix is the support: no support prompt, no fully fixed group added
    asked.clear()
    out2 = tmp_path / "d.yaml"
    assert main(["init", str(out2), "--mesh", msh, "--fix", "fixed", "100",
                 "--force", "loaded", "0", "0", "-5"]) == 0
    assert asked == []
    assert yaml.safe_load(out2.read_text(encoding="utf-8"))["constraints"] == [
        {"region": "fixed", "dofs": [1, 0, 0]}]


# ------------------------------------------------------ 5, 6: formfind
def test_formfind_needs_a_relaxation_and_an_inner_run_that_converged(tmp_path, capsys):
    dyn = _bridge(tmp_path, mode="dynamic", total_time=0.01, damping={"type": "none"})
    with pytest.raises(UserError, match="solver.mode = dynamic"):
        formfind.find(str(dyn), ["main_cable", "hangers"], log=lambda *_: None)
    short = _bridge(tmp_path, max_steps_per_load=30)
    with pytest.raises(UserError, match="did not reach equilibrium") as exc:
        formfind.find(str(short), ["main_cable", "hangers"], log=lambda *_: None)
    assert exc.value.code == "FORMFIND_NOT_RELAXED"
    out = tmp_path / "never.yaml"
    assert main(["formfind", str(short), "--groups", "main_cable,hangers", "-o", str(out)]) == 3
    assert "NOT CONVERGED" in capsys.readouterr().err and not out.exists()


def test_an_unconverged_formfind_says_so_in_the_file_it_writes(tmp_path, capsys):
    case = _bridge(tmp_path)
    out = tmp_path / "last.yaml"
    assert main(["formfind", str(case), "--groups", "main_cable,hangers", "-o", str(out),
                 "--tol", "1e-9", "--max-iter", "1"]) == 3
    head = out.read_text(encoding="utf-8").splitlines()[:5]
    assert head[0].startswith("# NOT CONVERGED") and not any("leave the mesh" in h for h in head)


def test_formfind_written_elsewhere_still_finds_its_mesh(tmp_path, capsys):
    case = _bridge(tmp_path)
    out = tmp_path / "found" / "deep" / "bridge_found.yaml"
    out.parent.mkdir(parents=True)
    assert main(["formfind", str(case), "--groups", "main_cable,hangers", "-o", str(out)]) == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["mesh"]["path"] == "../../bridge.msh"
    assert doc["output"]["path"] == str(tmp_path / "out.h5")          # the working directory's
    assert api.prepare_case(str(out)).preflight["status"] == "PASS"


def test_formfind_refuses_a_step_the_case_does_not_have(tmp_path, capsys):
    case = _bridge(tmp_path)
    assert main(["formfind", str(case), "--groups", "main_cable", "--step", "5"]) == 1
    assert "the case has 1 step(s)" in capsys.readouterr().err


# ------------------------------------------------------ 7, 8, 15: api
def test_a_study_keeps_a_good_run_when_extract_raises(tmp_path):
    case = api.load_case(CANTILEVER)
    rows = api.study(case, {"loads[0].vector[2]": [-1000.0]}, output_dir=tmp_path / "s",
                     extract=lambda o: {"w": o.monitors["no_such_monitor"][-1]})
    (row,) = rows
    assert row["status"] == "COMPLETED_DYNAMIC" and row["converged"] is True
    assert row["result"] and pathlib.Path(row["result"]).exists()
    assert row["extract_error"].startswith("KeyError")
    with open(tmp_path / "s" / "study.csv", newline="", encoding="utf-8") as fh:
        assert "extract_error" in next(csv.DictReader(fh))


def test_a_study_refuses_a_misspelt_field_before_the_first_run(tmp_path):
    case = api.load_case(CANTILEVER)
    for key in ("solver.total_tim", "sections.beam.widht", "materials.steel.EE"):
        started = []
        with pytest.raises(UserError, match=key.split(".")[-1]):
            api.study(case, {key: [1.0]}, output_dir=tmp_path / "never",
                      on_run=lambda i, v: started.append(i))
        assert started == [] and not (tmp_path / "never" / "study.csv").exists()


def test_the_documented_stop_raises_an_exception_the_api_exports(tmp_path):
    assert "AnalysisInterrupted" in api.__all__
    with pytest.raises(api.AnalysisInterrupted) as stop:
        api.run_case(str(CANTILEVER), output=str(tmp_path / "stop.h5"),
                     on_progress=lambda *a: False, progress_every=10)
    assert pathlib.Path(stop.value.result_path).exists()


# ------------------------------------------------------ 9, 10: view
def _line_result(path, mode=None, times=(0.0, 0.5, 1.0), mon_time=None):
    nodes = np.array([(0.0, 0, 0), (1.0, 0, 0), (2.0, 0, 0)])
    case = "name: v\n" + (f"solver: {{mode: {mode}}}\n" if mode else "")
    with h5py.File(path, "w") as f:
        f.attrs["element_family"] = "beam2"
        f.attrs["case_yaml"] = case
        f.create_dataset("mesh/nodes", data=nodes)
        f.create_dataset("mesh/elements", data=np.array([[0, 1], [1, 2]], np.int32))
        for i, t in enumerate(times):
            g = f.create_group(f"steps/{i + 1:08d}")
            g.attrs["time"] = t
            g.create_dataset("displacement", data=np.full((3, 3), -0.1 * i, np.float32))
            g.create_dataset("axial_force", data=np.array([1.0 * i, 2.0], np.float32))
        mt = list(times) if mon_time is None else mon_time
        f.create_dataset("monitors/time", data=np.asarray(mt, float))
        f.create_dataset("monitors/w", data=np.linspace(0, -1, len(mt)))


def test_view_refuses_a_case_region_field_or_name_it_cannot_use(tmp_path):
    from vfife.post import view
    h5 = tmp_path / "r.h5"
    _line_result(h5)
    with pytest.raises(UserError, match="not the model of this result"):
        view.load(str(h5), case=str(ROOT / "cases" / "case_01_beam_cantilever.yaml"))
    with pytest.raises(UserError, match="--case"):
        view.load(str(h5), case=str(tmp_path / "missing.yaml"))
    with pytest.raises(UserError, match="axial_forse"):
        view.load(str(h5), fields=["axial_forse"])
    with pytest.raises(UserError, match="2 name"):
        view.load(str(h5), names=["a", "b"])
    assert view.load(str(h5), fields=["axial_force"])["meta"]["acts"][0]["frames"] == 3
    # a region of the case that is not there, through the command line
    case = tmp_path / "two.yaml"
    msh = tmp_path / "two.msh"
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n2\n0 1 \"root\"\n"
                   "1 2 \"bar\"\n$EndPhysicalNames\n$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 2 0 0\n$EndNodes\n"
                   "$Elements\n3\n1 15 2 1 1 1\n2 1 2 2 2 1 2\n3 1 2 2 2 2 3\n$EndElements\n")
    doc = yaml.safe_load(CANTILEVER.read_text(encoding="utf-8"))
    doc["mesh"]["path"] = str(msh)
    doc["constraints"][0]["region"] = "root"
    doc["loads"] = []
    doc["output"]["monitors"] = {}
    case.write_text(yaml.safe_dump(doc), encoding="utf-8")
    assert main(["view", str(h5), "--case", str(case), "--hide", "nowhere", "-o", str(tmp_path / "x.html")]) == 1
    assert view.load(str(h5), case=str(case), hide=["bar"])["meta"]["hidden"] == [0, 1]


def test_view_draws_a_load_factor_as_one_not_as_seconds(tmp_path):
    from vfife.post import view
    arc = [0.4, 0.8, -0.3]                         # an arc-length path: back and below zero
    h5 = tmp_path / "a.h5"
    _line_result(h5, mode="static", times=arc)
    act = view.load(str(h5))["meta"]["acts"][0]
    assert (act["mode"], act["axis"], act["frame_mon"]) == ("static", "load", [0, 1, 2])
    html = view.render(view.load(str(h5)))
    assert "load factor" in html and "λ" in html
    dyn = tmp_path / "d.h5"
    _line_result(dyn, mode="dynamic")
    assert view.load(str(dyn))["meta"]["acts"][0]["axis"] == "time"


# ------------------------------------------------------ 11: check
def test_check_refuses_an_output_field_the_model_cannot_produce(tmp_path, capsys):
    case = _case01(tmp_path, fields=["displacement", "axial_forse"])
    report = tmp_path / "r.json"
    assert main(["check", str(case), "-o", str(report)]) == 2
    assert "axial_force" in capsys.readouterr().out
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["status"] == "FAIL" and data["issues"][-1]["code"] == "UNKNOWN_OUTPUT_FIELD"
    good = _case01(tmp_path, fields=["displacement", "axial_force"])
    assert main(["check", str(good), "-o", str(report)]) == 0


# ------------------------------------------------------ 12: one-line errors
def test_yaml_hdf5_and_eof_mistakes_are_one_line_not_a_traceback(tmp_path, monkeypatch, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nmesh: {type: box\n  bad: [\n", encoding="utf-8")
    assert main(["check", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "bad.yaml is not valid YAML (line 3" in err and "Traceback" not in err
    assert main(["plot", str(bad), "-o", str(tmp_path / "p.png")]) == 1
    assert "not an HDF5 file" in capsys.readouterr().err
    # (a .yaml given to `view` is a case for the model page since v1.20: read as one)
    assert main(["view", str(bad), "-o", str(tmp_path / "v.html")]) == 1
    err = capsys.readouterr().err
    assert "bad.yaml is not valid YAML (line 3" in err and "Traceback" not in err
    bad_h5 = tmp_path / "bad.h5"
    bad_h5.write_text("not a result", encoding="utf-8")
    assert main(["view", str(bad_h5), "-o", str(tmp_path / "v.html")]) == 1
    assert "cannot read" in capsys.readouterr().err

    def eof(prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    assert main(["init", str(tmp_path / "c.yaml"), "--mesh", str(ASSETS / "nafems_le10.msh")]) == 1
    assert "input ended" in capsys.readouterr().err


# ------------------------------------------------------ minor ones
def test_sweep_every_must_be_at_least_one(tmp_path):
    from vfife.sweep import write_series
    with pytest.raises(UserError, match="k >= 1"):
        write_series("never-read.yaml", str(tmp_path / "r.h5"), every=0)


def test_sweep_defaults_to_the_result_the_case_names(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake(case, result, **kw):
        seen["result"] = result
        return "x.pvd"
    monkeypatch.setattr("vfife.sweep.write_series", fake)
    case = _case01(tmp_path)
    assert main(["sweep", str(case)]) == 0
    assert seen["result"] == str(tmp_path / "c01.h5")


def test_a_frame_is_given_by_its_step_number_padded_or_not():
    from vfife.modal import frame_key
    frames = ["00000200", "00003338"]
    assert frame_key(frames, "3338", "r.h5") == "00003338"
    assert frame_key(frames, "00003338", "r.h5") == "00003338"
    assert frame_key(frames, None, "r.h5") == "00003338"
    with pytest.raises(UserError, match="3338"):
        frame_key(frames, "12", "r.h5")


def test_capabilities_answers_for_static(capsys):
    assert main(["capabilities", "--mode", "static", "--element", "beam2",
                 "--material", "linear_elastic"]) == 0
    assert "SUPPORTED   beam2" in capsys.readouterr().out
    assert main(["capabilities", "--mode", "static", "--element", "hexa8",
                 "--material", "linear_elastic", "--show-unsupported"]) == 0
    assert "UNSUPPORTED hexa8" in capsys.readouterr().out


def test_arc_length_progress_is_not_reported_as_done(tmp_path):
    calls = []

    def progress(step, n, *rest):
        calls.append((step, n))
        return False                                        # one increment is enough
    with pytest.raises(api.AnalysisInterrupted):
        api.run_case(str(ROOT / "cases" / "case_16b_lee_frame_arc_length.yaml"),
                     output=str(tmp_path / "lee.h5"), on_progress=progress)
    step, n = calls[0]
    assert step == 1 and n == 400                          # max_increments of the case


def test_a_material_value_that_is_not_a_number_is_named_by_its_key(tmp_path, capsys):
    case = _case01(tmp_path)
    doc = yaml.safe_load(case.read_text(encoding="utf-8"))
    doc["materials"]["steel"]["E"] = "2e11x"
    case.write_text(yaml.safe_dump(doc), encoding="utf-8")
    assert main(["check", str(case), "-o", str(tmp_path / "r.json")]) == 1
    assert "`E` is '2e11x'" in capsys.readouterr().err


# ====================================================== 1.0 re-review
# The v1.0 fixes to the user-facing layer, reviewed again: what they
# refused wrongly, said wrongly or left half-done.

#: two hexahedra beside each other; `fixed`/`tip` are faces (quad4), `body`
#: the whole volume
_HEX2 = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
3
2 1 "fixed"
2 2 "tip"
3 3 "body"
$EndPhysicalNames
$Nodes
12
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
5 0 0 1
6 1 0 1
7 1 1 1
8 0 1 1
9 2 0 0
10 2 1 0
11 2 0 1
12 2 1 1
$EndNodes
$Elements
5
1 3 2 1 1 1 4 8 5
2 3 2 2 2 9 10 12 11
3 5 2 3 3 1 2 3 4 5 6 7 8
{second}
$EndElements
"""
_HEX = "4 5 2 3 3 2 9 10 3 6 11 12 7"
_WEDGES = "4 6 2 3 3 2 9 10 6 11 12\n5 6 2 3 3 2 10 3 6 12 7"


def _hex_mesh(tmp_path, wedges=False, name="m.msh"):
    text = _HEX2.replace("$Elements\n5", "$Elements\n" + ("5" if wedges else "4")).format(
        second=_WEDGES if wedges else _HEX)
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_147_init_takes_a_whole_volume_group_named_by_an_option(tmp_path, capsys):
    msh = _hex_mesh(tmp_path)
    # a plane constraint on every node and a load over the body: `body` was
    # "not a Physical group in this mesh"
    out = tmp_path / "c.yaml"
    assert main(["init", str(out), "--mesh", str(msh), "--non-interactive",
                 "--fix", "fixed", "111", "--fix", "body", "010",
                 "--force", "body", "0", "0", "-100"]) == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert {"region": "body", "dofs": [0, 1, 0]} in doc["constraints"]
    capsys.readouterr()
    # a name that is no group lists every group, the volume too
    assert main(["init", str(tmp_path / "d.yaml"), "--mesh", str(msh), "--non-interactive",
                 "--fix", "nowhere", "111", "--force", "tip", "0", "0", "-1"]) == 1
    assert "available: body, fixed, tip" in capsys.readouterr().err
    # a mesh whose only group is the volume: fine when the options name it,
    # refused only when a prompt would have to offer a boundary group
    only = tmp_path / "only.msh"
    only.write_text(msh.read_text(encoding="utf-8").replace('3\n2 1 "fixed"\n2 2 "tip"\n', '1\n')
                    .replace("1 3 2 1 1 1 4 8 5\n2 3 2 2 2 9 10 12 11\n", "")
                    .replace("$Elements\n4", "$Elements\n2"), encoding="utf-8")
    assert main(["init", str(tmp_path / "e.yaml"), "--mesh", str(only), "--non-interactive",
                 "--fix", "body", "001", "--force", "body", "0", "0", "-1"]) == 0
    capsys.readouterr()
    assert main(["init", str(tmp_path / "f.yaml"), "--mesh", str(only)]) == 1
    assert "no boundary-sized" in capsys.readouterr().err


def _run_small(case_path, out):
    assert main(["run", str(case_path), "-o", str(out)]) == 0
    return out


def test_147_view_takes_the_case_of_a_mixed_block_mesh(tmp_path, capsys):
    from vfife.post import view
    msh = _hex_mesh(tmp_path, wedges=True)
    case = tmp_path / "mix.yaml"
    assert main(["init", str(case), "--mesh", str(msh), "--non-interactive",
                 "--fixed-region", "fixed", "--load-region", "tip"]) == 0
    doc = yaml.safe_load(case.read_text(encoding="utf-8"))
    doc["solver"]["n_steps"] = 3000
    case.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    h5 = _run_small(case, tmp_path / "mix.h5")
    # hexa8 (1) + wedge6 (2): "the case has 2 elements and the result 3"
    data = view.load(str(h5), case=str(case))
    assert data["meta"]["n_elements"] == 3


def _shell_line(tmp_path):
    """A wall panel (shell4) with two columns and a beam (Line2) round it."""
    import textwrap
    msh = tmp_path / "frame.msh"
    msh.write_text(textwrap.dedent("""\
        $MeshFormat
        2.2 0 8
        $EndMeshFormat
        $PhysicalNames
        3
        1 1 "columns"
        1 2 "beams"
        2 3 "walls"
        $EndPhysicalNames
        $Nodes
        6
        1 0 0 0
        2 0 0 3
        3 4 0 0
        4 4 0 3
        5 0 5 0
        6 0 5 3
        $EndNodes
        $Elements
        4
        1 1 2 1 1 1 2
        2 1 2 1 1 3 4
        3 1 2 2 2 2 4
        4 3 2 3 3 1 5 6 2
        $EndElements
        """), encoding="utf-8")
    case = {"version": 1, "name": "sl", "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "gmsh", "path": "frame.msh"},
            "materials": {"c": {"type": "linear_elastic", "E": 3.0e10, "nu": 0.2, "rho": 2400.0}},
            "sections": {"wall": {"type": "shell", "material": "c", "thickness": 0.2},
                         "col": {"type": "beam", "material": "c", "shape": "general", "area": 0.16,
                                 "iy": 2.1e-3, "iz": 2.1e-3, "j": 3.6e-3}},
            "assign": [{"region": "walls", "section": "wall"},
                       {"region": "columns", "section": "col", "orientation": [1, 0, 0]},
                       {"region": "beams", "section": "col", "orientation": [0, 1, 0]}],
            "regions": {"ground": {"box": [-1, 5, -1, 6, -0.01, 0.01]},
                        "top": {"box": [3.9, 4.1, -0.1, 0.1, 2.9, 3.1]}},
            "constraints": [{"region": "ground", "dofs": [1, 1, 1, 1, 1, 1]}],
            "loads": [{"region": "top", "vector": [1000.0, 0, 0]}],
            "solver": {"mode": "relaxation", "backend": "numpy", "n_steps": 3000,
                       "damping": {"type": "kinetic"}, "tolerance": 1e-6},
            "output": {"path": str(tmp_path / "sl.h5"), "every": 100000,
                       "monitors": {"u": {"kind": "displacement", "region": "top", "dof": "x"}}}}
    p = tmp_path / "sl.yaml"
    p.write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
    return p


def test_147_view_says_hide_is_not_for_a_shell_line_or_coupled_model(tmp_path):
    from vfife.post import view
    case = _shell_line(tmp_path)
    h5 = _run_small(case, tmp_path / "sl.h5")
    assert view.load(str(h5), case=str(case))["meta"]["n_elements"] == 1     # the case is its model
    # it said "this model has none" of a model with three line members
    with pytest.raises(UserError, match="not supported for a shell_line model"):
        view.load(str(h5), case=str(case), hide=["columns"])


def test_147_check_reports_a_model_it_cannot_build_whether_or_not_fields_are_written(tmp_path, capsys):
    for fields in (["displacement", "axial_force"], None):
        doc = yaml.safe_load((ROOT / "cases" / "case_02_truss_bar.yaml").read_text(encoding="utf-8"))
        doc["mesh"]["path"] = str(ASSETS / "beam_cantilever_v22.msh")
        # a truss refuses a ground motion outside a dynamic run when it is BUILT
        doc["loads"].append({"region": "all", "ground_motion": {
            "path": str(ASSETS / "elcentro_1940_ns_g.dat"), "direction": [1, 0, 0]}})
        if fields is None:
            doc["output"].pop("fields", None)
        else:
            doc["output"]["fields"] = fields
        case = tmp_path / "gm.yaml"
        case.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        report = tmp_path / "gm.json"
        assert main(["check", str(case), "-o", str(report)]) == 2          # was 1 (no report) or 0
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["status"] == "FAIL" and data["issues"][-1]["code"] == "MODEL_BUILD_FAILED"
        assert "ground_motion requires" in data["issues"][-1]["message"]
        assert "vfife check" not in capsys.readouterr().err


def test_147_an_os_error_names_its_own_path_and_a_broken_pipe_is_quiet(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "a_file"
    blocker.write_text("x", encoding="utf-8")
    case = _case01(tmp_path)
    assert main(["check", str(case), "-o", str(blocker / "r.json")]) == 1
    err = capsys.readouterr().err
    assert "cannot read or write" in err and str(blocker) in err and str(case) not in err

    def broken(args):
        raise BrokenPipeError(32, "Broken pipe")
    monkeypatch.setattr("vfife.cli.cmd_capabilities", broken)
    assert main(["capabilities"]) == 141
    assert capsys.readouterr().err == ""


def test_147_a_yaml_error_names_the_file_it_is_in(tmp_path, capsys):
    contract = tmp_path / "contract.yaml"
    contract.write_text("version: 1\nmetrics: [a, b\n", encoding="utf-8")
    case = _case01(tmp_path)
    assert main(["verify", str(case), "--contract", str(contract),
                 "--result", str(tmp_path / "none.h5")]) == 1
    err = capsys.readouterr().err
    assert f"{contract} is not valid YAML" in err and str(case) not in err


def test_147_a_study_refuses_a_misspelt_creep_key_or_unit_before_the_first_run(tmp_path):
    case = api.load_case(ROOT / "cases" / "case_29_creep_bar.yaml")
    for key, value, word in (("materials.concrete.creep.curvee", 1.0, "curvee"),
                             ("units.lenght", "m", "lenght")):
        started = []
        with pytest.raises(UserError, match=word) as exc:
            api.study(case, {key: [value]}, output_dir=tmp_path / "never",
                      on_run=lambda i, v: started.append(i))
        assert exc.value.code == "STUDY_UNKNOWN_FIELD" and started == []
    # a bad VALUE is still the run's own business
    from vfife.api import _check_field_names
    change = {"materials.concrete.creep.curve": [[0, 0], [1, -5]]}
    _check_field_names(case.with_changes(change), change)


def test_147_mesh_lists_line_members_beside_a_volume_not_boundary_faces(tmp_path, capsys):
    assert main(["mesh", str(_hex_mesh(tmp_path))]) == 0
    assert "also in file" not in capsys.readouterr().out           # quad4 faces are groups
    assert main(["mesh", str(ASSETS / "sze_roof_16_v22.msh")]) == 0
    assert "also in file" not in capsys.readouterr().out           # a shell's edge lines
    assert main(["mesh", str(ASSETS / "coupled_hex_truss_v22.msh")]) == 0
    assert "also in file  truss2 (1)" in capsys.readouterr().out


def test_147_view_fields_by_group_held_but_not_drawn_and_names_without_empties(tmp_path, capsys):
    from vfife.post import view
    h5 = tmp_path / "e.h5"
    nodes = np.array([(0.0, 0, 0), (1.0, 0, 0), (2.0, 0, 0)])
    with h5py.File(h5, "w") as f:
        f.attrs["element_family"] = "beam2"
        f.attrs["case_yaml"] = "name: e\n"
        f.create_dataset("mesh/nodes", data=nodes)
        f.create_dataset("mesh/elements", data=np.array([[0, 1], [1, 2]], np.int32))
        for i in range(2):
            g = f.create_group(f"steps/{i + 1:08d}")
            g.attrs["time"] = float(i)
            g.create_dataset("displacement", data=np.zeros((3, 3), np.float32))
            g.create_dataset("rotation", data=np.zeros((3, 3), np.float32))
            for c in ("axial_force_i", "axial_force_j"):
                g.create_dataset(c, data=np.array([1.0, 2.0], np.float32))
    names = {fl["name"] for fl in view.load(str(h5), fields=["end_actions"])["meta"]["acts"][0]["fields"]}
    assert {"axial_force_i", "axial_force_j"} <= names             # the group's held components
    with pytest.raises(UserError, match="holds but `vfife view` does not draw"):
        view.load(str(h5), fields=["rotation"])
    with pytest.raises(UserError, match="does not hold"):
        view.load(str(h5), fields=["von_mises"])
    # "a, b," was three names for two results; empty entries are dropped,
    # as --hide and --fields drop them
    assert main(["view", str(h5), str(h5), "--names", "a, b,", "-o", str(tmp_path / "p.html")]) == 0


def test_147_view_labels_a_creep_run_by_the_age(tmp_path):
    from vfife.post import view
    h5 = tmp_path / "c.h5"
    _line_result(h5, mode="relaxation", times=(1.0, 1.5, 2.0))
    with h5py.File(h5, "a") as f:
        f.create_dataset("monitors/age", data=np.array([28.0, 100.0, 3650.0]))
    act = view.load(str(h5))["meta"]["acts"][0]
    assert act["axis"] == "age" and act["ages"] == [28.0, 100.0, 3650.0]
    assert "龄期 age" in view.render(view.load(str(h5)))
    plain = tmp_path / "p.h5"
    _line_result(plain, mode="relaxation")
    assert view.load(str(plain))["meta"]["acts"][0]["axis"] == "load"


def test_147_arc_length_progress_ends_at_done(tmp_path):
    calls = []
    api.run_case(str(ROOT / "cases" / "case_16b_lee_frame_arc_length.yaml"),
                 output=str(tmp_path / "lee.h5"),
                 on_progress=lambda step, n, *rest: calls.append((step, n)), progress_every=1)
    step, n = calls[-1]
    assert step == n and step < 400                  # done, not 61 of 400


def test_147_static_formfind_is_quiet_says_static_and_makes_its_folder(tmp_path, capsys):
    case = _bridge(tmp_path, mode="static")
    lines = []
    found = formfind.find(str(case), ["main_cable", "hangers"], log=lines.append)
    assert found.converged and found.mode == "static"
    assert "static        Newton" not in capsys.readouterr().out     # log= is honoured
    out = tmp_path / "new" / "deep" / "found.yaml"                 # a folder that is not there
    assert main(["formfind", str(case), "--groups", "main_cable,hangers", "-o", str(out)]) == 0
    assert "static solutions" in out.read_text(encoding="utf-8").splitlines()[2]


def test_147_an_input_form_case_with_an_element_key_is_refused(tmp_path):
    from vfife.io import schema
    doc = yaml.safe_load((ROOT / "cases" / "case_01_beam_cantilever.yaml").read_text(encoding="utf-8"))
    doc["element"] = {"type": "beam2_timoshenko"}             # overwritten by the sections, silently
    with pytest.raises(UserError, match="`element`") as exc:
        schema.from_dict(doc)
    assert exc.value.code == "INTERNAL_KEY_IN_INPUT_FORM"
