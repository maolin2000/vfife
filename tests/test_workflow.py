"""The workflow a user actually walks: gmsh -> yaml -> run -> ParaView.

Every case in `cases/` was written by someone who already knew the answers, so
none of them exercises the parts that only bite a newcomer: not knowing the
region names, mistyping a material, being handed a scaffold that cannot run,
pressing Ctrl-C. This file walks the path from a mesh file with no case beside
it, and asserts the things that make the difference between "it works" and "it
works without being told anything first".
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from vfife.core.diagnostics import AnalysisInterrupted, StopAnalysis, UserError
from vfife.elements.library import ELEMENTS
from vfife.materials import base as mat_base
from vfife.scaffold import _formulation_for, generic_gmsh_case

DATA = os.path.join(os.path.dirname(__file__), "data")
TET = os.path.join(DATA, "box_tet4_v41.msh")
HEX = os.path.join(DATA, "box_hexa8_v41.msh")


def cli(*args, cwd=None):
    return subprocess.run([sys.executable, "-m", "vfife", *args],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8",
                               "PYTHONPATH": os.path.dirname(os.path.dirname(
                                   os.path.abspath(__file__)))})


# --------------------------------------------------------------------------- #
#  knowing what is in the mesh, before there is a case
# --------------------------------------------------------------------------- #


def test_mesh_command_names_the_regions_without_a_case_file():
    """The chicken-and-egg: you needed a region name to write the case that
    would have told you the region names."""
    r = cli("mesh", TET)
    assert r.returncode == 0, r.stderr
    for group in ("left_face", "right_face", "solid"):
        assert group in r.stdout
    assert "tet4" in r.stdout
    # and it says which formulations are legal, which is what stops the next
    # mistake before it is made
    assert "full" in r.stdout


def test_mesh_command_reports_the_extent_of_each_group():
    """`loaded` and `support` are indistinguishable by name alone."""
    r = cli("mesh", TET)
    assert "x=0" in r.stdout          # left_face is a plane at x = 0
    assert r.returncode == 0


def test_mesh_command_has_a_json_form_for_tools():
    import json
    r = cli("mesh", TET, "--format", "json")
    d = json.loads(r.stdout)
    assert d["element"] == "tet4"
    assert set(d["regions"]) >= {"left_face", "right_face", "solid"}
    assert d["formulations"] == ["full"]


# --------------------------------------------------------------------------- #
#  the scaffold must emit something that runs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("element", sorted(ELEMENTS))
def test_the_scaffold_never_picks_an_unsupported_formulation(element):
    """It used to hard-code `b-bar`, and Gmsh's default 3-D output is tet4 --
    which supports `full` only. The tool wrote a file the tool could not run."""
    assert _formulation_for(element) in ELEMENTS[element].formulations


def test_a_generated_case_for_a_tet_mesh_says_full():
    text = generic_gmsh_case("t", TET, "left_face", "right_face",
                             [0.0, 0.0, -1.0], element="tet4")
    assert "formulation: full" in text


def test_preflight_catches_an_unsupported_formulation_before_the_solver_does(tmp_path):
    """And says what to set instead, rather than raising from precompute after
    preflight has already printed PASS."""
    case = tmp_path / "bad.yaml"
    case.write_text(f"""
name: bad
mesh: {{type: gmsh, path: {TET}}}
materials:
  main: {{type: linear_elastic, E: 3.0e10, nu: 0.2, rho: 2400.0}}
sections:
  body: {{type: solid, material: main, formulation: b-bar}}
constraints: [{{region: left_face, dofs: [1, 1, 1]}}]
solver: {{mode: relaxation, n_steps: 10, backend: numpy}}
""", encoding="utf-8")
    r = cli("info", str(case))
    assert "FORMULATION_UNSUPPORTED" in r.stdout
    assert "tet4" in r.stdout and "full" in r.stdout
    assert "Traceback" not in r.stderr


# --------------------------------------------------------------------------- #
#  mistakes read as sentences
# --------------------------------------------------------------------------- #


def test_an_unknown_material_is_a_sentence_with_a_suggestion(tmp_path):
    case = tmp_path / "typo.yaml"
    case.write_text(f"""
name: typo
mesh: {{type: gmsh, path: {TET}}}
materials:
  main: {{type: linear_elasitc, E: 3.0e10, nu: 0.2, rho: 2400.0}}
sections:
  body: {{type: solid, material: main, formulation: full}}
constraints: [{{region: left_face, dofs: [1, 1, 1]}}]
solver: {{mode: relaxation, n_steps: 10, backend: numpy}}
""", encoding="utf-8")
    r = cli("run", str(case))
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "unknown material 'linear_elasitc'" in r.stderr
    assert "did you mean 'linear_elastic'" in r.stderr


def test_a_bad_field_name_names_the_field_and_not_pydantic(tmp_path):
    case = tmp_path / "field.yaml"
    case.write_text(f"""
name: f
mesh: {{type: gmsh, path: {TET}}}
materials:
  main: {{type: linear_elastic, E: 3.0e10, nu: 0.2, rho: 2400.0}}
sections:
  body: {{type: solid, material: main, formulation: full, Iz: 1.0}}
solver: {{mode: relaxation, n_steps: 10}}
""", encoding="utf-8")
    r = cli("info", str(case))
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "sections.body.Iz" in r.stderr


def test_a_missing_file_says_so():
    r = cli("info", "/nonexistent/case.yaml")
    assert r.returncode == 1
    assert "no such file" in r.stderr
    assert "Traceback" not in r.stderr


def test_debug_still_gives_the_traceback(tmp_path):
    """The stack is one flag away, not gone."""
    case = tmp_path / "typo.yaml"
    case.write_text(f"""
name: typo
mesh: {{type: gmsh, path: {TET}}}
materials:
  main: {{type: linear_elasitc, E: 1.0, nu: 0.2, rho: 1.0}}
sections:
  body: {{type: solid, material: main, formulation: full}}
solver: {{mode: relaxation, n_steps: 10}}
""", encoding="utf-8")
    r = cli("--debug", "run", str(case))
    assert "Traceback" in r.stderr


def test_unknown_material_carries_a_machine_readable_code():
    """A GUI branches on the code; it must not have to parse English."""
    with pytest.raises(UserError) as exc:
        mat_base.create("nonesuch")
    assert exc.value.code == "UNKNOWN_MATERIAL"
    assert exc.value.hint
    assert set(exc.value.as_dict()) == {"code", "message", "hint"}


# --------------------------------------------------------------------------- #
#  stopping without losing the run
# --------------------------------------------------------------------------- #


def test_a_cancelled_run_still_writes_a_readable_resumable_result(tmp_path):
    """`on_progress` returning False is the cancel token a GUI needs, and a
    cancelled run must leave a file, not a truncated one."""
    import h5py

    from vfife import api

    case = tmp_path / "c.yaml"
    case.write_text(f"""
name: c
mesh: {{type: gmsh, path: {HEX}}}
materials:
  main: {{type: linear_elastic, E: 3.0e10, nu: 0.2, rho: 2400.0}}
sections:
  body: {{type: solid, material: main, formulation: b-bar}}
constraints: [{{region: left_face, dofs: [1, 1, 1]}}]
loads: [{{region: right_face, vector: [0.0, 0.0, -1.0e3]}}]
solver: {{mode: relaxation, n_steps: 20000, backend: numpy}}
""", encoding="utf-8")
    prepared = api.prepare_case(str(case))
    out = tmp_path / "c.h5"

    def stop_at_200(step, n, resid, ke, resets):
        return False if step >= 200 else None

    with pytest.raises(AnalysisInterrupted) as exc:
        api.run_case(prepared, output=str(out), on_progress=stop_at_200,
                     progress_every=50)

    assert exc.value.report["status"] == "INTERRUPTED"
    with h5py.File(out, "r") as f:                # closed properly, not truncated
        assert {"diagnostics", "mesh", "monitors"} <= set(f)
    # the checkpoint is its own file beside the result
    with h5py.File(tmp_path / "c.ckpt.h5", "r") as f:
        assert int(f["/checkpoint/step"][0]) >= 200


def test_returning_true_or_none_from_on_progress_does_not_stop(tmp_path):
    """Only False cancels; a callback that just prints must not kill the run."""
    from vfife import api

    case = tmp_path / "c.yaml"
    case.write_text(f"""
name: c
mesh: {{type: gmsh, path: {HEX}}}
materials:
  main: {{type: linear_elastic, E: 3.0e10, nu: 0.2, rho: 2400.0}}
sections:
  body: {{type: solid, material: main, formulation: b-bar}}
constraints: [{{region: left_face, dofs: [1, 1, 1]}}]
solver: {{mode: relaxation, n_steps: 60, backend: numpy}}
""", encoding="utf-8")
    prepared = api.prepare_case(str(case))
    seen = []
    outcome = api.run_case(prepared, output=str(tmp_path / "c.h5"),
                           on_progress=lambda *a: seen.append(a) or None,
                           progress_every=20)
    assert seen                                   # it really was called
    assert outcome.result_path


# --------------------------------------------------------------------------- #
#  a mixed mesh is a mesh, not a mistake
# --------------------------------------------------------------------------- #


def _mixed_mesh(tmp_path):
    """tet4 + pyramid5, which is what one `Recombine Surface` produces.

    Gmsh caps every quadrilateral face that borders a tetrahedral volume with a
    pyramid, so recombining a single surface makes the whole mesh mixed. Written
    by hand here so the test does not need Gmsh installed.
    """
    p = tmp_path / "mixed.msh"
    nodes = "\n".join(
        f"{i + 1} {x} {y} {z}" for i, (x, y, z) in enumerate(
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
             (0.5, 0.5, 1), (0.5, 0.5, -1)]))
    p.write_text(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        '$PhysicalNames\n1\n3 1 "solid"\n$EndPhysicalNames\n'
        f"$Nodes\n6\n{nodes}\n$EndNodes\n"
        "$Elements\n2\n"
        "1 7 2 1 1 1 2 3 4 5\n"          # pyramid5
        "2 4 2 1 1 1 2 3 6\n"            # tet4
        "$EndElements\n", encoding="utf-8")
    return p


def test_a_mixed_mesh_imports_as_several_blocks(tmp_path):
    """It used to raise MIXED_ELEMENT_TYPES. Both element types are now kept."""
    from vfife.io import gmsh

    mesh, element = gmsh.read(str(_mixed_mesh(tmp_path)))
    assert set(mesh.cell_blocks) == {"tet4", "pyramid5"}
    assert mesh.cell_blocks["tet4"].shape == (1, 4)
    assert mesh.cell_blocks["pyramid5"].shape == (1, 5)
    # the reported name stays a real element -- the largest block -- so every
    # caller that looks an element up by name still finds one
    from vfife.elements.library import ELEMENTS
    assert element in ELEMENTS


def test_the_blocks_share_one_node_numbering(tmp_path):
    """Which is what makes them one model rather than two.

    The pyramid and the tetrahedron here share the face 1-2-3; if the reader
    renumbered per block they would be separate bodies touching nowhere, and
    the analysis would silently solve two disconnected pieces.
    """
    from vfife.io import gmsh

    mesh, _ = gmsh.read(str(_mixed_mesh(tmp_path)))
    shared = set(mesh.cell_blocks["tet4"].ravel().tolist()) & set(
        mesh.cell_blocks["pyramid5"].ravel().tolist())
    assert len(shared) == 3


def test_the_mesh_command_describes_every_block(tmp_path):
    r = cli("mesh", str(_mixed_mesh(tmp_path)))
    assert r.returncode == 0, r.stderr
    assert "mixed" in r.stdout
    assert "tet4" in r.stdout and "pyramid5" in r.stdout
    # one formulation setting has to satisfy every block, and tet4/pyramid5
    # have no volumetric part to average, so b-bar must not be offered
    forms = [l for l in r.stdout.splitlines() if "formulations" in l][0]
    assert "full" in forms and "b-bar" not in forms
    # and it says up front what a mixed mesh cannot do, rather than letting
    # the run find out several minutes in
    backends = [l for l in r.stdout.splitlines() if "backends" in l][0]
    assert "numpy" in backends and "does not run on numba" in backends


def test_solid_and_structural_elements_still_cannot_share_a_model(tmp_path):
    """Mixing hexahedra and tetrahedra is one solver; mixing solids and beams
    is two, and that refusal has to stay."""
    from vfife.core.diagnostics import UserError
    from vfife.io import gmsh

    p = tmp_path / "families.msh"
    nodes = "\n".join(f"{i + 1} {x} {y} {z}" for i, (x, y, z) in enumerate(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (3, 0, 0)]))
    p.write_text(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        f"$Nodes\n6\n{nodes}\n$EndNodes\n"
        "$Elements\n2\n"
        "1 4 2 1 1 1 2 3 4\n"            # tet4
        "2 3 2 1 1 1 2 3 4\n"            # quad4 shell
        "$EndElements\n", encoding="utf-8")
    # a quad4 beside a tet4 is the volume's own boundary face, which is legal;
    # it is only a conflict when the structural block is the primary topology
    mesh, element = gmsh.read(str(p))
    assert element == "tet4"


def test_a_mixed_mesh_runs_on_the_fused_backend(tmp_path):
    """Refused until v0.1 (MIXED_MESH_BACKEND): the per-block
    assembler sized its output from the block's own highest node id. Both
    blocks now assemble into the model's node count and the run proceeds."""
    import numpy as np
    import pytest as _pytest

    _pytest.importorskip("numba")
    from vfife.core.mesh import Mesh
    from vfife.core.solver import Solver
    from vfife.materials import create

    nodes = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                      [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                      [0.5, 0.5, 2.0]])
    mesh = Mesh(nodes=nodes,
                elements=np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32),
                cell_blocks={"hexa8": np.array([[0, 1, 2, 3, 4, 5, 6, 7]],
                                               dtype=np.int32),
                             "pyramid5": np.array([[4, 5, 6, 7, 8]],
                                                  dtype=np.int32)})
    mat = create("linear_elastic", E=3.0e10, nu=0.2, rho=2400.0)
    problem = {"element": {"type": "hexa8", "formulation": "full"},
               "solver": {"mode": "relaxation", "n_steps": 1,
                          "backend": "numba"}}
    s = Solver(problem, mesh, mat, log=lambda *_: None)
    assert [b.element for b in s.blocks] == ["hexa8", "pyramid5"]
    assert s.blocks[0].force.assemble(np.zeros((1, 8, 3))).shape == (9, 3)


# --------------------------------------------------------------------------- #
#  output.fields: a name the model cannot produce
# --------------------------------------------------------------------------- #

DATA = os.path.join(os.path.dirname(__file__), "data")
HEXA = os.path.join(DATA, "box_hexa8_v41.msh")


def _case(tmp_path, fields, material="linear_elastic", extra=""):
    p = tmp_path / "f.yaml"
    p.write_text(f"""
name: f
mesh: {{type: gmsh, path: {HEXA}}}
materials:
  main: {{type: {material}, E: 3.0e10, nu: 0.2, rho: 2400.0{extra}}}
sections:
  body: {{type: solid, material: main, formulation: b-bar}}
constraints: [{{region: left_face, dofs: [1, 1, 1]}}]
solver: {{mode: relaxation, n_steps: 5, backend: numpy}}
output: {{fields: [{", ".join(fields)}]}}
""", encoding="utf-8")
    return p


def test_a_mistyped_output_field_is_refused_before_the_run(tmp_path):
    """It used to be dropped in silence.

    `output.fields` was filtered by name at write time, so `von_misses` for
    `von_mises` gave a preflight PASS with no warnings, a complete analysis, and
    a result file with the field quietly absent -- found in ParaView, after the
    run. On a long analysis that is an hour spent on a typo.
    """
    r = cli("info", str(_case(tmp_path, ["displacement", "von_misses"])))
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "von_misses" in r.stderr
    assert "did you mean 'von_mises'" in r.stderr


def test_it_is_caught_by_info_so_you_never_start_the_run(tmp_path):
    """`info` exists to find mistakes before the solver does."""
    case = _case(tmp_path, ["displacement", "nonesuch"])
    assert cli("info", str(case)).returncode == 1
    # `run` reports it as a preflight failure, exit 2
    assert cli("run", str(case), "-o", str(tmp_path / "o.h5")).returncode == 2


def test_the_message_lists_what_this_model_can_produce(tmp_path):
    """Not a global catalogue: what THIS element and THIS material offer."""
    r = cli("info", str(_case(tmp_path, ["nonesuch"])))
    for field in ("displacement", "von_mises", "stress", "stress_nodal"):
        assert field in r.stderr
    # linear elastic has no damage, so a CDP-only field must not be listed
    assert "damage_t" not in r.stderr


def test_every_field_the_model_does_produce_is_accepted(tmp_path):
    """The check must not become a second place to forget an element."""
    r = cli("info", str(_case(tmp_path, [
        "displacement", "von_mises", "stress", "stress_nodal",
        "sigma_xx", "sigma_yy", "sigma_zz"])))
    assert r.returncode == 0, r.stderr


def test_a_material_specific_field_stays_legal(tmp_path):
    """The available set is asked of the model, not tabulated, so a field that
    only exists because of the MATERIAL has to pass.

    A static list would have had to enumerate every material's `output_fields`,
    which is exactly the kind of second copy that drifts.
    """
    from vfife.materials import create

    mat = create("j2", E=2.0e11, nu=0.3, rho=7850.0, yield_strength=2.5e8)
    extra = set(mat.output_fields(mat.init_state(1)))
    assert extra, "this test needs a material that contributes fields"

    p = tmp_path / "j2.yaml"
    p.write_text(f"""
name: j2case
mesh: {{type: gmsh, path: {HEXA}}}
materials:
  main: {{type: j2, E: 2.0e11, nu: 0.3, rho: 7850.0, yield_strength: 2.5e8}}
sections:
  body: {{type: solid, material: main, formulation: b-bar}}
constraints: [{{region: left_face, dofs: [1, 1, 1]}}]
solver: {{mode: relaxation, n_steps: 5, backend: numpy}}
output: {{fields: [displacement, {", ".join(sorted(extra))}]}}
""", encoding="utf-8")
    r = cli("info", str(p))
    assert r.returncode == 0, r.stderr

    # ...and the same field is NOT accepted for a material that lacks it
    r2 = cli("info", str(_case(tmp_path, ["displacement", sorted(extra)[0]])))
    assert r2.returncode == 1
    assert sorted(extra)[0] in r2.stderr


def test_an_empty_field_list_is_not_an_error(tmp_path):
    """Omitting `output.fields` means "write what you have", not "write none"."""
    from vfife.api import check_output_fields

    check_output_fields(object(), [])      # must not raise, must not probe
    check_output_fields(object(), None)


def test_a_field_that_only_appears_once_running_is_still_accepted(tmp_path):
    """The check must not be stricter than the truth.

    `reaction_force` is gated on a `last_reaction` attribute the solver creates
    when it first computes reactions -- during the run, after the probe. A
    naive probe reported it unavailable and refused a case that produces it
    perfectly well, which is worse than the silent drop it replaced: it turns
    working input into an error.
    """
    r = cli("info", str(_case(tmp_path, ["displacement", "reaction_force"])))
    assert r.returncode == 0, r.stderr


def test_a_field_the_case_did_not_enable_is_still_refused(tmp_path):
    """`active` exists only when erosion is configured, and that IS a real
    mistake -- so the leniency above must not become blanket permission."""
    r = cli("info", str(_case(tmp_path, ["displacement", "active"])))
    assert r.returncode == 1
    assert "active" in r.stderr


# --------------------------------------------------------------------------- #
#  the machine, not the input
# --------------------------------------------------------------------------- #


def _force_kw(backend, **kw):
    """Build a one-element force evaluator on the named backend."""
    import numpy as _np

    from vfife.elements import continuum
    from vfife.kernels import make_force
    from vfife.materials import create

    nodes = _np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    conn = _np.arange(8, dtype=_np.int32)[None, :]
    mat = create("linear_elastic", E=3.0e10, nu=0.2, rho=2400.0)
    pre = continuum.precompute(nodes, conn, element="hexa8",
                               formulation="full", material=mat)
    x0 = nodes[conn]
    x0_e = _np.ascontiguousarray(x0 - x0.mean(axis=1, keepdims=True))
    return make_force(pre, conn, x0_e, mat, backend, **kw)


def test_an_unknown_backend_is_a_sentence_that_lists_the_real_ones():
    with pytest.raises(UserError) as exc:
        _force_kw("cuda")                      # a plausible wrong guess
    assert exc.value.code == "UNKNOWN_BACKEND"
    assert "numpy" in exc.value.hint and "warp" in exc.value.hint


def test_a_backend_that_is_not_installed_says_how_to_get_it(monkeypatch):
    """numba and warp are optional installs. Their absence is a normal state of
    a machine, so it has to read as advice -- including the option that always
    works."""
    import builtins

    real = builtins.__import__

    def blocked(name, *a, **k):
        if "numba" in name:
            raise ImportError("no numba here")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(UserError) as exc:
        _force_kw("numba")
    assert exc.value.code == "BACKEND_UNAVAILABLE"
    assert "numba" in exc.value.message
    assert "solver.backend = numpy" in exc.value.hint


def test_asking_for_a_gpu_on_a_machine_without_one_is_a_sentence():
    """The one environment failure most users will actually meet.

    A shipped case asks for the warp backend; anyone without an NVIDIA card who
    ran it got forty-odd lines of Warp banner and then a bare RuntimeError
    about warp's own device model. The CLI turned input mistakes into sentences
    and left environment mistakes as stack traces, though both are things the
    user has to act on.
    """
    pytest.importorskip("warp")
    import warp as wp

    try:
        wp.init()
        has_cuda = wp.is_cuda_available()
    except Exception:
        has_cuda = False
    if has_cuda:
        pytest.skip("this machine has CUDA, so there is no failure to check")

    with pytest.raises(UserError) as exc:
        _force_kw("warp", device="cuda", precision="mixed")
    assert exc.value.code == "BACKEND_UNAVAILABLE"
    assert "warp" in exc.value.message
    assert "solver.backend = numpy" in exc.value.hint


def test_a_gpu_case_run_from_the_cli_fails_as_one_line_not_a_traceback():
    """End to end: the message has to survive the top-level handler."""
    pytest.importorskip("warp")
    import warp as wp

    try:
        wp.init()
        if wp.is_cuda_available():
            pytest.skip("this machine has CUDA")
    except Exception:
        pass

    # THE CASE IS WRITTEN HERE, not taken from `cases/`. This test used to
    # point at case_06 because that was the shipped case asking for CUDA;
    # when case_06 moved to numba (v0.1 -- a 256-element model that
    # needed a GPU could be run by nobody, and so had never been verified)
    # this test failed, which is a test coupled to a decision it is not
    # about. What it is about is the CLI's handling of a case that asks for
    # a device this machine has not got.
    import tempfile
    case = os.path.join(tempfile.mkdtemp(), "gpu.yaml")
    with open(case, "w", encoding="utf-8") as fh:
        fh.write("""
name: gpu
units: {length: m, force: N, stress: Pa}
mesh: {dimensions: [1.0, 0.2, 0.2], divisions: [4, 2, 2]}
materials: {steel: {type: linear_elastic, E: 2.0e+11, nu: 0.3, rho: 7850.0}}
sections: {body: {type: solid, material: steel}}
regions: {fix: {where: {x: 0.0}}}
constraints: [{region: fix, dofs: [1, 1, 1]}]
solver: {backend: warp, device: cuda, precision: mixed, mode: relaxation, n_steps: 10}
output: {path: gpu.h5, monitors: {}}
""")
    r = cli("info", case)
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "solver.backend = numpy" in r.stderr


# --------------------------------------------------------------------------- #
#  picking a run back up
# --------------------------------------------------------------------------- #


def _restartable(tmp_path, n_steps):
    p = tmp_path / f"rs{n_steps}.yaml"
    p.write_text(f"""
name: rs
mesh: {{type: gmsh, path: {HEX}}}
materials:
  main: {{type: linear_elastic, E: 3.0e10, nu: 0.2, rho: 2400.0}}
sections:
  body: {{type: solid, material: main, formulation: b-bar}}
constraints: [{{region: left_face, dofs: [1, 1, 1]}}]
loads: [{{region: right_face, vector: [0.0, 0.0, -1.0e3]}}]
solver: {{mode: relaxation, n_steps: {n_steps}, backend: numpy}}
output:
  checkpoint_every: 20
  monitors: {{tip: {{kind: displacement, region: right_face, dof: z}}}}
""", encoding="utf-8")
    return p


def test_a_restart_with_more_budget_continues_where_it_stopped(tmp_path):
    """`--restart` is advertised, and until now nothing ran it."""
    first = str(tmp_path / "a.h5")
    r = cli("run", str(_restartable(tmp_path, 40)), "-o", first)
    assert r.returncode == 3                      # MAX_STEPS, as intended here

    r2 = cli("run", str(_restartable(tmp_path, 400)), "-o",
             str(tmp_path / "b.h5"), "--restart", first)
    assert "restarted from step 40" in r2.stdout
    assert "Traceback" not in r2.stderr


def test_a_restart_with_no_budget_left_is_a_sentence(tmp_path):
    """The natural mistake: a run ends with MAX_STEPS, so you re-run the same
    command with `--restart` and change nothing else.

    That iterates zero times, and the reporting then read state the loop never
    produced -- it came out as `UnboundLocalError: F_un`, a Python internal
    name for what is really a scheduling mistake with an obvious remedy.
    """
    case = str(_restartable(tmp_path, 40))
    first = str(tmp_path / "a.h5")
    cli("run", case, "-o", first)

    r = cli("run", case, "-o", str(tmp_path / "b.h5"), "--restart", first)
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "UnboundLocalError" not in r.stderr
    assert "no iterations left" in r.stderr
    assert "solver.n_steps" in r.stderr


# --------------------------------------------------------------------------- #
#  which way a reaction points
# --------------------------------------------------------------------------- #


def test_a_reaction_opposes_the_load_and_matches_its_own_field():
    """A support holding a downward load pushes UP.

    The continuum solver reported the negation: its monitor was built from
    `-sum(F_int)` while the `reaction_force` field it writes to ParaView was
    `F_int - F_ext`. Two answers for one quantity, opposite in sign, and the
    monitor also dropped the -F_ext term -- wrong whenever a load acts directly
    on a constrained node.

    It survived because every verification contract that checks a reaction runs
    a structural element (beam, truss, shell), and those solvers already had the
    right sign. The one contract covering the solid path, case_06, had never
    been run: its `axial` metric expected +8.0e5 for an applied -800 kN and got
    -800049 N -- right magnitude, wrong sign.

    This test is the cheap version of that contract, so the next regression is
    caught in the fast tier rather than overnight.
    """
    import numpy as np

    from vfife.core.mesh import Mesh
    from vfife.core.solver import Solver
    from vfife.materials import create

    n = 3
    xs = np.linspace(0.0, 2.0, n + 1)
    nid, pts = {}, []
    for i, x in enumerate(xs):
        for j, y in enumerate((0.0, 0.5)):
            for k, z in enumerate((0.0, 0.5)):
                nid[(i, j, k)] = len(pts)
                pts.append((x, y, z))
    nodes = np.array(pts, float)
    conn = np.array([[nid[(i, 0, 0)], nid[(i + 1, 0, 0)], nid[(i + 1, 1, 0)],
                      nid[(i, 1, 0)], nid[(i, 0, 1)], nid[(i + 1, 0, 1)],
                      nid[(i + 1, 1, 1)], nid[(i, 1, 1)]] for i in range(n)],
                    dtype=np.int32)
    mesh = Mesh(nodes=nodes, elements=conn)
    mesh.regions["root"] = np.flatnonzero(np.abs(nodes[:, 0]) < 1e-12)
    mesh.regions["tip"] = np.flatnonzero(np.abs(nodes[:, 0] - 2.0) < 1e-12)

    P = -1000.0                               # downward
    solver = Solver(
        {"element": {"type": "hexa8", "formulation": "b-bar"},
         "constraints": [{"region": "root", "dofs": [1, 1, 1]}],
         "loads": [{"region": "tip", "vector": [0.0, 0.0, P]}],
         "solver": {"mode": "relaxation", "n_steps": 20000, "backend": "numpy"},
         "output": {"monitors": {
             "R": {"kind": "reaction", "region": "root", "dof": "z"}}}},
        mesh, create("linear_elastic", E=2.0e11, nu=0.3, rho=7850.0),
        log=lambda *_: None)
    solver.run()

    monitor = solver.monitors["R"].values[-1]
    field = float(solver.nodal_fields(None)["reaction_force"][
        mesh.regions["root"], 2].sum())

    assert monitor > 0, (
        f"a support holding a {P} N load reported {monitor:+.1f} N; a downward "
        f"load is held by an upward reaction")
    assert monitor == pytest.approx(-P, rel=1e-3)
    assert monitor == pytest.approx(field, rel=1e-9), (
        "the monitor and the reaction_force field must be the same quantity: "
        f"monitor {monitor:+.3f}, field {field:+.3f}")
