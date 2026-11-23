"""Form-finding: the unstressed lengths that make the mesh the dead-load
state.

`vfife formfind` relaxes the dead-load step, resets every cable element's
unstressed length by L_u <- L_u L0 / L, and repeats until no node moves
more than the tolerance. The anchor is a suspension bridge with weightless
cable and hangers, where the answer is closed-form (H_w = w l^2 / 8f,
every hanger carries w s, and L_u follows from the tension): the found
lengths must reproduce it, and a case written with them must sit still.

Also here: per-element `unstressed_lengths` (what form-finding writes),
and the Gmsh reader recording which ELEMENTS a line group holds -- without
that a hanger group, whose every node is a deck or cable node, would have
swallowed the deck's elements under the every-node-inside rule.
"""
import pathlib
import sys

import numpy as np
import pytest
import yaml

from vfife import formfind
from vfife.api import _with_element, prepare_case
from vfife.cli import main
from vfife.core.structural import make_structural_solver

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

CASE = ROOT / "cases" / "case_23_suspension_bridge.yaml"


def _solver(path):
    prep = prepare_case(path)
    p = _with_element(prep.case, prep.element, prep.case_path)
    return make_structural_solver(p, prep.mesh, prep.material)


def _small_bridge(td, n=10, with_lengths=False, dead_only=False, **kw):
    """The shipped case_23 with n hangers-and-one, written without (or with)
    the closed-form unstressed lengths, and with or without its live step."""
    from generate_suspension_bridge import build
    msh = pathlib.Path(td) / "bridge.msh"
    info = build(str(msh), n=n, **kw)
    doc = yaml.safe_load(CASE.read_text(encoding="utf-8"))
    doc["mesh"]["path"] = str(msh)
    doc["output"]["path"] = str(pathlib.Path(td) / "out.h5")
    doc["loads"][0]["vector"] = [0.0, 0.0, -info["deck_interior_load"] * (n - 1)]
    doc["steps"][1]["loads"][0]["vector"] = [0.0, 0.0, -info["params"]["p"] * info["s"] * (n - 1)]
    for a in doc["assign"]:
        a.pop("unstressed_lengths", None)
        if with_lengths and a["region"] == "main_cable":
            a["unstressed_lengths"] = [float(v) for v in info["Lu_cable"]]
        if with_lengths and a["region"] == "hangers":
            a["unstressed_lengths"] = [float(v) for v in info["Lu_hangers"]]
    if dead_only:
        doc["steps"] = doc["steps"][:1]
    yp = pathlib.Path(td) / "bridge.yaml"
    yp.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return str(yp), info


# --------------------------------------------------------------------------- #
#  the reader and the field
# --------------------------------------------------------------------------- #

def test_the_reader_records_which_elements_a_line_group_holds(tmp_path):
    """Every hanger node is a deck node or a cable node, so the node rule
    would give the hangers group the deck's and the cable's interior
    elements too; the recorded groups are exact."""
    path, _ = _small_bridge(tmp_path, n=6)
    mesh = prepare_case(path).mesh
    el = mesh.elements
    assert len(mesh.region_elements("deck", "truss2", el)) == 6
    assert len(mesh.region_elements("main_cable", "truss2", el)) == 6
    assert len(mesh.region_elements("hangers", "truss2", el)) == 5
    # the node rule alone would have handed the hangers the deck's and the
    # cable's interior elements as well: 5 + 4 + 4
    inside = np.zeros(mesh.n_nodes, bool); inside[mesh.regions["hangers"]] = True
    assert int(inside[el].all(axis=1).sum()) == 5 + 4 + 4
    s = _solver(path)                                    # no overlap refusal
    assert sum(1 for p in s.member_props if p["region"] == "hangers") == 5


def test_per_element_unstressed_lengths_are_taken_in_element_order(tmp_path):
    path, info = _small_bridge(tmp_path, n=6, with_lengths=True)
    s = _solver(path)
    ids = formfind._elements_of(s, "main_cable")
    assert np.allclose(s.member_Lu[ids], info["Lu_cable"], rtol=1e-14)
    ids = formfind._elements_of(s, "hangers")
    assert np.allclose(s.member_Lu[ids], info["Lu_hangers"], rtol=1e-14)
    assert s.has_cables


def test_a_wrong_number_of_unstressed_lengths_is_refused(tmp_path):
    path, _ = _small_bridge(tmp_path, n=6, with_lengths=True)
    doc = yaml.safe_load(open(path, encoding="utf-8"))
    for a in doc["assign"]:
        if a["region"] == "hangers":
            a["unstressed_lengths"] = a["unstressed_lengths"][:-1]
    p2 = pathlib.Path(tmp_path) / "short.yaml"
    p2.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unstressed_lengths"):
        _solver(str(p2))
    # and only one of the three statements
    doc = yaml.safe_load(open(path, encoding="utf-8"))
    for a in doc["assign"]:
        if a["region"] == "hangers":
            a["prestress"] = 1.0e8
    p2.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="ONE of"):
        prepare_case(str(p2))


# --------------------------------------------------------------------------- #
#  the search
# --------------------------------------------------------------------------- #

def test_the_closed_form_lengths_leave_the_bridge_exactly_where_it_is(tmp_path):
    """With the closed-form lengths the mesh IS the dead-load state: the
    deck does not move and the cable's horizontal tension is w l^2 / 8f."""
    path, info = _small_bridge(tmp_path, n=10, with_lengths=True, dead_only=True)
    s = _solver(path)
    s.run(output_every=1, progress_every=10 ** 9)
    u = s.dofs.view(s.q)[:, :3]
    assert float(np.max(np.abs(u))) < 1e-9
    assert -s.monitors["H_left"].values[-1] == pytest.approx(info["H"], rel=1e-9)
    ids = formfind._elements_of(s, "hangers")
    assert np.allclose(s.end_actions[ids, 6], info["P"], rtol=1e-9)


def test_formfind_reproduces_the_closed_form_from_the_mesh_lengths(tmp_path):
    """Started from L_u = L0 the bridge sags a metre; a handful of
    relaxations bring it back to the mesh, and the lengths that did so are
    the closed form's."""
    path, info = _small_bridge(tmp_path, n=10, dead_only=True)
    log = []
    found = formfind.find(path, ["main_cable", "hangers"], tol=1e-4, log=log.append)
    assert found.converged and found.iterations <= 8
    assert found.history[0] > 0.1 and found.max_displacement <= 1e-4
    assert np.allclose(found.unstressed_lengths["main_cable"], info["Lu_cable"], rtol=1e-5)
    assert np.allclose(found.unstressed_lengths["hangers"], info["Lu_hangers"], rtol=1e-5)
    assert np.allclose(found.forces["hangers"], info["P"], rtol=1e-4)
    assert len(log) == found.iterations

    # the written case runs and sits still under its dead load
    out = pathlib.Path(tmp_path) / "found.yaml"
    formfind.write_found_case(path, found, str(out))
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    lens = {a["region"]: a["unstressed_lengths"] for a in doc["assign"] if "unstressed_lengths" in a}
    assert set(lens) == {"main_cable", "hangers"} and len(lens["hangers"]) == 9
    s = _solver(str(out))
    s.run(output_every=1, progress_every=10 ** 9)
    assert float(np.max(np.abs(s.dofs.view(s.q)[:, :3]))) <= 1e-4


def test_formfind_refuses_a_beam_group_and_an_unknown_one(tmp_path):
    path, _ = _small_bridge(tmp_path, n=4)
    with pytest.raises(ValueError, match="not a cable"):
        formfind.find(path, ["deck"])
    with pytest.raises(ValueError, match="no member group"):
        formfind.find(path, ["stays"])


def test_write_found_case_needs_an_assign_entry_per_found_group(tmp_path):
    path, _ = _small_bridge(tmp_path, n=4)
    found = formfind.FormFound(unstressed_lengths={"stays": [1.0]}, iterations=1,
                               max_displacement=0.0, converged=True)
    with pytest.raises(ValueError, match="stays"):
        formfind.write_found_case(path, found, str(pathlib.Path(tmp_path) / "x.yaml"))


def test_the_cli_writes_the_found_case_and_says_when_it_did_not_converge(tmp_path, capsys):
    path, _ = _small_bridge(tmp_path, n=4)
    out = pathlib.Path(tmp_path) / "cli_found.yaml"
    rc = main(["formfind", path, "--groups", "main_cable,hangers", "-o", str(out), "--tol", "1e-3"])
    assert rc == 0 and out.exists()
    assert "CONVERGED" in capsys.readouterr().out
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert any("unstressed_lengths" in a for a in doc["assign"])
    # one relaxation cannot reach a micron
    rc = main(["formfind", path, "--groups", "main_cable", "-o", str(out), "--tol", "1e-9",
               "--max-iter", "1"])
    assert rc == 3 and "NOT CONVERGED" in capsys.readouterr().out
