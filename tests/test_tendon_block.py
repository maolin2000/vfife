"""A case's own `tendons:` block: each tendon one entry,
expanded on reading into its nodes, arms and jacked bars -- what the Midas
converter wrote out in full until then.

A 30 m girder of 15 members, simply supported, with a tendon draped as a
parabola, stressed in the case's second step."""
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

from vfife.prep import tendons as prep_tendons    # a submodule: imported, not reached as prep.tendons
from vfife.api import prepare_case, run_case
from vfife.io import gmsh, schema

N_EL, SPAN = 15, 30.0
P = 2.0e6
AREA, COUNT = 1.4e-4, 12


def _msh(path, n=N_EL):
    xs = np.linspace(0.0, SPAN, n + 1)
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "3",
             '1 1 "girder"', '0 2 "left"', '0 3 "right"', "$EndPhysicalNames",
             "$Nodes", str(n + 1)]
    lines += [f"{i + 1} {float(x)!r} 0 0" for i, x in enumerate(xs)]
    rows = [f"1 2 1 1 {i + 1} {i + 2}" for i in range(n)] + ["15 2 2 2 1", f"15 2 3 3 {n + 1}"]
    lines += ["$EndNodes", "$Elements", str(len(rows))] + [f"{i + 1} {r}" for i, r in enumerate(rows)]
    lines += ["$EndElements", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _tendon(**over):
    t = {"name": "T1", "nodes": "girder", "section": "strand", "count": COUNT,
         "line": {"origin": [0.0, 0.0, 0.0], "direction": [1.0, 0.0, 0.0]},
         "profile_z": [[0.0, 0.0, 0.0], [15.0, -0.8, 40.0], [30.0, 0.0, 0.0]],
         "stress": {"step": "stress", "ends": "both", "force": [P, P]},
         "friction": {"mu": 0.2, "k": 0.0015}, "draw_in": {"begin": 0.006, "end": 0.006}}
    t.update(over)
    return t


def _case(tmp_path, tendons, backend="numpy"):
    _msh(tmp_path / "girder.msh")
    return {
        "version": 1, "name": "tb", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": "girder.msh"},
        "materials": {"c50": {"type": "linear_elastic", "E": 3.45e10, "nu": 0.2, "rho": 2600.0},
                      "strand": {"type": "linear_elastic", "E": 1.95e11, "nu": 0.3, "rho": 7850.0}},
        "sections": {"box": {"type": "beam", "material": "c50", "shape": "general",
                             "area": 6.0, "iy": 4.0, "iz": 12.0, "j": 8.0},
                     "strand": {"type": "bar", "material": "strand", "shape": "general", "area": AREA}},
        "assign": [{"region": "girder", "section": "box", "orientation": [0, 0, 1]}],
        "constraints": [{"region": "left", "dofs": [1, 1, 1, 1, 0, 0]},
                        {"region": "right", "dofs": [0, 1, 1, 0, 0, 0]}],
        "loads": [{"region": "all", "gravity": [0, 0, -9.81]}],
        "steps": [{"name": "self weight"}, {"name": "stress"}],
        "tendons": tendons,
        "solver": {"mode": "static", "backend": backend, "tolerance": 1e-7},
        "output": {"path": str(tmp_path / "tb.h5"), "fields": ["displacement", "axial_force"]},
    }


def _write(tmp_path, d, name="case.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def _expanded(tmp_path, d):
    mesh, _ = gmsh.read(str(tmp_path / "girder.msh"))
    return prep_tendons.expand(d, mesh)


def test_a_tendon_is_its_nodes_arms_and_bars_after_the_meshs_own(tmp_path):
    d = _case(tmp_path, [_tendon()])
    p = _write(tmp_path, d)
    a = prepare_case(str(p))
    stops = N_EL + 1                                    # every girder node, the two anchors among them
    assert a.mesh.nodes.shape[0] == (N_EL + 1) + stops
    conn = a.mesh.cell_blocks["truss2"]
    assert conn.shape[0] == N_EL + stops + (stops - 1)
    # the girder's own rows are where they were
    ref, _ = gmsh.read(str(tmp_path / "girder.msh"))
    assert np.array_equal(a.mesh.nodes[:N_EL + 1], ref.nodes)
    assert np.array_equal(conn[:N_EL], ref.cell_blocks["truss2"])
    # each tendon node is on its line: z = -0.8 (1 - (2x/L - 1)^2) away from the corners' rounding
    t = a.mesh.nodes[N_EL + 1:]
    assert t[:, 0] == pytest.approx(np.linspace(0.0, SPAN, N_EL + 1), abs=1e-9)
    assert t[0, 2] == 0.0 and t[-1, 2] == 0.0 and t[N_EL // 2 + 1, 2] < -0.7
    # the bars come on in the step that stresses them, the arms with the girder (from the start)
    case = a.case
    stress = [st for st in case.steps if st.name == "stress"][0]
    assert sorted(stress.activate) == sorted(f"T1/bar{i + 1}" for i in range(stops - 1))


def test_without_friction_or_draw_in_every_bar_is_jacked_to_the_jack(tmp_path):
    d = _case(tmp_path, [_tendon(friction={}, draw_in={}, stress={"step": "stress", "ends": "begin",
                                                                   "stress": [1.3e9, 0.0]})])
    low, _ov, rep = _expanded(tmp_path, d)
    jacks = [a_["jack"] for a_ in low["assign"] if "jack" in a_]
    assert len(jacks) == N_EL and jacks == pytest.approx([1.3e9 * AREA * COUNT] * N_EL, rel=1e-12)


def test_friction_takes_what_the_curve_and_the_length_take(tmp_path):
    mu, k = 0.2, 0.0015
    d = _case(tmp_path, [_tendon(draw_in={}, friction={"mu": mu, "k": k},
                                 stress={"step": "stress", "ends": "begin", "force": [P, 0.0]})])
    low, _ov, rep = _expanded(tmp_path, d)
    jacks = np.array([a_["jack"] for a_ in low["assign"] if "jack" in a_])
    # the last bar's middle, 29 m along: past the rounded corner at midspan, so it has turned through
    # the whole corner, 2 atan(0.8 / 15), and run 29 m (the drape adds 0.1% of length)
    theta = 2.0 * math.atan(0.8 / 15.0)
    s = 29.0
    assert jacks[-1] == pytest.approx(P * math.exp(-(mu * theta + k * s)), rel=3e-3)
    assert np.all(np.diff(jacks) < 0)                                      # it only loses along the way


def test_the_expanded_case_is_the_case_bit_for_bit(tmp_path):
    d = _case(tmp_path, [_tendon(), _tendon(name="T2", profile_z=[[0.0, -0.2, 0.0], [30.0, -0.2, 0.0]],
                                            stress={"step": "stress", "ends": "end", "stress": [0.0, 1.2e9]})])
    p = _write(tmp_path, d)
    out = tmp_path / "flat.yaml"
    from vfife import cli
    assert cli.main(["expand", str(p), "-o", str(out)]) == 0
    flat = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert "tendons" not in flat
    a, b = prepare_case(str(p)), prepare_case(str(out))
    assert gmsh.same_mesh(a.mesh, b.mesh)
    ca, cb = a.case.as_dict(), b.case.as_dict()
    assert ca["mesh"].pop("path") == "girder.msh" and cb["mesh"].pop("path") == "flat.msh"
    assert ca == cb
    sa = _final_state(str(p)); sb = _final_state(str(out))
    assert np.array_equal(sa, sb)


def _final_state(path):
    got = {}
    out = run_case(path, on_ready=lambda s, _p: got.__setitem__("s", s))
    assert out.status == "CONVERGED", out.message
    return np.asarray(got["s"].q).copy()


def test_a_tendon_lifts_the_girder_by_its_kink_force(tmp_path):
    """Draped in two straight runs, the tendon pushes up at the corner with
    2 P sin(a), a = atan(0.8 / 15), P its force there after friction and
    draw-in: F L^3 / (48 E I) more camber than the girder has without it
    (the corner's 4 m rounding and the anchors on the centroid line)."""
    with_ = _final_state(str(_write(tmp_path, _case(tmp_path, [_tendon()]))))
    d = _case(tmp_path, [])
    d.pop("tendons")
    without = _final_state(str(_write(tmp_path, d, "plain.yaml")))
    mid = N_EL // 2
    lift = with_.reshape(-1, 6)[mid, 2] - without.reshape(-1, 6)[mid, 2]
    low, _ov, _rep = _expanded(tmp_path, _case(tmp_path, [_tendon()]))
    p_mid = max(a_["jack"] for a_ in low["assign"] if "jack" in a_ and a_["region"] in ("T1/bar7", "T1/bar8"))
    F = 2.0 * p_mid * math.sin(math.atan(0.8 / 15.0))
    # (local y is global z here, so the girder bends vertically about its local z: iz)
    assert lift == pytest.approx(F * SPAN ** 3 / (48.0 * 3.45e10 * 12.0), rel=0.05)


@pytest.mark.parametrize("bad, says", [
    ({"nodes": "nowhere"}, "not a group of the mesh"),
    ({"section": "box"}, "takes a bar section"),
    ({"stress": {"step": "never", "force": [P, P]}}, "not a step of the case"),
    ({"profile_z": [[0.0, 0.0, 0.0], [0.0, -0.5, 0.0]]}, "do not run forward"),
    ({"stress": {"step": "stress"}}, "one of `stress` or `force`"),
])
def test_what_cannot_be_a_tendon_is_said_by_name(tmp_path, bad, says):
    p = _write(tmp_path, _case(tmp_path, [_tendon(**bad)]))
    from vfife.core.diagnostics import UserError
    with pytest.raises(UserError, match=says):
        schema.load(str(p))


def test_a_case_without_tendons_reads_as_before(tmp_path):
    d = _case(tmp_path, [])
    d.pop("tendons")
    p = _write(tmp_path, d)
    case, _ = schema.load(str(p))
    assert getattr(case, "_mesh_overlay", None) is None


def test_what_is_followed_roughly_is_a_preflight_warning(tmp_path):
    """A tendon on a line off the girder's nodes: its arms are long, and said."""
    p = _write(tmp_path, _case(tmp_path, [_tendon(line={"origin": [0.0, 2.0, 0.0], "direction": [1.0, 0.0, 0.0]})]))
    a = prepare_case(str(p))
    codes = [i["code"] for i in a.preflight["issues"]]
    assert "TENDON_APPROXIMATE" in codes


def test_a_fibre_models_tendons_are_its_sections(tmp_path):
    d = _case(tmp_path, [_tendon()])
    d["sections"]["fib"] = {"type": "fiber", "material": "c50", "shape": "rect", "width": 1.0, "height": 1.0}
    from vfife.core.diagnostics import UserError
    with pytest.raises(UserError, match="fibre beam's tendons"):
        schema.load(str(_write(tmp_path, d)))


def _staged_msh(path):
    """The girder in three groups: two cantilevers and the closure between."""
    xs = np.linspace(0.0, SPAN, N_EL + 1)
    groups = {"L": range(0, 5), "C": range(5, 10), "R": range(10, 15)}
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "6",
             '1 1 "L"', '1 2 "C"', '1 3 "R"', '0 4 "left"', '0 5 "right"', '0 6 "girder"', "$EndPhysicalNames",
             "$Nodes", str(N_EL + 1)]
    lines += [f"{i + 1} {float(x)!r} 0 0" for i, x in enumerate(xs)]
    rows = [f"1 2 {t} {t} {i + 1} {i + 2}" for t, g in enumerate(groups.values(), 1) for i in g]
    rows += ["15 2 4 4 1", f"15 2 5 5 {N_EL + 1}"] + [f"15 2 6 6 {i + 1}" for i in range(N_EL + 1)]
    lines += ["$EndNodes", "$Elements", str(len(rows))] + [f"{i + 1} {r}" for i, r in enumerate(rows)]
    lines += ["$EndElements", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _staged(tmp_path, tendon):
    d = _case(tmp_path, [tendon])
    _staged_msh(tmp_path / "girder.msh")
    d["assign"] = [{"region": g, "section": "box", "orientation": [0, 0, 1]} for g in ("L", "C", "R")]
    d["steps"] = [{"name": "cantilevers", "activate": ["L", "R"]}, {"name": "closure", "activate": ["C"]}]
    return d


def test_unstated_a_tendon_is_stressed_when_the_last_member_it_runs_along_is_built(tmp_path):
    """Over a closure whose ends are the cantilevers' tips: not in the first
    step, where its end nodes already stand (found in review)."""
    t = _tendon(stress={"ends": "both", "force": [P, P]})
    d = _staged(tmp_path, t)
    d["steps"] = [{"name": "nothing"}] + d["steps"]
    low, _ov, rep = _expanded(tmp_path, d)
    assert rep["T1"]["step"] == 2
    bars = {f"T1/bar{i + 1}" for i in range(N_EL)}
    assert bars <= set(low["steps"][2]["activate"])
    # an arm is born with the girder at it: the cantilevers' in step 1, the closure's middle in step 2
    assert "T1/arm1" in low["steps"][1]["activate"] and "T1/arm8" in low["steps"][2]["activate"]


def test_a_step_that_builds_a_name_the_mesh_does_not_know_is_refused(tmp_path):
    d = _staged(tmp_path, _tendon(stress={"step": "closure", "ends": "both", "force": [P, P]}))
    d["steps"][1]["activate"] = ["somewhere"]
    from vfife.core.diagnostics import UserError
    with pytest.raises(UserError, match="not a group of the mesh"):
        schema.load(str(_write(tmp_path, d)))


def test_a_step_name_two_steps_have_is_refused(tmp_path):
    d = _case(tmp_path, [_tendon()])
    d["steps"] = [{"name": "stress"}, {"name": "stress"}]
    from vfife.core.diagnostics import UserError
    with pytest.raises(UserError, match="names 2 steps"):
        schema.load(str(_write(tmp_path, d)))


def test_its_relaxation_is_each_bars_own_curve(tmp_path):
    from vfife.codes import jtg3362
    rel = {"code": "jtg3362", "fpk": 1.86e9, "zeta": 0.3}
    d = _case(tmp_path, [_tendon(friction={}, draw_in={}, relaxation=rel,
                                 stress={"step": "stress", "ends": "begin", "stress": [0.72 * 1.86e9, 0.0]})])
    low, _ov, _rep = _expanded(tmp_path, d)
    curves = {k: v for k, v in low["materials"].items() if v.get("relaxation")}
    assert len(curves) == 1
    (m,) = curves.values()
    assert m["relaxation"]["curve"][-1][1] == pytest.approx(jtg3362.relaxation_final(0.72 * 1.86e9, 1.86e9, 0.3), abs=1e-4)
    assert m["rho"] == 1.0e-6


def test_in_millimetres_it_is_the_same_tendon(tmp_path):
    """The case's units: a mesh in mm read with `scale` and a case in N, mm,
    MPa give the girder in mm; the tendon's lengths, its weightless density
    and its jacking force are the same tendon (found in review)."""
    p_si = _write(tmp_path, _case(tmp_path, [_tendon()]))
    si, _o, rep_si = _expanded(tmp_path, _case(tmp_path, [_tendon()]))
    t = _tendon(line={"origin": [0.0, 0.0, 0.0], "direction": [1.0, 0.0, 0.0]},
                profile_z=[[0.0, 0.0, 0.0], [15000.0, -800.0, 40000.0], [30000.0, 0.0, 0.0]],
                draw_in={"begin": 6.0, "end": 6.0}, friction={"mu": 0.2, "k": 1.5e-6})
    d = _case(tmp_path, [t])
    d["units"] = {"length": "mm", "force": "N", "stress": "MPa"}
    d["mesh"]["scale"] = 1000.0
    d["materials"]["strand"]["E"] = 1.95e5
    d["materials"]["strand"]["rho"] = 7.85e-9
    d["sections"]["strand"]["area"] = AREA * 1e6
    mesh, _ = gmsh.read(str(tmp_path / "girder.msh"), scale=1000.0)
    mm, _o, rep_mm = prep_tendons.expand(d, mesh)
    assert rep_mm["T1"]["jacking"] == pytest.approx(rep_si["T1"]["jacking"], rel=1e-12)
    jsi = [a_["jack"] for a_ in si["assign"] if "jack" in a_]
    jmm = [a_["jack"] for a_ in mm["assign"] if "jack" in a_]
    assert jmm == pytest.approx(jsi, rel=1e-6)          # (the draw-in solve is smoothed over 1e-7 of the force)
    assert mm["materials"]["strand_tendon"]["rho"] == pytest.approx(1.0e-6 * 1e-12, rel=1e-12)   # kg/m^3 -> t/mm^3


def test_a_mesh_written_in_the_case_is_refused(tmp_path):
    d = _case(tmp_path, [_tendon()])
    d["mesh"] = {"type": "lines", "points": {"A": [0, 0, 0], "B": [30, 0, 0]},
                 "lines": {"girder": {"through": ["A", "B"], "size": 2.0}}}
    from vfife.core.diagnostics import UserError
    with pytest.raises(UserError, match="Gmsh mesh"):
        schema.load(str(_write(tmp_path, d)))


def test_expand_says_when_there_is_nothing_to_expand(tmp_path, capsys):
    d = _case(tmp_path, [])
    d.pop("tendons")
    from vfife import cli
    assert cli.main(["expand", str(_write(tmp_path, d))]) != 0
    assert "NOTHING_TO_EXPAND" in capsys.readouterr().err


def test_the_result_says_what_each_tendon_carries_step_by_step(tmp_path):
    """The result keeps each tendon's bars: `vfife tendons`
    reads their force at the end of each step, against what they were
    jacked to -- nothing before the step that stresses them, and in it the
    force each bar was jacked to (the jack holds it to the step's end)."""
    from vfife.post import tendons as post_tendons
    t1, t2 = _tendon(), _tendon(name="T2", stress={"step": "stress", "ends": "begin", "force": [P, P]})
    d = _case(tmp_path, [t1, t2])
    d["steps"] = [{"name": "self weight"}, {"name": "stress"}, {"name": "after"}]
    out = run_case(str(_write(tmp_path, d)))
    assert out.status == "CONVERGED", out.message
    got = post_tendons.read(out.result_path)
    assert [t["name"] for t in got] == ["T1", "T2"] and got[0]["stress_step"] == 1
    assert len(got[0]["elements"]) == len(got[0]["jack"]) == len(got[0]["regions"]) == N_EL
    rows = post_tendons.table(out.result_path)
    by = {(r["tendon"], r["step"]): r for r in rows}
    assert by[("T1", "self weight")]["mean"] is None                 # not yet stressed
    stressed = by[("T1", "stress")]
    assert stressed["least"] <= stressed["mean"] <= stressed["largest"]
    assert stressed["of_jack"] == pytest.approx(1.0, rel=1e-6)       # (forces stored in float32)
    assert by[("T1", "after")]["mean"] == pytest.approx(stressed["mean"], rel=1e-9)
    # the bars, one row each, the force where it is read from the result's own `axial_force`
    import h5py
    rows_b = post_tendons.bars(out.result_path)
    b = next(r for r in rows_b if r["tendon"] == "T2" and r["step"] == "stress" and r["bar"] == 3)
    with h5py.File(out.result_path, "r") as f:
        last = [k for k in sorted(f["steps"]) if int(f["steps"][k].attrs.get("analysis_step", 0)) == 1][-1]
        assert b["force"] == pytest.approx(float(f["steps"][last]["axial_force"][got[1]["elements"][2]]))
    # and the command
    from vfife import cli
    assert cli.main(["tendons", out.result_path, "--csv", str(tmp_path / "t.csv")]) in (0, None)
    assert (tmp_path / "t.csv").read_text(encoding="utf-8").splitlines()[0].startswith("tendon,step,time,least")
