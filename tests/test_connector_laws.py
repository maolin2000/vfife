"""Path-dependent connector laws: gap, bilinear, Coulomb friction.

Each test is a closed form: the sliding force of a Coulomb bearing and the
force it comes back with after an excursion, the parallelogram of a bilinear
loop, the free travel of a gap, and the stopping distance of a block pushed
across a frictional bearing.
"""
import h5py
import numpy as np
import pytest
import yaml

from vfife.api import prepare_case, run_case, _with_element
from vfife.core.structural import make_structural_solver
from vfife.elements import connector
from vfife.io.writer import read_checkpoint

G = 9.80665


def _write_msh(path, nodes, groups):
    names = list(groups)
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names))]
    out += [f'{groups[n][0]} {i + 1} "{n}"' for i, n in enumerate(names)]
    out += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    out += [f"{i + 1} {x:.12g} {y:.12g} {z:.12g}" for i, (x, y, z) in enumerate(nodes)]
    els = []
    for i, n in enumerate(names):
        dim, items = groups[n]
        for it in items:
            els.append((15, i + 1, [it + 1]) if dim == 0 else (1, i + 1, [it[0] + 1, it[1] + 1]))
    out += ["$EndNodes", "$Elements", str(len(els))]
    out += [f"{k + 1} {t} 2 {g} {g} " + " ".join(map(str, c)) for k, (t, g, c) in enumerate(els)]
    out += ["$EndElements", ""]
    path.write_text("\n".join(out), encoding="utf-8")


def _bearing_case(tmp_path, laws, stiffness, extra=None):
    """One connector from a fixed anchor at the origin to a block 1 m above it:
    the chord is vertical, so local ux is global z (vertical); with
    `orientation: [1, 0, 0]` local uy is global x, the direction dragged."""
    _write_msh(tmp_path / "b.msh", [(0, 0, 0), (0, 0, 1)],
               {"anchor": (0, [0]), "block": (0, [1]), "bearing": (1, [(0, 1)])})
    d = {
        "units": {"length": "m", "force": "N"},
        "mesh": {"type": "gmsh", "path": "b.msh"},
        "materials": {"steel": {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}},
        "sections": {"bearing": {"type": "connector", "material": "steel", "stiffness": stiffness, "laws": laws}},
        "assign": [{"region": "all", "section": "bearing", "orientation": [1, 0, 0]}],
        "constraints": [{"region": "anchor", "dofs": [1, 1, 1, 1, 1, 1]}],
        "output": {"fields": ["displacement", "connector_force_y", "connector_slip_uy"], "every": 1},
    }
    if extra:
        d.update(extra)
    return d


def _run(tmp_path, name, d):
    d = dict(d, name=name)
    d["output"] = dict(d["output"], path=str(tmp_path / f"{name}.h5"))
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return run_case(str(p)), p


W, MU, K_STICK, K_V = 1.0e5, 0.2, 1.0e8, 1.0e9


def _friction_static(tmp_path, amp, load_steps):
    d = _bearing_case(tmp_path,
                      {"uy": {"type": "friction", "mu": MU, "normal": "ux", "stiffness": K_STICK}},
                      [K_V, 0, 0, 0, 0, 0])
    # the block is held in all but the horizontal y; W presses it down, and a
    # prescribed horizontal displacement drags it
    d["constraints"].append({"region": "block", "dofs": [0, 1, 0, 1, 1, 1]})
    d["loads"] = [{"region": "block", "vector": [0, 0, -W], "amplitude": "hold"}]
    d["amplitudes"] = {"drag": amp}
    d["steps"] = [{"constraints": [{"region": "block", "dofs": [1, 0, 0, 0, 0, 0],
                                    "value": [1.0, 0.0, 0.0], "amplitude": "drag"}],
                   "load_steps": load_steps, "max_steps_per_load": 20000, "tolerance": 1e-8}]
    d["solver"] = {"backend": "numpy", "mode": "relaxation", "damping": {"type": "kinetic"},
                   "tolerance": 1e-8}
    d["output"]["monitors"] = {"u": {"kind": "displacement", "region": "block", "dof": "x"},
                               "H": {"kind": "reaction", "region": "anchor", "dof": "x"},
                               "V": {"kind": "reaction", "region": "anchor", "dof": "z"}}
    return d


def test_a_coulomb_bearing_sticks_then_slides_at_mu_n_and_comes_back_at_minus_mu_n(tmp_path):
    u_y = MU * W / K_STICK                      # where it starts to slide
    # drag to 3 u_y, back to 0, on to -3 u_y
    amp = [[0.0, 0.0], [1 / 3, 3 * u_y], [2 / 3, 0.0], [1.0, -3 * u_y]]
    out, _ = _run(tmp_path, "coulomb", _friction_static(tmp_path, amp, 12))
    u = np.asarray(out.monitors["u"]); H = np.asarray(out.monitors["H"]); V = np.asarray(out.monitors["V"])
    assert V[-1] == pytest.approx(W, rel=1e-6)                       # the anchor carries W
    # stage 2 (u = 1.5 u_y): sliding, |H| = mu W; stage 4 (u = 0): elastic return past
    # the origin re-yields the other way: H = -mu W exactly (see the note)
    at = lambda target: H[np.argmin(np.abs(u - target))]
    assert at(3 * u_y) == pytest.approx(-MU * W, rel=1e-6)          # anchor reaction opposes the drag
    assert at(0.0) == pytest.approx(+MU * W, rel=1e-6)
    assert at(-3 * u_y) == pytest.approx(+MU * W, rel=1e-6)
    with h5py.File(tmp_path / "coulomb.h5") as f:
        last = sorted(f["steps"].keys(), key=int)[-1]
        slip = float(np.asarray(f["steps"][last]["connector_slip_uy"])[0])
        assert "connector_slip" in read_checkpoint(str(tmp_path / "coulomb.h5"))
    assert slip == pytest.approx(-3 * u_y + u_y, rel=1e-6)          # -2 u_y of slip remains


def test_the_sliding_force_follows_the_normal_force(tmp_path):
    u_y = MU * W / K_STICK
    amp = [[0.0, 0.0], [1.0, 4 * u_y]]
    d = _friction_static(tmp_path, amp, 4)
    d["loads"][0]["vector"] = [0, 0, -2 * W]               # twice the weight
    out, _ = _run(tmp_path, "heavier", d)
    assert np.asarray(out.monitors["H"])[-1] == pytest.approx(-2 * MU * W, rel=1e-6)


def test_a_bilinear_isolator_traces_the_parallelogram_of_kinematic_hardening(tmp_path):
    k1, k2, fy = 2.0e7, 2.0e6, 5.0e4
    u_y = fy / k1; u_max = 5 * u_y
    d = _bearing_case(tmp_path, {"uy": {"type": "bilinear", "k1": k1, "k2": k2, "fy": fy}},
                      [K_V, 0, 0, 0, 0, 0])
    d["constraints"].append({"region": "block", "dofs": [0, 1, 1, 1, 1, 1]})
    d["amplitudes"] = {"cycle": [[0, 0], [0.2, u_max], [0.4, 0], [0.6, -u_max], [0.8, 0], [1.0, u_max]]}
    d["steps"] = [{"constraints": [{"region": "block", "dofs": [1, 0, 0, 0, 0, 0],
                                    "value": [1.0, 0.0, 0.0], "amplitude": "cycle"}],
                   "load_steps": 50, "max_steps_per_load": 20000, "tolerance": 1e-8}]
    d["solver"] = {"backend": "numpy", "mode": "relaxation", "damping": {"type": "kinetic"}, "tolerance": 1e-8}
    d["output"]["monitors"] = {"u": {"kind": "displacement", "region": "block", "dof": "x"},
                               "H": {"kind": "reaction", "region": "anchor", "dof": "x"}}
    out, _ = _run(tmp_path, "bilinear", d)
    u = np.asarray(out.monitors["u"]); F = -np.asarray(out.monitors["H"])   # force ON the block's spring
    i_max = int(np.argmax(u[:len(u) // 2])); i_min = int(np.argmin(u))
    F_max = fy + k2 * (u_max - u_y)
    assert F[i_max] == pytest.approx(F_max, rel=1e-6)
    assert F[i_min] == pytest.approx(-F_max, rel=1e-6)
    # unloading from u_max by 2 u_y is elastic (k1) and drops the force by 2 fy
    j = i_max + np.argmin(np.abs(u[i_max:] - (u_max - 2 * u_y)))    # on the way back
    assert F[j] == pytest.approx(F_max - 2 * fy, rel=1e-6)
    # one closed loop, u_max -> -u_max -> u_max: the parallelogram's area,
    # 4 (fy - k2 u_y) (u_max - u_y), is the energy it dissipates
    seg = slice(i_max, len(u))
    work = np.trapezoid(F[seg], u[seg])
    assert abs(work) == pytest.approx(4 * (fy - k2 * u_y) * (u_max - u_y), rel=0.01)


def test_a_gap_is_free_inside_and_a_stop_outside(tmp_path):
    k = 5.0e7; lo, hi = -0.02, 0.05
    d = _bearing_case(tmp_path, {"uy": {"type": "gap", "lo": lo, "hi": hi, "stiffness": k}},
                      [K_V, 0, 0, 0, 0, 0])
    d["constraints"].append({"region": "block", "dofs": [0, 1, 1, 1, 1, 1]})
    d["amplitudes"] = {"sweep": [[0, 0], [0.5, 0.08], [1.0, -0.05]]}
    d["steps"] = [{"constraints": [{"region": "block", "dofs": [1, 0, 0, 0, 0, 0],
                                    "value": [1.0, 0.0, 0.0], "amplitude": "sweep"}],
                   "load_steps": 26, "max_steps_per_load": 20000, "tolerance": 1e-8}]
    d["solver"] = {"backend": "numpy", "mode": "relaxation", "damping": {"type": "kinetic"}, "tolerance": 1e-8}
    d["output"]["monitors"] = {"u": {"kind": "displacement", "region": "block", "dof": "x"},
                               "H": {"kind": "reaction", "region": "anchor", "dof": "x"}}
    out, _ = _run(tmp_path, "gap", d)
    u = np.asarray(out.monitors["u"]); F = -np.asarray(out.monitors["H"])
    inside = (u > lo + 1e-6) & (u < hi - 1e-6)
    assert np.all(np.abs(F[inside]) < 1e-6 * k * hi)
    j = np.argmin(np.abs(u - 0.08)); assert F[j] == pytest.approx(k * (0.08 - hi), rel=1e-6)
    j = np.argmin(np.abs(u + 0.05)); assert F[j] == pytest.approx(k * (-0.05 - lo), rel=1e-6)


def test_a_pushed_block_slides_the_closed_form_distance_and_stops(tmp_path):
    """Dynamic, constant normal force: pushed by H > mu N for t1, then let go.
    a1 = (H - mu N) / m while pushed, a2 = -mu N / m after: it stops at
    s = a1 t1^2 / 2 + (a1 t1)^2 / (2 |a2|)."""
    m, N, H, t1, total = 1000.0, 1000.0 * G, 4000.0, 0.5, 2.0
    d = _bearing_case(tmp_path, {"uy": {"type": "friction", "mu": MU, "normal": N, "stiffness": 1.0e7}},
                      [K_V, 0, 0, 0, 0, 0])
    d["constraints"].append({"region": "block", "dofs": [0, 1, 1, 1, 1, 1]})
    d["nodal_masses"] = [{"region": "block", "mass": m}]
    d["amplitudes"] = {"push": [[0, 1.0], [t1 / total, 1.0], [t1 / total + 1e-6, 0.0], [1.0, 0.0]]}
    d["loads"] = [{"region": "block", "vector": [H, 0, 0], "amplitude": "push"}]
    d["solver"] = {"backend": "numpy", "mode": "dynamic", "total_time": total, "damping": {"type": "none"}}
    d["output"]["monitors"] = {"u": {"kind": "displacement", "region": "block", "dof": "x"}}
    out, _ = _run(tmp_path, "slide", d)
    with h5py.File(tmp_path / "slide.h5") as f:
        t = np.asarray(f["monitors/time"]); u = np.asarray(f["monitors/u"])
    a1 = (H - MU * N) / m; a2 = MU * N / m
    v1 = a1 * t1; stop = t1 + v1 / a2
    s = 0.5 * a1 * t1 ** 2 + v1 ** 2 / (2 * a2)
    # the two constant accelerations, read off the parabolas the block traces
    w = (t > 0.1) & (t < t1 - 0.05)
    assert 2 * np.polyfit(t[w], u[w], 2)[0] == pytest.approx(a1, rel=5e-3)
    w = (t > t1 + 0.05) & (t < stop - 0.05)
    assert 2 * np.polyfit(t[w], u[w], 2)[0] == pytest.approx(-a2, rel=5e-3)
    # it stops where the closed form says (to the pulse's own start-up
    # transient) and stays stopped: no creep under zero force
    assert u[-1] == pytest.approx(s, rel=0.03)
    late = t > stop + 0.2
    assert np.ptp(u[late]) < 1e-3 * s


def test_the_laws_are_idempotent_in_the_deformation_and_refused_where_they_would_be_ignored(tmp_path):
    # the same deformation evaluated twice from the same committed slip is the
    # same answer: that is what lets the static stages treat each evaluation as
    # a trial from the last equilibrium
    law = {"type": "bilinear", "k1": 100.0, "k2": 10.0, "fy": 50.0}
    f1, s1 = connector.law_force(law, 1.0, 0.0, 0.0)
    f2, s2 = connector.law_force(law, 1.0, s1, 0.0)
    assert (f1, s1) == pytest.approx((55.0, 0.45)) and (f2, s2) == pytest.approx((f1, s1))
    # refusals: a law on a beam section; a law and a stiffness on one component;
    # friction pressing on a component that has no spring
    for bad, msg in ((dict(type="beam", material="steel", shape="rect", b=0.1, h=0.1,
                           laws={"uy": {"type": "gap", "hi": 0.01, "stiffness": 1e6}}), "connector section"),
                     (dict(type="connector", material="steel", stiffness=[1e9, 1e9, 0, 0, 0, 0],
                           laws={"uy": {"type": "gap", "hi": 0.01, "stiffness": 1e6}}), "one of them"),
                     (dict(type="connector", material="steel", stiffness=[0, 0, 0, 0, 0, 0],
                           laws={"uy": {"type": "friction", "mu": 0.1, "normal": "ux", "stiffness": 1e6}}), "linear stiffness")):
        d = _bearing_case(tmp_path, {}, [1e9, 0, 0, 0, 0, 0])
        d["sections"]["bearing"] = bad
        d["solver"] = {"backend": "numpy", "mode": "relaxation", "n_steps": 10}
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(dict(d, name="bad"), sort_keys=False), encoding="utf-8")
        with pytest.raises(Exception, match=msg):
            prepare_case(str(p))


def test_the_slip_rides_in_the_checkpoint(tmp_path):
    u_y = MU * W / K_STICK
    out, p = _run(tmp_path, "ck", _friction_static(tmp_path, [[0.0, 0.0], [1.0, 3 * u_y]], 3))
    prep = prepare_case(str(p))
    s = make_structural_solver(_with_element(prep.case, prep.element, prep.case_path), prep.mesh, prep.material)
    s.restore(read_checkpoint(str(tmp_path / "ck.h5")))
    assert s._conn_committed[0, 1] == pytest.approx(2 * u_y, rel=1e-6)


def test_a_bearing_law_under_a_fibre_column_slides_at_mu_n_through_the_trial_discipline(tmp_path):
    """The fibre solver re-drives every evaluation of a static stage from the
    last equilibrium; the connector slip follows the same rule. A stiff fibre
    column standing on a Coulomb bearing, pressed by W and dragged at its top:
    the anchor's horizontal reaction saturates at mu W."""
    _write_msh(tmp_path / "c.msh", [(0, 0, 0), (0, 0, 1), (0, 0, 3)],
               {"anchor": (0, [0]), "foot": (0, [1]), "top": (0, [2]),
                "bearing": (1, [(0, 1)]), "column": (1, [(1, 2)])})
    d = {
        "units": {"length": "m", "force": "N"},
        "mesh": {"type": "gmsh", "path": "c.msh"},
        "materials": {"steel": {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}},
        "sections": {"col": {"type": "fiber", "material": "steel", "shape": "rect",
                             "width": 0.3, "height": 0.3, "ny": 4, "nz": 4},
                     "bearing": {"type": "connector", "material": "steel", "stiffness": [K_V, 0, 0, 0, 0, 0],
                                 "laws": {"uy": {"type": "friction", "mu": MU, "normal": "ux", "stiffness": K_STICK}}}},
        "assign": [{"region": "column", "section": "col", "orientation": [1, 0, 0]},
                   {"region": "bearing", "section": "bearing", "orientation": [1, 0, 0]}],
        "constraints": [{"region": "anchor", "dofs": [1, 1, 1, 1, 1, 1]},
                        {"region": "foot", "dofs": [0, 1, 0, 1, 1, 1]},
                        {"region": "top", "dofs": [0, 1, 0, 1, 1, 1]}],
        "loads": [{"region": "top", "vector": [0, 0, -W], "amplitude": "hold"}],
        "amplitudes": {"drag": [[0.0, 0.0], [1.0, 4 * MU * W / K_STICK]]},
        "steps": [{"constraints": [{"region": "top", "dofs": [1, 0, 0, 0, 0, 0],
                                    "value": [1.0, 0.0, 0.0], "amplitude": "drag"}],
                   "load_steps": 4, "max_steps_per_load": 40000, "tolerance": 1e-7}],
        "solver": {"backend": "numpy", "mode": "relaxation", "damping": {"type": "kinetic"}, "tolerance": 1e-7},
        "output": {"fields": ["displacement", "connector_slip_uy"], "every": 1,
                   "monitors": {"H": {"kind": "reaction", "region": "anchor", "dof": "x"},
                                "V": {"kind": "reaction", "region": "anchor", "dof": "z"}}},
    }
    out, _ = _run(tmp_path, "fibcol", d)
    H = np.asarray(out.monitors["H"]); V = np.asarray(out.monitors["V"])
    assert V[-1] == pytest.approx(W, rel=1e-5)
    assert H[-1] == pytest.approx(-MU * W, rel=1e-4)
    assert abs(H[0]) < MU * W * 0.999                 # the first quarter is still sticking
