"""Rigid surfaces: a node that meets the ground stops there.

Closed forms: a propped cantilever's prop reaction, a dropped rod's bounce
back to its drop height, a block sliding out at mu g, a solid box's weight
on the floor; and the refusals.
"""
import h5py
import numpy as np
import pytest
import yaml

from vfife.api import prepare_case, run_case

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


STEEL = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}


def _run(tmp_path, name, d):
    d = dict(d, name=name)
    d["output"] = dict(d.get("output", {}), path=str(tmp_path / f"{name}.h5"))
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    out = run_case(str(p))
    assert out.result_path is not None, out
    return out, p


def _beam_on_floor(tmp_path, gap=0.0, region="all", stiffness="auto"):
    """A 4 m beam clamped at x = 0 whose free end rests on the floor z = -gap."""
    n = 8
    xs = np.linspace(0, 4.0, n + 1)
    _write_msh(tmp_path / "b.msh", [(x, 0, 0) for x in xs],
               {"root": (0, [0]), "tip": (0, [n]), "beam": (1, [(k, k + 1) for k in range(n)])})
    return {
        "units": {"length": "m", "force": "N"},
        "mesh": {"type": "gmsh", "path": "b.msh"},
        "materials": {"steel": STEEL},
        "sections": {"b": {"type": "beam", "material": "steel", "shape": "rect", "b": 0.1, "h": 0.1}},
        "assign": [{"region": "all", "section": "b", "orientation": [0, 0, 1]}],
        "constraints": [{"region": "root", "dofs": [1, 1, 1, 1, 1, 1]},
                        {"region": "all", "dofs": [0, 1, 0, 1, 0, 1]}],
        "loads": [{"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"}],
        "contact": {"rigid": [{"plane": {"point": [0, 0, -gap], "normal": [0, 0, 1]},
                               "region": region, "stiffness": stiffness, "name": "floor"}]},
        "solver": {"backend": "numpy", "mode": "relaxation", "damping": {"type": "kinetic"},
                   "tolerance": 1e-8, "n_steps": 80000, "load_steps": 2, "max_steps_per_load": 40000},
        "output": {"fields": ["displacement", "contact_force", "contact_penetration"], "every": 1,
                   "monitors": {"w_tip": {"kind": "displacement", "region": "tip", "dof": "z"},
                                "R_root": {"kind": "reaction", "region": "root", "dof": "z"}}},
    }


def test_a_cantilever_whose_tip_rests_on_the_floor_is_a_propped_cantilever(tmp_path):
    """Prop reaction of a uniformly loaded propped cantilever: 3 w L / 8; the
    root then carries 5 w L / 8. The tip sinks by the prop's own penetration
    only (microns with `auto`)."""
    d = _beam_on_floor(tmp_path, region="tip")      # the beam itself lies ON z = 0: only the tip is checked
    out, _ = _run(tmp_path, "prop", d)
    w = 7850.0 * 0.01 * G; L = 4.0
    # gravity is lumped at the nodes (wL/8 at each interior node, wL/16 at
    # the tip): the prop reaction of those point loads, P a^2 (3L - a) / (2 L^3)
    # each, is 1160.7 N -- 0.5% above the uniform load's 3 w L / 8
    a = np.linspace(0, L, 9)[1:]
    P = np.full(8, w * L / 8); P[-1] /= 2
    R_prop = float(np.sum(P * a ** 2 * (3 * L - a) / (2 * L ** 3)))
    with h5py.File(tmp_path / "prop.h5") as f:
        last = sorted(f["steps"].keys(), key=int)[-1]
        fc = np.asarray(f["steps"][last]["contact_force"])
        pen = np.asarray(f["steps"][last]["contact_penetration"])
    assert fc[-1, 2] == pytest.approx(R_prop, rel=1e-3)             # the prop
    assert np.all(fc[:-1] == 0.0)                                    # only the tip touches
    assert np.asarray(out.monitors["R_root"])[-1] == pytest.approx(w * L - R_prop, rel=1e-3)
    assert 0.0 < pen[-1] < 1e-4                                      # microns
    assert abs(np.asarray(out.monitors["w_tip"])[-1]) < 1e-4


def test_a_floor_below_the_free_tip_is_not_touched_and_the_cantilever_hangs_free(tmp_path):
    d = _beam_on_floor(tmp_path, gap=1.0)                             # the tip sags 5.3 mm; 1 m clear
    out, _ = _run(tmp_path, "free", d)
    w = 7850.0 * 0.01 * G; L = 4.0
    with h5py.File(tmp_path / "free.h5") as f:
        last = sorted(f["steps"].keys(), key=int)[-1]
        assert np.all(np.asarray(f["steps"][last]["contact_force"]) == 0.0)
    assert np.asarray(out.monitors["R_root"])[-1] == pytest.approx(w * L, rel=1e-4)


def _rod(tmp_path, z0, zend, extra):
    """A 1 m steel rod, both nodes free in z (x, y held): a particle pair."""
    _write_msh(tmp_path / "r.msh", [(0, 0, z0), (0, 0, zend)],
               {"low": (0, [0]), "high": (0, [1]), "rod": (1, [(0, 1)])})
    d = {
        "units": {"length": "m", "force": "N"},
        "mesh": {"type": "gmsh", "path": "r.msh"},
        "materials": {"steel": STEEL},
        "sections": {"r": {"type": "bar", "material": "steel", "shape": "circle", "d": 0.05}},
        "assign": [{"region": "all", "section": "r"}],
        "loads": [{"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"}],
        "output": {"fields": ["displacement", "contact_force"], "every": 5,
                   "monitors": {"z_low": {"kind": "displacement", "region": "low", "dof": "z"},
                                "x_low": {"kind": "displacement", "region": "low", "dof": "x"}}},
    }
    d.update(extra)
    return d


def test_a_dropped_rod_bounces_back_to_its_drop_height(tmp_path):
    """Vertical rod dropped from h = 0.5 m onto an elastic floor with no
    damping: the apex after the bounce is the drop height; the fall takes
    sqrt(2 h / g) = 0.319 s. The floor is given a stiffness far below the
    rod's own (1e5 N/m against EA/L = 4e8), so the rod bounces as a rigid
    body; with `auto` the contact spring is as stiff as the rod and the
    impact leaves a tenth of the energy ringing in its axial mode."""
    h = 0.5
    d = _rod(tmp_path, h, h + 1.0, {
        "constraints": [{"region": "all", "dofs": [1, 1, 0]}],
        "contact": {"rigid": [{"plane": {"point": [0, 0, 0], "normal": [0, 0, 1]}, "stiffness": 1.0e5}]},
        "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 1.0, "damping": {"type": "none"}},
    })
    out, _ = _run(tmp_path, "drop", d)
    with h5py.File(tmp_path / "drop.h5") as f:
        t = np.asarray(f["monitors/time"]); z = np.asarray(f["monitors/z_low"])
    t_fall = np.sqrt(2 * h / G)
    m = 7850.0 * np.pi * 0.025 ** 2 * 1.0                            # the whole (rigid) rod presses the spring
    assert z.min() > -h - np.sqrt(2 * G * h) * np.sqrt(m / 1.0e5) * 1.1   # penetration v sqrt(m/k), 39 mm
    first_hit = t[np.argmax(z < -h)]
    assert first_hit == pytest.approx(t_fall, rel=0.02)
    after = t > first_hit + 0.05
    assert z[after].max() == pytest.approx(0.0, abs=0.01 * h)         # back up to the drop height


def test_a_pushed_block_decelerates_at_mu_g_on_a_rough_floor(tmp_path):
    """The rod lying on the floor (both nodes on z = 0), pushed along x by a
    force pulse, then free: it decelerates at mu g and stops."""
    mu, H, t1, total = 0.3, 100.0, 0.2, 1.0
    m = 7850.0 * np.pi * 0.025 ** 2 * 1.0                             # 15.4 kg
    d = _rod(tmp_path, 0.0, 0.0, {})
    _write_msh(tmp_path / "r.msh", [(0, 0, 0), (1, 0, 0)],
               {"low": (0, [0]), "high": (0, [1]), "rod": (1, [(0, 1)])})
    d.update({
        "constraints": [{"region": "all", "dofs": [0, 1, 0]}],
        "amplitudes": {"push": [[0, 1.0], [t1 / total, 1.0], [t1 / total + 1e-6, 0.0], [1.0, 0.0]]},
        "loads": [{"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"},
                  {"region": "all", "vector": [H, 0, 0], "amplitude": "push"}],
        "contact": {"rigid": [{"plane": {"point": [0, 0, 0], "normal": [0, 0, 1]}, "friction": mu}]},
        "solver": {"backend": "numpy", "mode": "dynamic", "total_time": total, "damping": {"type": "none"}},
    })
    out, _ = _run(tmp_path, "slide", d)
    with h5py.File(tmp_path / "slide.h5") as f:
        t = np.asarray(f["monitors/time"]); x = np.asarray(f["monitors/x_low"])
    a1 = H / m - mu * G; a2 = mu * G
    v1 = a1 * t1; stop = t1 + v1 / a2
    w = (t > 0.03) & (t < t1 - 0.02)
    assert 2 * np.polyfit(t[w], x[w], 2)[0] == pytest.approx(a1, rel=0.02)
    w = (t > t1 + 0.03) & (t < stop - 0.03)
    assert 2 * np.polyfit(t[w], x[w], 2)[0] == pytest.approx(-a2, rel=0.02)
    late = t > stop + 0.1
    assert np.ptp(x[late]) < 2e-3 * x[-1]                             # stopped (creeps at v_reg only)


def test_a_solid_box_on_the_floor_puts_its_weight_on_it(tmp_path):
    d = {
        "units": {"length": "m", "force": "N"},
        "mesh": {"type": "box", "dimensions": [0.2, 0.2, 0.4], "divisions": [2, 2, 4]},
        "materials": {"steel": STEEL},
        "sections": {"s": {"type": "solid", "material": "steel"}},
        "assign": [{"region": "all", "section": "s"}],
        "regions": {"bottom": {"where": {"z": 0.0}}},
        "constraints": [{"region": "all", "dofs": [1, 1, 0]}],
        "loads": [{"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"}],
        "contact": {"rigid": [{"plane": {"point": [0, 0, 0], "normal": [0, 0, 1]}, "region": "bottom"}]},
        "solver": {"backend": "numpy", "mode": "relaxation", "damping": {"type": "kinetic"},
                   "tolerance": 1e-8, "n_steps": 20000},
        "output": {"fields": ["displacement", "contact_force"], "every": 1},
    }
    out, _ = _run(tmp_path, "box", d)
    with h5py.File(tmp_path / "box.h5") as f:
        last = sorted(f["steps"].keys(), key=int)[-1]
        fc = np.asarray(f["steps"][last]["contact_force"])
    W = 7850.0 * 0.2 * 0.2 * 0.4 * G
    assert fc[:, 2].sum() == pytest.approx(W, rel=1e-4)
    assert np.all(fc[:, 2] >= 0.0)


def test_refusals(tmp_path):
    # a node behind the surface at the start
    d = _beam_on_floor(tmp_path, gap=-0.01)
    p = tmp_path / "in.yaml"; p.write_text(yaml.safe_dump(dict(d, name="in"), sort_keys=False), encoding="utf-8")
    with pytest.raises(Exception, match="start behind the surface"):
        prepare_case(str(p)) and run_case(str(p))
    # friction in a relaxation run
    d = _beam_on_floor(tmp_path)
    d["contact"]["rigid"][0]["friction"] = 0.3
    p.write_text(yaml.safe_dump(dict(d, name="fr"), sort_keys=False), encoding="utf-8")
    with pytest.raises(Exception, match="velocity law"):
        run_case(str(p))
    # an unknown region
    d = _beam_on_floor(tmp_path, region="nowhere")
    p.write_text(yaml.safe_dump(dict(d, name="reg"), sort_keys=False), encoding="utf-8")
    with pytest.raises(Exception, match="unknown region"):
        run_case(str(p))
    # a zero normal
    d = _beam_on_floor(tmp_path)
    d["contact"]["rigid"][0]["plane"]["normal"] = [0, 0, 0]
    p.write_text(yaml.safe_dump(dict(d, name="nz"), sort_keys=False), encoding="utf-8")
    with pytest.raises(Exception, match="zero vector"):
        prepare_case(str(p))
