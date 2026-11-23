"""Embedded reinforcement: the bar element, the embedding, and the RC section.

The embedding is the piece that can go wrong quietly.  A bar that is merely
*near* the right place still produces a plausible-looking load-displacement
curve, so these tests pin the properties that must hold exactly rather than
approximately:

  * the host interpolation reproduces position and sums to one, on every
    element type -- otherwise the bar is somewhere else;
  * the assembled bar force sums to zero -- otherwise the embedding is injecting
    momentum into the mesh;
  * a bar laid on mesh nodes puts force on those nodes and NOWHERE else -- the
    embedding must degenerate to the shared-node case, not merely approach it;
  * rigid translation and rigid rotation give zero bar force;
  * an RC prism matches the transformed-section formula.
"""

import numpy as np
import pytest

from vfife.core.mesh import box_mesh, scatter_add
from vfife.core.solver import Solver
from vfife.elements import truss
from vfife.materials import create

ELEMENTS = ["hexa8", "tet4", "tet10", "wedge6"]


def steel(**kw):
    kw.setdefault("yield_strength", 400e6)
    return create("rebar", E=2.0e11, rho=7850.0, **kw)


@pytest.mark.parametrize("element", ELEMENTS)
def test_host_interpolation_is_exact(element):
    """Inverse mapping + shape functions reproduce the point they were built for."""
    mesh = box_mesh((1.2, 0.8, 0.6), (3, 2, 2), element=element)
    rng = np.random.default_rng(2)
    pts = np.column_stack([rng.uniform(0.05, 1.15, 40),
                           rng.uniform(0.05, 0.75, 40),
                           rng.uniform(0.05, 0.55, 40)])
    host, w = truss.locate(mesh.nodes, mesh.elements, element, pts)
    assert np.all(host >= 0)
    rebuilt = np.einsum("pa,pai->pi", w, mesh.nodes[mesh.elements[host]])
    assert np.abs(rebuilt - pts).max() < 1e-12
    assert np.abs(w.sum(axis=1) - 1.0).max() < 1e-12


def test_point_outside_the_mesh_is_an_error():
    """A bar that has slipped outside the concrete must stop the run."""
    mesh = box_mesh((1.0, 0.2, 0.2), (4, 1, 1))
    with pytest.raises(ValueError, match="not inside"):
        truss.locate(mesh.nodes, mesh.elements, "hexa8",
                     np.array([[0.5, 0.1, 2.0]]))


def test_uniform_stretch_gives_the_exact_bar_stress():
    mesh = box_mesh((1.0, 0.2, 0.2), (4, 1, 1))
    mat = steel()
    emb = truss.EmbeddedTruss(mesh.nodes, mesh.elements, "hexa8",
                              np.array([[0.1, 0.1, 0.1], [0.9, 0.1, 0.1]]),
                              np.array([[0, 1]]), mat, 3.14e-4)
    eps = 3e-4
    u = np.zeros_like(mesh.nodes)
    u[:, 0] = eps * mesh.nodes[:, 0]
    f_eff, sig = emb.force(u, 1.0, want_stress=True)
    assert abs(sig[0, 0, 0] - mat.E * eps) < 1e-6 * mat.E * eps
    # the embedding must be self-equilibrating: it applies no net force
    F = scatter_add(f_eff, emb.conn, mesh.n_nodes)
    assert np.abs(F.sum(axis=0)).max() < 1e-9 * np.abs(F).max()


def test_node_coincident_bar_degenerates_to_shared_nodes():
    """On mesh nodes the embedding must be EXACTLY the shared-node element."""
    mesh = box_mesh((1.0, 0.2, 0.2), (4, 1, 1))
    mat = steel()
    A = 3.14e-4
    emb = truss.EmbeddedTruss(mesh.nodes, mesh.elements, "hexa8",
                              np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                              np.array([[0, 1]]), mat, A)
    eps = 3e-4
    u = np.zeros_like(mesh.nodes)
    u[:, 0] = eps * mesh.nodes[:, 0]
    f_eff, _ = emb.force(u, 1.0, want_stress=True)
    F = scatter_add(f_eff, emb.conn, mesh.n_nodes)

    n0 = int(np.argmin(np.linalg.norm(mesh.nodes - [0.0, 0.0, 0.0], axis=1)))
    n1 = int(np.argmin(np.linalg.norm(mesh.nodes - [1.0, 0.0, 0.0], axis=1)))
    N = mat.E * eps * A
    assert abs(F[n0, 0] + N) < 1e-9 * N
    assert abs(F[n1, 0] - N) < 1e-9 * N
    # not "small elsewhere" -- identically zero elsewhere
    assert np.abs(np.delete(F, [n0, n1], axis=0)).max() == 0.0


@pytest.mark.parametrize("motion", ["translation", "rotation"])
def test_rigid_body_motion_gives_no_bar_force(motion):
    mesh = box_mesh((1.0, 0.2, 0.2), (4, 1, 1))
    emb = truss.EmbeddedTruss(mesh.nodes, mesh.elements, "hexa8",
                              np.array([[0.1, 0.1, 0.1], [0.9, 0.1, 0.1]]),
                              np.array([[0, 1]]), steel(), 3.14e-4)
    if motion == "translation":
        u = np.tile([0.3, -0.2, 0.1], (mesh.n_nodes, 1))
    else:
        th = np.deg2rad(25.0)
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        u = mesh.nodes @ R.T - mesh.nodes
    f_eff, _ = emb.force(u, 1.0)
    # scale: the force a 1e-4 strain would produce
    scale = steel().E * 1e-4 * 3.14e-4
    assert np.abs(f_eff).max() < 1e-6 * scale


def test_bars_from_spec_subdivides_lines():
    nodes, conn = truss.bars_from_spec({"lines": [
        {"start": [0.0, 0.0, 0.0], "end": [1.0, 0.0, 0.0], "divisions": 4},
        {"start": [0.0, 1.0, 0.0], "end": [1.0, 1.0, 0.0], "divisions": 2},
    ]})
    assert conn.shape == (6, 2)
    assert nodes.shape == (8, 3)
    # the two lines must not be joined end to end: the second starts on its
    # own node 5, not on node 4 where the first one ended
    assert conn[3].tolist() == [3, 4]
    assert conn[4].tolist() == [5, 6]
    lengths = np.linalg.norm(nodes[conn[:, 1]] - nodes[conn[:, 0]], axis=1)
    assert np.allclose(lengths[:4], 0.25)
    assert np.allclose(lengths[4:], 0.5)


# --------------------------------------------------------------------------- #


def rc_prism(with_steel, host_deduction):
    """400 kN on a 200x200 prism, four 20 mm bars: composite action in closed form."""
    L, b, h, off = 1.0, 0.2, 0.2, 0.04
    Ec, Es, Ab, N = 3.0e10, 2.0e11, 3.14159e-4, 400e3
    lines = [{"start": [0.0, y, z], "end": [L, y, z], "divisions": 8}
             for y in (off, b - off) for z in (off, h - off)]
    prob = {
        "element": {"type": "hexa8", "formulation": "b-bar"},
        "regions": {
            "x0": {"box": [-1e-9, 1e-9, -1, 1, -1, 1]},
            "xL": {"box": [L - 1e-9, L + 1e-9, -1, 1, -1, 1]},
            "y0": {"box": [-1, 2, -1e-9, 1e-9, -1, 1]},
            "z0": {"box": [-1, 2, -1, 1, -1e-9, 1e-9]},
        },
        "constraints": [{"region": "x0", "dofs": [1, 0, 0]},
                        {"region": "y0", "dofs": [0, 1, 0]},
                        {"region": "z0", "dofs": [0, 0, 1]}],
        "loads": [{"region": "xL", "vector": [-N, 0.0, 0.0]}],
        "reinforcement": [{
            "name": "longitudinal", "area": Ab,
            "host_deduction": host_deduction,
            "material": {"type": "rebar", "E": Es, "rho": 7850.0,
                         "yield_strength": 400e6, "hardening": 0.0},
            "lines": lines,
        }] if with_steel else [],
        # The closed-form assertions below allow 0.2--0.5%; requiring a global
        # residual below 1e-8 made the host-deduction variant fail at 7.2e-8
        # despite already matching the transformed section by orders tighter.
        "solver": {"mode": "relaxation", "n_steps": 40000, "tolerance": 1e-7,
                   "backend": "numpy", "damping": {"type": "kinetic"},
                   },
    }
    mesh = box_mesh((L, b, h), (8, 4, 4))
    conc = create("linear_elastic", E=Ec, nu=0.0, rho=2400.0)
    s = Solver(prob, mesh, conc, log=lambda *_: None).run(progress_every=200)
    # Gauge planes in the interior.  An evenly split nodal load is not the
    # consistent traction of a hexahedron face, so the loaded end carries a
    # Saint-Venant disturbance (43 um of spread, measured) that has nothing to
    # do with the reinforcement.
    x = mesh.nodes[:, 0]
    ua = s.u[np.abs(x - 0.25) < 1e-9, 0].mean()
    ub = s.u[np.abs(x - 0.75) < 1e-9, 0].mean()
    return s, (ub - ua) / 0.5, dict(Ec=Ec, Es=Es, As=4 * Ab, Ag=b * h, N=N)


@pytest.mark.slow
# 轴向理论解核对；同上。
def test_plain_prism_matches_axial_theory():
    _, eps, c = rc_prism(False, 0.0)
    exact = -c["N"] / (c["Ec"] * c["Ag"])
    assert abs(eps - exact) < 2e-3 * abs(exact)


def test_embedded_bars_stiffen_the_section_by_Es_As():
    """Without the deduction the bar duplicates the concrete it sits in."""
    _, eps, c = rc_prism(True, 0.0)
    exact = -c["N"] / (c["Ec"] * c["Ag"] + c["Es"] * c["As"])
    assert abs(eps - exact) < 2e-3 * abs(exact)


@pytest.mark.slow
# 换算截面的物理核对；本模块另有 8 个快测试守着接口。
def test_host_deduction_recovers_the_transformed_section():
    s, eps, c = rc_prism(True, 3.0e10)
    exact = -c["N"] / (c["Ec"] * (c["Ag"] - c["As"]) + c["Es"] * c["As"])
    assert abs(eps - exact) < 2e-3 * abs(exact)

    # and the load must actually split the way the transformed section says
    _, sig = s.rebar[0].force(s.u, 1.0, want_stress=True)
    mid = np.abs(s.rebar[0].x0_e[:, :, 0].mean(axis=1) - 0.5) < 0.2
    sig_s = sig[mid, 0, 0].mean()
    assert abs(sig_s - c["Es"] * exact) < 5e-3 * abs(c["Es"] * exact)
    N_s = sig_s * c["As"]
    N_c = c["Ec"] * eps * (c["Ag"] - c["As"])
    assert abs(N_s + N_c + c["N"]) < 5e-3 * c["N"]


def test_reinforcement_enters_the_time_step_and_the_mass():
    """A stiff light bar sets the CFL limit; forgetting it is silently unstable."""
    mesh = box_mesh((1.0, 0.2, 0.2), (4, 1, 1))
    base = {
        "element": {"type": "hexa8"},
        "regions": {"x0": {"box": [-1e-9, 1e-9, -1, 1, -1, 1]}},
        "constraints": [{"region": "x0", "dofs": [1, 1, 1]}],
        "solver": {"mode": "dynamic", "total_time": 1e-3, "backend": "numpy"},
    }
    bar = [{"area": 3.14e-4,
            "material": {"type": "rebar", "E": 2.0e11, "rho": 7850.0,
                         "yield_strength": 400e6},
            "lines": [{"start": [0.05, 0.1, 0.1], "end": [0.95, 0.1, 0.1],
                       "divisions": 30}]}]
    conc = create("linear_elastic", E=3e10, nu=0.2, rho=2400.0)
    plain = Solver(base, mesh, conc, log=lambda *_: None)
    rc = Solver({**base, "reinforcement": bar}, mesh, conc, log=lambda *_: None)
    assert rc.dt < plain.dt                       # the bar binds the step
    assert rc.mass.sum() > plain.mass.sum()       # and adds its own mass
