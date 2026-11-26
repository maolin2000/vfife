"""The implicit solver's tangent from the members that feel each nudge
(1.15), and its sparsity built once.

Moving the nodes of one colour changes the force of the members that touch
them and of no other, so the columns can be read from those members alone:
a quarter of the arithmetic of differencing the whole force, and the same
matrix to round-off."""
import numpy as np
import pytest
import scipy.sparse as sp

import vfife.implicit as im
from vfife.api import run_case

pytest.importorskip("numba")

E, L = 2.06e11, 2.0


def _lattice(nx):
    pts, lines, n = {}, {}, 0
    name = lambda i, j, k: f"P{i}_{j}_{k}"
    for i in range(nx + 1):
        for j in range(2):
            for k in range(2):
                pts[name(i, j, k)] = [i * L, j * L, k * L]
    for i in range(nx + 1):
        for j in range(2):
            for k in range(2):
                pairs = []
                if i < nx: pairs.append(("c", name(i + 1, j, k)))
                if j < 1: pairs.append(("b", name(i, 1, k)))
                if k < 1: pairs.append(("b", name(i, j, 1)))
                if i < nx and k < 1: pairs.append(("d", name(i + 1, j, 1)))
                for g, other in pairs:
                    lines[f"{g}{n}"] = {"through": [name(i, j, k), other], "size": 10.0}; n += 1
    return pts, lines


def _case(tmp_path, name="lat", nx=5, backend="numba", steps=None, extra=None, jack=False):
    pts, lines = _lattice(nx)
    frame = [k for k in lines if k[0] in "cb"]; diag = [k for k in lines if k[0] == "d"]
    assign = [{"region": r, "section": "s", "orientation": [0.3, 0.5, 0.8]} for r in frame]
    for i, r in enumerate(diag):
        if i % 4 == 0:
            assign.append({"region": r, "section": "k", "orientation": [0, 1, 0]})
        else:
            a = {"region": r, "section": "t"}
            if i % 4 == 1: a["tension_only"] = True
            if jack and i % 4 == 2: a["jack"] = 3.0e4
            assign.append(a)
    d = {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines", "points": pts, "lines": lines},
         "materials": {"steel": {"type": "linear_elastic", "E": E, "nu": 0.3, "rho": 7850.0}},
         "sections": {"s": {"type": "beam", "material": "steel", "shape": "general",
                            "area": 0.02, "iy": 2e-4, "iz": 2e-4, "j": 4e-4},
                      "t": {"type": "bar", "material": "steel", "shape": "general", "area": 0.004},
                      "k": {"type": "connector", "stiffness": [5e8, 5e8, 5e8, 1e7, 1e7, 1e7]}},
         "assign": assign,
         "constraints": [{"region": k, "dofs": [1] * 6} for k in pts if k.startswith("P0_")],
         "loads": [{"region": "all", "gravity": [0, 0, -9.80665]}] +
                  [{"region": k, "vector": [0, 2e4, -6e4], "per_node": True} for k in pts if k.startswith(f"P{nx}_")],
         "steps": steps or [{"name": "a", "load_steps": 3}],
         "solver": {"mode": "static", "backend": backend, "tolerance": 1e-9},
         "output": {"path": str(tmp_path / f"{name}.h5")}}
    d.update(extra or {})
    return d


def _run(d, members, monkeypatch):
    monkeypatch.setattr(im, "_MEMBERS", members)
    got, seen = {}, []
    orig = im.Newton.tangent

    def spy(self, free):
        K = orig(self, free); seen.append(K); return K
    monkeypatch.setattr(im.Newton, "tangent", spy)
    out = run_case(d, on_ready=lambda s, _p: got.__setitem__("s", s))
    monkeypatch.setattr(im.Newton, "tangent", orig)
    assert out.status == "CONVERGED", out.message
    return got["s"], seen


def test_the_tangent_from_the_members_is_the_whole_forces_to_round_off(tmp_path, monkeypatch):
    """Beams, bars (some tension-only, slack under this load), connectors."""
    a, Ka = _run(_case(tmp_path, "a"), True, monkeypatch)
    b, Kb = _run(_case(tmp_path, "b"), False, monkeypatch)
    assert a._tangent_picks(a._newton.groups) is not None
    K0, K1 = Ka[0], Kb[0]                       # the first tangent: both at the same state
    assert (K0.indices == K1.indices).all() and (K0.indptr == K1.indptr).all()
    assert abs(K0 - K1).max() < 1e-10 * abs(K1).max()
    # the converged states: to well inside the solver tolerance (1.1e-9 with the symmetric LU, 1.22)
    assert np.abs(a.q - b.q).max() < 1e-8 * np.abs(b.q).max()
    assert a._newton.evaluations < 0.5 * b._newton.evaluations


def _both(d, monkeypatch):
    """One run on the members' tangent; at EVERY tangent the whole-force one
    is taken at the same state and compared. Returns (solver, largest
    relative difference, tangents compared)."""
    monkeypatch.setattr(im, "_MEMBERS", True)
    got, worst = {}, [0.0, 0]
    orig = im.Newton.tangent

    def spy(self, free):
        K = orig(self, free)
        monkeypatch.setattr(im, "_MEMBERS", False)
        Kw = orig(self, free)
        monkeypatch.setattr(im, "_MEMBERS", True)
        self.tangents -= 1
        worst[0] = max(worst[0], float(abs(K - Kw).max() / abs(Kw).max())); worst[1] += 1
        return K
    monkeypatch.setattr(im.Newton, "tangent", spy)
    out = run_case(d, on_ready=lambda s, _p: got.__setitem__("s", s))
    monkeypatch.setattr(im.Newton, "tangent", orig)
    assert out.status == "CONVERGED", out.message
    return got["s"], worst[0], worst[1]


def _staged(tmp_path, name):
    pts, lines = _lattice(5)
    diag = [k for k in lines if k[0] == "d"]
    late = diag[-6:]                       # bars and one connector (every fourth diagonal is one)
    assert any(diag.index(r) % 4 == 0 for r in late)
    steps = [{"name": "first", "load_steps": 2}, {"name": "more", "activate": late, "load_steps": 2},
             {"name": "load", "load_steps": 2, "loads": [{"region": "P5_1_1", "vector": [0, 0, -5e4], "keep": True}]}]
    return _case(tmp_path, name, steps=steps, jack=True,
                 extra={"solver": {"mode": "static", "backend": "numba", "tolerance": 1e-8}})


def test_a_staged_run_with_jacks_and_members_born_later_has_the_same_tangent_throughout(tmp_path, monkeypatch):
    """A bar being jacked holds a force whatever its length: it has no axial
    stiffness in its step, and its length follows every nudge (found in
    review: taken once a tangent, the bar's E A / L was in K and Newton went
    linear -- 232 iterations for 20). A connector born later has its offset;
    the batches are rebuilt when a step builds members."""
    s, worst, n = _both(_staged(tmp_path, "t"), monkeypatch)
    assert n >= 6 and worst < 1e-9
    assert (s._jack_T > 0).any() and s._conn_birth is not None and s.member_built.all()


def test_a_jack_converges_as_fast_on_the_members_tangent(tmp_path, monkeypatch):
    a, _ = _run(_staged(tmp_path, "a"), True, monkeypatch)
    b, _ = _run(_staged(tmp_path, "b"), False, monkeypatch)
    assert a._newton.iterations <= b._newton.iterations + 2
    assert np.abs(a.q - b.q).max() < 1e-7 * np.abs(b.q).max()
    assert np.abs(a.end_actions - b.end_actions).max() < 1e-6 * np.abs(b.end_actions).max()


def test_a_deflected_lattice_has_the_same_tangent_at_every_iteration(tmp_path, monkeypatch):
    d = _case(tmp_path, "big")
    for ld in d["loads"][1:]:
        ld["vector"] = [0, 4e5, -1.2e6]                           # metres of deflection: the frames turn
    s, worst, n = _both(d, monkeypatch)
    assert n >= 6 and worst < 1e-9 and np.abs(s.q).max() > 0.05


def test_what_the_kernels_do_not_evaluate_member_by_member_keeps_the_whole_force(tmp_path, monkeypatch):
    s, _ = _run(_case(tmp_path, "np", backend="numpy"), True, monkeypatch)
    assert s._tangent_picks(s._newton.groups) is None            # the NumPy backend
    monkeypatch.setattr(type(s), "backend", "numba", raising=False)
    s.backend = "numba"
    assert s._tangent_picks(s._newton.groups) is not None
    s._plastic_d = np.zeros((s.nel, 12))                         # plastic hinges
    assert s._tangent_picks(s._newton.groups) is None
    s._plastic_d = None
    assert s._tangent_picks(s._newton.groups) is not None
    # a force wrapped on the instance -- a rigid contact surface's penalty, a
    # script's own term -- is not a member's (found in review)
    inner = s.internal_force
    s.internal_force = lambda: inner()
    assert s._tangent_picks(s._newton.groups) is None
    del s.internal_force
    s.contact = object()
    assert s._tangent_picks(s._newton.groups) is None


def test_a_rigid_surface_under_the_lattice_keeps_the_whole_force_and_converges(tmp_path, monkeypatch):
    d = _case(tmp_path, "c", nx=3)
    d["contact"] = {"rigid": [{"plane": {"point": [0, 0, -5e-4], "normal": [0, 0, 1]}, "region": "all", "stiffness": 5e8}]}
    s, _ = _run(d, True, monkeypatch)
    assert s._tangent_picks(s._newton.groups) is None
    assert s._newton.iterations < 60                               # 6; without the contact's stiffness, 1720 and no answer
    assert np.abs(s.dofs.view(s.q)[:, 2]).max() > 5e-4             # it did reach the plane


def test_the_sparsity_built_once_gives_the_matrix_the_loop_gave(tmp_path, monkeypatch):
    """The whole-force path, values placed by a stored permutation, against
    the loop it replaced (one small array a node, a COO matrix a tangent):
    the same matrix to the bit."""
    s, seen = _run(_case(tmp_path, "w"), False, monkeypatch)
    nw = s._newton
    monkeypatch.setattr(im, "_MEMBERS", False)
    K = nw.tangent(np.ones(s.ndof, bool))
    base = nw.get()
    rows_all, cols_all, vals_all = [], [], []
    for group in nw.groups:
        for comp in range(nw.stride):
            h = nw.h_r if comp >= 3 else nw.h_t
            cols = nw.node_dofs[group, comp]
            dx = np.zeros(s.ndof); dx[cols] = h
            nw.put(base); nw.advance(dx); fp = nw.force()
            nw.put(base); nw.advance(-dx); fm = nw.force()
            df = (fp - fm) / (2.0 * h)
            for a_, c in zip(group, cols):
                r = nw._rows[a_]
                rows_all.append(r); cols_all.append(np.full(len(r), c)); vals_all.append(df[r])
    nw.put(base)
    ref = sp.coo_matrix((np.concatenate(vals_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
                        shape=(s.ndof, s.ndof)).tocsc()
    assert (K.indices == ref.indices).all() and (K.indptr == ref.indptr).all()
    assert np.array_equal(K.data, ref.data)
