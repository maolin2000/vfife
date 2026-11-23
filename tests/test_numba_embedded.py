"""Embedded reinforcement compiled under `backend: numba`.

`kernels/numba_embedded.BarForce` against the reference
`EmbeddedTruss.force` + `scatter_add`: the nodal force, the stress and the
law's state through a history that yields, unloads and reverses, with a
prestrain per bar, a host deduction and a tension-only set, for the bilinear
bar, with and without rupture -- to round-off; a law without a compiled
twin (Steel02) keeps the reference (the law's compiled
twin is not its NumPy update to the bit). And the solver uses it: the same
case on both paths agrees to round-off; `backend: numpy` is the reference.
"""
import numpy as np
import pytest

numba = pytest.importorskip("numba")

from vfife.core.mesh import box_mesh, scatter_add          # noqa: E402
from vfife.elements import truss                           # noqa: E402
from vfife.materials.base import create                    # noqa: E402

LAWS = {
    "rebar": lambda: create("rebar", E=2.0e11, rho=7850.0, yield_strength=300e6,
                            hardening=2.0e9, kinematic_fraction=0.5),
    "rebar_rupture": lambda: create("rebar", E=2.0e11, rho=7850.0, yield_strength=300e6,
                                    hardening=2.0e9, kinematic_fraction=0.2, rupture_strain=4e-3),
}


def _pair(law, prestrain, tonly, ded):
    mesh = box_mesh((1.0, 0.3, 0.3), (6, 2, 2))
    bn, bc = truss.bars_from_spec({"lines": [
        {"start": [0.02, 0.06, 0.06], "end": [0.98, 0.07, 0.05], "divisions": 12},
        {"start": [0.02, 0.24, 0.24], "end": [0.98, 0.23, 0.25], "divisions": 12}]})
    args = (mesh.nodes, mesh.elements, "hexa8", bn, bc)
    out = []
    for _ in range(2):
        e = truss.EmbeddedTruss(*args, LAWS[law](), 3.14e-4, host_deduction=ded, prestrain=prestrain)
        e.pre.tension_only = tonly
        out.append(e)
    return mesh, out


@pytest.mark.parametrize("law", sorted(LAWS))
@pytest.mark.parametrize("prestrain,tonly,ded", [(0.0, False, 0.0), ("profile", True, 3e10)])
def test_the_compiled_bars_are_the_reference_to_round_off(law, prestrain, tonly, ded):
    from vfife.kernels.numba_embedded import BarForce
    if prestrain == "profile":
        prestrain = np.linspace(-2e-4, 3e-3, 24)
    mesh, (ref, cmp_) = _pair(law, prestrain, tonly, ded)
    assert ref.material.numba_point_fn() is not None
    nb = BarForce(cmp_)
    rng = np.random.default_rng(3)
    shake = rng.standard_normal(mesh.nodes.shape) * 1e-5
    worst = 0.0
    for eps in [4e-4 * k for k in range(1, 7)] + [1.5e-3, 5e-4, -6e-4, -1.4e-3, 2e-3]:
        u = np.zeros_like(mesh.nodes)
        u[:, 0] = eps * mesh.nodes[:, 0]
        u[:, 2] = 0.2 * eps * mesh.nodes[:, 0] ** 2          # a little bending too
        u = np.ascontiguousarray(u + shake)
        fb, sb = ref.force(u, 1.0, want_stress=True)
        F_ref = scatter_add(fb, ref.conn, mesh.n_nodes)
        F = np.zeros((mesh.n_nodes, 3))
        s = nb(u, 1.0, F, want_stress=True)
        worst = max(worst, np.abs(F - F_ref).max() / 6.0e4)   # E A 1e-3, the force scale
        assert np.allclose(s, sb, rtol=1e-11, atol=1e-6)
        assert np.allclose(cmp_.state, ref.state, rtol=1e-11, atol=1e-15)
    assert worst < 1e-12, worst
    assert np.abs(ref.state).max() > 0.0                     # it did yield


def test_a_law_without_a_compiled_twin_keeps_the_reference():
    from vfife.kernels.numba_embedded import BarForce
    _mesh, (e, _) = _pair("rebar", 0.0, False, 0.0)
    e.material = create("steel02", E=2.0e11, Fy=3.0e8, b=0.01, R0=20.0, rho=7850.0)
    with pytest.raises(TypeError, match="no compiled"):
        BarForce(e)


def test_a_solid_with_bars_takes_the_compiled_bars_and_agrees(tmp_path):
    from vfife.api import load_case, run_case
    ends = {}
    # "0": `backend: numpy`, the reference (VFIFE_COMPILED_BARS=0 went in
    # v1.18)
    for flag in ("0", "1"):
        src = load_case("cases/case_13_pretensioned_beam.yaml")
        src.set("solver.backend", "numba" if flag == "1" else "numpy")
        src.set("output.path", str(tmp_path / f"b{flag}.h5"))
        box = {}

        def stop(*_a):
            return False if box["s"].step >= 3000 else None
        try:
            run_case(src, on_progress=stop, progress_every=500, on_ready=lambda s, p: box.setdefault("s", s))
        except Exception:
            pass
        s = box["s"]
        assert s.step == 3000
        kinds = {type(getattr(e, "_numba_force", None)).__name__ for e in s.rebar}
        assert kinds == ({"NoneType"} if flag == "0" else {"BarForce"})
        ends[flag] = (s.u.copy(), [e.state.copy() for e in s.rebar])
    u0, st0 = ends["0"]; u1, st1 = ends["1"]
    assert np.abs(u0 - u1).max() <= 1e-9 * np.abs(u0).max()
    for a, b in zip(st0, st1):
        assert np.allclose(a, b, rtol=1e-9, atol=1e-18)


def test_a_probe_of_a_solid_does_not_step_its_bars(tmp_path):
    """`testing.state.probe(...).internal` copies the blocks' states, "a
    question must not be a step" -- but called the bars' force on their own
    state, and a probe of a yielded bar moved its plastic strain (found in
    the review of v1.10)."""
    from vfife.api import load_case, run_case
    from vfife.testing.state import probe
    src = load_case("cases/case_13_pretensioned_beam.yaml")
    src.set("output.path", str(tmp_path / "p.h5"))
    box = {}
    try:
        run_case(src, on_progress=lambda *a: False, progress_every=50,
                 on_ready=lambda s, p: box.setdefault("s", s))
    except Exception:
        pass
    s = box["s"]
    p = probe(s)
    emb = s.rebar[0]
    before = emb.state.copy()
    u = np.zeros((s.nn, 3))
    u[:, 0] = 0.02 * s.mesh.nodes[:, 0]       # 2% along the beam: far past yield
    for _ in range(3):
        p.internal(u.ravel())
    assert np.array_equal(emb.state, before)
    emb.force(u, s.dt)                         # and that state is one a step does move
    assert not np.array_equal(emb.state, before)
