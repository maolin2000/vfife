"""The Numba backend must give the same answer as the NumPy reference.

This is the test that lets the fused kernel be trusted: the NumPy path is where
the physics is verified (rigid rotation, patch test, closed-form constitutive
solutions), and this file pins the fast path to it -- element force, stress and
material state, on every formulation and every material, on distorted meshes
and after a load history long enough to exercise the return maps.
"""

import os
import numpy as np
import pytest

from vfife import kernels
from vfife.core.mesh import box_mesh, scatter_add
from vfife.core.solver import Solver
from vfife.core.diagnostics import ConvergenceFailure
from vfife.elements import continuum
from vfife.materials import create

FORMULATIONS = ["full", "reduced", "b-bar"]


def materials():
    return {
        "linear_elastic": create("linear_elastic", E=3e10, nu=0.2, rho=2400.0),
        "chaboche": create(
            "chaboche", E=2.1e11, nu=0.3, rho=7800.0, yield_strength=250e6,
            Q_inf=100e6, b=20.0, H_iso=1e9, C1=30e9, gamma1=300.0,
            C2=5e9, gamma2=30.0, n_newton=3,
            damage_initiation_strain=0.002, damage_evolution_rate=20.0,
            damage_viscosity=1e-4,
        ),
        "cdp": create(
            "cdp", E=3e10, nu=0.2, rho=2400.0, l_ref=0.2,
            compression={"source": "gb50010", "fc": 30e6},
            tension={"source": "fracture_energy", "ft": 3e6, "Gf": 120.0},
        ),
    }


def setup(formulation, material, distort=0.12, seed=0):
    mesh = box_mesh((1.2, 0.6, 0.6), (5, 3, 3))
    rng = np.random.default_rng(seed)
    hi = np.array([1.2, 0.6, 0.6])
    inner = np.all((mesh.nodes > 1e-9) & (mesh.nodes < hi - 1e-9), axis=1)
    mesh.nodes[inner] += rng.uniform(-distort * 0.2, distort * 0.2,
                                     (int(inner.sum()), 3))
    pre = continuum.precompute(mesh.nodes, mesh.elements, formulation=formulation,
                           hourglass_coeff=0.1, material=material)
    x0 = mesh.nodes[mesh.elements]
    x0_e = np.ascontiguousarray(x0 - x0.mean(axis=1, keepdims=True))
    material.bind_points(np.repeat(pre.char_len, pre.n_gp))
    return mesh, pre, x0_e


def run(force, mesh, material, pre, u_path, dt=1e-4):
    """Push a displacement history through one backend; return the final state."""
    state = material.init_state(mesh.n_elem * pre.n_gp)
    quat = np.zeros((mesh.n_elem, 4))
    quat[:, 0] = 1.0
    F = None
    for i, u in enumerate(u_path):
        f_e, sig = force(u, quat, state, dt,
                         rot_iters=30 if i == 0 else 2, want_stress=True)
        F = scatter_add(f_e, mesh.elements, mesh.n_nodes)
    return F, np.array(sig, copy=True), state, quat


def displacement_history(mesh, n=25, scale=6e-4, seed=1):
    rng = np.random.default_rng(seed)
    x = mesh.nodes
    u = np.zeros_like(x)
    out = []
    # a rigid rotation on top of a growing random field: exercises the
    # co-rotational path as well as the constitutive one
    for i in range(n):
        th = 0.02 * (i + 1)
        c, s = np.cos(th), np.sin(th)
        rot = np.stack([c * x[:, 0] - s * x[:, 1] - x[:, 0],
                        s * x[:, 0] + c * x[:, 1] - x[:, 1],
                        np.zeros_like(x[:, 2])], axis=1)
        u = u + rng.normal(size=x.shape) * scale
        out.append(np.ascontiguousarray(u + rot))
    return out


@pytest.mark.parametrize("formulation", FORMULATIONS)
@pytest.mark.parametrize("name", ["linear_elastic", "chaboche", "cdp"])
def test_numba_matches_numpy(formulation, name):
    mat_np = materials()[name]
    mat_nb = materials()[name]
    mesh, pre, x0_e = setup(formulation, mat_np)
    _, pre_nb, _ = setup(formulation, mat_nb)

    path = displacement_history(mesh)
    f_numpy = kernels.make_force(pre, mesh.elements, x0_e, mat_np, "numpy")
    f_numba = kernels.make_force(pre_nb, mesh.elements, x0_e, mat_nb, "numba")

    F_a, sig_a, st_a, q_a = run(f_numpy, mesh, mat_np, pre, path)
    F_b, sig_b, st_b, q_b = run(f_numba, mesh, mat_nb, pre_nb, path)

    def rel(a, b):
        scale = max(float(np.abs(a).max()), 1e-30)
        return float(np.abs(a - b).max()) / scale

    # CDP is held to a looser bound for a physical reason, not a numerical
    # excuse: past the peak its softening branch is genuinely ill-conditioned,
    # so two orderings of the same arithmetic separate. Everything else is at
    # round-off.
    tol = 1e-9 if name == "cdp" else 1e-12
    assert rel(F_a, F_b) < tol, f"internal force: {rel(F_a, F_b):.3e}"
    assert rel(sig_a, sig_b) < tol, f"stress: {rel(sig_a, sig_b):.3e}"
    assert rel(q_a, q_b) < 1e-12, f"rotation: {rel(q_a, q_b):.3e}"
    if st_a.size:
        assert rel(st_a, st_b) < tol, f"state: {rel(st_a, st_b):.3e}"


@pytest.mark.parametrize("formulation", FORMULATIONS)
def test_numba_preserves_rigid_body_rotation(formulation):
    """The property the whole co-rotational formulation rests on, on the fast path."""
    mat = materials()["linear_elastic"]
    mesh, pre, x0_e = setup(formulation, mat, distort=0.15)
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    th = np.deg2rad(30.0)
    Kx = np.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)
    u = mesh.nodes @ R.T - mesh.nodes + np.array([0.1, -0.2, 0.3])

    force = kernels.make_force(pre, mesh.elements, x0_e, mat, "numba")
    state = mat.init_state(mesh.n_elem * pre.n_gp)
    quat = np.zeros((mesh.n_elem, 4))
    quat[:, 0] = 1.0
    f_e, sig = force(u, quat, state, 1e-6, rot_iters=60, want_stress=True)
    scale = mat.E * np.abs(mesh.nodes).max()
    assert np.abs(sig).max() / scale < 1e-9
    assert np.abs(f_e).max() / scale < 1e-9


def test_solver_backends_agree_end_to_end():
    """Same case, two backends, bit-comparable displacement field."""
    case = {
        "mesh": {"dimensions": [1.0, 0.05, 0.05], "divisions": [12, 1, 2]},
        "element": {"formulation": "b-bar", "hourglass_coeff": 0.05},
        "regions": {
            "clamp": {"box": [-1e-9, 1e-9, -1e9, 1e9, -1e9, 1e9]},
            "tip": {"box": [1 - 1e-9, 1 + 1e-9, -1e9, 1e9, -1e9, 1e9]},
        },
        "constraints": [{"region": "clamp", "dofs": [1, 1, 1], "value": None}],
        "loads": [{"region": "tip", "vector": [0.0, 0.0, -300.0]}],
        "solver": {"mode": "relaxation", "n_steps": 1500, "tolerance": 0.0,
                   "damping": {"type": "kinetic"}, "total_time": None,
                   },
        "output": {"monitors": {}},
    }
    out = {}
    for backend in ("numpy", "numba"):
        c = {**case, "solver": {**case["solver"], "backend": backend}}
        mesh = box_mesh(c["mesh"]["dimensions"], c["mesh"]["divisions"])
        mat = create("linear_elastic", E=2.1e11, nu=0.3, rho=7800.0)
        s = Solver(c, mesh, mat, log=lambda *_: None)
        with pytest.raises(ConvergenceFailure, match="without convergence"):
            s.run(output_every=10 ** 9, progress_every=500)
        assert s.diagnostics["status"] == "MAX_STEPS"
        out[backend] = s.u
    rel = np.abs(out["numpy"] - out["numba"]).max() / np.abs(out["numpy"]).max()
    assert rel < 1e-11, rel


def test_unknown_backend_is_rejected():
    """`cuda` is the plausible wrong guess -- the GPU backend is called `warp`.

    Asserted on the structured fields rather than a regex: the message now
    names `solver.backend`, which is the YAML key the user has to edit, and the
    list of real names moved into the hint where the CLI prints it.
    """
    from vfife.core.diagnostics import UserError

    with pytest.raises(UserError) as exc:      # a ValueError subclass
        kernels.make_force(None, None, None, None, "cuda")
    assert exc.value.code == "UNKNOWN_BACKEND"
    assert "solver.backend" in exc.value.message
    assert "warp" in exc.value.hint and "numpy" in exc.value.hint


def test_gather_assembly_matches_bincount():
    """The two assembly shapes must agree to summation-order noise, no more."""
    from vfife.kernels.assembly import GatherAssembler

    mesh = box_mesh((2.0, 1.0, 1.0), (6, 4, 4))
    rng = np.random.default_rng(4)
    f_e = rng.normal(size=(mesh.n_elem, 8, 3)) * 1e6

    a = scatter_add(f_e, mesh.elements, mesh.n_nodes)
    g = GatherAssembler(mesh.elements, mesh.n_nodes)
    b = g(f_e).copy()
    assert np.abs(a - b).max() / np.abs(a).max() < 1e-13


def test_gather_assembly_is_reproducible():
    """Fixed index order, no atomics: the same input must give the same bits.

    This is what makes a future CPU-vs-GPU comparison a real check.
    """
    from vfife.kernels.assembly import GatherAssembler

    mesh = box_mesh((2.0, 1.0, 1.0), (6, 4, 4))
    rng = np.random.default_rng(5)
    f_e = rng.normal(size=(mesh.n_elem, 8, 3)) * 1e6
    g = GatherAssembler(mesh.elements, mesh.n_nodes)
    first = g(f_e).copy()
    for _ in range(5):
        assert np.array_equal(g(f_e), first)


def test_inverse_connectivity_covers_every_slot():
    from vfife.kernels.assembly import inverse_connectivity

    mesh = box_mesh((2.0, 1.0, 1.0), (5, 3, 3))
    order, offsets = inverse_connectivity(mesh.elements, mesh.n_nodes)
    assert offsets[0] == 0
    assert offsets[-1] == mesh.n_elem * 8
    assert np.array_equal(np.sort(order), np.arange(mesh.n_elem * 8))
    flat = mesh.elements.ravel()
    for n in (0, mesh.n_nodes // 2, mesh.n_nodes - 1):
        assert np.all(flat[order[offsets[n]:offsets[n + 1]]] == n)


# --------------------------------------------------------------------------- #
#  Warp (GPU) backend
#
#  Run on the CPU device here: Warp compiles the same kernel source for both, so
#  correctness is fully checkable without a GPU.  What cannot be measured
#  without one is throughput.
# --------------------------------------------------------------------------- #

#: Only the tests that need warp are skipped without it. This used to be a
#: module-level `importorskip`, which skipped the whole file -- the numba
#: checks above included -- wherever warp-lang is not installed, CI among them.
try:
    import warp
    HAVE_WARP = True
except ImportError:
    warp = None
    HAVE_WARP = False
needs_warp = pytest.mark.skipif(not HAVE_WARP, reason="warp-lang not installed")
#: the backends the cross-element comparisons below run besides numpy
COMPARED = ("numba", "warp") if HAVE_WARP else ("numba",)
#: The device every Warp comparison runs on. It is "cpu" by default so the
#: suite is reproducible on a machine with no GPU -- but that means nothing in
#: this file has ever exercised CUDA, while `cases/case_06` ships asking for it.
#: Set VFIFE_WARP_DEVICE=cuda on a machine with a card to close that gap:
#:
#:     VFIFE_WARP_DEVICE=cuda pytest tests/test_backends.py
#:
#: fp64 is kept whatever the device, so a difference in the result is a
#: difference in the kernel and not in the precision.
WARP_DEVICE = os.environ.get("VFIFE_WARP_DEVICE", "cpu")


def _require_warp_device(name):
    """Fail loudly if the requested device is not the one we get.

    Warp falls back to the CPU when no CUDA device is present. Without this
    check a run asking for cuda on a machine with no card passes every
    assertion in this file while proving nothing about the GPU path -- the
    worst kind of green, since the whole point of setting the variable is to
    produce evidence about CUDA specifically.
    """
    if name == "cpu":
        return
    import warp
    warp.init()
    # Resolve through warp rather than matching strings: "cuda" is an alias
    # warp accepts and resolves to "cuda:0", so a membership test against
    # get_devices() rejects the most natural spelling of a device that IS
    # present -- which is exactly what happened the first time this guard
    # ran on a real card. Ask warp what the name means, then check what
    # came back.
    try:
        device = warp.get_device(name)
    except Exception as exc:
        raise RuntimeError(
            f"VFIFE_WARP_DEVICE={name!r} is not a device warp can resolve "
            f"({exc}). Warp sees {[str(d) for d in warp.get_devices()]}."
        ) from exc
    if not device.is_cuda:
        raise RuntimeError(
            f"VFIFE_WARP_DEVICE={name!r} resolved to {device!r}, which is not "
            "a CUDA device. Refusing to run: the comparisons would fall back "
            "to the CPU and report a pass that says nothing about that device."
        )


if HAVE_WARP:
    _require_warp_device(WARP_DEVICE)
WARP_CPU = dict(device=WARP_DEVICE, precision="fp64")


def run_warp(mesh, mat, pre, x0e, path, **kw):
    f = kernels.make_force(pre, mesh.elements, x0e, mat, "warp", **kw)
    state = mat.init_state(mesh.n_elem * pre.n_gp)
    quat = np.zeros((mesh.n_elem, 4))
    quat[:, 0] = 1.0
    for i, u in enumerate(path):
        f_e, sig = f(u, quat, state, 1e-4,
                     rot_iters=30 if i == 0 else 2, want_stress=True)
    return f.assemble(f_e), np.array(sig), state, quat


@needs_warp
@pytest.mark.parametrize("name,formulation", [
    ("linear_elastic", "full"),
    ("linear_elastic", "reduced"),
    ("linear_elastic", "b-bar"),
    ("chaboche", "b-bar"),
    ("cdp", "reduced"),
    ("cdp", "b-bar"),
])
def test_warp_fp64_matches_numpy(name, formulation):
    """In double precision the GPU port must reproduce the reference exactly.

    fp64 is not the production mode -- a consumer GeForce runs it at 1/64 of
    fp32 -- it exists so the port can be checked without a precision argument
    getting in the way.
    """
    mat_np = materials()[name]
    mat_wp = materials()[name]
    mesh, pre, x0_e = setup(formulation, mat_np)
    _, pre_wp, _ = setup(formulation, mat_wp)
    path = displacement_history(mesh)

    f_numpy = kernels.make_force(pre, mesh.elements, x0_e, mat_np, "numpy")
    F_a, sig_a, st_a, q_a = run(f_numpy, mesh, mat_np, pre, path)
    F_b, sig_b, st_b, q_b = run_warp(mesh, mat_wp, pre_wp, x0_e, path, **WARP_CPU)

    def rel(a, b):
        return float(np.abs(a - b).max()) / max(float(np.abs(a).max()), 1e-30)

    # CDP is looser because the softening branch is genuinely ill-conditioned:
    # once an element is past the peak, two orderings of the same arithmetic
    # separate.  Everything else is at round-off.
    tol = 1e-9 if name == "cdp" else 1e-11
    assert rel(F_a, F_b) < tol, f"force {rel(F_a, F_b):.2e}"
    assert rel(sig_a, sig_b) < tol, f"stress {rel(sig_a, sig_b):.2e}"
    assert rel(q_a, q_b) < 1e-12, f"rotation {rel(q_a, q_b):.2e}"
    if st_a.size:
        assert rel(st_a, st_b) < tol, f"state {rel(st_a, st_b):.2e}"


@needs_warp
@pytest.mark.parametrize("name", ["linear_elastic", "cdp"])
def test_warp_mixed_precision_stays_within_fp32(name):
    """fp32 elements must cost single-precision accuracy, not correctness.

    This is the production mode, so what matters is that it stays finite and
    close, not that it matches to round-off: about four significant digits
    survive, which is far more than the constitutive parameters are known to.
    """
    mesh, pre, x0_e = setup("reduced", materials()[name])
    path = displacement_history(mesh)
    out = {}
    for prec in ("fp64", "mixed"):
        m = materials()[name]
        _, p2, _ = setup("reduced", m)
        out[prec] = run_warp(mesh, m, p2, x0_e, path, device="cpu",
                             precision=prec)
    a, b = out["fp64"], out["mixed"]
    for i, nm in enumerate(("force", "stress")):
        assert np.isfinite(b[i]).all(), f"{nm} went non-finite in fp32"
        rel = np.abs(a[i] - b[i]).max() / max(np.abs(a[i]).max(), 1e-30)
        assert rel < 1e-3, f"{nm}: {rel:.2e}"


@needs_warp
@pytest.mark.parametrize("with_prescribed", [False, True])
def test_warp_device_loop_matches_cpu_loop(with_prescribed):
    """The whole time loop on the device must give the CPU loop's answer.

    The Warp backend keeps displacement, velocity, material state and element
    force resident and runs the integrator there too, because shipping them
    across the bus every step would cost more than the kernel it feeds
    (~280 MB/iteration at 10^6 elements). That means a second implementation of
    the integrator, so it gets its own equivalence test.
    """
    out = {}
    for backend, kw in (("numba", {}), ("warp", WARP_CPU)):
        case = {
            "mesh": {"dimensions": [1.0, 0.05, 0.05], "divisions": [12, 1, 2]},
            "element": {"formulation": "b-bar", "hourglass_coeff": 0.05},
            "regions": {
                "clamp": {"box": [-1e-9, 1e-9, -1e9, 1e9, -1e9, 1e9]},
                "tip": {"box": [1 - 1e-9, 1 + 1e-9, -1e9, 1e9, -1e9, 1e9]},
            },
            "constraints": [{"region": "clamp", "dofs": [1, 1, 1], "value": None}],
            "loads": [] if with_prescribed
                     else [{"region": "tip", "vector": [0.0, 0.0, -300.0]}],
            "solver": {"mode": "relaxation", "n_steps": 1200, "tolerance": 0.0,
                       "damping": {"type": "kinetic"}, "total_time": None,
                       "backend": backend, **kw,
                       },
            "output": {"monitors": {}},
        }
        if with_prescribed:
            # displacement control exercises the prescribed-BC path, which the
            # load-controlled case never touches -- and which had the device
            # kernels applying the value before zeroing the constrained dofs,
            # so nothing moved at all
            case["constraints"].append(
                {"region": "tip", "dofs": [0, 0, 1], "value": [0.0, 0.0, -2e-3]}
            )
        mesh = box_mesh(case["mesh"]["dimensions"], case["mesh"]["divisions"])
        mat = create("linear_elastic", E=2.1e11, nu=0.3, rho=7800.0)
        s = Solver(case, mesh, mat, log=lambda *_: None)
        with pytest.raises(ConvergenceFailure, match="without convergence"):
            s.run(output_every=10 ** 9, progress_every=300)
        assert s.diagnostics["status"] == "MAX_STEPS"
        assert np.abs(s.u).max() > 1e-9, f"{backend}: nothing moved"
        out[backend] = (s.u.copy(), s.n_resets, s.residuals[-1])

    a, b = out["numba"], out["warp"]
    assert a[1] == b[1], f"kinetic resets differ: {a[1]} vs {b[1]}"
    assert abs(a[2] / b[2] - 1.0) < 1e-6
    assert np.abs(a[0] - b[0]).max() / np.abs(a[0]).max() < 1e-10


@needs_warp
@pytest.mark.parametrize("name,formulation,limit_gb", [
    ("cdp", "reduced", 1.0),
    ("cdp", "b-bar", 3.0),
    ("linear_elastic", "reduced", 1.0),
])
def test_device_memory_fits_a_million_elements(name, formulation, limit_gb):
    """The resident buffers must leave room on a 16 GB card at 10^6 elements."""
    mesh = box_mesh((12.0, 0.5, 1.0), (20, 5, 5))
    mat = materials()[name]
    pre = continuum.precompute(mesh.nodes, mesh.elements, formulation=formulation,
                           material=mat)
    x0 = mesh.nodes[mesh.elements]
    x0_e = np.ascontiguousarray(x0 - x0.mean(axis=1, keepdims=True))
    mat.bind_points(np.repeat(pre.char_len, pre.n_gp))
    f = kernels.make_force(pre, mesh.elements, x0_e, mat, "warp",
                           device="cpu", precision="mixed")
    per_element = f.bytes_resident() / mesh.n_elem
    projected_gb = per_element * 1e6 / 1e9
    assert projected_gb < limit_gb, (
        f"{name}/{formulation}: {per_element:.0f} B/element -> "
        f"{projected_gb:.2f} GB at 1e6 elements"
    )


@needs_warp
def test_warp_rejects_bad_precision():
    mesh = box_mesh((1.0, 0.5, 0.5), (2, 1, 1))
    mat = materials()["linear_elastic"]
    pre = continuum.precompute(mesh.nodes, mesh.elements, formulation="reduced",
                           material=mat)
    x0 = mesh.nodes[mesh.elements]
    x0_e = np.ascontiguousarray(x0 - x0.mean(axis=1, keepdims=True))
    with pytest.raises(ValueError, match="unknown precision"):
        kernels.make_force(pre, mesh.elements, x0_e, mat, "warp",
                           device="cpu", precision="fp16")


@needs_warp
def test_warp_truss_matches_numpy_through_yield_and_reversal():
    """Embedded reinforcement on the device must track the reference exactly.

    The bars are the one part of the residency design that could have been
    skipped -- evaluating them on the host is only a few thousand elements --
    but that would mean pulling `u` down and pushing a force back every step,
    which is precisely the transfer the device-resident loop exists to avoid.
    So they run on the device, and this pins them to the NumPy path through a
    history that yields, unloads, and reverses into compression: the Bauschinger
    branch is where a mismatched state update would first show.
    """
    from vfife.core.mesh import box_mesh, scatter_add
    from vfife.elements import truss

    mesh = box_mesh((1.0, 0.3, 0.3), (6, 2, 2))
    conc = create("linear_elastic", E=3e10, nu=0.2, rho=2400.0)
    mk = lambda: create("rebar", E=2.0e11, rho=7850.0, yield_strength=300e6,
                        hardening=2.0e9, kinematic_fraction=0.5)
    bn, bc = truss.bars_from_spec({"lines": [
        {"start": [0.02, 0.06, 0.06], "end": [0.98, 0.06, 0.06], "divisions": 12},
        {"start": [0.02, 0.24, 0.24], "end": [0.98, 0.24, 0.24], "divisions": 12}]})
    args = (mesh.nodes, mesh.elements, "hexa8", bn, bc)
    emb_h = truss.EmbeddedTruss(*args, mk(), 3.14e-4, host_deduction=3e10)
    emb_d = truss.EmbeddedTruss(*args, mk(), 3.14e-4, host_deduction=3e10)

    pre = continuum.precompute(mesh.nodes, mesh.elements, material=conc)
    x0 = mesh.nodes[mesh.elements]
    x0_e = np.ascontiguousarray(x0 - x0.mean(axis=1, keepdims=True))
    conc.bind_points(np.repeat(pre.char_len, pre.n_gp))

    f_np = kernels.make_force(pre, mesh.elements, x0_e, conc, "numpy")
    f_wp = kernels.make_force(pre, mesh.elements, x0_e, conc, "warp", **WARP_CPU)
    f_wp.attach_truss(emb_d)
    # (mass, fixed, load groups, prescribed): loads arrive grouped by
    # amplitude curve now, and this comparison applies none
    f_wp.set_nodal(np.ones((mesh.n_nodes, 3)),
                   np.zeros((mesh.n_nodes, 3), np.int8), [], [])
    st = conc.init_state(pre.n_elem * pre.n_gp)
    q = np.zeros((pre.n_elem, 4))
    q[:, 0] = 1.0

    worst = 0.0
    for eps in [4e-4 * k for k in range(1, 7)] + [1.5e-3, 5e-4, -6e-4, -1.4e-3]:
        u = np.zeros_like(mesh.nodes)
        u[:, 0] = eps * mesh.nodes[:, 0]
        fe, _ = f_np(u, q, st, 1.0, rot_iters=30)
        fb, _ = emb_h.force(u, 1.0)
        F_host = f_np.assemble(fe) + scatter_add(fb, emb_h.conn, mesh.n_nodes)

        f_wp.set_u(u)
        f_wp.set_state(st, q)
        f_wp.force_resident(1.0, rot_iters=30)
        F_dev = f_wp.d_F.numpy()

        worst = max(worst, np.abs(F_host - F_dev).max()
                    / max(np.abs(F_host).max(), 1e-30))
        a_h = emb_h.state[:, 1]
        a_d = f_wp.trusses[0].d_state.numpy()[:, 1]
        assert np.abs(a_h - a_d).max() < 1e-14, f"plastic state at eps={eps}"

    assert worst < 1e-12, f"host vs device force: {worst:.3e}"
    # the history must actually have yielded, or this test proves nothing
    assert emb_h.state[:, 1].max() > 1e-3


@pytest.mark.parametrize("element,formulation", [
    ("tet4", "full"), ("tet10", "full"), ("tet10", "b-bar"),
    ("wedge6", "full"), ("wedge6", "b-bar"), ("hexa8", "reduced"),
])
@pytest.mark.parametrize("name", ["linear_elastic", "j2"])
def test_all_elements_agree_across_backends(element, formulation, name):
    """One kernel source, four element topologies, three backends, one answer.

    The element library is data -- dN/dxi, quadrature, node count -- so adding a
    topology must not need a new kernel.  This is the test that says so: the
    same fused Numba and Warp kernels compile for 4, 6, 8 and 10 nodes and land
    on the NumPy reference. Without warp-lang only Numba is compared.
    """
    def mat():
        if name == "j2":
            return create("j2", E=2.1e11, nu=0.3, rho=7800.0,
                          hardening={"source": "linear",
                                     "yield_strength": 250e6, "H": 2e9},
                          kinematic=1e9)
        return create("linear_elastic", E=3e10, nu=0.2, rho=2400.0)

    mesh = box_mesh((1.2, 0.6, 0.6), (3, 2, 2), element=element)
    rng = np.random.default_rng(3)
    hi = np.array([1.2, 0.6, 0.6])
    inner = np.all((mesh.nodes > 1e-9) & (mesh.nodes < hi - 1e-9), axis=1)
    mesh.nodes[inner] += rng.uniform(-0.02, 0.02, (int(inner.sum()), 3))

    x0 = mesh.nodes[mesh.elements]
    x0_e = np.ascontiguousarray(x0 - x0.mean(axis=1, keepdims=True))
    path = displacement_history(mesh, n=12, scale=8e-4)

    out = {}
    for backend, kw in (("numpy", {}), ("numba", {}), ("warp", WARP_CPU)):
        if backend not in ("numpy",) + COMPARED:
            continue
        m = mat()
        pre = continuum.precompute(mesh.nodes, mesh.elements, element=element,
                                   formulation=formulation, material=m)
        m.bind_points(np.repeat(pre.char_len, pre.n_gp))
        f = kernels.make_force(pre, mesh.elements, x0_e, m, backend, **kw)
        out[backend] = run(f, mesh, m, pre, path)

    ref = out["numpy"]
    for backend in COMPARED:
        got = out[backend]
        for k, label in enumerate(("force", "stress", "state")):
            a, b = ref[k], got[k]
            if a is None or not np.size(a):
                continue
            rel = np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
            assert rel < 1e-11, f"{backend} {element}/{formulation} {label}: {rel:.2e}"


# --------------------------------------------------------------------------- #
#  the three elements the catalogue said could not use the fused kernels
# --------------------------------------------------------------------------- #

#: hexa20, wedge15 and pyramid5 declared `backends = ("numpy",)`, and preflight
#: enforced it with CAPABILITY_MISMATCH -- so a case asking for numba on any of
#: them was refused. But the fused kernels are element-generic by construction:
#: `make_kernel(..., n_nodes=...)` bakes the node count in as a compile-time
#: constant and the element supplies only dN/dx and the quadrature. The
#: declaration was under-claiming, and the cost was running ~3x slower than
#: necessary with no way to opt out.
#:
#: `box_mesh` cannot emit these three, so they come from the real Gmsh fixtures
#: -- which is the right source anyway: a path no real file has travelled is a
#: path that has not been tested.
GMSH_ONLY = {
    "hexa20": "box_hexa20_v41.msh",
    "wedge15": "box_wedge15_v41.msh",
    "pyramid5": "box_pyramid5_v41.msh",
}

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def gmsh_block(element):
    """One element type, read from a file Gmsh wrote, with its nodes jittered.

    The jitter matters: on a perfectly regular box every Jacobian is the same
    and a kernel could agree with the reference for the wrong reason.
    """
    from vfife.core.mesh import Mesh
    from vfife.io import gmsh as gmsh_io

    mesh, _primary = gmsh_io.read(os.path.join(FIXTURE_DIR, GMSH_ONLY[element]))
    conn = mesh.cell_blocks[element]
    nodes = mesh.nodes.copy()

    lo, hi = nodes.min(axis=0), nodes.max(axis=0)
    inner = np.all((nodes > lo + 1e-9) & (nodes < hi - 1e-9), axis=1)
    # only the corner nodes may move: displacing a mid-side node off the edge
    # would make a quadratic element genuinely curved, and then the numpy
    # reference is no longer the same problem as a straight-sided one
    corners = np.zeros(len(nodes), bool)
    ncorner = {"hexa20": 8, "wedge15": 6, "pyramid5": 5}[element]
    corners[np.unique(conn[:, :ncorner])] = True
    movable = inner & corners
    if movable.any():
        rng = np.random.default_rng(5)
        nodes[movable] += rng.uniform(-0.01, 0.01, (int(movable.sum()), 3))

    return Mesh(nodes=nodes, elements=np.ascontiguousarray(conn))


@pytest.mark.parametrize("element", sorted(GMSH_ONLY))
@pytest.mark.parametrize("name", ["linear_elastic", "j2"])
def test_the_remaining_elements_also_agree_across_backends(element, name):
    """Same assertion as the four topologies above, for the other three.

    Runs a displacement history -- rigid rotation plus a growing random field --
    so the co-rotational path and the constitutive update are both exercised,
    not just one force evaluation on an undeformed mesh.
    """
    def mat():
        if name == "j2":
            return create("j2", E=2.1e11, nu=0.3, rho=7800.0,
                          hardening={"source": "linear",
                                     "yield_strength": 250e6, "H": 2e9},
                          kinematic=1e9)
        return create("linear_elastic", E=3e10, nu=0.2, rho=2400.0)

    mesh = gmsh_block(element)
    x0 = mesh.nodes[mesh.elements]
    x0_e = np.ascontiguousarray(x0 - x0.mean(axis=1, keepdims=True))
    path = displacement_history(mesh, n=12, scale=8e-4)

    out = {}
    for backend, kw in (("numpy", {}), ("numba", {}), ("warp", WARP_CPU)):
        if backend not in ("numpy",) + COMPARED:
            continue
        m = mat()
        pre = continuum.precompute(mesh.nodes, mesh.elements, element=element,
                                   formulation="full", material=m)
        m.bind_points(np.repeat(pre.char_len, pre.n_gp))
        f = kernels.make_force(pre, mesh.elements, x0_e, m, backend, **kw)
        out[backend] = run(f, mesh, m, pre, path)

    ref = out["numpy"]
    for backend in COMPARED:
        for k, label in enumerate(("force", "stress", "state")):
            a, b = ref[k], out[backend][k]
            if a is None or not np.size(a):
                continue
            rel = np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
            assert rel < 1e-11, f"{backend} {element} {label}: {rel:.2e}"

    if name == "j2":
        # the history must actually have yielded, or this proves nothing about
        # the constitutive path
        assert ref[2][:, 1].max() > 1e-4, "no plastic strain: history too gentle"
