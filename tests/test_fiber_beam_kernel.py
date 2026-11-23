"""The fused fibre-beam kernel must agree with the NumPy path it replaces.

`FiberBeamSolver._internal_force_true` stays the reference implementation; the
kernel in `kernels/numba_fiber_beam.py` is a transcription of it. These tests
pin the transcription, pin the conditions under which it is used at all, and
pin the fallback -- because the fallback is silent by design, and a silent
fallback that stops working is a run that takes twenty times longer with
nothing to say why.
"""
import numpy as np
import pytest

from vfife.api import prepare_case
from vfife.core.fiber_beam_solver import FiberBeamSolver

pytest.importorskip("numba")


SECTION = """
name: kernel_check
units: {length: m, force: N, stress: Pa}
mesh: {type: gmsh, path: MESH}
materials:
  concrete: {type: concrete02, fpc: 25.6e6, epsc0: 0.002, fcu: 15.0e6, epscu: 0.02,
             ft: 3.1e6, Ets: 1.2e9, rat: 0.1, rho: 2400}
  core: {type: concrete02, fpc: 42.0e6, epsc0: 0.0084, fcu: 8.4e6, epscu: 0.0345,
         ft: 3.1e6, Ets: 1.2e9, rat: 0.1, rho: 2400}
STEELS
sections:
BARSECTIONS
  rc:
    type: fiber
    material: concrete
    shape: rect
    width: 0.4
    height: 0.4
    ny: 8
    nz: NZ
    integration: lobatto3
CONFINED
    bars:
BARS
assign:
  - {region: all, section: rc}
regions:
  base: {box: [-0.01, 0.01, -0.01, 0.01, -0.01, 0.01]}
  head: {box: [1.19, 1.21, -0.01, 0.01, -0.01, 0.01]}
constraints:
  - {region: base, dofs: [1, 1, 1, 1, 1, 1]}
loads:
  - {region: head, vector: [-400.0e3, 0.0, 0.0]}
steps:
  - name: push
    constraints:
      - {region: head, dofs: [0, 0, 1, 0, 0, 0], value: 0.01}
solver: {backend: numpy, mode: relaxation, load_steps: 4, tolerance: 1.0e-3, n_steps: 50}
output:
  path: out.h5
  monitors:
    tip: {kind: displacement, region: head, dof: z}
"""


def _write_case(tmpdir, fy_list, first_bar_is_bilinear=False, confined=False):
    import pathlib
    d = pathlib.Path(tmpdir)
    d.mkdir(parents=True, exist_ok=True)
    msh = d / "col.msh"
    n = 3
    nodes = "\n".join(f"{i+1} {i*1.2/n:.10g} 0 0" for i in range(n + 1))
    elems = "\n".join(f"{e+1} 1 2 1 1 {e+1} {e+2}" for e in range(n))
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n"
                   f"{n+1}\n{nodes}\n$EndNodes\n$Elements\n{n}\n{elems}\n"
                   "$EndElements\n", encoding="utf-8")
    coords = [(0.138, 0.138), (-0.138, 0.138), (0.138, -0.138), (-0.138, -0.138)]
    steels, secs, bars = [], [], []
    for i, ((y, z), fy) in enumerate(zip(coords, fy_list)):
        if i == 0 and first_bar_is_bilinear:
            # `rebar` is a different law TYPE, and has no scalar twin
            steels.append(f"  steel{i}: {{type: rebar, E: 200.0e9, yield_strength: 474.0e6, "
                          "hardening: 1.0e9, rho: 7850}")
        else:
            steels.append(f"  steel{i}: {{type: steel02, E: 200.0e9, Fy: {fy}, b: 0.012, R0: 20.0, rho: 7850}}")
        secs.append(f"  bar{i}: {{type: bar, material: steel{i}, shape: general, area: 3.1416e-4}}")
        bars.append(f"      - {{y: {y}, z: {z}, section: bar{i}}}")
    text = (SECTION.replace("MESH", str(msh)).replace("STEELS", "\n".join(steels))
            .replace("BARSECTIONS", "\n".join(secs)).replace("BARS", "\n".join(bars)))
    if confined:
        text = text.replace("NZ", "8").replace("CONFINED", "    cover: 0.052\n    core_material: core")
    else:
        text = text.replace("NZ", "2").replace("CONFINED\n", "")
    yaml_p = d / "case.yaml"
    yaml_p.write_text(text, encoding="utf-8")
    return str(yaml_p)


def _solver(path, kernel):
    prep = prepare_case(path)
    problem = prep.case.model_dump(mode="python")
    sol = FiberBeamSolver(problem, prep.mesh, prep.material)
    if not kernel:
        sol._kernel = None
    return sol


def test_kernel_matches_the_numpy_reference_over_a_history(tmp_path):
    """Path dependent, so walk a history and compare at every step.

    Force, element actions and the fibre state are all checked: the state is
    what the next step reads, so two paths can agree on force for a while and
    still be diverging underneath.
    """
    path = _write_case(tmp_path, [474.0e6] * 4)
    fast = _solver(path, kernel=True)
    ref = _solver(path, kernel=False)
    assert fast._kernel is not None, fast.kernel_status

    rng = np.random.default_rng(7)
    worst_force = worst_actions = worst_state = 0.0
    for _ in range(40):
        dq = rng.normal(0.0, 3e-4, fast.ndof)
        fast.q[:] += dq
        ref.q[:] += dq
        f_fast, a_fast = fast.internal_force()
        f_ref, a_ref = ref.internal_force()
        worst_force = max(worst_force, np.max(np.abs(f_fast - f_ref))
                          / max(np.max(np.abs(f_ref)), 1.0))
        worst_actions = max(worst_actions, np.max(np.abs(a_fast - a_ref))
                            / max(np.max(np.abs(a_ref)), 1.0))
        s_fast = np.concatenate([s.ravel() for s in fast.fiber_states])
        s_ref = np.concatenate([s.ravel() for s in ref.fiber_states])
        worst_state = max(worst_state, np.max(np.abs(s_fast - s_ref))
                          / max(np.max(np.abs(s_ref)), 1e-30))

    assert worst_force < 1e-10, f"internal force drifted {worst_force:.2e}"
    assert worst_actions < 1e-10, f"element actions drifted {worst_actions:.2e}"
    assert worst_state < 1e-10, f"fibre state drifted {worst_state:.2e}"


def test_two_rebar_grades_stay_on_the_kernel_and_match_the_reference(tmp_path):
    """Different parameters are not a different law.

    The kernel inlines one law TYPE per fibre family; each fibre carries its own
    parameter row, so a section with two steel grades is an ordinary case. It
    was not always: the first version compared `numba_params()` across rebars
    and fell back whenever they differed, which would have pushed a hinge
    element with its own slip length -- or a cover concrete weaker than the
    core -- off the fast path for no reason.
    """
    path = _write_case(tmp_path, [474.0e6, 474.0e6, 335.0e6, 335.0e6])
    fast = _solver(path, kernel=True)
    ref = _solver(path, kernel=False)
    assert fast._kernel is not None, fast.kernel_status
    assert "2 rebar parameter sets" in fast.kernel_status

    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(25):
        dq = rng.normal(0.0, 3e-4, fast.ndof)
        fast.q[:] += dq
        ref.q[:] += dq
        f1, _ = fast.internal_force()
        f2, _ = ref.internal_force()
        worst = max(worst, np.max(np.abs(f1 - f2)) / max(np.max(np.abs(f2)), 1.0))
    assert worst < 1e-10, f"mixed grades drifted from the reference {worst:.2e}"


def test_the_two_grades_actually_change_the_answer(tmp_path):
    """Guard against the parameter rows being broadcast from rebar 0.

    If `spar` were filled from the first rebar and reused for the rest, the
    test above would still pass -- the reference path would be asked the same
    wrong question. This one checks that 335 MPa bars carry less than 474 MPa
    bars, i.e. that the per-fibre rows are reaching the right fibres.
    """
    same = _solver(_write_case(tmp_path / "a", [474.0e6] * 4), kernel=True)
    mixed = _solver(_write_case(tmp_path / "b", [474.0e6, 474.0e6, 335.0e6, 335.0e6]),
                    kernel=True)
    for sol in (same, mixed):
        sol.q[:] = 0.0
        sol.q[6 + 2] = 0.02              # push the free node hard enough to yield
        sol.internal_force()
    f_same = np.max(np.abs(same.internal_force()[0]))
    f_mixed = np.max(np.abs(mixed.internal_force()[0]))
    assert f_mixed < f_same, (
        f"weaker bars gave the same or more force ({f_mixed:.4e} vs {f_same:.4e}); "
        "the per-fibre parameter rows are probably all coming from rebar 0")


def test_kernel_declines_a_second_law_type(tmp_path):
    """Two law TYPES cannot be two inlined point functions -- the real limit.

    Parameters may differ freely; the law itself may not, because each law is
    compiled into the kernel as a separate inlined function and the choice
    cannot be made per fibre at run time.
    """
    path = _write_case(tmp_path, [474.0e6] * 4, first_bar_is_bilinear=True)
    sol = _solver(path, kernel=True)
    assert sol._kernel is None, f"expected a fallback, got {sol.kernel_status}"
    assert "type" in sol.kernel_status or "scalar twin" in sol.kernel_status, \
        sol.kernel_status
    # and declining the kernel must not decline the analysis
    force, actions = sol.internal_force()
    assert np.all(np.isfinite(force)) and np.all(np.isfinite(actions))


def test_kernel_state_survives_a_checkpoint_round_trip(tmp_path):
    """The consolidated rebar array is aliased by fiber_states as views.

    restore() rebinds fiber_states to fresh copies, which breaks that aliasing
    unless _relink_kernel_states puts it back. If it does not, the kernel then
    reads a stale array and the run silently continues from the wrong history.
    """
    path = _write_case(tmp_path, [474.0e6] * 4)
    sol = _solver(path, kernel=True)
    rng = np.random.default_rng(3)
    sol.q[:] += rng.normal(0.0, 5e-4, sol.ndof)
    sol.internal_force()

    ck = sol.checkpoint()
    before = sol.internal_force()[0].copy()

    sol.q[:] += rng.normal(0.0, 5e-4, sol.ndof)
    sol.internal_force()
    sol.restore(ck)
    after = sol.internal_force()[0]

    assert np.max(np.abs(after - before)) / max(np.max(np.abs(before)), 1.0) < 1e-12
    # and the aliasing is genuinely back, not merely consistent by luck
    sol._state_s[0, 0, 0, 0] = 12345.0
    assert sol.fiber_states[1][0, 0, 0, 0] == 12345.0


def test_confined_core_matches_the_reference_over_a_history(tmp_path):
    """Cover and core are two concrete groups; the kernel folds them into one.

    The reference path keeps them as separate material groups because one
    `update` call takes one parameter set. The kernel keeps a single inlined
    concrete law and reads `cpar[e, f]`, so the two consolidations have to
    agree fibre by fibre -- including which fibre got which parameter row.
    """
    path = _write_case(tmp_path, [474.0e6] * 4, confined=True)
    fast = _solver(path, kernel=True)
    ref = _solver(path, kernel=False)
    assert fast._kernel is not None, fast.kernel_status
    assert "cover + confined core" in fast.kernel_status

    rng = np.random.default_rng(19)
    worst_force = worst_state = 0.0
    for _ in range(30):
        dq = rng.normal(0.0, 4e-4, fast.ndof)
        fast.q[:] += dq
        ref.q[:] += dq
        f1, _ = fast.internal_force()
        f2, _ = ref.internal_force()
        worst_force = max(worst_force,
                          np.max(np.abs(f1 - f2)) / max(np.max(np.abs(f2)), 1.0))
        s1 = np.concatenate([s.ravel() for s in fast.fiber_states])
        s2 = np.concatenate([s.ravel() for s in ref.fiber_states])
        worst_state = max(worst_state,
                          np.max(np.abs(s1 - s2)) / max(np.max(np.abs(s2)), 1e-30))
    assert worst_force < 1e-10, f"force drifted {worst_force:.2e}"
    assert worst_state < 1e-10, f"state drifted {worst_state:.2e}"


def test_cover_fibres_are_the_outer_ones_and_carry_the_weaker_concrete(tmp_path):
    """Geometry, not order: the classification has to follow the coordinates.

    A version that split the fibres by index rather than by position would pass
    every equivalence check above -- both paths would be asked the same wrong
    question -- so this one checks the actual assignment.
    """
    path = _write_case(tmp_path, [474.0e6] * 4, confined=True)
    sol = _solver(path, kernel=True)
    assert sol.n_conc_groups == 2
    cover_idx, core_idx = sol.group_indices[0], sol.group_indices[1]
    half_w, half_h, cover = 0.2, 0.2, 0.052
    for i in cover_idx:
        assert (abs(sol.fiber_y[0, i]) > half_w - cover
                or abs(sol.fiber_z[0, i]) > half_h - cover), \
            f"fibre {i} at ({sol.fiber_y[0, i]:.3f}, {sol.fiber_z[0, i]:.3f}) is inside the core"
    for i in core_idx:
        assert (abs(sol.fiber_y[0, i]) <= half_w - cover
                and abs(sol.fiber_z[0, i]) <= half_h - cover)
    # and the parameter rows follow: cover fibres get the weaker peak
    fpc_cover = abs(sol._cpar[0, 0, 0])
    fpc_core = abs(sol._cpar[0, len(cover_idx), 0])
    assert fpc_core > fpc_cover * 1.4, (
        f"core peak {fpc_core/1e6:.1f} MPa is not above cover {fpc_cover/1e6:.1f} MPa; "
        "the parameter rows are not following the cover/core split")


def test_a_cover_that_swallows_the_section_is_refused(tmp_path):
    """Silently producing an all-cover section would be worse than failing."""
    import pathlib
    path = _write_case(tmp_path, [474.0e6] * 4, confined=True)
    text = pathlib.Path(path).read_text(encoding="utf-8").replace("cover: 0.052", "cover: 0.19")
    pathlib.Path(path).write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="no core fibres"):
        _solver(path, kernel=True)


def test_a_rigidly_moved_and_twisted_fibre_beam_carries_no_force(tmp_path):
    """The member rotated by a large skew rotation with every point's
    orientation the same rotation: no force on either path.
    Until then the global displacement was rotated into the chord frame and
    the Hermite functions cancelled the rigid part only to third order in
    the member's rotation, and a twist about the axis was not removed at
    all."""
    from vfife.core.rotations import exp_map, log_map
    path = _write_case(tmp_path, [474.0e6] * 4)
    Q = exp_map(np.array([0.9, -0.6, 1.3]))
    for kernel in (False, True):
        s = _solver(path, kernel)
        X = s.mesh.nodes
        x = (Q @ X.T).T + np.array([0.3, -0.2, 0.5])
        q = np.zeros((s.nn, 6)); q[:, :3] = x - X; q[:, 3:] = log_map(Q)
        s.q = q.ravel(); s.orient = np.broadcast_to(Q, (s.nn, 3, 3)).copy()
        f, actions = s.internal_force()
        scale = float(np.abs(s.K).max()) if hasattr(s, "K") else 1e12
        assert np.abs(f).max() < 1e-9 * scale, kernel
        assert np.abs(actions).max() < 1e-9 * scale, kernel


HOMOGENEOUS = """
name: homogeneous_kernel_check
units: {length: m, force: N, stress: Pa}
mesh: {type: gmsh, path: MESH}
materials:
  s: {type: steel_cyclic, grade: Q235B, E: 2.06e11, rho: 7850.0}
sections:
  m: {type: fiber, material: s, shape: tube, diameter: 0.114, t: 0.003, nt: 12}
assign: [{region: all, section: m, orientation: [0, 0, 1]}]
regions:
  base: {box: [-0.01, 0.01, -0.01, 0.01, -0.01, 0.01]}
  head: {box: [1.19, 1.21, -0.01, 0.01, -0.01, 0.01]}
constraints:
  - {region: base, dofs: [1, 1, 1, 1, 1, 1]}
loads:
  - {region: head, vector: [0.0, 0.0, -2.0e3]}
solver: {backend: numpy, mode: relaxation, load_steps: 2, tolerance: 1.0e-3, n_steps: 50}
output: {path: out.h5, monitors: {tip: {kind: displacement, region: head, dof: z}}}
"""


def _write_homogeneous(tmpdir):
    import pathlib
    d = pathlib.Path(tmpdir)
    d.mkdir(parents=True, exist_ok=True)
    msh = d / "tube.msh"
    n = 3
    nodes = "\n".join(f"{i+1} {i*1.2/n:.10g} 0 0" for i in range(n + 1))
    elems = "\n".join(f"{e+1} 1 2 1 1 {e+1} {e+2}" for e in range(n))
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n"
                   f"{n+1}\n{nodes}\n$EndNodes\n$Elements\n{n}\n{elems}\n"
                   "$EndElements\n", encoding="utf-8")
    p = d / "tube.yaml"
    p.write_text(HOMOGENEOUS.replace("MESH", str(msh)), encoding="utf-8")
    return str(p)


def test_a_section_with_no_rebars_gets_the_kernel(tmp_path):
    """The easy case used to be the excluded one.

    `_setup_kernel` opened with `if not self.is_hybrid: return`, so every
    homogeneous model in the library -- a steel tube, a box, anything without
    rebars -- ran on the NumPy reference while every reinforced-concrete one
    got the compiled kernel. Backwards: the kernel inlines at most two law
    types, and this section needs one and no rebar arrays at all. Measured on
    a K8 dome (2736 elements, 20 fibres): 1.95
    steps/s on the reference against 97.7 with the kernel, which is the
    difference between 159 hours and 3.2.

    What this asserts is the CONDITION, not the speed: a silent fallback that
    comes back is a run that is fifty times slower with nothing to say why.
    """
    sol = _solver(_write_homogeneous(tmp_path), kernel=True)
    assert sol.is_hybrid is False, "this section has no rebars, by construction"
    assert sol._kernel is not None, sol.kernel_status
    assert "no rebars" in sol.kernel_status and "numba fused kernel" in sol.kernel_status
    # and the rebar half of the kernel really is empty, not merely unused
    assert sol._sy.shape == (sol.nel, 0) and sol._sa.shape == (sol.nel, 0)
    assert sol._state_s.shape[2] == 0
    assert sol._spar.shape[1] == 0


def test_the_homogeneous_kernel_matches_the_numpy_reference_over_a_history(tmp_path):
    """Same transcription check as the reinforced case, on one law and no bars.

    Path dependent -- steel_cyclic carries eleven state entries per fibre --
    so it is walked, not evaluated once, and the state is compared as well as
    the force: two paths can agree on force for a while and still be
    diverging underneath.
    """
    path = _write_homogeneous(tmp_path)
    fast = _solver(path, kernel=True)
    ref = _solver(path, kernel=False)
    assert fast._kernel is not None, fast.kernel_status
    assert ref._kernel is None

    rng = np.random.default_rng(11)
    worst_force = worst_actions = worst_state = 0.0
    for _ in range(40):
        dq = rng.normal(0.0, 3e-4, fast.ndof)
        fast.q[:] += dq
        ref.q[:] += dq
        f_fast, a_fast = fast.internal_force()
        f_ref, a_ref = ref.internal_force()
        worst_force = max(worst_force, np.max(np.abs(f_fast - f_ref))
                          / max(np.max(np.abs(f_ref)), 1.0))
        worst_actions = max(worst_actions, np.max(np.abs(a_fast - a_ref))
                            / max(np.max(np.abs(a_ref)), 1.0))
        s_fast = np.concatenate([s.ravel() for s in fast.fiber_states])
        s_ref = np.concatenate([s.ravel() for s in ref.fiber_states])
        worst_state = max(worst_state, np.max(np.abs(s_fast - s_ref))
                          / max(np.max(np.abs(s_ref)), 1e-30))
        assert np.all(np.isfinite(f_fast))

    assert worst_force < 1e-10, f"internal force drifted {worst_force:.2e}"
    assert worst_actions < 1e-10, f"element actions drifted {worst_actions:.2e}"
    assert worst_state < 1e-10, f"fibre state drifted {worst_state:.2e}"
