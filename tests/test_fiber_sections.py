"""Several fibre sections on one model.

`one fiber section per model in phase 1` was the sentence that stopped the
Fig. 17 end stiffeners of the archived Tube A, and stops any frame whose
columns are not its beams. Lifting it is a change to the input model, the
solver's fibre arrays and the kernel's indexing, and the thing that keeps it
honest is that a ONE-section model is the same arithmetic it always was: the
arrays become (n_elements, n_fibers), a single section is the same row on
every element, and every sum runs over the same values in the same order.
That was checked to the bit against the previous code before this file was
written; what this file pins is the multi-section side.

THE PADDING RULE. Sections with different fibre counts are padded to the
widest with fibres of zero area. A padded fibre costs one law evaluation and
contributes exactly nothing -- sig * 0.0 -- so a two-section model's element
forces equal the single-section models' element by element, to the bit. The
one place a padded fibre could speak is the fracture fraction, and it is
masked there and the fraction taken over the member's own count.
"""
import pathlib

import numpy as np
import pytest

from vfife.api import prepare_case
from vfife.core.fiber_beam_solver import FiberBeamSolver

STEEL = "  s: {type: steel_cyclic, grade: Q235B, E: 2.06e11, rho: 7850.0}"
THIN = "{type: fiber, material: s, shape: tube, diameter: 0.100, t: 0.003, nt: 12}"
THICK = "{type: fiber, material: s, shape: tube, diameter: 0.100, t: 0.006, nt: 20}"


def _mesh(td, n=4, dx=0.3):
    msh = pathlib.Path(td) / "m.msh"
    nodes = "\n".join(f"{i+1} {i*dx:.6f} 0 0" for i in range(n + 1))
    elems = "\n".join(f"{e+1} 1 2 1 1 {e+1} {e+2}" for e in range(n))
    msh.write_text(f"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n{n+1}\n{nodes}\n"
                   f"$EndNodes\n$Elements\n{n}\n{elems}\n$EndElements\n", encoding="utf-8")
    return msh


def _case(td, sections, assigns, extra_regions="", fracture=""):
    msh = _mesh(td)
    yp = pathlib.Path(td) / "c.yaml"
    yp.write_text(f"""
name: sections
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
{STEEL}
sections:
{sections}
assign:
{assigns}
regions:
  base:  {{box: [-0.01, 0.01, -1, 1, -1, 1]}}
  inner: {{box: [-0.01, 0.61, -1, 1, -1, 1]}}
  outer: {{box: [0.59, 1.21, -1, 1, -1, 1]}}
  head:  {{box: [1.19, 1.21, -1, 1, -1, 1]}}
{extra_regions}constraints:
  - {{region: base, dofs: [1,1,1,1,1,1]}}
loads:
  - {{region: head, vector: [0, 0, -1.0e3]}}
solver: {{backend: numpy, mode: relaxation, load_steps: 2, tolerance: 1.0e-3, n_steps: 50}}
output: {{path: out.h5, monitors: {{}}}}
""", encoding="utf-8")
    return str(yp)


def _two(td, **kw):
    return _case(td, f"  thin:  {THIN}\n  thick: {THICK}",
                 "  - {region: inner, section: thin,  orientation: [0, 0, 1]}\n"
                 "  - {region: outer, section: thick, orientation: [0, 0, 1]}", **kw)


def _one(td, section):
    return _case(td, f"  m: {section}",
                 "  - {region: all, section: m, orientation: [0, 0, 1]}")


def _solver(path, kernel=True):
    prep = prepare_case(path)
    sol = FiberBeamSolver(prep.case.model_dump(mode="python"), prep.mesh, prep.material)
    if not kernel:
        sol._kernel = None
    return sol


def test_each_member_gets_the_section_its_region_names(tmp_path):
    sol = _solver(_two(tmp_path))
    assert sol._section_of.tolist() == [0, 0, 1, 1]
    assert sol.n_fibers == 20
    assert sol.fiber_valid.sum(axis=1).tolist() == [12, 12, 20, 20]
    # the padded fibres are zero area and nothing else
    assert np.all(sol.fiber_area[0, 12:] == 0.0)
    assert np.all(sol.fiber_area[2] > 0.0)
    # per-element section properties: a 3 mm and a 6 mm wall of the same tube
    assert sol.total_area[0] == pytest.approx(np.pi / 4 * (0.1 ** 2 - 0.094 ** 2), rel=1e-12)
    assert sol.total_area[2] == pytest.approx(np.pi / 4 * (0.1 ** 2 - 0.088 ** 2), rel=1e-12)
    assert "no rebars" in sol.kernel_status


def test_a_two_section_model_is_the_two_one_section_models_element_by_element(tmp_path):
    """The padding contributes nothing: force, actions AND state, both paths.

    Element e of the two-section model must equal element e of the
    single-section model built with e's section. On the KERNEL that is to the
    bit: its sums are sequential scalar loops, and adding exact zeros to a
    running sum changes nothing. On the NumPy reference it is to 1e-13:
    `np.sum` is pairwise, and the pairing tree of a 20-long row is not the
    tree of a 12-long one, so the same twelve non-zero terms round differently
    once eight zeros are appended. That is numpy's summation order, not the
    physics, and it is why the reference tolerance is stated rather than
    tightened to zero.
    """
    for kernel in (True, False):
        def same(a, b):
            if kernel:
                return np.array_equal(a, b)
            return np.allclose(a, b, rtol=1e-13, atol=1e-13 * max(1.0, float(np.abs(b).max())))
        two = _solver(_two(tmp_path), kernel)
        thin = _solver(_one(tmp_path, THIN), kernel)
        thick = _solver(_one(tmp_path, THICK), kernel)
        rng = np.random.default_rng(7)
        for _ in range(20):
            dq = rng.normal(0.0, 3e-4, two.ndof)
            for s in (two, thin, thick):
                s.q[:] += dq
            f2, a2 = two.internal_force()
            f_thin, a_thin = thin.internal_force()
            f_thick, a_thick = thick.internal_force()
            for e in range(two.nel):
                ref_a = a_thin[e] if two._section_of[e] == 0 else a_thick[e]
                ref_end = (thin if two._section_of[e] == 0 else thick).end_actions[e]
                assert same(a2[e], ref_a), (kernel, e, a2[e], ref_a)
                assert same(two.end_actions[e], ref_end), (kernel, e)
            # and the live fibre state: each section owns its material group,
            # so element e's state is read from ITS section's
            # group, whose width is the section's own
            for e in range(two.nel):
                k = int(two._section_of[e])
                st2 = two.fiber_states[two._layouts[k].conc_groups[0]]
                ref = (thin if k == 0 else thick).fiber_states[0][e]
                assert same(st2[e], ref)


def test_the_kernel_matches_the_reference_with_two_sections(tmp_path):
    fast, ref = _solver(_two(tmp_path), True), _solver(_two(tmp_path), False)
    assert fast._kernel is not None, fast.kernel_status
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(30):
        dq = rng.normal(0.0, 3e-4, fast.ndof)
        fast.q[:] += dq; ref.q[:] += dq
        ff, af = fast.internal_force(); fr, ar = ref.internal_force()
        worst = max(worst, np.max(np.abs(ff - fr)) / max(np.max(np.abs(fr)), 1.0),
                    np.max(np.abs(af - ar)) / max(np.max(np.abs(ar)), 1.0))
    assert worst < 1e-10, worst


def test_a_section_may_carry_its_own_orientation(tmp_path):
    path = _case(tmp_path, f"  thin:  {THIN}\n  thick: {THICK}",
                 "  - {region: inner, section: thin,  orientation: [0, 0, 1]}\n"
                 "  - {region: outer, section: thick, orientation: [0, 1, 0]}")
    sol = _solver(path)
    assert np.allclose(sol.R0[0][1], [0, 0, 1]) and np.allclose(sol.R0[3][1], [0, 1, 0])


def test_the_fracture_fraction_is_out_of_the_members_own_fibres(tmp_path):
    """A padded fibre cannot break, and 12 of 12 is 1.0, not 12 of 20."""
    from vfife.core import fracture
    sol = _solver(_two(tmp_path, fracture=""))
    # wire fracture by hand on the same layout the solver would build
    spec = dict(sol._elem_spec); spec["fracture"] = {"threshold": 1.0}
    sol._setup_fracture(spec)
    assert sol.fracture is not None
    assert sol.fracture.latch.over is not None
    assert sol.fracture.latch.over.tolist() == [12.0, 12.0, 20.0, 20.0]
    # the tracked columns are both sections' groups side by side (12 + 20);
    # a column is live on its own section's rows only
    assert sol._tracked_valid.shape == (sol.nel, 32, 1)
    assert sol._tracked_valid[:, :, 0].sum(axis=1).tolist() == [12.0, 12.0, 20.0, 20.0]
    # every fibre of element 0 reports failure, the other section's columns
    # and the padded ones included -- the mask keeps only its own 12
    pf = np.ones((sol.nel, sol.fracture.layout.points_per_element))
    pf[1:] = 0.0
    fac = fracture.fraction(pf * np.repeat(sol._tracked_valid, len(sol.gauss_s),
                                           axis=2).reshape(sol.nel, -1),
                            sol.fracture.layout, over=sol.fracture.latch.over)
    assert fac[0] == pytest.approx(1.0) and np.all(fac[1:] == 0.0)


@pytest.mark.parametrize("what, needle", [
    ("integration", "`integration` differs"),
    ("overlap", "overlap"),
    ("uncovered", "lie in no fibre section"),
    ("all", "assigned to `all` alongside"),
])
def test_the_limits_are_refused_with_a_reason(tmp_path, what, needle):
    """Bars, cover/core and a second fibre law used to be refused here too;
    since v0.2 each section carries its own (tests/test_fiber_sections_rc.py)."""
    sections = f"  thin:  {THIN}\n  thick: {THICK}"
    assigns = ("  - {region: inner, section: thin}\n"
               "  - {region: outer, section: thick}")
    extra = ""
    if what == "integration":
        sections += "\n  thick: {type: fiber, material: s, shape: tube, diameter: 0.1, t: 0.006, integration: lobatto3}"
    elif what == "overlap":
        assigns = ("  - {region: inner, section: thin}\n"
                   "  - {region: wide, section: thick}")
        extra = "  wide: {box: [0.29, 1.21, -1, 1, -1, 1]}\n"
    elif what == "uncovered":
        assigns = ("  - {region: inner, section: thin}\n"
                   "  - {region: head_only, section: thick}")
        extra = "  head_only: {box: [0.89, 1.21, -1, 1, -1, 1]}\n"
    elif what == "all":
        assigns = ("  - {region: all, section: thin}\n"
                   "  - {region: outer, section: thick}")
    path = _case(tmp_path, sections, assigns, extra_regions=extra)
    with pytest.raises(ValueError) as excinfo:
        _solver(path)
    assert needle in str(excinfo.value), str(excinfo.value)


# --------------------------------------------------------------------------- #
#  torsion is the section's, not the first section's
# --------------------------------------------------------------------------- #
#  v0.2 gave every fibre section its own material and the fibre laws honoured
#  it. Torsion is not a fibre quantity -- there is no fibre law to ask for it --
#  so `G J` kept reading `self.material`, which is the FIRST fibre section's
#  cover law. `self.J` was already per element; only the modulus was not. The
#  symptom is the sharpest one a defect can have: THE SAME MODEL, written with
#  its two sections in the other order, gave every member a different torque.
ELASTIC = ("  c: {type: linear_elastic, E: 3.0e10, nu: 0.2, rho: 2400.0}\n"
           "  s: {type: linear_elastic, E: 2.1e11, nu: 0.3, rho: 7850.0}")
RECT_C = "{type: fiber, material: c, shape: rect, width: 0.4, height: 0.4, ny: 4, nz: 4}"
RECT_S = "{type: fiber, material: s, shape: rect, width: 0.4, height: 0.4, ny: 4, nz: 4}"


def _elastic_case(td, sections, assigns):
    pathlib.Path(td).mkdir(parents=True, exist_ok=True)
    path = pathlib.Path(_case(str(td), sections, assigns))
    path.write_text(path.read_text(encoding="utf-8").replace(STEEL, ELASTIC), encoding="utf-8")
    return str(path)


def _twisted(path, rate=1e-4):
    from scipy.spatial.transform import Rotation
    prep = prepare_case(path)
    sol = FiberBeamSolver(prep.case.model_dump(mode="python"), prep.mesh, prep.material)
    q = sol.q.reshape(-1, 6)
    for n in range(sol.nn):
        ang = rate * sol.mesh.nodes[n, 0]
        q[n, 3] = ang
        sol.orient[n] = Rotation.from_rotvec([ang, 0, 0]).as_matrix()
    sol.internal_force()
    return sol, sol.end_actions[:, 9].copy()


def test_the_same_model_written_in_either_section_order_twists_the_same(tmp_path):
    """inner = concrete, outer = steel, written twice with the sections in
    opposite order. Before this fix the two files disagreed by 6.46x on EVERY
    member."""
    a = _twisted(_elastic_case(tmp_path / "ca", f"  a: {RECT_C}\n  b: {RECT_S}",
                               "  - {region: inner, section: a}\n"
                               "  - {region: outer, section: b}"))[1]
    b = _twisted(_elastic_case(tmp_path / "sc", f"  a: {RECT_S}\n  b: {RECT_C}",
                               "  - {region: inner, section: b}\n"
                               "  - {region: outer, section: a}"))[1]
    assert a == pytest.approx(b, rel=1e-14)


def test_each_member_twists_on_its_own_shear_modulus(tmp_path):
    """G = E / 2(1 + nu) of the member's OWN section, against the closed form."""
    sol, Mt = _twisted(_elastic_case(
        tmp_path, f"  a: {RECT_C}\n  b: {RECT_S}",
        "  - {region: inner, section: a}\n  - {region: outer, section: b}"))
    g_c, g_s = 3.0e10 / (2 * 1.2), 2.1e11 / (2 * 1.3)
    assert sol.member_G[:2] == pytest.approx([g_c, g_c], rel=1e-14)
    assert sol.member_G[2:] == pytest.approx([g_s, g_s], rel=1e-14)
    assert Mt[:2] == pytest.approx(g_c * sol.J[:2] * 1e-4, rel=1e-10)
    assert Mt[2:] == pytest.approx(g_s * sol.J[2:] * 1e-4, rel=1e-10)


def test_one_section_still_reads_the_model_material_exactly(tmp_path):
    """The input model writes a section's `material` key only when it differs
    from the first section's, so a single-material model -- which every
    committed fibre case is -- reads the same number it always did."""
    sol, _ = _twisted(_elastic_case(tmp_path, f"  a: {RECT_C}",
                                    "  - {region: all, section: a}"))
    assert np.all(sol.member_E == float(sol.material.E))
    assert np.all(sol.member_nu == float(sol.material.nu))
    assert np.array_equal(sol._G0J0, sol.material.E / (2 * (1 + sol.material.nu)) * sol.J)
