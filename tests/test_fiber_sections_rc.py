"""Several REINFORCED fibre sections on one model.

v0.1 let a model carry several fibre sections but refused bars, a confined
core and a second fibre law with them, because the material GROUPS -- one
`update` per law per section -- were one list for the whole model. Now each
section owns its groups: cover, core, one per bar, and `self.materials` is
their concatenation. A column with eight bars and a Mander core next to a
beam with four bars and a different cover is one model.

The pin is the one v0.1 used: element e of the two-section model must equal
element e of the single-section model built with e's section -- forces,
end actions and fibre state -- to the bit on the kernel and to 1e-13 on the
NumPy reference (its pairwise sums see a different padded length).
"""
import pathlib

import numpy as np
import pytest

from vfife.api import prepare_case
from vfife.core.fiber_beam_solver import FiberBeamSolver

MATERIALS = """  cover: {type: concrete02, fpc: 3.0e7, epsc0: 0.002, fcu: 6.0e6, epscu: 0.005, ft: 2.0e6, Ets: 1.0e9, rho: 2400}
  weak:  {type: concrete02, fpc: 2.5e7, epsc0: 0.002, fcu: 5.0e6, epscu: 0.005, ft: 1.8e6, Ets: 1.0e9, rho: 2400}
  core:  {type: concrete02, fpc: 3.8e7, epsc0: 0.0047, fcu: 7.6e6, epscu: 0.02, ft: 2.0e6, Ets: 1.0e9, rho: 2400}
  s420:  {type: steel02, E: 2.0e11, Fy: 4.2e8, b: 0.01, rho: 7850}
  s500:  {type: steel02, E: 2.0e11, Fy: 5.0e8, b: 0.008, rho: 7850}"""

COLUMN = """{type: fiber, material: cover, shape: rect, width: 0.3, height: 0.4, ny: 4, nz: 8,
     cover: 0.04, core_material: core, integration: lobatto3,
     bars: [{y: 0.1, z: 0.16, section: b20}, {y: -0.1, z: 0.16, section: b20},
            {y: 0.1, z: -0.16, section: b20}, {y: -0.1, z: -0.16, section: b20},
            {y: 0.1, z: 0.0, section: b20}, {y: -0.1, z: 0.0, section: b20}]}"""
BEAM = """{type: fiber, material: weak, shape: rect, width: 0.3, height: 0.4, ny: 4, nz: 8,
     integration: lobatto3,
     bars: [{y: 0.1, z: 0.15, section: b16}, {y: -0.1, z: 0.15, section: b16},
            {y: 0.1, z: -0.15, section: b16}, {y: -0.1, z: -0.15, section: b16}]}"""
BARS = """  b20: {type: bar, material: s420, shape: general, area: 3.14e-4}
  b16: {type: bar, material: s500, shape: general, area: 2.01e-4}"""


def _mesh(td, n=4, dx=0.5):
    msh = pathlib.Path(td) / "m.msh"
    nodes = "\n".join(f"{i+1} {i*dx:.6f} 0 0" for i in range(n + 1))
    elems = "\n".join(f"{e+1} 1 2 1 1 {e+1} {e+2}" for e in range(n))
    msh.write_text(f"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n{n+1}\n{nodes}\n"
                   f"$EndNodes\n$Elements\n{n}\n{elems}\n$EndElements\n", encoding="utf-8")
    return msh


def _case(td, sections, assigns, fracture=""):
    msh = _mesh(td)
    yp = pathlib.Path(td) / "c.yaml"
    yp.write_text(f"""
name: rc_sections
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
{MATERIALS}
sections:
{BARS}
{sections}
assign:
{assigns}
regions:
  base:  {{box: [-0.01, 0.01, -1, 1, -1, 1]}}
  inner: {{box: [-0.01, 1.01, -1, 1, -1, 1]}}
  outer: {{box: [0.99, 2.01, -1, 1, -1, 1]}}
  head:  {{box: [1.99, 2.01, -1, 1, -1, 1]}}
  mid:   {{box: [0.99, 1.01, -1, 1, -1, 1]}}
constraints:
  - {{region: base, dofs: [1,1,1,1,1,1]}}
loads:
  - {{region: head, vector: [0, 0, -1.0e3]}}
solver: {{backend: numpy, mode: relaxation, load_steps: 2, tolerance: 1.0e-3, n_steps: 50}}
output: {{path: out.h5, monitors: {{}}}}
""", encoding="utf-8")
    return str(yp)


def _two(td, anchorage=""):
    return _case(td, f"  col:  {COLUMN}\n  beam: {BEAM}",
                 f"  - {{region: inner, section: col,  orientation: [0, 0, 1]{anchorage}}}\n"
                 "  - {region: outer, section: beam, orientation: [0, 0, 1]}")


def _one(td, section, anchorage=""):
    return _case(td, f"  m: {section}",
                 f"  - {{region: all, section: m, orientation: [0, 0, 1]{anchorage}}}")


def _solver(path, kernel=True):
    prep = prepare_case(path)
    sol = FiberBeamSolver(prep.case.model_dump(mode="python"), prep.mesh, prep.material)
    if not kernel:
        sol._kernel = None
    return sol


def test_each_section_owns_its_groups(tmp_path):
    sol = _solver(_two(tmp_path))
    col, beam = sol._layouts
    assert [m.name for m in sol.materials] == (["concrete02"] * 2 + ["steel02"] * 6
                                                + ["concrete02"] + ["steel02"] * 4)
    assert col.conc_groups == [0, 1] and col.rebar_groups == [2, 3, 4, 5, 6, 7]
    assert beam.conc_groups == [8] and beam.rebar_groups == [9, 10, 11, 12]
    assert col.n_c == 32 and col.n_s == 6 and beam.n_c == 32 and beam.n_s == 4
    assert sol.n_fibers == 38 and sol.n_steel == 6
    assert sol.fiber_valid.sum(axis=1).tolist() == [38, 38, 36, 36]
    # the beam's cover law is its own object with its own parameters
    assert abs(sol.materials[8].fc) == pytest.approx(2.5e7) and abs(sol.materials[0].fc) == pytest.approx(3.0e7)
    assert "2 sections" in sol.kernel_status, sol.kernel_status


def test_a_two_section_rc_model_is_the_two_single_section_models_element_by_element(tmp_path):
    for kernel in (True, False):
        def same(a, b):
            if kernel:
                return np.array_equal(a, b)
            return np.allclose(a, b, rtol=1e-13, atol=1e-13 * max(1.0, float(np.abs(b).max())))
        two = _solver(_two(tmp_path), kernel)
        col = _solver(_one(tmp_path, COLUMN), kernel)
        beam = _solver(_one(tmp_path, BEAM), kernel)
        assert (two._kernel is not None) == kernel, two.kernel_status
        rng = np.random.default_rng(7)
        for _ in range(20):
            dq = rng.normal(0.0, 3e-4, two.ndof)
            for s in (two, col, beam):
                s.q[:] += dq
            _f2, a2 = two.internal_force()
            _fc, ac = col.internal_force()
            _fb, ab = beam.internal_force()
            for e in range(two.nel):
                k = int(two._section_of[e])
                ref = col if k == 0 else beam
                assert same(a2[e], (ac if k == 0 else ab)[e]), (kernel, e)
                assert same(two.end_actions[e], ref.end_actions[e]), (kernel, e)
                lay, lay_ref = two._layouts[k], ref._layouts[0]
                for g, g_ref in zip(lay.conc_groups + lay.rebar_groups,
                                    lay_ref.conc_groups + lay_ref.rebar_groups):
                    assert same(two.fiber_states[g][e], ref.fiber_states[g_ref][e]), (kernel, e, g)


def test_the_kernel_matches_the_reference_with_two_rc_sections(tmp_path):
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


def test_a_sections_anchorage_reaches_its_own_members_only(tmp_path):
    """`mid` is the node both sections share: the column's anchorage scales
    the column element touching it and not the beam element touching it."""
    sol = _solver(_two(tmp_path, anchorage=", anchorage: {region: mid, bar_diameter: 0.02}"))
    assert sol.strain_penetration["elements"] == [1]
    scale = sol._spar[:, :, 8]
    assert np.all(scale[1, :6] < 1.0) and np.all(scale[[0, 2, 3]] == 1.0)
    # and the single-section column with the same anchorage scales both
    # elements touching the node, i.e. the general rule is unchanged
    one = _solver(_one(tmp_path, COLUMN, anchorage=", anchorage: {region: mid, bar_diameter: 0.02}"))
    assert one.strain_penetration["elements"] == [1, 2]


def test_padded_rebar_slots_carry_a_real_parameter_row(tmp_path):
    """The beam has 4 bars in a 6-slot array: slots 4 and 5 hold the first
    bar's parameters at zero area, so the kernel evaluates them harmlessly."""
    sol = _solver(_two(tmp_path))
    beam_rows = sol._rows[1]
    assert np.array_equal(sol._spar[beam_rows][:, 4, :], sol._spar[beam_rows][:, 0, :])
    assert np.all(sol._sa[beam_rows][:, 4:] == 0.0)
    assert np.isfinite(sol.internal_force()[0]).all()


def test_checkpoint_round_trip_keeps_every_sections_state(tmp_path):
    """restore() rebinds the state arrays and relinks the kernel views; with
    two sections the relink must copy each group onto its own rows only."""
    sol = _solver(_two(tmp_path))
    rng = np.random.default_rng(5)
    for _ in range(10):
        sol.q[:] += rng.normal(0.0, 3e-4, sol.ndof)
        sol.internal_force()
    f_before, _ = sol.internal_force()
    before = [s.copy() for s in sol.fiber_states]     # after the last evaluation
    ck = sol.checkpoint()
    for s in sol.fiber_states:
        s[...] = 0.0
    sol.restore(ck)
    # a group's state is exact on its own section's rows; the rows outside
    # its section are another section's business (or a padded slot, which
    # comes back as the zero state every law starts from and never carries
    # area, so nothing downstream can see it)
    for g, (a, b) in enumerate(zip(before, sol.fiber_states)):
        rows = sol._rows[int(sol.group_section[g])]
        assert np.array_equal(a[rows], b[rows]), g
    f_after, _ = sol.internal_force()
    assert np.array_equal(f_before, f_after)


def test_two_sections_refuse_anchorage_without_bars(tmp_path):
    plain = "{type: fiber, material: cover, shape: rect, width: 0.3, height: 0.4, ny: 4, nz: 8, integration: lobatto3}"
    path = _case(tmp_path, f"  col: {COLUMN}\n  beam: {plain}",
                 "  - {region: inner, section: col}\n"
                 "  - {region: outer, section: beam, anchorage: {region: mid, bar_diameter: 0.02}}")
    with pytest.raises(ValueError, match="anchorage but no bars"):
        _solver(path)
