"""Particle splitting: the fragment leaves, and takes its mass with it.

v0.1 stopped one step short of a collapse analysis and said so in its own
docstring -- a fractured member stops carrying load, but "a beam cut in the
middle of a frame does not fall out of it". This file asserts the step that
finishes it, and every assertion here is a CONSERVATION LAW rather than a
number, because that is what a topology change can get wrong:

    the mass does not change      -- it moves from one particle to another
    the momentum does not change  -- the mass that moves keeps its velocity
    components only increase      -- a split separates, it never rejoins
    the untouched members do not notice -- bit for bit, not to a tolerance

The fourth is the one that catches a whole class of mistakes at once: if the
degrees of freedom of members that have nothing to do with the fracture move
by so much as an ulp, the split has renumbered something it should not have.
"""

import pathlib
import tempfile

import numpy as np
import pytest

from vfife.core import topology
from vfife.core.diagnostics import UserError


# --------------------------------------------------------------------------- #
#  the graph on its own
# --------------------------------------------------------------------------- #

def _chain(n_elements, **kw):
    conn = np.array([[i, i + 1] for i in range(n_elements)])
    return topology.BondGraph(conn, n_elements + 1, **kw)


def test_a_chain_is_one_component_until_something_is_cut():
    g = _chain(5, capacity=10)
    assert g.n_components() == 1
    assert g.attachments().tolist() == [1, 2, 2, 2, 2, 1]


def test_cutting_a_member_out_of_the_middle_makes_three_components():
    """Two halves of the frame and the member itself. THREE, not two: the
    fragment is its own component because the dead member still ties its two
    new particles together -- that is what makes it a segment with a length
    and a mass rather than two loose points (core/topology.py)."""
    g = _chain(5, capacity=10)
    made = g.detach(2)
    assert [(k, p) for k, p, _new in made] == [(0, 2), (1, 3)]
    assert g.n_components() == 3
    labels = g.components()
    assert labels[0] == labels[1] == labels[2]          # the left half
    assert labels[3] == labels[4] == labels[5]          # the right half
    assert labels[6] == labels[7]                       # the fragment
    assert len({labels[0], labels[3], labels[6]}) == 3


def test_an_end_that_is_already_alone_keeps_its_particle():
    """The rule that keeps the particle count equal to the number of
    SEPARATIONS: a cantilever tip is attached to one member only, so it is
    already that fragment's own particle and duplicating it would leave a
    massless orphan behind."""
    g = _chain(5, capacity=10)
    made = g.detach(4)
    assert [(k, p) for k, p, _new in made] == [(0, 4)]
    assert g.n_created == 1
    assert g.n_components() == 2


def test_the_component_count_never_falls():
    g = _chain(6, capacity=20)
    counts = [g.n_components()]
    for e in (3, 0, 5, 1):
        g.detach(e)
        counts.append(g.n_components())
    assert counts == sorted(counts), counts
    assert counts[0] == 1 and counts[-1] > 1


def test_running_out_of_room_is_an_error_with_a_way_out():
    """Only the allocating graph can run out; the solver reserves slots and
    cannot (see the next test). The message still has to say what to do."""
    g = _chain(5, capacity=1)
    with pytest.raises(UserError) as excinfo:
        g.detach(2)                     # needs two, has room for one
    assert excinfo.value.code == "SPLIT_CAPACITY"
    assert "capacity" in excinfo.value.hint
    assert g.n_created == 1, "the end that fitted was still moved"


def test_reserved_slots_cannot_run_out():
    """What the solver uses: two particles per element, allocated before the
    run. An end can separate once, so two per element is not a budget that
    might be too small -- it is the upper bound, and it is never all used:
    breaking a chain member by member costs one new particle each time,
    because the previous break already left that end alone."""
    conn = np.array([[i, i + 1] for i in range(5)])
    reserved = 6 + 2 * np.arange(5)[:, None] + np.arange(2)[None, :]
    g = topology.BondGraph(conn, 6 + 10, reserved=reserved)
    for e in range(5):
        g.detach(e)
    assert g.n_created == 4             # the two chain ends were already alone
    assert g.n_components() == 5        # five loose members
    assert g.n_particles == 16          # unchanged: nothing was allocated


# --------------------------------------------------------------------------- #
#  the solver: what a split does to mass, momentum and everybody else
# --------------------------------------------------------------------------- #

def _case(td, n_elem=4, name="split", split=True, threshold=1.0, rupture=2.0e-3,
          mode="dynamic", tip_mass=None, gravity=False, total_time=1.0e-3):
    """A straight bar of `n_elem` fibre members along x, fixed at x = 0."""
    L = 1.0
    msh = pathlib.Path(td) / f"{name}.msh"
    nodes = "\n".join(f"{i+1} {i*L/n_elem:.6f} 0 0" for i in range(n_elem + 1))
    elems = "\n".join(f"{i+1} 1 2 1 1 {i+1} {i+2}" for i in range(n_elem))
    msh.write_text(f"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n{n_elem+1}\n"
                   f"{nodes}\n$EndNodes\n$Elements\n{n_elem}\n{elems}\n$EndElements\n", encoding="utf-8")
    solver = (f"{{backend: numpy, mode: {mode}, n_steps: 10, "
              f"total_time: {total_time}, damping: {{type: viscous, coeff: 0.0}}}}"
              if mode == "dynamic" else
              f"{{backend: numpy, mode: {mode}, n_steps: 10}}")
    masses = ("nodal_masses: [{region: tip, mass: %g, per_node: true}]\n" % tip_mass
              if tip_mass else "")
    loads = ("loads: [{region: all, gravity: [0, 0, -9.81], amplitude: hold}]\n"
             if gravity else "")
    yp = pathlib.Path(td) / f"{name}.yaml"
    yp.write_text(f"""
name: {name}
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {str(msh).replace(chr(92), "/")}}}
amplitudes: {{hold: [[0, 1], [1, 1]]}}
materials:
  steel: {{type: rebar, E: 200.0e9, yield_strength: 4.0e8, hardening: 2.0e9,
           rho: 7850.0, rupture_strain: {rupture}}}
sections:
  rc: {{type: fiber, material: steel, shape: rect, width: 0.05, height: 0.05,
        ny: 3, nz: 3,
        fracture: {{threshold: {threshold}, split: {str(bool(split)).lower()}}}}}
assign: [{{region: all, section: rc}}]
regions:
  base: {{box: [-0.01, 0.01, -0.01, 0.01, -0.01, 0.01]}}
  tip: {{box: [{L-0.01}, {L+0.01}, -0.01, 0.01, -0.01, 0.01]}}
{masses}constraints: [{{region: base, dofs: [1,1,1,1,1,1]}}]
{loads}solver: {solver}
output: {{monitors: {{}}}}
""", encoding="utf-8")
    return str(yp)


def _solver(yaml_path):
    from vfife.api import prepare_case, _with_element
    from vfife.core.structural import make_structural_solver

    prep = prepare_case(yaml_path)
    assert prep.preflight["status"] == "PASS", prep.preflight
    p = _with_element(prep.case, prep.element, prep.case_path)
    return make_structural_solver(p, prep.mesh, prep.material)


def _fracture(solver, element, t=0.0):
    """Break `element` by hand and apply the split, without asking a material.

    The material's own route is exercised end to end at the bottom of this
    file; here what is under test is the topology change, and driving it
    directly keeps a failed assertion about mass from being a statement about
    a bar law's rupture strain."""
    solver.fracture.latch.fractured[element] = True
    solver.fracture.latch.n_fractured = int(solver.fracture.latch.fractured.sum())
    solver.fracture.active[element] = False
    solver._active[element] = 0
    solver._queue_split(np.array([element]), t)
    return solver.apply_pending_splits(t)


def test_two_spare_particles_per_element_are_reserved_and_park_on_their_end():
    """Nothing is sized at run time, and a spare sits exactly on the particle
    it may replace -- so the element's undeformed geometry does not change
    when a bond moves onto it."""
    with tempfile.TemporaryDirectory() as td:
        s = _solver(_case(td, n_elem=4))
        assert s.nn == 5 + 2 * 4
        assert s.ndof == s.nn * 6
        for e, conn in enumerate(s.mesh.elements):
            for k in (0, 1):
                spare = s._spare_of[e, k]
                assert np.array_equal(s.mesh.nodes[spare], s.mesh.nodes[conn[k]])
        assert s.bonds.n_components() == 1


def test_a_spare_is_in_no_region_so_it_cannot_be_born_clamped():
    """The reason the reservation happens after the regions are resolved: a
    spare parked on the fixed base sits inside `base`'s box, and a fragment
    that inherited that constraint would never fall."""
    with tempfile.TemporaryDirectory() as td:
        s = _solver(_case(td, n_elem=4))
        for name, idx in s.mesh.regions.items():
            assert np.asarray(idx).max(initial=-1) < 5, name


def test_a_split_moves_mass_and_does_not_create_or_destroy_any():
    with tempfile.TemporaryDirectory() as td:
        s = _solver(_case(td, n_elem=4, name="mass"))
        pm = s.physical_mass.reshape(s.nn, 6)
        before = float(pm[:, :3].sum())
        rot_before = pm[:, 3:].copy()
        parent_before = pm[2, 0]

        assert _fracture(s, 1)
        assert float(pm[:, :3].sum()) == pytest.approx(before, rel=1e-15)
        # the rotary inertia: what moved with the member, and on the piece
        # beyond the break (debris, joined to no support) the three axes
        # made one, the largest -- never less, and nothing
        # changed on the side still standing but the share that left
        rshare = s._element_mass_share(1)[3:]
        assert np.array_equal(pm[0, 3:], rot_before[0])
        assert pm[1, 3:] == pytest.approx(rot_before[1] - rshare, rel=1e-14)
        assert (pm[3:5, 3:] >= rot_before[3:5] - 1e-30).all()
        for n in range(2, 5):
            assert np.ptp(pm[n, 3:]) == 0.0
        share = s._element_mass_share(1)[0]
        assert pm[2, 0] == pytest.approx(parent_before - share, rel=1e-14)
        # element 1's bonds now sit on particles of its own
        assert set(s.mesh.elements[1]) & set(range(5)) == set()
        assert pm[s.mesh.elements[1], 0].tolist() == pytest.approx([share, share])


def test_a_split_conserves_momentum_exactly():
    """The fragment takes the velocity it already had. Anything else -- a
    separation velocity, a zeroed one -- is an impulse the model invented."""
    with tempfile.TemporaryDirectory() as td:
        s = _solver(_case(td, n_elem=4, name="mom"))
        rng = np.random.default_rng(0)
        s.v[:5 * 6] = rng.normal(size=5 * 6)

        def momentum():
            m = s.physical_mass.reshape(s.nn, 6)[:, :3]
            v = s.dofs.view(s.v)[:, :3]
            return (m * v).sum(axis=0)

        before = momentum()
        assert _fracture(s, 2)
        assert np.allclose(momentum(), before, rtol=0, atol=1e-12)


def test_the_members_that_did_not_fracture_do_not_move_by_one_ulp():
    """Bit-identical, on purpose. A split renumbers degrees of freedom, and
    the only way to be sure it renumbered nothing it should not have is to
    require the untouched members' internal force to be the same object it
    would have been."""
    with tempfile.TemporaryDirectory() as td:
        yp = _case(td, n_elem=4, name="bits")
        a, b = _solver(yp), _solver(yp)
        rng = np.random.default_rng(1)
        q = np.zeros(a.ndof)
        q[:5 * 6] = 1e-4 * rng.normal(size=5 * 6)
        a.q, b.q = q.copy(), q.copy()

        # both deactivate element 1; only `a` splits it
        for s in (a, b):
            s.fracture.latch.fractured[1] = True
            s.fracture.active[1] = False
            s._active[1] = 0
        assert _fracture(a, 1)

        fa, aa = a.internal_force()
        fb, ab = b.internal_force()
        untouched = np.setdiff1d(np.arange(4), [1])
        assert np.array_equal(aa[untouched], ab[untouched])
        # the surviving members still push on the same original particles
        for e in untouched:
            assert np.array_equal(a.mesh.elements[e], b.mesh.elements[e])
            dof = a.dofs.element_dofs(a.mesh.elements[e])
            assert np.array_equal(fa[dof], fb[dof])


def test_the_step_bound_after_a_split_is_measured_not_assumed():
    """The hazard the graphics survey flagged, and what it actually does here.

    A split takes mass OFF a particle, and the explicit bound is
    dt <= sqrt(min m/k), so the fear is that the step collapses for a member
    that is not there any more. Rebuilding K over the surviving members is the
    accounting -- a dead element contributes neither mass nor stiffness -- and
    the measurement is that it recovers the bound COMPLETELY ON THE AXIAL
    DEGREE OF FREEDOM AND ONLY PARTLY ON THE ROTATIONAL ONE:

        parent's m/k, axial       6.133e-10 -> 6.133e-10   (exactly restored)
        parent's m/k, rotational  4.089e-11 -> 2.271e-11   (0.556, not 1.0)

    The reason is that a row sum of the ASSEMBLED |K| is not the sum of the
    two members' row sums: at a shared node the two members' bending
    couplings (6EI/L^2, opposite signs) largely cancel, so two members'
    rotational row is only 1.11x one member's while their mass is 2x. Take
    one away and half the mass goes with a tenth of the stiffness.

    So: rebuilding is necessary and is not always sufficient. On this bar the
    axial dof governs and the model-wide bound does not move at all; on a
    model whose rotational dof governs it would fall by up to sqrt(1/2), and
    the solver says so rather than integrating past it silently.
    """
    with tempfile.TemporaryDirectory() as td:
        s = _solver(_case(td, n_elem=4, name="dt"))
        stale_K = s.K.copy()
        bound_before = s.dt_bound
        dof = s.dofs.node_dofs(2).ravel()     # element 1's far particle

        def ratios(K, mass):
            row = np.asarray(abs(K).sum(axis=1)).ravel()[dof]
            r = mass[dof] / np.maximum(row, 1e-20)
            return float(r[0]), float(r[4])   # axial, bending about y

        mass0 = s.mass.copy()
        axial0, rot0 = ratios(stale_K, mass0)
        assert _fracture(s, 1)
        # the far particle is on debris now, and its rotary inertia about y
        # is the largest of its three: the claims below are
        # about the stiffness, so they are taken per unit of that mass
        lift = float(s.mass[dof[4]] / (0.5 * mass0[dof[4]]))
        assert lift >= 1.0
        axial_stale, rot_stale = ratios(stale_K, s.mass)
        axial_new, rot_new = ratios(s.K, s.mass)

        # not rebuilding: half the mass against all of the stiffness
        assert axial_stale == pytest.approx(0.5 * axial0, rel=1e-12)
        assert rot_stale == pytest.approx(0.5 * lift * rot0, rel=1e-12)
        # rebuilding: the axial ratio is exactly back, the rotational one is not
        assert axial_new == pytest.approx(axial0, rel=1e-12)
        assert rot_new == pytest.approx(0.556 * lift * rot0, rel=1e-2)
        # here the axial dof governs, so the run's own bound does not move
        assert s.dt_bound == pytest.approx(bound_before, rel=1e-12)


def test_a_bound_that_did_fall_is_said_out_loud():
    """dt is not adapted mid-run -- the time grid of an explicit run is fixed
    when it starts -- so the one thing that must not happen is a bound that
    silently stops holding."""
    with tempfile.TemporaryDirectory() as td:
        said = []
        s = _solver(_case(td, n_elem=4, name="loud"))
        s.log = said.append
        s.dt = 10.0 * s.dt_bound                 # a step this model cannot take
        assert _fracture(s, 1)
        assert any("exceeds the stability bound" in line for line in said), said


def test_splitting_is_refused_in_a_relaxation_run():
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(UserError) as excinfo:
            _solver(_case(td, n_elem=2, name="relax", mode="relaxation"))
        assert excinfo.value.code == "SPLIT_NEEDS_DYNAMIC"
        assert "dynamic" in excinfo.value.hint


def _falling_case(td, push=0.05, threshold=0.34, total_time=0.01):
    """Push a two-member cantilever until its base member ruptures, then let go.

    Two steps, and the tip constraint lives in the FIRST one only: step 1
    bends the bar past the bar law's rupture strain, step 2 has nothing but
    gravity. What is left standing after the fracture is a dead member hanging
    off the support; what comes away is the outer member, and the point of the
    case is that it is now a body of its own.
    """
    L, n = 1.0, 2
    msh = pathlib.Path(td) / "fall.msh"
    nodes = "\n".join(f"{i+1} {i*L/n:.6f} 0 0" for i in range(n + 1))
    elems = "\n".join(f"{i+1} 1 2 1 1 {i+1} {i+2}" for i in range(n))
    msh.write_text(f"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n{n+1}\n"
                   f"{nodes}\n$EndNodes\n$Elements\n{n}\n{elems}\n$EndElements\n", encoding="utf-8")
    yp = pathlib.Path(td) / "fall.yaml"
    yp.write_text(f"""
name: fall
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {str(msh).replace(chr(92), "/")}}}
amplitudes: {{hold: [[0, 1], [1, 1]], ramp: [[0, 0], [1, 1]]}}
materials:
  steel: {{type: rebar, E: 200.0e9, yield_strength: 4.0e8, hardening: 2.0e9,
           rho: 7850.0, rupture_strain: 1.0e-3}}
sections:
  rc: {{type: fiber, material: steel, shape: rect, width: 0.05, height: 0.05,
        ny: 3, nz: 3, fracture: {{threshold: {threshold}, split: true}}}}
assign: [{{region: all, section: rc}}]
regions:
  base: {{box: [-0.01, 0.01, -0.01, 0.01, -0.01, 0.01]}}
  tip: {{box: [{L-0.01}, {L+0.01}, -0.01, 0.01, -0.01, 0.01]}}
constraints: [{{region: base, dofs: [1,1,1,1,1,1]}}]
loads: [{{region: all, gravity: [0, 0, -9.81], amplitude: hold}}]
steps:
  - name: push
    constraints: [{{region: tip, dofs: [0,0,1,0,0,0], value: [0, 0, -{push}],
                    amplitude: ramp}}]
  - name: fall
solver: {{backend: numpy, mode: dynamic, total_time: {total_time},
          damping: {{type: viscous, coeff: 0.0}}}}
output: {{monitors: {{}}}}
""", encoding="utf-8")
    return str(yp)


def test_the_member_comes_away_and_falls_at_g():
    """The end-to-end case, and the assertion is a physical one: after the
    base member ruptures, the centre of mass of what came away accelerates
    downward at 9.81 m/s^2.

    IT HAS TO BE THE CENTRE OF MASS. Individual particles keep vibrating --
    the fragment is a beam that was bent when it let go, and nothing damps it
    -- but the internal forces of a free body cancel in the sum, so its centre
    of mass does not care. Measured here: 9.806 m/s^2 against 9.81, the
    difference being the central-difference step, not the model.

    This is what v0.1 could not do and said so. Without the split the same
    run leaves the outer member hanging in the air on particles the frame
    still owns.
    """
    with tempfile.TemporaryDirectory() as td:
        s = _solver(_falling_case(td))
        total_mass = float(s.physical_mass.reshape(s.nn, 6)[:, 0].sum())
        assert s.bonds.n_components() == 1

        seen = []

        def watch(sol, _step, t, _actions):
            m = sol.physical_mass.reshape(sol.nn, 6)[:, 2]
            fragment = [1, 2]                       # the outer member's particles
            z = sol.dofs.view(sol.q)[fragment, 2]
            seen.append((t, float((m[fragment] * z).sum() / m[fragment].sum())))

        s.run(on_output=watch, output_every=100, progress_every=10 ** 9)

        assert s.fracture.latch.fractured.tolist() == [True, False]
        # both ends of the dead base member: the one on the clamped base
        # leaves it too, as a frame member's does (1.0 re-review)
        assert len(s.splits) == 2 and {sp[1] for sp in s.splits} == {0}
        assert s.bonds.n_components() == 2
        assert float(s.physical_mass.reshape(s.nn, 6)[:, 0].sum()) == pytest.approx(
            total_mass, rel=1e-14), "the run gained or lost weight"

        # the fragment carries the outer member and nothing of the dead one
        share = s._element_mass_share(1)[0]
        assert s.physical_mass.reshape(s.nn, 6)[1, 0] == pytest.approx(share, rel=1e-14)

        t, z = np.array(seen).T
        late = slice(-6, -1)                        # after the split, even spacing
        h = float(np.diff(t[late]).mean())
        assert np.allclose(np.diff(t[late]), h, rtol=1e-9), "uneven sampling"
        acc = (z[late][:-2] - 2 * z[late][1:-1] + z[late][2:]) / h ** 2
        assert np.allclose(acc, -9.81, rtol=5e-3), acc
        assert z[-1] < z[0], "the fragment did not move down at all"


def test_without_the_split_the_topology_never_changes():
    """v0.1's behaviour is still there and still the default: the member
    stops carrying load and its mass stays on the frame's particles."""
    with tempfile.TemporaryDirectory() as td:
        s = _solver(_case(td, n_elem=4, name="nosplit", split=False))
        assert s.bonds is None
        assert s.nn == 5
        before = s.mesh.elements.copy()
        s._active[1] = 0
        s.internal_force()
        assert np.array_equal(s.mesh.elements, before)
