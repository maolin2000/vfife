"""Region selectors, amplitude curves, and analysis steps.

These replaced a bounding box, one global load curve, and a single run. The
tests worth having are the ones that would pass anyway if the new machinery
merely *looked* right:

  * a selector expression must select the nodes its algebra says, checked
    against the coordinates rather than against a count;
  * splitting one analysis into two steps must reach the same answer, because
    if it does not then step state is not really being carried over;
  * a held load must still be held after the step that applied it;
  * a cyclic amplitude must produce a hysteresis loop with the reverse yield
    the material's kinematic hardening promises, and not a mirrored one.
"""

import numpy as np
import pytest

from vfife.core import amplitude as amp
from vfife.core import selectors as S
from vfife.core.mesh import box_mesh, surface_traction, surface_pressure, volume_body_force
from vfife.core.solver import Solver
from vfife.io.schema import Case, LoadSpec
from vfife.materials import create


def mesh():
    return box_mesh((1.2, 0.6, 0.4), (6, 3, 2))


# --------------------------------------------------------------------------- #
#  selectors
# --------------------------------------------------------------------------- #


def test_primitives_select_by_geometry():
    m = mesh()
    N = m.nodes
    r = S.resolve_all(N, {
        "base": {"plane": {"point": [0, 0, 0], "normal": [0, 0, 1]}},
        "slab": {"where": {"z": {"min": 0.2}}},
        "tube": {"cylinder": {"point": [0.6, 0.3, 0.0], "axis": [0, 0, 1],
                              "radius": 0.15}},
        "ball": {"sphere": {"centre": [0.6, 0.3, 0.2], "radius": 0.25}},
        "corner": {"nearest": [1.2, 0.6, 0.4]},
        "everything": {"all": True},
    })
    assert np.abs(N[r["base"], 2]).max() == 0.0
    assert N[r["slab"], 2].min() >= 0.2
    assert np.linalg.norm(N[r["tube"], :2] - [0.6, 0.3], axis=1).max() <= 0.15
    assert np.linalg.norm(N[r["ball"]] - [0.6, 0.3, 0.2], axis=1).max() <= 0.25
    assert r["corner"].size == 1
    assert np.allclose(N[r["corner"]][0], [1.2, 0.6, 0.4])
    assert r["everything"].size == m.n_nodes


def test_a_bare_number_in_where_pins_an_axis_and_line_selects_a_segment():
    """The support and load lines of every beam case were six-number boxes
    with hand-tuned 1e-3 tolerances (`[0.074, 0.126, -0.01, 0.21, -0.001,
    0.001]` for "the line x = 0.1 on the bottom face"). `{x: 0.1, z: 0}` is
    that line; three pinned axes are a point; `line` takes a segment that
    need not be axis-aligned and does not pick up the far end.

    On this 6x3x2 mesh of 1.2 x 0.6 x 0.4 the node lines are 0.2 apart in x
    and y and the bottom face has 7 x 4 nodes."""
    m = mesh()
    N = m.nodes
    r = S.resolve_all(N, {
        "support": {"where": {"x": 0.2, "z": 0}},
        "pin": {"where": {"x": 0.2, "y": 0.2, "z": 0}},
        "mixed": {"where": {"z": 0, "y": {"min": 0.3}}},
        "diag": {"line": {"from": [0.0, 0.0, 0.0], "to": [1.2, 0.6, 0.0]}},
        "short": {"line": {"from": [0.0, 0.0, 0.0], "to": [0.4, 0.0, 0.0]}},
    })
    assert r["support"].size == 4 and np.allclose(N[r["support"], 0], 0.2) \
        and np.allclose(N[r["support"], 2], 0.0)
    assert r["pin"].size == 1 and np.allclose(N[r["pin"]][0], [0.2, 0.2, 0.0])
    assert np.allclose(N[r["mixed"], 2], 0.0) and N[r["mixed"], 1].min() >= 0.3
    # the diagonal of the bottom face passes exactly through 4 nodes:
    # (0,0), (0.4,0.2), (0.8,0.4), (1.2,0.6)
    assert r["diag"].size == 4
    # a segment, not a line: 3 nodes on x = 0..0.4, not the 7 of the whole edge
    assert r["short"].size == 3 and N[r["short"], 0].max() <= 0.4


def test_boolean_algebra_and_forward_references():
    m = mesh()
    N = m.nodes
    r = S.resolve_all(N, {
        # `ends` is declared before the `right` it depends on
        "ends": {"union": ["left", "right"]},
        "left": {"where": {"x": {"max": 0.0}}},
        "right": {"where": {"x": {"min": 1.2}}},
        "base": {"plane": {"point": [0, 0, 0], "normal": [0, 0, 1]}},
        "left_base": {"intersect": ["left", "base"]},
        "base_only": {"difference": ["base", "left"]},
    })
    assert r["ends"].size == r["left"].size + r["right"].size
    assert np.abs(N[r["left_base"], 0]).max() == 0.0
    assert np.abs(N[r["left_base"], 2]).max() == 0.0
    assert np.intersect1d(r["base_only"], r["left"]).size == 0
    assert np.union1d(r["base_only"], r["left_base"]).size == r["base"].size


def test_surface_and_interior_partition_the_mesh():
    """A property with no tolerance in it: every node is inside or outside."""
    m = mesh()
    pre = {"surface": S.surface_nodes(m.elements, "hexa8", m.n_nodes)}
    r = S.resolve_all(m.nodes, {
        "all": {"all": True},
        "core": {"difference": ["all", "surface"]},
    }, pre)
    assert r["core"].size + pre["surface"].size == m.n_nodes
    assert np.intersect1d(r["core"], pre["surface"]).size == 0


@pytest.mark.parametrize("element", ["hexa8", "tet10"])
def test_surface_traction_is_area_consistent_and_not_node_count_dependent(element):
    """A 2 MPa top-face pressure must result in exactly p times area."""
    m = box_mesh((2.0, 3.0, 1.0), (2, 3, 1), element=element)
    top = np.flatnonzero(np.isclose(m.nodes[:, 2], 1.0))
    f = surface_traction(m.nodes, m.elements, element, top, [0.0, 0.0, -2e6])
    assert f.sum(axis=0) == pytest.approx([0.0, 0.0, -12e6], rel=1e-12)
    assert np.allclose(f[np.setdiff1d(np.arange(m.n_nodes), top)], 0.0)
    # Tet10's quadratic surface has its consistent load on the mid-side DOFs;
    # a uniform nodal-force approximation would fail this distinction.
    if element == "tet10":
        assert np.count_nonzero(np.linalg.norm(f[top], axis=1) > 1e-3) < top.size


@pytest.mark.parametrize("element", ["hexa8", "tet10"])
def test_positive_pressure_is_inward_and_area_consistent(element):
    m = box_mesh((2.0, 3.0, 1.0), (2, 3, 1), element=element)
    top = np.flatnonzero(np.isclose(m.nodes[:, 2], 1.0))
    f = surface_pressure(m.nodes, m.elements, element, top, 2e6)
    assert np.allclose(f.sum(axis=0), [0.0, 0.0, -12e6], rtol=1e-12, atol=1e-8)


@pytest.mark.parametrize("element", ["hexa8", "tet10"])
def test_body_force_is_volume_consistent(element):
    m = box_mesh((2.0, 3.0, 1.0), (2, 3, 1), element=element)
    mat = create("linear_elastic", E=2e11, nu=0.3, rho=1.0)
    p = {"element": {"type": element, "formulation": "full"},
         "solver": {"mode": "relaxation", "backend": "numpy", "n_steps": 1},
         "material": {"type": "linear_elastic", "E": 2e11, "nu": 0.3, "rho": 1.0}}
    s = Solver(p, m, mat, log=lambda *_: None)
    f = volume_body_force(m.nodes, m.elements, np.arange(m.n_nodes),
                          s.pre.shape_gp, s.pre.w_detJ, [0.0, 0.0, -4.0])
    assert f.sum(axis=0) == pytest.approx([0.0, 0.0, -24.0], rel=1e-12)


def test_solver_maps_pressure_body_force_and_gravity_to_consistent_loads():
    m = box_mesh((2.0, 3.0, 1.0), (2, 3, 1))
    top = np.flatnonzero(np.isclose(m.nodes[:, 2], 1.0))
    m.regions["top"] = top
    base = {"element": {"type": "hexa8", "formulation": "full"},
            "solver": {"mode": "relaxation", "backend": "numpy", "n_steps": 1},
            "material": {"type": "linear_elastic", "E": 2e11, "nu": 0.3, "rho": 2.0}}
    for load, target in (({"region": "top", "pressure": 2e6}, -12e6),
                         ({"body_force": [0, 0, -4]}, -24.0),
                         ({"gravity": [0, 0, -10]}, -120.0)):
        s = Solver({**base, "loads": [load]}, m,
                   create("linear_elastic", E=2e11, nu=0.3, rho=2.0),
                   log=lambda *_: None)
        assert s.load_groups[0][1].sum(axis=0)[2] == pytest.approx(target, rel=1e-12)


def test_load_schema_requires_one_physical_quantity_and_defaults_to_all():
    assert LoadSpec(body_force=[0, 0, -24000]).region == "all"
    with pytest.raises(ValueError, match="exactly one"):
        LoadSpec(pressure=1.0, gravity=[0, 0, -9.81])
    with pytest.raises(ValueError, match="greater than 0"):
        LoadSpec(pressure=0.0)
    case = Case(mesh={"dimensions": [1, 1, 1], "divisions": [1, 1, 1]},
                material={"type": "linear_elastic", "E": 1e9, "nu": 0.3,
                          "rho": 1000},
                loads=[{"gravity": [0, 0, -9.81]}],
                solver={"mode": "relaxation", "n_steps": 1})
    assert case.loads[0].region == "all"


def test_continuum_ground_motion_is_an_inertial_load(tmp_path):
    record = tmp_path / "motion.dat"
    record.write_text("0 0\n1 3\n", encoding="utf-8")
    m = box_mesh((1, 1, 1), (1, 1, 1))
    p = {"element": {"type": "hexa8", "formulation": "full"},
         "solver": {"mode": "dynamic", "backend": "numpy", "total_time": 0.01},
         "material": {"type": "linear_elastic", "E": 2e9, "nu": 0.3, "rho": 10.0},
         "loads": [{"ground_motion": {"path": str(record), "direction": [1, 0, 0]}}]}
    s = Solver(p, m, create("linear_elastic", E=2e9, nu=0.3, rho=10.0), log=lambda *_: None)
    assert np.allclose(s.external_force(0.5, physical_time=0.5),
                       -s.physical_mass * [1.5, 0, 0])


def test_nodal_stress_patch_recovery_reproduces_a_linear_field():
    m = box_mesh((1.0, 0.8, 0.6), (2, 2, 2), element="tet10")
    p = {"element": {"type": "tet10", "formulation": "full"},
         "solver": {"mode": "relaxation", "backend": "numpy", "n_steps": 1},
         "material": {"type": "linear_elastic", "E": 2e11, "nu": 0.3, "rho": 1.0}}
    s = Solver(p, m, create("linear_elastic", E=2e11, nu=0.3, rho=1.0),
               log=lambda *_: None)
    x = s.pre.gp_coords
    coeff = np.array([[2.0, -3.0, 5.0], [-1.0, 4.0, 2.0], [3.0, 0.5, -2.0],
                      [0.0, 1.0, 0.0], [1.0, 0.0, -1.0], [2.0, -1.0, 1.0]])
    sig = 7.0 + np.einsum("egj,kj->egk", x, coeff)
    got = s.recover_nodal_stress(sig)
    want = 7.0 + m.nodes @ coeff.T
    assert np.allclose(got, want, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("spec,match", [
    ({"a": {"union": ["b"]}, "b": {"union": ["a"]}}, "cycle"),
    ({"a": {"group": "nope"}}, "never defined"),
    ({"a": {"box": [9, 10, 9, 10, 9, 10]}}, "selected no nodes"),
    ({"a": {"box": [0, 1, 0, 1, 0, 1], "all": True}}, "exactly one"),
])
def test_bad_selectors_say_what_is_wrong(spec, match):
    with pytest.raises(ValueError, match=match):
        S.resolve_all(mesh().nodes, spec)


def test_an_empty_region_is_an_error_not_an_empty_set():
    """A tolerance slightly too tight selects nothing and looks like a support."""
    with pytest.raises(ValueError, match="selected no nodes"):
        S.resolve_all(mesh().nodes, {
            "miss": {"plane": {"point": [0, 0, 0.123], "normal": [0, 0, 1],
                               "tol": 1e-9}}})


# --------------------------------------------------------------------------- #
#  amplitudes
# --------------------------------------------------------------------------- #


def test_builtin_amplitudes():
    a = amp.build(None)
    assert set(a) == {"ramp", "hold", "ramp_hold"}
    assert a["ramp"](0.0) == 0.0 and a["ramp"](1.0) == 1.0
    assert a["hold"](0.0) == 1.0 and a["hold"](1.0) == 1.0
    # the default: on over the first half, steady after
    assert a["ramp_hold"](0.5) == pytest.approx(1.0)
    assert a["ramp_hold"](0.75) == pytest.approx(1.0)
    assert a["ramp_hold"](0.25) == pytest.approx(0.5)


def test_settled_is_what_the_convergence_test_needs():
    a = amp.build(None)
    assert not a["ramp"].settled(0.5)
    assert a["ramp"].settled(1.0)
    assert a["hold"].settled(0.0)
    assert a["ramp_hold"].settled(0.5)      # steady from halfway on


def test_the_default_depends_on_whether_the_step_is_staged():
    """A staged step must sample its loading evenly, one stage per level."""
    a = amp.build(None)
    assert amp.get(a, None, "x", staged=True).name == "ramp"
    assert amp.get(a, None, "x", staged=False).name == "ramp_hold"


def test_cyclic_generator_lands_on_its_peaks():
    a = amp.build({"c": {"cyclic": {"peaks": [0.5, 1.0], "repeats": 1}}})["c"]
    got = [a(p) for p in np.linspace(0, 1, 9)]
    assert got == pytest.approx([0, 0.5, 0, -0.5, 0, 1.0, 0, -1.0, 0])


def test_an_undefined_amplitude_lists_the_ones_that_exist():
    with pytest.raises(ValueError, match="Available: hold, ramp, ramp_hold"):
        amp.get(amp.build(None), "nope", "a load")


@pytest.mark.parametrize("table,match", [
    ([[0, 0]], "at least two"),
    ([[1, 0], [0, 1]], "must not decrease"),
])
def test_bad_amplitude_tables(table, match):
    with pytest.raises(ValueError, match=match):
        amp.Amplitude("x", table)


# --------------------------------------------------------------------------- #
#  steps
# --------------------------------------------------------------------------- #


def cantilever(**extra):
    p = {
        "element": {"type": "hexa8", "formulation": "b-bar"},
        "regions": {"x0": {"where": {"x": {"max": 0.0}}},
                    "xL": {"where": {"x": {"min": 1.0}}}},
        "constraints": [{"region": "x0", "dofs": [1, 1, 1]}],
        "solver": {"mode": "relaxation", "mass": "underwood", "backend": "numpy",
                   "n_steps": 20000, "tolerance": 1e-8,
                   "damping": {"type": "kinetic"}},
    }
    p.update(extra)
    return p


def solve(p):
    m = box_mesh((1.0, 0.2, 0.2), (8, 2, 2))
    mat = create("linear_elastic", E=2e11, nu=0.3, rho=7850.0)
    s = Solver(p, m, mat, log=lambda *_: None)
    return s.run(progress_every=200, output_every=10 ** 9)


def test_splitting_one_analysis_into_two_steps_reaches_the_same_answer():
    """If it does not, state is not really carrying over between steps."""
    one = solve(cantilever(loads=[{"region": "xL", "vector": [0, 0, -1e4]}]))
    two = solve(cantilever(steps=[
        {"name": "half", "loads": [{"region": "xL", "vector": [0, 0, -5e3]}]},
        {"name": "full", "loads": [{"region": "xL", "vector": [0, 0, -1e4]}]},
    ]))
    rel = np.abs(one.u - two.u).max() / np.abs(one.u).max()
    assert rel < 1e-6, f"{rel:.3e}"
    assert one.residuals[-1] < 1e-8 and two.residuals[-1] < 1e-8


def test_a_model_level_load_ramps_on_once_and_then_holds():
    """"The column carries this throughout" is what a base load means.

    Without the rule a base load either ramps up again from zero in every step,
    or arrives as a step change in the first one. The first is wrong and the
    second is a shock, and both look plausible in the output.
    """
    ref = solve(cantilever(loads=[{"region": "xL", "vector": [0, 0, -1e4]}]))
    two = solve(cantilever(
        loads=[{"region": "xL", "vector": [0, 0, -1e4]}],
        steps=[{"name": "down"},
               {"name": "push",
                "loads": [{"region": "xL", "vector": [5e3, 0, 0]}]}]))
    m = two.mesh
    # the vertical load is still exactly what it was, not doubled and not gone
    z_ref = ref.u[m.regions["xL"], 2].mean()
    z_two = two.u[m.regions["xL"], 2].mean()
    assert z_two == pytest.approx(z_ref, rel=2e-3)
    assert two.u[m.regions["xL"], 0].mean() > 0.0      # and the push happened


def test_a_step_can_drive_a_dof_the_base_only_held():
    p = cantilever(
        constraints=[{"region": "x0", "dofs": [1, 1, 1]},
                     {"region": "xL", "dofs": [1, 0, 0]}],
        steps=[{"name": "held"},
                         {"name": "pushed",
                    "constraints": [{"region": "xL", "dofs": [1, 0, 0],
                                     "value": 1e-4, "amplitude": "hold"}]}])
    # This test exercises declaration precedence, not quasi-static
    # equilibrium. A single prescribed corner leaves a poor relaxation problem
    # (legacy behavior silently returned at max steps), so use physical time.
    # Under `mass: underwood` the time step is 1.0, so 2e4 of "time" is the
    # 20 000 steps the run needs to walk the prescribed value in through its
    # approach ramp. (It used to ask for 1e-4 and get 20 000 steps anyway,
    # because the solver-block n_steps overrode a dynamic run's duration --
    # the defect v0.1 removed.)
    p["solver"].update(mode="dynamic", total_time=2.0e4,
                       damping={"type": "none"})
    s = solve(p)
    assert s.u[s.mesh.regions["xL"], 0].mean() == pytest.approx(1e-4, rel=1e-9)


def test_per_step_solver_settings_override_the_model():
    # each step's own budget and tolerance, not the model's: a tolerance no
    # residual reaches makes a step spend its whole budget and say so. (Until
    # 1.18 the steps ran in dynamic mode, where the schema refuses a step's
    # `n_steps`; a solver's case is validated now, core/problem.py.)
    from vfife.core.diagnostics import ConvergenceFailure
    load = [{"region": "xL", "vector": [0, 0, -1e3]}]
    with pytest.raises(ConvergenceFailure, match="reached 300 iterations"):
        solve(cantilever(loads=load, steps=[{"name": "a", "n_steps": 300, "tolerance": 1e-300},
                                            {"name": "b", "n_steps": 500}]))
    with pytest.raises(ConvergenceFailure, match="reached 500 iterations"):
        solve(cantilever(loads=load, steps=[{"name": "a", "n_steps": 20000},
                                            {"name": "b", "n_steps": 500, "tolerance": 1e-300}]))


# --------------------------------------------------------------------------- #
#  the thing none of this could do before
# --------------------------------------------------------------------------- #


def test_cyclic_loading_shows_the_bauschinger_effect():
    """A reversed cyclic protocol, which the old input format could not express.

    The bar yields in tension, unloads elastically, and then yields in
    compression EARLY -- because kinematic hardening has translated the yield
    surface. With purely isotropic hardening the reversal would mirror the
    tensile peak instead.
    """
    m = box_mesh((1.0, 0.1, 0.1), (8, 1, 1))
    mat = create("j2", E=2.0e11, nu=0.3, rho=7850.0,
                 hardening={"source": "linear", "yield_strength": 250e6,
                            "H": 2e9},
                 kinematic=8e9)
    p = {
        "element": {"type": "hexa8", "formulation": "b-bar"},
        "regions": {"x0": {"where": {"x": {"max": 0.0}}},
                    "xL": {"where": {"x": {"min": 1.0}}}},
        "amplitudes": {"cyc": {"cyclic": {"peaks": [1.0], "repeats": 1}}},
        "constraints": [{"region": "x0", "dofs": [1, 1, 1]}],
        "steps": [{"name": "cyclic",
                   "constraints": [{"region": "xL", "dofs": [1, 0, 0],
                                    "value": 4.0e-3, "amplitude": "cyc"}],
                   "load_steps": 16, "max_steps_per_load": 3000}],
            # This constitutive-path test uses a 3000-iteration stage budget;
            # its reversal stages reach about 7e-3. Use a matching acceptance
            # level instead of relying on legacy max-step-as-converged behavior.
            "solver": {"mode": "relaxation", "mass": "underwood", "backend": "numpy",
                       "n_steps": 100000, "tolerance": 1e-2,
                   "damping": {"type": "kinetic"}},
        "output": {"monitors": {
            "force": {"kind": "reaction", "region": "x0", "dof": "x"},
            "disp": {"kind": "displacement", "region": "xL", "dof": "x"}}},
    }
    s = Solver(p, m, mat, log=lambda *_: None).run(progress_every=500,
                                                   output_every=10 ** 9)
    d = np.array(s.monitors["disp"].values)
    sig = np.array(s.monitors["force"].values) / 0.01 / 1e6      # MPa

    # the protocol was followed: the extremes are the prescribed displacement
    assert d.max() == pytest.approx(4.0e-3, rel=1e-6)
    assert d.min() == pytest.approx(-4.0e-3, rel=1e-6)

    peak = sig.max()
    assert peak > 250.0                      # yielded, and hardened past fy

    # unloading from the peak is elastic: 1 mm on a 1 m bar is E*1e-3 = 200 MPa
    top = int(np.argmax(sig))
    assert sig[top] - sig[top + 1] == pytest.approx(200.0, rel=0.05)

    # and the reverse yield arrives EARLY -- before the mirrored -peak that
    # purely isotropic hardening would give
    reverse = sig.min()
    assert reverse > -(peak + 50.0), (
        f"reverse yield at {reverse:.1f} MPa is not early against a peak of "
        f"{peak:.1f}: the kinematic hardening is not translating the surface"
    )
    # the loop must have area: at zero drift on the way back the stress is not
    # zero, which is what makes it a hysteresis loop rather than a line
    back_at_zero = sig[np.argmin(np.abs(d[top:])) + top]
    assert abs(back_at_zero) > 50.0


def test_surface_is_offered_but_not_computed_until_it_is_used():
    """It costs a pass over every element face, and no case in the library
    names it -- so an analysis that does not ask for it must not pay for it.

    Measured: about 12 s at 10^6 elements, and it used to be computed twice per
    run, once in preflight and once in the solver.
    """
    calls = []
    real = S.surface_nodes

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)

    m = box_mesh((1.0, 1.0, 1.0), (4, 4, 4))
    monkey = getattr(S, "surface_nodes")
    try:
        S.surface_nodes = counted
        thunk = S.surface_provider(m, "hexa8")
        specs = {"base": {"plane": {"point": [0, 0, 0], "normal": [0, 0, 1]}}}
        out = S.resolve_all(m.nodes, specs, {}, lazy={"surface": thunk})
        assert calls == []                       # nothing referenced it
        assert "surface" not in out

        specs["skin"] = {"group": "surface"}
        out = S.resolve_all(m.nodes, specs, {}, lazy={"surface": thunk})
        assert len(calls) == 1                   # referenced once, computed once
        assert np.array_equal(np.sort(out["skin"]),
                              np.sort(real(m.elements, "hexa8", m.n_nodes)))

        # and asking again reuses the mesh's cache rather than recomputing
        S.resolve_all(m.nodes, specs, {}, lazy={"surface": thunk})
        assert len(calls) == 1
    finally:
        S.surface_nodes = monkey


def test_a_lazy_region_named_only_by_a_constraint_still_resolves():
    """A boundary condition may name `surface` with no region spec mentioning
    it; the solver must build it rather than report it unknown."""
    m = box_mesh((1.0, 1.0, 1.0), (4, 4, 4))
    mat = create("linear_elastic", E=2e11, nu=0.3, rho=7850.0)
    p = {
        "element": {"type": "hexa8", "formulation": "b-bar"},
        "regions": {"base": {"plane": {"point": [0, 0, 0], "normal": [0, 0, 1]}}},
        "constraints": [{"region": "base", "dofs": [1, 1, 1]},
                        {"region": "surface", "dofs": [0, 1, 0]}],
        "solver": {"mode": "relaxation", "mass": "underwood", "backend": "numpy",
                   "n_steps": 5, "damping": {"type": "kinetic"}},
    }
    s = Solver(p, m, mat, log=lambda *_: None)
    expected = S.surface_nodes(m.elements, "hexa8", m.n_nodes)
    assert int(s.fixed[:, 1].sum()) == len(expected)


def test_an_unknown_region_still_lists_the_lazy_ones():
    """`surface` is available even before it exists; the error must say so."""
    m = box_mesh((1.0, 1.0, 1.0), (3, 3, 3))
    mat = create("linear_elastic", E=2e11, nu=0.3, rho=7850.0)
    p = {
        "element": {"type": "hexa8"},
        "constraints": [{"region": "nope", "dofs": [1, 1, 1]}],
        "solver": {"mode": "relaxation", "n_steps": 5},
    }
    with pytest.raises(ValueError, match="surface"):
        Solver(p, m, mat, log=lambda *_: None)
