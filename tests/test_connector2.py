"""Engineering qualification tests for the six-component connector2."""
from pathlib import Path

import numpy as np
import pytest

from vfife.cli import _with_element, build, make_solver
from vfife.core.beam_solver import BeamSolver
from vfife.elements import connector
from vfife.io import schema


ROOT = Path(__file__).resolve().parents[1]


def _props(**updates):
    out = {
        "type": "connector2", "coordinate_system": "local",
        "stiffness": [1e5, 2e5, 3e5, 4e4, 5e4, 6e4],
        "connector_damping": [10, 20, 30, 4, 5, 6], "curves": {},
    }
    out.update(updates)
    return out


def _geometry():
    x0 = np.array([[0., 0., 0.], [1., 0., 0.]])
    R0, _ = BeamSolver._frame(x0[0], x0[1], [0, 0, 1])
    return x0, R0


def _standalone_case(stiffness=None, curves=None):
    return schema.Case.model_validate({
        "name": "connector_probe",
        "mesh": {"type": "gmsh", "path": "benchmarks/assets/beam_cantilever_v22.msh"},
        "element": {"type": "connector2", "formulation": "full",
                    "stiffness": stiffness or [1e8, 0, 0, 0, 0, 0],
                    "curves": curves or {}, "orientation": [0, 0, 1]},
        "material": {"type": "linear_elastic", "E": 2e11, "nu": 0.3, "rho": 7850},
        "constraints": [
            {"region": "fixed", "dofs": [1, 1, 1, 1, 1, 1]},
            {"region": "loaded", "dofs": [0, 1, 1, 1, 1, 1]},
        ],
        "loads": [{"region": "loaded", "vector": [1000, 0, 0]}],
        "solver": {"mode": "relaxation", "backend": "numpy", "n_steps": 5000,
                   "damping": {"type": "kinetic"}, "tolerance": 1e-9},
    })


def test_schema_accepts_vector_or_matrix_and_rejects_invalid_connector_data():
    e = schema.ElementSpec(type="connector2", formulation="full",
                           stiffness=np.diag([1, 2, 3, 4, 5, 6]).tolist())
    assert np.asarray(e.stiffness).shape == (6, 6)
    nonsymmetric=np.eye(6); nonsymmetric[0,1]=1
    with pytest.raises(ValueError, match="symmetric"):
        schema.ElementSpec(type="connector2", formulation="full",
                           stiffness=nonsymmetric.tolist())
    with pytest.raises(ValueError, match="origin"):
        schema.ElementSpec(type="connector2", formulation="full",
                           curves={"ux": [[-1, -2], [1, 2]]})
    with pytest.raises(ValueError, match="non-decreasing"):
        schema.ElementSpec(type="connector2", formulation="full",
                           curves={"ux": [[-1, -1], [0, 0], [1, -1]]})


def _state(q):
    """The two ends' translations and orientation MATRICES, from the twelve
    nodal numbers these tests were written in. Since v0.1 the
    connector takes the triads, not the rotation vectors -- the physical
    configuration each test describes is unchanged, only its spelling."""
    q = np.asarray(q, dtype=float).reshape(2, 6)
    from vfife.core.rotations import exp_map
    return q[:, :3], exp_map(q[:, 3:])


def test_six_linear_components_are_work_conjugate():
    x0, R0 = _geometry(); q = np.zeros(12)
    deformation = np.array([.01, -.02, .03, -.04, .05, -.06])
    q[6:9] = R0.T@deformation[:3]
    q[9:12] = R0.T@deformation[3:]
    fe, action, measured = connector.response(*_state(q), np.zeros(12), x0, R0, _props())
    assert measured[3:] == pytest.approx(deformation[3:])
    assert action == pytest.approx(np.diag(connector.matrix(_props()["stiffness"]))*measured)
    assert fe.reshape(2,6)[:,:3].sum(axis=0) == pytest.approx(np.zeros(3),abs=2e-6)


def test_rigid_translation_and_rotation_produce_no_connector_force():
    x0, R0 = _geometry(); angle=np.pi/3
    Q=np.array([[np.cos(angle),-np.sin(angle),0.],
                [np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    q=np.zeros((2,6)); q[:,:3]=x0@Q.T-x0+[.2,-.1,.4]; q[:,3:]=[0,0,angle]
    fe, action, deformation = connector.response(*_state(q), np.zeros(12), x0, R0, _props())
    assert np.abs(deformation).max() < 1e-12
    assert np.abs(action).max() < 1e-7
    assert np.abs(fe).max() < 1e-7


def test_a_rigid_rotation_of_a_DEFORMED_connector_is_free():
    """The one this element failed until v0.1, and the reason it was
    still failing after v0.1 fixed the same thing in the frame solver.

    Two ends turned by DIFFERENT, non-coaxial rotations, then the whole
    connector moved rigidly: the deformation must not notice. It did. The
    frame was `rotation_matrix(0.5 * (phi1 + phi2))` and the rotational
    deformation `phi2 - phi1`; rotation vectors neither average nor
    subtract, and both expressions happen to be exact when the two rotations
    are equal or coaxial -- which is every other test in this file, and
    every case in the library. The invariant harness of v0.1 asked the
    question generically and scored 7.5e-2 of the model's own stiffness
    scale."""
    from vfife.core.rotations import exp_map

    x0, R0 = _geometry()
    u = np.array([[0.004, -0.002, 0.001], [-0.001, 0.003, 0.002]])
    T = np.stack([exp_map(np.array([0.3, -0.2, 0.5])),
                  exp_map(np.array([-0.1, 0.4, 0.2]))])
    d0 = connector.deformation(u, T, x0, R0, _props())
    assert np.abs(d0).max() > 0.1, "the probe state has to be deformed to be a test"

    R = exp_map(np.array([0.7, 0.2, -0.9]))
    c = x0.mean(axis=0)
    u_r = ((R @ (x0 + u - c).T).T + c) - x0
    d1 = connector.deformation(u_r, np.einsum("ij,njk->nik", R, T), x0, R0, _props())
    assert np.abs(d1 - d0).max() < 1e-12 * np.abs(d0).max()

    # and the force it produces turns with the body, component by component
    f0 = connector.response(u, T, np.zeros(12), x0, R0, _props())[0].reshape(4, 3)
    f1 = connector.response(u_r, np.einsum("ij,njk->nik", R, T), np.zeros(12),
                            x0, R0, _props())[0].reshape(4, 3)
    assert np.abs(f1 - (R @ f0.T).T).max() < 1e-9 * np.abs(f0).max()


def test_the_frame_is_the_geodesic_mean_of_the_two_triads():
    """`T1 exp(0.5 log(T1^T T2))` -- equidistant from both ends, and equal to
    either when they agree. The mean of the two rotation vectors is neither."""
    from vfife.core.rotations import exp_map, log_map

    T1 = exp_map(np.array([0.3, -0.2, 0.5]))
    T2 = exp_map(np.array([-0.1, 0.4, 0.2]))
    Q = connector.mean_triad(np.stack([T1, T2]))
    assert np.allclose(Q @ Q.T, np.eye(3), atol=1e-14)
    left = np.linalg.norm(log_map((T1.T @ Q)[None])[0])
    right = np.linalg.norm(log_map((Q.T @ T2)[None])[0])
    assert left == pytest.approx(right, rel=1e-12)
    assert np.allclose(connector.mean_triad(np.stack([T1, T1])), T1, atol=1e-14)


def test_local_and_global_axes_select_different_physical_directions():
    x0, R0 = _geometry(); q=np.zeros(12); q[8]=.01  # global z == local y
    _, local, _=connector.response(*_state(q),np.zeros(12),x0,R0,_props(stiffness=[0,2e5,0,0,0,0]))
    _, global_action, _=connector.response(*_state(q),np.zeros(12),x0,R0,
        _props(coordinate_system="global",stiffness=[0,0,3e5,0,0,0]))
    assert local[1] == pytest.approx(2000.)
    assert global_action[2] == pytest.approx(3000.)


def test_piecewise_curve_replaces_diagonal_linear_component_and_extrapolates():
    x0,R0=_geometry(); props=_props(stiffness=[9e9,0,0,0,0,0],
        curves={"ux":[[-.02,-1000],[0,0],[.01,1000],[.03,1400]]})
    q=np.zeros(12); q[6]=.02
    _, action, _=connector.response(*_state(q),np.zeros(12),x0,R0,props)
    assert action[0] == pytest.approx(1200.)
    q[6]=.04; assert connector.response(*_state(q),np.zeros(12),x0,R0,props)[1][0] == pytest.approx(1600.)


def test_dashpot_force_uses_physical_relative_velocity_only_when_requested():
    x0,R0=_geometry(); q=np.zeros(12); v=np.zeros(12); v[6]=.5
    props=_props(stiffness=np.zeros(6).tolist(),connector_damping=[20,0,0,0,0,0])
    assert connector.response(*_state(q),v,x0,R0,props,include_damping=False)[1][0] == 0
    assert connector.response(*_state(q),v,x0,R0,props,include_damping=True)[1][0] == pytest.approx(10.)


def test_pure_dashpot_runs_on_physical_time_with_nodal_mass_and_target_dt():
    raw=_standalone_case().as_dict()
    raw["element"]["stiffness"]=[0,0,0,0,0,0]
    raw["element"]["connector_damping"]=[20,0,0,0,0,0]
    raw["loads"]=[]
    raw["nodal_masses"]=[{"region":"all","mass":2.0}]
    raw["solver"].update(mode="dynamic",total_time=.01,target_dt=.001,
                         damping={"type":"none"})
    case=schema.Case.model_validate(raw); mesh,mat,element=build(case)
    from vfife.preflight import check
    assert check(case,mesh,element)["status"] == "PASS"
    s=make_solver(_with_element(case,element),mesh,mat)
    s.v[6]=.5
    _,actions=s.internal_force()
    assert actions[0,0] == pytest.approx(10.)
    assert s.dt == pytest.approx(.001)


def test_fixed_anchor_models_node_to_ground_and_matches_f_over_k():
    case=_standalone_case(); mesh,mat,element=build(case)
    s=make_solver(_with_element(case,element),mesh,mat)
    s.run(output_every=999999,progress_every=999999)
    assert s.u[1,0] == pytest.approx(1e-5,rel=2e-6)
    assert s.monitors == {}
    fields=s.nodal_fields(s._cell_actions())
    assert fields["connector_force_x"][0] == pytest.approx(1000.,rel=2e-6)


def test_connector_uses_common_mixed_and_coupled_assembly_paths():
    path=ROOT/"cases"/"case_03_mixed_frame.yaml"
    case,_=schema.load(str(path)); mesh,mat,element=build(case,str(path)); p=_with_element(case,element)
    p["members"][0]={"region":"beam_members","type":"connector2",
                     "coordinate_system":"local","stiffness":[1e6,0,0,0,0,0],
                     "connector_damping":None,"curves":{},"orientation":[0,0,1]}
    s=make_solver(p,mesh,mat); assert s.member_kind.tolist()==["connector2","truss2"]
    s.q.reshape(-1,6)[1,0]=1e-3; force,actions=s.internal_force()
    assert actions[0,0] == pytest.approx(1000.)

    coupled_path=ROOT/"cases"/"case_07_coupled_solid_truss.yaml"
    coupled,_=schema.load(str(coupled_path))
    coupled.members[0]=schema.MemberGroupSpec(
        region="truss_members",type="connector2",coordinate_system="local",
        stiffness=[1e8,0,0,0,0,0],orientation=[0,0,1])
    cm,cmat,ce=build(coupled,str(coupled_path))
    from vfife.preflight import check
    assert check(coupled,cm,ce)["status"] == "PASS"
    cs=make_solver(_with_element(coupled,ce,str(coupled_path)),cm,cmat)
    cs.q[6]=1e-6
    total,_=cs.internal_force(); line_only,line_actions=super(type(cs),cs)._member_forces()
    assert line_actions[0,0] == pytest.approx(100.)
    assert abs(total[6]) > abs(line_only[6])


def test_physical_connector_damping_is_rejected_in_relaxation_preflight():
    raw=_standalone_case().as_dict()
    raw["element"]["connector_damping"]=[1,0,0,0,0,0]
    case=schema.Case.model_validate(raw); mesh,_,element=build(case)
    from vfife.preflight import check
    report=check(case,mesh,element)
    assert any(i["code"]=="CONNECTOR_PHYSICAL_DAMPING_IN_RELAXATION" for i in report["issues"])


def test_preflight_reports_unrestrained_inactive_connector_components():
    raw=_standalone_case().as_dict()
    raw["constraints"]=raw["constraints"][:1]
    case=schema.Case.model_validate(raw); mesh,_,element=build(case)
    from vfife.preflight import check
    assert any(i["code"]=="CONNECTOR_MECHANISM" for i in check(case,mesh,element)["issues"])


def _reference_jacobian(u, T, x0, R0, props):
    """The Jacobian as it was built before v0.4: 24 separate
    `deformation` calls, one central difference per column."""
    from vfife.core.rotations import exp_map
    h_u = 2.0e-7 * max(float(np.linalg.norm(u)), 1.0)
    h_w = connector._SPIN_STEP
    J = np.empty((6, 12))
    for n in range(2):
        for k in range(3):
            up = u.copy(); um = u.copy(); up[n, k] += h_u; um[n, k] -= h_u
            J[:, 6 * n + k] = (connector.deformation(up, T, x0, R0, props)
                               - connector.deformation(um, T, x0, R0, props)) / (2 * h_u)
            w = np.zeros(3); w[k] = h_w
            Tp = T.copy(); Tm = T.copy()
            Tp[n] = exp_map(w) @ T[n]; Tm[n] = exp_map(-w) @ T[n]
            J[:, 6 * n + 3 + k] = (connector.deformation(u, Tp, x0, R0, props)
                                   - connector.deformation(u, Tm, x0, R0, props)) / (2 * h_w)
    return J


@pytest.mark.parametrize("system", ["local", "global"])
def test_the_stacked_jacobian_is_the_column_by_column_one(system):
    """Version 0.4 evaluates the state and its 24 perturbations as ONE
    stack (it was 80% of a bearing model's step). Same perturbations, same
    differences: the columns agree with the one-call-per-column reference to
    the central difference's own rounding floor, eps / h ~ 1e-9, over small,
    finite and near-pi relative rotations and large translations."""
    from vfife.core.rotations import exp_map
    rng = np.random.default_rng(106)
    props = _props(coordinate_system=system)
    for scale_u, scale_t in [(0.0, 0.0), (1e-4, 1e-8), (1e-2, 1e-3), (1.0, 0.5), (5.0, 2.5)]:
        for _ in range(20):
            x0 = rng.normal(size=(2, 3)) * 3.0
            R0 = exp_map(rng.normal(size=3))
            u = rng.normal(size=(2, 3)) * scale_u
            T = exp_map(rng.normal(size=(2, 3)) * scale_t)
            d, J = connector._deformation_and_jacobian(u, T, x0, R0, props)
            assert np.allclose(d, connector.deformation(u, T, x0, R0, props), rtol=0, atol=1e-13)
            Jr = _reference_jacobian(u, T, x0, R0, props)
            assert np.abs(J - Jr).max() < 1e-8 * max(np.abs(Jr).max(), 1.0)
    # a relative rotation within 1e-7 of pi takes log_map's per-item branch
    x0, R0 = _geometry()
    T = np.stack([np.eye(3), exp_map(np.array([0.0, 0.0, np.pi - 1e-9]))])
    J = connector.jacobian(np.zeros((2, 3)), T, x0, R0, props)
    assert np.all(np.isfinite(J))
    assert np.allclose(J[:3], _reference_jacobian(np.zeros((2, 3)), T, x0, R0, props)[:3], atol=1e-7)
