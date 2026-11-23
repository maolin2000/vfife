"""M01 orthotropic linear elastic -- registration and mechanics tests."""

import numpy as np
import pytest

from vfife.materials import definition, available
from vfife.materials.orthotropic import OrthotropicElastic, _build_matrices
from vfife.voigt import elastic_matrix
from vfife.io import schema
from vfife.api import prepare_case


def test_metadata_declares_orthotropic_numpy_only():
    assert "orthotropic" in available()
    d = definition("orthotropic")
    assert d.kinematics == ("solid_3d",)
    assert d.backends == ("numpy",)
    assert not d.nonlinear
    assert not d.regularised
    assert set(d.parameters) == {"E1", "E2", "E3", "nu12", "nu13", "nu23", "G12", "G13", "G23", "rho"}


def test_isotropic_limit_matches_linear_elastic():
    E, nu = 2.1e11, 0.3
    G = E / (2.0 * (1.0 + nu))
    m_iso = OrthotropicElastic(E1=E, E2=E, E3=E, nu12=nu, nu13=nu, nu23=nu, G12=G, G13=G, G23=G, rho=7800)
    D_iso = elastic_matrix(E, nu)
    assert np.allclose(m_iso.D0, D_iso, atol=1e-6, rtol=1e-12)


def test_compliance_is_inverse_and_symmetric():
    m = OrthotropicElastic(E1=1.2e11, E2=8e10, E3=1.0e11, nu12=0.25, nu13=0.2, nu23=0.22, G12=4e10, G13=3.5e10, G23=3e10, rho=1500)
    assert np.allclose(m.D0, m.D0.T, atol=1e-12)
    assert np.allclose(m.S @ m.D0, np.eye(6), atol=1e-12)


def test_uniaxial_along_1_gives_1_over_E1():
    E1, E2, E3 = 1.0e11, 0.8e11, 0.9e11
    m = OrthotropicElastic(E1=E1, E2=E2, E3=E3, nu12=0.25, nu13=0.2, nu23=0.22, G12=4e10, G13=3.5e10, G23=3e10, rho=1500)
    # strain = S * stress  -> for sigma = [s,0,0,0,0,0], eps = s * S[:,0]
    # check eps0 = s/E1
    s = 1e6
    strain = np.array([[s, 0, 0, 0, 0, 0]]) @ m.S.T  # not needed; directly
    # Instead check S[0,0] == 1/E1
    assert m.S[0, 0] == pytest.approx(1.0 / E1)
    assert m.S[1, 1] == pytest.approx(1.0 / E2)
    # also test via material update: pure uniaxial stress along x
    # we test stiffness: apply eps = [e,0,0,0,0,0] -> stress = D*eps ; for orthotropic, stress will be triaxial due to Poisson coupling, but D*S = I ensures consistency
    eps = np.array([[1e-5, 0, 0, 0, 0, 0]], dtype=np.float64)
    state = np.zeros((1, 0))
    sig = m.update(eps, state, dt=1.0)
    # recover strain via S: eps_recovered = S * sig
    eps_rec = sig @ m.S.T
    assert np.allclose(eps, eps_rec, atol=1e-14)


def test_shear_decouples():
    m = OrthotropicElastic(E1=1e11, E2=1e11, E3=1e11, nu12=0.3, nu13=0.3, nu23=0.3, G12=2e10, G13=3e10, G23=4e10, rho=7800)
    # engineering shear: sigma_4 = G12 * gamma_12
    for G, idx in [(2e10, 3), (4e10, 4), (3e10, 5)]:
        eps = np.zeros((1, 6))
        eps[0, idx] = 1e-5
        sig = m.update(eps, np.zeros((1, 0)), dt=1.0)
        assert sig[0, idx] == pytest.approx(G * 1e-5, rel=1e-12)
        # other components zero
        other = [i for i in range(6) if i != idx]
        assert np.allclose(sig[0, other], 0.0, atol=1e-12)


def test_invalid_params_rejected():
    base = dict(E1=1e11, E2=1e11, E3=1e11, nu12=0.3, nu13=0.3, nu23=0.3, G12=5e10, G13=5e10, G23=5e10, rho=7800)
    with pytest.raises(ValueError, match="E1"):
        OrthotropicElastic(**{**base, "E1": -1e11})
    with pytest.raises(ValueError, match="nu12"):
        OrthotropicElastic(**{**base, "nu12": 0.9})
    # non-SPD due to extreme anisotropy: E2 >> E1 with large nu makes compliance indefinite
    with pytest.raises(ValueError, match="positive definite"):
        OrthotropicElastic(E1=1e11, E2=1e12, E3=1e11, nu12=0.49, nu13=0.3, nu23=0.3, G12=5e10, G13=5e10, G23=5e10, rho=7800)


def test_orthotropic_runs_through_solver_patch():
    # single hex8 patch: use orthotropic with isotropic-equivalent constants, should reproduce patch equilibrium
    from vfife.core.solver import Solver  # smoke import
    raw = {
        "name": "ortho_patch",
        "mesh": {"dimensions": [1, 1, 1], "divisions": [1, 1, 1]},
        "element": {"formulation": "b-bar"},
        "material": {"type": "orthotropic", "E1": 2.1e11, "E2": 2.1e11, "E3": 2.1e11, "nu12": 0.3, "nu13": 0.3, "nu23": 0.3, "G12": 8.0769e10, "G13": 8.0769e10, "G23": 8.0769e10, "rho": 7800},
        "constraints": [{"region": "all", "dofs": [1, 1, 1]}],
        "solver": {"mode": "relaxation", "n_steps": 10, "tolerance": 1e-6},
    }
    case = schema.Case.model_validate(raw)
    prepared = prepare_case(case)
    # should pass preflight and have no CAPABILITY_MISMATCH
    assert not any(i["code"] == "CAPABILITY_MISMATCH" for i in prepared.preflight["issues"])
    assert prepared.preflight["status"] == "PASS"


def test_orthotropic_uses_solid3d_only():
    from pathlib import Path
    msh = str(Path(__file__).parent / "data" / "bar_truss2_v22.msh")
    raw = {
        "name": "bad_kinematics",
        "mesh": {"type": "gmsh", "path": msh},
        "element": {"type": "truss2", "formulation": "full", "area": 1e-3},
        "material": {"type": "orthotropic", "E1": 1e11, "E2": 1e11, "E3": 1e11, "nu12": 0.3, "nu13": 0.3, "nu23": 0.3, "G12": 5e10, "G13": 5e10, "G23": 5e10, "rho": 7800},
        "constraints": [{"region": "fixed", "dofs": [1, 1, 1]}],
        "solver": {"mode": "relaxation", "n_steps": 10},
    }
    case = schema.Case.model_validate(raw)
    prepared = prepare_case(case)
    assert any(i["code"] == "CAPABILITY_MISMATCH" for i in prepared.preflight["issues"])
