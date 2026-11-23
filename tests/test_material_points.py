import numpy as np
import pytest

from vfife.materials import create
from vfife.materials.points import initialise_points, validate_restored_state


def test_solid_material_point_contract_pins_voigt_state_and_stability():
    material = create("j2", E=210e9, nu=0.3, rho=7800,
                      yield_strength=250e6)
    state, contract = initialise_points(
        material, 4, kinematics="solid_3d", backend="numpy",
        char_length=[0.1, 0.2, 0.3, 0.4],
    )
    assert state.shape == (4, 13)
    assert contract.strain_components == ("exx", "eyy", "ezz", "gxy", "gyz", "gxz")
    assert contract.stress_components == ("sxx", "syy", "szz", "sxy", "syz", "sxz")
    assert contract.state_fields == ("eps_p", "alpha", "X")
    assert np.allclose(contract.stable_time_step([0.1, 0.2]),
                       np.array([0.1, 0.2])/material.wave_speed)


def test_uniaxial_contract_and_backend_rejection_are_early():
    material = create("rebar", E=200e9, rho=7850, yield_strength=400e6)
    state, contract = initialise_points(
        material, 2, kinematics="uniaxial", backend="numpy",
        char_length=[0.5, 0.5],
    )
    assert state.shape == (2, 3)
    assert contract.strain_components == ("axial_strain",)
    with pytest.raises(ValueError, match="solid_3d"):
        initialise_points(material, 1, kinematics="solid_3d", backend="numpy",
                          char_length=[1.0])


def test_regularised_material_requires_real_characteristic_length():
    material = create(
        "cdp", E=30e9, nu=0.2, rho=2400,
        compression={"source": "gb50010", "fc": 30e6, "beta_c": 0.7},
        tension={"source": "fracture_energy", "ft": 3e6, "Gf": 120,
                 "beta_t": 0.9},
    )
    with pytest.raises(ValueError, match="requires characteristic length"):
        initialise_points(material, 1, kinematics="solid_3d", backend="numpy")
    with pytest.raises(ValueError, match="finite and positive"):
        initialise_points(material, 1, kinematics="solid_3d", backend="numpy",
                          char_length=[0.0])


def test_restored_state_must_match_contract_shape_and_be_finite():
    material = create("linear_elastic", E=1e9, nu=0.25, rho=1000)
    state, contract = initialise_points(
        material, 3, kinematics="solid_3d", backend="numpy",
        char_length=np.ones(3),
    )
    assert validate_restored_state(state, contract, 3).flags.c_contiguous
    with pytest.raises(ValueError, match="does not match"):
        validate_restored_state(np.zeros((2, 0)), contract, 3)
    # Linear elastic state has zero columns; use a shape-valid non-finite test
    # with J2 instead.
    plastic = create("j2", E=1e9, nu=0.25, rho=1000, yield_strength=1e6)
    pstate, pcontract = initialise_points(
        plastic, 1, kinematics="solid_3d", backend="numpy", char_length=[1.0])
    pstate[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_restored_state(pstate, pcontract, 1)
