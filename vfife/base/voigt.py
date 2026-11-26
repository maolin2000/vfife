"""Voigt conventions used throughout vfife.

There is exactly ONE convention in this package, and every array that carries a
symmetric second-order tensor obeys it:

    strain  eps = [exx, eyy, ezz, gxy, gyz, gzx]   <- ENGINEERING shear, g = 2*e
    stress  sig = [sxx, syy, szz, txy, tyz, tzx]   <- true shear stress

Consequences that the original MATLAB code got wrong (see docs):

  * The elasticity matrix D0 has G (not 2G) on the shear diagonal, because it
    multiplies engineering shear strain.
  * The plastic flow rule produces a TENSOR strain increment.  To store it in
    the engineering convention its shear components must be DOUBLED.
  * The J2 invariant of a STRESS-like Voigt vector is
        J2 = 0.5*(s1^2+s2^2+s3^2) + s4^2+s5^2+s6^2
    but of a STRAIN-like (engineering) vector it is
        J2 = 0.5*(e1^2+e2^2+e3^2) + 0.25*(e4^2+e5^2+e6^2)
    Never use one formula on the other kind of vector.

Helpers below make the doubling explicit so it cannot be forgotten silently.
"""

import numpy as np

# Multiply a tensor-convention Voigt vector by this to get engineering convention.
TENSOR_TO_ENG = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
ENG_TO_TENSOR = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

# [1,1,1,0,0,0] -- the Voigt form of the identity tensor.
IDENTITY = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def elastic_matrix(E: float, nu: float) -> np.ndarray:
    """Isotropic elasticity matrix for ENGINEERING shear strain (6x6)."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * G
    D[3, 3] = D[4, 4] = D[5, 5] = G
    return D


def dilatational_wave_speed(E: float, nu: float, rho: float) -> float:
    """P-wave speed sqrt(M/rho); this is what sets the explicit stability limit."""
    M = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return float(np.sqrt(M / rho))


def deviator(sig: np.ndarray) -> np.ndarray:
    """Deviatoric part of a STRESS-like Voigt array, shape (..., 6)."""
    p = (sig[..., 0] + sig[..., 1] + sig[..., 2]) / 3.0
    s = sig.copy()
    s[..., 0] -= p
    s[..., 1] -= p
    s[..., 2] -= p
    return s


def j2_stress(s: np.ndarray) -> np.ndarray:
    """J2 of a STRESS-like deviatoric Voigt array, shape (..., 6) -> (...)."""
    return 0.5 * (s[..., 0] ** 2 + s[..., 1] ** 2 + s[..., 2] ** 2) + (
        s[..., 3] ** 2 + s[..., 4] ** 2 + s[..., 5] ** 2
    )


def von_mises(sig: np.ndarray) -> np.ndarray:
    """von Mises equivalent of a STRESS-like Voigt array, shape (..., 6) -> (...)."""
    return np.sqrt(3.0 * j2_stress(deviator(sig)))


def tensor_norm_stress(s: np.ndarray) -> np.ndarray:
    """Frobenius norm ||s|| of a STRESS-like deviatoric Voigt array."""
    return np.sqrt(2.0 * j2_stress(s))
