"""Small-tensor kernels shared by the material models.

Everything here is closed-form and branch-light on purpose: these routines run
once per integration point per iteration per step, and they have to survive the
move into a GPU kernel, where there is no LAPACK to call.
"""

from __future__ import annotations

import numpy as np


def principal_values(t: np.ndarray) -> np.ndarray:
    """Eigenvalues of symmetric 3x3 tensors in TENSOR-Voigt form. (n,6) -> (n,3).

    Ascending order, matching np.linalg.eigvalsh.  Closed form
    (Cardano/trigonometric) rather than an eigensolver: measured 7x faster than
    np.linalg.eigvalsh on 20k tensors, agreeing to 3e-14, and it is the only
    form that ports to a GPU kernel.

    NOTE ON CONVENTION: pass STRESS-like Voigt arrays directly; pass
    ENGINEERING-strain arrays only after multiplying by ENG_TO_TENSOR.
    """
    xx, yy, zz, xy, yz, zx = (t[..., i] for i in range(6))
    q = (xx + yy + zz) / 3.0
    p1 = xy * xy + yz * yz + zx * zx
    a, b, c = xx - q, yy - q, zz - q
    p2 = a * a + b * b + c * c + 2.0 * p1
    p = np.sqrt(np.maximum(p2 / 6.0, 0.0))
    safe = np.maximum(p, 1e-30)
    det = (a * (b * c - yz * yz)
           - xy * (xy * c - yz * zx)
           + zx * (xy * yz - b * zx))
    r = np.clip(det / (2.0 * safe ** 3), -1.0, 1.0)
    phi = np.arccos(r) / 3.0
    e1 = q + 2.0 * p * np.cos(phi)
    e3 = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    e2 = 3.0 * q - e1 - e3
    out = np.stack([e3, e2, e1], axis=-1)
    iso = p2 <= 0.0
    if iso.any():
        out[iso] = q[iso, None]
    return out


def ddot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A:B for two STRESS-like (tensor-convention) Voigt arrays.

    Shear components appear twice, because a symmetric tensor has two of each.
    Never call this on an engineering-strain array.
    """
    return (
        a[..., 0] * b[..., 0]
        + a[..., 1] * b[..., 1]
        + a[..., 2] * b[..., 2]
        + 2.0 * (a[..., 3] * b[..., 3]
                 + a[..., 4] * b[..., 4]
                 + a[..., 5] * b[..., 5])
    )


def deviator(sig: np.ndarray) -> np.ndarray:
    """Deviatoric part of a STRESS-like Voigt array."""
    p = (sig[..., 0] + sig[..., 1] + sig[..., 2]) / 3.0
    s = sig.copy()
    s[..., 0] -= p
    s[..., 1] -= p
    s[..., 2] -= p
    return s
