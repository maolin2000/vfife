"""Orthotropic linear elastic material -- M01.

Implements the 3-D orthotropic Hooke law in engineering Voigt
notation (see vfife.voigt).  The local material axes are assumed to
coincide with the global axes in this first delivery; a future
extension will add an explicit local-basis rotation.  The caveat in
the registry makes that limitation machine-readable.

The 6x6 compliance S is

    S = [[ 1/E1,  -nu12/E1, -nu13/E1,    0,      0,      0    ],
         [-nu12/E1, 1/E2,  -nu23/E2,     0,      0,      0    ],
         [-nu13/E1,-nu23/E2, 1/E3,       0,      0,      0    ],
         [   0,       0,       0,     1/G12,    0,      0    ],
         [   0,       0,       0,       0,   1/G23,     0    ],
         [   0,       0,       0,       0,      0,   1/G31   ]]

with G31 == G13.  Symmetry S12==S21 etc. is exact by construction;
positive definiteness is checked from the eigenvalues of S (all >0)
so the user cannot create a material that inverts under load.

The stiffness is D = S^{-1} and is symmetric by construction.
"""

from __future__ import annotations

import numpy as np

from .base import Material, register


def _build_matrices(E1, E2, E3, nu12, nu13, nu23, G12, G13, G23):
    S = np.zeros((6, 6), dtype=np.float64)
    S[0, 0] = 1.0 / E1
    S[1, 1] = 1.0 / E2
    S[2, 2] = 1.0 / E3
    S[0, 1] = S[1, 0] = -nu12 / E1
    S[0, 2] = S[2, 0] = -nu13 / E1
    S[1, 2] = S[2, 1] = -nu23 / E2
    S[3, 3] = 1.0 / G12
    S[4, 4] = 1.0 / G23
    S[5, 5] = 1.0 / G13
    D = np.linalg.inv(S)
    # force exact symmetry (numerical noise)
    D = 0.5 * (D + D.T)
    return S, D


@register(
    "orthotropic",
    kinematics=("solid_3d",),
    backends=("numpy",),
    analysis_modes=("dynamic", "relaxation"),
    output_fields=("stress", "strain", "von_mises"),
    nonlinear=False,
    verification="analytical",
    caveat="Local material axes coincide with global axes; orientation rotation is not yet implemented.",
)
class OrthotropicElastic(Material):
    n_state = 0
    state_names: dict = {}

    def __init__(
        self,
        E1: float,
        E2: float,
        E3: float,
        nu12: float,
        nu13: float,
        nu23: float,
        G12: float,
        G13: float,
        G23: float,
        rho: float,
        **_ignored,
    ):
        # --- basic positivity -------------------------------------------
        for name, val in [
            ("E1", E1), ("E2", E2), ("E3", E3),
            ("G12", G12), ("G13", G13), ("G23", G23),
            ("rho", rho),
        ]:
            v = float(val)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"orthotropic: {name} must be finite and >0, got {val!r}")
        for name, val in [("nu12", nu12), ("nu13", nu13), ("nu23", nu23)]:
            v = float(val)
            if not np.isfinite(v) or not (-0.5 < v < 0.5):
                # allow slightly negative (auxetic) but keep within stability
                raise ValueError(f"orthotropic: {name} must be finite and in (-0.5, 0.5), got {val!r}")

        self.E1 = float(E1)
        self.E2 = float(E2)
        self.E3 = float(E3)
        self.nu12 = float(nu12)
        self.nu13 = float(nu13)
        self.nu23 = float(nu23)
        self.G12 = float(G12)
        self.G13 = float(G13)
        self.G23 = float(G23)
        self.rho = float(rho)

        S, D = _build_matrices(
            self.E1, self.E2, self.E3,
            self.nu12, self.nu13, self.nu23,
            self.G12, self.G13, self.G23,
        )
        # positive definiteness: S must be SPD
        # check eigenvalues (6) > 0 ; equivalently all leading principal minors >0
        # use eigenvalues for a clear message
        eig = np.linalg.eigvalsh(S)
        if np.min(eig) <= 0:
            raise ValueError(
                f"orthotropic: compliance matrix is not positive definite "
                f"(min eig {np.min(eig):.3e}); check E, nu, G combination. "
                f"S eigenvalues: {eig}"
            )
        # also check D symmetry already enforced
        self.S = S
        self.D0 = D
        # wave speed: conservative estimate from max diagonal stiffness
        # use max(D11,D22,D33) as longitudinal modulus
        M = float(np.max(np.diag(D)[:3]))
        self.wave_speed = float(np.sqrt(M / self.rho))

    # ------------------------------------------------------------------ #
    def update(self, strain: np.ndarray, state: np.ndarray, dt: float) -> np.ndarray:
        # strain (n,6) engineering, D0 (6,6)
        return strain @ self.D0.T

    def describe(self) -> str:
        return (
            f"E1={self.E1:.3e} E2={self.E2:.3e} E3={self.E3:.3e}  "
            f"nu12={self.nu12:.3f} nu13={self.nu13:.3f} nu23={self.nu23:.3f}  "
            f"G12={self.G12:.3e} G13={self.G13:.3e} G23={self.G23:.3e}"
        )
