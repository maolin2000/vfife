"""Linear viscoelastic (Prony) -- Maxwell chain, shear relaxation only."""
from __future__ import annotations
import numpy as np
from ..voigt import dilatational_wave_speed
from .base import Material, register

@register("viscoelastic", kinematics=("solid_3d",), backends=("numpy",), analysis_modes=("dynamic",), output_fields=("stress","strain","von_mises"), nonlinear=True, verification="analytical", caveat="Shear Prony only; bulk elastic. Dynamic analysis only (physical time).")
class Viscoelastic(Material):
    n_state = 0
    state_names: dict = {}

    def __init__(self, E: float, nu: float, rho: float, g: list[float] | None = None, tau: list[float] | None = None, **_ignored):
        self.E = float(E); self.nu = float(nu); self.rho = float(rho)
        g = list(g) if g is not None else []
        tau = list(tau) if tau is not None else []
        if len(g) != len(tau):
            raise ValueError("viscoelastic: g and tau must have same length")
        self.g = np.asarray(g, dtype=float)
        self.tau = np.asarray(tau, dtype=float)
        if np.any(self.g < 0) or np.any(self.g >= 1):
            raise ValueError("viscoelastic g must be in [0,1)")
        if np.any(self.tau <= 0):
            raise ValueError("viscoelastic tau must be >0")
        if self.g.size and self.g.sum() >= 1.0:
            raise ValueError("viscoelastic sum(g) must be <1 (g_inf>0)")
        self.n_prony = len(self.g)
        self.G0 = self.E/(2*(1+self.nu))
        self.K = self.E/(3*(1-2*self.nu)) if self.nu < 0.5 else self.E/3e-12  # avoid div by 0 for nu~0.5
        self.G_inf = self.G0*(1 - self.g.sum()) if self.n_prony else self.G0
        self.wave_speed = dilatational_wave_speed(self.E, self.nu, self.rho)
        # state: n_prony *6 for q + 6 for prev_strain
        self.n_state = self.n_prony*6 + 6
        # dynamic state_names for introspection
        names = {}
        for i in range(self.n_prony):
            names[f"q{i}"] = slice(i*6, (i+1)*6)
        names["prev_strain"] = slice(self.n_prony*6, self.n_prony*6+6)
        self.state_names = names

    def init_state(self, n_points: int, dtype=np.float64):
        return np.zeros((n_points, self.n_state), dtype=dtype)

    def update(self, strain: np.ndarray, state: np.ndarray, dt: float) -> np.ndarray:
        n = strain.shape[0]
        tr = strain[:,0] + strain[:,1] + strain[:,2]
        # deviatoric tensorial
        e_dev_xx = strain[:,0] - tr/3
        e_dev_yy = strain[:,1] - tr/3
        e_dev_zz = strain[:,2] - tr/3
        # prev strain
        if self.n_prony:
            prev = state[:, self.n_prony*6 : self.n_prony*6+6]
            prev_tr = prev[:,0] + prev[:,1] + prev[:,2]
            d_eps = strain - prev
            d_tr = tr - prev_tr
            d_e_xx = d_eps[:,0] - d_tr/3
            d_e_yy = d_eps[:,1] - d_tr/3
            d_e_zz = d_eps[:,2] - d_tr/3
            d_gam_xy = d_eps[:,3]
            d_gam_yz = d_eps[:,4]
            d_gam_xz = d_eps[:,5]
            # update each Prony branch
            for i in range(self.n_prony):
                sl = slice(i*6, (i+1)*6)
                q_old = state[:, sl]
                exp_i = np.exp(-dt/self.tau[i]) if dt>0 else 1.0
                fac = (1 - exp_i)*self.tau[i]/dt if dt>1e-12 else 1.0
                coeff = self.G0*self.g[i]*fac
                # in-place update via copy
                q_new = np.empty_like(q_old)
                q_new[:,0] = exp_i*q_old[:,0] + 2*coeff*d_e_xx
                q_new[:,1] = exp_i*q_old[:,1] + 2*coeff*d_e_yy
                q_new[:,2] = exp_i*q_old[:,2] + 2*coeff*d_e_zz
                q_new[:,3] = exp_i*q_old[:,3] + coeff*d_gam_xy
                q_new[:,4] = exp_i*q_old[:,4] + coeff*d_gam_yz
                q_new[:,5] = exp_i*q_old[:,5] + coeff*d_gam_xz
                state[:, sl] = q_new
            # sum q
            q_sum = np.zeros((n,6))
            for i in range(self.n_prony):
                q_sum += state[:, i*6:(i+1)*6]
            # update prev
            state[:, self.n_prony*6 : self.n_prony*6+6] = strain
        else:
            q_sum = np.zeros((n,6))
        # total stress
        stress = np.empty((n,6))
        stress[:,0] = self.K*tr + 2*self.G_inf*e_dev_xx + q_sum[:,0]
        stress[:,1] = self.K*tr + 2*self.G_inf*e_dev_yy + q_sum[:,1]
        stress[:,2] = self.K*tr + 2*self.G_inf*e_dev_zz + q_sum[:,2]
        stress[:,3] = self.G_inf*strain[:,3] + q_sum[:,3]
        stress[:,4] = self.G_inf*strain[:,4] + q_sum[:,4]
        stress[:,5] = self.G_inf*strain[:,5] + q_sum[:,5]
        return stress

    def describe(self) -> str:
        return f"  G0 {self.G0:.3e} G_inf {self.G_inf:.3e} K {self.K:.3e} prony {self.n_prony} sum_g {self.g.sum() if self.n_prony else 0:.3f}"
