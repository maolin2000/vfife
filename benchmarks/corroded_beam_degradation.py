"""M_u(rho) for Li et al.'s eight corroded beams, by SECTION EQUILIBRIUM.

    python benchmarks/corroded_beam_degradation.py

THIS IS THE DELIVERABLE, NOT A WORKAROUND (decision D3).

The four corrosion laws in `generate_corroded_pc_beam.py` turn one number --
the strand's mass loss -- into a different strand. This asks what that strand
does to the midspan section's capacity, and puts the answer next to the eight
measured capacities of Li et al. (2024) Table 2.

WHY THE SECTION AND NOT THE BEAM. The transient fibre-beam model gives the
wrong answer for every corroded beam, and v0.4 measured why: a bonded tendon
in a perfectly-bonded fibre beam sees, at a crack, the crack opening divided by
the ELEMENT LENGTH. That is a mesh quantity. Doubling h from 40 to 80 mm moves
the capacity 49% and the rupture deflection by 2.5x; tightening the residual
tolerance 100x changes nothing, so it is the converged solution's own
localisation, not numerics. A real strand shares the crack opening over a BOND
TRANSFER LENGTH (100-200 mm for a 15 mm strand), and giving the model that
length is a bond element the library does not have, and that work is shelved.

Section equilibrium has no element length, so it is not affected. For the
UNCORRODED control it agrees with the transient beam to 0.8% (58.96 against
58.54 kN m), which is what licenses its use on the corroded ones.

WHAT IT IS AND IS NOT. It is a moment-curvature analysis of the midspan
section, driven to N = 0 at each curvature, with the same fibres, the same
patch discretisation and the same constitutive laws the case file declares.
It therefore tests the CORROSION LAWS and the SECTION, and not the beam:
no localisation, no shear, no load-deflection curve, no descending branch.
The deflection axis of the measurement is unusable anyway (v0.4 --
about 1.4 mm of Figure 9's deflection is the testing machine), so capacity
against mass loss is what these eight beams can actually judge.

TWO COLUMNS, BECAUSE THE SECOND ONE IS THE POINT. `M area+f` applies only the
area and strength laws; `M +rupture` also stops the strand at its corroded
ultimate strain. The first is the control -- "would削面积 alone do?" -- and it
would not: 12.6% worst and the curve is NOT MONOTONE (A5 comes back up to 0.97
of A0 where the measurement is at 0.80). With the ductility law: 10.5% worst,
6.2% mean, the errors scattering 2.7 points against 6.3 without it. That is
the evidence that the ductility half of the corrosion model is necessary, and
it is the same evidence for which the beam model cannot be used.

(Since v1.0 the strand is OpenSees' Steel02, whose envelope stays under its
hardening asymptote; every capacity moved down 2.6-2.9%. Before it: 16.1%
worst without the ductility law, 9.9% worst and 4.1% mean with it.)
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT))
from generate_corroded_pc_beam import LI_BEAMS, corroded_strand   # noqa: E402
from vfife.elements.fiber_section import patch_fibers             # noqa: E402
from vfife.materials.base import create as mat_create             # noqa: E402

Y_P = -0.1185
PRESTRESS_0 = 7.786e+08
PATCH = [{"corners": [[0.0135, -0.1325], [0.0135, 0.1325],
                      [0.0885, 0.1325], [0.0885, -0.1325]], "ny": 4, "nz": 10},
         {"corners": [[-0.1515, -0.0445], [-0.1515, 0.0445],
                      [0.0135, 0.0445], [0.0135, -0.0445]], "ny": 12, "nz": 4}]
BAR_Y = np.array([0.0645, 0.0295, -0.1275])
BAR_A = np.array([4.0, 4.0, 2.0]) * 5.027e-5

YC, _ZC, AC, _ = patch_fibers(PATCH)


def moment_curvature(c, n=500, phi_max=0.30):
    """(phi, M, eps_strand) for one corroded strand `c`."""
    conc = mat_create("concrete02", fpc=3.57e+7, epsc0=0.002, fcu=7.1e+6,
                      epscu=0.0045, ft=3.22e+6, Ets=1.6e+9, rat=0.1, rho=2500)
    steel = mat_create("steel02", E=2.0e+11, Fy=3.224e+8, b=0.0055, R0=20.0)
    strand = mat_create("steel02", E=c["E"], Fy=c["Fy"], b=c["b"], R0=20.0)
    sc = np.zeros((len(YC), conc.n_state))
    ss = np.zeros((len(BAR_Y), steel.n_state))
    sp = np.zeros((1, strand.n_state))
    pre = c["prestress"] / c["E"]

    def forces(eps0, phi, commit):
        cs = sc if commit else sc.copy()
        st = ss if commit else ss.copy()
        pt = sp if commit else sp.copy()
        ec, es = eps0 - phi * YC, eps0 - phi * BAR_Y
        ep = eps0 - phi * Y_P + pre
        fc = conc.update(ec.reshape(-1, 1), cs, 0.0).ravel()
        fs = steel.update(es.reshape(-1, 1), st, 0.0).ravel()
        fp = strand.update(np.array([[ep]]), pt, 0.0).ravel()[0]
        n_ = (fc * AC).sum() + (fs * BAR_A).sum() + fp * c["area"]
        m_ = -((fc * AC * YC).sum() + (fs * BAR_A * BAR_Y).sum()
               + fp * c["area"] * Y_P)
        return n_, m_, ep

    rows, eps0 = [], 0.0
    for phi in np.linspace(phi_max / n, phi_max, n):
        a, b = eps0 - 2e-4, eps0 + 2e-4
        fa, fb = forces(a, phi, False)[0], forces(b, phi, False)[0]
        for _ in range(60):
            if abs(fb - fa) < 1e-30 or abs(fb) < 1.0:
                break
            x = b - fb * (b - a) / (fb - fa)
            a, fa, b = b, fb, x
            fb = forces(b, phi, False)[0]
        eps0 = b
        _n, m_, ep = forces(eps0, phi, True)
        rows.append((phi, m_, ep))
    return np.array(rows)


def main():
    print(f"{'beam':5}{'rho%':>6}{'M meas':>8}{'M area+f':>10}{'err':>7}"
          f"{'M +rupture':>12}{'err':>7}{'eps_u':>9}{'eps@Mmax':>10}  mode")
    out = []
    for nm, rho, m_meas in LI_BEAMS:
        c = corroded_strand(rho, PRESTRESS_0)
        r = moment_curvature(c)
        k = int(np.argmax(r[:, 1]))
        m_free = r[k, 1] / 1e+3
        alive = r[:, 2] < c["eps_u"]
        m_rupt = (r[alive, 1].max() / 1e+3) if alive.any() else float("nan")
        mode = "crush" if r[k, 2] < c["eps_u"] else "WIRE BREAK"
        print(f"{nm:5}{100*rho:6.2f}{m_meas:8.2f}{m_free:10.2f}"
              f"{100*(m_free-m_meas)/m_meas:6.1f}%{m_rupt:12.2f}"
              f"{100*(m_rupt-m_meas)/m_meas:6.1f}%{c['eps_u']:9.5f}"
              f"{r[k,2]:10.5f}  {mode}")
        out.append((rho, m_meas, m_free, m_rupt))
    a = np.array(out)
    print(f"\n normalised to A0:")
    print(f"{'rho%':>6}{'meas':>8}{'area+f':>9}{'+rupture':>10}")
    for rho, mm, mf, mr in a:
        print(f"{100*rho:6.2f}{mm/a[0,1]:8.3f}{mf/a[0,2]:9.3f}{mr/a[0,3]:10.3f}")
    for j, lbl in ((2, "area+strength"), (3, "+ rupture")):
        e = 100 * (a[:, j] - a[:, 1]) / a[:, 1]
        print(f" {lbl:14} worst {np.abs(e).max():5.1f}%   mean |err| "
              f"{np.abs(e).mean():5.1f}%")


if __name__ == "__main__":
    main()
