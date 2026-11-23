"""The two closed forms creep has, on one bar.

    python benchmarks/generate_creep_bar.py --check

CREEP, under a stress applied at age t0 and held: the strain grows to
`(1 + phi(t, t0)) sigma / Ec(t0)` and the member carries exactly what it
carried on day one. Nothing is redundant, so nothing redistributes.

RELAXATION, at a strain applied at t0 and held: the stress decays to
`R(t, t0)` times what it started at, because the creep strain eats the elastic
one. R is not `1/(1 + phi)` -- that is the effective-modulus guess -- and it is
not the age-adjusted `1 - phi/(1 + chi phi)` either except in so far as that
formula's `chi` was fitted to it. It is the solution of the superposition
integral, and `core.creep.relaxation` solves it on a fine log grid; the case
solves the same thing with the structure in the way.

The creep function used here is a STANDARD LINEAR SOLID,
`phi(d) = phi_inf (1 - exp(-d/tau))`, for one reason: its relaxation modulus
is exact,

    R(d) / E = [1 + phi_inf exp(-d (1 + phi_inf) / tau)] / (1 + phi_inf)

so the relaxation check is against algebra and not against another numerical
method. `core.creep.relaxation` reproduces it to 1e-4 with the 400-point curve
below and to 4e-7 with a 6400-point one -- the residual is the log-linear
interpolation of the creep function, not the integral -- and the case is then
checked against `relaxation`.
"""
from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np

#: the bar
L, A, E = 10.0, 0.01, 3.0e+10          #: m, m2, Pa at the reference age
P = 3.0e+5                             #: N, applied at T0 and held
T0, T_END = 28.0, 1028.0               #: days
PHI_INF, TAU = 2.0, 40.0               #: the standard linear solid
NEL = 8


def creep_spec(n=400):
    """The standard linear solid as a measured `curve`, dense enough in log
    time that the interpolation is not what is being tested."""
    d = np.r_[0.0, np.exp(np.linspace(math.log(1e-4), math.log(1e+6), n))]
    return {"model": "curve",
            "curve": np.column_stack([d, PHI_INF * (1.0 - np.exp(-d / TAU))]).tolist()}


def phi(t, t0=T0):
    return PHI_INF * (1.0 - math.exp(-(t - t0) / TAU))


def elongation(t):
    """P L / (E A) * (1 + phi): the free bar, which carries P throughout."""
    return P * L / (E * A) * (1.0 + phi(t))


def relaxation_exact(t, t0=T0):
    """R(t, t0) / E for the standard linear solid -- the Zener model's own
    relaxation modulus, which is where this creep function was chosen from."""
    d = t - t0
    return (1.0 + PHI_INF * math.exp(-d * (1.0 + PHI_INF) / TAU)) / (1.0 + PHI_INF)


# --------------------------------------------------------------------------- #
#  Prestress loss: the three time-dependent losses on one member, and the
#  classical closed form they are usually hand-computed with.
# --------------------------------------------------------------------------- #
#: the pretensioned bar of case_29f/29g. A_c is the gross patch area; the
#: tendon's own area is added on top of it by the fibre section, which is the
#: usual small double count and is worth 1.4% of A_c here.
A_C, A_P, E_P = 0.01, 1.40e-4, 1.95e+11
SIG_PI = 1.488e+9                       #: 0.8 f_pu of an 1860 MPa strand
EPS_SH_INF = -3.0e-4                    #: the shrinkage curve's asymptote
RHO_1000, K_REL = 0.031, 0.21           #: Ghali (2000) Fig. 7, CEB-FIP low-relax


def shrinkage_spec(tau=60.0, n=200):
    """`eps_sh(t) = eps_inf (1 - exp(-(t - ts)/tau))` as a measured curve, so
    the closed form below can use the same numbers the case does."""
    d = np.r_[0.0, np.exp(np.linspace(math.log(1e-3), math.log(1e+5), n))]
    return {"model": "curve", "drying_start": T0,
            "curve": np.column_stack([d, EPS_SH_INF * (1.0 - np.exp(-d / tau))]).tolist()}


def eps_sh(t, tau=60.0):
    return EPS_SH_INF * (1.0 - math.exp(-(t - T0) / tau)) if t > T0 else 0.0


def relax_loss(t, t0=T0):
    """r(t) = rho_1000 (hours/1000)^k -- the shape in `core.relaxation`."""
    h = (t - t0) * 24.0
    return RHO_1000 * (h / 1000.0) ** K_REL if h > 0.0 else 0.0


def prestress_loss(t, chi=0.8, creep_on=True, shrink_on=True, relax_on=True):
    """`(d_sigma_p, sigma_p)` at age `t` on the free pretensioned bar, by the
    age-adjusted effective modulus method.

    DERIVED HERE rather than quoted, so that every symbol in it is one of this
    file's own. A centroidal tendon on a bar free to shorten: the concrete's
    strain change between t0 and t is its creep under the stress it already
    had, plus the elastic response to the stress CHANGE at the age-adjusted
    modulus, plus shrinkage,

        d_eps_c = phi sig_c0 / Ec + d_sig_c (1 + chi phi) / Ec + eps_sh

    the tendon follows that strain and loses its own relaxation on top,

        d_sig_p = Ep d_eps_c - d_sig_pr

    and the member is self-equilibrated, `A_p d_sig_p + A_c d_sig_c = 0`. Put
    together:

        d_sig_p [1 + n (1 + chi phi)] = Ep [phi sig_c0 / Ec + eps_sh] - d_sig_pr

    with `n = Ep A_p / (Ec A_c)`. chi = 1 is the plain effective-modulus
    method and chi = 0 is no relaxation of the concrete's own response at all,
    so the two bracket the step-by-step integral the solver runs; chi = 0.8 is
    the usual fitted value.
    """
    sig_p0 = SIG_PI / (1.0 + E_P * A_P / (E * A_C))       # after elastic shortening
    sig_c0 = -A_P * sig_p0 / A_C
    ph = phi(t) if creep_on else 0.0
    esh = eps_sh(t) if shrink_on else 0.0
    dpr = SIG_PI * relax_loss(t) if relax_on else 0.0
    n = E_P * A_P / (E * A_C)
    d = (E_P * (ph * sig_c0 / E + esh) - dpr) / (1.0 + n * (1.0 + chi * ph))
    return d, sig_p0 + d


def member_strain(t, chi=0.8, **kw):
    """The concrete's total strain at `t` on the same bar: its elastic part at
    transfer plus what the tendon's stress change says the member did."""
    sig_p0 = SIG_PI / (1.0 + E_P * A_P / (E * A_C))
    d, _ = prestress_loss(t, chi=chi, **kw)
    dpr = SIG_PI * relax_loss(t) if kw.get("relax_on", True) else 0.0
    return -A_P * sig_p0 / (A_C * E) + (d + dpr) / E_P


def write_mesh(path, nel=NEL):
    x = np.linspace(0.0, L, nel + 1)
    pts = {"left": 0, "right": nel, "mid": nel // 2}
    names = list(pts) + ["bar"]
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names))]
    out += [f'0 {k + 1} "{nm}"' for k, nm in enumerate(pts)]
    out += [f'1 {len(names)} "bar"', "$EndPhysicalNames", "$Nodes", str(len(x))]
    out += [f"{k + 1} {v:.12g} 0 0" for k, v in enumerate(x)]
    out += ["$EndNodes", "$Elements", str(len(pts) + nel)]
    eid = 1
    for k, nm in enumerate(pts):
        out.append(f"{eid} 15 2 {k + 1} {k + 1} {pts[nm] + 1}")
        eid += 1
    for k in range(nel):
        out.append(f"{eid} 1 2 {len(names)} {len(names)} {k + 1} {k + 2}")
        eid += 1
    out += ["$EndElements", ""]
    pathlib.Path(path).write_text("\n".join(out), encoding="utf-8")
    return nel


def write_pair_mesh(path, nel=NEL):
    """Two bars side by side for case_29c: the same member, free at one end
    and held at both.

    They share a mesh so that one analysis answers both halves of the same
    question -- a free member shrinks and carries nothing, a held one carries
    what it was stopped from doing -- under one time axis and one material.
    """
    x = np.linspace(0.0, L, nel + 1)
    n = nel + 1
    pts = {"free_left": 1, "free_right": n, "held_left": n + 1, "held_right": 2 * n}
    names = list(pts) + ["bars"]
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names))]
    out += [f'0 {k + 1} "{nm}"' for k, nm in enumerate(pts)]
    out += [f'1 {len(names)} "bars"', "$EndPhysicalNames", "$Nodes", str(2 * n)]
    out += [f"{k + 1} {v:.12g} 0 0" for k, v in enumerate(x)]
    out += [f"{n + k + 1} {v:.12g} 2 0" for k, v in enumerate(x)]
    out += ["$EndNodes", "$Elements", str(len(pts) + 2 * nel)]
    eid = 1
    for k, nm in enumerate(pts):
        out.append(f"{eid} 15 2 {k + 1} {k + 1} {pts[nm]}")
        eid += 1
    for base in (0, n):
        for k in range(nel):
            out.append(f"{eid} 1 2 {len(names)} {len(names)} "
                       f"{base + k + 1} {base + k + 2}")
            eid += 1
    out += ["$EndElements", ""]
    pathlib.Path(path).write_text("\n".join(out), encoding="utf-8")
    return 2 * nel


def _check():
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from vfife.core import creep as C

    spec = creep_spec()
    print("CREEP under a held load: the bar carries P = %.0f kN throughout and"
          " simply gets longer" % (P / 1e+3))
    print("   age (days)      phi     elongation (mm)")
    for t in (T0, 38.0, 68.0, 128.0, 428.0, T_END):
        print("   %9.1f  %7.3f   %10.4f" % (t, phi(t), 1e+3 * elongation(t)))
    print("   the elastic part is %.4f mm; at %.0f days it is %.3f times that"
          % (1e+3 * P * L / (E * A), T_END, 1.0 + phi(T_END)))

    print("\nRELAXATION at a held strain: the Zener model's own relaxation"
          " modulus, and core.creep solving the superposition integral")
    print("   age (days)   exact R/E    core.creep    err")
    for t in (38.0, 68.0, 128.0, 428.0, T_END):
        ex = relaxation_exact(t)
        num = C.relaxation(spec, t, T0, n=2000)
        print("   %9.1f   %9.5f    %9.5f   %+.1e" % (t, ex, num, num - ex))
    print("   1/(1+phi_inf) = %.5f is the effective-modulus guess; the answer"
          " tends to it only as t -> inf" % (1.0 / (1.0 + PHI_INF)))
    print("   and what is left of that error is the CURVE, not the integral:"
          " the creep function")
    print("   reaches the solver as a table read log-linearly, and")
    for npts in (400, 1600, 6400):
        s_ = creep_spec(npts)
        worst = max(abs(C.relaxation(s_, t, T0, n=2000) - relaxation_exact(t))
                    for t in (38.0, 68.0, 128.0, 428.0, T_END))
        print("      %5d points -> worst |err| %.1e" % (npts, worst))

    print("\nPRESTRESS LOSS -- the three losses on one member (case_29f, case_29g)")
    print("  the tendon: A_p = %.2f mm2 at %.0f MPa = %.2f kN jacked; elastic"
          % (A_P * 1e+6, SIG_PI / 1e+6, SIG_PI * A_P / 1e+3))
    print("  shortening on A_c = %.0f cm2 takes it to %.1f MPa at transfer"
          % (A_C * 1e+4, SIG_PI / (1.0 + E_P * A_P / (E * A_C)) / 1e+6))
    print("  RELAXATION ALONE, both ends held (case_29f): sigma_p / sigma_pi"
          " = 1 - r(t), exactly")
    print("     t (h)       r(t)     sigma_p (MPa)")
    for h in (0.0, 1000.0, 10000.0, 100000.0):
        t = T0 + h / 24.0
        print("   %8.0f   %8.5f     %10.2f" % (h, relax_loss(t),
                                               SIG_PI * (1 - relax_loss(t)) / 1e+6))
    print("  ALL THREE, one end free (case_29g): the age-adjusted effective")
    print("  modulus method derived in this file. chi = 0 and chi = 1 bracket")
    print("  the step-by-step integral; chi = 0.8 is the usual fitted value.")
    print("     age (d)   chi=0     chi=0.8    chi=1.0    sigma_p (MPa)")
    for t in (T0, 128.0, 444.667, T_END):
        v = [prestress_loss(t, chi=c)[1] / 1e+6 for c in (0.0, 0.8, 1.0)]
        print("   %9.1f %9.2f %9.2f %9.2f" % (t, v[0], v[1], v[2]))
    print("     and the member strain at %.0f d: %.6f (chi = 0.8)"
          % (T_END, member_strain(T_END)))
    print("  EACH LOSS ALONE, at %.0f days, against all three together --" % T_END)
    print("  the total is LESS than the sum, because each one relieves the")
    print("  concrete and so reduces the next:")
    base = prestress_loss(T_END, creep_on=False, shrink_on=False, relax_on=False)[1]
    parts = {}
    for nm, kw in (("creep", dict(shrink_on=False, relax_on=False)),
                   ("shrinkage", dict(creep_on=False, relax_on=False)),
                   ("relaxation", dict(creep_on=False, shrink_on=False))):
        parts[nm] = base - prestress_loss(T_END, **kw)[1]
        print("     %-11s %8.2f MPa" % (nm, parts[nm] / 1e+6))
    total = base - prestress_loss(T_END)[1]
    print("     %-11s %8.2f MPa   sum of the three %8.2f (%.1f%% more)"
          % ("all three", total / 1e+6, sum(parts.values()) / 1e+6,
             100 * (sum(parts.values()) / total - 1)))

    print("\nWHY THE TWO ARE DIFFERENT: the same creep function takes the free"
          " bar to %.2f times"
          % (1.0 + phi(T_END)))
    print("   its elastic strain and the held bar to %.2f times its elastic"
          " stress. Creep is not a" % relaxation_exact(T_END))
    print("   softening of the material -- nothing about E changed -- it is"
          " strain arriving late,")
    print("   and what that does depends entirely on what is in the way.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "assets"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        _check()
        return 0
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "creep_bar_8.msh"
    print(p, write_mesh(p), "elements")
    p2 = out / "creep_bar_pair_8.msh"
    print(p2, write_pair_mesh(p2), "elements")
    print("\ncreep curve for the case (a standard linear solid, phi_inf = %.1f,"
          " tau = %.0f d):" % (PHI_INF, TAU))
    d = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 400.0, 1000.0]
    print("    curve: [" + ", ".join("[%g, %.5f]" % (v, PHI_INF * (1 - math.exp(-v / TAU)))
                                     for v in d) + "]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
