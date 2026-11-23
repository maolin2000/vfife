"""A simply supported beam crossed by a moving force, and the closed-form
answer to check it against (cases/case_26).

THE PROBLEM. A constant force P travels at constant speed c across a simply
supported Bernoulli-Euler beam of span L, bending stiffness EI and mass per
unit length mu, initially at rest. It is the first problem in the moving-load
literature (Willis and Stokes, 1849, on the Chester bridge collapse) and the
one every textbook solves, because separating the modes turns it into one
forced oscillator per mode:

    mu (L/2) q_j'' + mu (L/2) omega_j^2 q_j = P sin(Omega_j t),
        omega_j = (j pi / L)^2 sqrt(EI / mu),      Omega_j = j pi c / L

-- the modal force is P sin(j pi x_P / L) with x_P = c t, and the load enters
the j-th mode at the frequency at which it sweeps that mode's half-waves.
From rest,

    q_j(t) = (2P / (mu L)) / (omega_j^2 - Omega_j^2)
             * [ sin(Omega_j t) - (Omega_j / omega_j) sin(omega_j t) ]

and v(x, t) = sum_j q_j(t) sin(j pi x / L). The first bracket is the load
riding across (the quasi-static part); the second is the free vibration the
beam keeps once it has been disturbed, at its own frequency rather than the
load's. Their ratio alpha = Omega_1 / omega_1 = c / (2 f_1 L) is the speed
parameter: alpha -> 0 is the influence line, alpha -> 1 is resonance.
This is Fryba's solution (Vibration of Solids and Structures under Moving
Loads, section 1.2) written in the modal form it is derived in.

THE BEAM here is 20 m of it with f_1 = 2.0 Hz -- a short highway span -- and
the force crosses at 40 m/s, alpha = 0.50, fast enough that the beam's own
vibration is plainly in the answer: midspan peaks at 1.70 times the static
P L^3 / 48 EI, and it peaks a third of the crossing AFTER the load has passed
midspan.
"""
import argparse
import pathlib

import numpy as np

SPAN, NEL = 20.0, 20
E, RHO, AREA, I = 2.1e11, 7850.0, 0.10, 9.70e-4
SPEED, FORCE = 40.0, 1.0e5


def properties(E=E, rho=RHO, area=AREA, I=I, span=SPAN):
    mu = rho * area
    omega1 = (np.pi / span) ** 2 * np.sqrt(E * I / mu)
    return {"mu": mu, "EI": E * I, "omega1": float(omega1), "f1": float(omega1 / (2 * np.pi)),
            "static": FORCE * span ** 3 / (48 * E * I)}


def fryba(x, t, P=FORCE, c=SPEED, span=SPAN, E=E, I=I, mu=RHO * AREA, modes=40):
    """Midspan (or any x) deflection while the force is ON the beam, positive
    down. `t` may be an array; values with c*t outside [0, L] are not what
    this returns (the free vibration after the load leaves needs the exit
    state as an initial condition, which is not what is checked here)."""
    t = np.atleast_1d(np.asarray(t, float))
    out = np.zeros((len(t),), float)
    for j in range(1, modes + 1):
        om = (j * np.pi / span) ** 2 * np.sqrt(E * I / mu)
        Om = j * np.pi * c / span
        if abs(om - Om) < 1e-9 * om:
            raise ValueError(f"mode {j} is at resonance with the load; the series needs its limit")
        q = (2 * P / (mu * span)) / (om ** 2 - Om ** 2) * (np.sin(Om * t) - (Om / om) * np.sin(om * t))
        out += q * np.sin(j * np.pi * x / span)
    return out


def write_mesh(path, span=SPAN, nel=NEL):
    """A line mesh of the span with nodes at the ends, midspan and the
    quarter points, and every node in the `deck` group the load travels."""
    x = np.linspace(0.0, span, nel + 1)
    pts = {"left": 0, "right": nel, "mid": nel // 2, "quarter": nel // 4}
    names = list(pts) + ["deck"]
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names))]
    out += [f'0 {k + 1} "{nm}"' for k, nm in enumerate(names[:-1])]
    out += [f'1 {len(names)} "deck"', "$EndPhysicalNames", "$Nodes", str(len(x))]
    out += [f"{k + 1} {v:.12g} 0 0" for k, v in enumerate(x)]
    out += ["$EndNodes", "$Elements", str(len(pts) + nel)]
    eid = 1
    for k, nm in enumerate(names[:-1]):
        out.append(f"{eid} 15 2 {k + 1} {k + 1} {pts[nm] + 1}"); eid += 1
    for k in range(nel):
        out.append(f"{eid} 1 2 {len(names)} {len(names)} {k + 1} {k + 2}"); eid += 1
    out += ["$EndElements", ""]
    pathlib.Path(path).write_text("\n".join(out), encoding="utf-8")
    return nel, pts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "assets"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        p = properties()
        cross = SPAN / SPEED
        t = np.linspace(0, cross, 2001)
        v = fryba(SPAN / 2, t)
        print(f"span {SPAN} m, EI = {p['EI']:.3e} N m2, mu = {p['mu']:.1f} kg/m, "
              f"f1 = {p['f1']:.3f} Hz")
        print(f"force {FORCE / 1e3:.0f} kN at {SPEED} m/s: crossing {cross:.3f} s, "
              f"alpha = {np.pi * SPEED / SPAN / p['omega1']:.3f}")
        print(f"static midspan {p['static'] * 1e3:.3f} mm; moving peak "
              f"{v.max() * 1e3:.3f} mm at t = {t[v.argmax()]:.4f} s "
              f"(amplification {v.max() / p['static']:.3f})")
        print(f"at the exit (t = {cross:.3f} s) midspan is {v[-1] * 1e3:+.3f} mm")
        return
    out = pathlib.Path(a.out)
    nel, pts = write_mesh(out / "moving_load_beam_20.msh")
    print(out / "moving_load_beam_20.msh", nel, "elements", pts)


if __name__ == "__main__":
    main()
