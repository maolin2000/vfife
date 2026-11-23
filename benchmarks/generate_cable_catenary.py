"""Irvine's elastic catenary: the closed form and a Gmsh mesh on it (cases/case_22).

A cable of unstressed length L0, axial stiffness EA and weight w per unit
UNSTRESSED length hangs between (0, 0) and (l, h). With H the horizontal
tension (constant along the cable), V the vertical reaction at the left
support and s the Lagrangian coordinate along the unstressed cable
(Irvine, "Cable Structures", 1981, eqs 2.7-2.8; z measured downward):

    x(s) = H s / EA + (H / w) [asinh(V / H) - asinh((V - w s) / H)]
    z(s) = (w s / EA) (V / w - s / 2) + (1 / w) [sqrt(H^2 + V^2) - sqrt(H^2 + (V - w s)^2)]

H and V are the two unknowns of x(L0) = l, z(L0) = h. Everything a cable
verification can ask -- the reactions, the sag at any s, the tension
profile -- follows in closed form.

THE MESH IS THE STRAIGHT CHORD between the supports, n equal elements, and
the cable's `unstressed_length` L0 is longer than the chord: the run has
to FIND the catenary, 8.8 m of sag away from where it starts -- dynamic
relaxation doing what it was invented for. The weight the solver puts on
the cable is rho g A per MESH metre, i.e. w = rho g A l / L0 per unstressed
metre (uniform, since the chord elements are equal), and that is the w the
closed form is evaluated with.
"""
import argparse
import pathlib

import numpy as np
from scipy.optimize import fsolve

G = 9.80665


def catenary(l, h, L0, EA, w):
    """(H, V) of the elastic catenary, and x(s), z(s) as callables."""
    def xz(s, H, V):
        x = H * s / EA + (H / w) * (np.arcsinh(V / H) - np.arcsinh((V - w * s) / H))
        z = (w * s / EA) * (V / w - s / 2) + (1 / w) * (np.sqrt(H ** 2 + V ** 2)
                                                        - np.sqrt(H ** 2 + (V - w * s) ** 2))
        return x, z

    def residual(p):
        H, V = p
        x, z = xz(L0, H, V)
        return [x - l, z - h]

    # start from the parabola: sag d from L0 ~ l + 8 d^2 / 3 l, H = w l^2 / 8 d
    d = max(np.sqrt(3 * l * max(L0 - l, 1e-6) / 8), 1e-3)
    H0 = w * l ** 2 / (8 * d)
    H, V = fsolve(residual, [H0, w * L0 / 2], xtol=1e-13)
    # fsolve can report slow progress at the tolerance floor while the
    # residual is already exactly zero; what matters is checked here
    rx, rz = residual([H, V])
    if abs(rx) > 1e-9 * max(l, 1.0) or abs(rz) > 1e-9 * max(l, 1.0):
        raise RuntimeError(f"catenary did not converge: residual {rx:g}, {rz:g}")
    return float(H), float(V), (lambda s: xz(s, H, V))


def build(path, l=100.0, h=0.0, L0=102.0, E=2.0e10, A=0.05, rho=7850.0, n=40):
    """Write the straight-chord mesh; return the closed form it must reach."""
    EA = E * A
    chord = float(np.hypot(l, h))
    w = rho * G * A * chord / L0            # the solver's weight, per unstressed metre
    H, V, xz = catenary(l, h, L0, EA, w)
    t = np.linspace(0.0, 1.0, n + 1)
    nodes = np.column_stack([l * t, np.zeros_like(t), -h * t])
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat",
           "$PhysicalNames", "4", '0 1 "left"', '0 2 "right"', '0 3 "mid"', '1 4 "cable"',
           "$EndPhysicalNames", "$Nodes", str(n + 1)]
    out += [f"{k + 1} {px:.12g} {py:.12g} {pz:.12g}" for k, (px, py, pz) in enumerate(nodes)]
    out += ["$EndNodes", "$Elements", str(n + 3),
            "1 15 2 1 1 1", f"2 15 2 2 2 {n + 1}", f"3 15 2 3 3 {n // 2 + 1}"]
    out += [f"{k + 4} 1 2 4 4 {k + 1} {k + 2}" for k in range(n)]
    out += ["$EndElements", ""]
    pathlib.Path(path).write_text("\n".join(out), encoding="utf-8")
    # the mid NODE is s = L0/2 along the unstressed cable (n even): its
    # closed-form position, and the sag there
    xm, zm = xz(L0 / 2)
    return {"H": H, "V": V, "w": w, "sag_mid": float(zm), "x_mid": float(xm),
            "T_support": float(np.hypot(H, V)), "T_mid": float(np.hypot(H, V - w * L0 / 2)),
            "strain_mid": float(np.hypot(H, V - w * L0 / 2) / EA),
            "chord": chord, "L0": L0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="benchmarks/assets/cable_catenary_40.msh")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    info = build(a.out, n=a.n)
    print("wrote", a.out)
    for k, v in info.items():
        print(f"  {k:12s} {v:.9g}")
