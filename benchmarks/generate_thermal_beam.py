"""Temperature on beams: the meshes for cases 27/27b and the closed forms
they are checked against.

WHAT A TEMPERATURE FIELD DOES depends entirely on what is in the way, and the
two cases are the two halves of that:

  case_27   a TWO-SPAN CONTINUOUS elastic beam -- redundant, so the part of
            the free strain the structure refuses becomes force.
              * uniform dT with both ends held axially:  N = E A alpha dT
              * a linear gradient (curvature kappa = alpha dT/dy): the spans
                want to hog by kappa and the middle support will not let them,
                so a secondary moment appears there. Force method: each span's
                end rotation under the free curvature is kappa L / 2, so the
                kink at the support is kappa L, and a unit pair of moments
                opens 2 L / 3EI of it --
                    M_B = 3 E I kappa / 2,
                with end reactions -M_B / L (the ends lift) and 2 M_B / L at
                the middle support. Nothing else in the beam carries anything.

  case_27b  a SIMPLY SUPPORTED FIBRE beam under a NONLINEAR profile -- what a
            bridge code actually specifies. Determinate, so no reaction can
            resist it, and yet the section still cannot take the shape,
            because plane sections must stay plane. It keeps the linear part
            (an elongation and a curvature) and turns the rest into
            SELF-EQUILIBRATED stress:
                sigma(y) = E [eps0 - y kappa - alpha T(y)]
            with eps0 and kappa the two numbers that make int sigma dA = 0 and
            int sigma y dA = 0. Zero net force, zero net moment, real stress --
            tension at the soffit of a deck warmed on top, which is where the
            cracks a code's gradient is written to prevent would be.

`--check` prints every number the two contracts hold.
"""
import argparse
import pathlib

import numpy as np

# --- case_27: two 20 m spans, elastic ------------------------------------
SPAN, NEL_PER_SPAN = 20.0, 20
E, RHO, AREA, I = 2.1e11, 7850.0, 0.10, 9.70e-4
ALPHA = 1.2e-5
D_UNIFORM = 40.0          # K
GRADIENT = 15.0           # K/m of section depth

# --- case_27b: a 20 m simply supported fibre beam, 0.6 x 1.0 m -----------
E_C, ALPHA_C = 3.45e10, 1.0e-5
WIDTH, DEPTH = 0.6, 1.0
#: [y from the centroid (m), T (K)] -- a deck warmed on top, the shape a code
#: gradient has: all of it in the upper 0.4 m and nothing below.
PROFILE = [[-0.5, 0.0], [0.1, 0.0], [0.5, 20.0]]


def secondary_moment(alpha=ALPHA, gradient=GRADIENT, E=E, I=I):
    """The support moment of the two-span beam under a free curvature."""
    kappa = alpha * gradient
    M = 1.5 * E * I * kappa
    return {"kappa": kappa, "M_support": M, "R_end": -M / SPAN, "R_support": 2 * M / SPAN}


def restrained_force(alpha=ALPHA, dT=D_UNIFORM, E=E, area=AREA):
    return E * area * alpha * dT


def eigenstress(profile=None, depth=DEPTH, width=WIDTH, E=E_C, alpha=ALPHA_C, n=2000):
    """(eps0, kappa, y, sigma) of a determinate section under `profile`.

    The section is free, so the two resultants vanish; the two numbers that
    make them vanish are the elongation and the curvature the member takes on,
    and what is left over is the stress it carries anyway.
    """
    table = np.asarray(profile if profile is not None else PROFILE, float)
    y = np.linspace(-depth / 2, depth / 2, n)
    a = np.full(n, width * depth / n)
    T = np.interp(y, table[:, 0], table[:, 1])
    A = a.sum(); Iz = (a * y * y).sum()
    eps0 = (alpha * T * a).sum() / A               # int alpha T dA / A
    kappa = -(alpha * T * y * a).sum() / Iz        # sigma = E(eps0 - y kappa - alpha T)
    sigma = E * (eps0 - y * kappa - alpha * T)
    return {"eps0": float(eps0), "kappa": float(kappa), "y": y, "sigma": sigma,
            "N": float((sigma * a).sum()), "M": float((-sigma * y * a).sum()),
            "w_mid": float(-kappa * SPAN ** 2 / 8)}


def two_span_solution(alpha=ALPHA, gradient=GRADIENT, E=E, I=I, span=SPAN, nel=40):
    """The two-span beam solved here, by direct stiffness, away from the
    library entirely: 2 dofs per node (w, theta), the free curvature entered
    as each member's fixed-end moment pair, three supports.

    It is the same answer as the force-method closed form above -- printed
    side by side by `--check` -- and it also gives the deflections and
    reactions the contract holds, which the closed form would need three more
    integrations for.
    """
    kappa = alpha * gradient
    x = np.linspace(0.0, 2*span, nel + 1)
    n = len(x); K = np.zeros((2*n, 2*n)); f = np.zeros(2*n)
    for e in range(nel):
        le = x[e+1] - x[e]
        ke = E*I/le**3*np.array([[12, 6*le, -12, 6*le], [6*le, 4*le*le, -6*le, 2*le*le],
                                 [-12, -6*le, 12, -6*le], [6*le, 2*le*le, -6*le, 4*le*le]])
        d = [2*e, 2*e+1, 2*e+2, 2*e+3]
        K[np.ix_(d, d)] += ke
        # a member free to take the curvature carries the end moments
        # -EI kappa and +EI kappa; moving them to the right-hand side is the
        # equivalent load of the thermal field
        f[d[1]] += E*I*kappa; f[d[3]] -= E*I*kappa
    supports = [0, 2*(nel//2), 2*nel]                      # w at the three supports
    free = np.setdiff1d(np.arange(2*n), supports)
    u = np.zeros(2*n)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
    r = K @ u - f
    # THE MEMBER'S MOMENT IS NOT E I w''. The free curvature went into the
    # right-hand side, so it has to come back out at stress recovery:
    #     f_member = k u_e - f_eq,
    # and forgetting the second term is exactly the trap core/temperature.py
    # exists to avoid -- it was fallen into here first, and read 18.33 kN m
    # (E I kappa / 2) against the force method's 55.00.
    e = nel//2 - 1                                         # the element left of the support
    le = x[e+1] - x[e]
    d = [2*e, 2*e+1, 2*e+2, 2*e+3]
    ke = E*I/le**3*np.array([[12, 6*le, -12, 6*le], [6*le, 4*le*le, -6*le, 2*le*le],
                             [-12, -6*le, 12, -6*le], [6*le, 2*le*le, -6*le, 4*le*le]])
    f_eq = np.array([0.0, E*I*kappa, 0.0, -E*I*kappa])
    M_support = abs((ke @ u[d] - f_eq)[3])
    return {"kappa": kappa, "w": u[0::2], "x": x,
            "w_mid1": float(u[2*(nel//4)]), "R_end": float(r[0]),
            "R_support": float(r[2*(nel//2)]),
            "M_support": float(M_support)}


def write_two_span(path, span=SPAN, nel=NEL_PER_SPAN):
    """Two equal spans on three supports, with nodes at every quarter point."""
    x = np.concatenate([np.linspace(0.0, span, nel + 1),
                        np.linspace(span, 2 * span, nel + 1)[1:]])
    pts = {"left": 0, "support": nel, "right": 2 * nel,
           "mid1": nel // 2, "mid2": nel + nel // 2}
    return _write(path, x, pts, "beam")


def write_single_span(path, span=SPAN, nel=NEL_PER_SPAN):
    x = np.linspace(0.0, span, nel + 1)
    return _write(path, x, {"left": 0, "right": nel, "mid": nel // 2}, "beam")


def _write(path, x, pts, line_name):
    names = list(pts) + [line_name]
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names))]
    out += [f'0 {k + 1} "{nm}"' for k, nm in enumerate(names[:-1])]
    out += [f'1 {len(names)} "{line_name}"', "$EndPhysicalNames", "$Nodes", str(len(x))]
    out += [f"{k + 1} {v:.12g} 0 0" for k, v in enumerate(x)]
    out += ["$EndNodes", "$Elements", str(len(pts) + len(x) - 1)]
    eid = 1
    for k, nm in enumerate(names[:-1]):
        out.append(f"{eid} 15 2 {k + 1} {k + 1} {pts[nm] + 1}"); eid += 1
    for k in range(len(x) - 1):
        out.append(f"{eid} 1 2 {len(names)} {len(names)} {k + 1} {k + 2}"); eid += 1
    out += ["$EndElements", ""]
    pathlib.Path(path).write_text("\n".join(out), encoding="utf-8")
    return len(x) - 1, pts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "assets"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        print(f"case_27  two {SPAN:.0f} m spans, EA = {E*AREA:.3e} N, EI = {E*I:.3e} N m2, "
              f"alpha = {ALPHA:.1e}/K")
        print(f"  uniform {D_UNIFORM:.0f} K held at both ends: N = {restrained_force()/1e3:.1f} kN "
              f"(free elongation would be {ALPHA*D_UNIFORM*2*SPAN*1e3:.2f} mm)")
        s = secondary_moment(); d = two_span_solution()
        print(f"  the direct solve here agrees: M_support = {d['M_support']/1e3:.2f} kN m, "
              f"|R_end| = {abs(d['R_end'])/1e3:.2f} kN, midspan of span 1 moves "
              f"{d['w_mid1']*1e3:.3f} mm")
        print(f"  gradient {GRADIENT:.0f} K/m: kappa = {s['kappa']:.3e} /m, "
              f"M_support = {s['M_support']/1e3:.2f} kN m, "
              f"R_end = {s['R_end']/1e3:+.2f} kN, R_support = {s['R_support']/1e3:+.2f} kN")
        print(f"  (a single span under the same gradient would simply bow "
              f"{s['kappa']*SPAN**2/8*1e3:.2f} mm and carry nothing)")
        e = eigenstress()
        print(f"\ncase_27b {WIDTH} x {DEPTH} m fibre section, E = {E_C:.2e} Pa, "
              f"alpha = {ALPHA_C:.1e}/K, profile {PROFILE}")
        print(f"  it takes eps0 = {e['eps0']:.3e} and kappa = {e['kappa']:.3e} /m, "
              f"so midspan moves {e['w_mid']*1e3:+.3f} mm")
        print(f"  and carries N = {e['N']:.3e} N, M = {e['M']:.3e} N m -- nothing")
        for y in (-0.5, -0.25, 0.0, 0.1, 0.3, 0.5):
            k = int(np.argmin(abs(e["y"] - y)))
            print(f"     y = {y:+.2f}   sigma = {e['sigma'][k]/1e6:+7.3f} MPa")
        return
    out = pathlib.Path(a.out)
    n, pts = write_two_span(out / "thermal_two_span_40.msh")
    print(out / "thermal_two_span_40.msh", n, "elements", pts)
    n, pts = write_single_span(out / "thermal_span_20.msh")
    print(out / "thermal_span_20.msh", n, "elements", pts)


if __name__ == "__main__":
    main()
