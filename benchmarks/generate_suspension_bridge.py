"""A single-span suspension bridge (cases/case_23): mesh, exact dead-load
cable lengths, and the linearised deflection theory for a live load.

GEOMETRY (x along the span, z up, y out of plane)
    deck    31 nodes at z = 0, x = 0 .. l every s = l/n; beam2 elements;
            simply supported at both ends
    cable   31 nodes on the parabola z = c + f (1 - 4 x (l - x) / l^2),
            c the midspan clearance; truss2 elements; ends fixed (rigid
            tower tops, anchored)
    hangers 29 vertical truss2 elements, cable node k to deck node k

DEAD LOAD is the deck's weight, w per metre, applied as w s at every
interior deck node. Nothing else weighs anything: with the cable and
hangers weightless the dead-load cable is EXACTLY the parabola through the
hanger points and the whole form is closed-form (Irvine, "Cable Structures",
ch. 3; any bridge text):

    H_w = w l^2 / (8 f)            horizontal tension
    P   = w s                      every hanger's force
    T_e = H_w L0_e / s             tension of cable element e (funicular polygon)
    L_u = L0 / (1 + T / (E A))     the unstressed length that holds it there

The deck carries nothing at dead load -- the girder was built to profile
and the hangers took its weight -- so its unstressed lengths are its mesh
lengths and it needs no finding.

LIVE LOAD p per metre over the whole span: the linearised deflection theory
(Irvine ch. 6; Timoshenko) with an elastic cable, inextensible vertical
hangers and the girder simply supported:

    E I v'''' - (H_w + h) v'' = p - h (8 f / l^2)
    h L_E / (E_c A_c) = (8 f / l^2) integral_0^l v dx,   L_E = l (1 + 8 (f/l)^2)

v(x) has the closed form of a tensioned beam under a uniform load, and h
follows from the compatibility equation by a one-dimensional solve
(`deflection_theory` below). The nonlinear model is expected a few percent
off the linearised theory at p = w/5, and the contract says by how much.
"""
import argparse
import pathlib

import numpy as np
from scipy.optimize import brentq

DEFAULTS = dict(l=300.0, f=30.0, c=5.0, n=30, w=1.0e5, p=2.0e4,
                E=2.0e11, A_c=0.10, A_h=0.02, A_d=0.5, I_d=1.0)


def dead_load_form(l, f, c, n, w, E, A_c, A_h, **_):
    """Node coordinates and the exact unstressed lengths of cable and hangers."""
    s = l / n
    x = np.linspace(0.0, l, n + 1)
    z_c = c + f * (1.0 - 4.0 * x * (l - x) / l ** 2)
    H = w * l ** 2 / (8.0 * f)
    P = w * s
    # cable elements k -> k+1
    L0_c = np.hypot(np.diff(x), np.diff(z_c))
    T_c = H * L0_c / s
    Lu_c = L0_c / (1.0 + T_c / (E * A_c))
    # hangers at interior nodes 1 .. n-1
    L0_h = z_c[1:-1]
    Lu_h = L0_h / (1.0 + P / (E * A_h))
    return {"x": x, "z_cable": z_c, "H": H, "P": P, "T_cable": T_c,
            "Lu_cable": Lu_c, "Lu_hangers": Lu_h, "s": s}


def deflection_theory(l, f, p, w, E, A_c, I_d, **_):
    """(h, v_mid, v(x)) for a uniform live load p over the whole span."""
    H_w = w * l ** 2 / (8.0 * f)
    L_E = l * (1.0 + 8.0 * (f / l) ** 2)
    EI = E * I_d
    k = 8.0 * f / l ** 2

    def v_of(h):
        H = H_w + h
        q = p - h * k
        beta = np.sqrt(H / EI)

        def v(x):
            return (q / (2.0 * H)) * x * (l - x) - (q * EI / H ** 2) * (
                1.0 - np.cosh(beta * (x - l / 2.0)) / np.cosh(beta * l / 2.0))
        # integral of v over the span, closed form
        area = (q / (2.0 * H)) * l ** 3 / 6.0 - (q * EI / H ** 2) * (
            l - (2.0 / beta) * np.tanh(beta * l / 2.0))
        return v, area

    def compat(h):
        _v, area = v_of(h)
        return h * L_E / (E * A_c) - k * area

    h = brentq(compat, 0.0, p * l ** 2 / (8.0 * f) * 1.5)
    v, _area = v_of(h)
    return h, float(v(l / 2.0)), v


def build(path, **kw):
    prm = {**DEFAULTS, **kw}
    form = dead_load_form(**prm)
    n = prm["n"]; x = form["x"]; z_c = form["z_cable"]
    nodes = []
    for k in range(n + 1):
        nodes.append((x[k], 0.0, 0.0))            # deck node k -> id k
    for k in range(n + 1):
        nodes.append((x[k], 0.0, z_c[k]))         # cable node k -> id n+1+k
    lines = []                                    # (group tag, a, b)
    for k in range(n):
        lines.append((4, k, k + 1))               # deck
    for k in range(n):
        lines.append((5, n + 1 + k, n + 2 + k))   # cable
    for k in range(1, n):
        lines.append((6, n + 1 + k, k))           # hangers, cable -> deck
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "10",
           '0 1 "deck_left"', '0 2 "deck_right"', '0 3 "deck_mid"', '0 7 "deck_hung"',
           '0 8 "cable_left"', '0 9 "cable_right"', '0 10 "deck_probe"',
           '1 4 "deck"', '1 5 "main_cable"', '1 6 "hangers"', "$EndPhysicalNames",
           "$Nodes", str(len(nodes))]
    out += [f"{k + 1} {px:.12g} {py:.12g} {pz:.12g}" for k, (px, py, pz) in enumerate(nodes)]
    pts = [(1, 0), (2, n), (8, n + 1), (9, 2 * n + 1), (3, n // 2), (10, n // 4)]
    pts += [(7, k) for k in range(1, n)]          # the hung deck nodes
    out += ["$EndNodes", "$Elements", str(len(pts) + len(lines))]
    e = 1
    for tag, nd in pts:
        out.append(f"{e} 15 2 {tag} {tag} {nd + 1}"); e += 1
    for tag, a, b in lines:
        out.append(f"{e} 1 2 {tag} {tag} {a + 1} {b + 1}"); e += 1
    out += ["$EndElements", ""]
    pathlib.Path(path).write_text("\n".join(out), encoding="utf-8")
    h, v_mid, _v = deflection_theory(**prm)
    return {**form, "h_live": h, "v_mid_live": v_mid, "params": prm,
            "deck_interior_load": prm["w"] * form["s"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="benchmarks/assets/suspension_bridge_30.msh")
    a = ap.parse_args()
    info = build(a.out)
    print("wrote", a.out)
    print(f"  H_w          {info['H']:.6g} N")
    print(f"  P (hanger)   {info['P']:.6g} N")
    print(f"  T cable      {info['T_cable'].min():.6g} .. {info['T_cable'].max():.6g} N")
    print(f"  live: h      {info['h_live']:.6g} N,  v_mid {info['v_mid_live']:.6g} m")
    print("  Lu cable:", " ".join(f"{v:.9g}" for v in info["Lu_cable"]))
    print("  Lu hangers:", " ".join(f"{v:.9g}" for v in info["Lu_hangers"]))


def vertical_modes(n_modes=4, n_grid=600, rho=7850.0, **kw):
    """The linearised deflection theory's vertical modes (Hz, ascending):

        E I v'''' - H_w v'' + (8 f / l^2) h = omega^2 m(x) v
        h = (E A_c / L_E) (8 f / l^2) integral v dx

    on a finite-difference grid, hinged ends, with the mass the model
    carries per metre of span: the deck, the cable (per metre of arc,
    ds/dx), and the hangers (rho A_h z_c(x) / s, half of each on the deck
    and half on the cable, which move together in a vertical mode). The
    antisymmetric modes have h = 0 and reduce to the string-plus-beam
    closed form sqrt((E I k^4 + H_w k^2) / m) for uniform m.
    """
    prm = {**DEFAULTS, **kw}
    l, f, E, A_c, I_d = prm["l"], prm["f"], prm["E"], prm["A_c"], prm["I_d"]
    H = prm["w"] * l ** 2 / (8.0 * f)
    L_E = l * (1.0 + 8.0 * (f / l) ** 2)
    k8 = 8.0 * f / l ** 2
    n = n_grid
    x = np.linspace(0.0, l, n + 1)
    hx = l / n
    z_c = prm["c"] + f * (1.0 - 4.0 * x * (l - x) / l ** 2)
    dzdx = f * (-4.0 * (l - 2 * x) / l ** 2)
    s = l / prm["n"]
    m = rho * (prm["A_d"] + A_c * np.sqrt(1.0 + dzdx ** 2) + prm["A_h"] * z_c / s)
    # interior unknowns v_1 .. v_{n-1}; hinged: v_0 = v_n = 0, v'' = 0 at the ends
    N = n - 1
    K = np.zeros((N, N))
    EI = E * I_d
    for i in range(N):
        # fourth difference with the hinged ghost points (v_{-1} = -v_1)
        for j, c in ((i - 2, 1.0), (i - 1, -4.0), (i, 6.0), (i + 1, -4.0), (i + 2, 1.0)):
            if 0 <= j < N:
                K[i, j] += EI * c / hx ** 4
            elif j == -1:
                K[i, i + 0] += 0.0   # v_{-1} = -v_1 handled below
        for j, c in ((i - 1, -1.0), (i, 2.0), (i + 1, -1.0)):
            if 0 <= j < N:
                K[i, j] += H * c / hx ** 2
    # hinged end: v_{-1} = -v_1 -> the (i=0, j=-1) fourth-difference entry adds -1 to K[0,0]
    K[0, 0] -= EI / hx ** 4
    K[N - 1, N - 1] -= EI / hx ** 4
    # the cable's tension increment couples every unknown: h = (E A_c/L_E) k8 * sum(v) hx
    K += k8 * (E * A_c / L_E) * k8 * hx * np.ones((N, N))
    M = m[1:-1]
    d = 1.0 / np.sqrt(M)
    A = d[:, None] * K * d[None, :]
    w = np.linalg.eigvalsh(0.5 * (A + A.T))
    return np.sqrt(np.maximum(w[:n_modes], 0.0)) / (2.0 * np.pi)
