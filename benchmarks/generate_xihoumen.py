"""Xihoumen rail-cum-road bridge (main bridge): mesh, dead-load state, case data.

SOURCES
  [X] Xiao Hai-zhu, Gao Zong-yu, Liu Jun-feng, "西堠门公铁两用大桥主桥结构设计",
      Bridge Construction 50(S2), 2020 -- the DESIGNER'S paper. Everything
      geometric below is from it unless marked otherwise.
  [J] Jia et al., Structures 58 (2023) 105509 -- the tower wall thicknesses
      (Fig. 3(e)), the five moduli/densities (Table 1), and the ten natural
      periods this model is compared with (Table 4).

WHERE THE TWO DISAGREE, [X] WINS, and the disagreements are not small:

  spans          [J] 70+84+434+1488+434+84+70   [X] 70+112+406+1488+406+112+70
  main cable     [J] 151 x 127 x phi5.5         [X] 127 x 127 x phi5.96
  stays          [J] PES7-439                   [X] PES7-301 ... PES7-511
  "63 m, 27.5 m" [J] reads as the longitudinal spacing of stays and hangers;
                 [X] says 63.7 m and 27.4 m are their TRANSVERSE spacings on the
                 girder, and the longitudinal anchor spacing is 14 m for both.
                 The first version of this generator followed [J] here and put
                 one stay every 63 m -- a quarter of the real fan.

THE BRIDGE ([X] 2.1-2.3)
  girder   three separated streamlined steel boxes, level road/rail: rail box
           13 m on the centreline, two road boxes 16.9 m, gaps 8.1 m, fairings
           2.5 m, 68 m overall, 5 m deep; box crossbeams every 14 m
  stays    fan, 36 pairs per tower per plane (144 pairs), anchored on the
           girder at y = +-31.85 (63.7 m apart), 14 m apart along it, and in
           the tower at 2.5 m spacing over a 90 m anchor zone
  cable    two SPATIAL main cables, 6 m apart at the tower tops and 26.5 m at
           midspan, sag 1/6.5, spans 680 + 1488 + 677; hangers anchored at
           y = +-13.7 (27.4 m apart); pure suspended length 452 m, 9 pairs of
           crossed stays and hangers, hanger zone 0.45 L
  tower    A-shaped, C60, 294 m: upper column 114 m (single shaft, 10-16 m
           transverse), middle legs 126.5 m (7 m transverse), lower legs
           41.5 m; legs 33 m apart outside at the base; crossbeam at the
           lower/middle junction with corbels whose ends are 46 m apart
  supports towers separate from the girder: 3 vertical bearings (one under
           each box), transverse wind bearings at the rail box, longitudinal
           dampers (no static stiffness); end piers 1/8: 3 vertical bearings,
           the middle one also lateral; auxiliary piers 2/3/6/7: one vertical
           support under each road box

LUMPING. Anchors are 14 m apart; this model puts one node per TWO anchors
(28 m) and gives that stay or hanger the area of both. That halves the node
count of the girder and keeps the model inside what the finite-difference
modal tangent can afford (`modal.max_dofs`).

THE DEAD-LOAD STATE IS STATED, not searched: the girder is taken as rigidly
supported at every anchor (the classic zero-moment dead state), so each stay
or hanger carries its tributary 28 m of deck; the main cable is the exact
funicular polygon of its hanger forces and its own weight, in elevation AND in
plan (the spatial cable is pulled out by the inclined hangers), with the
midspan sag fixed at L/6.5 -- so its horizontal tension H follows. Every cable
element gets the unstressed length that holds its tension at the mesh length.

THE ONE NUMBER CALIBRATED: the deck's dead load per metre. [X] Table 2 gives
the dead-load axial force at a tower base, 2 172 059 kN; everything else in
that sum is geometry, so it fixes w. [X] Table 3's dead-load girder axial
forces are then an INDEPENDENT check, printed beside it.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

G = 9.80665

# --------------------------------------------------------------------------- #
#  [X] geometry
# --------------------------------------------------------------------------- #
SPANS = (70.0, 112.0, 406.0, 1488.0, 406.0, 112.0, 70.0)
BACK_SPANS = (680.0, 677.0)             # main cable, tower to anchorage
L_MAIN = 1488.0
SAG = L_MAIN / 6.5

Y_RAIL, Y_ROAD = 0.0, 23.05             # box centroids: 6.5 + 8.1 + 16.9/2
Y_STAY_G = 31.85                        # stay anchors on the girder (63.7/2)
Y_HANG_G = 13.7                         # hanger anchors on the girder (27.4/2)
Y_IP = 3.0                              # main cable at the tower top (6/2)
Y_MID_DESIGN = 13.25                    # ... and at midspan (26.5/2): a CHECK
Y_STAY_T = 3.5                          # stay anchors inside the upper column

Z_BASE = 1.0                            # top of cap
Z_TOP = Z_BASE + 294.0                  # tower top
Z_IP = Z_TOP + 4.0                      # cable IP, 4 m above the top [X]
Z_CB = 50.5                             # crossbeam centre (8 m deep)
Z_DECK = 57.5                           # girder centroid: 51.5 m clearance
Z_MERGE = Z_TOP - 114.0                 # upper column bottom
Z_WALL = Z_MERGE + 10.0                 # the 10 m partition wall
Y_LEG_BASE, Y_LEG_MERGE = 12.5, 4.5     # 33 m outside at the base, 8 m legs
Y_CORBEL = 23.0                         # corbel ends 46 m apart

N_STAY_REAL, STAY_DS, STAY_D0 = 36, 14.0, 28.0   # per side per plane
Z_ANCH_LO = Z_TOP - 9.5 - 90.0 + 1.5    # 90 m anchor zone, 2.5 m pitch
ANCH_PITCH = 2.5
HANG_HALF = 0.45 * L_MAIN / 2.0         # hanger zone: |d - L/2| <= 334.8
PURE_HALF = 452.0 / 2.0                 # pure suspended zone

Z_ANCHORAGE = 45.0                      # ASSUMED: splay saddle elevation
Y_ANCHORAGE = 12.0                      # ASSUMED: cable offset at the anchorage

# cable sections [X]
WIRE = lambda d: np.pi * (d / 2.0) ** 2
A_MAIN = 127 * 127 * WIRE(0.00596)                  # 0.44997 m^2
A_HANGER_PURE = 139 * WIRE(0.007)                   # per rope
A_HANGER_CROSS = 91 * WIRE(0.007)
ROPES_PER_HANGER = 2                    # ASSUMED: a pair of ropes per point
STAY_N_MIN, STAY_N_MAX = 301, 511       # PES7-301 (shortest) .. PES7-511

# --------------------------------------------------------------------------- #
#  [J] materials, tower walls
# --------------------------------------------------------------------------- #
MAT = {
    "rail_box": dict(E=2.10e11, nu=0.30),          # density set by calibration
    "road_box": dict(E=2.10e11, nu=0.30),
    "xbeam":    dict(E=2.10e11, nu=0.30, rho=1.0), # its steel is in the boxes'
    "rigid":    dict(E=2.10e12, nu=0.30, rho=1.0),
    "tower":    dict(E=3.60e10, nu=0.20, rho=2650.0),
    "stay":     dict(E=1.81e11, nu=0.30, rho=7810.0),
    "hanger":   dict(E=2.00e11, nu=0.30, rho=7850.0),
    "main":     dict(E=2.00e11, nu=0.30, rho=8400.0),
}
#  tower sections: (b longitudinal, h transverse, wall t) -- dims [X], walls [J]
T_LOWER = (17.5, 7.5, 1.2)              # [X] 16.98-18 x 7-8   [J] C-C t1.2
T_MIDDLE = (15.7, 7.0, 1.8)             # [X] 14.38-16.98 x 7  [J] B-B t1.8
T_WALL = (14.4, 16.0, 4.0)              # the partition-wall zone
T_UPPER = (13.2, 13.0, 1.4)             # [X] 12-14.38 x 10-16 [J] A-A t1.4
T_CROSSBEAM = (10.0, 8.0, 1.0)          # [X] 10 wide, 8 deep; wall ASSUMED

# --------------------------------------------------------------------------- #
#  ASSUMED girder sections -- [X] gives the outline, not the plate thicknesses
#  (equivalent thicknesses incl. ribs: deck 30 mm, bottom 22 mm, webs 16 mm)
# --------------------------------------------------------------------------- #
BOX = {   # A, I_vertical, I_lateral, J  (per box)
    "road": (1.06, 4.7, 18.3, 9.8),
    "rail": (0.88, 4.5, 9.6, 8.3),
}
DECK_SHARE = {"rail": 0.40, "road": 0.30}   # of the deck mass, per box
#  box crossbeam, 3 m x 5 m box, 20 mm plates [X dims]; two per 28 m node
XB = dict(A=0.32 * 2, Iv=1.17 * 2, Ih=0.48 * 2, J=1.125 * 2)
GAP = 8.1
TOWER_EXTRA = {"saddle_per_ip": 200e3, "anchor_steel_per_level": 2000e3 / 36}

# --------------------------------------------------------------------------- #
#  [X] Table 2 / Table 3: the calibration target and its independent check
# --------------------------------------------------------------------------- #
TOWER_BASE_N_DEAD = 2_172_059e3         # N
GIRDER_N_DEAD = {"road": 147_506e3, "rail": 123_499e3}


def bredt(b, h, t):
    """Hollow rectangle b x h, wall t: A, I (about the b-parallel axis, so
    against deflection along h), I (against deflection along b), J."""
    A = b * h - (b - 2 * t) * (h - 2 * t)
    I_b = (b * h ** 3 - (b - 2 * t) * (h - 2 * t) ** 3) / 12.0
    I_h = (h * b ** 3 - (h - 2 * t) * (b - 2 * t) ** 3) / 12.0
    Am = (b - t) * (h - t)
    return A, I_b, I_h, 4 * Am ** 2 * t / (2 * ((b - t) + (h - t)))


class Mesh:
    def __init__(self):
        self.nodes, self.index = [], {}
        self.lines, self.points = [], []
        self.Lu: dict[str, list] = {}
        self.groups: dict[str, int] = {}
        self.pgroups: dict[str, int] = {}

    def node(self, x, y, z):
        k = (round(x, 5), round(y, 5), round(z, 5))
        if k not in self.index:
            self.index[k] = len(self.nodes)
            self.nodes.append(k)
        return self.index[k]

    def line(self, group, a, b, Lu=None):
        tag = self.groups.setdefault(group, len(self.groups) + 1)
        self.lines.append((tag, a, b))
        if Lu is not None:
            self.Lu.setdefault(group, []).append(float(Lu))

    def point(self, group, n):
        tag = self.pgroups.setdefault(group, 100 + len(self.pgroups))
        self.points.append((tag, n))

    def length(self, a, b):
        return float(np.linalg.norm(np.subtract(self.nodes[b], self.nodes[a])))

    def write(self, path):
        names = [f'1 {t} "{n}"' for n, t in self.groups.items()] + \
                [f'0 {t} "{n}"' for n, t in self.pgroups.items()]
        out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames",
               str(len(names))] + names + ["$EndPhysicalNames", "$Nodes",
                                           str(len(self.nodes))]
        out += [f"{i+1} {x:.9g} {y:.9g} {z:.9g}" for i, (x, y, z) in enumerate(self.nodes)]
        out += ["$EndNodes", "$Elements", str(len(self.points) + len(self.lines))]
        e = 1
        for tag, n in self.points:
            out.append(f"{e} 15 2 {tag} {tag} {n+1}"); e += 1
        for tag, a, b in self.lines:
            out.append(f"{e} 1 2 {tag} {tag} {a+1} {b+1}"); e += 1
        out += ["$EndElements", ""]
        pathlib.Path(path).write_text("\n".join(out), encoding="utf-8")


# --------------------------------------------------------------------------- #
#  stations along the girder
# --------------------------------------------------------------------------- #
def stations():
    xs = np.cumsum((0.0,) + SPANS)
    XT = (xs[3], xs[4])
    d_stay = STAY_D0 + STAY_DS / 2 + 2 * STAY_DS * np.arange(N_STAY_REAL // 2)  # 35..511
    st = {}

    def add(x, **kw):
        k = round(float(x), 4)
        rec = st.setdefault(k, dict(x=k, stay=None, hanger=False, support=None))
        for key, v in kw.items():
            rec[key] = v

    for t, xt in enumerate(XT):
        add(xt, support="tower")
        for j, d in enumerate(d_stay):
            add(xt - (1 if t == 0 else -1) * d, stay=(t, "side", j))
    # main span: from each tower on the same 28 m grid, meeting at midspan
    for k in range(26):
        d = STAY_D0 + STAY_DS / 2 + 2 * STAY_DS * k                   # 35..735
        for t, x in ((0, XT[0] + d), (1, XT[1] - d)):
            add(x)
            if k < len(d_stay):
                add(x, stay=(t, "main", k))
            if abs((x - XT[0]) - L_MAIN / 2) <= HANG_HALF:
                add(x, hanger=True)
    for x in (xs[0], xs[-1]):
        add(x, support="end")
    for x in (xs[1], xs[2], xs[5], xs[6]):
        add(x, support="aux")
    for x in (xs[0] + 35.0, xs[-1] - 35.0):
        add(x)
    out = [st[k] for k in sorted(st)]
    xv = np.array([s["x"] for s in out])
    for i, s in enumerate(out):
        lo = xv[i] - xv[i - 1] if i else 0.0
        hi = xv[i + 1] - xv[i] if i + 1 < len(xv) else 0.0
        s["trib"] = 0.5 * (lo + hi)
        s["d_mid"] = abs(s["x"] - (XT[0] + L_MAIN / 2))
    return out, xs, XT, d_stay


def stay_share(s):
    """In the crossing zone the stay hands its load over to the hanger,
    linearly, from the first hanger (d = 409) to past the last stay."""
    if s["stay"] is None:
        return 0.0
    if not s["hanger"]:
        return 1.0
    d_tower = L_MAIN / 2 - s["d_mid"]
    lo, hi = L_MAIN / 2 - HANG_HALF, STAY_D0 + STAY_DS * N_STAY_REAL + 7.0
    return float(np.clip((hi - d_tower) / (hi - lo), 0.0, 1.0))


def funicular(x, P, x0, x1, H=None, sag=None, x_mid=None):
    """Simple-beam moment of nodal loads P at x on [x0, x1]; if `sag` is given
    H = M(x_mid)/sag. Returns (M at x, H)."""
    Lx = x1 - x0
    R0 = np.sum(P * (x1 - x)) / Lx
    M = np.array([R0 * (xi - x0) - np.sum(P[x < xi] * (xi - x[x < xi])) for xi in x])
    if sag is not None:
        Mm = np.interp(x_mid, np.r_[x0, x, x1], np.r_[0.0, M, 0.0])
        H = Mm / sag
    return M, H


# --------------------------------------------------------------------------- #
#  the build
# --------------------------------------------------------------------------- #
def build(path, w_deck, verbose=False):
    """w_deck: the deck's dead load, N/m (all three boxes and everything on
    them). Returns the info dict; writes the mesh and a JSON sidecar."""
    st, xs, XT, d_stay = stations()
    m = Mesh()
    E = {k: v["E"] for k, v in MAT.items()}
    info = dict(w_deck=w_deck)

    # ---------------- girder ----------------------------------------------- #
    gn = {}                                   # (station index, key) -> node
    for i, s in enumerate(st):
        x = s["x"]
        for key, y in (("R-", -Y_ROAD), ("C", Y_RAIL), ("R+", Y_ROAD)):
            gn[(i, key)] = m.node(x, y, Z_DECK)
        if s["hanger"]:
            for key, y in (("H-", -Y_HANG_G), ("H+", Y_HANG_G)):
                gn[(i, key)] = m.node(x, y, Z_DECK)
        if s["stay"] is not None:
            for key, y in (("S-", -Y_STAY_G), ("S+", Y_STAY_G)):
                gn[(i, key)] = m.node(x, y, Z_DECK)
    for i in range(len(st) - 1):
        m.line("road_box", gn[(i, "R-")], gn[(i + 1, "R-")])
        m.line("rail_box", gn[(i, "C")], gn[(i + 1, "C")])
        m.line("road_box", gn[(i, "R+")], gn[(i + 1, "R+")])
    for i, s in enumerate(st):
        for sg in ("-", "+"):
            if s["hanger"]:
                m.line("xbeam_in", gn[(i, "C")], gn[(i, "H" + sg)])
                m.line("xbeam_out", gn[(i, "H" + sg)], gn[(i, "R" + sg)])
            else:
                m.line("xbeam", gn[(i, "C")], gn[(i, "R" + sg)])
            if s["stay"] is not None:
                m.line("arm", gn[(i, "R" + sg)], gn[(i, "S" + sg)])

    # ---------------- loads carried by stays / hangers --------------------- #
    for s in st:
        V = 0.5 * w_deck * s["trib"]            # per plane
        sh = stay_share(s)
        s["V_stay"] = V * sh if s["stay"] is not None else 0.0
        s["V_hang"] = V * (1 - sh) if s["hanger"] else 0.0
        s["V_supp"] = V if (s["stay"] is None and not s["hanger"]) else 0.0

    # ---------------- towers ----------------------------------------------- #
    z_lower = list(np.linspace(Z_BASE, Z_CB, 4))
    z_middle = list(np.linspace(Z_CB, Z_MERGE, 8))
    y_leg = lambda z: Y_LEG_BASE - (Y_LEG_BASE - Y_LEG_MERGE) * (z - Z_BASE) / (Z_MERGE - Z_BASE)
    z_anch = Z_ANCH_LO + ANCH_PITCH * (2 * np.arange(N_STAY_REAL // 2) + 0.5)
    tw = {}
    tower_mass = 0.0
    for t, xt in enumerate(XT):
        for sg in (-1, 1):
            prev = m.node(xt, sg * y_leg(Z_BASE), Z_BASE)
            m.point("tower_base", prev)
            for z in sorted(z_lower[1:] + z_middle[1:] + [Z_DECK]):
                n = m.node(xt, sg * y_leg(z), z)
                m.line("leg_lower" if z <= Z_CB + 1e-9 else "leg_middle", prev, n)
                if abs(z - Z_DECK) < 1e-9:
                    tw[(t, "leg_deck", sg)] = n
                prev = n
            m.line("rigid", prev, m.node(xt, 0.0, Z_MERGE))
        col = [Z_MERGE, Z_WALL] + list(z_anch) + [Z_TOP]
        for za, zb in zip(col[:-1], col[1:]):
            m.line("wall" if za < Z_WALL - 1e-9 else "upper",
                   m.node(xt, 0.0, za), m.node(xt, 0.0, zb))
        for sg in (-1, 1):
            ip = m.node(xt, sg * Y_IP, Z_IP)
            m.line("rigid", m.node(xt, 0.0, Z_TOP), ip)
            m.point("ip", ip)
            tw[(t, "ip", sg)] = ip
            for j, z in enumerate(z_anch):
                a = m.node(xt, sg * Y_STAY_T, z)
                m.line("rigid", m.node(xt, 0.0, z), a)
                m.point("anchor_t", a)
                tw[(t, "anch", sg, j)] = a
        # crossbeam and corbels at the lower/middle junction
        yl = y_leg(Z_CB)
        cl, cc, cr = (m.node(xt, -yl, Z_CB), m.node(xt, 0.0, Z_CB), m.node(xt, yl, Z_CB))
        m.line("crossbeam", cl, cc); m.line("crossbeam", cc, cr)
        for sg, c in ((-1, cl), (1, cr)):
            tip = m.node(xt, sg * Y_ROAD, Z_CB)
            m.line("corbel", c, tip)
            tw[(t, "corbel", sg)] = tip
        tw[(t, "cb")] = cc
        # BEARINGS AS PIN-ENDED BARS. A bar carries force along its axis and
        # nothing else, so a vertical bar under a box IS a vertical bearing
        # that lets the box slide and rotate; two horizontal bars from the
        # rail box to the legs ARE the transverse wind bearings [X 2.2]. No
        # longitudinal link: [X] has dampers there, which carry nothing
        # statically.
        i = next(k for k, s in enumerate(st) if abs(s["x"] - xt) < 1e-6)
        m.line("bearing", cc, gn[(i, "C")])
        m.line("bearing", tw[(t, "corbel", -1)], gn[(i, "R-")])
        m.line("bearing", tw[(t, "corbel", 1)], gn[(i, "R+")])
        for sg in (-1, 1):
            m.line("bearing", gn[(i, "C")], tw[(t, "leg_deck", sg)])

    # ---------------- stays ------------------------------------------------ #
    stay_T, stay_sig, girder_N = [], [], {0: 0.0, 1: 0.0}
    tower_V = {0: 0.0, 1: 0.0}
    # EACH PAIR BALANCES AT THE TOWER. A main-span stay in the crossing zone
    # hands part of its load to a hanger; its side-span partner, left on its
    # tributary load, would then out-pull it and ~80 MN of net longitudinal
    # force reaches each tower (the relaxed model leaned 2.3 m). A design
    # balances every anchor beam [X 2.3.2: one pair per steel anchor beam]
    # and the auxiliary piers take the side span's difference. So the main-
    # span stay is sized on its load and the side-span stay on the pair.
    Hx_main = {}
    order = sorted((i for i, s in enumerate(st) if s["stay"] is not None),
                   key=lambda i: 0 if st[i]["stay"][1] == "main" else 1)
    for i in order:
        s = st[i]
        t, side, j = s["stay"]
        d_real = STAY_D0 + STAY_DS * (2 * j + 0.5)
        n_wires = 2 * (STAY_N_MIN + (STAY_N_MAX - STAY_N_MIN)
                       * (d_real - STAY_D0) / (STAY_DS * (N_STAY_REAL - 1)))
        A = n_wires * WIRE(0.007)
        for sg in (-1, 1):
            a, b = tw[(t, "anch", sg, j)], gn[(i, "S" + ("+" if sg > 0 else "-"))]
            L = m.length(a, b)
            dz = m.nodes[a][2] - m.nodes[b][2]
            dx = abs(m.nodes[a][0] - m.nodes[b][0])
            if side == "main":
                T = s["V_stay"] * L / dz
                Hx_main[(t, j, sg)] = T * dx / L
            else:
                T = Hx_main[(t, j, sg)] * L / dx
                s["V_stay"] = T * dz / L          # what it actually lifts
            m.line(f"stay_{j:02d}", a, b, Lu=L / (1 + T / (E["stay"] * A)))
            info.setdefault("stay_area", {})[f"stay_{j:02d}"] = A
            stay_T.append(T); stay_sig.append(T / A)
            tower_V[t] += T * dz / L
            if side == "main":
                girder_N[t] += T * abs(m.nodes[a][0] - m.nodes[b][0]) / L
    info["stay_sigma"] = (min(stay_sig), max(stay_sig))
    info["girder_N_at_tower"] = girder_N[0]

    # ---------------- main cable: funicular in elevation and plan ---------- #
    msi = [i for i, s in enumerate(st) if XT[0] < s["x"] < XT[1]]
    xm = np.array([st[i]["x"] for i in msi])
    Vh = np.array([st[i]["V_hang"] for i in msi])
    hanger_A = np.array([ROPES_PER_HANGER * 2 *
                         (A_HANGER_PURE if st[i]["d_mid"] <= PURE_HALF + 1e-6 else A_HANGER_CROSS)
                         for i in msi])
    w_c = MAT["main"]["rho"] * A_MAIN * G
    xn = np.r_[XT[0], xm, XT[1]]
    z = Z_IP - SAG * 4 * (xn - XT[0]) * (XT[1] - xn) / L_MAIN ** 2
    y = np.full_like(xn, Y_IP)
    for _ in range(40):
        seg = np.sqrt(np.diff(xn) ** 2 + np.diff(z) ** 2 + np.diff(y) ** 2)
        Pc = 0.5 * w_c * (seg[:-1] + seg[1:])            # cable weight at nodes
        Pz = Vh + Pc
        Mz, H = funicular(xm, Pz, XT[0], XT[1], sag=SAG, x_mid=XT[0] + L_MAIN / 2)
        z_new = Z_IP - Mz / H
        # PLAN: the inclined hangers are stiff lateral springs toward
        # y = Y_HANG_G, so y = Y_IP + G P_y(y) / H is solved as the linear
        # system it is -- a fixed-point iteration on it diverges (factor ~14)
        hz = np.where(Vh > 0, z_new - Z_DECK, 1.0)
        k = np.where(Vh > 0, Vh / hz, 0.0)            # lateral spring, N/m
        X0, X1 = XT[0], XT[1]
        Gm = np.where(xm[None, :] <= xm[:, None],
                      (xm[None, :] - X0) * (X1 - xm[:, None]),
                      (xm[:, None] - X0) * (X1 - xm[None, :])) / (X1 - X0)
        A_sys = np.eye(len(xm)) + Gm * k[None, :] / H
        y_new = np.linalg.solve(A_sys, Y_IP + Gm @ (k * Y_HANG_G) / H)
        z = np.r_[Z_IP, z_new, Z_IP]; y = np.r_[Y_IP, y_new, Y_IP]
    info["H"] = H
    info["cable_y_mid"] = float(np.interp(XT[0] + L_MAIN / 2, xn, y))
    info["cable_z_mid"] = float(np.interp(XT[0] + L_MAIN / 2, xn, z))
    info["cable_sigma_max"] = 0.0
    cable_V_ip = 0.0
    for sg in (-1, 1):
        nodes = [tw[(0, "ip", sg)]] + [m.node(x, sg * yy, zz) for x, yy, zz in
                                       zip(xm, y[1:-1], z[1:-1])] + [tw[(1, "ip", sg)]]
        for a, b in zip(nodes[:-1], nodes[1:]):
            L = m.length(a, b); dx = abs(m.nodes[b][0] - m.nodes[a][0])
            T = H * L / dx
            info["cable_sigma_max"] = max(info["cable_sigma_max"], T / A_MAIN)
            m.line("main_cable", a, b, Lu=L / (1 + T / (E["main"] * A_MAIN)))
        # hangers
        for k, i in enumerate(msi):
            if not st[i]["hanger"]:
                continue
            c = nodes[k + 1]
            g = gn[(i, "H" + ("+" if sg > 0 else "-"))]
            L = m.length(c, g)
            T = Vh[k] * L / (m.nodes[c][2] - m.nodes[g][2])
            grp = "hang_pure" if st[i]["d_mid"] <= PURE_HALF + 1e-6 else "hang_cross"
            m.line(grp, c, g, Lu=L / (1 + T / (E["hanger"] * hanger_A[k])))
            info.setdefault("hanger_area", {})[grp] = float(hanger_A[k])
            info.setdefault("hanger_sigma", []).append(T / hanger_A[k])
        # vertical reaction of the main-span cable at each IP = half its load
        cable_V_ip += 0.5 * (np.sum(Pz))
    info["cable_V_ip_main"] = cable_V_ip          # per tower, both cables

    # ---------------- back spans ------------------------------------------- #
    back_V = 0.0
    for t, (xt, Lb) in enumerate(zip(XT, BACK_SPANS)):
        xa = xt - Lb if t == 0 else xt + Lb
        for sg in (-1, 1):
            p0 = np.array([xt, sg * Y_IP, Z_IP]); p1 = np.array([xa, sg * Y_ANCHORAGE, Z_ANCHORAGE])
            s_plan = np.hypot(*(p1 - p0)[:2])
            H_plan = H * s_plan / abs(p1[0] - p0[0])      # same x-component as the main span
            ns = 12
            sv = np.linspace(0.0, s_plan, ns + 1)
            chord = p0[2] + (p1[2] - p0[2]) * sv / s_plan
            zb = chord.copy()
            for _ in range(20):
                seg = np.hypot(np.diff(sv), np.diff(zb))
                P = 0.5 * w_c * (seg[:-1] + seg[1:])
                M, _ = funicular(sv[1:-1], P, 0.0, s_plan, H=H_plan)
                zb = np.r_[p0[2], chord[1:-1] - M / H_plan, p1[2]]
            pts = [p0 + (p1 - p0) * (v / s_plan) for v in sv]
            nodes = [tw[(t, "ip", sg)]] + [m.node(p[0], p[1], zz) for p, zz in
                                           zip(pts[1:], zb[1:])]
            m.point("anchorage", nodes[-1])
            for a, b in zip(nodes[:-1], nodes[1:]):
                L = m.length(a, b)
                ds = np.hypot(*(np.subtract(m.nodes[b], m.nodes[a])[:2]))
                T = H_plan * L / ds
                m.line("main_cable", a, b, Lu=L / (1 + T / (E["main"] * A_MAIN)))
            # vertical force the back span puts on the IP
            back_V += H_plan * (zb[0] - zb[1]) / (sv[1] - sv[0]) + 0.5 * w_c * np.hypot(sv[1], zb[0] - zb[1])
    info["cable_V_ip_back"] = back_V / 2.0        # per tower

    # ---------------- supports and links ----------------------------------- #
    for i, s in enumerate(st):
        if s["support"] == "end":
            for key in ("R-", "C", "R+"):
                m.point("pier_z", gn[(i, key)])
            m.point("pier_y", gn[(i, "C")])
        elif s["support"] == "aux":
            for key in ("R-", "R+"):
                m.point("pier_z", gn[(i, key)])
    for i, s in enumerate(st):
        for key in ("R-", "R+"):
            m.point("road_nodes", gn[(i, key)])
        m.point("rail_nodes", gn[(i, "C")])

    # ---------------- tower dead-load base force --------------------------- #
    def tower_self_weight():
        W = 0.0
        for grp, sec in (("leg_lower", T_LOWER), ("leg_middle", T_MIDDLE),
                         ("wall", T_WALL), ("upper", T_UPPER), ("crossbeam", T_CROSSBEAM)):
            A = bredt(*sec)[0]
            tag = m.groups[grp]
            L = sum(m.length(a, b) for tg, a, b in m.lines if tg == tag)
            W += A * L * MAT["tower"]["rho"]
        tag = m.groups["corbel"]
        W += 8.0 * 7.3 * sum(m.length(a, b) for tg, a, b in m.lines if tg == tag) * MAT["tower"]["rho"]
        W += 4 * TOWER_EXTRA["saddle_per_ip"] + 2 * 36 * TOWER_EXTRA["anchor_steel_per_level"]
        return W * G / 2.0                        # per tower
    bearing_V = sum(0.5 * w_deck * s["trib"] * 2 for s in st if s["support"] == "tower") / 2.0
    info["tower_self_weight"] = tower_self_weight()
    info["tower_stays_V"] = tower_V[0]
    info["tower_bearing_V"] = bearing_V
    info["tower_base_N"] = (info["tower_self_weight"] + tower_V[0] + bearing_V
                            + info["cable_V_ip_main"] + info["cable_V_ip_back"])

    # ---------------- write ------------------------------------------------ #
    m.write(path)
    info.update(nodes=len(m.nodes), elements=len(m.lines))
    side = dict(
        Lu={k: v for k, v in m.Lu.items()},
        groups=m.groups, pgroups=m.pgroups,
        rail=[gn[(i, "C")] for i in range(len(st))],
        road_m=[gn[(i, "R-")] for i in range(len(st))],
        road_p=[gn[(i, "R+")] for i in range(len(st))],
        x=[s["x"] for s in st], XT=list(XT),
        cable=sorted({a for tg, a, b in m.lines if tg == m.groups["main_cable"]}
                     | {b for tg, a, b in m.lines if tg == m.groups["main_cable"]}),
        info={k: v for k, v in info.items() if np.isscalar(v)},
        stay_area=info.get("stay_area", {}), hanger_area=info.get("hanger_area", {}),
    )
    pathlib.Path(path).with_suffix(".json").write_text(json.dumps(side), encoding="utf-8")
    return info, side


def calibrate(path, target=TOWER_BASE_N_DEAD):
    """The deck load that makes the tower base carry [X]'s dead-load axial
    force. The base force is affine in w (every cable force is linear in the
    loads it carries), so two builds fix it."""
    f = lambda w: build(path, w)[0]["tower_base_N"]
    w0, w1 = 400e3, 800e3
    n0, n1 = f(w0), f(w1)
    return w0 + (target - n0) * (w1 - w0) / (n1 - n0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="benchmarks/assets/xihoumen.msh")
    ap.add_argument("--w", type=float, default=None, help="deck dead load, N/m")
    a = ap.parse_args()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    w = a.w if a.w else calibrate(a.out)
    info, _ = build(a.out, w)
    for k, v in info.items():
        print(f"  {k:22s} {v}")


# --------------------------------------------------------------------------- #
#  the case
# --------------------------------------------------------------------------- #
def _crossbeam(L_total, L_flexible, cap=200.0):
    """A crossbeam element spanning box centroid to box centroid, of which
    only the part in the 8.1 m gap bends: the rest is inside a box, braced by
    its diaphragms. Its sway stiffness 12 E I / l^3 over the flexible length l
    is reproduced by a uniform element of length L with I (L/l)^3 and its
    axial stiffness with A L/l."""
    f = min((L_total / L_flexible) ** 3, cap)
    a = L_total / L_flexible
    return dict(area=XB["A"] * a, iz=XB["Iv"] * f, iy=XB["Ih"] * f, j=XB["J"] * a)


def case_yaml(msh, side, w_deck, backend="numpy"):
    rho_rail = DECK_SHARE["rail"] * w_deck / G / BOX["rail"][0]
    rho_road = DECK_SHARE["road"] * w_deck / G / BOX["road"][0]
    mats = dict(MAT)
    mats["rail_box"] = dict(MAT["rail_box"], rho=rho_rail)
    mats["road_box"] = dict(MAT["road_box"], rho=rho_road)
    L = []
    L.append("name: xihoumen_rail_road")
    L.append("units: {length: m, force: N, stress: Pa}")
    L.append(f"mesh: {{type: gmsh, path: {msh}}}")
    L.append("materials:")
    for k, v in mats.items():
        L.append(f"  {k}: {{type: linear_elastic, E: {v['E']:.6g}, nu: {v['nu']}, rho: {v['rho']:.6g}}}")

    def beam(name, mat, A, iz, iy, j):
        L.append(f"  {name}: {{type: beam, material: {mat}, shape: general, area: {A:.6g}, "
                 f"iy: {iy:.6g}, iz: {iz:.6g}, j: {j:.6g}}}")

    def tower(name, sec):
        A, I_b, I_h, J = bredt(*sec)
        # vertical member, orientation [1, 0, 0]: local y is the bridge axis,
        # so iz resists LONGITUDINAL deflection (I_h) and iy transverse (I_b)
        beam(name, "tower", A, I_h, I_b, J)

    L.append("sections:")
    Ar, Ivr, Ilr, Jr = BOX["rail"]; Ao, Ivo, Ilo, Jo = BOX["road"]
    # boxes along x, orientation [0, 0, 1]: local y is vertical, iz vertical
    beam("rail_box", "rail_box", Ar, Ivr, Ilr, Jr)
    beam("road_box", "road_box", Ao, Ivo, Ilo, Jo)
    for name, (Lt, Lf) in (("xbeam", (Y_ROAD, GAP)),
                           ("xbeam_in", (Y_HANG_G, Y_HANG_G - 6.5)),
                           ("xbeam_out", (Y_ROAD - Y_HANG_G, 14.6 - Y_HANG_G))):
        s = _crossbeam(Lt, Lf)
        beam(name, "xbeam", s["area"], s["iz"], s["iy"], s["j"])
    beam("rigid", "rigid", 2.0, 20.0, 20.0, 20.0)
    tower("leg_lower", T_LOWER); tower("leg_middle", T_MIDDLE)
    tower("wall", T_WALL); tower("upper", T_UPPER)
    A, I_b, I_h, J = bredt(*T_CROSSBEAM)          # along y, orientation [0, 0, 1]
    beam("crossbeam", "tower", A, I_b, I_h, J)
    beam("corbel", "tower", 8.0 * 7.3, 8.0 * 7.3 ** 3 / 12, 7.3 * 8.0 ** 3 / 12, 400.0)
    for k, A in sorted(side["stay_area"].items()):
        L.append(f"  {k}: {{type: bar, material: stay, shape: general, area: {A:.6g}}}")
    for k, A in sorted(side["hanger_area"].items()):
        L.append(f"  {k}: {{type: bar, material: hanger, shape: general, area: {A:.6g}}}")
    L.append(f"  main_cable: {{type: bar, material: main, shape: general, area: {A_MAIN:.6g}}}")
    L.append("  bearing: {type: bar, material: rigid, shape: general, area: 0.05}")

    L.append("assign:")
    along_x = [("rail_box", "rail_box"), ("road_box", "road_box")]
    along_y = [("xbeam", "xbeam"), ("xbeam_in", "xbeam_in"), ("xbeam_out", "xbeam_out"),
               ("crossbeam", "crossbeam"), ("corbel", "corbel")]
    vertical = [("leg_lower", "leg_lower"), ("leg_middle", "leg_middle"),
                ("wall", "wall"), ("upper", "upper")]
    for r, s in along_x + along_y:
        L.append(f"  - {{region: {r}, section: {s}, orientation: [0, 0, 1]}}")
    for r, s in vertical + [("rigid", "rigid"), ("arm", "rigid")]:
        L.append(f"  - {{region: {r}, section: {s}, orientation: [1, 0, 0]}}")
    for g, lu in side["Lu"].items():
        sec = g
        vals = ", ".join(f"{v:.10g}" for v in lu)
        L.append(f"  - region: {g}")
        L.append(f"    section: {sec}")
        L.append(f"    unstressed_lengths: [{vals}]")
    L.append("  - {region: bearing, section: bearing}")

    L.append("nodal_masses:")
    L.append(f"  - {{region: ip, mass: {TOWER_EXTRA['saddle_per_ip']:.6g}, per_node: true}}")
    L.append(f"  - {{region: anchor_t, mass: {TOWER_EXTRA['anchor_steel_per_level']:.6g}, per_node: true}}")
    L.append("constraints:")
    L.append("  - {region: tower_base, dofs: [1, 1, 1, 1, 1, 1]}")
    L.append("  - {region: anchorage,  dofs: [1, 1, 1, 1, 1, 1]}")
    L.append("  - {region: pier_z,     dofs: [0, 0, 1, 0, 0, 0]}")
    L.append("  - {region: pier_y,     dofs: [0, 1, 0, 0, 0, 0]}")
    L.append("loads: []")
    L.append("steps: [{name: dead, load_steps: 1}]")
    L.append(f"solver: {{backend: {backend}, mode: relaxation, tolerance: 1.0e-4, n_steps: 1000}}")
    L.append("output:")
    L.append("  path: results/xihoumen.h5")
    L.append("  monitors:")
    L.append("    w_mid: {kind: displacement, region: rail_nodes, dof: z}")
    return "\n".join(L) + "\n"
