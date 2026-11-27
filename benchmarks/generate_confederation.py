"""Confederation Bridge: the P31-P32 portal frame, its sections, and an
independent plane-frame solution to check the model against.

THE SECTION IS RECONSTRUCTED, NOT PUBLISHED AS A FORMULA. The outline comes
from Dilger & Langohr (1997) CJCE 24: 898, Fig. 9 -- the fully dimensioned
section at the FACE of the pier -- and the two free thicknesses (bottom slab,
web) are calibrated on the gross-section table in Butcher (2009) Appendix E,
which states depth, area, centroid and second moment at two stations. Fitting
two numbers and predicting three keeps this a check and not a curve fit:

    station        depth      A            centroid        I
    pier face      13.990 m   22.247 m2    6752 mm         577.79 m4
    Sections 1/2   13.000 m   19.218 m2    6517 mm         454.96 m4
    midspan         4.500 m   --           2908 mm         --

With t_bot 1100 / t_web 460 at the pier face and 900 / 400 seven metres away
-- and 400 mm is the web thickness Dilger states outright -- the reconstruction
lands within 4% on all three, and the midspan centroid comes out at 2882 mm
against 2908 with nothing fitted at all.

    python benchmarks/generate_confederation.py --check
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
from vfife.base.compat import trapezoid

# --------------------------------------------------------------------------- #
#  The cross section (all lengths mm, x from the centreline, y DOWN from the
#  deck top).  Dilger & Langohr Fig. 9.
# --------------------------------------------------------------------------- #
HALF_DECK = 5820.0          #: half the deck width (total 11 640)
X_WEB_TOP_OUT = 3500.0      #: web outer face at deck level
X_BOT_OUT = 2500.0          #: web outer face at the soffit, 14 m down
WEB_LEAN = (X_WEB_TOP_OUT - X_BOT_OUT) / 14000.0
HAUNCH_START, T_DECK_CL, T_DECK_WEB = 820.0, 250.0, 650.0
#: cantilever soffit measured off Fig. 9: (x, thickness), 650 at the web to
#: 230 at the tip
CANT_X = np.array([3500, 3708, 3874, 4040, 4206, 4372, 4538, 4704, 4870,
                   5036, 5202, 5368, 5534, 5820], float)
CANT_T = np.array([650, 650, 616, 573, 508, 465, 422, 378, 335,
                   313, 292, 281, 275, 230], float)


def _edges(y, depth, t_bot, t_web):
    """(outer edge, void edge) in x at depth y below the deck top (vectorised).

    The cell's roof is the deck soffit, which drops from 250 mm at the
    centreline to 650 mm over the web.  Between those two depths the VOID runs
    from the centreline out to the haunch and everything outboard of it is
    still deck -- reading that band the other way round costs 0.68 m2 of area
    and 200 mm of centroid, and two independent area calculations is how it
    was caught.
    """
    y = np.asarray(y, float)
    xo = X_WEB_TOP_OUT - WEB_LEAN * y                    # web outer face
    xi = np.maximum(xo - t_web, 0.0)                     # web inner face
    out = np.interp(y, CANT_T[::-1], CANT_X[::-1])       # cantilever soffit
    out = np.where(y >= CANT_T[0], xo,
                   np.where(y <= CANT_T[-1], HALF_DECK, out))
    out = np.clip(out, xo, HALF_DECK)
    haunch = HAUNCH_START + (y - T_DECK_CL) / (T_DECK_WEB - T_DECK_CL) \
        * (X_WEB_TOP_OUT - t_web - HAUNCH_START)
    void = np.where(y <= T_DECK_CL, 0.0,
                    np.where(y >= T_DECK_WEB, xi, np.minimum(haunch, xi)))
    void = np.where(y >= depth - t_bot, 0.0, void)       # bottom slab is solid
    return out, void


def _half_width(y, depth, t_bot, t_web):
    out, void = _edges(y, depth, t_bot, t_web)
    return out - void


def section(depth, t_bot=None, t_web=None, n=4001):
    """(A mm2, centroid mm below the deck top, I mm4) at this girder depth."""
    t_bot = thickness(depth)[0] if t_bot is None else t_bot
    t_web = thickness(depth)[1] if t_web is None else t_web
    y = np.linspace(0.0, depth, n)
    w = _half_width(y, depth, t_bot, t_web)
    A = 2 * trapezoid(w, y)
    yc = 2 * trapezoid(w * y, y) / A
    return A, yc, 2 * trapezoid(w * y * y, y) - A * yc * yc


def section_weak(depth, n=4001):
    """Second moment about the VERTICAL axis (mm4): bending in plan.

    Not exercised by these cases -- the model is constrained to the vertical
    plane, as Ghali's is -- but the element asks for it.
    """
    t_bot, t_web = thickness(depth)
    y = np.linspace(0.0, depth, n)
    out, void = _edges(y, depth, t_bot, t_web)
    return 2.0 / 3.0 * trapezoid(out ** 3 - void ** 3, y)


def section_torsion(depth):
    """Bredt single-cell torsion constant (mm4), from the cell's mid-line."""
    t_bot, t_web = thickness(depth)
    t_top = 0.5 * (T_DECK_CL + T_DECK_WEB)
    h = depth - 0.5 * (t_top + t_bot)                  # cell mid-line height
    b_top = 2 * (X_WEB_TOP_OUT - 0.5 * t_web) - 0.0    # width at the roof
    b_bot = 2 * (X_WEB_TOP_OUT - WEB_LEAN * depth - 0.5 * t_web)
    Am = 0.5 * (b_top + b_bot) * h
    s_web = np.hypot(h, 0.5 * (b_top - b_bot))
    return 4 * Am ** 2 / (b_top / t_top + b_bot / t_bot + 2 * s_web / t_web)


def thickness(depth):
    """(bottom slab, web) in mm: the two numbers calibrated on Butcher's table,
    interpolated between the three stations where they are known."""
    d = np.asarray(depth, float)
    return (np.interp(d, [4500.0, 13000.0, 14000.0], [250.0, 900.0, 1102.0]),
            np.interp(d, [4500.0, 13000.0, 14000.0], [405.0, 400.0, 460.0]))


# --------------------------------------------------------------------------- #
#  The haunch.  Two published depths fix it; the third number checks it.
# --------------------------------------------------------------------------- #
D_PIER, D_MID = 14000.0, 4500.0
CANT_HALF = 96250.0         #: half the 192.5 m double cantilever (Ghali 2000)
#: d = D_MID + A0 (CANT_HALF - u)^P, fitted to 13 990 mm at the pier face
#: (u = 5.0 m) and 13 000 mm at Butcher's Sections 1/2 (u = 12.0 m)
P_HAUNCH = np.log(9490.0 / 8500.0) / np.log((CANT_HALF - 5000.0)
                                            / (CANT_HALF - 12000.0))
A_HAUNCH = 9490.0 / (CANT_HALF - 5000.0) ** P_HAUNCH


def depth_at(u):
    """Girder depth (mm) u mm from the pier centre, along the girder."""
    u = np.abs(np.asarray(u, float))
    d = D_MID + A_HAUNCH * np.maximum(CANT_HALF - u, 0.0) ** P_HAUNCH
    return np.minimum(d, D_PIER)        # the pier table is constant at 14.0 m


# --------------------------------------------------------------------------- #
#  The frame.  Ghali, Elbadry & Megally (2000) CJCE 27: 1139, Figs. 8 and 9.
# --------------------------------------------------------------------------- #
SPAN, OVERHANG = 250.0, 95.0            #: m; the girder runs -220 .. +220
X_PIER = SPAN / 2.0                     #: piers at +/- 125 m
DECK_Z = 40.8                           #: deck top, m above mean sea level
AXIS_BELOW_DECK = 1.52                  #: Ghali's reference axis
Z_BASE = {"P32": -21.7, "P31": -29.3}   #: pier bases (heights 60.98 / 68.58 m)

E_C = 38.7e+9                           #: Ec(28) measured, Ghali Fig. 2
#: the modulus the superimposed dead load of April 1997 actually met. Ghali's
#: own analysis ages Ec with MC90's exp[s (1 - sqrt(28/t))] and the s = 0.25 he
#: FITTED TO HIS MEASURED MODULI, so the only assumption here is the age; a
#: factor of four on it moves E by 5% (200 d -> 45.2 GPa, 800 d -> 47.4).
AGE_SDL = 400.0
E_SDL = E_C * np.exp(0.25 * (1.0 - np.sqrt(28.0 / AGE_SDL)))    # 46.5 GPa
RHO_C = 2450.0
#: the box outline holds 85% of the 8000 t Ghali gives for a 192.5 m double
#: cantilever; the rest is the pier diaphragm, the deviator and anchor blocks
#: and the tendons, which a beam model carries as mass and not as area
MASS_FACTOR = 8000.0 / 6811.0
ALPHA_C = 8.5e-6                        #: measured, Butcher (2009); design 8.3

#: pier shaft and base: hollow, octagonal above the ice shield, circular below,
#: 8 m outer at the top of the base growing to 22 m at the footing (Lin 2010).
#: The wall is not published; 0.6 m is the usual for a unit of this size.
PIER_T = 0.60
PIER_D_TOP, PIER_D_MID, PIER_D_BASE = 8.0, 8.0, 22.0
PIER_SHAFT_FRACTION = 0.62              #: of the height, above which D = 8 m

#: temperature rise on a hot summer day, Ghali Fig. 21: 27.3 C at the deck top
#: falling to 18.3 at the deck soffit, then to 17.33 at the girder soffit
T_TOP, T_SOFFIT_SLAB, T_BOTTOM = 27.3, 18.3, 17.33

#: the two trucks, Ghali Fig. 15. Axle loads (kN) and the spacings between the
#: centres of the rear tandem, the drive tandem and the steering axle; the
#: tandem's own spacing is not dimensioned and does not matter on a 440 m span.
TANDEM = 1.5
TRUCKS = {
    "blue": dict(total=379.0, tandem=(88.4, 77.9), steer=46.4, gaps=(8.30, 5.00)),
    "grey": dict(total=346.0, tandem=(88.0, 63.3), steer=43.4, gaps=(8.70, 4.70)),
}


def truck_axles(name):
    """[(offset from the resultant, force N), ...] for one truck."""
    t = TRUCKS[name]
    g1, g2 = t["gaps"]
    pos = [-TANDEM / 2, TANDEM / 2, g1 - TANDEM / 2, g1 + TANDEM / 2, g1 + g2]
    f = [t["tandem"][0]] * 2 + [t["tandem"][1]] * 2 + [t["steer"]]
    if abs(sum(f) - t["total"]) > 0.05:
        raise ValueError(f"{name} truck axles sum to {sum(f)}, not {t['total']}")
    x0 = sum(p * q for p, q in zip(pos, f)) / sum(f)
    return [(p - x0, q * 1e+3) for p, q in zip(pos, f)]


# --------------------------------------------------------------------------- #
#  Band table: the girder's properties are stepped, by distance from a pier.
# --------------------------------------------------------------------------- #
#: band edges as distance from the pier centre (m). The last one runs out to
#: the cantilever tip; beyond it the girder is the constant 4.5 m centre.
BAND_EDGES = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 55.0, 70.0, 85.0,
              95.0]


def stations(h=5.0):
    """Girder node stations (m): a uniform 5 m grid with the piers on it.

    Uniform on purpose. Relaxation's fictitious mass comes from the stiffest
    element, so a short element inside the deepest band (where EI is 22x the
    midspan value) costs iterations everywhere and buys nothing -- the pier
    table is a constant 14.0 m deep, so the two bands it spans are identical.
    """
    return np.round(np.arange(-220.0, 220.0 + 1e-9, h), 6)


def band_of(u):
    """Section band for a point u metres from the nearest pier."""
    for j in range(len(BAND_EDGES) - 1):
        if BAND_EDGES[j] <= u < BAND_EDGES[j + 1]:
            return f"b{j:02d}"
    return "bmid"


def bands():
    """[(name, u_lo, u_hi, depth m, A m2, iz m4, iy m4, j m4), ...]."""
    out = []
    for k in range(len(BAND_EDGES) - 1):
        lo, hi = BAND_EDGES[k], BAND_EDGES[k + 1]
        u = np.linspace(lo, hi, 21) * 1e+3
        out.append((f"b{k:02d}", lo, hi) + _avg(depth_at(u)))
    out.append(("bmid", BAND_EDGES[-1], X_PIER) + _avg(np.array([D_MID])))
    return out


def _avg(d):
    pr = np.array([section(v) for v in d])
    return (d.mean() / 1e+3, pr[:, 0].mean() / 1e+6,
            pr[:, 2].mean() / 1e+12,
            np.mean([section_weak(v) for v in d]) / 1e+12,
            np.mean([section_torsion(v) for v in d]) / 1e+12)


def pier_bands(nseg=2):
    """[(name, z_lo_fraction, z_hi_fraction, A m2, I m4), ...] for a pier."""
    #: 6 elements a pier: h=5 m girder with 6-element piers reproduces the
    #: 2.5 m / 12-element answer to 0.3%, and the short very stiff base
    #: elements are what set relaxation's time step for the whole model.
    out = [("pshaft", PIER_SHAFT_FRACTION, 1.0) + _annulus(PIER_D_TOP)]
    for k in range(nseg):
        a = PIER_SHAFT_FRACTION * k / nseg
        b = PIER_SHAFT_FRACTION * (k + 1) / nseg
        d = np.interp((a + b) / 2, [0.0, PIER_SHAFT_FRACTION],
                      [PIER_D_BASE, PIER_D_MID])
        out.append((f"pbase{k}", a, b) + _annulus(d))
    return out


def _annulus(d_out, t=None):
    t = PIER_T if t is None else t          # read at call time: the sensitivity
    di = d_out - 2 * t                      # sweep in --check rebinds PIER_T
    return (np.pi / 4 * (d_out ** 2 - di ** 2),
            np.pi / 64 * (d_out ** 4 - di ** 4))


# --------------------------------------------------------------------------- #
#  Temperature: the equivalent LINEAR field of Ghali's profile, on the real
#  section at each depth.  A `general` section has no fibres, so the nonlinear
#  remainder -- which is self-equilibrated and moves nothing -- is dropped
#  here rather than inside the solver.
# --------------------------------------------------------------------------- #
def temperature_field(depth, axis_below_deck=None, t_slab=None, n=4001):
    """(T at the element axis, dT/dy in K/m upwards) for Ghali's profile.

    `t_slab` is how thick the hot band at the deck is -- the depth at which the
    profile reaches 18.3 C. Fig. 21 does not dimension it; the default is the
    deck's 650 mm over the web, and the cantilever tip deflection is what reads
    the choice (250 mm moves it 25%).
    """
    t_bot, t_web = thickness(depth)
    y = np.linspace(0.0, depth, n)                      # down from the deck top
    w = _half_width(y, depth, t_bot, t_web)
    t_slab = T_DECK_WEB if t_slab is None else t_slab    # deck soffit over the web
    T = np.where(y <= t_slab,
                 T_TOP + (T_SOFFIT_SLAB - T_TOP) * y / t_slab,
                 T_SOFFIT_SLAB + (T_BOTTOM - T_SOFFIT_SLAB)
                 * (y - t_slab) / (depth - t_slab))
    A = 2 * trapezoid(w, y)
    yc = 2 * trapezoid(w * y, y) / A
    I = 2 * trapezoid(w * y * y, y) - A * yc * yc
    Tbar = 2 * trapezoid(T * w, y) / A
    g_down = 2 * trapezoid(T * (y - yc) * w, y) / I   # K per mm downwards
    y_axis = yc if axis_below_deck is None else axis_below_deck * 1e+3
    return Tbar + g_down * (y_axis - yc), -g_down * 1e+3


# --------------------------------------------------------------------------- #
#  An independent plane frame (ux, uz, ry), for cross-checking only.
# --------------------------------------------------------------------------- #
class Frame:
    def __init__(self):
        self.nodes, self.elems = [], []

    def node(self, x, z):
        for k, (a, b) in enumerate(self.nodes):
            if abs(a - x) < 1e-9 and abs(b - z) < 1e-9:
                return k
        self.nodes.append((x, z))
        return len(self.nodes) - 1

    def element(self, i, j, EA, EI, dth=(0.0, 0.0)):
        """dth = (free axial strain, free HOGGING curvature).

        Hogging positive, i.e. the sign of the library's `gradient`: a warmer
        top makes a simply supported span rise. Writing it the other way round
        -- the mathematical d2w/dx2 -- is a sign error that survives every
        check a uniform temperature can make, because a uniform rise has no
        curvature at all. It took disagreeing with the solver on case_28c to
        find, and on the way it produced a confident wrong reading of Ghali's
        Fig. 21.
        """
        self.elems.append([i, j, EA, EI, dth])
        return len(self.elems) - 1

    def _geom(self, e):
        i, j = e[0], e[1]
        dx = self.nodes[j][0] - self.nodes[i][0]
        dz = self.nodes[j][1] - self.nodes[i][1]
        L = np.hypot(dx, dz)
        return L, dx / L, dz / L

    def _T(self, c, s):
        T = np.zeros((6, 6))
        T[0, 0] = T[3, 3] = c
        T[0, 1] = T[3, 4] = s
        T[1, 0] = T[4, 3] = -s
        T[1, 1] = T[4, 4] = c
        T[2, 2] = T[5, 5] = 1.0
        return T

    def _ke(self, L, EA, EI):
        k = np.zeros((6, 6))
        k[0, 0] = k[3, 3] = EA / L
        k[0, 3] = k[3, 0] = -EA / L
        a, b, c = 12 * EI / L ** 3, 6 * EI / L ** 2, EI / L
        k[1, 1] = k[4, 4] = a
        k[1, 4] = k[4, 1] = -a
        k[1, 2] = k[2, 1] = k[1, 5] = k[5, 1] = b
        k[2, 4] = k[4, 2] = k[4, 5] = k[5, 4] = -b
        k[2, 2] = k[5, 5] = 4 * c
        k[2, 5] = k[5, 2] = 2 * c
        return k

    def solve(self, loads, fixed):
        """loads: {dof: value}; fixed: iterable of dofs. Returns u."""
        n = 3 * len(self.nodes)
        K, f = np.zeros((n, n)), np.zeros(n)
        for e in self.elems:
            L, c, s = self._geom(e)
            ke, T = self._ke(L, e[2], e[3]), self._T(c, s)
            d = np.array([-e[4][0] * L / 2, 0.0, e[4][1] * L / 2,
                          e[4][0] * L / 2, 0.0, -e[4][1] * L / 2])
            g = T.T @ ke @ T
            fe = T.T @ (ke @ d)
            ids = np.array([3 * e[0], 3 * e[0] + 1, 3 * e[0] + 2,
                            3 * e[1], 3 * e[1] + 1, 3 * e[1] + 2])
            K[np.ix_(ids, ids)] += g
            f[ids] += fe
        for d, v in loads.items():
            f[d] += v
        free = np.setdiff1d(np.arange(n), np.asarray(sorted(fixed), int))
        u = np.zeros(n)
        u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
        return u

    def point_load(self, e, xi, P):
        """Consistent (Hermite) nodal forces for a vertical load on element e."""
        L, c, s = self._geom(self.elems[e])
        a, b = float(np.clip(xi, 0, 1)), 1.0 - float(np.clip(xi, 0, 1))
        n1, n2 = 1 - 3 * a * a + 2 * a ** 3, 3 * a * a - 2 * a ** 3
        m1, m2 = L * a * b * b, -L * a * a * b
        i, j = self.elems[e][0], self.elems[e][1]
        return {3 * i + 1: P * n1, 3 * i + 2: P * m1,
                3 * j + 1: P * n2, 3 * j + 2: P * m2}


def reference_frame(nel_pier=6, axis="centroid", E=E_C):
    """Ghali's frame, built as straight elements.

    `axis` is where the girder nodes sit: "centroid" puts them on the section
    centroid (what a beam element is derived for, so the axis dips 4.3 m from
    midspan to the pier), "reference" puts them on Ghali's straight line 1.52 m
    below the deck top and accepts the offset error.
    """
    fr = Frame()
    xs = stations()

    def z_of(x):
        if axis == "reference":
            return DECK_Z - AXIS_BELOW_DECK
        return DECK_Z - section(depth_at(abs(abs(x) - X_PIER) * 1e+3))[1] / 1e+3

    gn = [fr.node(x, z_of(x)) for x in xs]
    ge = []
    for k in range(len(xs) - 1):
        u = abs(abs(0.5 * (xs[k] + xs[k + 1])) - X_PIER) * 1e+3
        d = depth_at(u) if u <= CANT_HALF else D_MID
        A, _, I = section(d)
        ge.append(fr.element(gn[k], gn[k + 1], E * A / 1e+6, E * I / 1e+12))
    piers = {}
    for name, sgn in (("P32", -1.0), ("P31", 1.0)):
        top = gn[int(np.argmin(np.abs(xs - sgn * X_PIER)))]
        zt, zb = fr.nodes[top][1], Z_BASE[name]
        nz = np.linspace(zt, zb, nel_pier + 1)
        prev = top
        for k in range(nel_pier):
            frac = 1.0 - (0.5 * (nz[k] + nz[k + 1]) - zb) / (zt - zb)
            frac = 1.0 - frac
            d = (PIER_D_TOP if frac >= PIER_SHAFT_FRACTION else
                 np.interp(frac, [0.0, PIER_SHAFT_FRACTION],
                           [PIER_D_BASE, PIER_D_MID]))
            A, I = _annulus(d)
            nxt = fr.node(sgn * X_PIER, nz[k + 1])
            fr.element(prev, nxt, E * A, E * I)
            prev = nxt
        piers[name] = prev
    fixed = [3 * piers[n] + k for n in piers for k in (0, 1, 2)]
    return fr, xs, gn, ge, fixed


def truck_deflection(x_res, axis="centroid", E=E_C, **kw):
    """Vertical deflection (m, down positive) of the girder nodes with both
    trucks' resultants at `x_res`."""
    fr, xs, gn, ge, fixed = reference_frame(axis=axis, E=E, **kw)
    loads = {}
    for name in TRUCKS:
        for off, P in truck_axles(name):
            x = x_res + off
            if x < xs[0] or x > xs[-1]:
                continue
            k = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
            xi = (x - xs[k]) / (xs[k + 1] - xs[k])
            for d, v in fr.point_load(ge[k], xi, -P).items():
                loads[d] = loads.get(d, 0.0) + v
    u = fr.solve(loads, fixed)
    return xs, -np.array([u[3 * n + 1] for n in gn])


#: superimposed dead load placed in April 1997, Ghali (2000) p. 1147
Q_SDL = 5.4e+4                          #: N/m, stated
DROP_IN = 60.0                          #: m, simply supported tip to tip
#: each overhang tip also carries half a drop-in beam's share of the same load
V_SDL = Q_SDL * DROP_IN / 2.0           #: 1620 kN


def sdl_deflection(q=Q_SDL, V=V_SDL, axis="centroid", E=E_C, **kw):
    """The superimposed dead load: [midspan, x=-60, tip W, tip E] mm, DOWN
    positive.

    `V` is the reaction the simply supported 60 m drop-in beam delivers to
    each overhang tip under the same q -- the V of Ghali's own Fig. 9. It is
    a separate argument because leaving it out is the obvious mistake and is
    worth being able to price: without it the tips take 28 mm and midspan
    104, with it 107 and 79.
    """
    fr, xs, gn, ge, fixed = reference_frame(axis=axis, E=E, **kw)
    loads = {}
    for e in ge:
        L, _, _ = fr._geom(fr.elems[e])
        i, j = fr.elems[e][0], fr.elems[e][1]
        for d, v in ((3 * i + 1, -q * L / 2), (3 * j + 1, -q * L / 2),
                     (3 * i + 2, -q * L * L / 12), (3 * j + 2, q * L * L / 12)):
            loads[d] = loads.get(d, 0.0) + v
    for n in (gn[0], gn[-1]):
        loads[3 * n + 1] = loads.get(3 * n + 1, 0.0) - V
    u = fr.solve(loads, fixed)
    w = -1e+3 * np.array([u[3 * n + 1] for n in gn])
    i = lambda x: int(np.argmin(np.abs(xs - x)))       # noqa: E731
    return [w[i(0.0)], w[i(-60.0)], w[0], w[-1]]


def thermal_deflection(uniform=1, gradient=1, t_slab=None, axis="centroid",
                       E=E_C, alpha=ALPHA_C, **kw):
    """[midspan, x=-60, x=-180, west tip, east tip] in mm, POSITIVE UP."""
    fr, xs, gn, ge, fixed = reference_frame(axis=axis, E=E, **kw)
    below = None if axis == "centroid" else AXIS_BELOW_DECK
    for k, e in enumerate(ge):
        d = depth_at(abs(abs(0.5 * (xs[k] + xs[k + 1])) - X_PIER) * 1e+3)
        T0, g = temperature_field(d, below, t_slab=t_slab)
        fr.elems[e][4] = (alpha * T0 * uniform, alpha * g * gradient)
    u = fr.solve({}, fixed)
    w = 1e+3 * np.array([u[3 * n + 1] for n in gn])
    i = lambda x: int(np.argmin(np.abs(xs - x)))       # noqa: E731
    return [w[i(0.0)], w[i(-60.0)], w[i(-180.0)], w[0], w[-1]]


# --------------------------------------------------------------------------- #
#  Mesh
# --------------------------------------------------------------------------- #
def write_mesh(path, nel_pier=6, axis="centroid"):
    """Gmsh 2.2 line mesh: one physical group per section band, node groups at
    the stations the contracts monitor."""
    fr, xs, gn, ge, _ = reference_frame(nel_pier=nel_pier, axis=axis)
    nodes = list(fr.nodes)
    gband = [band_of(abs(abs(0.5 * (xs[k] + xs[k + 1])) - X_PIER))
             for k in range(len(xs) - 1)]
    names = []
    pb = pier_bands()
    lines = []                       # (physical name, n1, n2)
    for k in range(len(xs) - 1):
        lines.append((gband[k], gn[k], gn[k + 1]))
    ne = len(fr.elems)
    for e in fr.elems[len(ge):]:
        zt = max(nodes[e[0]][1], nodes[e[1]][1])
        zb = min(nodes[e[0]][1], nodes[e[1]][1])
        x = nodes[e[0]][0]
        base = Z_BASE["P32" if x < 0 else "P31"]
        top = DECK_Z - AXIS_BELOW_DECK if axis == "reference" else \
            max(n[1] for n in nodes if abs(n[0] - x) < 1e-9)
        frac = 1.0 - (0.5 * (zt + zb) - base) / (top - base)
        frac = 1.0 - frac
        nm = "pshaft"
        for p in pb[1:]:
            if p[1] <= frac < p[2]:
                nm = p[0]
                break
        lines.append((nm, e[0], e[1]))
    del ne

    def near(x):
        return int(np.argmin([abs(n[0] - x) + abs(n[1] - (nodes[gn[0]][1]))
                              for n in nodes]))

    pts = {
        "tip_west": gn[0], "tip_east": gn[-1],
        "mid": gn[int(np.argmin(np.abs(xs)))],
        "p32": gn[int(np.argmin(np.abs(xs + X_PIER)))],
        "p31": gn[int(np.argmin(np.abs(xs - X_PIER)))],
        "q_west": gn[int(np.argmin(np.abs(xs + 60.0)))],
        "q_east": gn[int(np.argmin(np.abs(xs - 60.0)))],
        "o_west": gn[int(np.argmin(np.abs(xs + 180.0)))],
        "o_east": gn[int(np.argmin(np.abs(xs - 180.0)))],
        "base32": int(np.argmin([abs(n[0] + X_PIER) + abs(n[1] - Z_BASE["P32"])
                                 for n in nodes])),
        "base31": int(np.argmin([abs(n[0] - X_PIER) + abs(n[1] - Z_BASE["P31"])
                                 for n in nodes])),
    }
    del near
    band_names = sorted({nm for nm, _, _ in lines})
    names = list(pts) + band_names
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat",
           "$PhysicalNames", str(len(names))]
    out += [f'0 {k + 1} "{nm}"' for k, nm in enumerate(pts)]
    out += [f'1 {len(pts) + k + 1} "{nm}"' for k, nm in enumerate(band_names)]
    out += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    out += [f"{k + 1} {x:.9g} 0 {z:.9g}" for k, (x, z) in enumerate(nodes)]
    out += ["$EndNodes", "$Elements", str(len(pts) + len(lines))]
    eid = 1
    for k, nm in enumerate(pts):
        out.append(f"{eid} 15 2 {k + 1} {k + 1} {pts[nm] + 1}")
        eid += 1
    tag = {nm: len(pts) + k + 1 for k, nm in enumerate(band_names)}
    for nm, i, j in lines:
        out.append(f"{eid} 1 2 {tag[nm]} {tag[nm]} {i + 1} {j + 1}")
        eid += 1
    out += ["$EndElements", ""]
    pathlib.Path(path).write_text("\n".join(out), encoding="utf-8")
    return len(nodes), len(lines), band_names


# --------------------------------------------------------------------------- #
def _check():
    print("SECTION -- two thicknesses fitted on Butcher's table, three")
    print("          quantities predicted (Butcher 2009, Appendix E)")
    for nm, d, Ap, yp, Ip in (("pier face ", 13990.0, 22.247193e+6, 6752.0, 5.7779e+14),
                              ("Sections1/2", 13000.0, 19.218202e+6, 6517.0, 4.5496e+14)):
        A, yc, I = section(d, n=40001)
        print("  %s d=%6.0f  A %+5.1f%%   centroid %+5.1f%%   I %+5.1f%%"
              % (nm, d, 100 * (A / Ap - 1), 100 * ((d - yc) / yp - 1),
                 100 * (I / Ip - 1)))
    A, yc, I = section(D_MID, n=40001)
    print("  midspan     d=  4500  centroid %.0f mm vs Butcher 2908 (%+.1f%%),"
          " nothing fitted" % (D_MID - yc, 100 * ((D_MID - yc) / 2908.0 - 1)))

    print("\nHAUNCH -- d = 4500 + %.5g (96250 - u)^%.3f mm" % (A_HAUNCH, P_HAUNCH))
    slope = np.degrees(np.arctan(P_HAUNCH * A_HAUNCH
                                 * (CANT_HALF - 5000.0) ** (P_HAUNCH - 1)))
    print("  soffit slope at the pier face %.2f deg (Dilger states 7.80);"
          " a parabola would give 11.2" % slope)

    u = np.linspace(0, CANT_HALF, 241)
    A = np.array([section(v)[0] for v in depth_at(u)])
    mass = 2 * trapezoid(A, u) / 1e+9 * RHO_C / 1e+3
    print("\nMASS -- the 192.5 m double cantilever: %.0f t of box outline"
          " against Ghali's 8000 t (%.0f%%)" % (mass, 100 * mass / 8000))
    print("  the rest is the pier diaphragm, blisters and tendons: carried as"
          " mass (x%.3f), not as area" % MASS_FACTOR)

    print("\nWEB -- Dilger's w_D2 is the self weight of the WEBS, not of the")
    print("       girder: 0.248 MN/m at the pier face")
    y = np.linspace(0.0, 13990.0, 40001)
    tb, tw = thickness(13990.0)
    w = _half_width(y, 13990.0, tb, tw)
    m = (y > T_DECK_WEB) & (y < 13990.0 - tb)
    Aw = 2 * trapezoid(w[m], y[m]) / 1e+6
    print("  model %.3f MN/m (%+.0f%%)"
          % (Aw * RHO_C * 9.80665 / 1e+6,
             100 * (Aw * RHO_C * 9.80665 / 1e+6 / 0.248 - 1)))

    print("\nTRUCKS -- axle offsets from the resultant (m) and loads (kN)")
    for nm in TRUCKS:
        ax = truck_axles(nm)
        print("  %-5s %s" % (nm, ", ".join("%+.2f:%.1f" % (o, p / 1e+3)
                                           for o, p in ax)))

    print("\nTEMPERATURE -- Ghali Fig. 21 on the real section")
    for d in (D_PIER, 9000.0, D_MID):
        T0, g = temperature_field(d)
        print("  depth %5.1f m: mean rise at the centroid %5.2f K,"
              " gradient %+6.3f K/m" % (d / 1e+3, T0, g))

    print("\nINDEPENDENT PLANE FRAME (this file's own direct stiffness solve)")
    print("  vertical displacement in mm, POSITIVE = UP")
    print("  %-28s %7s %7s %7s %7s %7s"
          % ("", "midspan", "x=-60", "x=-180", "tip W", "tip E"))
    for nm, uT, ug in (("temperature: mean rise only", 1, 0),
                       ("             gradient only", 0, 1),
                       ("             whole profile", 1, 1)):
        print("  %-28s %+7.1f %+7.1f %+7.1f %+7.1f %+7.1f"
              % ((nm,) + tuple(thermal_deflection(uT, ug))))
    print("  %-28s %+7.1f %+7.1f %+7.1f %+7.1f %+7.1f"
          % ("  Ghali Fig. 21, digitised", 60.6, 47.0, -43.7, -73.5, -73.5))
    print("  the hot band at the deck is the one thing Fig. 21 does not")
    print("  dimension, and the tips are what reads it:")
    for t in (250.0, 400.0, 650.0):
        print("    deck band %4.0f mm            %+7.1f %+7.1f %+7.1f %+7.1f %+7.1f"
              % ((t,) + tuple(thermal_deflection(1, 1, t_slab=t))))

    print("\n  trucks at midspan (Fig. 17: +17.1 measured, +13.7 Ghali's analysis)")
    for axis in ("centroid", "reference"):
        xs, w = truck_deflection(0.0, axis=axis)
        i = int(np.argmin(np.abs(xs)))
        print("    girder axis on the %-9s  midspan %+6.1f  tips %+6.1f / %+6.1f"
              % (axis, 1e+3 * w[i], 1e+3 * w[0], 1e+3 * w[-1]))
    print("    measured                        midspan %+6.1f  tips %+6.1f / %+6.1f"
          % (17.1, -5.4, -7.6))
    for E in (38.7e+9, 46.4e+9):
        xs, w = truck_deflection(0.0, E=E)
        print("    E = %.1f GPa                   midspan %+6.1f"
              % (E / 1e+9, 1e+3 * w[int(np.argmin(np.abs(xs)))]))
    global PIER_T
    t0 = PIER_T
    for f in (0.5, 1.0, 2.0):
        PIER_T = t0 * f
        xs, w = truck_deflection(0.0)
        mid_t = thermal_deflection(1, 0)[0]
        print("    pier wall x%.1f                  trucks %+6.1f   uniform rise %+6.1f"
              % (f, 1e+3 * w[int(np.argmin(np.abs(xs)))], mid_t))
    PIER_T = t0

    print("\nSUPERIMPOSED DEAD LOAD -- %.0f kN/m, placed April 1997 (case_28d)"
          % (Q_SDL / 1e+3))
    print("  Ghali p. 1147 states his own analysis gives about 70 mm at"
          " midspan and 80 at the tips.")
    print("  %-34s %7s %7s %7s %7s" % ("", "midspan", "x=-60", "tip W", "tip E"))
    print("  %-34s %7.1f %7.1f %7.1f %7.1f"
          % (("without the drop-in reaction",) + tuple(sdl_deflection(V=0.0))))
    for E in (E_C, 44.65e+9, E_SDL, 48.4e+9):
        for axis in ("centroid", "reference"):
            print("  E = %4.1f GPa, axis on the %-9s %7.1f %7.1f %7.1f %7.1f"
                  % ((E / 1e+9, axis) + tuple(sdl_deflection(E=E, axis=axis))))
    print("  E = %.2f GPa is Ec(28) = 38.7 aged to %.0f days by MC90's own"
          % (E_SDL / 1e+9, AGE_SDL))
    print("  exp[s (1 - sqrt(28/t))] with the s = 0.25 Ghali FITTED TO MEASURED")
    print("  MODULI -- so it is not a free parameter, only the age is, and the")
    print("  age is worth little: t = 200 d gives 45.2 GPa and t = 800 d 47.4.")
    for f in (0.5, 1.0, 2.0):
        PIER_T = t0 * f
        print("    pier wall x%.1f                   %7.1f %7.1f %7.1f %7.1f"
              % ((f,) + tuple(sdl_deflection(E=E_SDL))))
    PIER_T = t0


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
    p = out / "confederation_frame.msh"
    n, ne, bn = write_mesh(p)
    print(p, n, "nodes", ne, "elements", len(bn), "bands:", " ".join(bn))
    print("  # girder bands: u is the distance from the pier centre")
    for nm, lo, hi, d, A, iz, iy, j in bands():
        print("    %-5s: {type: beam, material: concrete, shape: general,"
              " area: %.4f, iz: %.4g, iy: %.4g, j: %.4g}   # u %.1f-%.1f m,"
              " depth %.2f m" % (nm, A, iz, iy, j, lo, hi, d))
    for nm, lo, hi, A, I in pier_bands():
        print("    %-7s: {type: beam, material: concrete, shape: general,"
              " area: %.4f, iz: %.4g, iy: %.4g, j: %.4g}   # %.2f-%.2f of H"
              % (nm, A, I, I, 2 * I, lo, hi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
