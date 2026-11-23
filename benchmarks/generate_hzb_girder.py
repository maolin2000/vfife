"""The Hangzhou Bay Bridge 70 m box girder (cases/case_24): section, meshes,
tendons, loads, and the hand checks behind the case files.

SOURCES
  Xu Aimin, PhD thesis, Zhejiang University 2008, Fig. 2.4: the midspan
    section of the 70 m precast girder, dimensions in cm; 2140 t per
    girder; ANSYS beam188 simply supported modal frequencies 1.8498,
    4.7565, 6.7249 Hz (E 3.5e10, nu 1/6, rho 2500, section varying).
  Chen Hengzhi, PhD thesis, Zhejiang University 2005, §6.6: the 3 x 70 m
    continuous unit (precast simply supported, made continuous), its
    tendon scheme (Fig. 6.25), loads q_d1 306 + q_d2 52 kN/m, live
    F_L 818 kN + q_L 25 kN/m per JTG D60-2004, and the proportional
    push to failure: load case 1 (F_L at the middle span's midspan, q_L
    over the middle span) k_u = 16.8, delta_u = 727 mm; load case 2
    (side span, 40 m from the abutment, q_L over that span) k_u = 11.3,
    delta = 384 mm; the dead-load + prestress strain state of Fig. 6.29.

SECTION (metres, v up from the soffit, t across; the node line is at the
centroid). Single-cell box, 4.00 m deep, 15.80 m wide at the top, 6.25 m
at the soffit; webs 0.50 m thick, inclined so that the outer face runs
from t = 3.125 at the soffit to 3.998 at the top (a horizontal thickness
of 0.512 m); top slab 0.27 m between the webs, bottom slab 0.25 m; 0.5 x
0.5 m fillets in the four inner corners; cantilevers 3.9 m long, 0.30 m
at the root tapering to 0.20 m at the tip. The girder's end thickening
(10 m at each end: webs 1.10, top 0.55, bottom 0.95) and the diaphragms
are NOT in the section, which is why the fibre section weighs 243 kN/m
against the 306 kN/m the girder does; the dead load is applied as the
stated line load, not as gravity on the section.

Every quantity below is derived here so the case files carry numbers
with a source: run `python benchmarks/generate_hzb_girder.py --check`.
"""
import argparse
import pathlib

import numpy as np

# ---------------------------------------------------------------- section
H = 4.00                       # depth
T_SOFFIT = 3.125               # half width of the soffit (outer web face at v = 0)
T_TOP = 3.998                  # outer web face at v = H
WEB_H = 0.512                  # horizontal thickness of the inclined web
T_SLAB = 0.27                  # top slab between the webs
B_SLAB = 0.25                  # bottom slab
FILLET = 0.50
CANT_TIP = 7.90                # half of 15.80
CANT_ROOT_T, CANT_TIP_T = 0.30, 0.20
SLOPE = (T_TOP - T_SOFFIT) / H


def t_out(v):
    return T_SOFFIT + SLOPE * v


def t_in(v):
    return T_SOFFIT - WEB_H + SLOPE * v


def section_patches(refine=1):
    """The concrete as quadrilateral patches [{corners, ny, nz}] in (v, t):
    v up, t across; `patch_fibers` puts one fibre at every cell. `refine`
    multiplies the subdivision."""
    r = int(refine)
    P = []

    def quad(c, ny, nz):
        P.append({"corners": [[float(v), float(t)] for v, t in c], "ny": ny * r, "nz": nz * r})

    vb, vt = B_SLAB, H - T_SLAB
    # bottom slab between the inner web faces (a trapezoid)
    quad([(0, -t_in(0)), (0, t_in(0)), (vb, t_in(vb)), (vb, -t_in(vb))], 3, 8)
    # top slab between the inner web faces
    quad([(vt, -t_in(vt)), (vt, t_in(vt)), (H, t_in(H)), (H, -t_in(H))], 3, 10)
    for sgn in (-1, 1):
        # the web, soffit to deck, full depth
        quad([(0, sgn * t_in(0)), (0, sgn * t_out(0)), (H, sgn * t_out(H)), (H, sgn * t_in(H))], 12, 3)
        # bottom fillet: a triangle as a collapsed quad
        quad([(vb, sgn * t_in(vb)), (vb, sgn * (t_in(vb) - FILLET)), (vb + FILLET, sgn * t_in(vb + FILLET)),
              (vb + FILLET, sgn * t_in(vb + FILLET))], 2, 2)
        # top fillet
        quad([(vt, sgn * t_in(vt)), (vt - FILLET, sgn * t_in(vt - FILLET)), (vt, sgn * (t_in(vt) - FILLET)),
              (vt, sgn * (t_in(vt) - FILLET))], 2, 2)
        # cantilever, root at the outer web face
        quad([(H - CANT_ROOT_T, sgn * t_out(H)), (H - CANT_TIP_T, sgn * CANT_TIP), (H, sgn * CANT_TIP),
              (H, sgn * t_out(H))], 2, 6)
    return P


def section_properties(refine=4):
    """A, centroid height v_c, I about the horizontal centroidal axis (I_y,
    bending in the vertical plane), I about the vertical axis (I_z), from
    the patches at a fine subdivision."""
    from vfife.elements.fiber_section import patch_fibers
    v, t, a, _ = patch_fibers(section_patches(refine))
    A = a.sum(); vc = (a * v).sum() / A
    Iy = (a * (v - vc) ** 2).sum(); Iz = (a * t * t).sum()
    return {"A": float(A), "v_c": float(vc), "I_y": float(Iy), "I_z": float(Iz), "n": int(len(a))}


def torsion_constant():
    """Bredt's closed-cell J = 4 A0^2 / sum(s/t) for the single cell (the
    cantilevers add a little open-section J, ignored)."""
    vb, vt = B_SLAB, H - T_SLAB
    A0 = 0.5 * (2 * t_in(vb) + 2 * t_in(vt)) * (vt - vb)
    web_len = np.hypot(vt - vb, (t_in(vt) - t_in(vb)))
    s_over_t = 2 * t_in(vb) / B_SLAB + 2 * t_in(vt) / T_SLAB + 2 * web_len / 0.50
    return 4 * A0 ** 2 / s_over_t


# ---------------------------------------------------------------- tendons
#  ONE ENTRY PER DUCT. The first version lumped each group
#  into one steel row on the centreline -- mechanically identical for
#  bending in the vertical plane (the transverse coordinate enters only
#  I_z and the torsion), but the picture was wrong: the pier's 22 ducts
#  came out as four coincident lines at mid-width, and the web tendons ran
#  up the middle of the box instead of inside the webs. Every duct now
#  carries its own (y, z) profile, which also puts the PLAN path in: a web
#  duct follows the inclined web, so it moves half a metre sideways as it
#  rises, and its friction loss counts that angle.
STRAND = 140e-6                # m2, one 15.24 mm strand
SIGMA_K = 1395e6               # Pa, control stress at the jack (0.75 f_pk)
#: web ducts, one per web: height above the soffit at the anchor, at the low
#: point, where the low point is reached, and the strands in the duct
#: (Chen Fig. 6.25b elevation and 6.25c section)
WEB = {"W1": (3.00, 0.72, 24.0, 27), "W2": (2.50, 0.60, 21.0, 27),
       "W3": (2.05, 0.47, 18.0, 12), "W4": (1.55, 0.35, 14.0, 27),
       "W5": (1.05, 0.35, 12.0, 27)}
#: bottom-slab ducts, per side: (|t|, strands). Fig. 6.25c reads
#: "3xB1 3xB2 | 3xB2 3xB1" across the soffit -- B1 (19 strands) outboard,
#: B2 (12) inboard
BOTTOM = [(2.50, 19), (2.20, 19), (1.90, 19), (1.50, 12), (1.00, 12), (0.50, 12)]
V_BOTTOM = 0.12
#: deck continuity ducts over a pier, per side: (|t|, half-length from the
#: pier, strands). The transverse positions are measured off the plan panel
#: of Fig. 6.25b (11 ducts per half between 2.38 and 5.16 m from the
#: centreline at 0.28 m spacing, the outermost being the longest, T10/T11);
#: the four half-lengths are the elevation's four groups, which is the
#: resolution that figure supports.
DECK = [(5.16, 32.0), (4.88, 32.0),
        (4.60, 22.0), (4.35, 22.0), (4.07, 22.0),
        (3.67, 15.0), (3.43, 15.0), (3.19, 15.0),
        (2.90, 9.0), (2.62, 9.0), (2.38, 9.0)]
DECK_STRANDS = 12
V_TOP = H - 0.13               # the ducts sit in the top slab


def web_t(v):
    """The transverse position of a web duct at height v: the middle of the
    inclined web, so a rising tendon moves outboard as it goes."""
    return t_in(v) + 0.5 * WEB_H


def tendon_layout(spans, L=70.0, v_c=None, long_term=0.90):
    """Every duct of a girder of `spans` equal spans as `tendons:` entries
    (profiles [s, y, z] in the section axes: y up from the centroid, z
    across, s along the member from x = 0), plus the `bar` sections they
    need. `long_term` is the flat factor for relaxation, shrinkage and
    creep on the jacking stress; friction and anchor set the solver works
    out per profile, on the duct's own three-dimensional polyline."""
    if v_c is None:
        v_c = section_properties()["v_c"]
    out, bars = [], {}

    def bar(n):
        name = {27: "duct27", 19: "duct19", 12: "duct12"}[n]
        bars[name] = {"type": "bar", "material": "strand", "shape": "general",
                      "area": round(n * STRAND, 8)}
        return name

    jack = round(SIGMA_K * long_term, -5)
    friction = {"mu": 0.25, "k": 0.0015, "jack": "both", "anchor_set": 0.006}
    for k in range(spans):
        x0 = k * L
        for name, (va, vl, xl, n) in WEB.items():
            for sgn in (-1, 1):
                prof = [[x0, va - v_c, sgn * web_t(va)],
                        [x0 + xl, vl - v_c, sgn * web_t(vl)],
                        [x0 + L - xl, vl - v_c, sgn * web_t(vl)],
                        [x0 + L, va - v_c, sgn * web_t(va)]]
                out.append({"name": f"{name}{'L' if sgn < 0 else 'R'}_s{k + 1}",
                            "section": bar(n), "profile": prof,
                            "extent": [x0, x0 + L], "jack": jack, "friction": friction})
        for t, n in BOTTOM:
            for sgn in (-1, 1):
                out.append({"name": f"B{n}_{t:g}{'L' if sgn < 0 else 'R'}_s{k + 1}",
                            "section": bar(n),
                            "profile": [[x0, V_BOTTOM - v_c, sgn * t],
                                        [x0 + L, V_BOTTOM - v_c, sgn * t]],
                            "extent": [x0, x0 + L], "jack": jack, "friction": friction})
    for k in range(1, spans):
        xp = k * L
        for t, half in DECK:
            for sgn in (-1, 1):
                # A 18-64 m duct cannot spread a 6 mm anchor set by
                # reflection, so these carry a stated stress: the jacking
                # stress less the seat over the whole length and the mean
                # friction (see the case file's header).
                sigma = jack - 1.95e11 * 0.006 / (2 * half) - jack * 0.0015 * (2 * half) / 4.0
                out.append({"name": f"T{t:g}{'L' if sgn < 0 else 'R'}_p{k}",
                            "section": bar(DECK_STRANDS),
                            "profile": [[xp - half, V_TOP - v_c, sgn * t],
                                        [xp + half, V_TOP - v_c, sgn * t]],
                            "extent": [xp - half, xp + half],
                            "prestress": round(sigma, -5)})
    return out, bars


def rebar_layout(v_c):
    """Phi 16 grade II bars at 15 cm along every face (Chen, Fig. 6.25c),
    lumped: a `bars` entry per face segment with `count` bars at its
    middle. 389 bars in 48 rows, 78 200 mm2 -- 0.8% of the section."""
    rows = []

    def face(v, t0, t1, cover_in, n_lump):
        length = abs(t1 - t0)
        n_bars = max(1, int(round(length / 0.15)))
        per = [n_bars // n_lump + (1 if i < n_bars % n_lump else 0) for i in range(n_lump)]
        for i in range(n_lump):
            t = t0 + (t1 - t0) * (i + 0.5) / n_lump
            rows.append({"y": round(v - v_c, 4), "z": round(t, 4), "count": per[i]})

    c = 0.05
    face(H - c, -CANT_TIP, CANT_TIP, 0, 10)                    # deck top
    face(H - T_SLAB + c, -t_in(H - T_SLAB), t_in(H - T_SLAB), 0, 6)   # top slab soffit
    face(H - CANT_ROOT_T + c, -CANT_TIP, -t_out(H), 0, 2)        # cantilever undersides
    face(H - CANT_ROOT_T + c, t_out(H), CANT_TIP, 0, 2)
    face(c, -T_SOFFIT, T_SOFFIT, 0, 6)                          # bottom slab, bottom layer
    face(B_SLAB - c, -t_in(B_SLAB), t_in(B_SLAB), 0, 6)          # bottom slab, top layer
    # webs: two faces each, bars every 15 cm up the 4 m, lumped in 4 along the height
    for sgn in (-1, 1):
        n_bars = int(round(H / 0.15))
        for i in range(4):
            v = H * (i + 0.5) / 4
            for off in (c, 0.50 - c):
                rows.append({"y": round(v - v_c, 4), "z": round(sgn * (t_in(v) + off), 4),
                             "count": n_bars // 4 + (1 if i < n_bars % 4 else 0)})
    return rows


# ---------------------------------------------------------------- loads
Q_D1, Q_D2 = 306e3, 52e3       # N/m, girder weight; pavement and barriers
F_L, Q_L = 818e3, 25e3         # N, N/m: JTG D60-2004 lane load as Chen applies it


# ---------------------------------------------------------------- meshes
def write_line_mesh(path, spans, n_per_span, L=70.0, points=None,
                    pier_height=None, n_pier=6, bearing=None):
    """A gmsh 2.2 line mesh along x from 0 to spans*L, one physical line
    `girder`, physical points for the supports and the probes.

    WITH `pier_height`, EACH INTERIOR SUPPORT GROWS A PIER:
    a chain of `n_pier` elements running DOWN in z from the girder node to
    z = -pier_height, in a physical line group `piers`, with the bottom nodes
    in a physical point group `pier_base`. The girder part of the mesh is
    UNCHANGED, node for node and element for element, so a case that does not
    ask for piers gets the file it always got.

    WITHOUT `bearing` THE PIER TOP *IS* THE GIRDER NODE -- they share all six
    DOF, which is a monolithic connection, and v0.4 measured what that
    costs: the girder shortens under prestress, the piers are dragged with it
    and lean, and the moment they hand back DOUBLES the middle-span dead-load
    deflection.

    WITH `bearing = h` THE TWO ARE DIFFERENT NODES, h metres apart in z, joined
    by one element in the physical line group `bearings`. What
    that element IS -- rigid, a fixed pot bearing, a sliding one -- is the
    connector's six stiffnesses in the case file, not a property of the mesh.
    The link has to have a LENGTH: `connector2` builds its frame from the
    chord, so two coincident nodes have no frame. h is a modelling length, not
    the bearing's height; the pier still runs to z = -pier_height, so the base
    elevation and the total height are what they were.
    """
    n = spans * n_per_span
    x = np.linspace(0.0, spans * L, n + 1)
    pts = {"left": 0, "right": n}
    for k in range(1, spans):
        pts[f"pier_{k}"] = k * n_per_span
    for name, xx in (points or {}).items():
        i = int(round(xx / (spans * L) * n))
        assert abs(x[i] - xx) < 1e-9, f"{name}: x = {xx} is not a node with {n_per_span} per span"
        pts[name] = i
    # the pier chains, appended AFTER every girder node so the girder's own
    # numbering does not move
    nodes = [(xx, 0.0) for xx in x]                       # (x, z)
    pier_elems, base_nodes, bearing_elems, top_nodes = [], [], [], []
    if pier_height:
        h = float(bearing or 0.0)
        for k in range(1, spans):
            girder_node = pts[f"pier_{k}"]
            if h > 0.0:
                # the pier top is its OWN node, h below the girder node, and
                # one `bearings` element joins the two
                nodes.append((x[girder_node], -h))
                top = len(nodes) - 1
                bearing_elems.append((girder_node, top))
                top_nodes.append(top)
            else:
                top = girder_node
            prev = top
            for j in range(1, n_pier + 1):
                nodes.append((x[girder_node], -h - (pier_height - h) * j / n_pier))
                cur = len(nodes) - 1
                pier_elems.append((prev, cur))
                prev = cur
            base_nodes.append(prev)

    names = list(pts) + (["pier_base"] if base_nodes else []) + ["girder"]
    if pier_elems:
        names.append("pier_shafts")
    if bearing_elems:
        names.append("bearings")
    n_point_groups = len(pts) + (1 if base_nodes else 0)
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names))]
    for k, nm in enumerate(names[:n_point_groups]):
        out.append(f'0 {k + 1} "{nm}"')
    for k, nm in enumerate(names[n_point_groups:]):
        out.append(f'1 {n_point_groups + k + 1} "{nm}"')
    g_girder = n_point_groups + 1
    g_piers = n_point_groups + 2
    g_bearings = n_point_groups + 3
    n_pt_elems = len(pts) + len(base_nodes)
    out += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    out += [f"{k + 1} {xx:.12g} 0 {zz:.12g}" for k, (xx, zz) in enumerate(nodes)]
    out += ["$EndNodes", "$Elements",
            str(n_pt_elems + n + len(pier_elems) + len(bearing_elems))]
    eid = 1
    for k, nm in enumerate(names[:len(pts)]):
        out.append(f"{eid} 15 2 {k + 1} {k + 1} {pts[nm] + 1}"); eid += 1
    if base_nodes:
        g = len(pts) + 1
        for nd in base_nodes:
            out.append(f"{eid} 15 2 {g} {g} {nd + 1}"); eid += 1
    for k in range(n):
        out.append(f"{eid} 1 2 {g_girder} {g_girder} {k + 1} {k + 2}"); eid += 1
    for a, b in pier_elems:
        out.append(f"{eid} 1 2 {g_piers} {g_piers} {a + 1} {b + 1}"); eid += 1
    for a, b in bearing_elems:
        out.append(f"{eid} 1 2 {g_bearings} {g_bearings} {a + 1} {b + 1}"); eid += 1
    out += ["$EndElements", ""]
    pathlib.Path(path).write_text("\n".join(out), encoding="utf-8")
    return pts


# --------------------------------------------------------------------------- #
#  What a bearing is worth
#
#  A plane-frame price for the pier-girder joint, on the SAME direct-stiffness
#  frame `generate_confederation.py` uses and `tests/test_confederation.py`
#  checks against closed forms -- plus one thing that file does not have: a
#  spring block between two nodes, which is all a bearing is in a linear model.
#
#  It exists because v0.4 measured a cost and could not price the fix:
#  `case_24d`'s piers are MONOLITHIC with the girder, which doubles the
#  middle-span dead-load deflection, and a bearing is a `connector2` that the
#  fibre-beam solver cannot host. This is the elastic twin that says how much
#  the fix is worth before the solver is taught to hold one.
# --------------------------------------------------------------------------- #
#: THE PIER, as three independent papers state it rather than as v0.4
#: back-calculated it.
#:
#:   黄燕庆《水中低墩区引桥总体设计》§3 -- THIS unit's own paper:
#:     "墩身采用钢筋混凝土预制墩身，矩形空心断面，横向宽 6.25 m，纵向墩底宽
#:      2.6 m，墩顶宽 4.0 m，壁厚 0.5 m。最大预制高度为 17.4 m，最大吊装重量
#:      为 458 t。"
#:   赵毅《高墩区引桥下部结构及基础设计》§3 and 罗园辉《墩身的施工》§1 give the
#:   same 6.25 x (2.6 -> 4.0), wall 0.5, for the high-pier zone.
#:
#: v0.4 had no such statement and derived a SOLID 6.25 x 1.80 from the piers'
#: weights. That was wrong in the direction that mattered: the real section's
#: second moment IN THE PLANE OF THE FRAME is 4.4x larger, and v0.4's own
#: sensitivity sweep (1.62 to 2.08 m solid, I = 2.2 to 4.7 m4) does not contain
#: the true 13.4 m4 -- so "the assumed thickness is worth 10%" was measured on
#: a bracket that excluded the answer.
#: TRANSVERSELY it was right by luck, to 8%.
PIER_B = 6.25                  #: across the bridge, constant
PIER_D_BASE, PIER_D_TOP = 2.60, 4.00      #: along the bridge, base and top
PIER_WALL = 0.50
PIER_H = 12.0                  #: this case's representative height (7.4 - 17.4 m)


def pier_properties(depth, b=PIER_B, t=PIER_WALL):
    """(A, I_in_plane, I_transverse, J) of the hollow pier at a section whose
    longitudinal depth is `depth`. `J` is Bredt's closed-cell formula, which
    is what a thin-walled box has and a solid rectangle does not."""
    bi, di = b - 2 * t, depth - 2 * t
    A = b * depth - bi * di
    I_long = (b * depth ** 3 - bi * di ** 3) / 12      # bending along the bridge
    I_tran = (depth * b ** 3 - di * bi ** 3) / 12      # bending across it
    a0 = (b - t) * (depth - t)                         # mid-line enclosed area
    J = 4 * a0 ** 2 / (2 * ((b - t) + (depth - t)) / t)
    return A, I_long, I_tran, J


def pier_depth(height_above_base, total=None):
    """The linear taper 2.60 -> 4.00. 赵毅 says the top 6 m of a HIGH pier is a
    curve and the rest linear; this unit's own paper gives only the two end
    widths, so the taper is linear here and that is a documented reading, not
    a measurement."""
    h = PIER_H - 0.5 if total is None else total        # the shaft, bearing aside
    f = min(max(height_above_base / h, 0.0), 1.0)
    return PIER_D_BASE + (PIER_D_TOP - PIER_D_BASE) * f


#: the elastic twin's numbers, all read out of case_24d's own assembled model
E_TWIN = 3.4105e+10            #: concrete02's initial tangent, 2 fpc / epsc0
A_GIRDER, I_GIRDER = 9.8715, 22.2393
#: the twin is prismatic, so it takes the pier at mid-height
A_PIER, I_PIER = pier_properties(pier_depth((PIER_H - 0.5) / 2))[:2]
H_PIER_TWIN, N_PIER_TWIN, H_BEARING = 12.0, 6, 0.5
#: the tendons as their own axial free strain: case_24d's prestrain array,
#: element by element, length-averaged. 52.8 mm of free shortening over 210 m.
EPS_PRESTRESS = -2.5152e-4
#: a pot bearing, [longitudinal, vertical, rotational]. NOT published for
#: these piers -- swept below, and the sweep is the honest reading.
POT_FIXED = dict(kx=1.0e+9, kz=5.0e+10, kr=1.0e+7)
POT_SLIDING = dict(kx=7.1e+7, kz=5.0e+10, kr=1.0e+7)
RIGID_LINK = dict(kx=1.0e+14, kz=1.0e+14, kr=1.0e+14)


#: the PRIMARY prestress action of case_24f, element by element along the
#: girder (42 x 5 m, x increasing), read straight off the fibre model:
#:
#:     prep = prepare_case("cases/case_24f_hzb_girder_bearings.yaml")
#:     s = FiberBeamSolver(prep.case.model_dump(mode="python"), prep.mesh,
#:                         prep.material)
#:     _f, act = s.internal_force()          # at q = 0: the prestress alone
#:     N, M = act[girder, 0] / 1e6, act[girder, 5] / 1e6
#:
#: so it carries the real friction and anchor-set losses and the real duct
#: areas. The axial column reduces to EPS_PRESTRESS: -N/(E A) length-averaged
#: is -2.519e-4 against the -2.5152e-4 measured independently for case_24e.
N_PRESTRESS = np.array([                                              # MN
    68.110, 68.668, 69.345, 70.145, 70.817, 71.023, 70.493, 70.493,
    79.133, 78.927, 90.430, 101.746, 112.982, 112.424, 112.424, 112.982,
    101.746, 90.430, 78.927, 79.133, 70.493, 70.493, 79.133, 78.927,
    90.430, 101.746, 112.982, 112.424, 112.424, 112.982, 101.746, 90.430,
    78.927, 79.133, 70.493, 70.493, 71.023, 70.817, 70.145, 69.345,
    68.668, 68.110]) * 1e+6
M_PRESTRESS = np.array([                                              # MN.m
    96.476, 113.090, 129.958, 142.371, 150.734, 152.399, 151.262, 151.262,
    141.114, 139.449, 114.144, 84.873, 51.427, 34.813, 34.813, 51.427,
    84.873, 114.144, 139.449, 141.114, 151.262, 151.262, 141.114, 139.449,
    114.144, 84.873, 51.427, 34.813, 34.813, 51.427, 84.873, 114.144,
    139.449, 141.114, 151.262, 151.262, 152.399, 150.734, 142.371, 129.958,
    113.090, 96.476]) * 1e+6


def _twin(bearings=None, eps=0.0, kappa=0.0, n_per_span=14, spans=3, L=70.0):
    """(K, f, node map) for the three-span unit with piers, optionally on
    bearings. `bearings` is None (monolithic) or one spec per pier.

    `eps` and `kappa` are the girder's FREE axial strain and FREE curvature,
    a scalar or one value per girder element. `kappa` is what a draped tendon
    does and what `EPS_PRESTRESS` alone leaves out (see `camber_split`).
    """
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).parent))
    from generate_confederation import Frame
    fr = Frame()
    xs = np.linspace(0.0, spans * L, spans * n_per_span + 1)
    gn = [fr.node(x, 0.0) for x in xs]
    e_arr = np.broadcast_to(np.asarray(eps, float), (len(xs) - 1,))
    k_arr = np.broadcast_to(np.asarray(kappa, float), (len(xs) - 1,))
    for k in range(len(xs) - 1):
        fr.element(gn[k], gn[k + 1], E_TWIN * A_GIRDER, E_TWIN * I_GIRDER,
                   dth=(float(e_arr[k]), float(k_arr[k])))
    springs, bases = [], []
    for p, spec in zip(range(1, spans), bearings or [None] * (spans - 1)):
        gi = gn[p * n_per_span]
        z0, prev = 0.0, gi
        if spec is not None:
            top = fr.node(xs[p * n_per_span], -H_BEARING)
            springs.append((gi, top, spec)); z0, prev = -H_BEARING, top
        nz = np.linspace(z0, -H_PIER_TWIN, N_PIER_TWIN + 1)
        for k in range(N_PIER_TWIN):
            nxt = fr.node(xs[p * n_per_span], nz[k + 1])
            fr.element(prev, nxt, E_TWIN * A_PIER, E_TWIN * I_PIER)
            prev = nxt
        bases.append(prev)
    n = 3 * len(fr.nodes)
    K, f = np.zeros((n, n)), np.zeros(n)
    q = Q_D1 + Q_D2
    for m, e in enumerate(fr.elems):
        Lc, c, s = fr._geom(e)
        ke, T = fr._ke(Lc, e[2], e[3]), fr._T(c, s)
        d = np.array([-e[4][0] * Lc / 2, 0.0, e[4][1] * Lc / 2,
                      e[4][0] * Lc / 2, 0.0, -e[4][1] * Lc / 2])
        ids = np.array([3*e[0], 3*e[0]+1, 3*e[0]+2, 3*e[1], 3*e[1]+1, 3*e[1]+2])
        K[np.ix_(ids, ids)] += T.T @ ke @ T
        f[ids] += T.T @ (ke @ d)
        if m < len(xs) - 1:                      # the girder carries the load
            f[ids] += np.array([0.0, -q*Lc/2, -q*Lc*Lc/12,
                                0.0, -q*Lc/2, q*Lc*Lc/12])
    for a, b, spec in springs:
        for off, k in ((0, spec["kx"]), (1, spec["kz"]), (2, spec["kr"])):
            ia, ib = 3*a + off, 3*b + off
            K[ia, ia] += k; K[ib, ib] += k; K[ia, ib] -= k; K[ib, ia] -= k
    fixed = [3*gn[0] + 1, 3*gn[-1] + 1]          # abutments: vertical rollers
    for b in bases:
        fixed += [3*b, 3*b + 1, 3*b + 2]
    return K, f, fixed, gn, n_per_span, spans, L


def bearing_price(bearings=None, eps=0.0, **kw):
    """Middle- and side-span deflections (mm, DOWN NEGATIVE) and the girder's
    longitudinal movement at each pier."""
    K, f, fixed, gn, nps, spans, L = _twin(bearings, eps, **kw)
    free = np.setdiff1d(np.arange(len(f)), np.asarray(sorted(set(fixed)), int))
    u = np.zeros(len(f))
    u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
    at = lambda x: gn[int(round(x / (spans * L) * (spans * nps)))]   # noqa: E731
    return dict(mid=1e+3*u[3*at(1.5*L) + 1], side=1e+3*u[3*at(0.5*L) + 1],
                pier1=1e+3*u[3*gn[nps]], pier2=1e+3*u[3*gn[2*nps]])


def camber_split(bearings=None):
    """WHERE THE PRESTRESS CAMBER GOES, and why case_24e can miss it entirely
    in the middle span and be out by a factor of four in the side one
    (written after the v0.4 contract got the reason wrong).

    `EPS_PRESTRESS` is the tendons reduced to their AXIAL free strain, so the
    twin that case_24e prices the bearing with has no camber anywhere. Feed
    the same twin the free CURVATURE as well -- `M_PRESTRESS / EI`, the real
    primary moment of case_24f's 98 tendons -- and the camber appears:

        camber = (dead + axial + curvature) - (dead + axial)

    THE POINT IS THAT IT IS NOT SHARED OUT IN THE RATIO OF THE SPANS.
    M_PRESTRESS is very nearly the same in all three spans -- 151 MN.m at
    every mid-span, because the drape is the same in each -- and the dead-load
    deflections sit in the classical 10.4 : 1 ratio. The camber does not: it
    is ~100 : 1. A free curvature deflects a member only where its ends can
    ROTATE, and a constant free curvature between two rotationally fixed ends
    produces no deflection at all (the fixed-end moments cancel it exactly).
    The abutment is a real hinge; the middle span's two ends are held by the
    side spans. So the camber lands almost entirely in the side spans, and a
    model without it agrees on the middle span by construction rather than by
    luck.
    """
    kappa = M_PRESTRESS / (E_TWIN * I_GIRDER)
    eps = -N_PRESTRESS / (E_TWIN * A_GIRDER)
    bearings = bearings if bearings is not None else [POT_FIXED, POT_SLIDING]
    dead = bearing_price(bearings, 0.0)
    axial = bearing_price(bearings, eps)
    full = bearing_price(bearings, eps, kappa=kappa)
    return {"dead": dead, "axial": axial, "full": full,
            "camber_mid": full["mid"] - axial["mid"],
            "camber_side": full["side"] - axial["side"]}


def curvature_needs_a_free_end(fixed_ends, kappa=199.43e-6, L=70.0, n=28):
    """One 70 m span, no load, a uniform free curvature: `kappa L^2 / 8` with
    the ends free to rotate and EXACTLY ZERO with them fixed. The mechanism
    behind `camber_split`, on its own."""
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).parent))
    from generate_confederation import Frame
    fr = Frame()
    g = [fr.node(x, 0.0) for x in np.linspace(0.0, L, n + 1)]
    for i in range(n):
        fr.element(g[i], g[i + 1], E_TWIN * A_GIRDER, E_TWIN * I_GIRDER,
                   dth=(0.0, kappa))
    fixed = [3*g[0], 3*g[0] + 1, 3*g[-1] + 1]
    if fixed_ends:
        fixed += [3*g[0] + 2, 3*g[-1] + 2]
    return 1e+3 * fr.solve({}, fixed)[3 * g[n // 2] + 1]


def _four_rollers(n_per_span=14, spans=3, L=70.0):
    """The same girder on four vertical rollers and no piers: the check that
    the frame itself is right before anything is priced with it."""
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).parent))
    from generate_confederation import Frame
    fr = Frame()
    xs = np.linspace(0.0, spans * L, spans * n_per_span + 1)
    gn = [fr.node(x, 0.0) for x in xs]
    for k in range(len(xs) - 1):
        fr.element(gn[k], gn[k + 1], E_TWIN * A_GIRDER, E_TWIN * I_GIRDER)
    q = Q_D1 + Q_D2
    loads = {}
    for e in fr.elems:
        Lc, _, _ = fr._geom(e)
        i, j = e[0], e[1]
        for d, v in ((3*i+1, -q*Lc/2), (3*j+1, -q*Lc/2),
                     (3*i+2, -q*Lc*Lc/12), (3*j+2, q*Lc*Lc/12)):
            loads[d] = loads.get(d, 0.0) + v
    fixed = [3*gn[0]+1, 3*gn[-1]+1, 3*gn[n_per_span]+1, 3*gn[2*n_per_span]+1, 3*gn[0]]
    u = fr.solve(loads, fixed)
    return 1e+3 * u[3 * gn[int(round(1.5 * n_per_span))] + 1]


def _four_rollers_exact(L=70.0):
    """5 q L^4 / 384 EI less the hogging the pier moment takes off it."""
    q = Q_D1 + Q_D2
    m = continuous_three_span_moments(L=L)
    return 1e+3 * (5 * q * L ** 4 / (384 * E_TWIN * I_GIRDER)
                   - abs(m["uniform_all"][1]) * q * L ** 2 / (8 * E_TWIN * I_GIRDER)) * -1.0


# ---------------------------------------------------------------- checks
def euler_modes(E, I, rho, A, L, n=3):
    return [(k * np.pi) ** 2 / (2 * np.pi * L ** 2) * np.sqrt(E * I / (rho * A)) for k in range(1, n + 1)]


def continuous_three_span_moments(L=70.0, n=60):
    """Elastic bending moments of a three-span continuous beam of equal
    spans, from a Hermite beam solve (EI cancels): {load: (M at the middle
    span's midspan, M at the piers)} per unit of q over the whole length,
    P at the middle midspan, and q over the middle span only."""
    ne = 3 * n; h = L / n
    K = np.zeros((2 * (ne + 1), 2 * (ne + 1)))
    ke = np.array([[12, 6 * h, -12, 6 * h], [6 * h, 4 * h * h, -6 * h, 2 * h * h],
                   [-12, -6 * h, 12, -6 * h], [6 * h, 2 * h * h, -6 * h, 4 * h * h]]) / h ** 3
    for e in range(ne):
        d = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(d, d)] += ke
    supports = [0, n, 2 * n, 3 * n]
    free = np.array([i for i in range(2 * (ne + 1)) if i // 2 not in supports or i % 2 == 1])

    def solve(F):
        u = np.zeros(2 * (ne + 1)); u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
        # moment at node i from the element to its right: EI * v'' (Hermite)
        def M_at(i):
            e = min(i, ne - 1); d = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            xi = 0.0 if i < ne else 1.0
            B = np.array([(-6 + 12 * xi) / h ** 2, (-4 + 6 * xi) / h, (6 - 12 * xi) / h ** 2, (-2 + 6 * xi) / h])
            return -float(B @ u[d])
        return M_at(3 * n // 2), M_at(n)

    def uniform(e0, e1):
        F = np.zeros(2 * (ne + 1))
        for e in range(e0, e1):
            F[[2 * e, 2 * e + 2]] += h / 2; F[2 * e + 1] += h * h / 12; F[2 * e + 3] -= h * h / 12
        return F

    Fp = np.zeros(2 * (ne + 1)); Fp[2 * (3 * n // 2)] = 1.0
    return {"uniform_all": solve(uniform(0, ne)), "point_mid": solve(Fp),
            "uniform_mid": solve(uniform(n, 2 * n))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "assets"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--n-per-span", type=int, default=14)
    ap.add_argument("--pier-height", type=float, default=12.0,
                    help="representative pier height; the 474 precast piers "
                         "run 7.372 to 17.383 m")
    ap.add_argument("--n-pier", type=int, default=6)
    ap.add_argument("--bearing", type=float, default=0.5,
                    help="length of the girder-to-pier-top link element (m); "
                         "a modelling length, not the bearing's height")
    a = ap.parse_args(argv)
    props = section_properties()
    if a.check:
        E, rho = 3.5e10, 2500.0
        print(f"A = {props['A']:.4f} m2  v_c = {props['v_c']:.4f} m  I_y = {props['I_y']:.4f} m4  "
              f"I_z = {props['I_z']:.3f} m4  J = {torsion_constant():.3f} m4")
        print(f"weight {props['A'] * rho * 9.80665 / 1e3:.1f} kN/m (girder 306 kN/m: end thickening, diaphragms)")
        f = euler_modes(E, props["I_y"], rho, props["A"], 70.0)
        print("Euler simply supported vertical modes: " + ", ".join(f"{x:.4f}" for x in f) + " Hz")
        f = euler_modes(E, props["I_z"], rho, props["A"], 70.0, 1)
        print(f"first lateral: {f[0]:.4f} Hz; Xu's ANSYS 1.8498, 4.7565, 6.7249")
        tend, bars = tendon_layout(3, v_c=props["v_c"])
        A_span = sum(bars[t["section"]]["area"] for t in tend if t["section"].startswith(("web", "bottom"))) / 3
        A_cap = sum(bars[t["section"]]["area"] for t in tend if t["section"].startswith("cap")) / 2
        print(f"tendons per span {A_span * 1e6:.0f} mm2 (jack {SIGMA_K * A_span / 1e6:.1f} MN), per pier cap "
              f"{A_cap * 1e6:.0f} mm2 ({SIGMA_K * A_cap / 1e6:.1f} MN)")
        print("\nWHAT A BEARING IS WORTH (case_24e). Middle-span")
        print("deflection in mm, DOWN NEGATIVE; `short` adds the prestress as its own")
        print("axial free strain (%.4e, i.e. %.1f mm over 210 m) and D is what that"
              % (EPS_PRESTRESS, abs(EPS_PRESTRESS) * 210e+3))
        print("costs -- the quantity v0.4 measured at -9.88 mm on the fibre model.")
        print("  %-24s %8s %8s %9s  %s" % ("", "dead", "+short", "D", "girder at the piers"))
        for nm, b in (("monolithic (case_24d)", None),
                      ("rigid links", [RIGID_LINK] * 2),
                      ("both FIXED pots", [POT_FIXED] * 2),
                      ("fixed + sliding pots", [POT_FIXED, POT_SLIDING])):
            a = bearing_price(b, 0.0); c = bearing_price(b, EPS_PRESTRESS)
            print("  %-24s %8.3f %8.3f %+9.3f  %+6.2f %+6.2f mm"
                  % (nm, a["mid"], c["mid"], c["mid"] - a["mid"], c["pier1"], c["pier2"]))
        print("  the frame itself is checked first: on four vertical rollers and no")
        print("  piers the middle span reads %.3f mm against the superposition closed"
              % _four_rollers())
        print("  form 5 q L^4 / 384 EI - M_pier L^2 / 8 EI = %.3f" % _four_rollers_exact())

        print("\n  WHICH PROPERTY DOES IT, and it is not the one the name suggests.")
        print("  Sweeping one component at a time with the other two rigid:")
        for lbl, key, vals in (("rotational", "kr", (1e14, 1e11, 1e10, 1e9, 1e7)),
                               ("longitudinal", "kx", (1e14, 1e10, 1e9, 1e8, 1e5))):
            print("    %-13s" % lbl, end="")
            for v in vals:
                b = dict(RIGID_LINK); b[key] = v
                a = bearing_price([b] * 2, 0.0); c = bearing_price([b] * 2, EPS_PRESTRESS)
                print("  %.0e:%+.2f" % (v, c["mid"] - a["mid"]), end="")
            print()
        print("    EITHER release kills the penalty -- it is a MOMENT and THRUST")
        print("    transfer, not a longitudinal-restraint effect -- and a real pot has")
        print("    both, so the penalty is gone whichever pier is fixed.")
        print("\n  AND WHAT REPLACES IT AS THE SENSITIVE ASSUMPTION: the bearing's own")
        print("  VERTICAL stiffness, which does nothing to D and everything to the")
        print("  dead-load deflection.")
        print("    %-10s" % "kz", end="")
        for kz in (1e14, 1e11, 5e10, 1e10, 5e9, 2e9):
            b = dict(POT_FIXED); b["kz"] = kz
            print("  %.0e:%.2f" % (kz, bearing_price([b] * 2, 0.0)["mid"]), end="")
        print("\n    case_24d's own assumed number, the pier wall, was worth 10%;")
        print("    this one is worth 230% over a plausible range.")

        print("\nWHERE THE PRESTRESS CAMBER GOES. This twin has")
        print("  none: EPS_PRESTRESS is the tendons reduced to their axial free")
        print("  strain. Give it the free CURVATURE too -- M_PRESTRESS / EI, the")
        print("  real primary moment of case_24f's 98 tendons -- and it appears.")
        c = camber_split()
        print("    %-34s %9s %9s" % ("", "middle", "side"))
        for nm, key in (("dead load only", "dead"),
                        ("+ axial free strain (case_24e)", "axial"),
                        ("+ free curvature too", "full")):
            print("    %-34s %9.3f %9.3f" % (nm, c[key]["mid"], c[key]["side"]))
        print("    %-34s %+9.3f %+9.3f" % ("camber", c["camber_mid"], c["camber_side"]))
        print("    case_24f, the fibre model, reads    %9.3f %9.3f" % (-7.678, -20.990))
        print("    so the reconstruction is within 10% on both spans, and the")
        print("    camber is split %.0f : 1 while the dead load is split %.1f : 1."
              % (c["camber_side"] / c["camber_mid"], c["dead"]["side"] / c["dead"]["mid"]))
        print("    WHY: a free curvature deflects a member only where its ends can")
        print("    ROTATE. One 70 m span, uniform free curvature, no load:")
        print("      ends free to rotate %7.2f mm (kappa L^2/8 = %.2f), ends fixed %.2f mm"
              % (curvature_needs_a_free_end(False), 199.43e-6 * 70.0 ** 2 / 8 * 1e3,
                 curvature_needs_a_free_end(True)))
        print("    The abutment is a hinge; the middle span's ends are held by the")
        print("    side spans. So a twin with no camber agrees on the middle span")
        print("    BY CONSTRUCTION, and cannot agree on the side span at all.")

        m = continuous_three_span_moments()
        q = Q_D1 + Q_D2
        print(f"dead load continuous: M_mid(middle) = {m['uniform_all'][0] * q / 1e6:.1f} MN.m, "
              f"M_pier = {m['uniform_all'][1] * q / 1e6:.1f} MN.m; simply supported {q * 70 ** 2 / 8 / 1e6:.1f}")
        print(f"live load case 1 per unit k: M_mid = {(m['point_mid'][0] * F_L + m['uniform_mid'][0] * Q_L) / 1e6:.2f} MN.m, "
              f"M_pier = {(m['point_mid'][1] * F_L + m['uniform_mid'][1] * Q_L) / 1e6:.2f}; "
              f"simply supported {(F_L * 70 / 4 + Q_L * 70 ** 2 / 8) / 1e6:.2f}")
        return
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p1 = write_line_mesh(out / f"hzb_girder_1span_{a.n_per_span}.msh", 1, a.n_per_span,
                         points={"mid": 35.0, "x20": 20.0})
    p3 = write_line_mesh(out / f"hzb_girder_3span_{a.n_per_span}.msh", 3, a.n_per_span,
                         points={"mid_side": 35.0, "mid_middle": 105.0, "side_40m": 40.0})
    # the same three-span unit WITH PIERS. Representative
    # height 12 m: the 474 precast piers run 7.372 to 17.383 m
    # (章汉斌), and this is the middle of that range.
    p3p = write_line_mesh(out / f"hzb_girder_3span_{a.n_per_span}_piers.msh",
                          3, a.n_per_span,
                          points={"mid_side": 35.0, "mid_middle": 105.0,
                                  "side_40m": 40.0},
                          pier_height=a.pier_height, n_pier=a.n_pier)
    # ... and the same again with the pier top on its OWN node, so a bearing
    # element can sit between it and the girder. 0.5 m is a
    # modelling length for the link, not the bearing's height.
    p3b = write_line_mesh(out / f"hzb_girder_3span_{a.n_per_span}_bearings.msh",
                          3, a.n_per_span,
                          points={"mid_side": 35.0, "mid_middle": 105.0,
                                  "side_40m": 40.0},
                          pier_height=a.pier_height, n_pier=a.n_pier,
                          bearing=a.bearing)
    print(p1, p3, p3p, p3b)


if __name__ == "__main__":
    main()
