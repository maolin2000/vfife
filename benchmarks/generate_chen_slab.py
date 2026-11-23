"""Chen's 13 m pretensioned hollow-core slab (cases/case_25): section,
strands, mesh, and the hand checks behind the case files.

SOURCE
  Chen Hengzhi, "Computational analysis of large-span prestressed concrete
  box girder structures", PhD thesis, Zhejiang University, 2005, §6.4
  (pp. 135-139): two precast pretensioned hollow-core slabs, 12.96 m long
  on a 12.60 m span, loaded to failure by two synchronised 100 t jacks.
      1#  loads 4.00 m apart (2.00 m each side of midspan)
          failed at P = 245.6 kN per point, 155 mm at midspan (L/81)
      2#  loads 6.00 m apart
          failed at P = 315.0 kN per point, 158 mm (L/80)
  The measured load-deflection points are digitised in
  verification/Chen2005_hollow_slab_{1,2}.txt.

SECTION (cm in the thesis, metres here; v up from the soffit, t across,
t = 0 on the centreline). Fig. 6.16: 99 wide at the soffit and 85 at the
deck, 55 deep, two circular voids of R = 19 centred 29 above the soffit
and 22.5 either side of the centreline -- so a 10 cm bottom slab, a 7 cm
top slab, 8 cm outer webs and a 7 cm middle web. The bottom corners are
chamfered 4.5 x 4.5; the shoulders run from the full width at v = 39.3 to
the 85 cm deck. That last height is the ONE dimension not in the figure:
it is set so that the net area is 0.3046 m2, which is what Chen's own
model says (3046 concrete fibres of 1e-4 m2 each, p. 137).

STRANDS (Fig. 6.16, the `预应力钢筋截面布置` panel). Nine 15.24 mm
strands 5 cm above the soffit, at t = -42, -35, -14, -7, 0, +7, +14, +35,
+42 cm, each pulled to N_k = 195 kN (1393 MPa on 140 mm2). They are
PRETENSIONED WITH DIFFERENT BONDED LENGTHS -- the table gives four
`有效长度`, 1296 / 1175 / 1015 / 835 cm, and the strip above it says which
strand is which:
      t   -42  -35  -14   -7    0   +7  +14  +35  +42
    type    1    3    4    2    1    2    4    3    1
    length 1296 1015  835 1175 1296 1175  835 1015 1296
so five of the nine stop short of the ends. That is what a tendon's
`extent` is for.

REBAR (Fig. 6.16, the `普通钢筋截面布置` panel): 5 phi 12 (grade II) in
the deck at 4@20 cm, 4 phi 12 in the soffit, 2 phi 8 (grade I) on the
webs -- eleven bars, which is Chen's count.

Run `python benchmarks/generate_chen_slab.py --check` for the section
properties, the cracking load and the ultimate hand check.
"""
import argparse
import pathlib

import numpy as np

# ---------------------------------------------------------------- section
W_SOFFIT, W_DECK, DEPTH = 0.99, 0.85, 0.55
CHAMFER = 0.045                 # bottom corners
V_SHOULDER = 0.393              # where the sides start running in to the deck
VOID_R, VOID_V, VOID_T = 0.19, 0.29, 0.225
STRAND_V, STRAND_T = 0.05, [-0.42, -0.35, -0.14, -0.07, 0.0, 0.07, 0.14, 0.35, 0.42]
#: bonded length of each strand, in the order of STRAND_T (Fig. 6.16)
STRAND_LENGTH = [12.96, 10.15, 8.35, 11.75, 12.96, 11.75, 8.35, 10.15, 12.96]
STRAND_AREA, N_K = 140e-6, 195e3
LENGTH, SPAN, OVERHANG = 12.96, 12.60, 0.18


def section_patches(cell=0.02):
    """The concrete as three quadrilateral patches with the two voids as
    `holes`: a bottom band up to the chamfers, the full-width middle, and
    the shoulders up to the deck. `cell` is the target fibre size."""
    hs, hd = W_SOFFIT / 2, W_DECK / 2
    nz = int(round(W_SOFFIT / cell))
    out = [
        {"corners": [[0.0, -(hs - CHAMFER)], [0.0, hs - CHAMFER], [CHAMFER, hs], [CHAMFER, -hs]],
         "ny": max(1, int(round(CHAMFER / cell))), "nz": nz},
        {"corners": [[CHAMFER, -hs], [CHAMFER, hs], [V_SHOULDER, hs], [V_SHOULDER, -hs]],
         "ny": int(round((V_SHOULDER - CHAMFER) / cell)), "nz": nz},
        {"corners": [[V_SHOULDER, -hs], [V_SHOULDER, hs], [DEPTH, hd], [DEPTH, -hd]],
         "ny": int(round((DEPTH - V_SHOULDER) / cell)), "nz": nz},
    ]
    for p in out:
        p["holes"] = [[VOID_V, -VOID_T, VOID_R], [VOID_V, VOID_T, VOID_R]]
    return out


def case_patches(cell=0.02, v_c=None):
    """`section_patches` shifted so that y = 0 is the centroid, which is
    where the fibre section's axis is."""
    v_c = section_properties()["v_c"] if v_c is None else v_c
    out = []
    for q in section_patches(cell):
        out.append({"corners": [[round(v - v_c, 4), round(t, 4)] for v, t in q["corners"]],
                    "ny": q["ny"], "nz": q["nz"],
                    "holes": [[round(h[0] - v_c, 4), h[1], h[2]] for h in q["holes"]]})
    return out


def section_properties(cell=0.01):
    from vfife.elements.fiber_section import patch_fibers
    v, t, a, _ = patch_fibers(section_patches(cell))
    A = a.sum(); vc = (a * v).sum() / A
    return {"A": float(A), "v_c": float(vc), "I": float((a * (v - vc) ** 2).sum()),
            "n": int(len(a)), "W_bot": float((a * (v - vc) ** 2).sum() / vc),
            "W_top": float((a * (v - vc) ** 2).sum() / (DEPTH - vc))}


def rebars(v_c):
    """The eleven ordinary bars, as `bars` entries in the section axes."""
    phi12, phi8 = "phi12", "phi8"
    rows = [{"y": round(0.501 - v_c, 4), "z": t, "section": phi12} for t in (-0.40, -0.20, 0.0, 0.20, 0.40)]
    rows += [{"y": round(0.049 - v_c, 4), "z": t, "section": phi12} for t in (-0.45, -0.175, 0.175, 0.45)]
    rows += [{"y": round(0.269 - v_c, 4), "z": t, "section": phi8} for t in (-0.46, 0.46)]
    return rows


def tendons(v_c, long_term=0.88):
    """The nine strands, each with its own bonded length centred on the
    beam (`extent`). `long_term` is the flat factor for relaxation,
    shrinkage and creep on the 1393 MPa at the jack; the model works out
    the elastic shortening itself."""
    sigma = round(N_K / STRAND_AREA * long_term, -5)
    out = []
    for t, L in zip(STRAND_T, STRAND_LENGTH):
        a, b = 0.5 * (LENGTH - L), 0.5 * (LENGTH + L)
        out.append({"section": "strand", "profile": [[a, round(STRAND_V - v_c, 4), t],
                                                     [b, round(STRAND_V - v_c, 4), t]],
                    "extent": [round(a, 3), round(b, 3)], "prestress": sigma})
    return out


# ------------------------------------------------------------------- mesh
def write_mesh(path, load_a, target=0.45):
    """A line mesh of the 12.96 m slab with nodes at the bearings, the two
    load points and midspan. `load_a` is each load's distance from
    midspan."""
    mid = LENGTH / 2
    must = [0.0, OVERHANG, mid - load_a, mid, mid + load_a, LENGTH - OVERHANG, LENGTH]
    x = list(must)
    for a, b in zip(must, must[1:]):
        n = max(1, int(round((b - a) / target)))
        x += [a + (b - a) * k / n for k in range(1, n)]
    x = np.array(sorted(set(round(v, 6) for v in x)))
    pts = {"left": 0, "right": len(x) - 1,
           "bearing_left": int(np.argmin(abs(x - OVERHANG))),
           "bearing_right": int(np.argmin(abs(x - (LENGTH - OVERHANG)))),
           "load_left": int(np.argmin(abs(x - (mid - load_a)))),
           "load_right": int(np.argmin(abs(x - (mid + load_a)))),
           "mid": int(np.argmin(abs(x - mid)))}
    names = list(pts) + ["slab"]
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names))]
    out += [f'0 {k + 1} "{nm}"' for k, nm in enumerate(names[:-1])]
    out += [f'1 {len(names)} "slab"', "$EndPhysicalNames", "$Nodes", str(len(x))]
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


# ----------------------------------------------------------------- checks
def hand_checks(fc=28.0e6, ft=2.2e6, E_c=34.3e9, long_term=0.88):
    p = section_properties()
    P = 9 * STRAND_AREA * N_K / STRAND_AREA * long_term
    e = p["v_c"] - STRAND_V
    sigma_bot = P / p["A"] + P * e * p["v_c"] / p["I"]
    w_sw = p["A"] * 2500 * 9.80665
    M_sw = w_sw * SPAN ** 2 / 8
    M_cr = (ft + sigma_bot) * p["I"] / p["v_c"] - M_sw
    T_u = 9 * STRAND_AREA * 1860e6                       # strands at their tensile strength
    a_u = T_u / (0.85 * fc * W_DECK)                     # equivalent stress block in the deck
    M_u = T_u * (p["v_c"] + (DEPTH - p["v_c"]) - STRAND_V - 0.5 * a_u) - 0 * M_sw
    return {"props": p, "P_eff": P, "e": e, "sigma_bot": sigma_bot, "w_sw": w_sw,
            "M_cr": M_cr, "T_u": T_u, "a_u": a_u, "M_u": M_u}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "assets"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        h = hand_checks(); p = h["props"]
        print(f"section: {p['n']} fibres, A = {p['A']:.4f} m2 (Chen's 3046 x 1e-4 = 0.3046), "
              f"centroid {p['v_c']:.4f} m, I = {p['I']:.5f} m4")
        print(f"self weight {h['w_sw'] / 1e3:.2f} kN/m; prestress {h['P_eff'] / 1e3:.0f} kN at e = {h['e']:.3f} m, "
              f"soffit compression {h['sigma_bot'] / 1e6:.2f} MPa")
        for name, A in (("1#", 2.00), ("2#", 3.00)):
            arm = SPAN / 2 - A
            print(f"  {name}: loads {2 * A:.1f} m apart, each over an arm of {arm:.2f} m; "
                  f"cracking at P = {h['M_cr'] / arm / 1e3:6.1f} kN, "
                  f"ultimate at P = {h['M_u'] / arm / 1e3:6.1f} kN "
                  f"(measured {245.6 if name == '1#' else 315.0})")
        return
    out = pathlib.Path(a.out)
    for name, A in (("1", 2.00), ("2", 3.00)):
        n, pts = write_mesh(out / f"chen_slab_{name}.msh", A)
        print(out / f"chen_slab_{name}.msh", n, "elements", pts)


if __name__ == "__main__":
    main()
