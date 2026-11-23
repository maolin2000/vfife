"""The three Confederation Bridge cases against what is published about them.

    python benchmarks/plot_confederation.py [-o results/confederation]

Writes three PNGs:

  <out>_section.png   the reconstructed cross section at the pier and at
                      midspan, and A / centroid / I along the girder with
                      Butcher's published values marked on them
  <out>_trucks.png    case_28: the deflection profile with the trucks at
                      midspan against Ghali's Fig. 17, and the influence lines
                      the 22-station sweep produces
  <out>_thermal.png   case_28b/c: the published temperature profile resolved
                      on the real section, and the deflections against Fig. 21

Published points are READ OFF the figures at 600 dpi and are marked as such;
they are the same numbers the contracts in verification/case_28*.yaml assert.
"""
import argparse
import pathlib
import sys

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: read off Ghali (2000) Fig. 17, trucks at midspan; mm, DOWN positive
FIG17 = {"measured": [(-220, -5.4), (-180, -3.2), (-125, 0.0), (-60, 6.5),
                      (0, 17.1), (60, 6.8), (125, 0.0), (180, -4.0), (220, -7.6)],
         "analysis": [(-220, -4.0), (-180, -3.0), (-125, 0.0), (-60, 8.5),
                      (0, 13.7), (60, 7.0), (125, 0.0), (180, -3.0), (220, -3.5)]}
#: read off Fig. 21, the summer temperature rise; mm, DOWN positive
FIG21 = [(-220, 73.5), (-180, 43.7), (-125, 0.0), (-60, -47.0), (0, -60.6),
         (60, -47.0), (125, 0.0), (180, 43.7), (220, 73.5)]


def _axis(cb, h5):
    """(x of every girder node, the frame index of each output step)."""
    with h5py.File(h5) as f:
        xyz = f["mesh/nodes"][:]
        keys = sorted(f["steps"].keys(), key=int)
        w = np.array([f["steps"][k]["displacement"][:, 2] for k in keys])
    girder = np.abs(xyz[:, 0]) <= 220.0 + 1e-6
    girder &= np.abs(xyz[:, 2] - np.interp(np.abs(xyz[:, 0]), [0, 440], [40, 40])) < 100
    girder = xyz[:, 2] > 20.0                      # the piers run down to -29 m
    order = np.argsort(xyz[girder, 0])
    return xyz[girder, 0][order], w[:, girder][:, order]


def main(argv=None):
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import generate_confederation as cb

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--out", default=str(ROOT / "results/confederation"))
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
    matplotlib.rcParams["axes.grid"] = True
    matplotlib.rcParams["grid.alpha"] = 0.25

    # ------------------------------------------------------------ section
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for depth, colour, label in ((cb.D_PIER, "#1f4e79", "at the pier, 14.0 m"),
                                 (cb.D_MID, "#c0504d", "at midspan, 4.5 m")):
        t_bot, t_web = cb.thickness(depth)
        y = np.linspace(0.0, depth, 4001)
        o, v = cb._edges(y, depth, t_bot, t_web)
        z = -y / 1e3
        for sgn in (1, -1):
            ax[0].fill_betweenx(z, sgn * v / 1e3, sgn * o / 1e3, color=colour,
                                alpha=0.35, lw=0.0,
                                label=label if sgn == 1 else None)
        yc = cb.section(depth)[1]
        ax[0].plot([-o.max() / 1e3, o.max() / 1e3], [-yc / 1e3] * 2, colour,
                   lw=1.0, ls="--")
        ax[0].annotate("centroid %.2f m" % (yc / 1e3), (o.max() / 1e3, -yc / 1e3),
                       fontsize=7, color=colour, ha="right", va="bottom")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x from the centreline (m)")
    ax[0].set_ylabel("below the deck top (m)")
    ax[0].set_title("reconstructed section\n(Dilger & Langohr Fig. 9)")
    ax[0].legend(fontsize=8, loc="lower right")

    u = np.linspace(0.0, 125.0, 400) * 1e+3
    d = cb.depth_at(u)
    pr = np.array([cb.section(v) for v in d])
    for k, (col, lab, pub) in enumerate((
            (0, "A (m$^2$)", [(13.990, 22.247), (13.000, 19.218)]),
            (2, "I (m$^4$)", [(13.990, 577.79), (13.000, 454.96)]))):
        axx = ax[1 + k]
        axx.plot(d / 1e3, pr[:, col] / (1e+6 if col == 0 else 1e+12), "#1f4e79", lw=1.6,
                 label="reconstruction")
        axx.plot([p[0] for p in pub], [p[1] for p in pub], "o", ms=7, mfc="none",
                 mec="#c0504d", mew=1.8, label="Butcher (2009) App. E")
        axx.set_xlabel("girder depth (m)")
        axx.set_ylabel(lab)
        axx.legend(fontsize=8)
    ax[1].set_title("two thicknesses fitted on this table,\nA / centroid / I then predicted")
    ax[2].set_title("within 4% at both depths;\nmidspan centroid 0.9% with nothing fitted")
    plt.tight_layout()
    plt.savefig(str(out) + "_section.png", dpi=130)
    print(str(out) + "_section.png")

    # ------------------------------------------------------------- trucks
    h5 = ROOT / "results/case_28_confederation_trucks.h5"
    if h5.exists():
        x, w = _axis(cb, h5)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        ax[0].plot(x, -1e+3 * w[-1], "#1f4e79", lw=1.8, label="this model")
        for key, style, lab in (("measured", "o", "measured (Fig. 17, read off)"),
                                ("analysis", "s", "Ghali's own analysis")):
            p = np.array(FIG17[key], float)
            ax[0].plot(p[:, 0], p[:, 1], style, ms=6, mfc="none", mew=1.6,
                       mec="#c0504d" if key == "measured" else "#7f7f7f", label=lab)
        for xp in (-125, 125):
            ax[0].axvline(xp, color="k", lw=0.8, ls="-.", alpha=0.5)
        ax[0].invert_yaxis()
        ax[0].set_xlabel("distance from midspan (m)")
        ax[0].set_ylabel("deflection (mm), down positive")
        ax[0].set_title("case_28: two trucks (725 kN) at midspan")
        ax[0].legend(fontsize=8)

        with h5py.File(h5) as f:
            mon = {k: f["monitors"][k][:] for k in
                   ("w_mid", "w_tip_west", "w_tip_east")}
        pos = np.r_[-220.0 + 20.0 * np.arange(1, 23), 0.0]
        for k, (nm, col) in enumerate((("w_mid", "#1f4e79"),
                                       ("w_tip_west", "#c0504d"),
                                       ("w_tip_east", "#4f8f3f"))):
            ax[1].plot(pos[:22], -1e+3 * mon[nm][:22], col, marker=".", lw=1.4,
                       label=nm.replace("w_", "").replace("_", " "))
        for xp in (-125, 125):
            ax[1].axvline(xp, color="k", lw=0.8, ls="-.", alpha=0.5)
        ax[1].invert_yaxis()
        ax[1].set_xlabel("position of the trucks' resultant (m)")
        ax[1].set_ylabel("deflection (mm), down positive")
        ax[1].set_title("the influence lines the 22-station sweep gives")
        ax[1].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(str(out) + "_trucks.png", dpi=130)
        print(str(out) + "_trucks.png")

    # ------------------------------------------------------------ thermal
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for depth, colour in ((cb.D_PIER, "#1f4e79"), (9000.0, "#7f7f7f"),
                          (cb.D_MID, "#c0504d")):
        y = np.linspace(0.0, depth, 400)
        T = np.where(y <= cb.T_DECK_WEB,
                     cb.T_TOP + (cb.T_SOFFIT_SLAB - cb.T_TOP) * y / cb.T_DECK_WEB,
                     cb.T_SOFFIT_SLAB + (cb.T_BOTTOM - cb.T_SOFFIT_SLAB)
                     * (y - cb.T_DECK_WEB) / (depth - cb.T_DECK_WEB))
        T0, g = cb.temperature_field(depth)
        yc = cb.section(depth)[1]
        ax[0].plot(T, -y / 1e3, colour, lw=1.6, label="%.1f m deep" % (depth / 1e3))
        ax[0].plot(T0 + g * (yc - y) / 1e3, -y / 1e3, colour, lw=1.0, ls="--")
    ax[0].set_xlabel("temperature rise (K)")
    ax[0].set_ylabel("below the deck top (m)")
    ax[0].set_title("Ghali Fig. 21's profile (solid) and the equivalent\n"
                    "linear field it becomes on each section (dashed)")
    ax[0].legend(fontsize=8, loc="lower right")

    for nm, colour, lab in (("case_28b_confederation_thermal", "#1f4e79",
                             "whole profile (case_28b)"),
                            ("case_28c_confederation_uniform", "#4f8f3f",
                             "mean rise only (case_28c)")):
        h5 = ROOT / ("results/%s.h5" % nm)
        if not h5.exists():
            continue
        x, w = _axis(cb, h5)
        ax[1].plot(x, -1e+3 * w[-1], colour, lw=1.8, label=lab)
    p = np.array(FIG21, float)
    ax[1].plot(p[:, 0], p[:, 1], "o", ms=6, mfc="none", mec="#c0504d", mew=1.6,
               label="Fig. 21 (read off)")
    for xp in (-125, 125):
        ax[1].axvline(xp, color="k", lw=0.8, ls="-.", alpha=0.5)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("distance from midspan (m)")
    ax[1].set_ylabel("deflection (mm), down positive")
    ax[1].set_title("case_28b / 28c: a summer temperature rise")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(out) + "_thermal.png", dpi=130)
    print(str(out) + "_thermal.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
