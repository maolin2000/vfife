"""The two moving-load cases against their closed forms.

    python benchmarks/plot_moving_load.py [-o results/moving_load.png]

Left: case_26, the crossing at 40 m/s against Fryba's series. Right:
case_26b, the quasi-static influence line against P a (3L^2 - 4a^2)/48EI,
with the load's station read off the left reaction (R = P (1 - a/L), which is
statics and needs nothing from the model's displacements).
"""
import argparse
import pathlib

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main(argv=None):
    import sys
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from generate_moving_load_beam import FORCE, SPAN, SPEED, fryba, properties

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--out", default=str(ROOT / "results/moving_load.png"))
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans"]
    p = properties()
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    try:
        with h5py.File(ROOT / "results/case_26_moving_load_beam.h5") as h:
            t = h["monitors/time"][...]; w = -h["monitors/w_mid"][...] * 1e3
        exact = fryba(SPAN / 2, t) * 1e3
        print("case_26  crossing at %.0f m/s (alpha = %.2f)" % (SPEED, np.pi*SPEED/SPAN/p["omega1"]))
        print("   t [s]    Fryba [mm]   vfife [mm]    ratio")
        for tt in (0.10, 0.20, 0.25, 0.30, 1.0/3, 0.40):
            k = int(np.argmin(abs(t - tt)))
            print(f"  {t[k]:6.3f} {exact[k]:12.3f} {w[k]:12.3f} {w[k]/exact[k]:9.4f}")
        print(f"  peak   {exact.max():12.3f} {w.max():12.3f} {w.max()/exact.max():9.4f}"
              f"   (static {p['static']*1e3:.3f} mm, amplification {w.max()/p['static']/1e3:.3f})")
        ax[0].plot(t, exact, "-", lw=3.0, color="crimson", alpha=0.45, label="Fryba 闭合解")
        ax[0].plot(t, w, "-", lw=1.4, color="#1f77b4", label="vfife")
        ax[0].axhline(p["static"] * 1e3, ls=":", color="0.4", lw=1.2,
                      label=f"静力 P L³/48EI = {p['static']*1e3:.1f} mm")
        ax[0].set_xlabel("时间 t / s（过桥 0 → 0.5 s）"); ax[0].set_ylabel("跨中挠度 / mm")
        ax[0].set_title(f"case_26 恒力过桥 {SPEED:.0f} m/s（α = 0.50）")
        ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8, loc="upper left")
    except OSError:
        print("case_26: no result yet")

    try:
        with h5py.File(ROOT / "results/case_26b_moving_load_influence.h5") as h:
            w = -h["monitors/w_mid"][...] * 1e3; R = h["monitors/R_left"][...]
        s = SPAN * (1.0 - R / FORCE)                 # statics, not the model
        b = SPAN - s
        x = SPAN / 2
        ex = np.where(x <= s, FORCE*b*x*(SPAN**2 - b**2 - x**2)/(6*SPAN*p["EI"]),
                      FORCE*s*(SPAN-x)*(SPAN**2 - s**2 - (SPAN-x)**2)/(6*SPAN*p["EI"])) * 1e3
        print("\ncase_26b influence line: %d stations, worst relative error %.2e"
              % (len(s), np.abs(w/ex - 1).max()))
        ax[1].plot(s, ex, "-", lw=3.0, color="crimson", alpha=0.45, label="P a (3L²−4a²)/48EI")
        ax[1].plot(s, w, "o-", ms=4, lw=1.2, color="#1f77b4", label="vfife（39 个静力位置）")
        for node in np.arange(0, SPAN + 0.1, 1.0):
            ax[1].axvline(node, color="0.85", lw=0.6, zorder=0)
        ax[1].set_xlabel("荷载位置 a / m（细线为节点；一半的位置在单元内部）")
        ax[1].set_ylabel("跨中挠度 / mm")
        ax[1].set_title("case_26b 影响线（准静力扫掠）")
        ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8, loc="upper left")
    except OSError:
        print("case_26b: no result yet")

    plt.tight_layout(); plt.savefig(a.out, dpi=130)
    print("\nwritten", a.out)


if __name__ == "__main__":
    main()
