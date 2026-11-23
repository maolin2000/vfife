"""Chen's two 13 m hollow-core slabs: the model's P-delta against the
measured points.

    python benchmarks/plot_chen_slab.py [-o results/chen_slab.png]

Reads results/case_25{,b}_chen_hollow_slab.h5 and
verification/Chen2005_hollow_slab_{1,2}.txt, prints the comparison table
and draws both beams.

THE DEFLECTION IS MEASURED FROM THE START OF LOADING. The dial gauges
were zeroed on the slab as it stood -- self-weight on, prestress in, the
midspan cambered up a few millimetres -- so the model's curve is shifted
by its own state at the end of the transfer step. P is the force at ONE
jack, which is the reaction at a pushed node.
"""
import argparse
import pathlib

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BEAMS = (("1", "case_25_chen_hollow_slab", 245.6, 155.0),
         ("2", "case_25b_chen_hollow_slab", 315.0, 158.0))


def load(stem):
    with h5py.File(ROOT / f"results/{stem}.h5") as h:
        t = h["monitors/time"][...]
        w = -h["monitors/w_mid"][...] * 1e3          # mm, down positive
        P = -h["monitors/P_jack"][...] / 1e3         # kN, the jack's force
    push = t > 1.0                                   # the loading step
    zero = w[np.argmin(abs(t - 1.0))]
    return w[push] - zero, P[push] - P[np.argmin(abs(t - 1.0))], zero


def stable(w, P):
    """The branch the test can be compared with: the frames up to the point
    where the slab snaps. The push is displacement controlled, so while the
    slab takes load the midspan goes down monotonically; when it lets go,
    one frame comes back UP a millimetre and shows a force that is the
    catch, not a capacity (beam 2# records 370.6 kN there against a smooth
    branch that ends at 298.4). Keeping only the frames that are still at
    their running maximum deflection drops that frame and everything past
    it -- and leaves the curve untouched for a beam that never snaps."""
    k = w >= np.maximum.accumulate(w) - 1e-9
    if k.all():
        return w, P, None
    end = int(np.argmin(k))                          # first frame off the branch
    return w[:end], P[:end], float(P[end:].max())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--out", default=str(ROOT / "results/chen_slab.png"))
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for k, (beam, stem, P_u, d_u) in enumerate(BEAMS):
        meas = np.loadtxt(ROOT / f"verification/Chen2005_hollow_slab_{beam}.txt", encoding="utf-8")
        try:
            w, P, camber = load(stem)
        except (OSError, KeyError):
            print(f"beam {beam}#: no result yet"); continue
        w_all, P_all = w, P
        w, P, snap = stable(w, P)
        print(f"\nbeam {beam}#  (camber before loading {camber:+.1f} mm)")
        print("  delta [mm]   measured P   model P    ratio")
        for d in (20, 40, 60, 80, 100, 120, 140, d_u):
            if d > w.max() or d > meas[:, 0].max():
                continue
            pm = np.interp(d, meas[:, 0], meas[:, 1]); pc = np.interp(d, w, P)
            print(f"  {d:8.1f} {pm:12.1f} {pc:10.1f} {pc / pm:8.3f}")
        top = int(np.argmax(P))
        print(f"  the model peaks at {P[top]:.1f} kN / {w[top]:.1f} mm "
              f"(measured failure {P_u} at {d_u} mm)"
              + (f"; one frame at the snap records {snap:.1f} kN" if snap else ""))
        b = ax[k]
        b.plot(meas[:, 0], meas[:, 1], "o", ms=5, mfc="none", mec="crimson", mew=1.4,
               label=f"陈衡治 2005 图6.20({'ab'[k]}) 实测")
        b.plot([d_u], [P_u], "*", ms=14, color="crimson", label=f"实测破坏 {P_u} kN / {d_u:.0f} mm")
        if snap:
            b.plot(w_all[len(w):], P_all[len(w):], ":", lw=1.2, color="#1f77b4", alpha=0.6,
                   label=f"（断裂瞬间的一帧 {snap:.0f} kN 与其后的卸载）")
        b.plot(w, P, "-", lw=1.8, color="#1f77b4", label="vfife")
        b.set_xlabel("跨中挠度（自加载起）δ / mm"); b.set_ylabel("单点荷载 P / kN")
        b.set_title(f"{beam}# 试验梁：两点加载 {'400' if beam == '1' else '600'} cm")
        b.grid(alpha=0.3); b.set_xlim(0, 175); b.set_ylim(0, 1.35 * P_u)
        b.legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(a.out, dpi=130)
    print("\nwritten", a.out)


if __name__ == "__main__":
    main()
