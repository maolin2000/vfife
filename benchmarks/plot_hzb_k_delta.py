"""The Hangzhou Bay 3 x 70 m unit under load case 1: the two vfife runs
against Chen's digitised curve.

    python benchmarks/plot_hzb_k_delta.py [-o results/hzb_k_delta.png]

Reads results/case_24_hzb_girder.h5 (staged relaxation to k = 10),
results/case_24c_hzb_girder_collapse.h5 (dynamic ramp to collapse) and
verification/Chen2005_hzb_girder_case1.txt, and prints the comparison
table it plots.

WHY THREE PANELS. k = 0 IS NOT THE ORIGIN: the dead load and the
prestress are already on, and the curves start at their own dead-load
deflection -- 9 mm (dynamic) / 6 mm (relaxation) here, 21 mm for Chen.
The 15 mm between his and this model's is the construction sequence
(先简后连: q_d1 acts on a SIMPLY SUPPORTED 70 m span, 126 mm down
against 114 mm of camber from the same span's tendons; continuous from
the start, the interior span takes 0.00052 q L^4 / EI, a twenty-fifth of
the simply supported value). So the third panel compares INCREMENTS from
each curve's own k = 0, which is the only comparison that means the same
thing on both sides.
"""
import argparse
import pathlib

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load():
    with h5py.File(ROOT / "results/case_24c_hzb_girder_collapse.h5") as h:
        t = h["monitors/time"][...]; w = -h["monitors/w_mid_middle"][...] * 1e3
    with h5py.File(ROOT / "results/case_24_hzb_girder.h5") as h:
        ts = h["monitors/time"][...]; ws = -h["monitors/w_mid_middle"][...] * 1e3
    chen = np.loadtxt(ROOT / "verification/Chen2005_hzb_girder_case1.txt", encoding="utf-8")
    return t, w, ts, ws, chen


def table(t, w, ts, ws, chen):
    k = np.clip((t - 10.0) / 2.5, 0.0, None)
    d0 = w[np.argmin(abs(t - 10.0))]
    lines = [f"dead state (k = 0): dynamic {d0:.1f} mm, relaxation "
             f"{ws[np.argmin(abs(ts - 1.0))]:.1f} mm, Chen {chen[0, 0]:.1f} mm", "",
             "   k   model total   model incr    Chen total    Chen incr    ratio"]
    for kk in (2, 4, 6, 8, 10, 12, 14, 14.5):
        m = float(np.interp(kk, k[t >= 10], w[t >= 10]))
        c = float(np.interp(kk, chen[:, 1], chen[:, 0]))
        lines.append(f"{kk:5.1f} {m:12.1f} {m - d0:12.1f} {c:13.1f} {c - chen[0, 0]:12.1f}"
                     f" {(m - d0) / (c - chen[0, 0]):8.2f}")
    for lim in (400.0, 727.0, 2000.0):
        i = int(np.argmax(w > lim))
        lines.append(f"passes {lim:6.0f} mm at t = {t[i]:6.2f} s, k = {k[i]:5.2f}")
    return "\n".join(lines)


def plot(path, t, w, ts, ws, chen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans"]
    k = np.clip((t - 10.0) / 2.5, 0.0, None); m = t >= 10
    d0 = w[np.argmin(abs(t - 10.0))]; s0 = ws[np.argmin(abs(ts - 1.0))]
    sm = ts >= 1.0; ks = (ts[sm] - 1.0) * 10.0; wss = ws[sm]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for a in ax[:2]:
        a.plot(w[m], k[m], "-", lw=1.7, color="#1f77b4", label="vfife case_24c 动力斜坡")
        a.plot(wss, ks, "o", ms=4.5, color="#ff7f0e", label="vfife case_24 逐级松弛（静力）")
        a.plot(chen[:, 0], chen[:, 1], "--", lw=1.7, color="crimson", label="陈衡治 2005 图6.30(a) 数字化")
        a.plot([d0], [0], "o", ms=9, mfc="none", mec="#1f77b4", mew=2)
        a.plot([s0], [0], "s", ms=9, mfc="none", mec="#ff7f0e", mew=2)
        a.plot([chen[0, 0]], [0], "^", ms=9, mfc="none", mec="crimson", mew=2)
        a.set_xlabel("中跨跨中总挠度 δ / mm（含恒载）"); a.set_ylabel("活载系数 k"); a.grid(alpha=0.3)
    ax[0].set_xlim(-15, 460); ax[0].set_ylim(0, 16.5)
    ax[1].set_xlim(-60, 3500); ax[1].set_ylim(0, 20)
    ax[0].set_title("k = 0 不是原点：恒载 + 预应力已经在了")
    ax[1].set_title("到倒塌")
    ax[0].annotate("恒载状态 k=0：\n本模型 9（动力）/ 6 mm（静力），陈 21 mm\n差的 15 mm 是先简后连：\n"
                   "q$_{d1}$=306 kN/m 作用在简支的 70 m 梁上（下挠 126）\n对束在同一简支梁上的上拱（114），\n连续梁上只有 1/25",
                   xy=(chen[0, 0], 0.15), xytext=(120, 1.0), fontsize=8.5, va="bottom",
                   arrowprops=dict(arrowstyle="->", lw=0.9))
    ax[0].legend(fontsize=8, loc="upper left")
    ax[1].annotate("k=14.8 墩顶屈服，跳到束强化段", xy=(600, 14.9), xytext=(950, 11.5),
                   fontsize=8.5, arrowprops=dict(arrowstyle="->"))
    ax[1].annotate("k=18.1 机构，悬在束上", xy=(2100, 18.2), xytext=(1500, 19.3),
                   fontsize=8.5, arrowprops=dict(arrowstyle="->"))
    kk = np.arange(0, 14.6, 0.05)
    ax[2].plot(np.interp(kk, k[m], w[m]) - d0, kk, "-", lw=1.7, color="#1f77b4", label="vfife case_24c")
    ax[2].plot(np.interp(kk, chen[:, 1], chen[:, 0]) - chen[0, 0], kk, "--", lw=1.7, color="crimson", label="陈衡治")
    ax[2].plot(wss[ks <= 10.01] - s0, ks[ks <= 10.01], "o", ms=4.5, color="#ff7f0e", label="vfife case_24（静力）")
    ax[2].set_xlim(0, 420); ax[2].set_ylim(0, 14.5); ax[2].grid(alpha=0.3)
    ax[2].set_xlabel("自各自恒载状态起的活载挠度增量 / mm"); ax[2].set_ylabel("活载系数 k")
    ax[2].set_title("同一基准比：模型是陈的 0.65–0.72")
    ax[2].legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(path, dpi=130)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--out", default=str(ROOT / "results/hzb_k_delta.png"))
    a = ap.parse_args(argv)
    data = load()
    print(table(*data))
    print("written", plot(a.out, *data))


if __name__ == "__main__":
    main()
