"""Figures for cases/case_17 (hinged roof) and case_18 (hemisphere).

    python benchmarks/plot_sze_shells.py roof|hemisphere [result.h5] [outdir]
"""
import sys, pathlib
import h5py, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

which = sys.argv[1]
src = sys.argv[2] if len(sys.argv) > 2 else {"roof": "results/case_17_hinged_cylindrical_roof.h5",
                                            "hemisphere": "results/case_18_hemisphere_pinched.h5"}[which]
out = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else "results/figures"); out.mkdir(parents=True, exist_ok=True)
cjk = [f for f in font_manager.findSystemFonts() if any(k in f.lower() for k in ("noto", "cjk", "wqy", "droid"))]
if cjk: plt.rcParams["font.family"] = font_manager.FontProperties(fname=cjk[0]).get_name()
plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.titleweight": "bold", "axes.titlesize": 11, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": "#e8e7e1", "grid.linewidth": 0.6, "savefig.dpi": 200})
BLUE, ORANGE, INK, MUTED, LINE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#c3c2b7"
ARROW = dict(arrowstyle="-", color=MUTED, lw=0.8)
f = h5py.File(src); m = f["monitors"]

if which == "roof":
    ref = np.loadtxt("verification/Sze2004_roof_12p7_isotropic.txt", encoding="utf-8")
    w = -np.asarray(m["w_C"]); P = -4 * np.asarray(m["P4"]) / 3000.0
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhline(0, color=LINE, lw=0.8, zorder=1)
    ax.plot(w, P, color=BLUE, lw=2, zorder=3, label="VFIFE 松弛求解，位移控制，60 级")
    ax.plot(w, P, "o", ms=3.5, color=BLUE, mec="#fcfcfb", mew=0.7, zorder=4)
    ax.plot(ref[:, 1], ref[:, 0], "s", ms=6, color=ORANGE, mec="#fcfcfb", mew=1, zorder=5, label="Sze et al. 2004, Table 9(a)（Abaqus S4R, Riks）")
    i = int(np.argmax(P[:40])); j = int(np.argmin(P[15:50])) + 15
    ax.annotate(f"极限点 {P[i]:.3f} P_max（Sze 0.742）", (w[i], P[i]), (2, 0.9), color=INK, fontsize=9, arrowprops=ARROW)
    ax.annotate(f"最低点 {P[j]:.3f} P_max（Sze 0.173）：\n翻转中最平处，模型偏高 10%", (w[j], P[j]), (12, -0.15), color=INK, fontsize=9, arrowprops=ARROW)
    ax.annotate("翻转完成，重新变刚", (w[-1], P[-1]), (18, 1.0), color=INK, fontsize=9, arrowprops=ARROW)
    ax.set_xlabel("−w_C / mm（顶点竖向位移）"); ax.set_ylabel("P / P_max   (P_max = 3000 N)")
    ax.set_ylim(-0.3, 1.2); ax.grid(True, zorder=0); ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("铰支圆柱壳屋盖：snap-through 全路径（case_17，h = 12.7）")
    fig.tight_layout(); fig.savefig(out / "case_17_P_w.png"); plt.close(fig)

    # shapes: the symmetry line y = 0 (the arc through the crown) at four levels
    X = np.asarray(f["mesh/nodes"]); keys = sorted(f["steps"], key=int)
    on_axis = np.flatnonzero(np.abs(X[:, 1]) < 1e-9); order = on_axis[np.argsort(X[on_axis, 0])]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(X[order, 0], X[order, 2], color=LINE, ls="--", lw=1, label="参考构形")
    cmap = plt.get_cmap("Blues")
    for k, (idx, lab) in enumerate([(9, "w_C = 5"), (21, "11（极限点）"), (38, "19.5（最低点）"), (57, "29")]):
        d = np.asarray(f["steps"][keys[idx]]["displacement"]); x = X + d
        ax.plot(x[order, 0], x[order, 2], color=cmap(0.4 + 0.55 * k / 3), lw=2, label=lab)
    ax.set_xlabel("x / mm（沿弧）"); ax.set_ylabel("z / mm（纵轴拉伸约 5×）"); ax.grid(True, zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="upper right"); ax.set_title("对称线 y = 0 上的弧线：从拱到反拱（位移不放大，纵轴拉伸）")
    fig.tight_layout(); fig.savefig(out / "case_17_arc.png"); plt.close(fig)
else:
    ref = np.loadtxt("verification/Sze2004_hemisphere.txt", encoding="utf-8")
    uA = -np.asarray(m["u_A"]); uB = np.asarray(m["u_B"]); P = np.linspace(1, len(uA), len(uA)) / len(uA)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(uA, P, color=BLUE, lw=2, label="VFIFE：A 点内凹 V_A"); ax.plot(uB, P, color=BLUE, lw=2, ls=(0, (5, 3)), label="VFIFE：B 点外凸 −U_B")
    ax.plot(ref[:, 1], ref[:, 0], "s", ms=6, color=ORANGE, mec="#fcfcfb", mew=1, label="Sze 2004 Table 5(a)：V_A")
    ax.plot(ref[:, 2], ref[:, 0], "D", ms=6, color=ORANGE, mec="#fcfcfb", mew=1, label="Sze 2004：−U_B")
    ax.set_xlabel("径向位移"); ax.set_ylabel("P / P_max   (P_max = 400)"); ax.grid(True, zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="lower right"); ax.set_title("带孔半球壳受交替径向力：荷载—位移（case_18，16×16）")
    fig.tight_layout(); fig.savefig(out / "case_18_P_u.png"); plt.close(fig)
print("wrote", sorted(p.name for p in out.glob(f"case_1{'7' if which=='roof' else '8'}_*.png")))
