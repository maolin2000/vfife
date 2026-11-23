"""Figures for cases/case_16 (Lee's frame): the load path in two projections,
the deformed shapes level by level, and the corner's motion through the jump.

    python benchmarks/plot_lee_frame.py [results/case_16_lee_frame.h5] [outdir]
"""
import sys, pathlib
import h5py, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

src = sys.argv[1] if len(sys.argv) > 1 else "results/case_16_lee_frame.h5"
out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "results/figures"); out.mkdir(parents=True, exist_ok=True)
cjk = [f for f in font_manager.findSystemFonts() if any(k in f.lower() for k in ("noto", "cjk", "wqy", "droid"))]
if cjk:
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=cjk[0]).get_name()
plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.titleweight": "bold", "axes.titlesize": 11, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": "#e8e7e1", "grid.linewidth": 0.6, "savefig.dpi": 200})
BLUE, ORANGE, INK, MUTED, RED, LINE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e34948", "#c3c2b7"
E, I, L = 71.74e9, 2.0e-8, 1.2
f = h5py.File(src); m = f["monitors"]
v = -np.asarray(m["v_A"])/L; u = np.asarray(m["u_A"])/L; lam = -np.asarray(m["P"])*L*L/(E*I)
uc = np.asarray(m["u_C"])/L; wc = -np.asarray(m["w_C"])/L
jump = int(np.argmax(np.diff(u)))
i_max = int(np.argmax(lam[:jump+1])); i_min = int(np.argmin(lam))
ARROW = dict(arrowstyle="-", color=MUTED, lw=0.8)

def path(ax, x, y, xlabel):
    ax.axhline(0, color=LINE, lw=0.8, zorder=1)
    ax.plot(x[:jump+1], y[:jump+1], color=BLUE, lw=2, zorder=3)
    ax.plot(x[jump:], y[jump:], color=BLUE, lw=2, zorder=3)
    ax.plot([x[jump], x[jump+1]], [y[jump], y[jump+1]], color=MUTED, lw=1.2, ls=(0, (5, 4)), zorder=2)
    ax.plot(x, y, "o", ms=4, color=BLUE, mec="#fcfcfb", mew=0.8, zorder=4)
    ax.set_xlabel(xlabel); ax.set_ylabel("P L² / EI"); ax.grid(True, zorder=0); ax.set_ylim(-14, 40)

# 1. P against v_A
fig, ax = plt.subplots(figsize=(7.2, 4.6))
path(ax, v, lam, "v_A / L   (A 点竖向位移)")
for x0, y0 in [(0.407, 18.552), (0.784, 31.887)]:
    ax.plot(x0, y0, "s", ms=9, color=ORANGE, mec="#fcfcfb", mew=1.2, zorder=5)
ax.annotate("NAFEMS 极限点\n(18.552, 0.407)", (0.407, 18.552), (0.16, 24), color=INK, fontsize=9, arrowprops=ARROW)
ax.annotate("NAFEMS 第二点\n(31.887, 0.784)", (0.784, 31.887), (0.56, 33), color=INK, fontsize=9, arrowprops=ARROW)
ax.annotate(f"模型极限点 {lam[i_max]:.2f}\n(v/L = {v[i_max]:.3f})", (v[i_max], lam[i_max]), (0.30, 6),
            color=BLUE, fontsize=9, arrowprops=ARROW)
ax.annotate("snap-back：位移控制跟不上折返，\n松弛动态跳到远支（灰虚线不是平衡路径）",
            ((v[jump]+v[jump+1])/2, (lam[jump]+lam[jump+1])/2), (0.55, -6), color=INK, fontsize=9, arrowprops=ARROW)
ax.annotate(f"荷载最小值 {lam[i_min]:.2f}\n（刚架需被拉住）", (v[i_min], lam[i_min]), (0.60, -12.5),
            color=INK, fontsize=9, arrowprops=ARROW)
ax.set_title("Lee 刚架：荷载—竖向位移路径（case_16，位移控制，47 级）")
ax.text(0.01, 0.02, "蓝：VFIFE 松弛求解　橙：NAFEMS NL7 参考点", transform=ax.transAxes, color=MUTED, fontsize=8.5)
fig.tight_layout(); fig.savefig(out / "case_16_P_v.png"); plt.close(fig)

# 2. P against u_A
fig, ax = plt.subplots(figsize=(7.2, 4.6))
path(ax, u, lam, "u_A / L   (A 点水平位移)")
ax.annotate(f"极限点 {lam[i_max]:.2f}", (u[i_max], lam[i_max]), (0.05, 24), color=BLUE, fontsize=9, arrowprops=ARROW)
ax.annotate(f"一级之内 u_A/L 从 {u[jump]:.2f} 跳到 {u[jump+1]:.2f}：\n这就是 snap-back——折返段在位移控制下无法追踪",
            ((u[jump]+u[jump+1])/2, (lam[jump]+lam[jump+1])/2), (0.08, -8), color=INK, fontsize=9, arrowprops=ARROW)
ax.annotate("远支：u 回缩、P 反号后陡升\n（两根铰接杆被拉成索）", (u[-1], lam[-1]), (0.40, 36), color=INK, fontsize=9, arrowprops=ARROW)
ax.set_title("Lee 刚架：荷载—水平位移路径（snap-back 在这个投影里最明显）")
fig.tight_layout(); fig.savefig(out / "case_16_P_u.png"); plt.close(fig)

# 3. deformed shapes, small multiples
X = np.asarray(f["mesh/nodes"]); keys = sorted(f["steps"], key=int)
def shape(idx): return X + np.asarray(f["steps"][keys[idx]]["displacement"])
panels = [([11], "v/L = 0.20"), ([23], f"0.40  极限点 P L²/EI = {lam[23]:.1f}"), ([29], f"0.50  跳跃前 (P = {lam[29]:.1f})"),
          ([29, 30], f"0.50 → 0.52  一级之内的跳跃"), ([41], f"0.70  P = {lam[41]:.1f}（负）"), ([46], f"0.78  P = {lam[46]:.1f}")]
fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), sharex=True, sharey=True)
for ax, (ids, title) in zip(axes.ravel(), panels):
    ax.plot(X[:, 0], X[:, 2], color=LINE, lw=1.0, ls="--", zorder=1)
    for n, idx in enumerate(ids):
        x = shape(idx); col = BLUE if n == len(ids)-1 else "#9cc3ef"
        ax.plot(x[:, 0], x[:, 2], color=col, lw=2, zorder=3)
        ax.plot(x[24, 0], x[24, 2], "o", ms=6, color=col, mec="#fcfcfb", mew=0.8, zorder=4)
    if len(ids) == 2:
        a, b = shape(ids[0])[24], shape(ids[1])[24]
        ax.annotate("", (b[0], b[2]), (a[0], a[2]), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.plot([0, L], [0, L], "^", ms=8, color=ORANGE, mec="#fcfcfb", mew=1, zorder=5)
    ax.set_aspect("equal"); ax.grid(True, zorder=0); ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.3, 1.6); ax.set_ylim(-0.35, 1.3)
for ax in axes[1]: ax.set_xlabel("x / m")
for ax in axes[:, 0]: ax.set_ylabel("z / m")
fig.suptitle("Lee 刚架：各载荷级的变形构形（蓝点 = A 点，橙 = 铰支，虚线 = 参考构形）", fontweight="bold")
fig.tight_layout(); fig.savefig(out / "case_16_shapes.png"); plt.close(fig)

# 4. the corner through the jump
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(v, uc, color=BLUE, lw=2, label="角点水平位移 u_C / L")
ax.plot(v, wc, color=ORANGE, lw=2, label="角点竖向位移 −w_C / L（向下为正）")
ax.axvspan(v[jump], v[jump+1], color="#e8e7e1", zorder=0)
ax.text((v[jump]+v[jump+1])/2, 0.02, "跳跃", ha="center", color=MUTED, fontsize=9)
ax.set_xlabel("v_A / L"); ax.set_ylabel("角点位移 / L"); ax.grid(True, zorder=0)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("角点随 A 点下压的运动：跳跃处两个分量同时突变")
fig.tight_layout(); fig.savefig(out / "case_16_corner.png"); plt.close(fig)
print("wrote", sorted(p.name for p in out.glob("case_16_*.png")))
