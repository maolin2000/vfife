"""The 45-degree bend cantilever of Bathe & Bolourchi (1979) as a Gmsh 2.2
line mesh (cases/case_18).

One eighth of a circle of radius R = 100 in the x-y plane, centre of
curvature on the x axis at (R, 0, 0): clamped at the origin where its
tangent is +y, free at 45 degrees -- the frame the references use, in
which the tip moves (-13.4, -23.5, +53.4) under P = 600. The
transverse tip load is out of the plane (+z), so the beam bends about two
axes and twists at once -- the reason this is THE finite-rotation benchmark
for 3-D beams (NAFEMS 3DNLG-5; Simo & Vu-Quoc 1986; Cardona & Geradin 1988;
Crisfield 1990).

    root   (0, 0, 0)
    tip    (R (1 - cos 45), R sin 45, 0) = (29.289, 70.711, 0)

`n` straight elements along the arc; the classical discretisation is 8.
"""
import argparse
import pathlib

import numpy as np

R = 100.0
ANGLE = np.pi / 4


def build(path, n=8):
    th = ANGLE * np.arange(n + 1) / n
    nodes = [(R * (1.0 - np.cos(t)), R * np.sin(t), 0.0) for t in th]
    lines = [(i, i + 1) for i in range(n)]
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat",
           "$PhysicalNames", "3", '0 1 "root"', '0 2 "tip"', '1 3 "members"', "$EndPhysicalNames",
           "$Nodes", str(len(nodes))]
    out += [f"{k + 1} {x:.12g} {y:.12g} {z:.12g}" for k, (x, y, z) in enumerate(nodes)]
    out += ["$EndNodes", "$Elements", str(2 + len(lines))]
    out.append("1 15 2 1 1 1")
    out.append(f"2 15 2 2 2 {n + 1}")
    for e, (a, b) in enumerate(lines, 3):
        out.append(f"{e} 1 2 3 3 {a + 1} {b + 1}")
    out += ["$EndElements", ""]
    pathlib.Path(path).write_text("\n".join(out), encoding="utf-8")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="benchmarks/assets/bend45_8_v22.msh")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    print("wrote", build(a.out, a.n))
