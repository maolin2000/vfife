"""The Vecchio & Emara (1992) frame as a Gmsh 2.2 line mesh (cases/case_21).

One bay, two storeys, in the x-z plane (y out of plane). Centreline
dimensions from Fig. 10 of Guner & Vecchio (2010): columns at x = 0 and
x = 3.5 m from the top of the base beam (z = 0) to the second-storey beam
centreline (z = 4.0 m); beams at z = 2.0 and z = 4.0 m (clear storey heights
1800 and 1600 mm plus the 400 mm beam depth). The base beam is bolted to the
strong floor and is not modelled: the columns are built in at z = 0.

    base_l, base_r   (0, 0, 0), (3.5, 0, 0)     column bases, fixed
    top_l, top_r     (0, 0, 4), (3.5, 0, 4)     column tops: 700 kN each
    drift            the node the lateral displacement is applied to (top_l)

`n` elements per member: 4 gives 0.5 m column elements and 0.875 m beam
elements, 24 elements in all -- the ten-minute model. The joint nodes belong
to both a column and a beam; the regions are boxes in the case file, so an
element is a column element when BOTH its nodes lie on a column line.
"""
import argparse
import pathlib

SPAN, STOREY = 3.5, 2.0


def build(path, n=4):
    nodes, index = [], {}

    def node(x, z):
        key = (round(x, 9), round(z, 9))
        if key not in index:
            index[key] = len(nodes); nodes.append((x, 0.0, z))
        return index[key]

    lines = []
    for x in (0.0, SPAN):                                   # two columns, two storeys
        for s in range(2):
            for i in range(n):
                a = node(x, STOREY * s + STOREY * i / n)
                b = node(x, STOREY * s + STOREY * (i + 1) / n)
                lines.append((a, b))
    for z in (STOREY, 2 * STOREY):                         # two beams
        for i in range(n):
            lines.append((node(SPAN * i / n, z), node(SPAN * (i + 1) / n, z)))
    points = {"base_l": node(0.0, 0.0), "base_r": node(SPAN, 0.0),
              "top_l": node(0.0, 2 * STOREY), "top_r": node(SPAN, 2 * STOREY)}
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat",
           "$PhysicalNames", str(len(points) + 1)]
    out += [f'0 {k + 1} "{name}"' for k, name in enumerate(points)]
    out += [f'1 {len(points) + 1} "members"', "$EndPhysicalNames",
            "$Nodes", str(len(nodes))]
    out += [f"{k + 1} {x:.10g} {y:.10g} {z:.10g}" for k, (x, y, z) in enumerate(nodes)]
    out += ["$EndNodes", "$Elements", str(len(points) + len(lines))]
    e = 1
    for k, (name, nd) in enumerate(points.items()):
        out.append(f"{e} 15 2 {k + 1} {k + 1} {nd + 1}"); e += 1
    for a, b in lines:
        out.append(f"{e} 1 2 {len(points) + 1} {len(points) + 1} {a + 1} {b + 1}"); e += 1
    out += ["$EndElements", ""]
    pathlib.Path(path).write_text("\n".join(out), encoding="utf-8")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="benchmarks/assets/vecchio_emara_frame_4.msh")
    ap.add_argument("--n", type=int, default=4, help="elements per member")
    a = ap.parse_args()
    print("wrote", build(a.out, a.n))
