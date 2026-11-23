"""Xihoumen modes at the design dead-load tensions, against both papers.

    python benchmarks/run_xihoumen_modal.py [outdir] [--modes N] [--relax] [--paraview]

outdir defaults to results/xihoumen (run from the repository root). --paraview
also writes the mode shapes to <outdir>/paraview/, one .pvd per mode named after
its classification, and xihoumen_all.pvd with every mode on one time axis.

Builds the mesh and case with `generate_xihoumen` (deck load calibrated to
the design's tower base force), asks `vfife.modal` for the lowest modes at
the stated dead-load state, classifies each one by where its kinetic energy
is, and prints them beside [J] Table 4 and the three frequencies [X] 4.1
states.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))

import generate_xihoumen as gx                  # noqa: E402
from vfife import modal                          # noqa: E402
from vfife.api import prepare_case               # noqa: E402
from vfife.core.structural import make_structural_solver   # noqa: E402

JIA = [(11.274, "girder lateral, sym"), (9.990, "girder vertical, anti"),
       (6.618, "girder vertical, sym"), (6.028, "main cable lateral, anti"),
       (5.903, "main cable lateral, sym"), (5.811, "main cable lateral, sym"),
       (5.675, "main cable vertical, anti"), (5.669, "main cable vertical, anti"),
       (5.647, "side cable lateral, sym"), (5.635, "side cable lateral, sym")]
XIAO = {"girder lateral, sym": 0.077, "girder torsion, sym": 0.254,
        "girder torsion, anti": 0.381}


def classify(sol, side, phi):
    """Where the mode's kinetic energy lives, and whether the deck part is
    symmetric about midspan."""
    m = sol.physical_mass.reshape(-1, 6)[:, 0]
    u = phi.reshape(-1, 6)[:, :3]
    ke = m[:, None] * u ** 2
    tot = ke.sum()
    rail, rm, rp = (np.array(side[k]) for k in ("rail", "road_m", "road_p"))
    deck = np.r_[rail, rm, rp]
    cable = np.array(side["cable"])
    x = np.array(side["x"])
    # deck: split vertical into bending (mean of the three boxes) and torsion
    uz_mean = (u[rail, 2] + u[rm, 2] + u[rp, 2]) / 3.0
    uz_tor = 0.5 * (u[rp, 2] - u[rm, 2])
    md = m[rail] + m[rm] + m[rp]
    e = {
        "girder lateral": float((ke[deck, 1]).sum()),
        "girder vertical": float((md * uz_mean ** 2).sum()),
        "girder torsion": float(((m[rm] + m[rp]) * uz_tor ** 2).sum()),
        "girder longitudinal": float((ke[deck, 0]).sum()),
        "cable lateral": float(ke[cable, 1].sum()),
        "cable vertical": float(ke[cable, 2].sum() + ke[cable, 0].sum()),
    }
    e["tower"] = float(tot - sum(e.values()))
    kind = max(e, key=e.get)
    share = e[kind] / tot
    # symmetry of the deck signal that defines the mode
    sig = {"girder lateral": u[rail, 1], "girder vertical": uz_mean,
           "girder torsion": uz_tor, "girder longitudinal": u[rail, 0]}.get(kind)
    xc = side["XT"][0] + gx.L_MAIN / 2
    if sig is None:                              # cable / tower: use the cable nodes
        xs_c = sol.mesh.nodes[cable, 0]
        comp = 1 if "lateral" in kind else 2
        order = np.argsort(xs_c)
        xs, sg = xs_c[order], u[cable, comp][order]
        # main span only, for the symmetry test
        sel = (xs > side["XT"][0]) & (xs < side["XT"][1])
        xs, sg = xs[sel], sg[sel]
    else:
        xs, sg = x, sig
    mirror = np.interp(2 * xc - xs, xs, sg)
    denom = np.sqrt(np.sum(sg ** 2) * np.sum(mirror ** 2)) or 1.0
    corr = float(np.sum(sg * mirror) / denom)
    sym = "sym" if corr > 0.5 else ("anti" if corr < -0.5 else "mixed")
    # is it the main-span cable or a back-span (side) cable?
    if kind.startswith("cable"):
        xc_nodes = sol.mesh.nodes[cable, 0]
        back = (xc_nodes < side["XT"][0]) | (xc_nodes > side["XT"][1])
        comp = 1 if "lateral" in kind else 2
        eb = ke[cable[back], comp].sum(); em = ke[cable[~back], comp].sum()
        kind = ("side " if eb > em else "main ") + kind
    return kind, sym, share


def main(argv):
    out = pathlib.Path(argv[1]) if len(argv) > 1 and not argv[1].startswith("--") else \
        pathlib.Path("results") / "xihoumen"
    n_modes = int(argv[argv.index("--modes") + 1]) if "--modes" in argv else 40
    out.mkdir(parents=True, exist_ok=True)
    msh = out / "xihoumen.msh"
    w = gx.calibrate(str(msh))
    info, side = gx.build(str(msh), w)
    text = gx.case_yaml(msh.name, side, w, backend="numba")   # beside the yaml
    relax = "--relax" in argv
    if relax:
        # the gravity-equilibrium state: the towers shorten, the girder takes
        # its axial compression from the stays -- the P-delta the stated
        # (u = 0) state leaves out of the tangent
        text = text.replace("loads: []", "loads:\n  - {region: all, gravity: [0, 0, -9.80665]}")
        text = text.replace("steps: [{name: dead, load_steps: 1}]",
                            "steps: [{name: dead, load_steps: 1}]")
        text = text.replace("tolerance: 1.0e-4, n_steps: 1000", "tolerance: 1.0e-5, n_steps: 200000,\n         max_steps_per_load: 100000")
    (out / "xihoumen.yaml").write_text(text, encoding="utf-8")
    prep = prepare_case(str(out / "xihoumen.yaml"))
    sol = make_structural_solver(prep.case.model_dump(mode="python"), prep.mesh, prep.material)
    if relax:
        from vfife.core.diagnostics import ConvergenceFailure
        try:
            sol.run(output_every=10**9, progress_every=10**9)
            print("  relaxed under gravity: converged")
        except ConvergenceFailure as exc:
            print(f"  relaxed under gravity: NOT converged -- {exc}")
        u = sol.dofs.view(sol.q)[:, :3]
        rail = np.array(side["rail"])
        print(f"  max |u| {np.abs(u).max():.3f} m; deck vertical {u[rail, 2].min():.3f} .. "
              f"{u[rail, 2].max():.3f} m; deck longitudinal {u[rail, 0].min():.3f} .. {u[rail, 0].max():.3f} m")
    print(f"  deck dead load    {w/1e3:.1f} kN/m (calibrated to the tower base force)")
    print(f"  model             {sol.nn} nodes, {sol.nel} elements, {sol.ndof} dofs, "
          f"{sol.physical_mass.reshape(-1, 6)[:, 0].sum()/1e6:.1f} kt")
    res = modal.analyse(sol, n_modes=n_modes, max_dofs=6000)
    rows = []
    print(f"\n  {'#':>3s} {'f [Hz]':>8s} {'T [s]':>8s}   {'where the energy is':<28s} {'share':>5s}")
    for i, (f, T) in enumerate(zip(res.frequencies, res.periods)):
        kind, sym, share = classify(sol, side, res.shapes[i])
        rows.append((i + 1, f, T, kind, sym, share))
        print(f"  {i+1:3d} {f:8.4f} {T:8.3f}   {kind + ', ' + sym:<28s} {share:5.2f}")
    np.savez(out / "modes.npz", f=res.frequencies, shapes=res.shapes)
    if "--paraview" in argv:
        from vfife.post import modes_vtk
        modes_vtk.write(sol, res, out / "paraview", stem="xihoumen",
                        labels=[f"{r[3]} {r[4]}" for r in rows])

    # against the papers: the first mode of each kind
    print("\n  [J] Table 4, each paper mode matched to the model's first unused mode of the same kind:")
    used = set()
    for k, (Tj, label) in enumerate(JIA, 1):
        want_kind, want_sym = label.rsplit(", ", 1)
        want_kind = want_kind.replace("girder", "girder").strip()
        cand = [r for r in rows if r[0] not in used and r[3].endswith(want_kind.split(" ", 1)[-1])
                and want_kind.split()[0] in r[3] and r[4] == want_sym]
        if cand:
            r = cand[0]; used.add(r[0])
            print(f"    J{k:<2d} {1/Tj:.4f} Hz {label:<28s} -> model #{r[0]:<3d} {r[1]:.4f} Hz "
                  f"({100*(r[1]*Tj-1):+.1f}%)")
        else:
            print(f"    J{k:<2d} {1/Tj:.4f} Hz {label:<28s} -> no mode of that kind among the first {n_modes}")
    print("\n  [X] 4.1:")
    for label, fx in XIAO.items():
        kind, sym = label.rsplit(", ", 1)
        cand = [r for r in rows if r[3] == kind and r[4] == sym]
        if cand:
            r = cand[0]
            print(f"    {label:<24s} {fx:.3f} Hz -> model #{r[0]:<3d} {r[1]:.4f} Hz ({100*(r[1]/fx-1):+.1f}%)")
        else:
            print(f"    {label:<24s} {fx:.3f} Hz -> none among the first {n_modes}")
    return rows, info


if __name__ == "__main__":
    main(sys.argv)
