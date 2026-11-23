"""Generate the M02 cyclic constitutive comparison as CSV and dependency-free SVG."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vfife.materials import create


def response(mode, history):
    material = create(
        "piecewise_uniaxial", E=2.0e11, rho=7850.0,
        tension=[[0.0, 2.0e8], [0.005, 2.6e8], [0.02, 3.2e8]],
        compression=[[0.0, 1.8e8], [0.005, 2.4e8], [0.02, 3.0e8]],
        hardening_mode=mode, kinematic_fraction=0.5,
    )
    state = material.init_state(1)
    stress = []
    for value in history:
        strain = np.zeros((1, 6)); strain[0, 0] = value
        stress.append(material.update(strain, state, 1.0)[0, 0])
    return np.asarray(stress), material.output_fields(state)


def render(path, history, curves):
    width, height = 900, 640
    left, right, top, bottom = 115, 45, 60, 90
    pw, ph = width-left-right, height-top-bottom
    x = history*1e3
    all_y = np.concatenate(list(curves.values()))/1e6
    x0, x1 = 1.08*x.min(), 1.08*x.max()
    y0, y1 = 1.08*all_y.min(), 1.08*all_y.max()
    sx = lambda v: left+(v-x0)/(x1-x0)*pw
    sy = lambda v: top+(y1-v)/(y1-y0)*ph
    colours = {"isotropic": "#2563eb", "combined": "#111827", "kinematic": "#dc2626"}
    paths = []
    for name, values in curves.items():
        points = " ".join(("M" if i == 0 else "L")+f" {sx(xv):.3f} {sy(yv/1e6):.3f}"
                          for i, (xv, yv) in enumerate(zip(x, values)))
        paths.append(f'<path d="{points}" fill="none" stroke="{colours[name]}" stroke-width="2"/>' )
    legend = "".join(
        f'<line x1="{left+18+i*190}" y1="{top+18}" x2="{left+53+i*190}" y2="{top+18}" stroke="{colours[n]}" stroke-width="3"/>'
        f'<text x="{left+61+i*190}" y="{top+24}" class="tick">{n}</text>'
        for i, n in enumerate(curves)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>text{{font-family:Arial,sans-serif;fill:#1f2937}}.tick{{font-size:14px}}.label{{font-size:18px}}.title{{font-size:21px;font-weight:bold}}</style>
<text x="{width/2}" y="30" text-anchor="middle" class="title">M02 asymmetric piecewise-linear cyclic response</text>
<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#4b5563"/>
<line x1="{sx(0)}" y1="{top}" x2="{sx(0)}" y2="{top+ph}" stroke="#9ca3af"/>
<line x1="{left}" y1="{sy(0)}" x2="{left+pw}" y2="{sy(0)}" stroke="#9ca3af"/>
{''.join(paths)}{legend}
<text x="{left+pw/2}" y="{height-25}" text-anchor="middle" class="label">Axial strain (10^-3)</text>
<text x="28" y="{top+ph/2}" text-anchor="middle" class="label" transform="rotate(-90 28 {top+ph/2})">Axial stress (MPa)</text>
<text x="{left}" y="{height-55}" class="tick">{x0:.3g}</text><text x="{left+pw}" y="{height-55}" text-anchor="end" class="tick">{x1:.3g}</text>
<text x="{left-10}" y="{top+5}" text-anchor="end" class="tick">{y1:.3g}</text><text x="{left-10}" y="{top+ph}" text-anchor="end" class="tick">{y0:.3g}</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/m02_piecewise_uniaxial")
    args = parser.parse_args()
    prefix = Path(args.output); prefix.parent.mkdir(parents=True, exist_ok=True)
    turns = [0.0, 0.004, -0.004, 0.008, -0.008, 0.0]
    history = np.concatenate([np.linspace(a, b, 41, endpoint=False)
                              for a, b in zip(turns[:-1], turns[1:])] + [[turns[-1]]])
    curves, energies = {}, {}
    for mode in ("isotropic", "combined", "kinematic"):
        curves[mode], fields = response(mode, history)
        energies[mode] = fields["hysteretic_energy"][0]
    np.savetxt(prefix.with_suffix(".csv"),
               np.column_stack([history]+[curves[k] for k in curves]), delimiter=",",
               header="strain,stress_isotropic_Pa,stress_combined_Pa,stress_kinematic_Pa", comments="", encoding="utf-8")
    render(prefix.with_suffix(".svg"), history, curves)
    for name, value in energies.items():
        print(f"{name:10s} hysteretic_energy = {value:.6e} J/m^3")
    print(prefix.with_suffix(".svg"))


if __name__ == "__main__":
    main()
