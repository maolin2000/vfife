"""Write an SVG shear--drift hysteresis plot from a vfife result file.

The implementation intentionally uses only h5py and the standard library so a
benchmark run does not acquire a plotting dependency just to leave behind its
most useful engineering diagnostic.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import h5py
import numpy as np


def _domain(values: np.ndarray) -> tuple[float, float]:
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo
    pad = 0.05 * span if span else max(abs(lo) * 0.05, 1.0)
    return lo - pad, hi + pad


def render(result: str | Path, output: str | Path, title: str | None = None) -> Path:
    """Render the ``drift``/``shear`` monitors in *result* as an SVG file."""
    result, output = Path(result), Path(output)
    with h5py.File(result, "r") as h5:
        monitors = h5["monitors"]
        drift = np.asarray(monitors["drift"], dtype=float)
        shear = np.asarray(monitors["shear"], dtype=float)

    if drift.size != shear.size or drift.size < 2:
        raise ValueError("hysteresis plot needs at least two matching drift and shear samples")
    finite = np.isfinite(drift) & np.isfinite(shear)
    if finite.sum() < 2:
        raise ValueError("hysteresis monitors contain fewer than two finite samples")
    # A failed relaxation step can remain numerically finite while its reaction
    # grows by dozens of orders of magnitude.  Such a value would flatten every
    # valid loop point on the plot, so retain the stable prefix and label it as
    # truncated.  The HDF5 data is intentionally left untouched for diagnosis.
    stable = finite.copy()
    finite_shear = np.abs(shear[finite])
    reference = max(float(np.median(finite_shear)), 1.0)
    stable &= np.abs(shear) <= 100.0 * reference
    if stable.sum() < 2:
        stable = finite
    truncated = not bool(stable.all())
    drift, shear = drift[stable], shear[stable]

    # Engineering display units, while the HDF5 file remains in SI units.
    x, y = drift * 1e3, shear * 1e-3
    x0, x1 = _domain(x)
    y0, y1 = _domain(y)
    width, height = 900, 640
    left, right, top, bottom = 115, 45, 55, 95
    plot_w, plot_h = width - left - right, height - top - bottom

    def sx(v):
        return left + (v - x0) / (x1 - x0) * plot_w

    def sy(v):
        return top + (y1 - v) / (y1 - y0) * plot_h

    path = " ".join(
        ("M" if i == 0 else "L") + f" {sx(xv):.3f} {sy(yv):.3f}"
        for i, (xv, yv) in enumerate(zip(x, y))
    )
    zero_x = sx(0.0) if x0 <= 0.0 <= x1 else None
    zero_y = sy(0.0) if y0 <= 0.0 <= y1 else None
    caption = html.escape(title or result.stem)
    if truncated:
        caption += " (finite prefix; run ended non-finite)"
    axes = []
    if zero_x is not None:
        axes.append(f'<line x1="{zero_x:.3f}" y1="{top}" x2="{zero_x:.3f}" y2="{top + plot_h}" class="zero"/>')
    if zero_y is not None:
        axes.append(f'<line x1="{left}" y1="{zero_y:.3f}" x2="{left + plot_w}" y2="{zero_y:.3f}" class="zero"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Shear drift hysteresis">
<style>
  text {{ font-family: Arial, sans-serif; fill: #1f2937; }}
  .frame {{ fill: none; stroke: #4b5563; stroke-width: 1.2; }}
  .zero {{ stroke: #9ca3af; stroke-width: 1; }}
  .curve {{ fill: none; stroke: #1565c0; stroke-width: 2.2; stroke-linejoin: round; stroke-linecap: round; }}
  .label {{ font-size: 18px; }} .title {{ font-size: 21px; font-weight: bold; }} .tick {{ font-size: 14px; }}
</style>
<title>{caption}</title>
<text x="{width / 2:.0f}" y="30" text-anchor="middle" class="title">{caption}</text>
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="frame"/>
{''.join(axes)}
<path d="{path}" class="curve"/>
<text x="{left + plot_w / 2:.0f}" y="{height - 30}" text-anchor="middle" class="label">Drift (mm)</text>
<text x="30" y="{top + plot_h / 2:.0f}" text-anchor="middle" class="label" transform="rotate(-90 30 {top + plot_h / 2:.0f})">Base shear (kN)</text>
<text x="{left}" y="{height - 62}" class="tick">{x0:.3g}</text>
<text x="{left + plot_w}" y="{height - 62}" text-anchor="end" class="tick">{x1:.3g}</text>
<text x="{left - 12}" y="{top + 5}" text-anchor="end" class="tick">{y1:.3g}</text>
<text x="{left - 12}" y="{top + plot_h}" text-anchor="end" class="tick">{y0:.3g}</text>
</svg>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return output


def main() -> int:
    ap = argparse.ArgumentParser(description="plot vfife drift/shear monitors as SVG")
    ap.add_argument("result", help="HDF5 result containing monitors/drift and monitors/shear")
    ap.add_argument("-o", "--output", help="SVG path; defaults beside the result")
    ap.add_argument("--title")
    args = ap.parse_args()
    result = Path(args.result)
    output = Path(args.output) if args.output else result.with_name(result.stem + "_hysteresis.svg")
    print(render(result, output, args.title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
