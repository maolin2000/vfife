"""Dependency-free SVG plot of any two scalar HDF5 monitors."""
import argparse,html
from pathlib import Path
import h5py,numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument("result"); p.add_argument("x"); p.add_argument("y")
    p.add_argument("-o","--output",required=True); p.add_argument("--xscale",type=float,default=1.); p.add_argument("--yscale",type=float,default=1.)
    p.add_argument("--xlabel",default="x"); p.add_argument("--ylabel",default="y"); p.add_argument("--title",default="Response curve")
    a=p.parse_args()
    with h5py.File(a.result,"r") as f: x=np.asarray(f["monitors"][a.x])*a.xscale; y=np.asarray(f["monitors"][a.y])*a.yscale
    if len(x)<2 or len(x)!=len(y): raise ValueError("two matching monitor histories are required")
    W,H=900,640; L,R,T,B=115,45,55,90; pw,ph=W-L-R,H-T-B
    def domain(v):
        lo,hi=float(v.min()),float(v.max()); pad=.06*(hi-lo or max(abs(lo),1)); return lo-pad,hi+pad
    x0,x1=domain(x); y0,y1=domain(y); sx=lambda v:L+(v-x0)/(x1-x0)*pw; sy=lambda v:T+(y1-v)/(y1-y0)*ph
    path=" ".join(("M" if i==0 else "L")+f" {sx(xv):.3f} {sy(yv):.3f}" for i,(xv,yv) in enumerate(zip(x,y)))
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><style>text{{font-family:Arial;fill:#1f2937}}.l{{font-size:18px}}.t{{font-size:21px;font-weight:bold}}</style><text x="{W/2}" y="30" text-anchor="middle" class="t">{html.escape(a.title)}</text><rect x="{L}" y="{T}" width="{pw}" height="{ph}" fill="none" stroke="#4b5563"/><path d="{path}" fill="none" stroke="#1565c0" stroke-width="2.2"/><text x="{L+pw/2}" y="{H-25}" text-anchor="middle" class="l">{html.escape(a.xlabel)}</text><text x="28" y="{T+ph/2}" text-anchor="middle" class="l" transform="rotate(-90 28 {T+ph/2})">{html.escape(a.ylabel)}</text><text x="{L}" y="{H-55}">{x0:.3g}</text><text x="{L+pw}" y="{H-55}" text-anchor="end">{x1:.3g}</text><text x="{L-10}" y="{T+5}" text-anchor="end">{y1:.3g}</text><text x="{L-10}" y="{T+ph}" text-anchor="end">{y0:.3g}</text></svg>'''
    Path(a.output).write_text(svg,encoding="utf-8"); print(a.output)
if __name__=="__main__": main()
