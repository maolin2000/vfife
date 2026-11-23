"""Runner for the pure-bending specimen.

    python verification/scripts/case_12_pure_bending_run.py NX NY NZ THETA TAG [z0=..] [bond=1] [key=value ...]

THETA < 1 is the end-plate rotation in radians (displacement control);
key=value pairs override CDP material parameters (dilation_angle, Kc, ...).
PB_L in the environment sets the specimen length (default 1.0 m).
"""
import sys, time, numpy as np, yaml, os
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2])); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from case_12_pure_bending_specimen import case, L, H
from vfife.api import prepare_case
from vfife.core.solver import Solver
nx,ny,nz=(int(v) for v in sys.argv[1:4]); Mt=float(sys.argv[4]); tag=sys.argv[5] if len(sys.argv)>5 else ""
theta=None
if Mt<1: theta=Mt; Mt=None
over={}
for kv in sys.argv[6:]:
    k,v=kv.split("="); over[k]=float(v)
z0=over.pop('z0',0.15); bond={'type':'bond_ceb','fc':44.0e6,'condition':'good'} if over.pop('bond',0) else None
c=case(nx,ny,nz,f"results/pb_{nx}{ny}{nz}_{tag}.h5",M_target=Mt,load_steps=50,theta=theta,z0=z0,bond=bond,**over)
p=f"cases/_pb_{os.getpid()}.yaml"; yaml.safe_dump(c,open(p,"w",encoding="utf-8"),allow_unicode=True)
try: prep=prepare_case(p)
finally: os.remove(p)
s=Solver(prep.case.model_dump(mode="python"),prep.mesh,prep.material,log=lambda *a: None)
t0=time.perf_counter(); st="完成"
try: s.run(on_progress=lambda *a: None, progress_every=2000)
except Exception as e:
    m=str(e).splitlines()[0]; st=f"{type(e).__name__} {m.split('load step ')[1].split(' ')[0] if 'load step ' in m else ''}"
n=len(s.monitors["mid_bot"].values); steps=np.arange(1,n+1)
mb=np.array(s.monitors["mid_bot"].values)
th=lambda a,b: (np.array(s.monitors[a].values)-np.array(s.monitors[b].values))/H
kap=(th("Lbot","Ltop")+th("Rtop","Rbot"))/L
nz_=nz; zs=np.array([k*H/nz_ for k in range(nz_+1)])
R=np.array([s.monitors[f"RL{k}"].values for k in range(nz_+1)])      # (rows, frames)
N=R.sum(axis=0); M=(R*(zs[:,None]-z0)).sum(axis=0)
print(f"[L={L} {nx}x{ny}x{nz} {tag} z0={z0} bond={bool(bond)} {over}] {st}  收敛 {n}/50 步   峰值 |M|={np.abs(M).max()/1e3:.1f} kN·m（截面分析 101.7；P=2M/0.55={2*np.abs(M).max()/0.55/1e3:.0f} kN）")
for i in range(0,n,max(1,n//16)): print(f"    κ={abs(kap[i]):.4f}  M={abs(M[i])/1e3:6.1f} kN·m  N={N[i]/1e3:7.1f} kN  δ_mid={abs(mb[i])*1e3:.2f} mm")
print(f"    末帧 κ={abs(kap[-1]):.4f} M={abs(M[-1])/1e3:.1f} N={N[-1]/1e3:.1f}  {time.perf_counter()-t0:.0f}s")
