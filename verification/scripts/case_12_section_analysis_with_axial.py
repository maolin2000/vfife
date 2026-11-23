"""Section analysis with axial force: M at a given (kappa, N), for comparing the
pure-bending solid specimen (case_12_pure_bending_specimen.py) at the same state.
The rows below are the z0 = 0.15 run; v0.1 has the z0 = 0.21 rows."""
import numpy as np
b,h=0.200,0.300; fc=44e6; Ec=4500*np.sqrt(44)*1e6
def gb(eps, alpha_c=None):
    fc_M=fc/1e6; E_M=Ec/1e6; eps_cr=(700+172*np.sqrt(fc_M))*1e-6
    if alpha_c is None: alpha_c=0.157*fc_M**0.785-0.905
    rho=fc_M/(E_M*eps_cr); n=1/(1-rho); x=np.maximum(eps,0)/eps_cr
    with np.errstate(divide="ignore"):
        d=np.where(x<=1, 1-rho*n/(n-1+x**n), 1-rho/(alpha_c*np.maximum(x-1,0)**2+x))
    return (1-d)*Ec*np.maximum(eps,0)
bars=[(0.0443,3*200e-6),(0.0853,2*200e-6),(0.3-0.025-0.0113-0.0113/2,2*100e-6)]
def steel(e,E=192e9,fy=449e6,H=8e9):
    ey=fy/E; a=np.abs(e); return np.sign(e)*np.where(a<ey,E*a,fy+H*(a-ey))
def steel10(e): return steel(e,195e9,404e6,7e9)
zc=np.linspace(0,h,300); dz=h/300
def NM(kappa,c):
    eps_c=kappa*(zc-(h-c)); sc=-gb(eps_c)      # 拉为正
    N=np.sum(sc)*b*dz; M=np.sum(sc*b*dz*(h/2-zc))
    for i,(z,A) in enumerate(bars):
        e=-kappa*(z-(h-c)); sg=(steel10(e) if i==2 else steel(e)); N+=sg*A; M+=sg*A*(h/2-z)
    return N,M
def M_at(kappa,N_target):
    lo,hi=1e-4,h-1e-4
    for _ in range(80):
        m=0.5*(lo+hi)
        if NM(kappa,m)[0]>N_target: lo=m       # N 随 c 增大而更压（更负）
        else: hi=m
    return NM(kappa,0.5*(lo+hi))[1]
rows=[(0.0020,24.2,35.3),(0.0040,40.8,147.9),(0.0060,57.0,263.6),(0.0080,72.6,373.8),(0.0100,86.4,467.3),
      (0.0120,96.1,522.9),(0.0140,99.9,519.0),(0.0160,100.4,480.8),(0.0180,100.1,428.4),(0.0200,82.3,254.8)]
print("   κ       N(kN,压)   M_模型   M_截面(同κ,同N)   比值")
for k,Mm,Nc in rows:
    Ms=M_at(k,-Nc*1e3)/1e3
    print(f"  {k:.4f}   {Nc:7.1f}   {Mm:6.1f}     {Ms:6.1f}          {Mm/Ms:.3f}")
print("\n对照：N=0 时截面分析", [f"κ={k}: {M_at(k,0.0)/1e3:.1f}" for k in (0.01,0.02,0.03,0.045)])
