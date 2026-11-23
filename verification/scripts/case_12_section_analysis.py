"""Section analysis of the case_12 beam with the case's own uniaxial laws.

WHY THIS FILE IS IN verification/
    The solid model of case_12 peaks at 276-304 kN and the beam at 399. The
    question that decides where to look is whether the SECTION, with exactly
    the materials the case uses, can carry the beam. It can:

        GB50010 compression, alpha_c = 2.16 (the case)   M = 101.7 kN.m -> P = 370 kN
        alpha_c = 0.5 (a confined, ductile descent)      M = 110.3        -> P = 401
        measured                                                            399

    so the shortfall is in three dimensions, not in the material curves. The
    contract for case_12 carries the rest of that argument. Run it:

        python verification/scripts/case_12_section_analysis.py

    Layers of concrete (300 over the depth), the same five 15M bars in two
    layers and the two 10M hangers as the case, plane sections, no tension
    in the concrete (the crack-band tension of the solid model contributes
    under 10% of M at yield and nothing at the peak). Equilibrium is found by
    bisection on the neutral axis at each curvature.
"""
import numpy as np
b,h=0.200,0.300; fc=44e6; Ec=4500*np.sqrt(44)*1e6
def gb(eps, alpha_c=None):
    fc_M=fc/1e6; E_M=Ec/1e6
    eps_cr=(700+172*np.sqrt(fc_M))*1e-6
    if alpha_c is None: alpha_c=0.157*fc_M**0.785-0.905
    rho=fc_M/(E_M*eps_cr); n=1/(1-rho)
    x=np.maximum(eps,0)/eps_cr
    d=np.where(x<=1, 1-rho*n/(n-1+x**n), 1-rho/(alpha_c*np.maximum(x-1,0)**2+x))
    return (1-d)*Ec*np.maximum(eps,0)
bars=[(0.0443,3*200e-6),(0.0853,2*200e-6),(0.3-0.025-0.0113-0.0113/2, 2*100e-6)]  # z从底, 面积
def steel(e,E=192e9,fy=449e6,H=8e9):
    ey=fy/E; a=np.abs(e)
    return np.sign(e)*np.where(a<ey, E*a, fy+H*(a-ey))
def steel10(e): return steel(e,195e9,404e6,7e9)
zc=np.linspace(0,h,300); dz=h/300
def section(kappa, alpha_c, top_lim=None):
    # 找中和轴 c（从顶量），使轴力=0
    def N(c):
        eps_c=kappa*(zc-(h-c))            # 顶部受压为正
        sc=gb(eps_c,alpha_c); Nc=-np.sum(sc)*b*dz
        Ns=0
        for i,(z,A) in enumerate(bars):
            e=-kappa*(z-(h-c))            # 底部受拉为正
            Ns+= (steel10(e) if i==2 else steel(e))*A
        return Nc+Ns
    lo,hi=1e-4,h-1e-4
    for _ in range(60):
        m=0.5*(lo+hi)
        if N(m)>0: lo=m
        else: hi=m
    c=0.5*(lo+hi)
    eps_c=kappa*(zc-(h-c)); sc=-gb(eps_c,alpha_c)        # 应力：拉为正
    M=np.sum(sc*b*dz*(h/2-zc))
    for i,(z,A) in enumerate(bars):
        e=-kappa*(z-(h-c)); M+=(steel10(e) if i==2 else steel(e))*A*(h/2-z)
    return M, c, kappa*c, steel(-kappa*(bars[0][0]-(h-c)))
for ac,label in ((None,"GB50010 α_c=2.16（算例用）"),(0.5,"α_c=0.5"),(0.05,"α_c=0.05（近似平台）")):
    best=(0,0,0,0,0)
    print(f"\n{label}:")
    print("   κ(1/m)   M(kN·m)   P(kN,a=0.55)   c(mm)   ε_top    σ_s底层(MPa)")
    for k in np.concatenate([np.linspace(0.005,0.05,10),np.linspace(0.06,0.3,25)]):
        M,c,et,ss=section(k,ac)
        if M>best[1]: best=(k,M,c,et,ss)
        if abs(k-0.02)<1e-9 or abs(k-0.05)<1e-9 or k in (0.06,0.1,0.14,0.2,0.3): pass
        print(f"   {k:6.3f}   {M/1e3:7.1f}   {2*M/0.55/1e3:8.1f}     {c*1e3:5.1f}   {et:.4f}   {ss/1e6:6.0f}")
    print(f"   峰值: M={best[1]/1e3:.1f} kN·m → P={2*best[1]/0.55/1e3:.1f} kN @ κ={best[0]:.3f}, ε_top={best[3]:.4f}")
