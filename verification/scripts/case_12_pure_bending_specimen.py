"""The case_12 section in PURE BENDING, as a solid: no supports, no load pads.

WHY
    Every explanation of case_12's post-yield ceiling that lived at the
    supports or the load points -- knife-edge bearing, pad clamping -- had to
    be separated from the behaviour of the CDP solid itself. This specimen
    has neither: the same 200x300 section, bars, stirrups and material, 1 m
    long, bent by rigid end plates rotated about z0 (displacement control;
    force-controlled couples cannot pass first cracking under kinetic
    damping -- measured, NaN at M = 15 kN.m).

    A rigid plate pinned at z0 also restrains the axial elongation of a
    cracked section, so an axial force N appears. z0 = 0.21 m keeps |N|
    under 60 kN to the peak; the comparison is made against a section
    analysis AT THE SAME (kappa, N) (case_12_section_analysis_with_axial.py):

        kappa    N (kN)   M_solid   M_section   ratio
        0.010    -58       59.6      55.8        1.07
        0.014    -52       78.2      79.5        0.98
        0.016    -24       81.3      86.6        0.94     <- peak of the solid
        0.018    108       70.7     101.5        0.70     <- collapse

    The solid follows plane sections to 2% until kappa = 0.014, peaks at
    0.016 and collapses at 0.018, where the section analysis is still on its
    ascending branch and goes on to 101.7 kN.m at kappa = 0.045. The peak of
    81.3 kN.m is 296 kN in the beam's rig -- the ceiling every beam run hit.
    The wall is the CDP hinge in three dimensions, not the rig.

    Reuses the reinforcement layout of the case_12 generator; the constants it
    needs are inlined below. NO host_deduction: with a cracking host it turns
    the bars' post-yield tangent negative, and with it in place
    this specimen peaked at 80.7 kN.m and collapsed at kappa = 0.017; without
    it, 100.3 against a plane-section 101.7.
"""
import sys, yaml
class G:      # constants from the case_12 generator
    B,H=0.200,0.300; COVER,D_STIR,D15,D10=0.025,0.0113,0.016,0.0113
    Z1=COVER+D_STIR+D15/2; Z2=Z1+D15+0.025; YL,YR,YM=Z1,B-Z1,B/2; ZTOP=H-COVER-D_STIR-D10/2
import os
L=float(os.environ.get('PB_L','1.0')); B,H=G.B,G.H
def case(nx,ny,nz,out,M_target=130e3,load_steps=40,theta=None,z0=0.15,bond=None,**mat_over):
    Ec=4500.0*44**0.5*1e6
    def line(y,z,seg=0.025): return {"start":[0.0,y,z],"end":[L,y,z],"segment_length":seg}
    tension=[line(G.YL,G.Z1),line(G.YM,G.Z1),line(G.YR,G.Z1),line(G.YL,G.Z2),line(G.YR,G.Z2)]
    hangers=[line(G.YL,G.ZTOP),line(G.YR,G.ZTOP)]
    n_st=int(round((L-2*0.05)/0.075))+1
    stir=[{"start":[0.05,y,G.Z1],"end":[0.05,y,G.ZTOP],"divisions":8,
           "repeat":{"count":n_st-1,"offset":[0.075,0,0]}} for y in (G.YL,G.YR)]
    steel15={"type":"rebar","E":192.0e9,"rho":7850.0,"yield_strength":449.0e6,"hardening":8.0e9,"kinematic_fraction":0.5}
    steel10={"type":"rebar","E":195.0e9,"rho":7850.0,"yield_strength":404.0e6,"hardening":7.0e9,"kinematic_fraction":0.5}
    rein=[{"name":"tension","area":200.0e-6,"diameter":G.D15,"material":steel15,"lines":tension,
           **({"bond":bond} if bond else {})},
          {"name":"hangers","area":100.0e-6,"material":steel10,"lines":hangers},
          {"name":"stirrups","area":100.0e-6,"material":steel10,"lines":stir}]
    regions={}; loads=[]
    dz=H/nz; zs=[k*dz for k in range(nz+1)]
    w=[(z-H/2)*(0.5 if k in (0,nz) else 1.0) for k,z in enumerate(zs)]
    norm=sum(wk*(z-H/2) for wk,z in zip(w,zs))         # Σ F_k (z_k−H/2) = M  →  F_k = M w_k / norm
    cons=[]
    for end,x in (("L",0.0),("R",L)):
        for k,z in enumerate(zs):
            name=f"{end}{k}"
            regions[name]={"box":[x-0.001,x+0.001,-0.01,B+0.01,z-0.001,z+0.001]}
            if theta is None:
                Fx=M_target*w[k]/norm*(+1 if end=="L" else -1)
                loads.append({"region":name,"vector":[Fx,0.0,0.0]})
            else:
                # 刚性端板绕 (z0) 转动 θ：正弯矩时左端面上部向 +x（压入梁内）、下部向 −x；右端相反
                ux=theta*(z-z0)*(+1 if end=="L" else -1)
                cons.append({"region":name,"dofs":[1,0,0],"value":ux})
    regions["A"]={"box":[-0.001,0.001,B/2-0.026,B/2+0.026,-0.001,0.001]}
    regions["Bn"]={"box":[L-0.001,L+0.001,B/2-0.026,B/2+0.026,-0.001,0.001]}
    regions["C"]={"box":[L/2-0.045,L/2+0.045,B/2-0.026,B/2+0.026,H-0.001,H+0.001]}
    regions["mid_bot"]={"box":[L/2-0.045,L/2+0.045,-0.01,B+0.01,-0.001,0.001]}
    regions["mid_top"]={"box":[L/2-0.045,L/2+0.045,-0.01,B+0.01,H-0.001,H+0.001]}
    regions["Ltop"]={"box":[-0.001,0.001,-0.01,B+0.01,H-0.001,H+0.001]}
    regions["Lbot"]={"box":[-0.001,0.001,-0.01,B+0.01,-0.001,0.001]}
    regions["Rtop"]={"box":[L-0.001,L+0.001,-0.01,B+0.01,H-0.001,H+0.001]}
    regions["Rbot"]={"box":[L-0.001,L+0.001,-0.01,B+0.01,-0.001,0.001]}
    material={"type":"cdp","E":Ec,"nu":0.2,"rho":2400.0,"fb0_fc0":1.16,"Kc":0.6667,
              "dilation_angle":36.0,"eccentricity":0.1,"wt":0.0,"wc":1.0,"viscosity":1.0,"n_return":5,
              "crack_band":"both","l_ref":"auto","damage_max":0.80,
              "compression":{"source":"gb50010","fc":44e6,"beta_c":0.7,"sigma_c0_ratio":0.4},
              "tension":{"source":"fracture_energy","ft":5.7e6*0.6,"Gf":140.0,"beta_t":0.9}}
    for k,v in mat_over.items():
        if k=="alpha_c": material["compression"]["alpha_c"]=v
        else: material[k]=v
    mons={"mid_bot":{"kind":"displacement","region":"mid_bot","dof":"z"},
          "mid_top":{"kind":"displacement","region":"mid_top","dof":"z"}}
    for r in ("Ltop","Lbot","Rtop","Rbot"): mons[r]={"kind":"displacement","region":r,"dof":"x"}
    if theta is not None:
        for k in range(nz+1): mons[f"RL{k}"]={"kind":"reaction","region":f"L{k}","dof":"x"}
    return {"name":"pure_bending","units":{"length":"m","force":"N","stress":"Pa"},
            "mesh":{"dimensions":[L,B,H],"divisions":[nx,ny,nz]},"element":{"formulation":"b-bar"},
            "material":material,"reinforcement":rein,"regions":regions,
            "constraints":[{"region":"A","dofs":[0,1,1]},{"region":"Bn","dofs":[0,1,1]},{"region":"C","dofs":[0,1,0]}],
            "steps":[{"name":"bend",**({"loads":loads} if theta is None else {"constraints":cons}),
                      "load_steps":load_steps,"max_steps_per_load":20000,"tolerance":1.0e-3}],
            "solver":{"backend":"numba","mode":"relaxation","mass":"underwood","n_steps":load_steps*20000+10,
                      "tolerance":1.0e-3,"relaxation_safety":0.3,"damping":{"type":"kinetic"}},
            "output":{"path":out,"every":500,"monitors":mons}}
