"""Four-node flat Mindlin--Reissner shell reference kernel.

The element uses full 2x2 integration for membrane/bending, selective one-point
integration for transverse shear, an energy-scaled transverse hourglass mode,
and a drilling constraint based on the physical in-plane continuum spin.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Shell4Data:
    frame: np.ndarray
    centroid: np.ndarray
    local_nodes: np.ndarray
    stiffness: np.ndarray
    area: float
    membrane: np.ndarray
    bending: np.ndarray
    shear: np.ndarray


_SIGNS = np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])


def shape(xi, eta):
    N=.25*(1+_SIGNS[:,0]*xi)*(1+_SIGNS[:,1]*eta)
    d=np.empty((2,4))
    d[0]=.25*_SIGNS[:,0]*(1+_SIGNS[:,1]*eta)
    d[1]=.25*_SIGNS[:,1]*(1+_SIGNS[:,0]*xi)
    return N,d


def frame(nodes, warpage_tolerance=0.05):
    x=np.asarray(nodes,float); c=x.mean(axis=0)
    ex=(x[1]-x[0])+(x[2]-x[3]); nx=np.linalg.norm(ex)
    normal=np.cross(x[2]-x[0],x[3]-x[1]); nn=np.linalg.norm(normal)
    if nx<=1e-14 or nn<=1e-14: raise ValueError("shell4 has a degenerate midsurface")
    ex/=nx; ez=normal/nn; ey=np.cross(ez,ex); ey/=np.linalg.norm(ey)
    R=np.vstack((ex,ey,ez)); local=(R@(x-c).T).T
    span=max(np.ptp(local[:,0]),np.ptp(local[:,1]),1e-30)
    if np.ptp(local[:,2])>float(warpage_tolerance)*span:
        raise ValueError("shell4 reference face is too warped; refine the surface mesh")
    local[:,2]=0.0
    return R,c,local


def _matrices(xy,E,nu,t,kappa,drilling_factor,hourglass_coeff):
    C=E/(1-nu*nu)*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    Dm=t*C; Db=t**3*C/12; G=E/(2*(1+nu)); Ds=kappa*G*t*np.eye(2)
    K=np.zeros((24,24)); shear_full=np.zeros((24,24)); area=0.0
    gp=1/np.sqrt(3)
    for xi in (-gp,gp):
        for eta in (-gp,gp):
            N,dn=shape(xi,eta); J=dn@xy[:,:2]; det=np.linalg.det(J)
            if det<=1e-14: raise ValueError("shell4 has a non-positive in-plane Jacobian")
            dxy=np.linalg.solve(J,dn); Bm=np.zeros((3,24)); Bb=np.zeros((3,24)); Bd=np.zeros((1,24))
            for a in range(4):
                i=6*a; dx,dy=dxy[:,a]
                Bm[:,i:i+2]=[[dx,0],[0,dy],[dy,dx]]
                Bb[:,i+3:i+5]=[[0,dx],[-dy,0],[-dx,dy]]
                # rz equals the continuum spin 0.5*(dv/dx-du/dy).
                Bd[0,i]=.5*dy; Bd[0,i+1]=-.5*dx; Bd[0,i+5]=N[a]
            Bs_full=np.zeros((2,24))
            for a in range(4):
                i=6*a; dx,dy=dxy[:,a]
                Bs_full[:,i:i+6]=[[0,0,dx,0,N[a],0],[0,0,dy,-N[a],0,0]]
            K+=(Bm.T@Dm@Bm+Bb.T@Db@Bb+drilling_factor*G*t*(Bd.T@Bd))*det
            shear_full+=Bs_full.T@Ds@Bs_full*det
            area+=det
    N,dn=shape(0,0); J=dn@xy[:,:2]; det=np.linalg.det(J); dxy=np.linalg.solve(J,dn)
    Bs=np.zeros((2,24))
    for a in range(4):
        i=6*a; dx,dy=dxy[:,a]
        Bs[:,i:i+6]=[[0,0,dx,0,N[a],0],[0,0,dy,-N[a],0,0]]
    shear_reduced=Bs.T@Ds@Bs*(4*det)
    # Stabilised selective integration: only the full-minus-reduced shear
    # modes are added, scaled by t²/A.  Their stiffness is therefore O(E t³)
    # like bending, eliminating zero-energy modes without reintroducing the
    # O(E t) thin-shell shear locking of ordinary full integration.
    scale=hourglass_coeff*min(1.0,t*t/max(area,1e-30))
    K+=shear_reduced+scale*(shear_full-shear_reduced)
    return .5*(K+K.T),area,Dm,Db,Ds


def precompute(nodes,E,nu,thickness,kappa=5/6,drilling_factor=1.0,
               hourglass_coeff=0.1,warpage_tolerance=0.05):
    E=float(E); nu=float(nu); t=float(thickness)
    if E<=0 or not -1<nu<.5 or t<=0: raise ValueError("shell4 needs E>0, -1<nu<0.5 and thickness>0")
    if kappa<=0 or drilling_factor<=0 or hourglass_coeff<0:
        raise ValueError("shell4 kappa/drilling_factor must be positive and hourglass_coeff non-negative")
    R,c,local=frame(nodes,warpage_tolerance)
    K,A,Dm,Db,Ds=_matrices(local,E,nu,t,kappa,drilling_factor,hourglass_coeff)
    return Shell4Data(R,c,local,K,A,Dm,Db,Ds)


def resultants(data, local_q):
    N,dn=shape(0,0); J=dn@data.local_nodes[:,:2]; dxy=np.linalg.solve(J,dn)
    Bm=np.zeros((3,24)); Bb=np.zeros((3,24)); Bs=np.zeros((2,24))
    for a in range(4):
        i=6*a; dx,dy=dxy[:,a]
        Bm[:,i:i+2]=[[dx,0],[0,dy],[dy,dx]]
        Bb[:,i+3:i+5]=[[0,dx],[-dy,0],[-dx,dy]]
        Bs[:,i:i+6]=[[0,0,dx,0,N[a],0],[0,0,dy,-N[a],0,0]]
    return np.concatenate((data.membrane@(Bm@local_q),data.bending@(Bb@local_q),data.shear@(Bs@local_q)))

# --------------------------------------------------------------------------- #
#  shell3 -- 3-node flat Mindlin--Reissner (constant strain triangle)
# --------------------------------------------------------------------------- #

@dataclass
class Shell3Data:
    frame: np.ndarray
    centroid: np.ndarray
    local_nodes: np.ndarray
    stiffness: np.ndarray
    area: float
    membrane: np.ndarray
    bending: np.ndarray
    shear: np.ndarray

def shape_tri(xi, eta):
    N=np.array([1-xi-eta, xi, eta], dtype=float)
    d=np.array([[-1.,1.,0.],[-1.,0.,1.]])
    return N,d

def frame_tri(nodes, warpage_tolerance=0.05):
    x=np.asarray(nodes,float); c=x.mean(axis=0)
    ex=x[1]-x[0]; nx=np.linalg.norm(ex)
    normal=np.cross(x[1]-x[0], x[2]-x[0]); nn=np.linalg.norm(normal)
    if nx<=1e-14 or nn<=1e-14: raise ValueError("shell3 has a degenerate midsurface")
    ex/=nx; ez=normal/nn; ey=np.cross(ez,ex); ey/=np.linalg.norm(ey)
    R=np.vstack((ex,ey,ez)); local=(R@(x-c).T).T
    local[:,2]=0.0
    return R,c,local

def _matrices_tri(xy,E,nu,t,kappa,drilling_factor):
    C=E/(1-nu*nu)*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    Dm=t*C; Db=t**3*C/12; G=E/(2*(1+nu)); Ds=kappa*G*t*np.eye(2)
    N,d = shape_tri(1/3,1/3)
    J=d@xy[:,:2]; det=np.linalg.det(J)
    if det<=1e-14: raise ValueError("shell3 has a non-positive in-plane Jacobian")
    dxy=np.linalg.solve(J,d); area=0.5*det
    K=np.zeros((18,18))
    Bm=np.zeros((3,18)); Bb=np.zeros((3,18)); Bd=np.zeros((1,18))
    for a in range(3):
        i=6*a; dx,dy=dxy[:,a]
        Bm[:,i:i+2]=[[dx,0],[0,dy],[dy,dx]]
        Bb[:,i+3:i+5]=[[0,dx],[-dy,0],[-dx,dy]]
        Bd[0,i]=.5*dy; Bd[0,i+1]=-.5*dx; Bd[0,i+5]=N[a]
    Bs=np.zeros((2,18))
    for a in range(3):
        i=6*a; dx,dy=dxy[:,a]
        Bs[:,i:i+6]=[[0,0,dx,0,N[a],0],[0,0,dy,-N[a],0,0]]
    K=(Bm.T@Dm@Bm+Bb.T@Db@Bb+drilling_factor*G*t*(Bd.T@Bd)+Bs.T@Ds@Bs)*det*0.5
    # det*0.5 == area, already included as factor ; keep as area scaling
    return 0.5*(K+K.T),area,Dm,Db,Ds

def precompute_tri(nodes,E,nu,thickness,kappa=5/6,drilling_factor=1.0,warpage_tolerance=0.05):
    E=float(E); nu=float(nu); t=float(thickness)
    if E<=0 or not -1<nu<.5 or t<=0: raise ValueError("shell3 needs E>0, -1<nu<0.5 and thickness>0")
    if kappa<=0 or drilling_factor<=0: raise ValueError("shell3 kappa/drilling must be positive")
    R,c,local=frame_tri(nodes,warpage_tolerance)
    K,A,Dm,Db,Ds=_matrices_tri(local,E,nu,t,kappa,drilling_factor)
    return Shell3Data(R,c,local,K,A,Dm,Db,Ds)

def resultants_tri(data, local_q):
    N,d=shape_tri(1/3,1/3); J=d@data.local_nodes[:,:2]; dxy=np.linalg.solve(J,d)
    Bm=np.zeros((3,18)); Bb=np.zeros((3,18)); Bs=np.zeros((2,18))
    for a in range(3):
        i=6*a; dx,dy=dxy[:,a]
        Bm[:,i:i+2]=[[dx,0],[0,dy],[dy,dx]]
        Bb[:,i+3:i+5]=[[0,dx],[-dy,0],[-dx,dy]]
        Bs[:,i:i+6]=[[0,0,dx,0,N[a],0],[0,0,dy,-N[a],0,0]]
    return np.concatenate((data.membrane@(Bm@local_q),data.bending@(Bb@local_q),data.shear@(Bs@local_q)))
