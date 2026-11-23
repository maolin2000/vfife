"""E03 Shell4 mechanics, co-rotation, import and loading contracts."""
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import eigsh, spsolve

from vfife.core.mesh import Mesh
from vfife.core.shell_solver import ShellSolver
from vfife.core.shell_line_solver import ShellLineSolver
from vfife.elements import shell
from vfife.materials import create
from vfife.io import gmsh


E, NU, T = 210e9, .3, .1
NODES = np.array([[0.,0,0],[2,0,0],[2,1,0],[0,1,0]])


def problem(mode="relaxation"):
    return {
        "element":{"type":"shell4","formulation":"selective","thickness":T,
                   "shear_correction":5/6,"drilling_factor":1.,"hourglass_coeff":.1,
                   "warpage_tolerance":.05},
        "solver":{"backend":"numpy","mode":mode,"n_steps":10,
                  "damping":{"type":"kinetic"},"tolerance":1e-5,"kinetic_tolerance":1e-10},
        "constraints":[],"loads":[],"regions":{},"output":{"monitors":{}},
    }


def plate(n,thickness=.01):
    axis=np.linspace(0,1,n+1); nodes=np.array([[x,y,0.] for y in axis for x in axis]); conn=[]
    for j in range(n):
        for i in range(n):
            a=j*(n+1)+i; conn.append([a,a+1,a+n+2,a+n+1])
    p=problem(); p["element"]["thickness"]=thickness
    return Mesh(nodes,np.asarray(conn,np.int32),{"all":np.arange(len(nodes))}),p


def test_shell4_has_exactly_six_rigid_modes_and_no_hourglass_mechanism():
    data=shell.precompute(NODES,E,NU,T)
    eig=np.linalg.eigvalsh(data.stiffness); scale=eig[-1]
    assert np.count_nonzero(np.abs(eig)<scale*1e-10)==6
    assert eig[6]>0


def test_membrane_and_bending_constant_patch_resultants_are_exact():
    data=shell.precompute(NODES,E,NU,T); q=np.zeros((4,6))
    ex,ey,gxy=1e-4,-2e-4,3e-4
    x,y=data.local_nodes[:,0],data.local_nodes[:,1]
    q[:,0]=ex*x+.5*gxy*y; q[:,1]=ey*y+.5*gxy*x
    got=shell.resultants(data,q.ravel())
    C=E/(1-NU**2)*np.array([[1,NU,0],[NU,1,0],[0,0,(1-NU)/2]])
    assert got[:3]==pytest.approx(T*C@np.array([ex,ey,gxy]),rel=2e-13)
    q.fill(0); kx,ky,kxy=2e-3,-1e-3,3e-3
    q[:,4]=kx*x+.5*kxy*y; q[:,3]=-ky*y-.5*kxy*x
    got=shell.resultants(data,q.ravel())
    assert got[3:6]==pytest.approx(T**3/12*C@np.array([kx,ky,kxy]),rel=2e-13)


def test_finite_rigid_translation_and_rotation_produce_no_internal_force():
    mesh=Mesh(NODES.copy(),np.array([[0,1,2,3]],np.int32),{"all":np.arange(4)})
    solver=ShellSolver(problem(),mesh,create("linear_elastic",E=E,nu=NU,rho=7850))
    a=.63; axis=np.array([.2,.4,.7]); axis/=np.linalg.norm(axis)
    A=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    Q=np.eye(3)+np.sin(a)*A+(1-np.cos(a))*(A@A); shift=np.array([3.,-2.,1.])
    view=solver.q.reshape(4,6); view[:,:3]=(Q@NODES.T).T+shift-NODES; view[:,3:]=a*axis
    solver.sync_orientation()                       # the state is the orientation
    force,_=solver.internal_force()
    # 2e-4 was the residue of subtracting rotation vectors; the relative
    # rotation of the triads leaves round-off
    assert np.linalg.norm(force)<1e-9*np.abs(solver.K).max()


def test_pressure_and_traction_are_area_consistent():
    mesh=Mesh(NODES.copy(),np.array([[0,1,2,3]],np.int32),{"all":np.arange(4)})
    solver=ShellSolver(problem(),mesh,create("linear_elastic",E=E,nu=NU,rho=7850))
    pressure=solver._surface_load(np.arange(4),{"pressure":10.})
    traction=solver._surface_load(np.arange(4),{"traction":[3.,4.,5.]})
    assert solver.dofs.field(pressure,("ux","uy","uz")).sum(axis=0)==pytest.approx([0,0,-20])
    assert solver.dofs.field(traction,("ux","uy","uz")).sum(axis=0)==pytest.approx([6,8,10])


def test_warped_reference_face_is_rejected_with_refinement_advice():
    x=NODES.copy(); x[2,2]=.3
    with pytest.raises(ValueError,match="too warped"):
        shell.precompute(x,E,NU,T,warpage_tolerance=.05)


@pytest.mark.parametrize(("n","minimum"),[(4,.90),(8,.96)])
def test_clamped_thin_square_plate_converges_to_classical_solution_without_locking(n,minimum):
    mesh,p=plate(n,.01); solver=ShellSolver(p,mesh,create("linear_elastic",E=E,nu=NU,rho=7850))
    force=solver._surface_load(np.arange(mesh.n_nodes),{"pressure":1e4})
    edge=np.flatnonzero(np.isclose(mesh.nodes[:,0],0)|np.isclose(mesh.nodes[:,0],1)|
                        np.isclose(mesh.nodes[:,1],0)|np.isclose(mesh.nodes[:,1],1))
    fixed=solver.dofs.element_dofs(edge); free=np.setdiff1d(np.arange(solver.ndof),fixed)
    q=np.zeros(solver.ndof); q[free]=spsolve(solver.K[free][:,free],force[free])
    centre=np.argmin(np.linalg.norm(mesh.nodes[:,:2]-.5,axis=1))
    D=E*.01**3/(12*(1-NU**2)); reference=-.00126*1e4/D
    assert q.reshape(-1,6)[centre,2]/reference>minimum
    assert q.reshape(-1,6)[centre,2]/reference<1.04


def test_simply_supported_square_first_frequency_matches_navier_solution():
    mesh,p=plate(12,.01); rho=7850.; solver=ShellSolver(p,mesh,create("linear_elastic",E=E,nu=NU,rho=rho))
    edge=np.isclose(mesh.nodes[:,0],0)|np.isclose(mesh.nodes[:,0],1)|np.isclose(mesh.nodes[:,1],0)|np.isclose(mesh.nodes[:,1],1)
    free=[]
    for node in range(mesh.n_nodes):
        if not edge[node]: free.append(6*node+2)
        free.extend((6*node+3,6*node+4))
    free=np.asarray(free); mass=sparse.diags(solver.physical_mass[free])
    value=eigsh(solver.K[free][:,free],k=1,M=mass,sigma=0,which="LM",return_eigenvectors=False)[0]
    D=E*.01**3/(12*(1-NU**2)); reference=2*np.pi**2*np.sqrt(D/(rho*.01))
    assert np.sqrt(value)==pytest.approx(reference,rel=.01)


def test_gmsh_quad4_with_line_boundaries_imports_as_shell(tmp_path):
    path=tmp_path/"plate.msh"
    path.write_text(
        '$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n2\n2 1 "plate"\n1 2 "edge"\n$EndPhysicalNames\n'
        '$Nodes\n4\n1 0 0 0\n2 2 0 0\n3 2 1 0\n4 0 1 0\n$EndNodes\n'
        '$Elements\n5\n1 1 2 2 1 1 2\n2 1 2 2 2 2 3\n3 1 2 2 3 3 4\n4 1 2 2 4 4 1\n'
        '5 3 2 1 1 1 2 3 4\n$EndElements\n',encoding="utf-8")
    mesh,kind=gmsh.read(str(path))
    assert kind=="shell4" and mesh.elements.tolist()==[[0,1,2,3]]
    assert set(mesh.regions["edge"])=={0,1,2,3}


def test_shell_beam_shared_dof_assembly_adds_both_physical_blocks():
    blocks={"shell4":np.array([[0,1,2,3]],np.int32),"truss2":np.array([[0,1]],np.int32)}
    mesh=Mesh(NODES.copy(),blocks["shell4"],{"all":np.arange(4),"beam":np.array([0,1])},blocks)
    p=problem(); p["element"]["type"]="shell_line"
    p["members"]=[{"region":"beam","type":"beam2","area":.01,"iy":1e-5,"iz":2e-5,
                   "j":3e-5,"orientation":[0,0,1],"releases":[]}]
    solver=ShellLineSolver(p,mesh,create("linear_elastic",E=E,nu=NU,rho=7850))
    assert np.allclose(solver.K.toarray(),
                       ShellSolver({**p,"element":{**p["element"],"type":"shell4"}},
                                   Mesh(NODES.copy(),blocks["shell4"],dict(mesh.regions)),solver.material).K.toarray()
                       +solver.line.K.toarray())
    solver.q.reshape(4,6)[1,0]=1e-5
    total,_=solver.internal_force(); shell_only,_=ShellSolver._member_forces(solver)
    solver.line.q=solver.q; line_only,_=solver.line.internal_force()
    assert total==pytest.approx(shell_only+line_only)


def test_coupled_import_keeps_only_line2_groups_assigned_as_members(tmp_path):
    path=tmp_path/"stiffened_plate.msh"
    path.write_text(
        '$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n3\n2 1 "plate"\n1 2 "beam"\n1 3 "boundary"\n$EndPhysicalNames\n'
        '$Nodes\n4\n1 0 0 0\n2 2 0 0\n3 2 1 0\n4 0 1 0\n$EndNodes\n'
        '$Elements\n3\n1 1 2 2 1 1 2\n2 1 2 3 2 2 3\n3 3 2 1 1 1 2 3 4\n$EndElements\n',encoding="utf-8")
    mesh,kind=gmsh.read_coupled(str(path),line_regions={"beam"})
    assert kind=="shell_line"
    assert mesh.cell_blocks["shell4"].tolist()==[[0,1,2,3]]
    assert mesh.cell_blocks["truss2"].tolist()==[[0,1]]


def test_distorted_quad_preserves_constant_membrane_patch():
    x=np.array([[0.,0,0],[2.2,.1,0],[1.8,1.1,0],[-.2,.8,0]])
    data=shell.precompute(x,E,NU,T); q=np.zeros((4,6)); ex,ey,g=.2e-3,-.1e-3,.3e-3
    X,Y=data.local_nodes[:,0],data.local_nodes[:,1]
    q[:,0]=ex*X+.5*g*Y; q[:,1]=ey*Y+.5*g*X
    C=E/(1-NU**2)*np.array([[1,NU,0],[NU,1,0],[0,0,(1-NU)/2]])
    assert shell.resultants(data,q.ravel())[:3]==pytest.approx(T*C@np.array([ex,ey,g]),rel=2e-12)


def test_a_lumped_mass_in_a_shell_line_model_is_added_once(tmp_path):
    """`lp` cleared loads, steps, constraints and output for the line half but
    not `nodal_masses`; the shell half had already added them over the same
    node array and line 36 sums both, so a lump arrived at exactly twice its
    weight."""
    lump = 1.0e6
    blocks = {"shell4": np.array([[0, 1, 2, 3]], np.int32),
              "truss2": np.array([[0, 1]], np.int32)}
    totals = []
    for masses in ([], [{"region": "all", "mass": lump}]):
        mesh = Mesh(NODES.copy(), blocks["shell4"],
                    {"all": np.arange(4), "beam": np.array([0, 1])}, blocks)
        p = problem(); p["element"]["type"] = "shell_line"
        p["members"] = [{"region": "beam", "type": "beam2", "area": .01, "iy": 1e-5,
                         "iz": 2e-5, "j": 3e-5, "orientation": [0, 0, 1], "releases": []}]
        p["nodal_masses"] = masses
        s = ShellLineSolver(p, mesh, create("linear_elastic", E=E, nu=NU, rho=7850))
        totals.append(s.physical_mass.reshape(-1, 6)[:, 0].sum())
    assert (totals[1] - totals[0]) == pytest.approx(lump, rel=1e-12)


@pytest.mark.parametrize("extra", [{"removals": [{"region": "beam", "time": 0.1}]},
                                   {"failure": {"yield_strength": 3.5e8, "extreme_fibre": [.05, .05]}}])
def test_shell_line_refuses_breaking_so_its_checkpoint_has_no_parked_particles(extra):
    """ShellLine's checkpoint saves the line half's member state but not
    `_parked` or the appendages: only a member that breaks
    or is removed makes them, and the line half cannot -- its `step` stays
    0 and its queued splits are never applied. That holds while the schema
    refuses both for shell_line."""
    from vfife.core.problem import Problem
    p = problem("dynamic"); p["element"]["type"] = "shell_line"
    p["solver"]["total_time"] = 0.1; p["solver"]["damping"] = {"type": "none"}
    member = {"region": "beam", "type": "beam2", "area": .01, "iy": 1e-5, "iz": 2e-5,
              "j": 3e-5, "orientation": [0, 0, 1], "releases": []}
    if "failure" in extra:
        member["failure"] = extra["failure"]
    else:
        p.update(extra)
    p["members"] = [member]
    with pytest.raises(Exception, match="implemented for line models"):
        Problem.of(p)
