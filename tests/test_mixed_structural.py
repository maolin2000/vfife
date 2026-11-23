from pathlib import Path
import numpy as np
import pytest

from vfife.cli import build, _with_element, make_solver
from vfife.io import schema

ROOT=Path(__file__).resolve().parents[1]
CASE=ROOT/"cases"/"case_03_mixed_frame.yaml"


def test_physical_curves_assign_one_beam_and_one_truss():
    case,_=schema.load(str(CASE)); mesh,mat,element=build(case,str(CASE))
    assert element=="mixed_line"
    s=make_solver(_with_element(case,element,str(CASE)),mesh,mat)
    assert s.member_kind.tolist()==["beam2","truss2"]
    s.q.reshape(-1,6)[2,2]=-1e-4
    force,actions=s.internal_force()
    assert actions[1,0]==pytest.approx(mat.E*1e-4*1e-4,rel=1e-4)
    assert np.allclose(actions[1,1:],0)


def test_mixed_assignment_must_cover_each_line_once():
    case,_=schema.load(str(CASE)); mesh,mat,element=build(case,str(CASE))
    p=_with_element(case,element,str(CASE)); p["members"]=p["members"][:1]
    with pytest.raises(ValueError,match="unassigned"):
        make_solver(p,mesh,mat)


def test_mixed_frame_solution_is_beam_bending_plus_truss_shortening():
    case,_=schema.load(str(CASE)); mesh,mat,element=build(case,str(CASE))
    s=make_solver(_with_element(case,element,str(CASE)),mesh,mat)
    s.run(output_every=999999,progress_every=999999)
    exact=-(1000/(3*2e11*1e-4)+1000/(2e11*1e-4))
    assert s.u[2,2]==pytest.approx(exact,rel=2e-6)
