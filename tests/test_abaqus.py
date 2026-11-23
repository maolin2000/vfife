from pathlib import Path
import numpy as np
import pytest

from vfife.io import abaqus, schema
from vfife.cli import build

ROOT=Path(__file__).resolve().parents[1]
INP=ROOT/"tests"/"data"/"abaqus_hex_truss.inp"


def test_inp_coupled_preserves_sparse_tags_sets_and_elsets():
    mesh, element=abaqus.read(str(INP),coupled=True)
    assert element=="coupled"
    assert set(mesh.cell_blocks)=={"hexa8","truss2"}
    assert mesh.cell_blocks["truss2"].tolist()==[[0,1]]
    assert mesh.regions["FIXED"].tolist()==[0]
    assert mesh.regions["BRACE"].tolist()==[0,1]
    assert mesh.regions["ALL_SOLID"].tolist()==list(range(8))


def test_inp_requires_explicit_coupled_mode_for_mixed_topology():
    with pytest.raises(ValueError,match="element.type: coupled"):
        abaqus.read(str(INP))


def test_schema_and_build_accept_abaqus_mesh():
    case=schema.Case.model_validate({
      "mesh":{"type":"abaqus","path":str(INP)},
      "element":{"type":"coupled","formulation":"full"},
      "members":[{"region":"BRACE","type":"truss2","area":1e-3}],
      "material":{"type":"linear_elastic","E":2e11,"nu":.3,"rho":7850},
      "solver":{"mode":"relaxation","backend":"numpy","n_steps":10}})
    mesh,_mat,element=build(case,str(INP))
    assert element=="coupled" and np.allclose(mesh.nodes[1],[1,0,0])
