"""The Midas Civil `.mct` reader: a small stage model
written here -- a two-member cantilever hung from a stay, erected in three
stages -- read back as plain data; id lists; continuation lines; an
encoding that is not UTF-8; what is not understood is counted, not dropped;
a model that does not hang together is said to be so."""
import pytest

from vfife.io import midas

MCT = """*VERSION
   9.1.0
*UNIT    ; Unit System
; FORCE, LENGTH, HEAT, TEMPER
   KN   , MM, KJ, C
*NODE    ; Nodes
; iNO, X, Y, Z
     1, 0, 0, 0
     2, 10000, 0, 0
     3, 20000, 0, 0
     4, 0, 0, 15000
    11, 20000, 0, 0
*ELEMENT    ; Elements
     1, BEAM  ,    1,     1,     1,     2,     0,     0
     2, BEAM  ,    1,     2,     2,     3,    90,     0
    10, TRUSS ,    2,     3,     4,     3,     0,     0
    20, PLATE ,    1,     1,     1,     2,     3,     0,     1,     0
*GROUP    ; Group
; NAME, NODE_LIST, ELEM_LIST, PLANE_TYPE
   节段1      , 1 2, 1, 0
   节段2      , 2to3, 2 \\
        20, 0
   扣索1      , , 10, 0
*BNDR-GROUP
   拱脚固结
   扣地锚
*LOAD-GROUP
   自重
   K1
   平台
*MATERIAL    ; Material
    1, STEEL, 拱肋Q420q , 60, 0, , C, NO, 0.02, 2,  2.0600e+02,   0.3,  1.2000e-05, 7.904e-08,     0
    2, STEEL, 钢绞线 , 0, 0, , C, NO, 0.02, 1, JTG3362-18(S),            , Strand1860    , NO, 195
*SECTION    ; Section
    1, DBUSER    , 弦管1300_22         , CC, 0, 0, 0, 0, 0, 0, YES, NO, P  , 2, 1300, 22, 0, 0, 0, 0, 0, 0, 0, 0
    2, DBUSER    , 箱400         , CC, 0, 0, 0, 0, 0, 0, YES, NO, B  , 2, 400, 300, 12, 16, 0, 0, 0, 0, 0, 0
    3, DBUSER    , 索            , CC, 0, 0, 0, 0, 0, 0, YES, NO, SR , 2, 42.2, 0, 0, 0, 0, 0, 0, 0, 0, 0
    4, SRC       , 钢管混凝土 , CC, 0, 0, 0, 0, 0, 0, YES, NO, CP, 36, 2.5e-08, 0.2, 0.3, 1, 1e-05
       1300, 22, 0, 0
*STLDCASE
   自重 , CS, 
   K1 , CS, 
   平台 , CS, 
*CONSTRAINT    ; Supports
   1, 111111, 拱脚固结
   4, 111000, 扣地锚
*ELASTICLINK    ; Elastic Link
     1, 3, 11, GEN  ,     0, NO, NO, NO, NO, NO, NO, 0.1, 1e+06, 1e+06, 0.01, 0.01, 0.02, NO, 0.5, 0.5, 扣地锚
     2, 3, 11, RIGID,     0, NO, 0.5, 0.5, 扣地锚
*USE-STLD, 自重
*SELFWEIGHT
0, 0, -1, 自重
*USE-STLD, K1
*PRETENSION    ; Pretension Loads
 10, 2104.2, K1
*USE-STLD, 平台
*CONLOAD
 2to3, 0, 0, -5, 0, 0, 0, 平台,
*BEAMLOAD
   1 2, BEAM   , UNILOAD, GZ, NO , NO, LY, , , , 0, -0.0011, 1, -0.0011, 0, 0, 0, 0, 平台, NO, 0, 0, NO, 
*SYSTEMPER
5, 平台
*STAGE    ; Define Construction Stage
   NAME=安装S1, 90, YES, NO, NO, 5
   AELEM=节段1, 0
   ABNDR=拱脚固结, DEFORMED
   ALOAD=自重, FIRST
   NAME=安装S2, 0, YES, NO, NO, 5
   AELEM=节段2, 0, 扣索1, 0
   ABNDR=扣地锚, DEFORMED
   ALOAD=K1, FIRST, 
         平台, FIRST
   NAME=松索, 0, YES, NO, NO, 5
   DELEM=扣索1, 100
   DBNDR=扣地锚
   DLOAD=平台, FIRST
*STAGE-CTRL
  YES, , EXTERNAL, YES
*ENDDATA
"""


@pytest.fixture
def model(tmp_path):
    p = tmp_path / "model.mct"
    p.write_bytes(MCT.encode("gbk"))                   # as Midas writes it on a Chinese Windows
    return midas.read(str(p))


def test_an_id_list():
    assert midas.expand("3 5 10to14 30to50by10") == [3, 5, 10, 11, 12, 13, 14, 30, 40, 50]
    assert midas.expand("") == []
    with pytest.raises(ValueError, match="not an id list"):
        midas.expand("3 five")


def test_the_model_is_read_as_plain_data(model):
    m = model
    assert m.units == ("KN", "MM") and m.si() == (1000.0, 0.001)
    assert m.nodes[3] == (20000.0, 0.0, 0.0) and len(m.nodes) == 5
    assert m.elements[2].kind == "BEAM" and m.elements[2].nodes == (2, 3) and m.elements[2].angle == 90.0
    assert m.elements[10].kind == "TRUSS" and m.elements[10].section == 3
    assert m.elements[20].kind == "PLATE" and m.elements[20].nodes == (1, 2, 3)
    steel, strand = m.materials[1], m.materials[2]
    assert (steel.E, steel.nu, steel.alpha, steel.weight_density) == (206.0, 0.3, 1.2e-5, 7.904e-8)
    assert strand.E == 195.0 and "Strand1860" in strand.standard and strand.weight_density is None
    pipe, box = m.sections[1], m.sections[2]
    assert (pipe.kind, pipe.shape, pipe.dims[:2]) == ("DBUSER", "P", (1300.0, 22.0))
    assert box.shape == "B" and box.dims[:4] == (400.0, 300.0, 12.0, 16.0)
    assert m.sections[4].kind == "SRC" and m.sections[4].lines == ["1300, 22, 0, 0"]      # its further lines, kept
    assert m.groups["节段1"] == ([1, 2], [1])
    assert m.groups["节段2"] == ([2, 3], [2, 20])                                         # a line that went on
    assert m.groups["扣索1"] == ([], [10])
    assert m.boundary_groups == ["拱脚固结", "扣地锚"] and m.load_groups == ["自重", "K1", "平台"]
    assert m.constraints == [([1], "111111", "拱脚固结"), ([4], "111000", "扣地锚")]
    gen, rigid = m.links
    assert gen.kind == "GEN" and gen.nodes == (3, 11) and gen.stiffness == (0.1, 1e6, 1e6, 0.01, 0.01, 0.02)
    assert gen.group == "扣地锚" and rigid.kind == "RIGID" and rigid.stiffness == ()


def test_the_loads_keep_their_load_case(model):
    m = model
    assert m.load_cases == {"自重": "CS", "K1": "CS", "平台": "CS"}
    assert m.self_weight == [("自重", (0.0, 0.0, -1.0), "自重")]
    assert m.pretension == [("K1", 10, 2104.2, "K1")]
    assert m.nodal_loads == [("平台", 2, (0.0, 0.0, -5.0, 0.0, 0.0, 0.0), "平台"),
                             ("平台", 3, (0.0, 0.0, -5.0, 0.0, 0.0, 0.0), "平台")]
    assert m.beam_loads == [("平台", 1, "GZ", -0.0011, -0.0011, "平台"), ("平台", 2, "GZ", -0.0011, -0.0011, "平台")]
    assert m.temperatures == [("平台", 5.0, "平台")]


def test_the_stages(model):
    s1, s2, s3 = model.stages
    assert (s1.name, s1.duration) == ("安装S1", 90.0)
    assert s1.activate_elements == [("节段1", 0.0)] and s1.activate_boundaries == [("拱脚固结", "DEFORMED")]
    assert s1.activate_loads == [("自重", "FIRST")]
    assert s2.activate_elements == [("节段2", 0.0), ("扣索1", 0.0)]
    assert s2.activate_loads == [("K1", "FIRST"), ("平台", "FIRST")]                      # over two lines
    assert s3.deactivate_elements == [("扣索1", 100.0)] and s3.deactivate_boundaries == ["扣地锚"]
    assert s3.deactivate_loads == [("平台", "FIRST")]


def test_what_is_not_read_is_counted_and_the_model_is_checked(model, tmp_path):
    assert model.other == {"STAGE-CTRL": 1}
    assert midas.check(model) == []
    text = midas.summary(model)
    assert "3 stages" not in text and "stages      3: 安装S1 ... 松索" in text
    assert "not read    STAGE-CTRL (1)" in text and "check       consistent" in text
    assert "2 BEAM, 1 TRUSS, 1 PLATE" in text or "2 BEAM, 1 PLATE, 1 TRUSS" in text
    broken = MCT.replace("   AELEM=节段2, 0, 扣索1, 0", "   AELEM=节段2, 0, 扣索9, 0").replace(
        "     2, BEAM  ,    1,     2,     2,     3,    90,     0", "     2, BEAM  ,    7,     2,     2,    99,    90,     0")
    p = tmp_path / "broken.mct"
    p.write_text(broken, encoding="utf-8")                                               # and UTF-8 is read too
    problems = midas.check(midas.read(str(p)))
    assert any("not in *NODE" in x for x in problems)
    assert any("material(s) [7]" in x for x in problems)
    assert any("element group '扣索9' is not in *GROUP" in x for x in problems)


def test_vfife_mesh_says_what_is_in_an_mct(tmp_path, capsys):
    from vfife import cli
    p = tmp_path / "model.mct"
    p.write_bytes(MCT.encode("gbk"))
    assert cli.main(["mesh", str(p)]) in (0, None)
    out = capsys.readouterr().out
    assert "units       KN, MM" in out and "stages      3" in out and "check       consistent" in out


def test_the_stages_in_another_order(model):
    names = [s.name for s in model.stages]
    other = midas.with_stage_order(model, [names[0], names[2], names[1]])
    assert [s.name for s in other.stages] == [names[0], names[2], names[1]] and len(model.stages) == 3
    assert [s.name for s in model.stages] == names                      # the model itself is as it was
    with pytest.raises(ValueError, match="every stage once"):
        midas.with_stage_order(model, names[:2])
    with pytest.raises(ValueError, match="every stage once"):
        midas.with_stage_order(model, names[:2] + ["no such stage"])
