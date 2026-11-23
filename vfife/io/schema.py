"""Case definition schema -- the single source of truth for a problem.

A case is a YAML file validated against these models.  That one change buys
three things at once:

  * batch parameter studies are just generated YAML, not edited source code
  * a future GUI builds its property editor from these models, so adding a
    material parameter never means touching interface code
  * runs are reproducible: the YAML goes in git and the full text is embedded
    in the result file

Everything the original `setup_beam_4_point_disp.m` expressed is expressible
here, with named regions replacing hard-coded bounding boxes and constraint
indices.
"""

from __future__ import annotations

import math

import re
import os
from typing import Union, Any, Dict, List, Literal, Optional

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, model_validator
from pydantic import field_validator

from ..elements.library import element_names


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def coerce_numbers(obj):
    """Turn numeric-looking strings into floats, recursively.

    YAML 1.1 only recognises an exponent as a float when it carries an explicit
    sign, so `3.0e10` parses as the STRING "3.0e10" while `3.0e+10` parses as a
    float.  That trips up every engineer who writes a modulus the obvious way,
    and it surfaces as a TypeError deep inside a material constructor.  The
    material block is free-form (Dict[str, Any]) so pydantic cannot coerce it;
    this does, once, at the edge.
    """
    if isinstance(obj, str) and _NUM.match(obj.strip()):
        return float(obj)
    if isinstance(obj, dict):
        return {k: coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [coerce_numbers(v) for v in obj]
    return obj


_CONNECTOR_COMPONENTS_ORDER = ("ux", "uy", "uz", "rx", "ry", "rz")
_CONNECTOR_COMPONENTS = set(_CONNECTOR_COMPONENTS_ORDER)


def _connector_matrix(value, label):
    if value is None:
        return np.zeros((6, 6))
    a = np.asarray(coerce_numbers(value), dtype=float)
    if a.shape == (6,):
        a = np.diag(a)
    if a.shape != (6, 6):
        raise ValueError(f"{label} must be a 6-vector or 6x6 matrix")
    if not np.isfinite(a).all():
        raise ValueError(f"{label} must contain finite values")
    scale = max(float(np.abs(a).max(initial=0.0)), 1.0)
    if not np.allclose(a, a.T, rtol=0.0, atol=scale*1e-12):
        raise ValueError(f"{label} matrix must be symmetric")
    if np.linalg.eigvalsh(a).min(initial=0.0) < -scale*1e-12:
        raise ValueError(f"{label} matrix must be positive semidefinite")
    return a


class ConnectorLawSpec(Strict):
    """A path-dependent law on ONE component of a connector:
    `gap` (free travel, then a stop), `bilinear` (elastic-plastic with
    kinematic hardening: an isolator, a yielding device) or `friction`
    (Coulomb: sticks, then slides at mu times the compression another
    component of the same connector carries). elements/connector.py has the
    mechanics; one scalar of history per component, the slipped deformation."""
    #: 律的种类：`gap` 间隙/限位，`bilinear` 双线性滞回（隔震支座、屈服耗能装置），`friction` 库仑摩擦滑动，`damper` 非线性粘滞阻尼器 F = c·sign(v)·|v|^α（1.10），`pendulum` 摩擦摆支座（1.10）
    #: `gap`, `bilinear`, `friction`, `damper` (a fluid viscous damper, F = c sign(v) |v|^alpha; 1.10) or `pendulum` (a friction pendulum bearing; 1.10)
    type: Literal["gap", "bilinear", "friction", "damper", "pendulum"]
    #: gap：自由行程的下限（m 或 rad），不给即该侧无限位（friction 的粘着刚度是 `stiffness`，不读 `lo`）
    #: gap: lower bound of the free travel (None = no stop that side); friction does not read it (its stick stiffness is `stiffness`)
    lo: Optional[float] = None
    #: gap：自由行程的上限；在局部 ux（伸长为正）上，`{lo: 0}` 是只受压的支承（压紧受力、拉开自由），`{hi: 0}` 是只受拉的
    #: gap: upper bound of the free travel; on the local ux (elongation positive) `{lo: 0}` is a compression-only seat and `{hi: 0}` a tension-only tie
    hi: Optional[float] = None
    #: gap：碰到限位后的刚度（N/m 或 N·m/rad）；friction、pendulum：粘着刚度；damper：与阻尼器并联的弹簧（可不写）
    #: gap: the stop's stiffness once a bound is reached; friction, pendulum: the stick stiffness before sliding; damper: a spring beside the damper (optional)
    stiffness: Optional[float] = Field(None, gt=0.0)
    #: bilinear：初始刚度
    #: bilinear: initial (elastic) stiffness
    k1: Optional[float] = Field(None, gt=0.0)
    #: bilinear：屈服后刚度（0 = 理想塑性；须小于 k1）
    #: bilinear: post-yield stiffness (0 = perfectly plastic; must be below k1)
    k2: Optional[float] = Field(None, ge=0.0)
    #: bilinear：屈服力（N 或 N·m）
    #: bilinear: yield force
    fy: Optional[float] = Field(None, gt=0.0)
    #: friction、pendulum：摩擦系数
    #: friction, pendulum: coefficient of friction
    mu: Optional[float] = Field(None, gt=0.0)
    #: pendulum：滑动球面的等效曲率半径（m）；恢复刚度是 N/radius，N 为支座当时的压力
    #: pendulum: effective radius of the sliding surface; the restoring stiffness is N / radius, N the compression the bearing carries at that instant
    radius: Optional[float] = Field(None, gt=0.0)
    #: damper：阻尼系数 c（N·(s/m)^α），F = c·sign(v)·|v|^α
    #: damper: the coefficient c of F = c sign(v) |v|^alpha (force per velocity^alpha)
    c: Optional[float] = Field(None, gt=0.0)
    #: damper：速度指数 α，0 < α ≤ 1（1 即线性阻尼器；桥梁常用 0.2–0.5）
    #: damper: the velocity exponent alpha, 0 < alpha <= 1 (1 is the linear dashpot; bridges use 0.2-0.5)
    exponent: Optional[float] = Field(None, gt=0.0, le=1.0)
    #: damper：低于这个速度（m/s）时按线性 F = c·v0^(α-1)·v；α < 1 时必须写（v → 0 时幂律的斜率无穷大，显式积分会在零速附近抖动）。取设计速度的 1% 左右
    #: damper: below this velocity the law is linear, F = c v0^(alpha-1) v; required when alpha < 1 (the power law's slope is infinite at v = 0 and an explicit step chatters there). About 1% of the design velocity
    linear_below: Optional[float] = Field(None, gt=0.0)
    #: friction：压紧力从哪来——`ux`（同一连接器局部 ux 线性弹簧当前的压力；只有它的正负与节点顺序无关），或一个常数（N）
    #: friction: where the compression comes from -- `ux` (the current compression of this connector's local ux spring, the one component whose sign does not depend on the node order) or a constant force
    normal: Optional[Union[str, float]] = None

    @model_validator(mode="after")
    def _by_type(self):
        need = {"gap": ("stiffness",), "bilinear": ("k1", "k2", "fy"), "friction": ("stiffness", "mu", "normal"),
                "damper": ("c", "exponent"), "pendulum": ("stiffness", "mu", "normal", "radius")}
        allowed = {"gap": {"lo", "hi", "stiffness"}, "bilinear": {"k1", "k2", "fy"},
                   "friction": {"stiffness", "mu", "normal"},
                   "damper": {"c", "exponent", "linear_below", "stiffness"},
                   "pendulum": {"stiffness", "mu", "normal", "radius"}}
        given = {k for k in ("lo", "hi", "stiffness", "k1", "k2", "fy", "mu", "normal", "radius", "c",
                             "exponent", "linear_below")
                 if getattr(self, k) is not None}
        missing = [k for k in need[self.type] if getattr(self, k) is None]
        if missing:
            raise ValueError(f"a {self.type} connector law needs " + ", ".join(missing))
        extra = given - allowed[self.type]
        if extra:
            raise ValueError(f"a {self.type} connector law does not take " + ", ".join(sorted(extra)))
        if self.type == "gap":
            if self.lo is None and self.hi is None:
                raise ValueError("a gap law needs lo and/or hi, or it is no law at all")
            if self.lo is not None and self.hi is not None and self.lo >= self.hi:
                raise ValueError("a gap law needs lo < hi")
        if self.type == "bilinear" and self.k2 >= self.k1:
            raise ValueError("a bilinear law needs k2 < k1 (k2 = k1 is a linear spring: use stiffness)")
        if self.type == "damper" and self.exponent < 1.0 and self.linear_below is None:
            raise ValueError("a damper law with exponent < 1 needs `linear_below`, the velocity under which it "
                             "is linear (the power law's slope is infinite at zero velocity); about 1% of "
                             "the design velocity")
        if self.type in ("friction", "pendulum") and isinstance(self.normal, str) and self.normal != "ux":
            # uy, uz and the rotations flip sign with the Line2's node order,
            # so their "compression" is a coin toss
            raise ValueError("friction `normal` is `ux` (the connector's local elongation: its "
                             "compression is what presses the bearing) or a force")
        if self.type in ("friction", "pendulum") and not isinstance(self.normal, str) and float(self.normal) <= 0.0:
            raise ValueError("a constant friction `normal` must be a positive compression")
        return self


def _validate_connector(stiffness, damping, curves, owner, laws=None):
    K = _connector_matrix(stiffness, f"{owner}.stiffness")
    C = _connector_matrix(damping, f"{owner}.connector_damping")
    curves = coerce_numbers(curves or {})
    laws = laws or {}
    bad = set(laws) - _CONNECTOR_COMPONENTS
    if bad:
        raise ValueError("connector law keys must be ux,uy,uz,rx,ry,rz; bad: " + ", ".join(sorted(bad)))
    for name, law in laws.items():
        law = law if isinstance(law, dict) else law.model_dump()
        i = list(_CONNECTOR_COMPONENTS_ORDER).index(name)
        if K[i, i] != 0.0 or name in curves:
            raise ValueError(f"connector component {name} has a law AND a stiffness or curve: "
                             "a component follows one of them")
        if law["type"] in ("friction", "pendulum") and isinstance(law["normal"], str):
            j = list(_CONNECTOR_COMPONENTS_ORDER).index(law["normal"])
            if law["normal"] == name or K[j, j] <= 0.0 or law["normal"] in laws or law["normal"] in curves:
                raise ValueError(f"friction on {name}: `normal: {law['normal']}` must be a different "
                                 "component carried by a plain linear stiffness")
    bad = set(curves) - _CONNECTOR_COMPONENTS
    if bad:
        raise ValueError("connector curve keys must be ux,uy,uz,rx,ry,rz; bad: "
                         + ", ".join(sorted(bad)))
    for name, values in curves.items():
        table = np.asarray(values, dtype=float)
        if table.ndim != 2 or table.shape[1] != 2 or len(table) < 2:
            raise ValueError(f"connector curve {name} needs at least two [deformation, force] points")
        if not np.isfinite(table).all() or np.any(np.diff(table[:, 0]) <= 0):
            raise ValueError(f"connector curve {name} deformation values must be finite and strictly increasing")
        zero = np.flatnonzero(np.isclose(table[:, 0], 0.0, rtol=0.0, atol=1e-14))
        if len(zero) != 1 or abs(table[zero[0], 1]) > 1e-12:
            raise ValueError(f"connector curve {name} must contain the origin [0, 0]")
        if np.any(np.diff(table[:, 1]) < 0):
            raise ValueError(f"connector curve {name} must be non-decreasing in the first version")
    if not np.any(K) and not np.any(C) and not curves and not laws:
        raise ValueError("connector2 needs stiffness, connector_damping, curves and/or laws")


class ArcSpec(Strict):
    """A line drawn as a curve between its two end points (1.1)."""
    #: 矢高：弦中点处曲线离弦的距离，沿 `up` 量（负值向下）
    #: the rise: how far the curve is from the chord at its middle, along `up` (negative: below)
    rise: float
    #: `parabola`（按弦等分，桥梁拱轴的常用形状）或 `circle`（按圆心角等分）
    #: `parabola` (nodes evenly along the chord, the usual arch axis) or `circle` (evenly by angle)
    shape: Literal["parabola", "circle"] = "parabola"
    #: 矢高的方向（与弦不能平行；默认整体 +z）
    #: the direction of the rise (not along the chord; default global +z)
    up: List[float] = Field(default=[0.0, 0.0, 1.0], min_length=3, max_length=3)


class LineGroupSpec(Strict):
    """One named line of members, straight or an arc, generated (1.1). Its name is
    a region: the members' (for `assign`) and their nodes' (for constraints,
    loads, monitors), as a Gmsh physical line group's is."""
    #: 依次经过的点：`points` 里的名字或坐标 [x, y, z]；每两个相邻点之间是一段
    #: the points the line passes through, in order: names from `points` or
    #: coordinates [x, y, z]; each pair of neighbours is one segment
    through: List[Union[str, List[float]]] = Field(..., min_length=2)
    #: 每一段分成的单元数（与 `size` 二选一；都不写为 1）
    #: elements per segment (or `size`; neither: 1)
    divisions: Optional[int] = Field(None, ge=1)
    #: 单元的目标长度：每一段分成 ceil(段长 / size) 个单元
    #: the target element length: each segment is divided into ceil(length / size)
    size: Optional[float] = Field(None, gt=0.0)
    #: 画成曲线（只用于两点之间）：抛物线或圆弧，见 ArcSpec
    #: a curve instead of a straight line (two points only): see ArcSpec
    arc: Optional[ArcSpec] = None

    @model_validator(mode="after")
    def _one_division(self):
        if self.divisions is not None and self.size is not None:
            raise ValueError("a line takes `divisions` or `size`, not both")
        if self.arc is not None and len(self.through) != 2:
            raise ValueError("an `arc` is drawn between two points: `through` gives its two ends")
        for p in self.through:
            if not isinstance(p, str) and (len(p) != 3 or not all(math.isfinite(float(v)) for v in p)):
                raise ValueError(f"a point in `through` is a name or [x, y, z] (finite), not {p!r}")
        return self


class MeshSpec(Strict):
    #: 网格来源：`box` 结构化盒子（dimensions/divisions/origin）、`gmsh` 读 .msh、`abaqus` 读 .inp、`lines` 在算例里写线单元网格（points/lines，1.1）
    #: `box` (dimensions/divisions/origin), `gmsh` (path to .msh), `abaqus` (path to .inp), or `lines`: a line-member mesh written in the case (points/lines, 1.1)
    type: Literal["box", "gmsh", "abaqus", "lines"] = "box"
    #: lines：命名的点 {名字: [x, y, z]}；每个名字是一个只含该节点的区域；两个点不能重合（支座两侧要错开一点，用连接器相连）
    #: lines: named points {name: [x, y, z]}; each name is a region of that one
    #: node. No two at one place: a bearing's two sides are two points a little
    #: apart, joined by a connector
    points: Optional[Dict[str, List[float]]] = None
    #: lines：命名的构件组 {名字: LineGroupSpec}；名字是这些单元和它们节点的区域
    #: lines: named groups of line members {name: LineGroupSpec}; the name is a
    #: region of those members and of their nodes
    lines: Optional[Dict[str, LineGroupSpec]] = None
    #: lines：几条线合成的组 {名字: [线名, ...]}，与 Gmsh 物理组一样是**单元**区域（供 assign、removals）；`regions:` 里的 union 只是节点集。gmsh：几个物理组合成的组（顶层 `groups:` 降级到这里）
    #: lines: groups of lines {name: [line names]}, an ELEMENT region like a
    #: Gmsh physical group of several pieces (for `assign` and `removals`); a
    #: `regions:` union is a set of nodes only, and a member whose two nodes
    #: are in it belongs to it whichever line it is on. gmsh: groups of the
    #: file's physical groups, the same kind of region (the top-level `groups:`
    #: lowers to this)
    groups: Optional[Dict[str, List[str]]] = None
    #: 盒子网格的三个边长 [Lx, Ly, Lz]（m）
    #: box only
    dimensions: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    #: 盒子网格沿 x、y、z 的单元数
    #: box only: elements along x, y, z
    divisions: Optional[List[int]] = Field(None, min_length=3, max_length=3)
    #: 盒子网格 x、y、z 最小的那个角点
    #: box only: corner with the smallest x, y, z
    origin: List[float] = Field(default=[0.0, 0.0, 0.0], min_length=3, max_length=3)
    #: 网格文件路径，相对算例文件
    #: gmsh only -- path to a .msh, relative to the case file
    path: Optional[str] = None
    #: 坐标统一乘以它：网格按 mm 画、分析按 m 算时写 0.001
    #: multiplies every coordinate. Meshing in mm and analysing in m is the
    #: normal case, and a silent factor of 1000 in the geometry is not the kind
    #: of mistake that announces itself.
    scale: float = Field(1.0, gt=0.0)

    @model_validator(mode="after")
    def _by_type(self):
        if self.type != "lines" and (self.points is not None or self.lines is not None):
            raise ValueError("mesh.points and mesh.lines are for type 'lines'")
        if self.type not in ("lines", "gmsh") and self.groups is not None:
            # (a Gmsh mesh's groups join its physical groups)
            raise ValueError("mesh.groups are for type 'lines' or 'gmsh'")
        if self.type == "lines":
            if not self.lines:
                raise ValueError("mesh type 'lines' needs `lines`: the named groups of members")
            if self.path is not None or self.dimensions is not None or self.divisions is not None:
                raise ValueError("mesh type 'lines' is written in the case: path, dimensions and "
                                 "divisions do not apply")
            for name, pt in (self.points or {}).items():
                if len(pt) != 3 or not all(math.isfinite(float(v)) for v in pt):
                    raise ValueError(f"mesh.points.{name} is [x, y, z], three finite numbers")
            names = [("point", n) for n in (self.points or {})] + [("line", n) for n in self.lines] \
                + [("group", n) for n in (self.groups or {})]
            seen = {}
            for kind, n in names:
                if n in ("all", "surface"):
                    raise ValueError(f"{n!r} is a region every mesh has (`all` every node, `surface` the "
                                     f"outer ones): name the {kind} something else")
                if n in seen:
                    raise ValueError(f"{n!r} names both a {seen[n]} and a {kind}: one name, one region")
                seen[n] = kind
            for name, members in (self.groups or {}).items():
                for m in members:
                    if m not in self.lines:
                        raise ValueError(f"mesh.groups.{name} lists {m!r}, which is not in mesh.lines")
            for name, line in self.lines.items():
                for p in line.through:
                    if isinstance(p, str) and p not in (self.points or {}):
                        raise ValueError(f"mesh.lines.{name} passes through {p!r}, which is not in "
                                         "mesh.points")
            return self
        if self.type == "box":
            if self.dimensions is None or self.divisions is None:
                raise ValueError("mesh type 'box' needs dimensions and divisions")
            if self.path is not None:
                raise ValueError("mesh.path is for type 'gmsh'")
        else:
            if not self.path:
                raise ValueError(f"mesh type {self.type!r} needs a path")
            if self.dimensions is not None or self.divisions is not None:
                raise ValueError(
                    f"mesh type {self.type!r} takes its geometry from the file; "
                    "dimensions/divisions do not apply"
                )
        return self


#: Built from the element library rather than restated, so a new element is
#: accepted by the schema the moment it is defined -- and, just as importantly,
#: an element that is NOT defined can never be accepted here.  `Literal` takes
#: a tuple, so the JSON Schema a GUI reads still carries the full enum.
ElementName = Literal[element_names()]


class FractureSpec(Strict):
    """When an element stops carrying load, and what happens then.

    ONE NUMBER, because everything else about fracture is decided elsewhere on
    purpose (core/fracture.py): WHEN a material point is finished is the
    material's business and is written in the material block (a bar's
    `rupture_strain`, a damage law's own limit), and HOW the points of this
    element family add up is the element's and is not a choice at all. What is
    left for the analyst is how much of the section has to be gone before the
    member is treated as severed, and the honest default is all of it.
    """
    #: 截面纤维（或实体的高斯点）失效的比例达到它即断；1.0 = 整个截面
    #: fraction of the section's fibres (or a solid's Gauss points) that must
    #: have failed. 1.0 = the whole section; lower cuts the member earlier,
    #: which is a modelling shortcut and should be said out loud in a report.
    threshold: float = Field(1.0, gt=0.0, le=1.0)
    #: 断开时把构件从框架上分离出来：它的质量转到自己的粒子上、碎片下落（需要 mode: dynamic；没有接触）
    #: separate the member from the frame when it fractures, instead of only
    #: switching its forces off. The member's share of the mass moves onto
    #: particles of its own and the fragment falls. Off by
    #: default because it is a different analysis: it needs solver.mode =
    #: dynamic, it reserves two particles per element, and what the debris
    #: then lands on is not modelled yet -- there is no contact.
    split: bool = False


class ElementSpec(Strict):
    #: any entry in elements/library.ELEMENTS or LINE_ELEMENTS
    type: ElementName = "hexa8"
    formulation: Literal["full", "reduced", "b-bar", "selective"] = "b-bar"
    hourglass_coeff: float = 0.1
    #: Homogeneous Mindlin shell section and stabilisation controls.
    thickness: Optional[float] = Field(None, gt=0.0)
    shear_correction: float = Field(5.0/6.0, gt=0.0)
    drilling_factor: float = Field(1.0, gt=0.0)
    warpage_tolerance: float = Field(0.05, gt=0.0)
    #: Cross-sectional area (m²), required by a standalone two-node truss.
    area: Optional[float] = Field(None, gt=0.0)
    #: A standalone truss2: the EFFECTIVE prestress every bar
    #: carries at u = 0 (Pa, tension positive) -- a strain offset sigma/E,
    #: the same primitive an embedded tendon uses -- and whether the bars
    #: are cables that carry no compression.
    prestress: Optional[float] = None
    tension_only: bool = False
    #: Section properties for the two-node Euler--Bernoulli space-frame member.
    iy: Optional[float] = Field(None, gt=0.0)
    iz: Optional[float] = Field(None, gt=0.0)
    j: Optional[float] = Field(None, gt=0.0)
    #: Timoshenko shear areas and correction factors in local y/z.
    ay: Optional[float] = Field(None, gt=0.0)
    az: Optional[float] = Field(None, gt=0.0)
    kappa_y: float = Field(1.0, gt=0.0)
    kappa_z: float = Field(1.0, gt=0.0)
    #: Fiber section for beam2_fiber (rect: width along local y, height along local z)
    fiber_section: Optional[Dict[str, Any]] = None
    #: 单一纤维截面指派到的区域（内部键，由 `assign` 降解生成）；求解器核对它覆盖全部纤维构件
    #: the region a single fibre section is assigned to (internal, from `assign`); the solver checks it owns every fibre member
    fiber_region: Optional[str] = None
    #: SEVERAL fibre sections on one model: [{region,
    #: fiber_section, material?, fiber_rebars?, anchorage?, orientation?}],
    #: each region naming the members that carry that section. Exclusive
    #: with `fiber_section`. A section carries its own cover law (`material`,
    #: when not the model's), confined core (inside `fiber_section`), bars
    #: and anchorage; the element-level settings (integration, fracture,
    #: formulation) are one for the model.
    fiber_sections: Optional[List[Dict[str, Any]]] = None
    #: `connector2` members inside a beam2_fiber model:
    #: [{region, stiffness, connector_damping, curves, orientation,
    #: coordinate_system}]. A bearing is two nodes and six stiffnesses -- no
    #: section to integrate and no material to drive -- so it is the one
    #: element type that can share a mesh with fibre sections. Its elements
    #: lie in NO fibre section's region, which is what tells the solver they
    #: are not fibre beams.
    fiber_connectors: Optional[List[Dict[str, Any]]] = None
    #: Bar anchorage for beam2_fiber: {region, bar_diameter}. The elements
    #: touching `region` get their rebar strain reduced by Le/(Le + Lsp) with
    #: Lsp = 0.022 fy db, the length over which a bar keeps straining past the
    #: member face. Declared here rather than inside `fiber_section` because it
    #: is a property of the anchorage, not of the section geometry.
    strain_penetration: Optional[Dict[str, Any]] = None
    #: Optional steel bars inside the fiber section (RC): [{y,z,area,material:{type:rebar,...}}]
    fiber_rebars: Optional[List[Dict[str, Any]]] = None
    #: Prestressing tendons of the fibre section, as the
    #: input form lowers them: [{profile: [[s,y,z],...], interpolation,
    #: normalised, area, material, prestress | jack + friction,
    #: elastic_shortening}]. With `fiber_sections` they go inside each entry
    #: as `fiber_tendons`.
    fiber_tendons: Optional[List[Dict[str, Any]]] = None
    #: Gauss-Legendre (gauss2, no endpoints) vs Gauss-Lobatto (lobatto3/4/5,
    #: endpoints sampled; five stations resolve a hinge a quarter-way in)
    fiber_integration: Optional[Literal["gauss2", "lobatto3", "lobatto4", "lobatto5"]] = None
    #: Element fracture, off unless asked for. beam2_fiber only
    #: so far; the validator below refuses it on anything else rather than
    #: accepting it and doing nothing.
    fracture: Optional[FractureSpec] = None
    #: 弹脆性断裂判据：单一截面的 beam2/beam2_timoshenko 给 yield_strength + extreme_fibre，truss2 给 tensile_strength；纤维梁用 fracture
    #: elastic-brittle breaking of every member: beam2/beam2_timoshenko take yield_strength + extreme_fibre, truss2 tensile_strength; a fibre beam breaks through `fracture`
    failure: Optional[FailureSpec] = None
    #: A vector used to orient the local y axis; it must not be parallel to the bar.
    orientation: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    #: [width, height] of a rect/box beam section, written by the lowering
    #: for `vfife info`'s axes note only
    section_size: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    #: End rotation releases, e.g. ``[i_rz, j_rz]``.  Translational releases are
    #: deliberately not accepted: they create an unconnected member.
    releases: List[str] = []
    #: connector2: six local/global stiffness and damping components, either a
    #: diagonal vector or a symmetric 6x6 matrix.
    stiffness: Optional[Any] = None
    connector_damping: Optional[Any] = None
    coordinate_system: Literal["local", "global"] = "local"
    #: Reversible force-displacement/rotation-moment tables keyed by
    #: ux,uy,uz,rx,ry,rz. A curve replaces that diagonal stiffness component.
    curves: Dict[str, List[List[float]]] = {}
    #: connector2: path-dependent laws per component (gap, bilinear, friction), see ConnectorLawSpec
    laws: Dict[str, ConnectorLawSpec] = {}

    @model_validator(mode="after")
    def _truss_fields(self):
        beam_types = ("beam2", "beam2_timoshenko", "beam2_fiber")
        if self.type in ("truss2",) + beam_types:
            if self.type == "beam2_fiber":
                if not self.fiber_section and not self.fiber_sections:
                    raise ValueError("beam2_fiber requires element.fiber_section")
                if self.fiber_section and self.fiber_sections:
                    raise ValueError("element.fiber_section and element.fiber_sections are exclusive")
                if self.fiber_sections:
                    if len(self.fiber_sections) < 1:
                        raise ValueError("element.fiber_sections is empty")
                    for k, m in enumerate(self.fiber_sections):
                        if not isinstance(m, dict) or not m.get("region") or not m.get("fiber_section"):
                            raise ValueError(f"element.fiber_sections[{k}] needs region and fiber_section")
                    if self.fiber_rebars or self.strain_penetration or self.fiber_tendons:
                        raise ValueError("with element.fiber_sections, rebars, tendons and anchorage "
                                         "go inside each section entry, not on the element")
                if self.formulation != "full":
                    raise ValueError("beam2_fiber supports element.formulation = full only")
            else:
                if self.area is None:
                    raise ValueError(f"element.area is required for {self.type}")
                if self.formulation != "full":
                    raise ValueError(f"{self.type} supports element.formulation = full only")
        if self.type in ("connector2", "mixed_line") and self.formulation != "full":
            raise ValueError(f"{self.type} supports element.formulation = full only")
        if self.type in {"shell3","shell4","shell_line"}:
            if self.thickness is None:
                raise ValueError("shell3/shell4 requires element.thickness")
            if self.formulation != "selective":
                raise ValueError("shell3/shell4 supports element.formulation = selective only")
        elif self.thickness is not None:
            raise ValueError("element.thickness applies to shell3/shell4 only")
        if self.type in beam_types:
            if self.type != "beam2_fiber" and (self.iy is None or self.iz is None or self.j is None):
                raise ValueError(f"{self.type} requires element.iy, element.iz and element.j")
            if self.type == "beam2_timoshenko" and (self.ay is None or self.az is None):
                raise ValueError("beam2_timoshenko requires element.ay and element.az")
            bad = set(self.releases) - {"i_rx", "i_ry", "i_rz", "j_rx", "j_ry", "j_rz"}
            if bad:
                raise ValueError("beam2 releases must be rotational end names; bad: "
                                 + ", ".join(sorted(bad)))
        if self.type not in ("truss2",) + beam_types and self.area is not None:
            raise ValueError("element.area applies to truss2 and beam elements only")
        if self.type != "beam2_timoshenko" and (self.ay is not None or self.az is not None):
            raise ValueError("element.ay/az apply to beam2_timoshenko only")
        if self.type == "beam2_fiber":
            if self.fiber_section:
                fs = self.fiber_section
                if fs.get("type", "rect") == "rect":
                    if fs.get("width") is not None and float(fs["width"]) <= 0:
                        raise ValueError("fiber_section.width must be > 0")
                    if fs.get("height") is not None and float(fs["height"]) <= 0:
                        raise ValueError("fiber_section.height must be > 0")
            if self.fiber_rebars is not None:
                for i, rb in enumerate(self.fiber_rebars):
                    rb = coerce_numbers(rb)
                    self.fiber_rebars[i] = rb
                    if "area" not in rb or float(rb["area"]) <= 0:
                        raise ValueError("fiber_rebars[].area must be >0")
                    if "material" not in rb or "type" not in rb["material"]:
                        raise ValueError("fiber_rebars[].material.type is required")
                    # y/z default 0 if missing (center)
                if self.fiber_section is None:
                    raise ValueError("fiber_rebars requires element.fiber_section")
            if self.fiber_tendons is not None:
                if self.fiber_section is None:
                    raise ValueError("fiber_tendons requires element.fiber_section")
                for i, t in enumerate(self.fiber_tendons):
                    t = coerce_numbers(t)
                    self.fiber_tendons[i] = t
                    for key in ("profile", "area", "material"):
                        if key not in t:
                            raise ValueError(f"fiber_tendons[{i}] needs {key}")
                    has_pre = t.get("prestress") is not None or t.get("prestress_profile") is not None
                    if has_pre == (t.get("jack") is not None):
                        raise ValueError(f"fiber_tendons[{i}] needs one of prestress or jack")
        if self.failure is not None:
            if self.type == "beam2_fiber":
                raise ValueError("element.failure: a fibre member's section knows its own "
                                 "strength -- use element.fracture (the laws' failure), or "
                                 "`removals` for a member cut at a time")
            if self.type == "truss2":
                if self.failure.tensile_strength is None:
                    raise ValueError("element.failure on truss2 takes tensile_strength")
            elif self.type in ("beam2", "beam2_timoshenko"):
                if self.failure.yield_strength is None:
                    raise ValueError("element.failure on a beam takes yield_strength and extreme_fibre")
            elif self.type == "mixed_line":
                raise ValueError("a mixed_line model states failure per member group (assign[].failure)")
            else:
                raise ValueError(f"element.failure is for line members (beam2, beam2_timoshenko, "
                                 f"truss2), not {self.type!r}")
        if self.fracture is not None and self.type != "beam2_fiber":
            # Silently ignoring it would be the worst outcome available: a
            # collapse run that never collapses looks exactly like a structure
            # that survived.
            raise ValueError(
                "element.fracture is implemented for beam2_fiber only "
                f"(got {self.type}); the solid path erodes elements instead")
        # the bearings of a fibre model are held to the rules a connector2
        # member is held to: they were built without them, and a law on a
        # component with a stiffness, or friction pressed by a component with
        # none, ran
        for k, fc in enumerate(self.fiber_connectors or []):
            if isinstance(fc, dict):
                _validate_connector(coerce_numbers(fc.get("stiffness")),
                                    coerce_numbers(fc.get("connector_damping")),
                                    coerce_numbers(fc.get("curves")),
                                    f"element.fiber_connectors[{k}] ({fc.get('region')!r})",
                                    {n: (ConnectorLawSpec(**v) if isinstance(v, dict) else v)
                                     for n, v in (fc.get("laws") or {}).items()})
        if self.type == "connector2":
            self.stiffness = coerce_numbers(self.stiffness)
            self.connector_damping = coerce_numbers(self.connector_damping)
            self.curves = coerce_numbers(self.curves)
            _validate_connector(self.stiffness, self.connector_damping, self.curves, "element", self.laws)
        elif self.stiffness is not None or self.connector_damping is not None or self.curves or self.laws:
            raise ValueError("stiffness, connector_damping, curves and laws apply to connector2 only")
        return self


class FailureSpec(Strict):
    """When a line member breaks: ELASTIC-BRITTLE -- linear
    until the criterion, then gone. Dynamic runs only; per member group in a
    mixed_line model (`assign[].failure`), for every member of a single-
    section beam2/beam2_timoshenko/truss2 model (`element.failure`).

    A bar (cable) breaks when its tension reaches `tensile_strength` x area.
    A beam breaks when, at either end, |N|/(A f_y) + |M_y|/M_y,cap +
    |M_z|/M_z,cap >= 1 with M_cap = shape_factor f_y I / c: `extreme_fibre`
    is [c_y, c_z], the distances from the member axis to the extreme fibres
    along the section's local y and z. Shear and torsion are not checked
    against a strength; but a beam twisted a quarter turn end to end is
    removed (the co-rotational frame cannot follow half a turn), and a
    piece broken at both ends is removed once its demand reaches twice its
    capacity (it is still steel).

    Why not a strain limit: in an elastic member the strain at which steel
    actually tears (15%) is reached only at 30 000 MPa, so the member would
    never break first; a CAPACITY is what an elastic member can be checked
    against, and it needs only A, I and the depth of the section."""

    #: 杆/索：拉应力达到它即断（Pa）
    #: bars and cables: the tension at which the member breaks, Pa
    tensile_strength: Optional[float] = Field(None, gt=0.0)
    #: 梁：承载力判据 |N|/N_p + |M_y|/M_y,cap + |M_z|/M_z,cap ≥ 1 用的屈服强度（Pa）
    #: beams: the yield strength (Pa) behind N_p = f_y A and M_cap = shape_factor f_y I / c
    yield_strength: Optional[float] = Field(None, gt=0.0)
    #: [c_y, c_z]：构件轴到局部 y、z 方向最外纤维的距离（m），M_cap = 形状系数·f_y·I/c
    #: beams: [c_y, c_z], the distances (m) from the member axis to the extreme fibres along the section's local y and z
    extreme_fibre: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    #: 塑性/弹性弯矩之比（1 = 首次屈服；箱形约 1.12）
    #: plastic over elastic moment (1 = first yield; a thin-walled box about 1.12)
    shape_factor: float = Field(1.0, ge=1.0)
    #: 梁（mixed_line）：给了它，梁端就是塑性铰——端弯矩按 |N|/N_p + |M_y|/M_y,p + |M_z|/M_z,p ≤ 1 封顶，超出的转成塑性转角，累计塑性转角（rad）到它才在该端断开；不写就是弹脆性，到承载力立即断开
    #: beams (mixed_line): given, every end is a PLASTIC HINGE -- the end moments are capped on |N|/N_p + |M_y|/M_y,p + |M_z|/M_z,p <= 1, what would exceed it becomes plastic rotation, and the end breaks when its accumulated plastic rotation (rad) reaches this; left out, elastic-brittle: the end breaks the moment the capacity is reached
    rotation_capacity: Optional[float] = Field(None, gt=0.0)
    #: 断了之后：true 质点分裂——构件两端移到自己的备用质点上、带着质量脱开、继续作为一段自由的弹性构件；false 只是不再传力、质量留在节点。不写：梁/mixed_line 分裂，独立桁架不分裂
    #: after the break: true splits the member off onto its own particles with its mass (a free elastic segment), false only deactivates it (mass stays on the nodes); left out, frames and mixed_line split, the standalone truss deactivates
    split: Optional[bool] = None

    @model_validator(mode="after")
    def _one_kind(self):
        beam = self.yield_strength is not None or self.extreme_fibre is not None
        if beam and (self.yield_strength is None or self.extreme_fibre is None):
            raise ValueError("a beam's failure needs both yield_strength and extreme_fibre [c_y, c_z]")
        if beam == (self.tensile_strength is not None):
            raise ValueError("failure: give tensile_strength (a bar) OR yield_strength and "
                             "extreme_fibre (a beam)")
        if self.extreme_fibre is not None and min(self.extreme_fibre) <= 0:
            raise ValueError("extreme_fibre distances must be positive")
        if self.rotation_capacity is not None and not beam:
            raise ValueError("rotation_capacity is a beam's (a plastic hinge at each end): "
                             "give it with yield_strength and extreme_fibre")
        return self


class RemovalSpec(Strict):
    """A member taken out at a given time: the members wholly
    in `region` stop carrying load at `time` seconds after the run starts --
    a cut cable, a burnt hanger. Dynamic runs of any line model (mixed_line,
    beam2, beam2_timoshenko, truss2, beam2_fiber)."""

    #: 要移除的构件（所有节点都在区域内的）
    #: the members (every node inside the region) to remove
    region: str
    #: 移除后是否质点分裂（同 FailureSpec.split）
    #: whether the removed member splits off onto its own particles (as FailureSpec.split)
    split: Optional[bool] = None
    #: 它们停止传力的时刻（s）
    #: the instant (s) they stop carrying load
    time: float = Field(ge=0.0)


class MemberGroupSpec(Strict):
    """Mechanical assignment for one Gmsh Physical Curve group."""
    region: str
    type: Literal["truss2", "beam2", "beam2_timoshenko", "connector2"]
    #: THIS GROUP'S OWN MATERIAL. Absent means the model's
    #: `material` -- which is what every single-material case has, and what
    #: keeps such a case identical to the bit. Written by the input model when
    #: a line section names a material other than the first one's; a cable-
    #: stayed bridge's steel cables and concrete tower are two entries here.
    #: The spec is a resolved material dictionary, exactly as `BlockSpec`
    #: carries one for a continuum block.
    material: Optional[Dict[str, Any]] = None
    area: Optional[float] = Field(None, gt=0.0)
    iy: Optional[float] = Field(None, gt=0.0)
    iz: Optional[float] = Field(None, gt=0.0)
    j: Optional[float] = Field(None, gt=0.0)
    ay: Optional[float] = Field(None, gt=0.0)
    az: Optional[float] = Field(None, gt=0.0)
    kappa_y: float = Field(1.0, gt=0.0)
    kappa_z: float = Field(1.0, gt=0.0)
    orientation: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    #: [width, height] of a rect/box beam section, for `vfife info`
    section_size: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    releases: List[str] = []
    stiffness: Optional[Any] = None
    connector_damping: Optional[Any] = None
    coordinate_system: Literal["local", "global"] = "local"
    curves: Dict[str, List[List[float]]] = {}
    #: connector2: path-dependent laws per component, see ConnectorLawSpec
    laws: Dict[str, ConnectorLawSpec] = {}
    #: truss2 members as CABLES. `prestress` is the effective
    #: stress the bars carry in the mesh geometry (a strain offset sigma/E);
    #: `unstressed_length` is the group's total unstressed length, shared
    #: out over its elements in proportion to their mesh lengths -- the two
    #: are the same statement (L_u = L_0 / (1 + sigma/E)) and only one may be
    #: given. `tension_only` makes the members carry no compression.
    prestress: Optional[float] = None
    unstressed_length: Optional[float] = Field(None, gt=0.0)
    #: one unstressed length PER ELEMENT of the group, in the group's element
    #: order -- what form-finding writes
    unstressed_lengths: Optional[List[float]] = None
    tension_only: bool = False
    #: the FORCE (N) the members are stressed to in the analysis step that
    #: builds them (the first, when built from the start): at that step's end
    #: each carries exactly this, whatever the structure gave under it, and
    #: from then on it is a bar of the unstressed length it ended with:
    jack: Optional[float] = Field(None, gt=0.0)
    #: elastic-brittle breaking
    failure: Optional[FailureSpec] = None

    @model_validator(mode="after")
    def _section(self):
        if self.material is not None:
            self.material = coerce_numbers(self.material)
            if "type" not in self.material:
                raise ValueError(f"member group {self.region!r}: material.type is required")
            # A CONNECTOR HAS NO MATERIAL, and saying so here is cheaper than
            # letting it be built and silently ignored: a connector's six
            # stiffnesses ARE its law.
            if self.type == "connector2":
                raise ValueError("connector2 has no material; its stiffness and "
                                 "connector_damping are its law")
        if self.type in ("truss2", "beam2", "beam2_timoshenko") and self.area is None:
            raise ValueError(f"{self.type} member group requires area")
        stated = [k for k in ("prestress", "unstressed_length", "unstressed_lengths", "jack")
                  if getattr(self, k) is not None]
        if self.type != "truss2" and (stated or self.tension_only):
            raise ValueError("prestress, unstressed_length(s), jack and tension_only apply to truss2 "
                             "members (cables) only")
        if len(stated) > 1:
            raise ValueError("give a cable ONE of `prestress`, `unstressed_length`, "
                             "`unstressed_lengths` or `jack`: each says what length it is cut to")
        if self.unstressed_lengths is not None and any(v <= 0 for v in self.unstressed_lengths):
            raise ValueError("unstressed_lengths must be positive")
        if self.type in ("beam2", "beam2_timoshenko") and any(v is None for v in (self.iy, self.iz, self.j)):
            raise ValueError(f"{self.type} member group requires iy, iz and j")
        if self.failure is not None:
            if self.type == "connector2":
                raise ValueError("a connector has no failure criterion here")
            if (self.type == "truss2") != (self.failure.tensile_strength is not None):
                raise ValueError(f"member group {self.region!r}: a bar breaks on tensile_strength, "
                                 "a beam on yield_strength + extreme_fibre")
        if self.type == "beam2_timoshenko" and any(v is None for v in (self.ay, self.az)):
            raise ValueError("beam2_timoshenko member group requires ay and az")
        if self.type != "beam2_timoshenko" and (self.ay is not None or self.az is not None):
            raise ValueError("member ay/az apply to beam2_timoshenko only")
        allowed = {"i_rx", "i_ry", "i_rz", "j_rx", "j_ry", "j_rz"}
        bad = set(self.releases)-allowed
        if bad:
            raise ValueError("member releases must be rotational end names; bad: "+", ".join(sorted(bad)))
        if self.type == "truss2" and self.releases:
            raise ValueError("truss2 has no rotational end releases")
        if self.type == "connector2":
            if self.area is not None:
                raise ValueError("connector2 does not use area")
            self.stiffness = coerce_numbers(self.stiffness)
            self.connector_damping = coerce_numbers(self.connector_damping)
            self.curves = coerce_numbers(self.curves)
            _validate_connector(self.stiffness, self.connector_damping, self.curves, "member", self.laws)
        elif self.stiffness is not None or self.connector_damping is not None or self.curves or self.laws:
            raise ValueError("stiffness, connector_damping, curves and laws apply to connector2 only")
        return self


class RegionSpec(Strict):
    """One primitive, or a boolean combination of other regions.

    See core/selectors.py. Exactly one selector per region; `tol` is an
    absolute length and defaults to 1e-6 of the mesh diagonal, so the same
    case works whether the mesh is in millimetres or metres.
    """

    #: [xmin, xmax, ymin, ymax, zmin, zmax]：盒子内的节点（按轴成对给）
    #: [xmin, xmax, ymin, ymax, zmin, zmax]: nodes inside the box, the bounds in
    #: pairs per axis (it was documented as all minima, then all maxima, which
    #: selected something else)
    box: Optional[List[float]] = Field(default=None, min_length=6, max_length=6)
    #: 逐轴给 {min, max}（可只给一个）或一个数（该轴取值在 `tol` 内）：{x: 0.1, z: 0} 是一条线，三个轴都给是一点
    #: per axis either {min, max} (either optional) or a bare number, which
    #: pins that axis within `tol`: {x: 0.1, z: 0} is a line, three axes a point
    where: Optional[Dict[str, Union[float, Dict[str, Optional[float]]]]] = None
    #: `{point: [x,y,z], normal: [nx,ny,nz]}`：离平面 `tol` 内的节点
    #: `{point: [x,y,z], normal: [nx,ny,nz]}`: nodes within `tol` of the plane
    plane: Optional[Dict[str, Any]] = None
    #: `{from: [x,y,z], to: [x,y,z]}`：离线段 `tol` 内的节点
    #: {from: [x,y,z], to: [x,y,z]}: nodes within `tol` of the segment
    line: Optional[Dict[str, Any]] = None
    #: `{point, axis, radius, rmin?, length?}`：圆柱内的节点（`point` 在轴上，`length` 从它沿 `axis` 量）
    #: `{point, axis, radius, rmin?, length?}`: nodes inside the cylinder (`point`
    #: on the axis, `length` measured from it along `axis`)
    cylinder: Optional[Dict[str, Any]] = None
    #: `{centre, radius, rmin?}`：球内的节点（`center` 也收）
    #: `{centre, radius, rmin?}`: nodes inside the sphere (`center` is taken too)
    sphere: Optional[Dict[str, Any]] = None
    #: Gmsh 物理组名（该组单元的节点）
    #: a Gmsh physical group name (nodes of its elements)
    group: Optional[str] = None
    #: 直接给节点编号（从 0 起）
    #: explicit 0-based node numbers
    node_ids: Optional[List[int]] = None
    #: 离 [x, y, z] 最近的那一个节点
    #: the one node nearest [x, y, z]
    nearest: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)
    #: `true`：全部节点
    #: `true`: every node
    all: Optional[bool] = None
    #: 并集：要合并的区域名
    #: names of regions to unite
    union: Optional[List[str]] = None
    #: 交集：区域名
    #: names of regions to intersect
    intersect: Optional[List[str]] = None
    #: [a, b, ...]：第一个区域减去其余
    #: [a, b, ...]: the first region minus the others
    difference: Optional[List[str]] = None
    #: 几何选择器的绝对容差（默认网格对角线的 1e-6）
    #: absolute tolerance for the geometric selectors (default 1e-6 of the mesh diagonal)
    tol: Optional[float] = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _one_of(self):
        keys = ("box", "where", "plane", "line", "cylinder", "sphere", "group",
                "node_ids", "nearest", "all", "union", "intersect", "difference")
        given = [k for k in keys if getattr(self, k) not in (None, False)]
        if len(given) != 1:
            raise ValueError(
                "a region needs exactly one of " + ", ".join(keys)
                + "; got " + (", ".join(given) if given else "nothing")
            )
        # THE SELECTORS' OWN KEYS, checked here: a documented `center` or a
        # `where: {X: 0}` was a KeyError out of the solver
        shapes = {"plane": ({"point", "normal"}, {"tol"}),
                  "line": ({"from", "to"}, {"tol"}),
                  "cylinder": ({"point", "axis", "radius"}, {"rmin", "length"}),
                  "sphere": ({"centre", "radius"}, {"rmin"})}
        aliases = {"sphere": {"center": "centre"}, "cylinder": {"axis_point": "point"}}
        for kind, (need, may) in shapes.items():
            spec = getattr(self, kind)
            if spec is None:
                continue
            spec = {aliases.get(kind, {}).get(k, k): v for k, v in spec.items()}
            setattr(self, kind, spec)
            missing = sorted(need - set(spec)); extra = sorted(set(spec) - need - may)
            if missing or extra:
                raise ValueError(f"a `{kind}` region takes {', '.join(sorted(need))}"
                                 + (f" (and optionally {', '.join(sorted(may))})" if may else "")
                                 + (f"; missing {', '.join(missing)}" if missing else "")
                                 + (f"; unknown {', '.join(extra)}" if extra else ""))
        if self.where is not None:
            for axis, lim in self.where.items():
                if axis not in ("x", "y", "z"):
                    raise ValueError(f"`where` takes the axes x, y and z; got {axis!r}")
                if isinstance(lim, dict) and set(lim) - {"min", "max"}:
                    raise ValueError(f"`where.{axis}` takes min and max; got "
                                     f"{', '.join(sorted(set(lim) - {'min', 'max'}))}")
        return self


class WaveSpec(Strict):
    """A travelling wave: each node of the region gets the record late by
    (x - origin) . direction / speed."""

    #: 视波速（m/s）
    #: the apparent wave speed (m/s)
    speed: float = Field(gt=0.0)
    #: 传播方向（整体坐标，自动归一化）
    #: the direction the wave travels in (global, normalised)
    direction: List[float] = Field(min_length=3, max_length=3)
    #: 零时延的点
    #: the point the record arrives at first (zero delay)
    origin: List[float] = Field([0.0, 0.0, 0.0], min_length=3, max_length=3)


class RecordMotionSpec(Strict):
    """A support's own ground motion: the flagged
    translational dofs of the constraint's region follow a record in time,
    in ABSOLUTE coordinates -- the multiple-support form, in which every
    support can move differently and the members between them are strained
    by the difference."""

    #: 两列记录（时间 s、数值），相对算例文件
    #: a two-column record (time s, value), relative to the case file
    path: str
    #: 记录的量：`displacement`（推荐：多点激励要的是位移，且已做基线校正）、`velocity` 或 `acceleration`
    #: what the record is: `displacement` (recommended -- a multiple-support run needs the displacement, baseline-corrected), `velocity` or `acceleration`
    quantity: Literal["displacement", "velocity", "acceleration"] = "displacement"
    #: 乘在记录上的系数（记录以 g 给出时写 9.80665）
    #: multiplies the record (9.80665 for an acceleration in g)
    scale: float = 1.0
    #: 这个支承收到记录的时延（s）
    #: how late this support receives the record (s)
    delay: float = Field(0.0, ge=0.0)
    #: 行波：区域里每个节点按位置再加时延
    #: a travelling wave: each node of the region gets a further delay from its position
    wave: Optional[WaveSpec] = None


class ConstraintSpec(Strict):
    #: 约束作用的节点
    #: the nodes the constraint applies to
    region: str
    #: 1 = 约束/强制，0 = 自由，顺序 [ux, uy, uz, rx, ry, rz]；只有平动的模型给三个
    #: 1 = held/prescribed, 0 = free, in the order [ux, uy, uz, rx, ry, rz]; three entries on a translation-only model
    dofs: List[int] = Field(..., min_length=3, max_length=6)
    #: 强制位移；标量或三分量向量。不给 = 固定为零
    #: prescribed displacement; scalar or 3-vector. None = held at zero.
    value: Optional[Any] = None
    #: 强制的运动量 `value × amplitude(progress)`：位移、速度或加速度；后两者只用于动力分析
    #: Kinematic quantity prescribed by ``value * amplitude(progress)``.
    #: Velocity and acceleration are dynamic-analysis conditions.
    motion: Literal["displacement", "velocity", "acceleration"] = "displacement"
    #: 只用于步内约束（1.2）：以后各步继续保持（解除见步的 `release`）；不写则只在本步有效
    #: a step's constraint only (1.2): it holds in every later step, until a step's `release` frees its nodes; without it a step's constraint is that step's alone
    keep: bool = False
    #: 幅值曲线名；默认：分级松弛的步里 `ramp`（一步内 0 到 1），连续松弛与动力分析里 `ramp_hold`（前段升到 1 后保持），后续步里的模型级荷载和带 `to_age` 的时间步里的一切 `hold`
    #: name of an amplitude curve; by default `ramp` (0 to 1 over the step) in a staged relaxation, `ramp_hold` (up to 1, then held) in a continuous relaxation or a dynamic run, and `hold` for a model-level load in a later step and for everything in a time step (`to_age`)
    amplitude: Optional[str] = None
    #: 支承自己的地面运动记录（多点激励）：被标记的平动自由度按记录随时间运动
    #: the support's own ground-motion record (multiple-support excitation): the flagged translational dofs follow it in time
    record: Optional[RecordMotionSpec] = None

    @model_validator(mode="after")
    def _motion_value(self):
        if self.record is not None:
            if self.value is not None or self.amplitude is not None or self.motion != "displacement":
                raise ValueError("a constraint with a `record` takes its motion from the record: "
                                 "drop value, motion and amplitude")
            if any(self.dofs[3:]) or not any(self.dofs[:3]):
                raise ValueError("a `record` drives translations: flag one or more of ux, uy, uz "
                                 "and no rotation (hold rotations with a constraint of their own)")
            return self
        if self.motion != "displacement" and self.value is None:
            raise ValueError(f"{self.motion} constraint requires value")
        return self


class MovingLoadSpec(Strict):
    """A group of axles travelling along the members of a region.

    The region's members must form one open chain; `s` is arc length along it
    from the end that comes first in (x, y, z), so a bridge laid out along +x
    is entered at its left end by a positive `speed`.

    `axles` are `[offset, force]` pairs: `offset` is the axle's position along
    the path RELATIVE to the group's reference point (so a trailing axle 4.2 m
    behind a lead axle is -4.2, whichever way the group travels), and `force`
    is its magnitude along `direction`, which defaults to straight down. An
    axle whose station is off the path carries nothing, which is how a vehicle
    drives on at one end and off at the other.
    """
    #: [[沿路径的偏移 (m), 轴重 (N)], ...]；偏移沿**行进方向**量，后轴为负
    #: [[offset along the path (m), force (N)], ...]
    axles: List[List[float]] = Field(..., min_length=1)
    #: 步开始时参考点所在的里程
    #: station of the group's reference point at the start of the step
    start: float = 0.0
    #: 沿路径的速度（m/s）：s = start + speed·t，动力过桥
    #: m/s along the path: s = start + speed * t. Dynamic crossings.
    speed: Optional[float] = None
    #: 代替速度：在步的进度内从 `start` 扫到 `stop`——配 `load_steps: n` 就是 n 个静力位置，即影响线
    #: instead of a speed, sweep s from `start` to `stop` over the step's
    #: progress -- with `load_steps: n`, n static positions, an influence line.
    stop: Optional[float] = None
    #: 力的方向（单位化）；默认 −z
    #: unit vector the force pushes along (it is normalised); default is -z
    direction: List[float] = Field([0.0, 0.0, -1.0], min_length=3, max_length=3)

    @model_validator(mode="after")
    def _speed_or_stop(self):
        if len(self.axles) and any(len(a) != 2 for a in self.axles):
            raise ValueError("each axle is [offset along the path (m), force (N)]")
        if (self.speed is None) == (self.stop is None):
            raise ValueError("a moving_load needs exactly one of speed (a crossing in "
                             "time) or stop (a quasi-static sweep over the step)")
        if self.speed == 0.0:
            raise ValueError("a moving_load with speed 0 never moves; use a vector load")
        if not any(self.direction):
            raise ValueError("moving_load.direction must not be the zero vector")
        return self


class SpringDashpotSpec(Strict):
    #: 刚度（N/m）
    #: stiffness (N/m)
    stiffness: float = Field(gt=0.0)
    #: 阻尼（N·s/m）
    #: damping (N s/m)
    damping: float = Field(0.0, ge=0.0)


class VehicleAxleSpec(Strict):
    #: 沿行进方向相对参考点的偏移（m），后轴为负
    #: offset along the direction of travel from the reference point (m); a trailing axle is negative
    offset: float
    #: 簧下质量（kg）：车轮和车轴；有轮胎时必填，无轮胎的刚性轴不给
    #: unsprung mass (kg): the wheel and the axle; with a tyre only
    unsprung_mass: Optional[float] = Field(None, gt=0.0)
    #: 悬挂
    #: the suspension, between the body and the axle
    suspension: SpringDashpotSpec
    #: 轮胎，车轴与路面之间，只压不拉；不给则车轴刚性贴着路面（文献里的"移动簧上质量"）
    #: the tyre, between the axle and the road; it pushes and never pulls. Without one the axle rides the road rigidly (the literature's moving sprung mass)
    tyre: Optional[SpringDashpotSpec] = None

    @model_validator(mode="after")
    def _wheel(self):
        if (self.tyre is None) != (self.unsprung_mass is None):
            raise ValueError("an axle on a tyre needs its unsprung_mass, and an axle without a tyre "
                             "rides the road rigidly and has none: give both or neither")
        return self


class VehicleBodySpec(Strict):
    #: 簧上质量（kg）
    #: sprung mass (kg)
    mass: float = Field(gt=0.0)
    #: 俯仰转动惯量（kg·m²），两轴以上必填
    #: pitch inertia (kg m^2); needed with more than one axle
    pitch_inertia: Optional[float] = Field(None, gt=0.0)
    #: 质心相对参考点的偏移（m，沿行进方向）
    #: offset of the centre of mass from the reference point (m, along travel)
    cg: float = 0.0


class RoughnessPsdSpec(Strict):
    """The displacement PSD G_d(n) = g0 (n/n0)^-w of a road, sampled as a sum
    of `count` cosines with random phases. Its own block so
    a misspelt key is refused (v1.0: `sead` and `nmax` were read as
    nothing)."""
    #: n0 处的功率谱密度（m³），必填；等级表不在库里
    #: the PSD at n0 (m^3); required. No class table lives here
    g0: float = Field(gt=0.0)
    #: 参考空间频率（周/长度单位）（默认值按 SI 给出，声明了非 SI 的 `units` 时换算到声明的单位）
    #: reference spatial frequency (cycles per length unit) (the default is SI, converted to the declared `units` when they are not SI)
    n0: float = Field(0.1, gt=0.0)
    #: 频率指数
    #: the frequency exponent
    w: float = 2.0
    #: 最低空间频率（周/长度单位）（默认值按 SI 给出，声明了非 SI 的 `units` 时换算到声明的单位）
    #: lowest spatial frequency (cycles per length unit) (the default is SI, converted to the declared `units` when they are not SI)
    n_min: float = Field(0.01, gt=0.0)
    #: 最高空间频率（周/长度单位）（默认值按 SI 给出，声明了非 SI 的 `units` 时换算到声明的单位）
    #: highest spatial frequency (cycles per length unit) (the default is SI, converted to the declared `units` when they are not SI)
    n_max: float = Field(3.0, gt=0.0)
    #: 余弦项数
    #: number of cosines
    count: int = Field(400, ge=1)
    #: 相位的随机种子
    #: seed of the random phases
    seed: int = 0

    @model_validator(mode="after")
    def _band(self):
        if self.n_max <= self.n_min:
            raise ValueError("roughness psd: n_max must be above n_min")
        return self


class RoughnessSpec(Strict):
    #: 两列文件（里程 m，高程 m），线性插值，范围外为零
    #: a two-column file (station m, elevation m), linear, zero outside
    profile: Optional[str] = None
    #: 高程的乘数（profile 或 psd 都乘）
    #: multiplies the elevation (of a profile or a psd)
    scale: float = 1.0
    #: 位移功率谱 G_d(n) = g0 (n/n0)^-w（见 RoughnessPsdSpec）
    #: a displacement PSD G_d(n) = g0 (n/n0)^-w (see RoughnessPsdSpec)
    psd: Optional[RoughnessPsdSpec] = None

    @model_validator(mode="after")
    def _one(self):
        if (self.profile is None) == (self.psd is None):
            raise ValueError("roughness needs a `profile` OR a `psd`")
        return self


class VehicleSpec(Strict):
    """A vehicle with mass, suspension and tyres crossing the members of the
    region: a rigid body (heave, and pitch with more than one
    axle) on per-axle suspensions, unsprung masses and tyres that touch the
    deck plus a roughness profile. Dynamic only; the same carriageway rules
    as `moving_load`."""
    #: 名字（监测器用 `of` 引用它）；不写则每一步里依次为 vehicle、vehicle_2……，同一步里不得重名
    #: its name (a monitor's `of`); unnamed ones are vehicle, vehicle_2, ... in each step, and two in one step may not share one
    name: Optional[str] = None
    #: 车身
    #: the body
    body: VehicleBodySpec
    #: 车轴
    #: the axles
    axles: List[VehicleAxleSpec] = Field(..., min_length=1)
    #: 沿路径的速度（m/s）
    #: speed along the path (m/s)
    speed: float
    #: 步开始时参考点的里程（负值在桥外的引道上）
    #: station of the reference point at the start of the step (negative: on the approach)
    start: float = 0.0
    #: 重力方向（车辆向下压的方向）
    #: the direction gravity pulls (the way the vehicle presses)
    direction: List[float] = Field([0.0, 0.0, -1.0], min_length=3, max_length=3)
    #: 重力加速度：默认 9.80665 m/s²，声明了非 SI 的 `units` 时换算（mm 下为 9806.65；1.0 版前一律 9.80665）
    #: gravitational acceleration: 9.80665 m/s^2 by default, converted to the declared `units` when they are not SI (9806.65 in mm; it was 9.80665 whatever the units before v1.0)
    gravity: float = Field(9.80665, gt=0.0)
    #: 路面不平度
    #: road roughness
    roughness: Optional[RoughnessSpec] = None

    @model_validator(mode="after")
    def _moves(self):
        if self.speed == 0.0:
            raise ValueError("a vehicle with speed 0 never crosses; park it with a vector load")
        if len(self.axles) > 1 and self.body.pitch_inertia is None:
            raise ValueError("a body on more than one axle pitches: give body.pitch_inertia")
        if not any(self.direction):
            raise ValueError("vehicle.direction must not be the zero vector: it is the way gravity pulls")
        return self


class TemperatureSpec(Strict):
    """A temperature change over a member's section.

    Temperature is not a force. It is a FREE STRAIN the section would take on
    if nothing held it: eps_th(y, z) = alpha * T(y, z). What that produces
    depends entirely on what is in the way --

      * nothing in the way (a determinate member, a linear T): the member
        simply changes shape. It elongates by alpha dT L, it bows to a
        curvature alpha dT/dy, and it carries NO stress at all.
      * something in the way (a restrained end, a redundant support): the part
        of the free strain the structure refuses becomes force -- an axial
        N = EA alpha dT, a continuous beam's secondary moments.
      * a NONLINEAR T through the depth (which is what a bridge code's design
        profile is): even a simply supported beam cannot take that shape,
        because plane sections must stay plane. The section keeps the linear
        part as elongation and curvature and turns the rest into
        self-equilibrated stresses -- zero net force, zero net moment, and
        real stress. Only a fibre section can show them: a `profile` on a
        `beam2` / `beam2_timoshenko` section (shaped or `general`) is refused,
        not reduced -- reducing it needs the area distribution a beam section
        does not carry.

    The field is `uniform` + `gradient`, or a `profile` sampled through the
    section. `gradient` is K per metre of section depth rather than "top minus
    bottom", because a `general` beam section states area and second moments
    and has no depth to divide by.
    """
    #: 整体温差（K），正 = 比无应力状态热
    #: K, the whole section (positive = warmer than the stress-free state)
    uniform: float = 0.0
    #: [dT/dy, dT/dz]（K/m），沿截面自身坐标；给标量即沿 `axis`（默认 y）的梯度
    #: [dT/dy, dT/dz] in K/m along the section's own axes; a scalar is the
    #: gradient along `axis` (y by default: [g, 0]; `axis: z` gives [0, g]).
    gradient: Optional[Any] = None
    #: [[s, T], ...] 沿截面高度的温度分布（m，K），点间线性、两端外常数——规范的设计温度分布
    #: [[s, T], ...] through the section (m, K), linear between the points and
    #: constant beyond the ends -- a code's design profile written literally.
    #: Mutually exclusive with uniform/gradient.
    profile: Optional[List[List[float]]] = None
    #: profile 或标量 gradient 沿截面哪个轴：`rect`/`box` 的高度在 z，`patches` 在角点坐标所放的那个轴
    #: which of the section's own axes the profile, or a scalar gradient, runs
    #: along. A `rect`/`box` has its depth on z (width along y); a `patches`
    #: section on whichever axis its corners put it -- so a deck's vertical
    #: profile names the axis rather than assuming one. Refused with a
    #: two-component gradient, which names its axes already.
    axis: Literal["y", "z"] = "y"

    @model_validator(mode="after")
    def _one_form(self):
        if self.profile is not None:
            if self.uniform or self.gradient is not None:
                raise ValueError("a temperature is either uniform/gradient or a profile, not both")
            pts = [list(map(float, p)) for p in self.profile]
            if len(pts) < 2 or any(len(p) != 2 for p in pts):
                raise ValueError("temperature profile needs at least two [y, T] points")
            if any(b[0] <= a[0] for a, b in zip(pts, pts[1:])):
                raise ValueError("temperature profile coordinates must be strictly increasing")
        elif not self.uniform and self.gradient is None:
            raise ValueError("a temperature load needs uniform, gradient or profile")
        if self.gradient is not None:
            # `axis` names the axis a SCALAR gradient runs along, as it does a
            # profile's; it was accepted and ignored, so `gradient: 20, axis: z`
            # on a rect (depth on z) bowed the member across its width (found
            # in review). With both components given there is
            # nothing for it to say, and it is refused rather than ignored.
            scalar = not hasattr(self.gradient, "__len__")
            if not scalar and self.axis == "z":
                raise ValueError("temperature `axis` goes with a `profile` or a scalar `gradient`; "
                                 "[dT/dy, dT/dz] already names both axes")
            if scalar:
                g = [0.0, self.gradient] if self.axis == "z" else [self.gradient, 0.0]
                # stored as the pair, which names its axes: `axis` back to its
                # default so the validated spec reads the same when re-read
                self.axis = "y"
            else:
                g = self.gradient
            if len(g) != 2:
                raise ValueError("temperature gradient is [dT/dy, dT/dz] in K/m, or one scalar")
            self.gradient = [float(v) for v in g]
        return self


class LoadSpec(Strict):
    #: 荷载作用的节点、面（traction/pressure）、体（body_force/gravity）或构件（line_load/moving_load）
    #: the nodes, faces (traction/pressure), volume (body_force/gravity) or members (line_load/moving_load) the load acts on
    region: str = "all"
    #: 只用于步内荷载（1.2）：以后各步继续作用，取本步结束时的值；不写则只在本步有效
    #: a step's load only (1.2): it stays on in every later step at the value it ended this one with; without it a step's load is that step's alone
    keep: bool = False
    #: 默认是区域上的**总**力，分摊到各节点；`per_node` 时是每个节点上的力
    #: by default the TOTAL force on the region, spread over its nodes; with
    #: per_node it is the force applied at each node instead
    vector: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    #: 总力矩（N·m）：梁节点上的力偶，或实体表面上按 Mz/I 分布的弯曲面力（可随动）
    #: Total moment (N m): a couple on the nodes of a beam2 model, or on a solid
    #: FACE a couple about M's axis with a potential -- in the reference
    #: configuration the linear bending traction M z / I (and, for the
    #: component along the normal, Saint-Venant's torsion shear) as nodal
    #: forces; see core/mesh.FaceMoment.
    moment: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    #: 沿构件单位长度的均布力（N/m），一致地化为梁端力和力矩；默认全局分量，`coordinate_system: local` 时为局部
    #: Uniform force per unit member length (N/m), consistently converted to
    #: beam end forces and moments. Components are global unless
    #: ``coordinate_system: local`` is selected.
    line_load: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    #: line_load：`local` 用构件自身坐标（x 沿构件）给分量
    #: line_load: `local` gives the components in the member's own axes (x along it)
    coordinate_system: Literal["global", "local"] = "global"
    #: line_load：这个荷载是某样东西的重量（二期恒载、铺装、栏杆），也带上它的质量 |q|/g，按构件长度分到两端节点、三个平动方向；这份质量不再称重（重量就是这个荷载本身），地震、模态和反应谱里有惯性。g 为 9.80665 m/s²，按声明的单位换算。只能写在模型级的 `loads` 里（1.1）
    #: line_load: this load is the weight of something (superimposed dead load, surfacing, parapets), so carry its mass |q|/g too, lumped to the members' end nodes in all three translations. The mass weighs nothing (the load IS its weight), so `gravity` does not count it twice; it gives the load inertia in a dynamic run, a modal analysis and a response spectrum. g is 9.80665 m/s² in the declared units. Model-level `loads` only (1.1)
    with_mass: bool = False
    #: 常面力（N/m²），在区域的外表面上积分；与 `vector` 不同，不随表面节点密度变
    #: constant traction (N/m²) integrated over exterior faces in ``region``.
    #: Unlike ``vector``, this is independent of surface-node density.
    traction: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    #: 正标量压力（Pa），沿外法线向内作用
    #: positive scalar pressure (Pa), acting inward along the exterior normal.
    pressure: Optional[float] = Field(None, gt=0.0)
    #: 压力：每步在当前位置的面上重新积分，始终垂直于表面；面力矩：力偶随面转动并保持大小。默认按参考构形
    #: pressure: re-integrate it over the faces at their CURRENT positions
    #: every step, so it stays normal to the surface it acts on. Face
    #: moment: the couple turns with the face and keeps its magnitude
    #: through any rotation (the potential -|M| theta of the face's angle
    #: about M's axis); without it the reference distribution is a dead
    #: load whose couple decays with the cosine of the rotation. The
    #: default keeps the reference geometry. Solid elements, numpy/numba.
    follower: bool = False
    #: 常体力密度（N/m³），在体上积分
    #: constant force density (N/m³), consistently integrated over volume.
    body_force: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    #: 重力加速度（m/s²），在所选体上化为 ρg
    #: acceleration (m/s²); converted to rho * gravity in the selected volume.
    gravity: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    #: 截面上的温度变化（见 TemperatureSpec）
    #: A temperature change over the section (see TemperatureSpec).
    temperature: Optional[TemperatureSpec] = None
    #: 沿区域构件行驶的轴重组（见 MovingLoadSpec）
    #: Axles travelling along the members of `region` (see MovingLoadSpec).
    moving_load: Optional[MovingLoadSpec] = None
    #: 有质量、悬挂和轮胎的车辆（见 VehicleSpec）
    #: a vehicle with mass, suspension and tyres (see VehicleSpec)
    vehicle: Optional[VehicleSpec] = None
    #: 整个模型的一致地面加速度（动力；框架、纤维梁、桁架、壳、实体）：`path` 两列（秒，加速度），`scale` 换算到算例的加速度单位（SI 下记录以 g 给出写 9.80665），`direction` 给激振方向；只收这三个键
    #: Uniform base acceleration of the whole model in a dynamic run (frames,
    #: fibre beams, trusses, shells, solids): ``path`` holds two columns
    #: (seconds, acceleration), ``scale`` brings it to the case's acceleration
    #: unit (9.80665 for a record in g in an SI case) and ``direction`` gives
    #: the excitation axis. Only these three keys; no amplitude, region ``all``.
    ground_motion: Optional[Dict[str, Any]] = None
    #: `vector`/`moment` 作为**每个**节点上的力，而不是区域上的总和
    #: `vector`/`moment` as the force at EACH node rather than the total over the region
    per_node: bool = False
    #: `amplitudes` 里的名字，或 `hold`（从第一步起全量）；默认：分级松弛 `ramp`（一步内 0 到 1），连续松弛与动力 `ramp_hold`，后续步里的模型级荷载和带 `to_age` 的时间步里的荷载 `hold`
    #: name in `amplitudes`, or `hold` (full from the first step); by default `ramp` (0 to 1 over the step) in a staged relaxation, `ramp_hold` in a continuous relaxation or a dynamic run, and `hold` for a model-level load in a later step and for a load in a time step (`to_age`)
    amplitude: Optional[str] = None

    @model_validator(mode="after")
    def _one_load_kind(self):
        kinds = (self.vector, self.moment, self.line_load, self.traction, self.pressure, self.body_force,
                 self.gravity, self.ground_motion, self.moving_load, self.temperature, self.vehicle)
        if sum(v is not None for v in kinds) != 1:
            raise ValueError("a load needs exactly one of vector, moment, line_load, traction, pressure, "
                             "body_force, gravity, ground_motion, moving_load, vehicle or temperature")
        if self.per_node and self.vector is None and self.moment is None:
            raise ValueError("per_node applies to vector or moment loads only")
        if self.coordinate_system == "local" and self.line_load is None:
            raise ValueError("coordinate_system applies to line_load only")
        if self.with_mass and self.line_load is None:
            raise ValueError("with_mass applies to line_load only: a nodal weight with its mass is "
                             "a `nodal_masses` entry (it weighs under `gravity` by default)")
        if self.follower and self.pressure is None and self.moment is None:
            raise ValueError("follower applies to pressure and moment loads only")
        if self.ground_motion is not None:
            # A GROUND MOTION IS THE WHOLE MODEL'S BASE, ON ITS OWN RECORD:
            # `amplitude`, `region` and any misspelt key were
            # accepted and ignored, and `direction: [0, 0, 2]` doubled the
            # response.
            gm = self.ground_motion
            extra = set(gm) - {"path", "scale", "direction"}
            if extra:
                raise ValueError("ground_motion takes path, scale and direction; not "
                                 + ", ".join(sorted(extra)))
            if "path" not in gm:
                raise ValueError("ground_motion needs `path`, the (time, acceleration) record")
            if "direction" in gm:
                d = [float(x) for x in gm["direction"]]
                n = math.sqrt(sum(x * x for x in d))
                if len(d) != 3 or abs(n - 1.0) > 1e-6:
                    raise ValueError(f"ground_motion.direction must be a unit vector (its length is "
                                     f"{n:.6g}); the record's size is `scale`")
            if self.amplitude is not None:
                raise ValueError("a ground_motion follows its own record in time; it takes no amplitude")
            if self.region != "all":
                raise ValueError("a ground_motion shakes the whole model's base (region: all); a "
                                 "support that moves on its own record is a constraint with `record`")
        if self.vehicle is not None and self.amplitude is not None:
            # it was accepted and ignored: a vehicle's weight
            # is its mass, which no curve scales
            raise ValueError("a vehicle takes no amplitude: its axle loads are its weight and "
                             "its dynamics; to bring it on later give it a negative `start`")
        return self


class NodalMassSpec(Strict):
    """Lumped mass attached to nodes in a named region.

    Isotropic and weighing by default, as before v1.0. `components`
    makes it directional (added mass: a fluid's, or a deck's along one axis),
    `weight: false` keeps it out of `gravity`, `inertia` adds a rotary
    inertia about the global axes on the six-dof families."""
    #: 带质量的节点
    #: the nodes carrying the mass
    region: str = "all"
    #: 区域上的总质量（kg），`per_node` 时为每个节点的质量
    #: kg total over the region, or kg at every node with ``per_node``.
    mass: float = Field(0.0, ge=0.0)
    #: `mass` 是每个节点的，而不是区域总量分摊
    #: `mass` at each node rather than the total spread over the region
    per_node: bool = False
    #: 质量作用在哪些平动分量上 [x, y, z]；默认三向
    #: which translations carry the mass, [x, y, z]; all three by default
    components: List[int] = Field([1, 1, 1], min_length=3, max_length=3)
    #: 重力是否看见它；默认三向质量有重量、定向质量（附加质量）没有
    #: whether gravity sees it; by default a mass on all three translations weighs and a directional (added) one does not
    weight: Optional[bool] = None
    #: 绕整体轴的转动惯量（kg·m²，区域总量或 `per_node` 每节点），只用于六自由度的梁、壳
    #: rotary inertia about the global axes (kg m^2, over the region or per node with per_node); six-dof families only
    inertia: Optional[List[float]] = Field(None, min_length=3, max_length=3)

    @model_validator(mode="after")
    def _something_to_add(self):
        if any(c not in (0, 1) for c in self.components) or not any(self.components):
            raise ValueError("nodal mass `components` are three 0/1 flags [x, y, z], at least one set")
        if self.mass <= 0.0 and not (self.inertia and any(v > 0 for v in self.inertia)):
            raise ValueError("a nodal mass needs a `mass` > 0 or an `inertia`")
        if self.inertia is not None and any(v < 0 for v in self.inertia):
            raise ValueError("nodal mass `inertia` cannot be negative")
        if self.weight and not all(self.components):
            raise ValueError("a directional nodal mass cannot weigh: a body that weighs has its "
                             "inertia in every direction. A mass on fewer than three components is "
                             "added mass -- leave `weight` out (it defaults to false) or carry the "
                             "weight with a mass of its own")
        return self


class HydrodynamicMassSpec(Strict):
    """The Morison added mass of members under water:
    C_a rho_w A per unit length of the submerged part, across the member and
    not along it, weighing nothing. For piers and piles: the members must lie
    within 5 degrees of a global axis (a lumped mass is diagonal)."""

    #: 构件区域（两端都在区域内的构件）
    #: the members (both ends in the region)
    region: str
    #: 圆截面外径（m）；或给 `area`
    #: outer diameter of a round section (m); or give `area`
    diameter: Optional[float] = Field(None, gt=0.0)
    #: 排开水的截面积（m²），非圆截面用
    #: the displaced cross-section (m^2), for a section that is not round
    area: Optional[float] = Field(None, gt=0.0)
    #: 附加质量系数 C_a = C_M − 1；圆柱 1.0
    #: added-mass coefficient C_a = C_M - 1; 1.0 for a circular cylinder
    coefficient: float = Field(1.0, gt=0.0)
    #: 水的密度：默认 1025 kg/m³，声明了非 SI 的 `units` 时换算（mm/N/s 下为 1.025e-9 t/mm³）
    #: water density: 1025 kg/m^3 by default, converted to the declared `units` when they are not SI (1.025e-9 t/mm^3 in mm/N/s)
    density: float = Field(1025.0, gt=0.0)
    #: 水面高程：竖向坐标低于它的构件部分计入
    #: the free surface: the part of a member below it counts
    water_level: float = 0.0
    #: 竖直轴
    #: the vertical axis
    vertical: Literal["x", "y", "z"] = "z"

    @model_validator(mode="after")
    def _one_size(self):
        if (self.diameter is None) == (self.area is None):
            raise ValueError("hydrodynamic_mass needs a `diameter` OR an `area`")
        return self


class PYCurveSpec(Strict):
    """One p-y curve, at one depth (1.2)."""
    #: 深度：地面以下的距离，沿 `vertical` 向下量
    #: depth below the ground surface, measured down along `vertical`
    depth: float = Field(..., ge=0.0)
    #: [[y, p], ...]：桩的水平位移 y 与单位桩长的土抗力 p（按算例的单位，SI 下 m 与 N/m）；从 [0, 0] 起，y 递增，第二点 p > 0；最后一点之后 p 不再增加
    #: [[y, p], ...]: the pile's horizontal displacement y and the soil's resistance per unit length of pile p (in the case's units: m and N/m in SI); from [0, 0], y increasing, p > 0 at the second point; past the last point p stays at its last value
    p_y: List[List[float]] = Field(..., min_length=2)

    @model_validator(mode="after")
    def _a_curve(self):
        for row in self.p_y:
            if len(row) != 2 or not all(math.isfinite(float(v)) for v in row):
                raise ValueError("a p_y row is [y, p], two finite numbers")
        y = [float(r[0]) for r in self.p_y]
        p = [float(r[1]) for r in self.p_y]
        if y[0] != 0.0 or p[0] != 0.0:
            raise ValueError("a p_y curve starts at [0, 0]: no displacement, no resistance")
        if any(b <= a for a, b in zip(y, y[1:])):
            raise ValueError("a p_y curve's y increases from row to row")
        if any(v < 0.0 for v in p):
            raise ValueError("a p_y curve's p is a resistance, not negative: the sign follows the "
                             "displacement")
        if not p[1] > 0.0:
            # a curve with no initial stiffness let the pile move freely at
            # first, and the static solver failed ten cutbacks later with a
            # "limit point" (review of 1.2)
            raise ValueError("a p_y curve rises from [0, 0]: its second point has p > 0 (a soil with "
                             "no initial stiffness does not hold the pile)")
        return self


class SoilSpringSpec(Strict):
    """Soil springs on a pile: p-y curves by depth, nonlinear elastic (1.2).
    Each node of the region below the ground gets a horizontal spring of the
    curve at its depth times the pile length it stands for."""
    #: 桩的构件区域（两端都在区域内的构件）
    #: the pile's members (both ends in the region)
    region: str
    #: 地面高程：沿 `vertical` 的坐标，其下的桩段受土弹簧
    #: the ground surface, as a coordinate on `vertical`: the pile below it carries springs
    ground: float = 0.0
    #: 竖直轴
    #: the vertical axis
    vertical: Literal["x", "y", "z"] = "z"
    #: `horizontal`：水平面内径向（按两个水平位移的合量取抗力）；`x`、`y`（或 `z`）：只沿一个轴，平面模型用
    #: `horizontal`: radial in the horizontal plane (the resistance follows the resultant of the two horizontal displacements); `x`, `y` (or `z`): one axis only, for a plane model
    direction: Literal["horizontal", "x", "y", "z"] = "horizontal"
    #: 各深度的 p–y 曲线；深度之间按同一 y 线性插值，浅于第一条取第一条，深于最后一条取最后一条
    #: the p-y curves by depth: between two depths interpolated at the same y; above the first the first, below the last the last
    curves: List[PYCurveSpec] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _axes(self):
        if self.direction == self.vertical:
            raise ValueError(f"soil_springs: direction {self.direction!r} is the vertical axis; p-y "
                             "springs resist the pile's horizontal displacement")
        d = [c.depth for c in self.curves]
        if len(set(d)) != len(d):
            raise ValueError("soil_springs: two curves at one depth")
        return self


class DiaphragmSpec(Strict):
    """A floor that is rigid in its own plane.

    The nodes of `region` move together as one rigid body in the plane normal
    to `normal`: two translations and the rotation about `normal`. Everything
    else -- the out-of-plane translation and the two bending rotations -- stays
    free, which is what makes this a diaphragm and not a rigid body.

    It is how a building's floor slab is written down when the slab itself is
    not meshed, and it is what an OpenSees model means by `rigidDiaphragm`.
    """
    #: 绑在一起的节点
    #: the nodes tied together
    region: str
    #: 平面法向：节点在垂直于它的平面内作刚体运动
    #: the plane's normal: the nodes move as a rigid body within the plane normal to it
    normal: Literal["x", "y", "z"] = "z"


class StepSpec(Strict):
    """One analysis step. Steps run in sequence and share the model state.

    The model-level `constraints` and `loads` are the base and apply in every
    step; these are added to them, and where both prescribe the same (node, dof)
    the step wins.
    """

    #: 显示在运行日志里，并随该步结果保存
    #: shown in the run log and stored with the step's results
    name: Optional[str] = None
    #: 本步的约束，叠加在模型级约束上（冲突时本步优先）
    #: this step's constraints, added to the model-level ones (the step wins on a conflict)
    constraints: List[ConstraintSpec] = []
    #: 本步的荷载，叠加在模型级荷载上
    #: this step's loads, added to the model-level ones
    loads: List[LoadSpec] = []
    #: 逐步覆盖求解器里「做多少」的设置；mode、mass、backend、damping 属于模型，不能逐步改
    #: per-step overrides of the solver settings that describe HOW MUCH to do;
    #: mode, mass, backend and damping are properties of the model, not the step
    n_steps: Optional[int] = None
    #: 松弛：本步内分这么多荷载级，逐级保持到平衡
    #: relaxation: hold this many load levels to equilibrium within the step
    load_steps: Optional[int] = Field(None, ge=1)
    #: 松弛：每级的迭代预算
    #: relaxation: iteration budget per load level
    max_steps_per_load: Optional[int] = None
    #: 每个新荷载级渐入所用的迭代数（默认 max_steps_per_load/5）
    #: iterations over which each new load level is blended in (default max_steps_per_load / 5)
    stage_ramp_iterations: Optional[int] = None
    #: 本步的残差容差
    #: residual tolerance for this step
    tolerance: Optional[float] = None
    #: 本步起点的混凝土龄期（天）；默认接上一步结束处
    #: The step's time window in days of concrete age:
    #: `age` defaults to where the last step left off, and `to_age` makes the
    #: step a stretch of TIME whose `load_steps` are its time stations --
    #: log-spaced, because that is the axis creep happens on.
    age: Optional[float] = Field(None, gt=0.0)
    #: 本步终点的混凝土龄期（天）：给了本步就是一段时间，`load_steps` 是时间站
    #: concrete age in days at the end of the step: makes it a stretch of time
    to_age: Optional[float] = Field(None, gt=0.0)
    #: 时间站的分布：`log`（默认，徐变的轴）或 `linear`
    #: `log` (default, the creep axis) or `linear` spacing of the time stations
    time_spacing: Literal["log", "linear"] = "log"
    # --- analysis steps (1.2): each step its own procedure, and model change ---
    #: 本步的分析类型（1.2）：relaxation、dynamic 或 static；不写则用 solver.mode。线单元模型（beam、fiber、mixed_line）
    #: this step's procedure (1.2): relaxation, dynamic or static; left out, solver.mode's. Line models (beam, fibre, mixed_line)
    procedure: Optional[Literal["relaxation", "dynamic", "static"]] = None
    #: 动力步的时长（s）；不写则用 solver.total_time
    #: a dynamic step's duration (s); left out, solver.total_time
    total_time: Optional[float] = Field(None, gt=0.0)
    #: 本步出生的构件（所有节点都在区域内的）：在当时的变形位置无应力出生，自重与线荷载在本步内加上
    #: members born in this step (every node inside the region): stress-free where the structure has moved to, their weight and line loads coming on over the step
    activate: List[str] = []
    #: 只连着出生构件的节点放在哪里：tangent（默认，随相邻已有节点做刚体平移与转动——悬臂施工的切线安装）或 design（留在设计坐标）
    #: where the nodes that only the born members reach are put: tangent (default: carried rigidly by the neighbouring node that already stands -- a cantilever's tangential installation) or design (left at their design coordinates)
    birth: Literal["tangent", "design"] = "tangent"
    #: 出生构件的混凝土龄期（天）；不写则用 solver.age
    #: the concrete age (days) of the members born in this step; left out, solver.age
    activate_age: Optional[float] = Field(None, gt=0.0)
    #: 本步移除的构件：它们对其余结构的作用力（含自重）在本步内卸到零
    #: members removed in this step: what they did to the rest of the structure (their weight included) is taken off over the step
    deactivate: List[str] = []
    #: 解除这些区域节点上继承下来的约束（模型级与 keep 的）：支反力在本步内卸到零；本步自己的约束照常施加
    #: frees the inherited constraints (model-level and kept ones) on these regions' nodes: the reaction is taken off over the step; the step's own constraints still apply
    release: List[str] = []

    @model_validator(mode="after")
    def _model_change(self):
        both = sorted(set(self.activate) & set(self.deactivate))
        if both:
            raise ValueError(f"step {self.name or ''}: {both[0]!r} is both activated and deactivated")
        if self.total_time is not None and self.procedure not in (None, "dynamic"):
            raise ValueError("a step's total_time is a dynamic step's duration; this step is "
                             f"{self.procedure}")
        if self.activate_age is not None and not self.activate:
            raise ValueError("activate_age is the age of the members a step activates; it activates none")
        return self

    @model_validator(mode="after")
    def _no_mass_in_a_step(self):
        # a mass is the model's, like `solver.mass`: one that appeared with a
        # later step would have been in the explicit step and the modes from
        # the first (1.1)
        if any(ld.with_mass for ld in self.loads):
            raise ValueError("with_mass is the model's: give the line_load in the model-level "
                             "`loads` (with an `amplitude` if its weight comes on later) -- the "
                             "mass is there from the first step")
        return self


#: the element types whose steps may change the model or the procedure (1.2)
STEP_CHANGE_ELEMENTS = frozenset(("beam2", "beam2_timoshenko", "mixed_line", "beam2_fiber"))

#: Sentinel: a solver setting the schema does not give a default for.
_NO_SCHEMA_DEFAULT = object()


def solver_setting(sol: dict | None, key: str):
    """A solver setting, with the default taken from THIS FILE and nowhere else.

    (The solver classes read the typed case instead since v1.19 --
    `self.p.solver.<key>`, core/problem.py -- which carries the same defaults;
    this is for the dict readers that remain while P2 moves them over.)

    WHY THIS EXISTS, AND WHY EVERY SOLVER SHOULD USE IT
    ---------------------------------------------------
    A pydantic model always emits every field, so a validated case dict has
    the key PRESENT even when the analyst never wrote it -- carrying either
    the schema's default or `None`. That makes `sol.get(key, <literal>)` in a
    solver dead code: the key is there, so the literal never fires, and the
    schema silently wins.

    It has already cost this project twice, in two different shapes:

      * `mass`: the schema emits `mass: None`, so `sol.get("mass", ...)` never
        fired and relaxation cases silently ran with PHYSICAL mass and a CFL
        time step -- real dynamics, with n_steps counted as if they were
        relaxation iterations, and no error anywhere. Patched locally with
        `or` and a comment (core/solver.py, _setup_mass_and_dt).

      * `damping.type`: the schema defaults it to "kinetic", so SIX solvers'
        `damp.get("type", "kinetic" if relaxation else "viscous")` were all
        dead and every `mode: dynamic` run got kinetic damping -- a device for
        finding a STATIC answer, which removes the dynamics it is applied to.

    Two places each declaring a default is the defect; the fix is to have one.
    The schema is that one, because it is what a GUI reads, what the JSON
    Schema exports, and what the analyst is told. A solver asks for the value
    and gets the documented default when the case is silent -- including for a
    hand-built dict that never went through validation, which is how most of
    the unit tests construct their cases.

    THE ONE LEGITIMATE EXCEPTION
        A default that depends on something the schema cannot see -- the
        analysis mode, another field, the mesh -- cannot live here. Those
        fields carry `None` in the schema and the solver resolves them, with
        `or` rather than a `get` default so the present-but-None key cannot
        trip it again. `damping.type` is exactly that case and is the reason
        it is `Optional` below.
    """
    value = (sol or {}).get(key)
    if value is not None:
        return value
    field = SolverSpec.model_fields.get(key)
    if field is None:
        return None
    default = field.default
    # PLAIN DATA OUT, ALWAYS. A solver consumes the case as `model_dump`
    # output -- nested blocks arrive as dicts -- so handing back the pydantic
    # model that sits in `field.default` would give `damping` an object where
    # every caller does `damp.get("type")`. Caught by tests/test_bcs.py, which
    # builds its case dicts by hand and is therefore the only place the schema
    # default is ever the value actually returned.
    return default.model_dump() if isinstance(default, BaseModel) else default


class DampingSpec(Strict):
    #: 阻尼类型：viscous、kinetic、rayleigh、none。不给时按模式定——松弛 kinetic，动力 none（动力分析不要用 kinetic，它会抹掉动力响应）
    #: NO DEFAULT HERE, DELIBERATELY, AND IT IS THE ONE FIELD THAT NEEDS NONE.
    #:   The right default depends on the analysis mode -- kinetic for
    #:   relaxation, undamped for dynamic -- which this file cannot see. It
    #:   used to default to "kinetic", and because a pydantic model always
    #:   emits its fields, that made the mode-dependent expression in SIX
    #:   solvers dead code: every `mode: dynamic` run got kinetic damping,
    #:   which zeroes the velocity at every kinetic-energy peak and so removes
    #:   the dynamics it is applied to.
    #:
    #:   `None` here, resolved in the solver with `or` -- never with a `get`
    #:   default, which a present-but-None key walks straight past. See
    #:   `solver_setting` above for the general rule.
    type: Optional[Literal["viscous", "kinetic", "rayleigh", "none"]] = None
    #: viscous：单位质量的阻尼系数（1/s）
    #: viscous: damping ratio-like coefficient per unit mass (1/s)
    coeff: float = 0.0
    #: 质量比例阻尼（1/s）。用于 `type: rayleigh`，以及松弛里与 `kinetic` 叠加（0.1 版为软化问题设计的组合）；`viscous`、`none` 下写了会被拒绝。软化问题上它会选择解的分支
    #: Mass-proportional damping, 1/s. Applied under `type: rayleigh`, and
    #: ALONGSIDE `type: kinetic` in a relaxation -- refused under `viscous`
    #: and `none`, where the frame solvers ignored it.
    #: `type: kinetic` with a small alpha is the combination it exists for: kinetic damping only fires when the kinetic energy
    #: peaks, so a mode whose energy grows monotonically -- a softening
    #: element -- is never touched by it, while alpha*M*v opposes velocity
    #: whatever the energy is doing. It is not free: on a softening problem
    #: it selects a solution branch (v0.1 has the measurements).
    alpha: float = Field(0.0, ge=0.0)
    #: 刚度比例阻尼（s）；只用于动力分析的 `type: rayleigh`，会缩小稳定步长
    #: stiffness-proportional damping, s; `type: rayleigh` in a dynamic analysis only; makes the stable step smaller
    beta: float = Field(0.0, ge=0.0)
    #: Rayleigh 的另一种写法：在 `frequencies` 的两个频率上各给阻尼比——一个数（两处相同）或 [ζ1, ζ2]；alpha、beta 由此算出
    #: Rayleigh stated by its damping ratio at the two `frequencies` (one number, or [zeta_1, zeta_2]); alpha and beta are computed from it
    ratio: Optional[Any] = None
    #: 给 `ratio` 的两个频率（Hz）：通常取第一阶和参与显著的最高阶
    #: the two frequencies (Hz) `ratio` is stated at: usually the first mode and the highest one that matters
    frequencies: Optional[List[float]] = Field(None, min_length=2, max_length=2)

    @model_validator(mode="after")
    def _rayleigh_coefficients(self):
        if (self.ratio is None) != (self.frequencies is None):
            raise ValueError("damping: `ratio` and `frequencies` go together")
        if self.ratio is not None:
            if self.type not in (None, "rayleigh"):
                raise ValueError("damping: `ratio` / `frequencies` state Rayleigh damping (type: rayleigh)")
            # alpha and beta beside the ratio are refused -- unless they are the
            # ones this computes from it: a validated case dumped and validated
            # again (a solver's case, core/problem.py)
            given = (self.alpha, self.beta)
            self.type = "rayleigh"
            f1, f2 = (float(x) for x in self.frequencies)
            if f1 <= 0.0 or f2 <= 0.0 or f1 == f2:
                raise ValueError("damping.frequencies: two distinct positive frequencies (Hz)")
            z = self.ratio
            z1, z2 = (float(z), float(z)) if not isinstance(z, (list, tuple)) else (float(z[0]), float(z[1]))
            if z1 < 0.0 or z2 < 0.0 or (z1 == 0.0 and z2 == 0.0):
                raise ValueError("damping.ratio: non-negative, not both zero")
            w1, w2 = 2.0 * math.pi * f1, 2.0 * math.pi * f2
            # zeta_i = alpha / (2 w_i) + beta w_i / 2 at both frequencies
            self.alpha = 2.0 * (z1 * w2 - z2 * w1) * w1 * w2 / (w2 * w2 - w1 * w1)
            self.beta = 2.0 * (z2 * w2 - z1 * w1) / (w2 * w2 - w1 * w1)
            if any(given) and given != (self.alpha, self.beta):
                raise ValueError("damping: give alpha/beta OR ratio/frequencies, not both")
            if self.alpha < 0.0 or self.beta < 0.0:
                raise ValueError("damping.ratio: these two ratios at these two frequencies need a negative "
                                 "alpha or beta -- Rayleigh cannot give a ratio that falls that fast with frequency")
            self.ratio = [z1, z2]
        if self.type == "rayleigh" and self.alpha == 0.0 and self.beta == 0.0:
            raise ValueError("Rayleigh damping needs alpha and/or beta > 0")
        return self


class BarRepeatSpec(Strict):
    """Step a line along a vector -- one stirrup leg plus a spacing."""

    count: int = Field(..., ge=1)
    offset: List[float] = Field(..., min_length=3, max_length=3)


class BarLineSpec(Strict):
    """One bar: a straight run (`start`/`end`) or a connected route (`points`).

    A ROUTE IS ONE BAR, AND THAT IS THE WHOLE REASON `points` EXISTS.
    Two entries whose ends touch are two bars: `bars_from_spec` gives every
    entry its own nodes, so the segments are not joined and each one is free
    to move on its own. For a perfectly bonded bar that is harmless -- every
    bar node takes the host's interpolated displacement, so coincident nodes
    move together anyway -- and for a BONDED bar it is not: a tendon drawn as
    thirty-two touching entries is thirty-two independent stubs, each free to
    slip and each pulling out at tau_max, and `anchorage: ends` ties the ends
    of every one of them instead of the two ends of the tendon.

    `points` lays the same route with its segments SHARING nodes. `closed`
    joins the last point back to the first, which is what a hoop is -- and it
    is what makes the "a closed hoop has no free end" refusal in
    elements/bonded.py true, where before every leg of every hoop had two.
    """
    start: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    end: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    #: a connected route through three or more points; `divisions` and
    #: `segment_length` then apply to each leg
    points: Optional[List[List[float]]] = None
    #: close the route back to its first point (a tie, a hoop)
    closed: bool = False
    repeat: Optional[BarRepeatSpec] = None
    #: segments along the line.  Each carries one constitutive point, so a bar
    #: modelled as a single element yields over its whole length at once.
    #:
    #: PREFER `segment_length` IN A MEMBER THAT CRACKS.  These bars are
    #: perfectly bonded, so a crack of opening w crossing a segment of length L
    #: puts strain w/L into it with no slip to spread it: the segment length is
    #: a physical smearing length, not a discretisation choice.  Measured on the
    #: solid RC column, refining the mesh 100 -> 50 mm while letting `divisions`
    #: follow halved the drift the model survived (32.0 -> 18.0 mm) at a bar
    #: plastic strain that did not move; holding the segments at 100 mm put it
    #: back to 32.0 exactly.  A count invites being scaled with the mesh; a
    #: length in metres does not.
    divisions: Optional[int] = Field(None, ge=1)
    #: segment length in metres, rounded to the nearest whole number of
    #: segments.  Give this or `divisions`, not both.
    segment_length: Optional[float] = Field(None, gt=0.0)

    @model_validator(mode="after")
    def _one_or_the_other(self):
        if self.divisions is not None and self.segment_length is not None:
            raise ValueError(
                "a reinforcement line takes `divisions` or `segment_length`, "
                "not both -- they are different answers to the same question "
                "and one of them would silently win")
        straight = self.start is not None and self.end is not None
        if straight == (self.points is not None):
            raise ValueError("a reinforcement line is `start` + `end`, or `points`")
        if self.points is not None:
            if len(self.points) < 2:
                raise ValueError("a reinforcement route needs at least two points")
            if any(len(p) != 3 for p in self.points):
                raise ValueError("every point of a reinforcement route is [x, y, z]")
        elif self.closed:
            raise ValueError("`closed` describes a route: give `points`")
        return self


class BondSpec(Strict):
    """Bond stress-slip between an embedded bar and its host.

    Present means the bar gets its OWN degrees of freedom and may slip; absent
    means perfect bond, which is cheaper, runs on every backend, and is right
    whenever the member is not expected to crack. See materials/bond.py for the
    law and v0.1 for why the choice matters more than it looks.

    TWO LAWS SHIP, AND THE DEFAULT IS THE MONOTONIC ONE
        `bond_ceb`     the MC2010 envelope with a linear unloading rule. Right
                       for a pushover; under a long cyclic history it
                       overstates the bond that survives, which is the unsafe
                       direction.
        `bond_cyclic`  the same envelope with Eligehausen-type degradation and
                       a pinched reversal. Identical to `bond_ceb` on a
                       monotonic path, bit for bit, so switching costs nothing
                       but the extra state.

        The default stays `bond_ceb` because changing it would silently
        re-rate every model that already names a bond block.
    """
    model_config = ConfigDict(extra="allow")     # the law's own parameters
    #: 粘结–滑移律：`bond_ceb`（MC2010 单调）或 `bond_cyclic`（往复退化）；其余键是该材料的参数
    #: the bond-slip law: `bond_ceb` (MC2010 monotonic) or `bond_cyclic`; every other key is that material's parameter
    type: str = "bond_ceb"


class TendonFrictionSpec(Strict):
    """Post-tensioning friction: sigma(s) = sigma_con exp(-(mu theta + k s)).

    With this present, `prestress` is sigma_con AT THE JACK and the solver
    derives the stress in every segment; without it, `prestress` is the value
    the whole tendon carries. That is the one place the meaning of a field
    depends on another, and preflight says which reading is in force.
    """
    #: 曲率摩擦系数（每弧度角度变化）
    #: curvature coefficient, per radian of accumulated angle change
    mu: float = Field(..., ge=0.0)
    #: 摆动摩擦系数（每米束长）
    #: wobble coefficient, per metre of tendon
    k: float = Field(0.0, ge=0.0)
    #: 张拉端在哪头；`both` 两端张拉，每点取较小的损失
    #: which end the jack is at; `both` takes the smaller loss at each point
    jack: Literal["start", "end", "both"] = "start"
    #: 锚固回缩量，**米**（6 mm 写 0.006）
    #: draw-in as the wedges seat, in METRES (6 mm is 0.006)
    anchor_set: Optional[float] = Field(None, ge=0.0)


class ReinforcementSpec(Strict):
    """An embedded bar block: steel that lives inside the concrete mesh.

    The bars do not need nodes of their own and the mesh does not have to line
    up with them -- their displacement is interpolated from whatever host
    element each end falls in (elements/truss.py).  Perfect bond is assumed.
    """

    name: Optional[str] = None
    #: cross-section of ONE bar, m^2
    area: float = Field(..., gt=0.0)
    #: must be a uniaxial law; `rebar` is the one that ships
    material: Dict[str, Any]
    #: subtract the host modulus the bar duplicates (see elements/truss.py,
    #: THE OVERLAP).  Right while the concrete around the bar is elastic;
    #: leave at 0 when it is expected to crack.
    host_deduction: float = 0.0
    #: bar diameter, required only when `bond` is present: a bond force is a
    #: stress times the bar SURFACE, which the area alone does not give
    diameter: Optional[float] = Field(None, gt=0.0)
    #: absent = perfect bond (every backend); present = the bar gets its own
    #: dofs and may slip (numpy and numba; warp keeps its time loop on the
    #: device and is refused with a message that says so)
    bond: Optional[BondSpec] = None
    #: stiffness restraining slip ACROSS the bar. A numerical device, not a
    #: dowel model. Defaults to the BAR's own axial stiffness over the length
    #: each node speaks for, E*A/L_trib. An earlier default of 100x the bond
    #: stiffness came to 5e11 N/m on the Tanaka column, a fictitious mass of
    #: 1.4e12 kg, and a first load step that never converged -- a penalty's
    #: only job is to be stiff enough, and beyond that it buys nothing but a
    #: smaller time step (see elements/bonded.py).
    transverse_penalty: Optional[float] = Field(None, gt=0.0)
    #: bonded bars only: tie the bar's free ends to the host (`ends`), or
    #: only each line's first (`start`) or last (`end`) node. A bar that
    #: stops inside the mesh with no anchorage holds by the bond on its last
    #: half-segment alone and pulls out at tau_max; real bars run past the
    #: support, hook, or bear on a plate (elements/bonded.py).
    anchorage: Optional[Literal["ends", "start", "end"]] = None
    #: EFFECTIVE prestress in the bar after losses, Pa, positive in tension.
    #: Pretensioning: the bar is already strained at u = 0, so the first
    #: equilibrium IS transfer and the elastic-shortening loss comes out of
    #: it. Friction, anchor set, relaxation and creep are NOT computed --
    #: state the stress the tendon actually has (elements/truss.py, PRESTRESS).
    prestress: Optional[float] = None
    #: present = `prestress` is the jacking stress and the profile is computed
    friction: Optional[TendonFrictionSpec] = None
    lines: Optional[List[BarLineSpec]] = None
    nodes: Optional[List[List[float]]] = None
    elements: Optional[List[List[int]]] = None

    @model_validator(mode="after")
    def _geometry(self):
        self.material = coerce_numbers(self.material)
        if self.lines:
            if self.nodes or self.elements:
                raise ValueError(
                    "reinforcement: give either `lines` or `nodes` + `elements`,"
                    " not both"
                )
        elif not (self.nodes and self.elements):
            raise ValueError(
                "reinforcement needs either `lines` or `nodes` + `elements`"
            )
        return self


class ErosionSpec(Strict):
    """Element deletion, after Abaqus `ELEMENT DELETION` / LS-DYNA `ERODE`.

    A softening model has no answer for a fully degraded element: it still has
    nodes and still carries whatever residual force they carry, but no stiffness
    to resist it.  Both commercial codes answer by removing the element.
    """

    #: 损伤达到或超过它时积分点算失效（Abaqus 保留单元时 MAX DEGRADATION 为 0.99）
    #: damage at or above which the point counts as failed (Abaqus MAX
    #: DEGRADATION is 0.99 when elements are kept, 1.0 when they are deleted;
    #: MAT_CSCM erodes at d > 0.99)
    damage: float = Field(0.99, gt=0.0, le=1.0)
    #: 读哪个损伤变量：`d`（应力用的刚度退化）、`dt` 受拉、`dc` 受压
    #: which damage variable `damage` is read from: `d`, the stiffness
    #: degradation the stress carries (capped at the material's damage_max,
    #: so a threshold above the cap never fires); or `dt` / `dc`, the tensile
    #: and compressive variables on their own (CDP's DAMAGET / DAMAGEC,
    #: uncapped). `dc` is the spalling rule: crushed concrete goes, cracked
    #: concrete stays and keeps holding its bars -- on the Alberta beam a
    #: rule on `d` removed the cracked tension zone first.
    measure: Literal["d", "dt", "dc"] = "d"
    #: 同时必须超过的最大主 Green–Lagrange 应变，取单元中心
    #: max principal Green-Lagrange strain that must ALSO be exceeded, measured
    #: at the element centre. MAT_CSCM spells this as ERODE - 1 and FHWA's own
    #: deck uses 0.05. It is a KINEMATIC measure on purpose -- a plastic strain
    #: is rescaled by the crack band, so a threshold set for one element size
    #: would not carry to another (see continuum.strain_measures).
    strain: float = Field(0.05, ge=0.0)
    #: 单元内要有多少个积分点达到 `damage`——LS-DYNA 的 NUMFIP，默认 1
    #: how many of the element's integration points must reach `damage`
    #: -- LS-DYNA NUMFIP,
    #: whose default of 1 means one failed point out of eight deletes a fully
    #: integrated hexahedron. Defaults here to half the points, rounded up.
    points: Optional[int] = Field(None, ge=1)
    #: 无关判据的保险：体积塌缩或翻转到低于这个比例的单元无论材料怎么说都删
    #: the criterion-free safety net: an element whose volume has collapsed or
    #: inverted goes whatever the material says (LS-DYNA *CONTROL_TIMESTEP ERODE
    #: with a non-zero DTMIN; Abaqus distortion control exists for this too)
    min_volume_ratio: float = Field(0.2, gt=0.0)
    #: 体积膨胀超过这个倍数的单元删除（纠缠）
    #: an element whose volume has grown by more than this factor is deleted (tangling)
    max_volume_ratio: float = Field(5.0, gt=1.0)
    #: 每隔多少次迭代检查一次
    #: iterations between checks -- LS-DYNA checks DR convergence every
    #: NRCYCK = 250 cycles for the same reason
    every: int = Field(250, ge=1)


class PlaneSpec(Strict):
    #: 平面上一点 [x, y, z]
    #: a point on the plane
    point: List[float] = Field(min_length=3, max_length=3)
    #: 法向 [nx, ny, nz]，指向结构所在的一侧（允许的一侧）；自动单位化
    #: the normal, pointing to the side the structure lives on (the allowed side); normalised
    normal: List[float] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def _unit(self):
        if float(np.linalg.norm(self.normal)) <= 0.0:
            raise ValueError("a contact plane's normal must not be the zero vector")
        return self


class RigidSurfaceSpec(Strict):
    """One rigid, immovable surface the nodes of `region` cannot pass through:
    a penalty spring on the penetration along the normal,
    and Coulomb friction against the tangential velocity in a dynamic run.
    core/rigid_contact.py has the mechanics and the sizing of `auto`."""
    #: 平面：一点加法向；法向指向允许的一侧
    #: the plane, a point and the normal; the normal points to the allowed side
    plane: PlaneSpec
    #: 哪些节点受它约束（默认全部）
    #: the nodes it holds back (default: all)
    region: str = "all"
    #: 罚刚度（N/m）：`auto` 按节点质量和步长取，不缩小稳定步长（core/rigid_contact.py）
    #: penalty stiffness (N/m): `auto` sizes it per node from its mass and the step so the step never shrinks for it
    stiffness: Union[Literal["auto"], float] = "auto"
    #: 库仑摩擦系数（只用于动力计算；松弛拒绝）
    #: Coulomb friction coefficient (dynamic runs only; refused in relaxation)
    friction: float = Field(0.0, ge=0.0)
    #: 接触阻尼比 ζ（只用于动力计算）：节点在面内时并联一个阻尼器 2ζ√(km)，碰撞不再完全弹性，碎片会停下来；0.5 时恢复系数约 0.16
    #: contact damping ratio zeta (dynamic only): a dashpot 2 zeta sqrt(k m) while the node is inside, so an impact loses energy and debris settles; restitution about 0.16 at 0.5
    damping: float = Field(0.0, ge=0.0, le=1.0)
    #: 名字，用于报错和摘要
    #: a name for messages and the summary
    name: Optional[str] = None

    @model_validator(mode="after")
    def _stiffness(self):
        if self.stiffness != "auto" and float(self.stiffness) <= 0.0:
            raise ValueError("a contact stiffness is `auto` or a positive number")
        return self


class ContactBodySpec(Strict):
    """A set of line members that take part in the contact between members,
    and the radius of the capsule each of them is (1.11)."""
    #: 参与接触的构件所在的区域（两端节点都在区域内的构件；连接器不参与）
    #: the region whose members take part (a member is in when both its points are; connectors take no part)
    region: str = "all"
    #: 胶囊体半径（m）：构件表面到轴线的距离。`auto` 只用于杆和索（取与截面等面积的圆）；梁必须写
    #: the capsule's radius, the distance of the member's surface from its axis. `auto` is for bars and cables alone (the circle of their area); a beam must say
    radius: Union[Literal["auto"], float] = "auto"

    @model_validator(mode="after")
    def _radius(self):
        if self.radius != "auto" and float(self.radius) <= 0.0:
            raise ValueError("a contact radius is `auto` or a positive number")
        return self


class MemberContactSpec(Strict):
    """Contact between line members (1.11): each member a
    capsule about its axis; two that overlap push each other apart, with a
    dashpot and Coulomb friction in a dynamic step. Pairs that overlap in
    the reference configuration keep that overlap as an allowance until
    they have separated. core/member_contact.py has the mechanics."""
    #: 参与接触的构件组，每组一个区域和一个半径；各组之间、组内（不共节点的构件）都检查
    #: the bodies, a region and a radius each; every member is tested against every other, its own body's too (not the ones it shares a point with)
    bodies: List[ContactBodySpec] = Field(min_length=1)
    #: 罚刚度（N/m）：`auto` 按两构件的折算质量和步长取，不缩小稳定步长；写数时以它为上限
    #: penalty stiffness: `auto` sizes it from the pair's reduced mass and the step so the step never shrinks for it; a number is capped there
    stiffness: Union[Literal["auto"], float] = "auto"
    #: 库仑摩擦系数（只在动力步里起作用）
    #: Coulomb friction coefficient (acts in dynamic steps only)
    friction: float = Field(0.0, ge=0.0)
    #: 接触阻尼比 ζ（只在动力步里起作用）：碰撞不再完全弹性
    #: contact damping ratio zeta (dynamic steps only): an impact loses energy
    damping: float = Field(0.0, ge=0.0, le=1.0)
    #: 初始就重叠的构件对（支座上下的梁与盖梁、节点处的相邻构件）在接触起作用之前还能再靠近多少：较小半径的倍数（默认 5%）
    #: how much closer a pair that overlaps at the start (a girder and the cap under its bearing, neighbours at a joint) may come before the contact acts, as a fraction of the smaller of the two radii (default 5%)
    slack: float = Field(0.05, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _stiffness(self):
        if self.stiffness != "auto" and float(self.stiffness) <= 0.0:
            raise ValueError("a contact stiffness is `auto` or a positive number")
        return self


class ContactSpec(Strict):
    #: 刚性面列表（地面、海底、承台顶面……）
    #: the rigid surfaces (ground, sea floor, a pile cap ...)
    rigid: List[RigidSurfaceSpec] = []
    #: 线单元构件之间的接触（1.11）：落梁、梁体碰撞、倒塌堆积
    #: contact between line members (1.11): a falling span, pounding girders, the pile a collapse leaves
    members: Optional[MemberContactSpec] = None


class SpectrumSpec(Strict):
    """One design spectrum and the direction it acts in.
    The table is the user's: the library holds no code's spectrum."""

    #: 地震作用的方向（整体坐标，自动归一化）
    #: the direction the ground acceleration acts in (global, normalised)
    direction: List[float] = Field(min_length=3, max_length=3)
    #: 谱表 [[周期 s, 加速度], ...]，周期递增；与 `path` 二选一
    #: the spectrum as [[period s, pseudo-acceleration], ...], periods increasing; or `path`
    table: Optional[List[List[float]]] = None
    #: 文本文件，前两列为周期、加速度（空格、制表符或逗号分隔，其余列不读），相对算例文件；与 `table` 二选一
    #: a text file with period and acceleration in its first two columns (spaces, tabs or commas; further columns are not read), relative to the case file; or `table`
    path: Optional[str] = None
    #: 谱值乘的系数（谱以 g 给出时写 9.80665）
    #: multiplies the spectral values (9.80665 when the table is in g)
    scale: float = 1.0

    @model_validator(mode="after")
    def _one_source(self):
        if (self.table is None) == (self.path is None):
            raise ValueError("a spectrum is a `table` OR a `path`, one of them")
        if all(abs(c) < 1e-300 for c in self.direction):
            raise ValueError("a spectrum direction cannot be zero")
        if self.table is not None:
            if len(self.table) < 2 or any(len(r) != 2 for r in self.table):
                raise ValueError("a spectrum table is two or more [period, acceleration] rows")
            T = [r[0] for r in self.table]
            if any(b <= a for a, b in zip(T, T[1:])) or T[0] < 0:
                raise ValueError("spectrum periods must start at 0 or more and increase")
            if any(r[1] < 0 for r in self.table):
                raise ValueError("spectral accelerations cannot be negative")
        return self


class ResponseSpectrumSpec(Strict):
    """Response-spectrum analysis at the state the run ends in:
    the modes of the tangent there (as `vfife modal`), each mode's peak from
    the spectrum at its period, S_a / omega^2 times its participation, and
    the modal peaks combined -- per mode pair by CQC (Der Kiureghian, one
    damping ratio) or SRSS, then the directions by SRSS. Displacements,
    member end actions and support reactions are combined component by
    component; the peaks go to the result's `response_spectrum` group."""

    #: 谱与方向（每个方向一张谱；多个方向的结果按 SRSS 组合）
    #: the spectra and their directions (one per direction; directions combine by SRSS)
    spectra: List[SpectrumSpec] = Field(min_length=1)
    #: 振型组合：`cqc`（默认）或 `srss`
    #: how the modes combine: `cqc` (default) or `srss`
    combination: Literal["cqc", "srss"] = "cqc"
    #: CQC 用的振型阻尼比（谱本身应是按同一阻尼比给的）
    #: the modal damping ratio CQC's correlation uses (the spectrum should be for the same ratio)
    damping: float = Field(0.05, ge=0.0, lt=1.0)
    #: 取多少阶；不写：取到每个受激方向的有效质量累计达到 `mass_fraction`
    #: how many modes, lowest first; left out, as many as bring every excited direction to `mass_fraction`
    modes: Optional[int] = Field(None, ge=1)
    #: 不写 `modes` 时要达到的有效质量比
    #: the effective-mass fraction `modes` left out has to reach
    mass_fraction: float = Field(0.9, gt=0.0, le=1.0)
    #: 周期超出谱表时：`refuse`（默认，报出是哪几阶）或 `hold`（取端值）
    #: a period outside the table: `refuse` (default, naming the modes) or `hold` (the end value)
    beyond_table: Literal["refuse", "hold"] = "refuse"
    #: 有限差分切线的自由度上限（2n 次内力计算）
    #: refuse above this many dofs (the finite-difference tangent costs 2n force evaluations)
    max_dofs: int = Field(6000, ge=1)

    @model_validator(mode="after")
    def _modes_or_fraction(self):
        # `modes` silently overrode a written `mass_fraction`;
        # the default, as a dumped case carries it, is no conflict
        default = type(self).model_fields["mass_fraction"].default
        if self.modes is not None and "mass_fraction" in self.model_fields_set \
                and self.mass_fraction != default:
            raise ValueError("response_spectrum: give `modes` (how many) or `mass_fraction` "
                             "(how much mass), not both -- `modes` would decide alone")
        return self


class MassScalingSpec(Strict):
    """Selective mass scaling to a target explicit step:
    only the DOFs whose own stable step is below `target_dt` get mass, and
    exactly as much as lifts them to it. The mass is real -- gravity and a
    ground motion see it -- and more than `max_added_fraction` of the model's
    translational mass is refused. See vfife/core/mass_scaling.py."""

    #: 要达到的显式步长（s）
    #: the explicit step to reach (s)
    target_dt: float = Field(gt=0.0)
    #: 补的平动质量超过全模型的这个比例时拒绝
    #: refuse when the added translational mass exceeds this fraction of the model's
    max_added_fraction: float = Field(0.01, gt=0.0, le=1.0)


class ArcStopSpec(Strict):
    """Where an arc-length step ends when the load factor never reaches 1:
    the mean displacement of a region along a direction reaching `value`."""
    #: 节点区域
    #: the nodes
    region: str
    #: 方向
    #: the direction
    dof: Literal["x", "y", "z"] = "z"
    #: 到这个位移（带符号）就停
    #: stop when it reaches this displacement (signed)
    value: float


class ArcControlSpec(Strict):
    """The displacement an arc is measured on (1.3): one
    node's translation, instead of the norm of the whole state. Displacement
    control with the load factor as the unknown -- OpenSees' DisplacementControl
    integrator -- so the pushed point moves on at every increment and the
    path cannot turn back down an unloading branch; it cannot follow a
    snap-back of that displacement."""
    #: 一个节点的区域
    #: a region of ONE node
    region: str
    #: 方向
    #: the direction
    dof: Literal["x", "y", "z"] = "z"


class ArcLengthSpec(Strict):
    """Crisfield's cylindrical arc length: the load factor
    becomes an unknown, so the path can pass a limit point -- a pushover's
    peak, a snap-through, a snap-back."""
    #: 第一个增量的荷载因子（据此定弧长）
    #: the first increment's load factor (which sets the arc length)
    initial: float = Field(0.1, gt=0.0, le=1.0)
    #: 每步增量数上限
    #: increments per step, at most
    max_increments: int = Field(200, ge=1)
    #: 每个增量期望的迭代次数：弧长按 sqrt(期望/实际) 伸缩
    #: iterations an increment should take: the arc scales by sqrt(desired / taken)
    target_iterations: int = Field(5, ge=1)
    #: 荷载因子到不了 1 时的停止条件
    #: where to stop when the load factor does not reach 1
    stop: Optional[ArcStopSpec] = None
    #: 用弧长法的分析步名；不给则所有步都用（其余步用荷载控制的 Newton）
    #: the steps that use it, by name; all of them when left out (the others take load-controlled Newton)
    steps: Optional[List[str]] = None
    #: 弧长只量这一个位移（1.3）：位移控制，荷载因子是未知量；被控位移不会往回走，跟不了它自己的回跳；不给则量整个状态
    #: measure the arc on this one displacement (1.3): displacement control with the load factor unknown; the controlled displacement never goes back, and cannot follow its own snap-back; the whole state's norm when left out
    control: Optional[ArcControlSpec] = None


class StaticSpec(Strict):
    """The implicit static solver's settings."""
    #: 弧长法：荷载因子成为未知量，能越过极限点
    #: arc length: the load factor becomes an unknown and the path passes limit points
    arc_length: Optional[ArcLengthSpec] = None
    #: 每个增量的 Newton 迭代上限
    #: Newton iterations per increment
    max_iterations: int = Field(30, ge=1)
    #: 不收敛时增量减半的次数上限
    #: how many times an increment may be halved when it does not converge
    cutbacks: int = Field(10, ge=0)
    #: 回溯线搜索（只用于荷载控制的增量；弧长法的校正步没有线搜索）
    #: backtracking line search on the residual norm (load-controlled increments only; the arc-length corrector has none)
    line_search: bool = True
    #: `full`：每次迭代重算切线；`modified`：每个增量开头算一次，收敛变慢（残差没减半）时重算
    #: `full`: a new tangent every iteration; `modified`: once per increment, renewed when an iteration fails to halve the residual
    tangent: Literal["full", "modified"] = "full"
    #: 有限差分切线的步长：平动为它乘单元长度的中位数，转动为它乘 10（弧度）
    #: the finite-difference step of the tangent: this times the median element length for a translation, ten times it (radians) for a rotation
    perturbation: float = Field(1.0e-7, gt=0.0)
    #: Newton 用尽减半次数仍不收敛时怎么办（1.7）：`none`（默认）报极限点停下；`relaxation` 从上一个收敛状态起，这一级和本步余下的各级都改用动力松弛（虚拟质量、动能阻尼，与 `mode: relaxation` 同一算法；下一步再从 Newton 开始）——越过跳跃之后落在哪一支由这段虚拟的动力过程决定，软化截面上未必是 `mode: relaxation` 走的那支
    #: what happens when Newton has used up its cutbacks (1.7): `none` (the default) stops at the limit point; `relaxation` takes this level and the rest of the step by dynamic relaxation from the last converged state (fictitious mass, kinetic damping, the algorithm of `mode: relaxation`; the next step starts with Newton again) -- which branch it lands on past a jump is the fictitious transient's choice, and on a softening section not necessarily the one `mode: relaxation` follows
    fallback: Literal["relaxation", "none"] = "none"
    #: 回落松弛每级的迭代上限（默认 solver.max_steps_per_load）
    #: the iteration budget of each relaxed level (default solver.max_steps_per_load)
    fallback_iterations: Optional[int] = Field(None, ge=1)


class SolverSpec(Strict):
    #: `relaxation` 动力松弛求静力解，`static` 隐式静力（Newton），或 `dynamic` 显式时程
    #: `relaxation` (dynamic relaxation to a static state), `static` (implicit, Newton) or `dynamic` (explicit time history)
    mode: Literal["dynamic", "relaxation", "static"] = "relaxation"
    #: 隐式静力的设置（见 StaticSpec）
    #: the implicit static solver's settings (see StaticSpec)
    static: StaticSpec = StaticSpec()
    #: 松弛：`underwood`（参考刚度的 Gershgorin 行和，默认）或 `eigenvalue`（各单元最大特征值，大转动下仍成立）；动力：`physical`
    #: relaxation: `underwood` (Gershgorin row sums of the reference
    #: stiffness, the default) or `eigenvalue` (each element's largest
    #: eigenvalue, a bound that survives large rotations -- see
    #: elements/continuum.stiffness_max_eigenvalues); dynamic: `physical`
    mass: Optional[Literal["physical", "underwood", "eigenvalue"]] = None
    #: 见 DampingSpec；松弛默认 kinetic，动力默认无阻尼
    #: see DampingSpec; kinetic by default under relaxation, none under dynamic
    damping: DampingSpec = DampingSpec()

    # dynamic mode
    #: 动力：模拟时长（s），必填
    #: dynamic: simulated duration (s); required
    total_time: Optional[float] = None
    #: 实际步长占稳定步长的比例
    #: fraction of the stable explicit step actually used
    dt_safety: float = 0.6
    #: 动力、连续体：均匀放大质量把步长抬到这个值（选择性的见 `mass_scaling`）
    #: dynamic, continuum: raise the step to this by uniformly scaling the mass (see `mass_scaling` for the selective form)
    target_dt: Optional[float] = None

    # relaxation mode
    #: 松弛：迭代预算；每个分析步都自带预算时可省
    #: relaxation: iteration budget; may be left out when every step carries its own
    n_steps: Optional[int] = None
    #: 松弛：残差与荷载之比低于它即收敛
    #: relaxation: converged when the residual-to-load ratio falls below this
    tolerance: float = 1e-4
    #: 松弛：同时动能比要低于它
    #: relaxation: and the kinetic-energy ratio below this
    kinetic_tolerance: float = Field(1e-10, gt=0.0)
    #: 松弛：稳定步长的比例，同动力的 `dt_safety`
    #: relaxation: fraction of the stable step, as `dt_safety` for dynamic
    relaxation_safety: float = 0.6
    #: 单元删除；不写即不删
    #: omit entirely to disable element deletion
    erosion: Optional[ErosionSpec] = None
    #: 分级：每一级保持到平衡，每个输出帧都是荷载–位移曲线上的真实点
    #: hold each load level until equilibrium; every output frame is then
    #: a genuine point on the load-displacement curve
    load_steps: Optional[int] = Field(None, ge=1)
    #: 带 `load_steps` 的松弛：每级迭代预算
    #: relaxation with `load_steps`: iteration budget per level
    max_steps_per_load: int = 5000
    #: 每级荷载渐入所用的迭代数（默认 max_steps_per_load/5）
    #: iterations over which each load level is blended in (default max_steps_per_load / 5)
    stage_ramp_iterations: Optional[int] = None

    #: 开始时的混凝土龄期（天）；只有材料声明了 creep/shrinkage/ageing 才读
    #: concrete age in days at the start of the run. Only a
    #: model whose material declares `creep`, `shrinkage` or `ageing` reads it.
    age: Optional[float] = Field(None, gt=0.0)

    #: 选择性质量缩放：只在 beam2、beam2_timoshenko、beam2_fiber、mixed_line 上；其他拒绝
    #: selective mass scaling: beam2, beam2_timoshenko,
    #: beam2_fiber and mixed_line only; refused elsewhere
    mass_scaling: Optional[MassScalingSpec] = None
    #: 从本模型早先一次结果的终态起步——位置、转角、材料状态，静止、t = 0。典型用法：先松弛得恒载状态，再从它做动力。相对算例文件；结果的网格必须是本算例的网格
    #: start from the final state of an earlier result of THIS model -- its
    #: positions, rotations and material state, at rest, at t = 0 -- instead
    #: of from the mesh. The typical pair: a relaxation run
    #: for the dead state, then a dynamic run from it. Relative to the case
    #: file; the result's mesh must be this case's mesh.
    initial_state: Optional[str] = None

    #: 实体单元：每步从变形梯度提取转动的极分解迭代次数；梁、壳、桁架不读它
    #: solid elements: polar-decomposition iterations that extract the
    #: rotation from the deformation gradient each step. Line, shell and truss
    #: models do not read it (it was documented as a beam2 setting)
    rotation_iterations: int = 2
    #: `numpy`（参考实现）、`numba`（编译内核，同样答案）或 `warp`（GPU）
    #: 'numpy' (reference) or 'numba' (fused scalar kernel, same answer)
    backend: Literal["numpy", "numba", "warp"] = "numpy"
    #: numba 纤维梁与框架内核的线程数：1 串行（默认），0 用全部核，n 用 n 个；结果与串行逐位相同
    #: threads for the numba fibre-beam and frame kernels: 1 serial (the default), 0 all cores, n that many; the answer is the serial one to the bit
    threads: int = Field(1, ge=0)
    #: 只对 warp："cuda"、"cuda:0"、"cpu" 等
    #: warp only: "cuda", "cuda:0", "cpu" ...
    device: str = "cuda"
    #: 只对 warp：`mixed`（单元 fp32、节点 fp64）或 `fp64`（验证用）
    #: warp only: "mixed" (fp32 elements, fp64 nodal) or "fp64" (verification)
    precision: Literal["mixed", "fp64"] = "mixed"

    @model_validator(mode="after")
    def _mode_fields(self):
        if self.mode == "dynamic" and self.total_time is None:
            raise ValueError("solver.total_time is required when mode = dynamic")
        # DAMPING THAT MEANS SOMETHING IN THIS MODE, OR A REFUSAL.
        # `alpha` and `beta` were applied under some types
        # by some solvers and silently dropped by the rest; `kinetic` in a
        # dynamic run did nothing on frames and zeroed the velocity on trusses
        # and solids.
        d = self.damping
        if self.mode != "static" and d is not None:
            t = d.type or ("kinetic" if self.mode == "relaxation" else "viscous")
            if self.mode == "dynamic" and t == "kinetic":
                raise ValueError("damping.type: kinetic is a relaxation device (it zeroes the velocity "
                                 "at every kinetic-energy peak); a dynamic run takes viscous, rayleigh "
                                 "or none")
            # a relaxation model may have dynamic steps (1.2): its damping is
            # theirs, and the Case checks that there are some
            if d.beta and not (t == "rayleigh" and self.mode in ("dynamic", "relaxation")):
                raise ValueError("damping.beta is stiffness-proportional Rayleigh damping: it needs "
                                 "type: rayleigh in a dynamic run")
            if d.alpha and t not in ("rayleigh", "kinetic"):
                raise ValueError(f"damping.alpha under type: {t} would not be applied: use "
                                 "type: rayleigh (alpha M v), or alongside kinetic damping in a "
                                 "relaxation")
        # `n_steps` under relaxation is checked on the Case, not here: a staged
        # step declares its own budget (load_steps * max_steps_per_load), and
        # making the analyst copy that product into a second place was how
        # every case file grew an `n_steps: 750010`.
        return self


class MonitorSpec(Strict):
    """One scalar time history, written to `monitors/<name>` in the result.

    Four kinds, and the split is which THING the monitor reduces over:

      reaction, displacement   the NODES of a region
      element                  the ELEMENTS a region owns -- every node of the
                               element inside it, the rule a section
                               assignment already uses
      bar                      the SEGMENTS of one reinforcement set

    The last two exist because a verification contract can only read monitors
    and whole-model field extrema. Before them, case_15 could not assert the
    tendon stress it is built around (bar stress lives in `rebar_steps/`,
    which contracts cannot read) and had to stand in the camber for it, and it
    could not say "the constant-moment region has not crushed" -- only "no
    element anywhere has", which the anchorage zone decides. Both of those
    limits are recorded in that case's contract; both are now expressible.
    """

    #: reaction / displacement（区域的节点）、element（区域的单元）、bar（钢筋组的段）、fiber（截面上一点）、vehicle（车辆，`of` 为车名）
    #: reaction / displacement (nodes of a region), element (elements of a region), bar (segments of a reinforcement set), fiber (one section point), vehicle (a vehicle named by `of`)
    kind: Literal["reaction", "displacement", "element", "bar", "fiber", "vehicle"]
    #: 读的区域（节点类必填；element/bar 是缩小范围）
    #: the region read (nodal kinds: required; element/bar: narrows to it)
    region: Optional[str] = None
    #: 节点类：几个区域一起读
    #: nodal kinds: several regions read together
    regions: Optional[List[str]] = None
    #: reaction/displacement：哪个分量
    #: reaction/displacement: the component
    dof: Literal["x", "y", "z"] = "z"
    #: element：实体是材料的任一单元输出（`dt`、`dc`、`d`……）加 `stress`、`von_mises`；线单元模型是任一构件场（`axial_force`、`bending_y`、`*_max_abs`、`*_i/_j`、`capacity_ratio`……）；bar：`axial_stress`、`axial_force`、`plastic_strain`
    #: element: on solids any per-element output of the material (`dt`, `dc`,
    #: `d`, ...) plus `stress` and `von_mises`; on line models any member field
    #: (`axial_force`, `bending_y`, `*_max_abs`, `*_i/_j`, `capacity_ratio`, ...,
    #: v1.0). bar: `axial_stress`, `axial_force`, `plastic_strain`.
    field: Optional[str] = None
    #: 张量场取哪个分量（本库统一的 Voigt 顺序 xx, yy, zz, xy, yz, zx；`xz` 作 `zx` 读）
    #: which component of a tensor field, in this package's one Voigt order xx, yy, zz, xy, yz, zx (`xz` is read as `zx`)
    component: Optional[Literal["xx", "yy", "zz", "xy", "yz", "zx", "xz"]] = None
    #: bar：钢筋组的名字，即其 assign 上的 `name`
    #: bar: the reinforcement set's name, i.e. the `name` on its assignment
    of: Optional[str] = None
    #: vehicle：第几根轴（contact_force、wheel、station），从 0 起
    #: vehicle: which axle (contact_force, wheel, station), from 0
    axle: int = Field(0, ge=0)
    #: element 监测量（实体与线单元模型）和实体的 bar 监测量：单元或钢筋段上的值怎么归约（max 默认、min、mean、absmax=绝对值最大者带符号）；其余监测量不接受 reduce
    #: element monitors (solids and line models) and bar monitors on solids: how the values over the elements/segments are reduced (max by default; absmax is the signed value of largest magnitude); the other kinds refuse it
    reduce: Optional[Literal["max", "min", "mean", "absmax"]] = None
    #: displacement：相对这个区域的平均位移报告——一个部件相对另一个的运动
    #: displacement: report it relative to the mean displacement of this
    #: region -- the motion of one part of a body seen from another
    relative_to: Optional[str] = None
    #: fiber（只对 beam2_fiber）：离区域节点最近的积分站的截面，读 [y, z] 处：`field: strain` 是该点的平截面应变，`field: stress` 是最近纤维的应力——桥面或底缘应变片读到的量
    #: fiber (beam2_fiber only): the section at the station
    #: nearest the region's node, read at the point [y, z] of the section:
    #: `field: strain` is the plane-section strain eps0 - y kz + z ky there
    #: (the concrete's, no tendon prestrain), `field: stress` the stress of
    #: the fibre closest to the point. What a strain gauge on the deck or the
    #: soffit reads, and what a thesis reports as 上缘/下缘.
    at: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    #: fiber、`field: stress`：只在混凝土纤维（`concrete`）或钢筋/预应力束（`steel`）里找最近的纤维；不给则在全部纤维里找。读的是哪根纤维（坐标、材料、种类）记在结果的监测量属性里
    #: fiber, `field: stress`: look for the nearest fibre among the concrete
    #: fibres only (`concrete`) or the bars and tendons only (`steel`); left
    #: out, among all of them. Which fibre was read (coordinates, material,
    #: kind) is recorded on the monitor in the result.
    fiber_kind: Optional[Literal["concrete", "steel"]] = None

    @model_validator(mode="after")
    def _one_of(self):
        if self.kind == "vehicle":
            if not self.of:
                raise ValueError("a 'vehicle' monitor needs 'of': the vehicle's name")
            if self.field not in ("heave", "pitch", "acceleration", "contact_force", "wheel", "station"):
                raise ValueError("a 'vehicle' monitor reads field: heave, pitch, acceleration, "
                                 "contact_force, wheel or station")
            if self.region or self.regions or self.at is not None:
                raise ValueError("a 'vehicle' monitor reads the vehicle, not a region")
            return self
        if self.kind == "fiber":
            if self.region is None or self.regions:
                raise ValueError("a 'fiber' monitor needs one 'region': the node whose section it reads")
            if self.field not in ("strain", "stress"):
                raise ValueError("a 'fiber' monitor reads field: strain or field: stress")
            if self.at is None:
                raise ValueError("a 'fiber' monitor needs at: [y, z], the point of the section it reads")
            if self.fiber_kind is not None and self.field != "stress":
                raise ValueError("fiber_kind chooses the fibre a `field: stress` reading takes; "
                                 "a strain is the plane section's at the point, whatever is there")
            return self
        if self.at is not None:
            raise ValueError("'at' belongs to a fiber monitor")
        if self.fiber_kind is not None:
            raise ValueError("'fiber_kind' belongs to a fiber monitor")
        nodal = self.kind in ("reaction", "displacement")
        if self.relative_to is not None and self.kind != "displacement":
            raise ValueError("relative_to applies to displacement monitors only")
        if nodal:
            if self.region is None and not self.regions:
                raise ValueError("a monitor needs 'region' or 'regions'")
            for f in ("field", "component", "of"):
                if getattr(self, f) is not None:
                    raise ValueError(
                        f"a {self.kind!r} monitor reads nodes and has no {f!r}; "
                        "did you mean kind: element or kind: bar?")
            return self
        # element and bar: the region is OPTIONAL and means "narrow it to
        # here". Left out, the reduction runs over the whole model, which is
        # what a contract's `source: field` already does.
        if self.regions:
            raise ValueError(
                f"a {self.kind!r} monitor takes one 'region', not 'regions': "
                "a union of regions has no single set of elements")
        if self.field is None:
            raise ValueError(f"a {self.kind!r} monitor needs 'field'")
        if self.kind == "bar" and not self.of:
            raise ValueError(
                "a 'bar' monitor needs 'of': the name of the reinforcement "
                "set to read, i.e. the `name` on its assignment")
        if self.kind == "element" and self.of is not None:
            raise ValueError("'of' names a reinforcement set; use it with kind: bar")
        if self.component is not None and self.field != "stress":
            raise ValueError(
                f"'component' selects part of a tensor and {self.field!r} is a "
                "scalar; drop it")
        if self.field == "stress" and self.component is None:
            raise ValueError(
                "field: stress is a tensor -- add `component:` (xx, yy, zz, "
                "xy, yz or zx)")
        return self


#: series the result writes under monitors/ itself
RESERVED_MONITOR_NAMES = ("time", "residual", "kinetic_energy", "age")


class OutputSpec(Strict):
    #: HDF5 结果文件；相对路径按当前工作目录（运行命令的目录）解析，不是算例文件所在目录；旁边生成 .xdmf 供 ParaView
    #: the HDF5 result file; a relative path is relative to the working directory (where the command is run), not to the case file; an .xdmf beside it for ParaView
    path: str = "results.h5"
    #: 每隔多少步写一帧；动力与松弛分析的监测量在同样的帧上记录，静力（mode: static）分析的监测量在每个增量都记录、只有帧被稀疏（`time` 总与每条监测量一样长）；静力分析不写 `every` 时每个增量一帧
    #: write a frame every this many steps; in dynamic and relaxation runs the monitors are recorded at the same frames, in static runs (mode: static) at every increment and only the frames are thinned (`time` is always as long as each monitor); a static run that does not write `every` writes a frame at every increment
    every: int = 200
    #: 要写的节点/单元场；`vfife describe <单元>` 列出模型能给的；写出来的名字模型给不了即拒绝（UNKNOWN_OUTPUT_FIELD）。不写时按模型取默认（1.1）：位移、转角、反力、实体与杆的 von_mises、壳的 von_mises_top/bottom、构件的带符号内力，有塑性铰、连接器、可断构件时另加它们的场；模型给不了的不写、也不报
    #: nodal/element fields to write; `vfife describe <element>` lists what the model can produce, and a name written here that the model cannot produce is refused (UNKNOWN_OUTPUT_FIELD). Left out, the default is the model's (1.1): displacement, rotation, reactions, von_mises on solids and bars, von_mises_top/bottom on shells, the members' signed forces, and the fields of its plastic hinges, connectors and breakable members where it has them; a default name the model does not produce is skipped without a word
    fields: List[str] = ["displacement", "von_mises"]
    #: 每隔多少步写一个重启检查点到结果旁的 `<结果>.ckpt.h5`（在帧或进度检查处取，0 = 只在结束时）
    #: write a restart checkpoint to `<result>.ckpt.h5` beside the result every this many steps (taken at a frame or progress check; 0 = only at the end)
    checkpoint_every: int = 0
    #: 命名的标量时程，见 MonitorSpec；`time`、`residual`、`kinetic_energy`、`age` 是保留名
    #: named scalar time histories, see MonitorSpec; `time`, `residual`, `kinetic_energy` and `age` are reserved
    monitors: Dict[str, MonitorSpec] = {}

    @field_validator("monitors")
    @classmethod
    def _reserved_monitor_names(cls, v):
        # the result writes these series itself; a monitor of the same name
        # crashed the run's end after the whole analysis (`time`, `residual`,
        # `kinetic_energy`) or was silently replaced (`age`)
        bad = [n for n in v if n in RESERVED_MONITOR_NAMES]
        if bad:
            raise ValueError(f"monitor name {bad[0]!r} is reserved (the result writes "
                             f"{', '.join(RESERVED_MONITOR_NAMES)} itself); rename the monitor")
        return v


class BlockSpec(Strict):
    """One continuum material block: the elements of `region` (every node of
    the element inside it) carry `material`. Produced by the input model when
    more than one solid section is assigned, or one to a region other than
    `all`; listed in assignment order, and a later block takes the elements
    an earlier one also covers (Abaqus' section-assignment rule)."""
    name: str
    region: str
    material: Dict[str, Any]
    formulation: Optional[Literal["full", "reduced", "b-bar", "selective"]] = None
    hourglass_coeff: Optional[float] = None

    @model_validator(mode="after")
    def _typed(self):
        self.material = coerce_numbers(self.material)
        if "type" not in self.material:
            raise ValueError(f"block {self.name!r}: material.type is required")
        return self


#: the case-file format this version reads and writes
CASE_FORMAT = 1


class Case(Strict):
    #: 算例格式版本：1（1.0 版起）。不写时与 1 按同样的规则读（旧拼写带 DEPRECATED_FIELD 警告照读）；写了比 1 大的数则拒绝（那是更新的 vfife 写的）
    #: the case-file format: 1 (since v1.0). Absent, the case is read by the same rules as 1 (old spellings read with a DEPRECATED_FIELD warning); a larger number is refused (a newer vfife wrote it)
    version: Optional[int] = None
    name: str = "case"
    units: Dict[str, str] = {}
    #: old spellings the case used: (where, old, new) -- preflight warns
    _deprecated: List[Any] = PrivateAttr(default_factory=list)
    #: unit-carrying defaults written in the declared units (core/units.py)
    _unit_defaults: List[Any] = PrivateAttr(default_factory=list)
    #: what the high-level blocks add to the mesh (vfife/prep): `cli.build` reads it with the file
    _mesh_overlay: Any = PrivateAttr(None)
    #: (who, what) a high-level block followed only roughly -- preflight warns
    _expansion_notes: List[Any] = PrivateAttr(default_factory=list)
    #: each tendon of `tendons:` as expanded: its bars' regions and forces -- the result's `tendons` group
    _tendon_report: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _format(self):
        if self.version is not None and self.version < 0:
            raise ValueError(f"`version: {self.version}` is not a case format; this vfife reads "
                             f"format {CASE_FORMAT} (write `version: {CASE_FORMAT}`)")
        if self.version is not None and not 0 <= self.version <= CASE_FORMAT:
            raise ValueError(
                f"`version: {self.version}`: this case was written for a newer case format than "
                f"this vfife reads ({CASE_FORMAT}); upgrade vfife, or rewrite the case for format {CASE_FORMAT}")
        return self
    mesh: MeshSpec
    element: ElementSpec = ElementSpec()
    material: Dict[str, Any]
    #: per-region continuum materials; empty means `material` is the whole mesh
    blocks: List[BlockSpec] = []
    regions: Dict[str, RegionSpec] = {}
    constraints: List[ConstraintSpec] = []
    loads: List[LoadSpec] = []
    nodal_masses: List[NodalMassSpec] = []
    #: 水下构件的 Morison 附加质量
    #: the Morison added mass of members under water
    hydrodynamic_mass: List[HydrodynamicMassSpec] = []
    #: 桩的土弹簧：按深度给的 p–y 曲线，非线性弹性（1.2，见 SoilSpringSpec）
    #: soil springs on piles: p-y curves by depth, nonlinear elastic (1.2, see SoilSpringSpec)
    soil_springs: List[SoilSpringSpec] = []
    #: floors that are rigid in their own plane
    diaphragms: List[DiaphragmSpec] = []
    #: named amplitude curves; see core/amplitude.py. `ramp` and `hold` exist
    #: without being declared.
    amplitudes: Dict[str, Any] = {}
    steps: List[StepSpec] = []
    reinforcement: List[ReinforcementSpec] = []
    members: List[MemberGroupSpec] = []
    #: members taken out at given times
    removals: List[RemovalSpec] = []
    #: 刚性面接触：节点碰到地面/海底/承台即受罚力
    #: rigid-surface contact: a node meeting the ground, the sea floor, a pile cap
    contact: Optional[ContactSpec] = None
    #: 反应谱分析：在计算结束时的状态上做模态、按谱组合
    #: response-spectrum analysis at the state the run ends in
    response_spectrum: Optional[ResponseSpectrumSpec] = None
    solver: SolverSpec
    output: OutputSpec = OutputSpec()

    @model_validator(mode="after")
    def _spectrum_needs_an_equilibrium(self):
        """A response spectrum is linear about a state: the modes of the
        tangent there. A relaxation run ends at an equilibrium (the loaded
        state, or the drawing when nothing is loaded); a dynamic run ends
        wherever the motion was, which is not a state to linearise about."""
        if self.response_spectrum is not None and self.solver.mode == "dynamic":
            raise ValueError("response_spectrum needs solver.mode = relaxation (or static): the modes are "
                             "those of an equilibrium, and a dynamic run does not end at one")
        return self

    @model_validator(mode="after")
    def _breaking_is_line_dynamics(self):
        """Member failure and scheduled removal are implemented in the line
        solvers' dynamic runs (v0.4 mixed_line, v0.5 the rest:
        beam2, beam2_timoshenko, truss2, and removals on beam2_fiber, whose
        members break through `fracture`). Anywhere else they would be read
        and ignored -- a model that SAYS its cables can snap and whose cables
        cannot is the kind of silent gap v0.4 went after. Under relaxation
        a break inside the iterations would be a break in a state that is not
        an equilibrium, so it is refused too."""
        failing = ([m.region for m in self.members if m.failure is not None]
                   + (["element"] if self.element.failure is not None else []))
        if not (failing or self.removals):
            return self
        what = "member failure" if failing else "removals"
        allowed = {"mixed_line", "beam2", "beam2_timoshenko", "truss2"} | ({"beam2_fiber"} if not failing else set())
        if self.element.type not in allowed:
            raise ValueError(f"{what} is implemented for line models ({', '.join(sorted(allowed))}), "
                             f"not {self.element.type!r}")
        if self.element.type == "mixed_line" and self.element.failure is not None:
            raise ValueError("a mixed_line model states failure per member group (assign[].failure)")
        if self.element.failure is not None and self.element.failure.rotation_capacity is not None:
            raise ValueError("failure.rotation_capacity (plastic hinges) is implemented for "
                             "mixed_line member groups (assign[].failure), not for element.failure")
        wants_split = ([m.region for m in self.members if m.failure is not None and m.failure.split]
                       + [r.region for r in self.removals if r.split]
                       + (["element"] if self.element.failure is not None and self.element.failure.split else []))
        if wants_split and self.element.type not in ("mixed_line", "beam2", "beam2_timoshenko"):
            raise ValueError(f"split: true on {', '.join(wants_split)}: a broken member splits off onto "
                             f"its own particles in the frame solvers (beam2, beam2_timoshenko, mixed_line), "
                             f"not in {self.element.type!r} (a fibre member splits through element.fracture.split)")
        if self.solver.mode == "static":
            # the implicit static solver (1.7) breaks a member in an ACCEPTED
            # equilibrium and solves the level again without it -- the
            # alternate-path analysis; what needs a clock or a motion stays
            # with mode: dynamic
            if self.removals:
                raise ValueError("removals at a time need solver.mode = dynamic (a static step has no clock); "
                                 "under mode: static take a member out with a step's `deactivate`")
            if wants_split:
                raise ValueError(f"split: true on {', '.join(wants_split)} needs solver.mode = dynamic: a member "
                                 "that splits off falls, and a fragment has no static equilibrium; under "
                                 "mode: static a broken member is taken out")
            return self
        if self.solver.mode != "dynamic":
            # said for the mode it is (v1.0: `mode: static` was told
            # about relaxation iterations)
            raise ValueError(f"{what} needs solver.mode = dynamic or static (this case is mode: {self.solver.mode}): "
                             "a member that breaks inside relaxation iterations breaks in a state that is not "
                             "an equilibrium, and a removed member's load is released as a jump only a "
                             "time history can follow")
        return self

    @model_validator(mode="after")
    def _dynamic_steps_are_time(self):
        """A dynamic step lasts `solver.total_time` at the stable step: it has
        no iteration budget and no load levels. `load_steps`
        was accepted there, staged nothing and switched the default amplitude
        (the response at 0.25 s halved); a step's `n_steps` ran 10 steps on
        a solid, 337 on a frame and 60 on a truss."""
        if self.solver.mode == "dynamic" and self.solver.load_steps:
            raise ValueError("solver.load_steps holds load levels to equilibrium in a relaxation; "
                             "a dynamic run has no levels (shape the load with an amplitude)")
        for st in self.steps:
            # each step by its OWN procedure (1.2)
            if (st.procedure or self.solver.mode) != "dynamic":
                if st.total_time is not None:
                    raise ValueError(f"step {st.name!r}: total_time is a dynamic step's duration; "
                                     f"this step is {st.procedure or self.solver.mode}")
                continue
            if st.load_steps:
                raise ValueError(f"step {st.name!r}: load_steps is a relaxation setting; a dynamic "
                                 "step has no load levels (shape the load with an amplitude)")
            if st.n_steps is not None:
                raise ValueError(f"step {st.name!r}: n_steps is a relaxation budget; a dynamic step "
                                 "lasts solver.total_time at the stable time step")
        return self

    @model_validator(mode="after")
    def _relaxation_has_a_budget(self):
        """Under relaxation every step must have an iteration budget from
        SOMEWHERE: the solver's `n_steps`, the step's own `n_steps`, or a
        staged step's `load_steps * max_steps_per_load`. The last is the
        common case and needs nothing at solver level."""
        if self.solver.mode != "relaxation" or self.solver.n_steps is not None:
            # a relaxation STEP of another mode's model has its own budget (1.2)
            for st in self.steps:
                if st.procedure == "relaxation" and self.solver.mode != "relaxation" \
                        and self.solver.n_steps is None and st.n_steps is None and not st.load_steps:
                    raise ValueError(f"step {st.name!r} relaxes in a {self.solver.mode} model: give it "
                                     "`load_steps` or its own `n_steps`")
            return self
        if not self.steps:
            raise ValueError("solver.n_steps is required when mode = relaxation "
                             "and there are no steps to take a budget from")
        relaxing = [st for st in self.steps if (st.procedure or "relaxation") == "relaxation"]
        for st in relaxing:
            if st.n_steps is None and not st.load_steps:
                raise ValueError(
                    f"step {st.name!r} has no iteration budget: give it "
                    "`load_steps` (staged, budget = load_steps * max_steps_per_load), "
                    "its own `n_steps`, or set solver.n_steps")
        # AND WRITE THE SUM DOWN, HERE, ONCE. Saying the budget may be left out
        # was only true of the continuum solver, which derived it for itself;
        # the five line and shell solvers read `sol["n_steps"]` and died on
        # `int(None)` -- a TypeError from inside the solver for a file the
        # schema had just accepted (found writing the user manual, 2026-09-25).
        # Deriving it where it is validated gives every solver the same number
        # and leaves none of them a default to disagree about.
        self.solver.n_steps = sum(
            int(st.n_steps) if st.n_steps is not None
            else int(st.load_steps) * int(st.max_steps_per_load or self.solver.max_steps_per_load)
            for st in relaxing) or 1
        return self

    @model_validator(mode="after")
    def _travelling_loads(self):
        """What a `moving_load` or `vehicle` is refused with, by name, before
        the solver sees it: each of these used to run and
        answer something else."""
        steps = self.steps or [StepSpec()]
        every = [ld for ld in self.loads] + [ld for st in steps for ld in st.loads]
        vehicles = [ld.vehicle for ld in every if ld.vehicle is not None]
        moving = [ld.moving_load for ld in every if ld.moving_load is not None]
        if not vehicles and not moving and not any(
                m.kind == "vehicle" for m in self.output.monitors.values()):
            return self
        if (vehicles or moving) and self.element.type in ("shell4", "shell3", "shell_line"):
            raise ValueError(f"moving_load and vehicle travel along beam members; a "
                             f"{self.element.type!r} model has none to carry them (a shell deck "
                             "takes its wheels as vector loads)")
        if vehicles and self.solver.mode != "dynamic":
            raise ValueError("a vehicle needs solver.mode = dynamic: it has mass and a clock")
        if self.solver.mode != "dynamic" and any(m.speed is not None for m in moving):
            raise ValueError("a moving_load with `speed` crosses in physical time and needs solver.mode = "
                             "dynamic; a static or relaxation run has no clock (its `time` is the load "
                             "factor) -- give `stop` instead to sweep the stations")
        arc = self.solver.static.arc_length if self.solver.mode == "static" else None
        if arc is not None:
            for st in steps:
                if arc.steps and st.name not in arc.steps:
                    continue
                if any(ld.moving_load is not None for ld in list(self.loads) + list(st.loads)):
                    raise ValueError("a moving_load under static.arc_length: the arc makes the load factor "
                                     "an unknown, and a travelling load's factor is its POSITION, not its "
                                     "size -- sweep it under load control (no arc_length, or leave this "
                                     "step out of arc_length.steps)")
        # names: the explicit ones reserved first, then vehicle, vehicle_2, ...
        # -- two unnamed vehicles were one vehicle counted twice. PER STEP:
        # the first unnamed vehicle of every step is
        # `vehicle`, so `of: vehicle` follows each step's own; numbered over
        # the whole case, step 2's was `vehicle_2` and the monitor read step
        # 1's frozen one. A model-level vehicle is in every step and is
        # named first, clear of every explicit name.
        def _name(unnamed, taken):
            k = 1
            for v in unnamed:
                while (name := "vehicle" if k == 1 else f"vehicle_{k}") in taken:
                    k += 1
                v.name = name; taken.add(name); k += 1
        top = [ld.vehicle for ld in self.loads if ld.vehicle is not None]
        _name([v for v in top if v.name is None],
              {v.name for v in vehicles if v.name is not None})
        for st in steps:
            own = [ld.vehicle for ld in st.loads if ld.vehicle is not None]
            _name([v for v in own if v.name is None],
                  {v.name for v in top + own if v.name is not None})
        for st in steps:
            seen = set()
            for ld in list(self.loads) + list(st.loads):
                if ld.vehicle is None:
                    continue
                if ld.vehicle.name in seen:
                    raise ValueError(f"two vehicles named {ld.vehicle.name!r} in one step: each vehicle "
                                     "needs its own name (a monitor's `of` reads it)")
                seen.add(ld.vehicle.name)
        axles = {}
        for v in vehicles:
            axles[v.name] = min(axles.get(v.name, len(v.axles)), len(v.axles))
        for key, m in self.output.monitors.items():
            if m.kind != "vehicle":
                continue
            if m.of not in axles:
                raise ValueError(f"monitor {key!r} reads vehicle {m.of!r}, which no load names"
                                 + (f" (the vehicles are {', '.join(sorted(axles))})" if axles else ""))
            if m.axle >= axles[m.of]:
                raise ValueError(f"monitor {key!r}: vehicle {m.of!r} has {axles[m.of]} axle(s), "
                                 f"numbered from 0; there is no axle {m.axle}")
        return self

    @model_validator(mode="after")
    def _analysis_steps(self):
        """Analysis steps (1.2): `keep` is a step's word, and a step that
        changes the model or its procedure is a line model's for now."""
        kept = [x.region for x in list(self.loads) + list(self.constraints) if x.keep]
        if kept:
            raise ValueError(f"`keep` on a model-level load or constraint (on {kept[0]!r}): the "
                             "model-level ones apply in every step already; `keep` carries a "
                             "STEP's own into the steps after it")
        changing = [k for k, st in enumerate(self.steps)
                    if st.activate or st.deactivate or st.release or st.total_time is not None
                    or any(x.keep for x in list(st.loads) + list(st.constraints))
                    or (st.procedure is not None and st.procedure != self.solver.mode)]
        for k, st in enumerate(self.steps):
            if st.birth != "tangent" and not st.activate:
                raise ValueError(f"steps[{k}].birth says where the members a step activates are put; "
                                 "this step activates none")
            for ld in st.loads:
                if ld.keep and any(getattr(ld, f) is not None for f in
                                   ("moving_load", "vehicle", "ground_motion")):
                    raise ValueError(f"steps[{k}]: `keep` on a moving load, a vehicle or a ground "
                                     "motion -- a history has no end value to hold; repeat it in the "
                                     "steps it acts in")
        d = self.solver.damping
        if d is not None and d.beta and self.solver.mode == "relaxation" and not any(
                st.procedure == "dynamic" for st in self.steps):
            raise ValueError("damping.beta is stiffness-proportional Rayleigh damping: it needs "
                             "type: rayleigh in a dynamic run (or dynamic steps)")
        if changing and self.element.type not in STEP_CHANGE_ELEMENTS:
            st = self.steps[changing[0]]
            what = ("a procedure of its own" if not (st.activate or st.deactivate or st.release)
                    else "activate / deactivate / release")
            raise ValueError(f"steps[{changing[0]}] has {what}: analysis steps that change the "
                             f"model or the procedure run line models "
                             f"({', '.join(sorted(STEP_CHANGE_ELEMENTS))}), not {self.element.type}")
        if any(st.procedure == "dynamic" or (st.procedure is None and self.solver.mode == "dynamic")
               for st in self.steps) and self.solver.mode != "dynamic" \
                and self.solver.total_time is None and any(
                    st.total_time is None for st in self.steps
                    if (st.procedure or self.solver.mode) == "dynamic"):
            raise ValueError("a dynamic step needs its duration: give the step `total_time` "
                             "(or solver.total_time)")
        return self

    @model_validator(mode="after")
    def _regions_exist(self):
        self.material = coerce_numbers(self.material)
        if self.element.type in {"mixed_line", "coupled", "shell_line"} and not self.members:
            raise ValueError(f"element.type={self.element.type} requires at least one members assignment")
        # ``all`` is an implicit region supplied by every mesh loader.  Keeping
        # it here also makes the default LoadSpec region usable for structured
        # meshes whose YAML need not spell out a redundant selector.
        if self.soil_springs:
            moved = [c.region for c in list(self.constraints) + [c for st in self.steps for c in st.constraints]
                     if getattr(c, "record", None) is not None]
            if moved:
                # the springs act on the pile's absolute displacement: soil
                # fixed in space while the supports follow their records
                # (review of 1.2)
                raise ValueError(f"soil_springs with a support `record` (on {moved[0]!r}): the springs "
                                 "hold the pile to ground that does not move, while the supports move "
                                 "with the record; use a uniform `ground_motion` (relative motion) with "
                                 "soil springs")
        known = set(self.regions)
        known.add("all")
        if self.mesh.type == "lines":
            # its points and lines are regions, by name (1.1)
            known |= set(self.mesh.points or {}) | set(self.mesh.lines or {}) | set(self.mesh.groups or {})
        if self.mesh.type in {"gmsh", "abaqus"}:
            # the file's physical groups are regions too, and the file is the
            # authority on what they are. Checking them here would mean parsing
            # the mesh during schema validation; the solver names the available
            # groups instead, when it actually has them.
            return self
        for k, st in enumerate(self.steps):
            for key in ("activate", "deactivate", "release"):
                for r in getattr(st, key):
                    if r not in known:
                        raise ValueError(f"steps[{k}].{key} references unknown region {r!r}")
        for c in self.constraints:
            if c.region not in known:
                raise ValueError(f"constraint references unknown region {c.region!r}")
        for ld in self.loads:
            if ld.region not in known:
                raise ValueError(f"load references unknown region {ld.region!r}")
        for lump in self.nodal_masses:
            if lump.region not in known:
                raise ValueError(f"nodal mass references unknown region {lump.region!r}")
        for h in self.hydrodynamic_mass:
            if h.region not in known:
                raise ValueError(f"hydrodynamic_mass references unknown region {h.region!r}")
        for k, sspec in enumerate(self.soil_springs):
            if sspec.region not in known:
                raise ValueError(f"soil_springs[{k}] references unknown region {sspec.region!r}")
        for d in self.diaphragms:
            if d.region not in known:
                raise ValueError(f"diaphragm references unknown region {d.region!r}")
        for name, m in self.output.monitors.items():
            # an element or bar monitor may leave the region out entirely,
            # which means the whole model
            for r in (m.regions or ([m.region] if m.region else [])):
                if r not in known:
                    raise ValueError(
                        f"monitor {name!r} references unknown region {r!r}"
                    )
        if "type" not in self.material:
            raise ValueError("material.type is required")
        return self

    def as_dict(self) -> Dict[str, Any]:
        """Every field, defaults included -- the form the solver reads."""
        return self.model_dump(mode="python")

    def as_written(self) -> Dict[str, Any]:
        """Only what the case actually said.

        `as_dict` fills in every default, so validating its output back into a
        Case turns "I did not say" into "I said". That matters for exactly one
        field and it matters a lot: `element.type` defaults to `hexa8`, and the
        solver refuses a case that CLAIMS an element the mesh does not contain.
        A case that never mentioned `type`, round-tripped through a dict, is
        therefore accused of claiming `hexa8` -- and since Gmsh's default 3-D
        output is tetrahedra, that is most meshes.

        Hand this to `prepare_case`/`run_case` when passing a Case back as a
        dict; use `as_dict` when something needs every value filled in.
        """
        return self.model_dump(mode="python", exclude_unset=True)


def from_dict(d: dict, base_dir: Optional[str] = None) -> Case:
    """The one entry for a case dict: the input form (materials / sections /
    assign, io/model.py) is lowered to this internal model; the internal
    form is validated directly. Everything -- CLI, API, scaffold -- comes
    through here, so the lowering cannot be skipped by accident."""
    from . import model as input_model
    from .. import prep

    overlay = None
    expanded_report: Dict[str, Any] = {}
    notes: List[Any] = []
    if prep.has_high_level(d):
        # the high-level blocks (`tendons`), expanded onto the
        # case's mesh; `base_dir` is where its relative paths start
        try:
            d, overlay, expanded_report = prep.expand_case(d, base_dir, warn=lambda who, t: notes.append((who, t)))
        except (ValueError, ValidationError) as exc:
            from ..core.diagnostics import UserError
            raise UserError("HIGH_LEVEL_BLOCK", str(exc),
                            "see the manual, 11.14 (`tendons`), 11.15 (`include`, `groups`, `table`) and 11.16 (`offset_arms`); "
                            "`vfife expand` writes what the case expands to") from None
    if not input_model.is_input_form(d):
        # The old form (element / material / reinforcement / members) is the
        # solver's INTERNAL layout and is not accepted at the boundary any
        # more. `vfife migrate` rewrites a file and checks
        # that the result lowers back to the same model.
        from ..core.diagnostics import UserError
        raise UserError(
            "OLD_CASE_FORM",
            "this case has no `materials` / `sections` / `assign`: it is written in the "
            "old form (element / material / reinforcement / members)",
            "run `vfife migrate <case.yaml> --in-place` to rewrite it as "
            "materials / sections / assign, then run it again",
        )
    # the internal keys beside the input form were overwritten by the
    # lowering in silence: `element: {type: beam2_timoshenko}` next to beam
    # sections ran as the sections said
    internal = [k for k in ("element", "material", "members", "blocks") if k in d]
    if internal:
        from ..core.diagnostics import UserError
        raise UserError(
            "INTERNAL_KEY_IN_INPUT_FORM",
            f"this case has `materials` / `sections` / `assign` and also "
            f"{', '.join(f'`{k}`' for k in internal)}, a key of the old (internal) form that "
            "the sections decide and would overwrite",
            "remove " + ", ".join(f"`{k}`" for k in internal) + "; the element comes from the "
            "sections (`type: beam` with `theory: timoshenko`, `type: fiber`, ...) and the "
            "mesh, the materials from `materials`",
        )
    # a default that carries a unit, in the units the case declares
    from ..core.units import fill_defaults
    d, unit_defaults = fill_defaults(d)
    lowered, prov = input_model.lower_with_provenance(d)
    try:
        case = Case.model_validate(lowered)
    except ValidationError as exc:
        # The internal form's locations -- `members.0.failure`,
        # `reinforcement.0.friction` -- are keys the user never wrote. Said
        # in the input form's terms when the lowering made them up;
        # an error at a key the user did write is left as
        # pydantic reports it.
        lines = input_model.translate_errors(exc.errors(), prov)
        if not any(w for w, _l, _m in lines):
            raise
        from ..core.diagnostics import UserError
        body = "\n".join(f"  {w or l}: {m}" for w, l, m in lines)
        raise UserError("CASE_INVALID", "this case is not valid:\n" + body,
                        "internal locations: " + "; ".join(l for _w, l, _m in lines)) from None
    case._deprecated = deprecated_spellings(d)
    case._unit_defaults = unit_defaults
    case._mesh_overlay = overlay
    case._expansion_notes = notes
    case._tendon_report = dict((expanded_report or {}).get("tendons") or {})
    return case


def deprecated_spellings(d: dict) -> List[tuple]:
    """(where, old, new) for every old spelling a case uses:
    read as before, reported by preflight, rewritten by `vfife migrate`."""
    from .model import DEPRECATED_SECTION_KEYS
    out = []
    for name, sec in (d.get("sections") or {}).items():
        if isinstance(sec, dict):
            for old, new in DEPRECATED_SECTION_KEYS.items():
                if old in sec:
                    out.append((f"sections.{name}", old, new))
    return out


def safe_load_named(text: str, path) -> Any:
    """`yaml.safe_load(text)`, with a syntax error marked with the file it
    came from: a string's mark says "<unicode string>", and the command line
    then named whichever file the command was given -- the case, for a
    broken verification contract."""
    try:
        return yaml.safe_load(text)
    except yaml.MarkedYAMLError as exc:
        for mark in (exc.problem_mark, exc.context_mark):
            if mark is not None:
                mark.name = str(path)
        raise


def load(path: str) -> tuple[Case, str]:
    """Read and validate a YAML case. Returns (case, raw_text)."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    return from_dict(safe_load_named(text, path), os.path.dirname(os.path.abspath(path))), text


def loads(text: str) -> Case:
    return from_dict(yaml.safe_load(text))
