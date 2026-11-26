"""One minimal model per registered element.

The invariant harness in tests/invariants asks the same questions of every
element and every load: is the internal force the gradient of something, is
its tangent symmetric, does a rigid motion leave it at zero, does the total
external force equal the total reaction. Those questions do not depend on the
element, but every one of them needs a MODEL to be asked on -- a mesh, a
support, a load that produces a displacement in the range the element is
meant for -- and until now each test built its own, which is how five
element types were never asked anything at all.

This registry is that model, once per element:

    build(element)      a Model: the internal-form case dict `cli.make_solver`
                        reads, the in-memory Mesh it reads it against, and the
                        material, with the regions the harness may load
    make(model)         the solver, through `cli.make_solver`, on copies
    elements()          every element the registry covers

Every model is a cantilever of one kind or another: one end or face held
completely (`regions["fixed"]`), a small transverse force on the far end
(`regions["loaded"]`) sized so the linear response is about a thousandth of
the model's length, and for solids and shells the loaded exterior face
(`regions["face"]`) for a pressure, a traction or a face moment. The mesh is
built in memory -- `core.mesh.box_mesh` for the elements it emits, the
tests/data fixtures for the quadratic and pyramid solids, a few lines of
numpy for the frames and shells -- and the case is written in the INPUT form
(materials / sections / assign) and lowered through `schema.from_dict`, so a
model here is the same object a YAML case becomes. The one exception is
`shell_line`, which the input form cannot express (a shell section beside a
beam section lowers to `coupled` and the schema refuses the thickness); it
is validated in the internal form directly.

THE POINT IS THE ENUMERATION. `elements()` is checked against
`elements.library.element_names()` by tests/invariants: a new element that is
registered in the library but has no builder here fails that suite, which is
the moment to write the six lines its model needs -- not later, when it has
already shipped with nobody having asked it whether its force has a
potential.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from ..core.mesh import Mesh, box_mesh
from ..elements.library import element_names, families
from ..io import schema

# --------------------------------------------------------------------------- #
#  materials
# --------------------------------------------------------------------------- #

#: Steel for the frames, bars, connectors and shells.
STEEL = {"E": 2.0e11, "nu": 0.3, "rho": 7850.0}
#: Concrete for the solids: the elements' own compatible laws (cdp, j2 with a
#: concrete-like yield) are written for it, and it keeps the solid models in
#: the same displacement range as the structural ones at a sane force.
CONCRETE = {"E": 3.0e10, "nu": 0.2, "rho": 2400.0}

#: A reasonable parameter set per material, keyed by the registered name.
#: `linear_elastic` is STEEL here and CONCRETE on a solid (see `_material`).
#: The numbers are the ones the material's own tests use, so a model built
#: with them is a model the law has already been exercised on.
MATERIAL_PARAMS: Dict[str, dict] = {
    "linear_elastic": dict(STEEL),
    "orthotropic": {"E1": 3.0e10, "E2": 3.0e10, "E3": 3.0e10,
                    "nu12": 0.2, "nu13": 0.2, "nu23": 0.2,
                    "G12": 1.25e10, "G13": 1.25e10, "G23": 1.25e10, "rho": 2400.0},
    "viscoelastic": {**CONCRETE, "g": [0.3], "tau": [0.01]},
    # `H`, not `modulus`: `curves.hardening_linear(yield_strength, H)`. The
    # wrong spelling sat here through most of the 0.1 development, because until the
    # material invariants nothing ever built a model with a material other
    # than the default -- `build(element, material=...)` raised for `j2` and
    # no test asked it to.
    "j2": {**CONCRETE, "hardening": {"source": "linear", "yield_strength": 3.0e7,
                                     "H": 3.0e9}},
    "chaboche": {"E": 2.1e11, "nu": 0.3, "rho": 7800.0, "yield_strength": 2.5e8,
                 "Q_inf": 1.0e8, "b": 20.0, "H_iso": 1.0e9,
                 "C1": 3.0e10, "gamma1": 300.0, "C2": 5.0e9, "gamma2": 30.0},
    "cdp": {**CONCRETE, "compression": {"source": "gb50010", "fc": 3.0e7},
            "tension": {"source": "gb50010", "ft": 3.0e6}, "l_ref": 0.2},
    "rebar": {"E": 2.0e11, "rho": 7850.0, "yield_strength": 4.0e8, "hardening": 1.0e9},
    "steel02": {"E": 2.0e11, "rho": 7850.0, "Fy": 4.0e8, "b": 0.01},
    "piecewise_uniaxial": {"E": 2.0e11, "rho": 7850.0,
                           "tension": [[0.0, 2.0e8], [0.01, 3.0e8]],
                           "compression": [[0.0, 1.5e8], [0.01, 2.4e8]]},
    "concrete01": {"fpc": 3.0e7, "epsc0": 0.002, "fpcu": 6.0e6, "epscu": 0.0035, "rho": 2400.0},
    "concrete02": {"fpc": 3.0e7, "epsc0": 0.002, "fcu": 6.0e6, "epscu": 0.0035,
                   "ft": 3.0e6, "Ets": 1.5e9, "rho": 2400.0},
    # Cai et al. 2023, Table 1. A grade name is enough -- the eleven numbers
    # behind it are in materials/steel_cyclic.GRADES.
    "steel_cyclic": {"grade": "Q345B", "E": 2.06e11, "rho": 7850.0},
    # the composites (1.21): a yielding core beside an elastic spring, and an
    # elastic link in series with it -- parts that are laws already exercised
    "parallel": {"of": [{"type": "steel02", "E": 2.0e11, "rho": 7850.0, "Fy": 4.0e8, "b": 0.01},
                        {"type": "linear_elastic", "E": 2.0e10, "nu": 0.3, "rho": 7850.0}]},
    "series": {"of": [{"type": "linear_elastic", "E": 4.0e11, "nu": 0.3, "rho": 7850.0},
                      {"type": "steel02", "E": 2.0e11, "rho": 7850.0, "Fy": 4.0e8, "b": 0.01}]},
    "gap": {"E": 2.0e11, "rho": 7850.0, "lo": -1.0e-3, "hi": 0.0},       # a stop at the origin, slack behind it
}


def _material(name: str, element: str) -> dict:
    """The material block for `name` on `element`, checked for compatibility
    first so an impossible pair is refused with the catalogue's reasons rather
    than a solver's traceback."""
    from ..catalog import compatibility

    ok, reasons = compatibility(element, name)
    if not ok:
        raise ValueError(f"{name!r} cannot be used with {element!r}: " + "; ".join(reasons))
    if name not in MATERIAL_PARAMS:
        raise ValueError(f"no parameter set for material {name!r} in "
                         "vfife.testing.models.MATERIAL_PARAMS")
    params = dict(MATERIAL_PARAMS[name])
    if name == "linear_elastic" and families()[element] == "continuum":
        params = dict(CONCRETE)
    return {"type": name, **params}


# --------------------------------------------------------------------------- #
#  the registry entry and the public Model
# --------------------------------------------------------------------------- #

@dataclass
class Model:
    element: str            # element type name
    problem: dict           # the internal-form case dict `cli.make_solver` takes
    mesh: Mesh              # its own copy; solvers mutate mesh.regions
    material: object        # what `cli.make_solver` expects as `material`
    dofs_per_node: int      # 3 or 6
    regions: dict           # {"fixed": name, "loaded": name, "face": name or None, "all": "all"}
    length: float           # characteristic length, for scaling perturbations


@dataclass
class _Draft:
    """What a builder produces; `build` turns it into a Model."""
    mesh: Mesh
    #: the case's `mesh` block: a true description where one exists (a box, a
    #: fixture file), otherwise a gmsh block naming this module -- nothing
    #: reads it, `make_solver` takes the Mesh
    mesh_block: dict
    sections: dict
    assign: list
    fixed: np.ndarray
    loaded: np.ndarray
    length: float
    #: the default transverse load, total on `loaded`
    load: List[float]
    dofs_per_node: int
    face: Optional[np.ndarray] = None
    #: further named node sets (member groups) the sections are assigned to
    regions: Dict[str, np.ndarray] = field(default_factory=dict)
    #: how the material is named in the sections: `material` for every
    #: section that carries one
    material_sections: tuple = ()
    #: applied to the lowered internal dict before validation, for what the
    #: input form cannot say (shell_line)
    internal: Optional[Callable[[dict], None]] = None
    #: the element's default formulation, when the schema's is not valid for it
    formulation: Optional[str] = None


_REGISTRY: Dict[str, Callable[[], _Draft]] = {}


def _register(name):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def _placeholder_mesh(element: str) -> dict:
    return {"type": "gmsh", "path": f"vfife.testing.models:{element}"}


# --------------------------------------------------------------------------- #
#  solids
# --------------------------------------------------------------------------- #

_BOX = (0.8, 0.2, 0.2)            # a cantilever four deep in x, square section


def _box_draft(element: str, divisions, load: float, formulation: str) -> _Draft:
    dims = _BOX
    mesh = box_mesh(dims, divisions, (0.0, 0.0, 0.0), element=element)
    x = mesh.nodes[:, 0]
    fixed = np.flatnonzero(np.isclose(x, 0.0))
    loaded = np.flatnonzero(np.isclose(x, dims[0]))
    return _Draft(
        mesh=mesh,
        mesh_block={"type": "box", "dimensions": list(dims), "divisions": list(divisions)},
        sections={"body": {"type": "solid", "material": "mat", "formulation": formulation}},
        assign=[{"region": "all", "section": "body"}],
        fixed=fixed, loaded=loaded, face=loaded, length=dims[0],
        load=[0.0, 0.0, load], dofs_per_node=3, material_sections=("body",),
    )


@_register("hexa8")
def _hexa8():
    return _box_draft("hexa8", (4, 2, 2), 1.8e4, "b-bar")


@_register("wedge6")
def _wedge6():
    return _box_draft("wedge6", (4, 2, 2), 1.8e4, "b-bar")


@_register("tet4")
def _tet4():
    return _box_draft("tet4", (4, 1, 1), 1.8e4, "full")


@_register("tet10")
def _tet10():
    return _box_draft("tet10", (4, 1, 1), 1.8e4, "b-bar")


_FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "tests", "data")


def _fixture_draft(element: str, filename: str, load: float) -> _Draft:
    """A unit cube from tests/data, held on `left_face` (x = 0) and loaded
    on `right_face` (x = 1). The pyramid file mixes pyramids and tetrahedra,
    which is what a pyramid is for; it is kept as the reader reads it."""
    from ..io import gmsh as gmsh_io

    path = os.path.join(_FIXTURES, filename)
    mesh, _read_as = gmsh_io.read(path)
    fixed = np.asarray(mesh.regions["left_face"], dtype=np.int64)
    loaded = np.asarray(mesh.regions["right_face"], dtype=np.int64)
    length = float(np.ptp(mesh.nodes[:, 0]))
    return _Draft(
        mesh=mesh, mesh_block={"type": "gmsh", "path": path},
        sections={"body": {"type": "solid", "material": "mat", "formulation": "full"}},
        assign=[{"region": "all", "section": "body"}],
        fixed=fixed, loaded=loaded, face=loaded, length=length,
        load=[0.0, 0.0, load], dofs_per_node=3, material_sections=("body",),
    )


@_register("hexa20")
def _hexa20():
    return _fixture_draft("hexa20", "box_hexa20_v41.msh", 4.0e6)


@_register("wedge15")
def _wedge15():
    return _fixture_draft("wedge15", "box_wedge15_v41.msh", 4.0e6)


@_register("pyramid5")
def _pyramid5():
    """Eight cubes, each split into six pyramids around its own centre.

    NOT the tests/data fixture, which mixes six pyramids into a tetrahedral
    box: in that mesh no complete pyramid face lies on the boundary, so the
    block cannot carry a pressure, a traction or any other face load, and a
    model that cannot be loaded cannot be asked the load questions. Splitting
    every cube into six pyramids puts a pyramid's quadrilateral base on every
    exterior face, which is the arrangement a pyramid is for -- the transition
    from a hexahedral region to a tetrahedral one.
    """
    from ..core.mesh import box_mesh

    box = box_mesh((1.0, 1.0, 1.0), (2, 2, 2))
    hexes = box.elements
    centres = box.nodes[hexes].mean(axis=1)
    nodes = np.vstack([box.nodes, centres])
    apex = box.n_nodes + np.arange(len(hexes))
    # the six faces of a hexahedron, wound so each base normal points out of
    # the hexahedron and therefore away from its centre
    # wound so the base normal points INTO the hexahedron, i.e. toward the
    # apex: that is the VTK pyramid convention, which the precompute checks
    # by refusing a non-positive Jacobian
    faces = ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
             (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))
    conn = np.vstack([np.column_stack([hexes[:, list(f)], apex]) for f in faces])
    mesh = Mesh(nodes=np.ascontiguousarray(nodes),
                elements=np.ascontiguousarray(conn, np.int32))
    X = mesh.nodes
    fixed = np.flatnonzero(np.isclose(X[:, 0], 0.0)).astype(np.int64)
    loaded = np.flatnonzero(np.isclose(X[:, 0], 1.0)).astype(np.int64)
    return _Draft(
        mesh=mesh, mesh_block=_placeholder_mesh("pyramid5"),
        sections={"body": {"type": "solid", "material": "mat", "formulation": "full"}},
        assign=[{"region": "all", "section": "body"}],
        fixed=fixed, loaded=loaded, face=loaded, length=1.0,
        load=[0.0, 0.0, -2.0e5], dofs_per_node=3, material_sections=("body",),
    )


# --------------------------------------------------------------------------- #
#  lines
# --------------------------------------------------------------------------- #

_LINE_L = 1.0
_LINE_N = 4
#: 0.1 x 0.1 square: A = 0.01, I = 8.33e-6, J = 0.1406 a^4 = 1.41e-5
_BEAM_SECTION = {"shape": "rect", "width": 0.1, "height": 0.1}


def _line_mesh(n: int = _LINE_N, length: float = _LINE_L) -> Mesh:
    nodes = np.zeros((n + 1, 3))
    nodes[:, 0] = np.linspace(0.0, length, n + 1)
    conn = np.column_stack([np.arange(n), np.arange(1, n + 1)]).astype(np.int32)
    return Mesh(nodes=nodes, elements=conn)


def _cantilever_load(EI: float, length: float = _LINE_L, target: float = 1e-3) -> float:
    """The tip force that deflects a cantilever `target * length`."""
    return target * length * 3.0 * EI / length**3


def _line_draft(element: str, section: dict, load: float, dofs: int = 6) -> _Draft:
    mesh = _line_mesh()
    return _Draft(
        mesh=mesh, mesh_block=_placeholder_mesh(element),
        sections={"member": section},
        assign=[{"region": "all", "section": "member", "orientation": [0.0, 0.0, 1.0]}],
        fixed=np.array([0]), loaded=np.array([_LINE_N]), length=_LINE_L,
        load=[0.0, 0.0, load], dofs_per_node=dofs, material_sections=("member",),
    )


_EI = STEEL["E"] * 0.1**4 / 12.0


@_register("beam2")
def _beam2():
    return _line_draft("beam2", {"type": "beam", "material": "mat", **_BEAM_SECTION},
                       _cantilever_load(_EI))


@_register("beam2_timoshenko")
def _beam2_timoshenko():
    return _line_draft("beam2_timoshenko",
                       {"type": "beam", "material": "mat", "theory": "timoshenko", **_BEAM_SECTION},
                       _cantilever_load(_EI))


@_register("beam2_fiber")
def _beam2_fiber():
    # 4 x 4 fibres over the same square; their second moment is a little
    # under the exact I, so the tip goes a little further than the beam2's
    return _line_draft("beam2_fiber",
                       {"type": "fiber", "material": "mat", "shape": "rect",
                        "width": 0.1, "height": 0.1, "ny": 4, "nz": 4},
                       _cantilever_load(_EI))


@_register("connector2")
def _connector2():
    # four connectors in series: each shears by F / k_t and rotates by the
    # moment at its station over k_r, so the tip goes about
    # F (4 / k_t + sum x_i^2 / k_r) with x_i the stations from the tip
    k_t, k_r = 1.0e7, 1.0e5
    stations = (np.arange(_LINE_N) + 0.5) * (_LINE_L / _LINE_N)
    flexibility = _LINE_N / k_t + float((stations**2).sum()) / k_r
    return _line_draft("connector2",
                       {"type": "connector", "material": "mat",
                        "stiffness": [k_t, k_t, k_t, k_r, k_r, k_r]},
                       1e-3 * _LINE_L / flexibility)


@_register("truss2")
def _truss2():
    """A space-truss boom: four bays of a triangular section, chords,
    rings and one diagonal per face. A straight chain of bars would be the
    obvious four-member line, and a mechanism under a transverse load; this
    is the smallest truss that is stiff in every direction."""
    n_bay, length, side = 4, _LINE_L, 0.2
    r = side / math.sqrt(3.0)
    ring = np.array([[0.0, r * math.cos(a), r * math.sin(a)]
                     for a in (math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3)])
    nodes = np.vstack([ring + [k * length / n_bay, 0.0, 0.0] for k in range(n_bay + 1)])
    conn = []
    for k in range(n_bay + 1):
        a = 3 * k
        conn += [(a, a + 1), (a + 1, a + 2), (a + 2, a)]               # ring
        if k < n_bay:
            b = a + 3
            conn += [(a, b), (a + 1, b + 1), (a + 2, b + 2)]           # chords
            conn += [(a, b + 1), (a + 1, b + 2), (a + 2, b)]           # diagonals
    mesh = Mesh(nodes=nodes, elements=np.asarray(conn, dtype=np.int32))
    fixed = np.arange(3)
    loaded = np.arange(3 * n_bay, 3 * n_bay + 3)
    # chords of area A at radius r: I ~ 3 A r^2 / 2, then shear and the
    # coarse bays add about a third to the deflection
    area = 1.0e-3
    EI = STEEL["E"] * 1.5 * area * r**2
    return _Draft(
        mesh=mesh, mesh_block=_placeholder_mesh("truss2"),
        sections={"bar": {"type": "bar", "material": "mat", "shape": "general", "area": area}},
        assign=[{"region": "all", "section": "bar"}],
        fixed=fixed, loaded=loaded, length=length,
        load=[0.0, 0.0, 0.75 * _cantilever_load(EI, length)], dofs_per_node=3,
        material_sections=("bar",),
    )


@_register("mixed_line")
def _mixed_line():
    """A four-member cantilever beam with a bar stay from a support above
    the root to the tip: the two member groups `mixed_line` exists for, and
    a truss member that does something under the transverse load."""
    mesh = _line_mesh()
    stay_root = mesh.n_nodes
    nodes = np.vstack([mesh.nodes, [[0.0, 0.0, 0.5]]])
    conn = np.vstack([mesh.elements, [[stay_root, _LINE_N]]]).astype(np.int32)
    mesh = Mesh(nodes=nodes, elements=conn)
    beams = np.arange(_LINE_N + 1)
    bars = np.array([stay_root, _LINE_N])
    # the stay carries most of a downward tip load: its axial stiffness
    # projected on z, plus the beam
    area = 1.0e-4
    L_stay = math.hypot(_LINE_L, 0.5)
    k_stay = STEEL["E"] * area / L_stay * (0.5 / L_stay)**2
    k_beam = 3.0 * _EI / _LINE_L**3
    return _Draft(
        mesh=mesh, mesh_block=_placeholder_mesh("mixed_line"),
        sections={"member": {"type": "beam", "material": "mat", **_BEAM_SECTION},
                  "stay": {"type": "bar", "material": "mat", "shape": "general", "area": area}},
        assign=[{"region": "beams", "section": "member", "orientation": [0.0, 0.0, 1.0]},
                {"region": "bars", "section": "stay"}],
        fixed=np.array([0, stay_root]), loaded=np.array([_LINE_N]), length=_LINE_L,
        load=[0.0, 0.0, -1e-3 * _LINE_L * (k_stay + k_beam)], dofs_per_node=6,
        regions={"beams": beams, "bars": bars}, material_sections=("member", "stay"),
    )


# --------------------------------------------------------------------------- #
#  shells
# --------------------------------------------------------------------------- #

_PLATE_L = 0.4
_PLATE_N = 4
_PLATE_T = 0.01


def _plate_nodes(n: int = _PLATE_N, length: float = _PLATE_L):
    axis = np.linspace(0.0, length, n + 1)
    nodes = np.array([[x, y, 0.0] for y in axis for x in axis])
    quads = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            quads.append([a, a + 1, a + n + 2, a + n + 1])     # counter-clockwise, normal +z
    return nodes, np.asarray(quads, dtype=np.int32)


def _plate_draft(element: str, conn: np.ndarray, nodes: np.ndarray, **extra) -> _Draft:
    mesh = Mesh(nodes=nodes, elements=conn)
    x = nodes[:, 0]
    fixed = np.flatnonzero(np.isclose(x, 0.0))
    loaded = np.flatnonzero(np.isclose(x, _PLATE_L))
    EI = STEEL["E"] * _PLATE_L * _PLATE_T**3 / 12.0
    return _Draft(
        mesh=mesh, mesh_block=_placeholder_mesh(element),
        sections={"plate": {"type": "shell", "material": "mat", "thickness": _PLATE_T,
                            "formulation": "selective"}},
        assign=[{"region": "all", "section": "plate"}],
        fixed=fixed, loaded=loaded, face=np.arange(mesh.n_nodes), length=_PLATE_L,
        load=[0.0, 0.0, _cantilever_load(EI, _PLATE_L)], dofs_per_node=6,
        material_sections=("plate",), **extra,
    )


@_register("shell4")
def _shell4():
    nodes, quads = _plate_nodes()
    return _plate_draft("shell4", quads, nodes)


@_register("shell3")
def _shell3():
    nodes, quads = _plate_nodes()
    tris = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]).astype(np.int32)
    return _plate_draft("shell3", tris, nodes)


@_register("shell_line")
def _shell_line():
    """The shell4 plate with a beam rib along its centre line. The input
    form cannot write this (see the module docstring), so the rib is added
    to the lowered internal case by `internal`."""
    nodes, quads = _plate_nodes()
    rib = np.flatnonzero(np.isclose(nodes[:, 1], _PLATE_L / 2))
    rib = rib[np.argsort(nodes[rib, 0])]
    lines = np.column_stack([rib[:-1], rib[1:]]).astype(np.int32)
    draft = _plate_draft("shell_line", quads, nodes, regions={"rib": rib})
    draft.mesh.cell_blocks = {"shell4": quads, "truss2": lines}
    rib_section = {"shape": "rect", "width": 0.02, "height": 0.04}
    from .. import sections as sec_geom
    g = sec_geom.properties(rib_section)
    EI = STEEL["E"] * (g["iy"] + _PLATE_L * _PLATE_T**3 / 12.0)
    draft.load = [0.0, 0.0, _cantilever_load(EI, _PLATE_L)]

    def internal(p: dict):
        p["element"]["type"] = "shell_line"
        p["members"] = [{"region": "rib", "type": "beam2", "orientation": [0.0, 0.0, 1.0],
                         **{k: g[k] for k in ("area", "iy", "iz", "j")}}]
    draft.internal = internal
    return draft


# --------------------------------------------------------------------------- #
#  the coupled solid
# --------------------------------------------------------------------------- #

@_register("coupled")
def _coupled():
    """The hexa8 cantilever with a beam along its top edge: one solid block
    and one Line2 block on shared nodes."""
    draft = _box_draft("hexa8", (4, 2, 2), 1.8e4, "full")
    mesh = draft.mesh
    top = np.flatnonzero(np.isclose(mesh.nodes[:, 1], 0.0) & np.isclose(mesh.nodes[:, 2], _BOX[2]))
    top = top[np.argsort(mesh.nodes[top, 0])]
    lines = np.column_stack([top[:-1], top[1:]]).astype(np.int32)
    mesh.cell_blocks = {"hexa8": mesh.elements, "truss2": lines}
    draft.mesh_block = _placeholder_mesh("coupled")
    draft.sections["rib"] = {"type": "beam", "material": "mat", "shape": "rect", "width": 0.02, "height": 0.02}
    draft.assign.append({"region": "rib", "section": "rib", "orientation": [0.0, 0.0, 1.0]})
    draft.regions["rib"] = top
    draft.material_sections = ("body", "rib")
    draft.dofs_per_node = 6
    return draft


# --------------------------------------------------------------------------- #
#  public API
# --------------------------------------------------------------------------- #

def elements() -> tuple[str, ...]:
    """The element names the registry covers, sorted like `element_names()`."""
    return tuple(sorted(_REGISTRY))


def build(element: str, *, material: str = "linear_elastic", mode: str = "relaxation",
          backend: str = "numpy", formulation: str | None = None,
          loads: list[dict] | None = None, constraints: list[dict] | None = None,
          solver: dict | None = None) -> Model:
    """The minimal model of `element`.

    `loads` and `constraints` replace the defaults (a full fixity on
    `regions["fixed"]`, a small transverse `vector` on `regions["loaded"]`);
    `solver` is merged over the default solver block. `material` is any
    registered name the element accepts (`catalog.compatibility`), with the
    parameters of `MATERIAL_PARAMS`.
    """
    if element not in _REGISTRY:
        raise KeyError(f"no model for element {element!r}; the registry covers {elements()}")
    draft = _REGISTRY[element]()
    mesh = draft.mesh
    names = {"fixed": "fixed", "loaded": "loaded", "face": "face" if draft.face is not None else None,
             # the region a `line_load` may be applied to: on a mixed model it
             # is the beam members only, since a line load "can only be applied
             # to beam members" and a region that also holds bars is refused
             "line": "beams" if "beams" in draft.regions else "all",
             "all": "all"}

    node_sets = {"fixed": draft.fixed, "loaded": draft.loaded, **draft.regions}
    if draft.face is not None:
        node_sets["face"] = draft.face
    for name, idx in node_sets.items():
        mesh.regions[name] = np.unique(np.asarray(idx, dtype=np.int64))
    mesh.regions.setdefault("all", np.arange(mesh.n_nodes, dtype=np.int64))

    mat = _material(material, element)
    sections = copy.deepcopy(draft.sections)
    if formulation is not None:
        for name, sec in sections.items():
            if sec["type"] in ("solid", "shell"):
                sec["formulation"] = formulation
    for name in draft.material_sections:
        sections[name]["material"] = "mat"

    # 200 000 is not a real budget: the structural models are a fictitious-mass
    # relaxation of a slender cantilever and take 30-80 000 iterations to reach
    # 1e-6, and a harness that has to guess a per-element budget is a harness
    # nobody adds an element to.
    sol = {"mode": mode, "backend": backend, "n_steps": 200000, "tolerance": 1e-6,
           "damping": {"type": "kinetic"}}
    if mode == "dynamic":
        sol.pop("n_steps")
        sol["total_time"] = 1e-4
        sol["damping"] = {"type": "none"}
    sol.update(solver or {})

    dofs = draft.dofs_per_node
    case = {
        "name": f"testing_{element}",
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": dict(draft.mesh_block),
        "materials": {"mat": mat},
        "sections": sections,
        "assign": copy.deepcopy(draft.assign),
        "regions": {name: {"node_ids": [int(i) for i in idx]}
                    for name, idx in mesh.regions.items() if name != "all"},
        "constraints": constraints if constraints is not None
        else [{"region": "fixed", "dofs": [1] * dofs}],
        "loads": loads if loads is not None
        else [{"region": "loaded", "vector": list(draft.load)}],
        "solver": sol,
        "output": {"fields": ["displacement"]},
    }
    # the gmsh fixtures carry their own physical groups; the schema does not
    # check regions against a file it never opens, so nothing more is needed
    problem = schema.from_dict(case).as_dict()
    problem["element"] = dict(problem["element"])
    if draft.internal is not None:
        draft.internal(problem)
        problem = schema.Case.model_validate(problem).as_dict()
    problem["element"]["type"] = element

    from ..materials.factory import create_material
    material_obj = create_material(problem["material"])
    return Model(element=element, problem=problem, mesh=mesh, material=material_obj,
                 dofs_per_node=dofs, regions=names, length=float(draft.length))


def _copy_mesh(mesh: Mesh) -> Mesh:
    return Mesh(nodes=mesh.nodes.copy(), elements=mesh.elements.copy(),
                regions={k: np.asarray(v).copy() for k, v in mesh.regions.items()},
                cell_blocks={k: np.asarray(v).copy() for k, v in (mesh.cell_blocks or {}).items()})


def make(model: Model):
    """The solver for `model`, through `cli.make_solver`.

    On COPIES: a solver rewrites `mesh.regions` and `mesh.cell_blocks`, and a
    solid material binds its per-point data when the points are initialised,
    so a second solver on the same objects would inherit the first one's
    state. The Model stays what `build` made it, and can be made again.
    """
    from ..cli import make_solver
    from ..materials.factory import create_material

    return make_solver(copy.deepcopy(model.problem), _copy_mesh(model.mesh),
                       create_material(model.problem["material"]))


# a builder for every element the library registers, and nothing else: the
# invariant suite asserts the equality, this only makes the failure earlier
assert set(_REGISTRY) <= set(element_names()), sorted(set(_REGISTRY) - set(element_names()))
