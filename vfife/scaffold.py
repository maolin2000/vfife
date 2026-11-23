"""Generic, mesh-driven case scaffolds for the command-line interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any



_HEADER = """# yaml-language-server: $schema=vfife.schema.json
# VFIFE generic analysis case
#
# The first line points an editor at the case schema (VS Code: the YAML
# extension by Red Hat); `vfife schema` writes vfife.schema.json here.
#
# This is intentionally not a beam/column template.  Geometry and named
# regions come from your Gmsh volume mesh and its Physical groups.  Add more
# constraints, loads, steps, amplitudes, or replace the material as needed.
# Run `python -m vfife info <this-file>` before solving.

"""


def physical_regions(regions, n_nodes: int | None = None) -> list[str]:
    """Return likely boundary groups, excluding importer and whole-volume sets.

    `n_nodes` is the mesh's node count: a Gmsh mesh has no `all` region, so
    a whole-volume group was compared with an empty set and offered as a
    support (found in review)."""
    all_nodes = set(regions.get("all", ()))
    if not all_nodes and n_nodes:
        all_nodes = set(range(int(n_nodes)))
    return sorted(
        name for name, nodes in regions.items()
        if name not in {"all", "surface"} and set(int(n) for n in nodes) != all_nodes
    )


@dataclass
class GenericGmshDraft:
    """GUI-neutral problem definition assembled from Gmsh Physical groups.

    A future GUI should edit this data object (and validate its emitted case),
    rather than reimplement command-line parsing or YAML field conventions.
    ``constraints``, ``loads``, ``materials``, ``sections`` and ``solver``
    use the public case-dictionary shapes (the input form: materials ->
    sections -> assign) so every currently supported analysis feature remains
    representable.
    """

    name: str
    mesh_path: str
    constraints: list[dict[str, Any]] = field(default_factory=list)
    loads: list[dict[str, Any]] = field(default_factory=list)
    scale: float = 1.0
    materials: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "steel": {"type": "linear_elastic", "E": 2.1e11, "nu": 0.3, "rho": 7850.0},
    })
    #: the one section the whole mesh gets; a solid by default, the
    #: formulation chosen for the element type the mesh contains
    sections: dict[str, dict[str, Any]] = field(default_factory=dict)
    assign: list[dict[str, Any]] = field(default_factory=list)
    solver: dict[str, Any] = field(default_factory=lambda: {
        "backend": "numpy", "mode": "relaxation", "n_steps": 40000,
        "tolerance": 1.0e-6, "damping": {"type": "kinetic"},
    })
    regions: dict[str, Any] = field(default_factory=dict)
    amplitudes: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    reinforcement: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    #: element type the mesh actually contains; the formulation follows from it
    element: str = "hexa8"

    def as_dict(self) -> dict[str, Any]:
        slug = "".join(c if c.isalnum() or c in "_-" else "_" for c in self.name)
        sections = dict(self.sections) or {
            "body": {"type": "solid", "material": next(iter(self.materials)),
                     "formulation": _formulation_for(self.element)}}
        from .io.schema import CASE_FORMAT
        case = {
            "version": CASE_FORMAT,
            "name": self.name,
            "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "gmsh", "path": self.mesh_path, "scale": self.scale},
            "materials": self.materials,
            "sections": sections,
            "constraints": self.constraints,
            "loads": self.loads,
            "solver": self.solver,
            "output": {"path": f"results/{slug}.h5", "every": 100000,
                       "fields": ["displacement", "von_mises"],
                       "monitors": self.default_monitors(), **self.output},
        }
        for name, value in (("regions", self.regions), ("amplitudes", self.amplitudes),
                            ("steps", self.steps), ("reinforcement", self.reinforcement),
                            ("assign", self.assign)):
            if value:
                case[name] = value
        return case

    def default_monitors(self) -> dict[str, Any]:
        """A load-displacement curve, without the analyst having to ask.

        A scaffolded case used to carry no monitors at all, so `vfife plot`
        on its result could draw the residual and the kinetic energy -- the
        solver's own diagnostics -- and nothing about the structure. The
        first thing anyone wants from a run is how much the loaded end moved
        and what the support carried, and both are one line each; leaving
        them out taught the wrong lesson about what the tool is for.

        The dof is the one the load acts along, so `vfife plot --x u --y R`
        is a load-deflection curve rather than two unrelated histories. A
        case with several loads or no load falls back to the vertical.
        """
        axis = 2
        vec = None
        for load in self.loads:
            v = load.get("vector")
            if v is not None and any(v):
                vec = v if vec is None else vec
        if vec is not None:
            axis = int(max(range(3), key=lambda i: abs(float(vec[i]))))
        dof = "xyz"[axis]
        out: dict[str, Any] = {}
        loaded = next((ld.get("region") for ld in self.loads if ld.get("region")), None)
        held = next((c.get("region") for c in self.constraints if c.get("region")), None)
        if loaded:
            out["u"] = {"kind": "displacement", "region": loaded, "dof": dof}
        if held:
            out["R"] = {"kind": "reaction", "region": held, "dof": dof}
        return out

    def to_yaml(self) -> str:
        from .io.yamlfmt import dump_compact

        # the style a person writes: 3-vectors on one line, short mappings
        # inline, 1.92e+11 rather than 192000000000.0 (io/yamlfmt.py)
        return _HEADER + dump_compact(self.as_dict())

    def validate(self):
        """Return the schema-validated case; the GUI can call this before save."""
        from .io.schema import from_dict
        return from_dict(self.as_dict())


def generic_gmsh_case(name: str, mesh_path: str, fixed_region: str | None = None,
                      load_region: str | None = None, load: list[float] | None = None,
                      *, constraints: list[dict[str, Any]] | None = None,
                      loads: list[dict[str, Any]] | None = None,
                      scale: float = 1.0, material: dict[str, Any] | None = None,
                      solver: dict[str, Any] | None = None,
                      element: str | None = None) -> str:
    """Render a generic case; legacy single support/load arguments still work."""
    if constraints is None:
        constraints = ([{"region": fixed_region, "dofs": [1, 1, 1]}]
                       if fixed_region else [])
    if loads is None:
        loads = ([{"region": load_region, "vector": load}]
                 if load_region and load is not None else [])
    defaults = GenericGmshDraft("", "")
    draft = GenericGmshDraft(name, mesh_path, constraints, loads, scale,
                             defaults.materials if material is None
                             else {"main": material},
                             solver=defaults.solver if solver is None else solver)
    draft.element = element or draft.element
    return draft.to_yaml()


def _formulation_for(element: str) -> str:
    """A formulation the element actually supports.

    This used to be the constant "b-bar", and Gmsh's default 3-D output is
    tet4 -- whose strain is constant, so it has no volumetric part to average
    and supports `full` only.  The scaffold therefore wrote a case that could
    not run, for the commonest mesh anyone would feed it, and the failure came
    out as a traceback after preflight had already said PASS.
    """
    from .elements.library import ELEMENTS

    # `element` may name several types for a mixed mesh ("hexa8+tet4"); the
    # one setting has to be legal for all of them, which for hex+tet means
    # `full` -- the tetrahedron has no volumetric part to average.
    names = [n for n in str(element).split("+") if n in ELEMENTS]
    if not names:                          # structural element: not our choice
        return "b-bar"
    for preferred in ("b-bar", "full", "reduced"):
        if all(preferred in ELEMENTS[n].formulations for n in names):
            return preferred
    return ELEMENTS[names[0]].formulations[0]


def write_generic_gmsh_case(destination: str | Path, *, name: str,
                            mesh_path: str, fixed_region: str | None = None,
                            load_region: str | None = None,
                            load: list[float] | None = None,
                            constraints: list[dict[str, Any]] | None = None,
                            loads: list[dict[str, Any]] | None = None,
                            scale: float = 1.0,
                            element: str | None = None) -> Path:
    out = Path(destination)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(generic_gmsh_case(name, mesh_path, fixed_region, load_region,
                                     load, constraints=constraints, loads=loads,
                                     scale=scale, element=element),
                   encoding="utf-8")
    return out


# --------------------------------------------------------------------------- `vfife new` (1.2)
#: the scenario templates: name -> one line. Each is a case written with a
#: line mesh in the case (`mesh: {type: lines}`), commented, that passes
#: `vfife check` and runs as it stands -- a starting point to change, not a
#: catalogue of bridge types (tests/test_v12_new.py runs every one).
TEMPLATES = {
    "girder": "three-span continuous girder under its dead load (static; modes)",
    "pier_on_piles": "a pier on a pile in p-y soil, pushed at its top (static; buckling)",
    "arch": "a parabolic arch rib under a deck load (static; buckling, modes)",
    "cable": "a cable hanging under its own weight, by its unstressed length (relaxation)",
}


def template_text(name: str, case_name: str, file_name: str) -> str:
    """The template `name` with the case's name and file put in."""
    from importlib import resources
    if name not in TEMPLATES:
        raise KeyError(name)
    import json
    text = (resources.files("vfife") / "templates" / f"{name}.yaml").read_text(encoding="utf-8")
    # QUOTED where YAML reads them: a case file named 2024.yaml, yes.yaml or
    # "#1.yaml" was written and then refused as a number, a boolean or a
    # comment (review of 1.2)
    text = text.replace("name: @NAME@", "name: " + json.dumps(case_name, ensure_ascii=False))
    text = text.replace("path: results/@NAME@.h5", "path: " + json.dumps(f"results/{case_name}.h5",
                                                                          ensure_ascii=False))
    return text.replace("@NAME@", case_name).replace("@FILE@", file_name)
