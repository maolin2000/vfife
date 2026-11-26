"""Putting a model together (1.22): a case and its file to the mesh and the
material (`build`), to the problem dict the solvers read (`_with_element`),
and to the solver of its family (`make_solver`). Every entry point goes
through these -- the command line, the Python API, the analyses -- and until
1.22 they lived in `cli.py`, so the API imported the command line to build a
model. `vfife.cli` still exports the three names.
"""
from __future__ import annotations

import os

from .core.diagnostics import UserError
from .core.mesh import box_mesh
from .core.solver import Solver
from .io import schema
from . import materials as _materials  # noqa: F401  (registers all)


def build(case: schema.Case, case_path: str | None = None):
    """Build the mesh and material. Returns (mesh, material, element type).

    For a Gmsh mesh the ELEMENT TYPE comes from the file, not from the case:
    the file already says what it contains, and a case that disagreed with it
    would be describing a mesh that does not exist.
    """
    if getattr(case, "response_spectrum", None) is not None:
        # a spectrum file is checked before the run, not read after it
        from . import spectrum
        try:
            spectrum.check_files(case, case_path)
        except ValueError as exc:
            raise UserError("SPECTRUM_FILE", str(exc),
                            "write the file as two columns, period (s) and acceleration, "
                            "periods increasing; the path is relative to the case file") from None
    if case.mesh.type in {"gmsh", "abaqus", "lines"}:
        from .io import gmsh as gmsh_io

        path = case.mesh.path or "the case's mesh.lines"
        if case_path and case.mesh.path and not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.abspath(case_path)), path)
        if case.mesh.type == "lines":
            # a line mesh written in the case (1.1): what gmsh.read gives for
            # a Line2 file
            if float(getattr(case.mesh, "scale", 1.0) or 1.0) != 1.0:
                raise UserError("MESH_SCALE_UNUSED",
                                "mesh.scale applies to a mesh read from a file; `lines` are written "
                                "in the case's own length unit",
                                "write the points in that unit and remove `scale`")
            if case.element.type in {"coupled", "shell_line"}:
                raise UserError("LINES_ARE_LINES",
                                "mesh type `lines` writes line members only; a model with shells or "
                                "solids needs its mesh from Gmsh",
                                "keep `mesh: {type: gmsh, path: ...}` for this model")
            from .io import line_mesh
            try:
                mesh = line_mesh.build(case.mesh)
            except ValueError as exc:
                raise UserError("LINE_MESH", str(exc),
                                "see the manual, chapter 5 (mesh type `lines`)") from None
            element = "truss2"
        elif case.mesh.type == "abaqus":
            from .io import abaqus
            mesh, element = abaqus.read(path, scale=case.mesh.scale,
                                        coupled=case.element.type == "coupled")
        elif case.element.type in {"coupled","shell_line"}:
            mesh, element = gmsh_io.read_coupled(path, scale=case.mesh.scale,
                line_regions={m.region for m in case.members})
        else:
            mesh, element = gmsh_io.read(path, scale=case.mesh.scale,
                                         overlay=case._mesh_overlay)
        if case.mesh.type == "gmsh" and case.mesh.groups:
            try:
                gmsh_io.join_groups(mesh, case.mesh.groups)
            except ValueError as exc:
                raise UserError("MESH_GROUP", str(exc),
                                "a group joins the mesh's physical groups (or other groups) by name") from None
        # A case region that reuses a Gmsh group's name replaced the group's
        # NODES but not its ELEMENTS, so a section assigned to it went on the
        # group's elements. One name, one meaning.
        clash = sorted(set(case.regions) & set(getattr(mesh, "element_regions", {}) or {}))
        if case.mesh.type == "lines":
            # a point's name too: a case region named B replaced the point's
            # one node with whatever it selected (review of 1.1)
            clash = sorted(set(case.regions) & (set(mesh.regions) | set(mesh.element_regions)))
            if clash:
                raise UserError("REGION_NAME_CLASH",
                                f"region {clash[0]!r} is both a point, line or group of mesh.lines and "
                                "a region defined in the case",
                                "rename the case region; to refer to the point or line, use its name")
        if clash:
            raise UserError("REGION_NAME_CLASH",
                            f"region {clash[0]!r} is both a physical group in "
                            f"{os.path.basename(path)} and a region defined in the case",
                            "rename the case region (or the physical group); to refer to the "
                            "group, use its name without redefining it")
        # only complain if the case set element.type ITSELF -- an `element:`
        # block that only sets `formulation` is not a claim about topology
        # Gmsh Line2 is topology, not mechanics.  A YAML case chooses whether
        # that same line mesh represents axial bars or frame members.
        compatible_line = element == "truss2" and case.element.type in {
            "truss2", "beam2", "beam2_timoshenko", "beam2_fiber", "connector2", "mixed_line"
        }
        compatible_coupled = element in {"coupled","shell_line"} and case.element.type == element
        # a shell section lowers to shell4; a triangle mesh makes it shell3
        compatible_shell = element in {"shell3", "shell4"} and case.element.type in {"shell3", "shell4"}
        if "type" not in case.element.model_fields_set and element == "truss2" and not case.members:
            # THE SECTIONS ARE SOLID AND THE MESH IS LINES: a solid section on
            # a Line2 mesh went through preflight and ran as a truss of area
            # NaN (found in review). Said in section terms.
            raise UserError(
                "SECTION_MESH_MISMATCH",
                f"the case's sections are solid or shell, but {os.path.basename(path)} contains only "
                "Line2 elements",
                "a line mesh takes `bar`, `beam`, `fiber` or `connector` sections; a solid section "
                "needs a mesh of volume elements")
        if "type" not in case.element.model_fields_set and element in {"shell3", "shell4"}:
            # the same for a solid section on a shell mesh: it passed
            # preflight and the run died on the missing thickness (found in
            # review)
            raise UserError(
                "SECTION_MESH_MISMATCH",
                f"the case's sections are solid, but {os.path.basename(path)} contains only "
                f"{element} elements",
                "a shell mesh takes a `shell` section with its `thickness`, e.g. "
                "`plate: {type: shell, material: steel, thickness: 0.01}`; a solid section "
                "needs a mesh of volume elements")
        if "type" in case.element.model_fields_set and case.element.type != element and not compatible_line and not compatible_coupled and not compatible_shell:
            default = type(case.element).model_fields["type"].default
            raise UserError(
                "ELEMENT_TYPE_MISMATCH",
                f"the case says element.type = {case.element.type!r} but "
                f"{os.path.basename(path)} contains {element!r}",
                "remove element.type; the file decides."
                + (f"  If you did not write `type: {case.element.type}` "
                   "yourself, something round-tripped this case through "
                   "`Case.as_dict()`, which fills in every default and so turns "
                   "an unset field into a claim -- use `Case.as_written()` for "
                   "that instead." if case.element.type == default else ""),
            )
    else:
        if case.element.type in {"truss2", "beam2", "beam2_timoshenko", "shell4", "connector2", "mixed_line"}:
            raise ValueError(f"{case.element.type} requires mesh.type = gmsh or abaqus with Line2 elements")
        if float(getattr(case.mesh, "scale", 1.0) or 1.0) != 1.0:
            # it was accepted and ignored: the box stayed its written size
            raise UserError("MESH_SCALE_UNUSED",
                            "mesh.scale applies to a mesh read from a file; a box mesh is "
                            "built in the case's own length unit",
                            "write `dimensions` (and `origin`) in that unit and remove `scale`")
        mesh = box_mesh(case.mesh.dimensions, case.mesh.divisions,
                        case.mesh.origin, element=case.element.type)
        element = case.element.type
    from .materials.factory import create_material, wants_auto_length

    # char_length: "auto" resolves to the smallest element V/A_max in the mesh.
    # Crack-band models need the real element size, and hard-coding it in the
    # case file (as the MATLAB setup did) silently goes stale the moment the
    # mesh is refined -- which is precisely when it matters.
    lc = None
    if wants_auto_length(case.material) or any(wants_auto_length(b.material) for b in case.blocks):
        from .elements import continuum
        probe_element = ([k for k in mesh.cell_blocks if k != "truss2"][0]
                         if element == "coupled" else element)
        probe = continuum.precompute(mesh.nodes, mesh.elements,
                                     element=probe_element, formulation="reduced")
        lc = float(probe.char_len.min())

    try:
        material = create_material(case.material, lc)
    except (ValueError, TypeError) as exc:
        # "could not convert string to float: '2e11x'" named the value and
        # not the key it was written under (found in review)
        bad = [k for k, v in (case.material or {}).items()
               if isinstance(v, str) and k != "type" and repr(v) in str(exc)]
        if not bad or isinstance(exc, UserError):
            raise
        raise UserError("MATERIAL_NOT_A_NUMBER",
                        f"material {case.material.get('type', '')!r}: `{bad[0]}` is "
                        f"{case.material[bad[0]]!r}, which is not a number",
                        "write it as a number, e.g. `E: 2.0e+11`") from None
    if element == "truss2" and case.element.type in {"beam2", "beam2_timoshenko", "beam2_fiber", "connector2", "mixed_line"}:
        element = case.element.type
    return mesh, material, element


def _with_element(case: schema.Case, element: str, case_path: str | None = None) -> dict:
    """The problem dict, with the element type the mesh actually contains."""
    p = case.as_dict()
    p["element"] = dict(p["element"])
    p["element"]["type"] = element
    if p["solver"].get("mode") == "static":
        # the implicit static solver drives a solver built
        # as for relaxation: the same state, forces and constraints, and no
        # time; `implicit.run` takes the run over
        p["solver"] = dict(p["solver"], mode="relaxation", static_run=True)
        if p["solver"].get("n_steps") is None:
            p["solver"]["n_steps"] = 1
        # THE DEFAULT TEN LEVELS ARE STATED, NOT ASSUMED. A step without
        # `load_steps` is an UNSTAGED one to `step_data`, whose loads take the
        # `ramp_hold` amplitude -- full at progress 0.5 -- and the ten Newton
        # levels then put the whole load on in five and repeated it in five,
        # labelled 0.6 ... 1.0 (found in review).
        if p["solver"].get("load_steps") is None:
            p["solver"]["load_steps"] = 10
    # Ground-motion files live beside a case just like a Gmsh mesh. Resolve
    # them before handing the plain problem dictionary to a solver.
    if case_path:
        for loads in [p.get("loads") or []] + [s.get("loads") or [] for s in p.get("steps") or []]:
            for load in loads:
                gm = load.get("ground_motion")
                if gm and not os.path.isabs(gm["path"]):
                    gm["path"] = os.path.join(os.path.dirname(os.path.abspath(case_path)), gm["path"])
                # a vehicle's road profile
                rough = (load.get("vehicle") or {}).get("roughness") or {}
                if rough.get("profile") and not os.path.isabs(rough["profile"]):
                    rough["profile"] = os.path.join(os.path.dirname(os.path.abspath(case_path)),
                                                    rough["profile"])
        # and so do the supports' own records
        for cons in [p.get("constraints") or []] + [s.get("constraints") or [] for s in p.get("steps") or []]:
            for c in cons:
                rec = c.get("record")
                if rec and not os.path.isabs(rec["path"]):
                    rec["path"] = os.path.join(os.path.dirname(os.path.abspath(case_path)), rec["path"])
    return p


def make_solver(problem, mesh, material):
    """Choose the independent structural path from the imported topology."""
    from .core.problem import Problem
    from .core.structural import make_structural_solver
    problem = Problem.of(problem)          # validated once, typed
    from .elements.library import families
    # `continuum` is the solid path; every other family is structural.  The
    # membership used to be a literal set here, which meant a new structural
    # element ran silently through the solid solver until someone remembered.
    if families().get(problem.case.element.type) not in (None, "continuum"):
        return make_structural_solver(problem, mesh, material)
    from .core.structural import check_mass_scaling, _refuse_soil
    from .core.rigid_contact import attach_contact
    _refuse_soil(problem, problem.case.element.type)
    check_mass_scaling(problem)
    from .core.creep import refuse_unclocked
    refuse_unclocked(problem, material)      # v1.0
    if (problem.get("contact") or {}).get("members"):
        raise ValueError("contact.members acts between line members (beams, fibre beams, bars, cables); "
                         "a solid model has none -- contact with a solid mesh is not modelled")
    solver = Solver(problem, mesh, material)
    attach_contact(solver, problem, mesh)
    return solver
