"""Command line entry point.

    python -m vfife run   cases/case_01_beam_cantilever.yaml
    python -m vfife info  cases/case_01_beam_cantilever.yaml  # setup only
    python -m vfife list                             # registered materials
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np
import yaml

from pydantic import ValidationError

from .core.diagnostics import AnalysisInterrupted, UserError
from .io.writer import open_result
from .assemble import build, make_solver, _with_element  # noqa: F401  (the names `vfife.cli` exported)
from .io import schema
from . import materials as _materials  # noqa: F401  (registers all)
from .verification import verify as verify_result, write_reports
from .preflight import check as preflight_check, print_summary as print_preflight, write_json as write_preflight


def _ask_region(prompt, choices):
    """Prompt until a named imported physical group is selected."""
    while True:
        value = input(f"{prompt} [{', '.join(choices)}]: ").strip()
        if value in choices:
            return value
        print("Choose one of the listed physical groups.", file=sys.stderr)


def _dof_mask(text: str) -> list[int]:
    """Parse a GUI-friendly/CLI-friendly mask such as ``110`` into dofs."""
    if len(text) != 3 or any(c not in "01" for c in text):
        raise ValueError("DOFS must be a three-character 0/1 mask, e.g. 111 or 100")
    if text == "000":
        raise ValueError("DOFS 000 constrains nothing")
    return [int(c) for c in text]


_DEFAULT_LOAD = [0.0, 0.0, -1000.0]


def _dropped_blocks(path, mesh, element) -> dict:
    """{element type: count} of the blocks in the file that `gmsh.read`
    did not keep as elements (Line2 beside a volume mesh, a shell mesh's
    boundary lines): it reads them only as node sets for their physical
    groups."""
    from .io import gmsh as gmsh_io
    try:
        counts = gmsh_io.survey(path)["counts"]
    except Exception:
        return {}
    kept = set((getattr(mesh, "cell_blocks", None) or {}) or {element: None})
    return {k: int(n) for k, n in sorted(counts.items()) if k not in kept}


def _unassigned_lines(dropped: dict, element: str) -> dict:
    """The Line2 blocks of `_dropped_blocks` beside a VOLUME mesh -- members a
    case solves only when it assigns them a `bar` section. A volume mesh's
    boundary faces and a shell mesh's boundary lines are the node sets of
    its physical groups, and were listed as if they were lost members;
    `init` and `mesh` say the same thing."""
    from .elements.library import families
    fam = families()
    if fam.get(element) != "continuum":
        return {}
    return {k: n for k, n in dropped.items() if fam.get(k) == "line"}


def _ask_vector(prompt, default):
    """Three numbers, or blank to accept the default."""
    shown = " ".join(f"{v:g}" for v in default)
    while True:
        raw = input(f"{prompt} [{shown}]: ").strip()
        if not raw:
            return list(default)
        parts = raw.replace(",", " ").split()
        try:
            if len(parts) != 3:
                raise ValueError
            return [float(v) for v in parts]
        except ValueError:
            print("  give three numbers, e.g. `0 0 -5000`, or press enter.")


def cmd_new(args):
    """A case from a scenario template (1.2): a line mesh written in the case,
    commented, that runs as it stands."""
    from .scaffold import TEMPLATES, template_text
    if args.template:
        args.template = args.template.lower()
    if not args.template or args.list:
        print("templates (vfife new <template> <case.yaml>):")
        w = max(len(k) for k in TEMPLATES)
        for k, v in TEMPLATES.items():
            print(f"  {k:<{w}}  {v}")
        return 0
    if args.template not in TEMPLATES:
        import difflib
        close = difflib.get_close_matches(args.template, list(TEMPLATES), n=1)
        raise UserError("NEW_UNKNOWN_TEMPLATE", f"no template {args.template!r}"
                        + (f" (did you mean {close[0]!r}?)" if close else ""),
                        "`vfife new --list` lists them: " + ", ".join(TEMPLATES))
    case = args.case or f"{args.template}.yaml"
    if not os.path.splitext(case)[1] and not case.endswith(("/", "\\")) \
            and not os.path.isdir(os.path.abspath(case)):
        case += ".yaml"          # `vfife new girder g1` wrote a file named g1 (1.4.1)
    out = os.path.abspath(case)
    if os.path.isdir(out):
        raise UserError("NEW_EXISTS", f"{case} is a folder", "give the case file's name, e.g. "
                        f"{os.path.join(case, args.template + '.yaml')}")
    if os.path.exists(out) and not args.overwrite:
        raise UserError("NEW_EXISTS", f"{case} exists; new does not write over a file",
                        "choose another name, or add --overwrite to replace it")
    stem = os.path.splitext(os.path.basename(out))[0]
    text = template_text(args.template, stem, os.path.basename(out))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("# yaml-language-server: $schema=vfife.schema.json\n" + text)
    print(f"  written       {out} (template {args.template!r})")
    _write_schema_beside(out)
    shown = f'"{case}"' if any(c in case for c in " &()") else case
    print(f"  next          vfife info {shown}   then   vfife run {shown}")
    return 0


def cmd_init(args):
    """Create a generic case from any 3-D Gmsh mesh and its groups."""
    from .io import gmsh as gmsh_io
    from .scaffold import physical_regions, write_generic_gmsh_case

    mesh_arg = args.mesh
    mesh_path = os.path.abspath(mesh_arg)
    out = os.path.abspath(args.case)
    # IT WROTE OVER WHATEVER WAS THERE, its own mesh included -- `init
    # bridge.msh --mesh bridge.msh` left a YAML where the mesh had been
    # (found in review)
    if out == mesh_path or (os.path.exists(out) and os.path.samefile(out, mesh_path)):
        raise UserError("INIT_OVERWRITES_MESH",
                        f"{args.case} is the mesh file itself; the case would replace it",
                        "give the case its own name, e.g. `vfife init case.yaml --mesh "
                        f"{mesh_arg}`")
    if os.path.exists(out) and not args.overwrite:
        raise UserError("INIT_EXISTS", f"{args.case} exists; init does not write over a file",
                        "choose another name, or add --overwrite to replace it")
    if mesh_path.lower().endswith(".mct"):
        # a Midas Civil stage model: its case, its line mesh and the map of
        # what became what, side by side
        from .io import midas, midas_case
        stem = os.path.splitext(os.path.basename(out))[0]
        folder = os.path.dirname(out)
        for ext in (".msh", ".map.json"):
            other = os.path.join(folder, stem + ext)
            if os.path.exists(other) and not args.overwrite:
                raise UserError("INIT_EXISTS", f"{other} exists; init does not write over a file",
                                "choose another name, or add --overwrite to replace it")
        report = midas_case.convert(midas.read(mesh_path), folder, name=stem)
        print(f"wrote {out} (+ {stem}.msh, {stem}.map.json)")
        for key in ("nodes", "members", "rigid_arms", "rigid_link_slaves", "connectors", "regions", "sections", "steps",
                    "jacked_cables", "cables_cut_to_length", "stubs_made_joints", "links_made_to_turn_with_their_ends"):
            print(f"  {key.replace('_', ' '):<36}{report[key]}")
        for what, n in report["left_out"].items():
            print(f"  {'left out':<36}{n} x {what}")
        for note in report["notes"]:
            print(f"  {'note':<36}{note}")
        return 0
    mesh, element = gmsh_io.read(mesh_path, scale=args.scale)
    # THE SCAFFOLD WRITES A SOLID SECTION, whatever the mesh held: a shell
    # mesh passed `check` and crashed on its missing thickness, a line mesh
    # was refused by `info`. Said here, with what to write.
    from .elements.library import families
    family = families().get(element)
    if family != "continuum":
        what = ("a `shell` section with its `thickness`, e.g. "
                "`deck: {type: shell, material: steel, thickness: 0.01}`"
                if family == "surface" else
                "`bar`, `beam`, `fiber` or `connector` sections with their size, e.g. "
                "`girder: {type: beam, material: steel, shape: general, area: 0.01, "
                "iy: 1.0e-4, iz: 1.0e-4, j: 2.0e-4}`, and an `assign` per region")
        raise UserError("INIT_NOT_A_VOLUME_MESH",
                        f"{os.path.basename(mesh_path)} contains {element} elements; `vfife init` "
                        "writes a solid section, which needs volume elements",
                        "write the case by hand (see a case in cases/ with the same element) "
                        "using " + what)
    # a volume mesh with Line2 (or other) elements beside it: the reader
    # keeps the volume, and the rest went unmentioned
    dropped = _dropped_blocks(mesh_path, mesh, element)
    # the groups a PROMPT offers are the boundary-sized ones (a whole-volume
    # group is no support); a group NAMED by an option may be any physical
    # group -- `--fix body 010` is a plane constraint on every node, and
    # `--force body ...` a load spread over the body. Checking the named
    # ones against the prompt's list refused both, saying `body` was no
    # physical group
    choices = physical_regions(mesh.regions, mesh.n_nodes)
    every_group = sorted(n for n in mesh.regions if n not in {"all", "surface"})

    def _no_boundary_group():
        return UserError(
            "NO_PHYSICAL_GROUPS",
            "the mesh has no boundary-sized Gmsh Physical groups to choose from"
            + (f" (it has only {', '.join(every_group)}, the whole volume)" if every_group else ""),
            'add `Physical Surface("support") = {...};` for the faces you will '
            "constrain and load, re-mesh, then run `vfife mesh` to check; or name the "
            "groups with --fix / --force",
        )

    fixed = args.fixed_region
    loaded = args.load_region
    extra_fixes = args.fix or []
    extra_forces = args.force or []
    if args.non_interactive:
        missing = []
        if not fixed and not extra_fixes:
            missing.append("--fixed-region or --fix")
        if not loaded and not extra_forces:
            missing.append("--load-region or --force")
        if missing:
            raise UserError(
                "INIT_INCOMPLETE",
                "non-interactive init needs " + ", ".join(missing),
                f"run `vfife mesh {mesh_arg}` to see the group names, "
                "or drop --non-interactive to be prompted",
            )
    else:
        # --fix / --force ARE the supports and the loads: prompting on top of
        # them added a fully fixed support and a -1000 N load nobody asked
        # for (found in review)
        ask_fixed = not extra_fixes and not fixed
        ask_loaded = not extra_forces and not loaded
        if (ask_fixed or ask_loaded) and not choices:
            raise _no_boundary_group()
        print("Imported Gmsh Physical groups: " + ", ".join(choices or every_group))
        if ask_fixed:
            fixed = _ask_region("Fully fixed support region", choices)
        if ask_loaded:
            loaded = _ask_region("Loaded region", choices)
        # ...and the load itself.  There was no prompt for it, so whatever the
        # user meant, the file got the -1000 N default.
        if loaded and args.load == _DEFAULT_LOAD:
            args.load = _ask_vector(
                f"Total load on {loaded!r} as Fx Fy Fz in N", _DEFAULT_LOAD)
    if not every_group:
        raise _no_boundary_group()
    named_regions = [("fixed", fixed), ("load", loaded)]
    named_regions += [("fix", item[0]) for item in extra_fixes]
    named_regions += [("force", item[0]) for item in extra_forces]
    for flag, value in named_regions:
        if value is None:
            continue
        if value not in every_group:
            raise UserError(
                "UNKNOWN_GROUP",
                f"{flag} region {value!r} is not a Physical group in this mesh",
                "available: " + ", ".join(every_group),
            )

    # Keep the mesh reference relocatable when both files live in a project.
    try:
        ref = os.path.relpath(mesh_path, os.path.dirname(out) or os.curdir)
    except ValueError:
        # Windows, the mesh on another drive: there is no relative path
        ref = os.path.abspath(mesh_path)
    name = args.name or os.path.splitext(os.path.basename(out))[0]
    constraints = []
    if fixed:
        constraints.append({"region": fixed, "dofs": [1, 1, 1]})
    constraints.extend({"region": region, "dofs": _dof_mask(dofs)}
                       for region, dofs in extra_fixes)
    loads = []
    if loaded:
        loads.append({"region": loaded, "vector": list(args.load)})
    loads.extend({"region": item[0], "vector": [float(v) for v in item[1:]]}
                 for item in extra_forces)
    write_generic_gmsh_case(out, name=name, mesh_path=ref,
                            constraints=constraints, loads=loads, scale=args.scale,
                            element=element)
    print(f"  created       {out}")
    _write_schema_beside(out)                # the case's first line names it
    # Line2 beside a volume: members the case does not assign (boundary
    # faces of a volume mesh are its surface groups, not a lost block)
    for name, n in _unassigned_lines(dropped, element).items():
        print(f"  warning       the mesh also holds {n} Line2 element(s); this case does not "
              "assign them, so they are read only as node sets. If they are members, add a "
              "`bar` section and an `assign` entry naming their physical group "
              "(see cases/case_07_coupled_solid_truss.yaml)", file=sys.stderr)
    print("  next          python -m vfife info " + out)
    return 0


def _modeline(schema_path: str, case_dir: str) -> str:
    """The first line of a case that points the YAML extension at the schema."""
    try:
        rel = os.path.relpath(schema_path, case_dir)
    except ValueError:                       # Windows, another drive
        rel = os.path.abspath(schema_path)
    return "# yaml-language-server: $schema=" + rel.replace(os.sep, "/")


def _write_schema_beside(case_out: str) -> None:
    """`init`: vfife.schema.json next to the new case, which its first line
    names; one already there is kept."""
    from .reference import SCHEMA_FILE, schema_text
    path = os.path.join(os.path.dirname(case_out), SCHEMA_FILE)
    text = schema_text()
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"  created       {path} (editor completion; see the manual, Editor completion)")
        return
    try:
        with open(path, encoding="utf-8") as f:
            same = f.read() == text
    except (OSError, UnicodeDecodeError):
        same = False
    if not same:
        print(f"  kept          {path}, which differs from this vfife's schema; "
              "`vfife schema` in that folder refreshes it")


def cmd_schema(args):
    """Write the case file's JSON Schema for an editor."""
    from .reference import SCHEMA_FILE, schema_text
    given = args.output or SCHEMA_FILE
    target = os.path.abspath(given)
    # a folder: one that exists, or a name with no extension (`-o schemas`)
    if os.path.isdir(target) or given.endswith(("/", os.sep)) or not os.path.splitext(target)[1]:
        target = os.path.join(target, SCHEMA_FILE)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        f.write(schema_text())
    print(f"  wrote         {target}")
    print("  first line    " + _modeline(target, os.getcwd()))
    print("                (for a case in this folder: the path is relative to the case; "
          "VS Code reads it with the YAML extension by Red Hat)")
    return 0


def cmd_plot(args):
    """Look at the answer without writing h5py.

    The solver wrote every monitor into the result file and stopped there, so
    seeing a hysteresis loop meant opening HDF5 by hand. That is the chain
    breaking one step before the thing anyone actually wanted.
    """
    from .post import plot, read_monitors

    path = os.path.abspath(args.result)
    out = args.output or os.path.splitext(path)[0] + ".png"
    written = plot(path, out, x=args.x, y=args.y)
    names = sorted(read_monitors(path))
    print(f"written  {written}")
    print(f"  monitors  {', '.join(names)}")
    if args.x is None and not args.y:
        print("  next      --x <monitor> --y <monitor> draws one against the "
              "other, e.g. a load-displacement or hysteresis curve")
    return 0


def cmd_export(args):
    """The monitors as a CSV, for a spreadsheet or someone else's plotting."""
    from .post import to_csv

    path = os.path.abspath(args.result)
    out = args.output or os.path.splitext(path)[0] + ".csv"
    written = to_csv(path, out, monitors=args.monitor)
    with open(written, encoding="utf-8") as fh:
        header = fh.readline().strip()
        rows = sum(1 for _ in fh)
    print(f"written  {written}")
    print(f"  columns  {header}")
    print(f"  rows     {rows}")
    return 0


def cmd_mesh(args):
    """Report what is in a mesh file, before any case file exists.

    This closes a chicken-and-egg: the region names live in the mesh, but the
    only way to see them was to name one correctly in a case file, which is the
    thing you needed them for.  `init --non-interactive` demanded a name it had
    never shown you, and the interactive prompt was the one place they appeared.
    """
    from .io import gmsh as gmsh_io
    from .elements.library import ELEMENTS as _ELEMENTS, element_definition

    path = os.path.abspath(args.mesh)
    if path.lower().endswith(".mct"):
        # a Midas Civil model: what is in it, stages and all
        from .io import midas
        print(midas.summary(midas.read(path)))
        return 0
    try:
        mesh, element = gmsh_io.read(path, scale=args.scale)
    except UserError as exc:
        # A mesh this solver cannot use is exactly the mesh you most need
        # described, so report the census rather than repeating the refusal.
        if exc.code not in ("MIXED_ELEMENT_TYPES", "MIXED_ELEMENT_FAMILIES"):
            raise
        info = gmsh_io.survey(path)
        print(f"mesh: {path}")
        print(f"  nodes         {info['n_nodes']}")
        print("  element types")
        for name, n in sorted(info["counts"].items(), key=lambda kv: -kv[1]):
            print(f"    {name:<16} {n:>8}")
        if info["physical_names"]:
            print("  physical groups " + ", ".join(info["physical_names"]))
        print(f"\n  unusable: {exc.message}")
        print(f"  hint: {exc.hint}")
        return 1
    lo, hi = mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)
    cap = element_definition(element)
    # `element` is the largest block; a mixed mesh solves every block, so the
    # report has to name them all -- and the formulation line has to be the
    # intersection, because one setting has to satisfy all of them.
    solid = {k: len(v) for k, v in (mesh.cell_blocks or {}).items()
             if k in _ELEMENTS}
    blocks = solid if len(solid) > 1 else {element: int(mesh.n_elem)}
    forms = [f for f in cap.formulations
             if all(f in element_definition(b).formulations for b in blocks)]
    # every block in the file, the ones the reader keeps only as node sets
    # too: a volume + Line2 mesh reported the volume alone
    unsolved = _dropped_blocks(path, mesh, element)

    if args.format == "json":
        print(json.dumps({
            "path": path, "element": element, "family": cap.family,
            "blocks": {k: int(v) for k, v in sorted(blocks.items())},
            "node_set_blocks": unsolved,
            "nodes": int(mesh.n_nodes),
            "elements": int(sum(blocks.values())),
            "bounds": {"min": lo.tolist(), "max": hi.tolist()},
            "regions": {k: int(len(v)) for k, v in sorted(mesh.regions.items())},
            "formulations": forms,
            "backends": (list(cap.backends) if len(blocks) == 1 else ["numpy"]),
        }, indent=2))
        return 0

    print(f"mesh: {path}")
    if len(blocks) > 1:
        print("  element       mixed: "
              + " + ".join(f"{k} ({v})" for k, v in sorted(blocks.items()))
              + f", {sum(blocks.values())} elements, {mesh.n_nodes} nodes")
        print("                every block is solved; the blocks share nodes "
              "on their common faces")
    else:
        print(f"  element       {element} ({cap.family}), "
              f"{mesh.n_elem} elements, {mesh.n_nodes} nodes")
    lines = _unassigned_lines(unsolved, element)
    if lines:
        print("  also in file  " + ", ".join(f"{k} ({v})" for k, v in lines.items())
              + " -- read only as node sets for their physical groups; a case solves them "
                "only when it assigns them a section (e.g. a `bar` on Line2 beside a volume)")
    print(f"  formulations  {', '.join(forms) or '(none common to all blocks)'}")
    print("  backends      "
          + (", ".join(cap.backends) if len(blocks) == 1
             else "numpy   (a mixed mesh does not run on numba/warp yet)"))
    print(f"  extent        x [{lo[0]:.4g}, {hi[0]:.4g}]  "
          f"y [{lo[1]:.4g}, {hi[1]:.4g}]  z [{lo[2]:.4g}, {hi[2]:.4g}]")
    if mesh.regions:
        print("  physical groups (use these names in `region:`)")
        for name, idx in sorted(mesh.regions.items()):
            c = mesh.nodes[idx]
            span = ", ".join(
                f"{ax}={c[:, k].min():.4g}" if np.ptp(c[:, k]) < 1e-12
                else f"{ax} [{c[:, k].min():.4g}, {c[:, k].max():.4g}]"
                for k, ax in enumerate("xyz"))
            print(f"    {name:<16} {len(idx):>6} nodes   {span}")
    else:
        print("  physical groups  (none -- add `Physical Surface(\"name\")` "
              "in the .geo and re-mesh)")
    if cap.family in ("line", "surface"):
        # `init` writes a solid section and refuses a line or shell mesh: the
        # hint led straight to that refusal
        print("  next          write the case by hand: `bar`, `beam`, `fiber` or `connector` "
              "sections (a `shell` section for surfaces) and an `assign` per physical group; "
              "a shipped case with the same element is the quickest start (cases/)")
    else:
        print(f"  next          python -m vfife init <case>.yaml --mesh {args.mesh}")
    return 0


def cmd_info(args):
    case, _ = schema.load(args.case)
    mesh, material, element = build(case, args.case)
    preflight = preflight_check(case, mesh, element)
    print(f"case: {case.name}")
    print_preflight(preflight)
    if preflight["status"] != "PASS":
        return 2
    solver = make_solver(_with_element(case, element, args.case), mesh, material)
    # `info` exists so a mistake is found before the run, not after it. A field
    # name the model cannot produce is exactly that kind of mistake, and it used
    # to be discovered in ParaView at the end of a long analysis.
    from .api import check_output_fields
    check_output_fields(solver, case)
    print(solver.summary())
    # Regions that are offered but not yet built are named without being
    # computed -- `surface` costs a pass over every element face, and `info`
    # exists to tell you what is available, not to do the work.
    shown = [f"{k}({len(v)})" for k, v in mesh.regions.items()]
    shown += [f"{k}(available)" for k in sorted(getattr(solver, "_lazy_regions", {}))
              if k not in mesh.regions]
    print(f"  regions       {', '.join(shown)}")
    return 0


def cmd_run(args):
    # Import locally: api.py deliberately reuses the model builders in this
    # module, while the CLI delegates the execution lifecycle back to the API.
    # This keeps one authoritative prepare/run/result path without a cycle at
    # module import time.
    from .api import prepare_case, run_case

    prepared = prepare_case(args.case)
    print(f"case: {prepared.case.name}")
    if prepared.preflight["status"] != "PASS":
        print_preflight(prepared.preflight)
        return 2
    # A PASS is printed once the model is built and its output fields are
    # checked, which `check` counts as preflight: `run` printed "preflight
    # PASS" and then refused the same case for UNKNOWN_OUTPUT_FIELD, where
    # `check` said preflight FAIL
    shown = {"preflight": False}

    def show_preflight():
        if not shown["preflight"]:
            shown["preflight"] = True
            print_preflight(prepared.preflight)
    t0 = time.perf_counter()
    holder = {}

    def on_ready(solver, _prepared):
        holder["solver"] = solver
        show_preflight()
        print(solver.summary())
        if args.restart:
            print(f"  restarted from step {solver.step}")

    static_run = str(getattr(getattr(prepared.case, "solver", None), "mode", "")) == "static"

    def on_progress(step, n, resid, ke, resets):
        now = time.perf_counter()
        elapsed = max(now - t0, 1e-9)
        # an incremental relaxation reports its load INCREMENT here, not the
        # iteration: the rate read "0 steps/s" for a whole run
        it = int(getattr(holder.get("solver"), "step", step) or step)
        if static_run and not getattr(holder.get("solver"), "_in_fallback", False):
            # Newton statics has no kinetic energy and no resets to report:
            # the line said "KE 0.000e+00  resets 3" for the increment count
            sys.stdout.write(f"\r  increment {step:>4}/{n}  {100 * step / max(n, 1):5.1f}%  resid {resid:.2e}  "
                             f"{elapsed:6.1f}s   ")
        elif it != step:
            rate = it / elapsed
            eta = elapsed / max(step, 1) * max(n - step, 0)
            sys.stdout.write(
                f"\r  increment {step:>4}/{n}  "
                f"resid {resid:.2e}  KE {ke:.3e}  resets {resets:<4d} "
                f"{rate:7.0f} iterations/s  eta {eta:6.0f}s   "
            )
        else:
            rate = step / elapsed
            eta = (n - step) / max(rate, 1e-9)
            sys.stdout.write(
                f"\r  step {step:>8}/{n}  {100 * step / n:5.1f}%  "
                f"resid {resid:.2e}  KE {ke:.3e}  resets {resets:<4d} "
                f"{rate:7.0f} steps/s  eta {eta:6.0f}s   "
            )
        sys.stdout.flush()

    try:
        outcome = run_case(prepared, output=args.output, restart=args.restart,
                           on_ready=on_ready, on_progress=on_progress,
                           progress_every=args.progress_every)
    except UserError as exc:
        if not shown["preflight"] and exc.code in ("UNKNOWN_OUTPUT_FIELD", "UNKNOWN_MONITOR_FIELD"):
            rep = prepared.preflight
            rep["issues"].append({"severity": "ERROR", "code": exc.code, "message": exc.message,
                                  "details": {"hint": exc.hint} if exc.hint else None})
            rep["counts"]["ERROR"] += 1
            rep["status"] = "FAIL"
            show_preflight()
            if exc.hint:
                print(f"  hint: {exc.hint}", file=sys.stderr)
            return 2
        show_preflight()
        raise
    except AnalysisInterrupted as stop:
        # the writer was closed and the checkpoint saved before this was raised
        sys.stdout.write("\n")
        print(f"  stopped       at step {holder['solver'].step}", file=sys.stderr)
        # the checkpoint as a restart will find it: this run's own, or none
        # (a stale one beside it is not)
        from .io.writer import load_checkpoint
        try:
            held = load_checkpoint(stop.result_path)
        except Exception:
            held = None
        if held is not None and not held.legacy:
            print(f"  partial result {stop.result_path} "
                  f"(checkpoint {os.path.basename(held.path)}; resume with --restart {stop.result_path} -o <new file>)")
        else:
            print(f"  partial result {stop.result_path} (no checkpoint: stopped at once, before one was written)")
        # 128 + the signal, as a shell reports it: 143 for SIGTERM (a
        # cluster's time limit), 130 for Ctrl-C or a GUI's stop
        import signal as _signal
        if stop.signal is not None and stop.signal == getattr(_signal, "SIGTERM", None):
            return 128 + int(stop.signal)
        return 130
    sys.stdout.write("\n")
    solver = holder["solver"]
    if not outcome.converged:
        print(f"  analysis      {outcome.status}: {outcome.message}", file=sys.stderr)
        print(f"  failed result {outcome.result_path} "
              "(diagnostics embedded; not a valid equilibrium result)")
        return 3
    # THE FIRST QUESTION, ANSWERED FIRST. This block used to print the wall
    # time, the displacement, the monitors and the file -- and never the one
    # word an engineer reads first. The status was in the exit code and in
    # the result file's diagnostics, which is no help to someone watching the
    # terminal.
    report = outcome.diagnostics or {}
    resid = report.get("residual")
    where = ""
    if report.get("load_step") and len(getattr(solver, "steps", [])) >= 1:
        # it is the ANALYSIS step the run ended in (v1.0: it read
        # "2 load step(s)" for a run of seven increments in two steps)
        where = f", analysis step {report['load_step']} of {len(solver.steps)}"
    print(f"  analysis      {outcome.status}"
          + (f" (residual {resid:.2e}{where})" if resid is not None else where))
    print(f"  wall time     {solver.wall_time:.2f} s "
          f"({solver.step / solver.wall_time:.0f} steps/s, "
          f"{solver.step * solver.nel / solver.wall_time / 1e6:.2f} M element-steps/s)")
    umax = float(abs(solver.u).max())
    print(f"  max |u|       {umax:.6e}")
    # erosion is a continuum thing; a structural solver has no counter, and
    # reading one it does not have crashed EVERY `vfife run` of a frame, truss
    # or shell case after its result was already written -- exit 1 under an
    # "analysis CONVERGED" line (the tests run those cases
    # through the API, which is why nothing caught it)
    if getattr(solver, "n_eroded", 0):
        print(f"  eroded        {solver.n_eroded} of {solver.nel} elements "
              "(the answer is for what was left)")
    # and the same line for the OTHER way an element can stop carrying load.
    # Reported separately because the two differ in the thing that matters
    # afterwards: an eroded element's mass is gone, a fractured one's is not.
    frac = getattr(solver, "fracture", None)
    if frac is not None and frac.n_fractured:
        first = frac.events[0]
        print(f"  fractured     {frac.n_fractured} of {solver.nel} elements "
              f"(first: element {first[1]} at t = {first[0]:.6g}; mass kept)")
    # a vehicle that lost the road: the manual says it is recorded, and it
    # was only counted
    for vname, c in (report.get("vehicle_contact") or {}).items():
        print(f"  vehicle       {vname!r}: tyre off the road {c['tyre_lift_off_steps']} axle-step(s), "
              f"rigid axle in tension {c['rigid_axle_tension_steps']} (first at t = {c['first_time']:.6g})")
    for name, mon in solver.monitors.items():
        if mon.values:
            print(f"  {name:<13} final {mon.values[-1]:.6e}")
    extra = ", _members.xdmf" if prepared.element in {"coupled","shell_line"} else ""
    print(f"  written       {outcome.result_path} "
          f"(+ .xdmf time series, _last.xdmf fallback{extra} for ParaView; .ckpt.h5 checkpoint)")
    return 0


def cmd_list(args):
    from .catalog import library
    data = library()
    kind = getattr(args, "kind", "all")
    selected = {}
    if kind in {"all", "elements"}:
        selected["elements"] = sorted(data["elements"])
    if kind in {"all", "materials"}:
        selected["materials"] = sorted(data["materials"])
    if getattr(args, "format", "text") == "json":
        print(json.dumps({"schema_version": data["schema_version"], **selected}, indent=2))
    else:
        for name, values in selected.items():
            print(f"{name}: " + ", ".join(values))
    return 0


def cmd_describe(args):
    """Describe one registered element or material without initializing it."""
    from .catalog import library
    data = library()
    matches = [(kind, values[args.name]) for kind, values in
               (("element", data["elements"]), ("material", data["materials"]))
               if args.name in values]
    if not matches:
        choices = sorted((*data["elements"], *data["materials"]))
        raise ValueError(f"unknown element or material {args.name!r}; available: {choices}")
    kind, item = matches[0]
    from .elements.library import takes
    if kind == "element" and takes(args.name, "static") and "static" not in item.get("analysis_modes", ()):
        # the implicit static solver runs these; `describe`
        # listed only dynamic and relaxation for them
        item = {**item, "analysis_modes": list(item["analysis_modes"]) + ["static"]}
    payload = {"schema_version": data["schema_version"], "kind": kind, **item}
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(f"{kind}: {args.name}")
        for key, value in item.items():
            if key == "name":
                continue
            if isinstance(value, (list, tuple)):
                value = ", ".join(str(v) for v in value) or "none"
            elif isinstance(value, bool):
                value = "yes" if value else "no"
            print(f"  {key:<22} {value}")
    return 0


def cmd_capabilities(args):
    """Query valid element/material/backend/mode combinations."""
    from .catalog import compatibility, library
    from .elements.library import takes, taking
    data = library()
    elements = [args.element] if args.element else sorted(data["elements"])
    materials = [args.material] if args.material else sorted(data["materials"])
    unknown_e = [x for x in elements if x not in data["elements"]]
    unknown_m = [x for x in materials if x not in data["materials"]]
    if unknown_e or unknown_m:
        raise ValueError("unknown catalogue selection: " + ", ".join(unknown_e + unknown_m))
    rows = []
    # `static` is a case's mode too, and was refused here:
    # the implicit solver drives the model built for relaxation, on the
    # frame and fibre-beam paths only (implicit.check)
    static = args.mode == "static"
    for element in elements:
        for material in materials:
            ok, reasons = compatibility(element, material, backend=args.backend,
                                        mode="relaxation" if static else args.mode)
            if static and not takes(element, "static"):
                ok, reasons = False, tuple(reasons) + (
                    f"solver.mode = static runs {', '.join(taking('static'))} models; "
                    f"{element} takes mode: relaxation",)
            if ok or args.show_unsupported:
                rows.append({"element": element, "material": material,
                             "backend": args.backend, "mode": args.mode,
                             "supported": ok, "reasons": list(reasons)})
    payload = {"schema_version": data["schema_version"], "filters": {
        "element": args.element, "material": args.material,
        "backend": args.backend, "mode": args.mode,
    }, "combinations": rows}
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    elif not rows:
        print("no supported combinations match; use --show-unsupported for reasons")
    else:
        for row in rows:
            state = "SUPPORTED" if row["supported"] else "UNSUPPORTED"
            suffix = "" if row["supported"] else ": " + "; ".join(row["reasons"])
            filters = ""
            if args.backend:
                filters += f", backend={args.backend}"
            if args.mode:
                filters += f", mode={args.mode}"
            print(f"{state:<11} {row['element']} + {row['material']}{filters}{suffix}")
    return 0


def cmd_qualify_check(args):
    from .qualification import assess, load
    report = assess(load(args.manifest), root=os.getcwd())
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(f"qualification {report['status']}: {report['kind']} {report['name']} "
              f"-> {report['target_level']}")
        if report["missing"]:
            print("  missing gates: " + ", ".join(report["missing"]))
        if report["invalid_references"]:
            print("  invalid references: " + ", ".join(report["invalid_references"]))
        if report["missing_backend_coverage"]:
            print("  backend coverage missing: " + ", ".join(report["missing_backend_coverage"]))
    return 0 if report["status"] == "PASS" else 2


def cmd_qualify_template(args):
    from .qualification import template
    text = yaml.safe_dump(template(args.kind, args.name, args.level), sort_keys=False)
    if args.output:
        path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.force:
            raise ValueError(f"refusing to overwrite {path}; use --force")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"written       {path}")
    else:
        print(text, end="")
    return 0


def cmd_expand(args):
    """Write what a case's high-level blocks expand to: the case and its mesh
    as the solver reads them. To look at, not to edit: the
    case itself is what is kept."""
    from . import prep
    from .io import gmsh as gmsh_io
    with open(args.case, "r", encoding="utf-8") as fh:
        d = schema.safe_load_named(fh.read(), args.case)
    base = os.path.dirname(os.path.abspath(args.case))
    if not prep.has_high_level(d):
        raise UserError("NOTHING_TO_EXPAND", f"{args.case} has no high-level block ({', '.join(prep.HIGH_LEVEL)}, `table:`)",
                        "the case is already what the solver reads")
    notes = []
    try:
        low, overlay, report = prep.expand_case(d, base, warn=lambda who, t: notes.append(f"{who}: {t}"))
    except ValueError as exc:
        raise UserError("HIGH_LEVEL_BLOCK", str(exc), "see the manual, 11.14, 11.15 and 11.16") from None
    out = args.output or os.path.splitext(args.case)[0] + "_expanded.yaml"
    low = dict(low)
    here = os.path.dirname(os.path.abspath(out))
    if overlay is not None:
        msh = os.path.splitext(out)[0] + ".msh"
        scale = float((d.get("mesh") or {}).get("scale") or 1.0)
        try:
            gmsh_io.write_expanded(prep.mesh_path(d, base), msh, scale=scale, overlay=overlay)
        except ValueError as exc:
            raise UserError("EXPAND_MESH", str(exc), "write the mesh as MSH 2.2 ASCII and expand again") from None
        low["mesh"] = {k: v for k, v in dict(low.get("mesh") or {}).items() if k != "scale"}
        low["mesh"]["path"] = os.path.relpath(msh, here)
    elif (low.get("mesh") or {}).get("path") and not os.path.isabs(low["mesh"]["path"]):
        # the same mesh, from where the expanded case is
        low["mesh"] = dict(low["mesh"], path=os.path.relpath(os.path.join(base, low["mesh"]["path"]), here))
        msh = None
    with open(out, "w", encoding="utf-8") as fh:
        # every float as it is (the compact style rounds to 10 digits)
        yaml.safe_dump(low, fh, sort_keys=False, allow_unicode=True, width=120)
    for n in notes:
        print(f"  note          {n}")
    tendons = report.get("tendons", {})
    if tendons:
        print(f"  tendons       {len(tendons)}: {sum(len(t['segments']) for t in tendons.values())} bars, "
              f"{sum(len(t['segments']) + 1 for t in tendons.values())} arms")
    print(f"written       {out}")
    if overlay is not None:
        print(f"written       {msh}")
    return 0


def cmd_migrate(args):
    """Rewrite a case in the input form (materials / sections / assign)."""
    from .io.migrate import migrate, check_round_trip
    from .io.yamlfmt import dump_compact

    with open(args.case, "r", encoding="utf-8") as fh:
        text = fh.read()
    old = schema.safe_load_named(text, args.case)
    from .io.model import is_input_form
    if is_input_form(old):
        # already the input form: bring it to the current case format,
        # in place in the text, and prove the model did
        # not change
        from .io.migrate import upgrade_text, same_model
        try:
            out, notes = upgrade_text(text)
        except (yaml.YAMLError, ValueError) as exc:
            raise UserError("MIGRATE_FAILED", f"{args.case}: {exc}",
                            "fix the file by hand (it is valid YAML to read, but not to rewrite this way)")
        if not notes:
            print(f"{args.case} is already case format {old.get('version')}")
            return 0
        diffs = same_model(text, out)
        if diffs:
            raise UserError(
                "MIGRATE_MISMATCH",
                f"the rewritten case does not lower to the same model ({len(diffs)} differences)",
                "first difference: " + diffs[0] + " -- report this with the case file")
        target = args.output or args.case
        if os.path.abspath(target) == os.path.abspath(args.case) and not args.in_place:
            print(out, end="")
            for n in notes:
                print(f"# migrate: {n}", file=sys.stderr)
            return 0
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(out)
        for n in notes:
            print(f"  {n}")
        print(f"written       {target}")
        return 0
    new = migrate(old)
    diffs = check_round_trip(old, new)
    if diffs:
        raise UserError(
            "MIGRATE_MISMATCH",
            f"the migrated case does not lower back to the original ({len(diffs)} differences)",
            "first difference: " + diffs[0] + " -- report this with the case file",
        )
    # keep the file's leading comment block: it is the analyst's own record
    header = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            header.append(line)
        else:
            break
    body = dump_compact(new)
    out = "\n".join(header).rstrip("\n") + ("\n\n" if header else "") + body
    target = args.output or args.case
    if os.path.abspath(target) == os.path.abspath(args.case) and not args.in_place:
        print(out, end="")
        return 0
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"written       {target}")
    return 0


def _default_contract(case_path, stem):
    """The contract beside the case, not beside the working directory:
    `verification/<stem>.yaml` next to the case's folder
    (the shipped layout, cases/ and verification/ side by side), in it, then
    under the working directory."""
    here = os.path.dirname(os.path.abspath(case_path))
    tried = [os.path.join(os.path.dirname(here), "verification", stem + ".yaml"),
             os.path.join(here, "verification", stem + ".yaml"),
             os.path.join("verification", stem + ".yaml")]
    for c in tried:
        if os.path.exists(c):
            return c
    raise UserError("NO_CONTRACT", f"no verification contract for {stem}: looked for "
                    + ", ".join(dict.fromkeys(os.path.normpath(c) for c in tried)),
                    "name it with --contract <file.yaml>")


def _default_result(case_path, stem):
    """The result the case's `output.path` names (relative to the working
    directory, as `vfife run` writes it), else results/<stem>.h5."""
    try:
        from .io.schema import safe_load_named
        with open(case_path, encoding="utf-8") as fh:
            out = (safe_load_named(fh.read(), case_path) or {}).get("output") or {}
        if out.get("path") and os.path.exists(out["path"]):
            return out["path"]
    except Exception:
        pass
    return os.path.join("results", stem + ".h5")


def cmd_verify(args):
    """Evaluate an HDF5 result against an independent verification contract."""
    case_stem = os.path.splitext(os.path.basename(args.case))[0]
    contract = args.contract or _default_contract(args.case, case_stem)
    result = args.result or _default_result(args.case, case_stem)
    report_dir = args.report_dir or os.path.join("results", "verification")
    report = verify_result(contract, result)
    backend = report["execution"]["backend"]
    device = report["execution"].get("device")
    tag = backend + (("_" + str(device).replace(":", "")) if device else "")
    stem = os.path.join(report_dir, case_stem + "_" + tag)
    write_reports(report, stem + ".json", stem + ".md")
    print(f"verification  {report['status']} ({len(report['checks'])} checks)")
    print(f"written       {stem}.json (+ .md)")
    for check in report["checks"]:
        print(f"  {check['status']:4s}  {check['id']}: {check['value']}")
    return 0 if report["status"] == "PASS" else 2


def cmd_probe(args):
    """Ask the invariants of the user's own model.

    `vfife check` looks at the case file; `verify` looks at a finished result.
    Between them is the case that RAN and did not converge, where the message
    is a residual and a node number and the question is what kind of trouble
    it is. This asks the model itself: is the force field a gradient (if not,
    kinetic damping has nothing to descend and no budget will help), does a
    rigid motion of the deformed body cost anything, does the fictitious mass
    still bound the stiffest mode in the configuration the model has reached,
    and is that configuration stable at all.

    With no `--result` the questions are asked at the reference configuration
    and at an arc of `--bend` radians; with one they are asked at the state
    the run reached, which for a failed run is the state it stalled in.

    The loads are the FIRST step's, at full amplitude: this is a question
    about the model, not a replay of its history.
    """
    from .testing import invariants
    from .testing.state import probe as make_probe

    case, case_text = schema.load(args.case)
    mesh, material, element = build(case, args.case)
    solver = make_solver(_with_element(case, element, args.case), mesh, material)
    p = make_probe(solver)
    print(f"case: {case.name}")
    print(f"  model         {element}, {p.nn} nodes, {p.n} dofs, "
          f"{int(p.fixed.sum())} held")
    states = []
    if args.result:
        with open_result(args.result) as h:
            frames = sorted(h["steps"], key=int)
            from .modal import frame_key
            frame = frame_key(frames, args.frame, args.result)     # 3338 or 00003338
            group = h["steps"][frame]
            u = group["displacement"][...].astype(float)
            rot = (group["rotation"][...].astype(float)
                   if p.rotation.size and "rotation" in group else None)
            d = dict(h["diagnostics"].attrs) if "diagnostics" in h else {}
        # the full state (materials, creep, connector slips) when the
        # checkpoint holds this frame: a converged frame read as positions on
        # virgin materials reported a relative residual of 66
        from .modal import restore_state
        x, frame, _k, full, missing, note = restore_state(
            solver, case, element, args.case, args.result, frame)
        p = make_probe(solver)
        if note:
            print(f"  note          {note}")
        if missing and p.rotation.size:
            print("  note          the result carries no `rotation` field, so the "
                  "points' triads are read as the reference ones; add it to "
                  "`output.fields` for a frame this can be read from")
        states.append((f"the state of frame {frame}"
                       + (f" ({d['status']})" if "status" in d else ""), x))
    else:
        states.append(("the reference configuration", np.zeros(p.n)))
        if args.bend:
            states.append((f"an arc of {args.bend} rad", p.bent(args.bend)))
    failed = False
    for label, x in states:
        r = np.abs(p.residual(x)).max()
        e = np.abs(p.external(x)).max()
        print(f"\n  at {label}"
              + (f"  (relative residual {r / e:.2e})" if e > 0 else ""))
        # a self-stressed model (prestress, temperature, creep...) is not
        # force-free at its drawing geometry, and that geometry or an arc of
        # it is no equilibrium to ask for stability; a --result frame is
        self_stressed = "" if args.result else invariants.self_stress(case_text)
        for f in invariants.report(p, x, max_dofs=args.max_dofs,
                                   reference_state=(label == states[0][0]),
                                   self_stressed=self_stressed):
            print(f)
            failed |= not f.passed
    if not args.no_materials:
        failed |= _probe_materials(case, material)
    print("\n  A skew tangent on the LOAD is a load that is not the gradient of a\n"
          "  potential; kinetic damping is a descent method and cannot relax it.\n"
          "  An element tangent is unsymmetric away from equilibrium in proportion\n"
          "  to the out-of-balance force -- read it at a converged state.\n"
          "  See docs/convergence_debugging.md.")
    return 2 if failed else 0


def _probe_materials(case, built) -> bool:
    """The constitutive laws of this case, asked the questions of v0.1.

    The model questions above are about the mesh, the supports and the loads;
    these are about the law, on a strain path of the harness's choosing, with
    the ANALYST's parameters. A law that is fine in the library can fail here
    on a case's own numbers -- a curve whose branches do not meet, a reference
    length far from the element size -- and that is the point of asking it on
    the case rather than only in the test suite.
    """
    from .testing import materials as mat_invariants

    blocks = {}
    if getattr(case, "material", None):
        blocks["the case material"] = (case.material, built)
    for b in getattr(case, "blocks", None) or []:
        if getattr(b, "material", None):
            blocks[f"block {b.name!r}"] = (b.material, None)
    failed = False
    for label, (props, resolved) in blocks.items():
        try:
            spec = mat_invariants.spec_for(props, resolved)
        except ValueError as exc:
            print(f"\n  {label}: not asked ({exc})")
            continue
        print(f"\n  {label}: {spec.name}, driven to a strain of {spec.amplitude:.1e}")
        for f in mat_invariants.report_spec(spec):
            print(f)
            failed |= not f.passed
    return failed


def cmd_check(args):
    """Run model checks and build the model on numpy -- no GPU, no solve."""
    case, _ = schema.load(args.case)
    mesh, material, element = build(case, args.case)
    report = preflight_check(case, mesh, element)
    if report["status"] == "PASS":
        _check_output_fields_into(report, case, mesh, material, element, args.case)
    print(f"case: {case.name}")
    print_preflight(report)
    output = args.output or os.path.join("results", "preflight", case.name + ".json")
    write_preflight(report, output)
    print(f"  written       {output}")
    return 0 if report["status"] == "PASS" else 2


def _check_output_fields_into(report, case, mesh, material, element, case_path) -> None:
    """The model built (on numpy, so no GPU is initialised) and the
    `output.fields` check of `info` and `run`, as preflight issues.

    The manual (3.3) says `check` refuses a field name the model cannot
    produce; it did not, because the check asks a built solver (found in
    review). The build is part of EVERY check that passed
    preflight: built only when the case named its fields, a
    model the run refuses at construction passed `check` without them and
    failed it with them -- and a build ValueError left no report at all."""
    from .api import check_output_fields
    def fail(code, message, hint=None):
        report["issues"].append({"severity": "ERROR", "code": code, "message": message,
                                 "details": {"hint": hint} if hint else None})
        report["counts"]["ERROR"] += 1
        report["status"] = "FAIL"
    try:
        p = _with_element(case, element, case_path)
        p["solver"] = dict(p["solver"], backend="numpy")
        solver = make_solver(p, mesh, material)
    except UserError as exc:
        fail("MODEL_BUILD_FAILED", f"the model could not be built ({exc.code}): {exc.message}", exc.hint)
        return
    except ValueError as exc:
        # an input mistake the solver refuses when it is built: the run
        # would stop here too
        fail("MODEL_BUILD_FAILED", f"the model could not be built: {exc}",
             "run `vfife --debug info <case>` for the stack")
        return
    except Exception as exc:
        fail("MODEL_BUILD_FAILED", f"the model could not be built: {type(exc).__name__}: {exc}",
             "run `vfife --debug info <case>` for the stack")
        return
    try:
        check_output_fields(solver, case)
    except UserError as exc:
        if exc.code not in ("UNKNOWN_OUTPUT_FIELD", "UNKNOWN_MONITOR_FIELD"):
            raise
        fail(exc.code, exc.message, exc.hint)


def cmd_modal(args):
    """Natural frequencies and mode shapes at a state."""
    from . import modal
    from .testing.state import probe as make_probe

    case, _ = schema.load(args.case)
    mesh, material, element = build(case, args.case)
    solver = make_solver(_with_element(case, element, args.case), mesh, material)
    p = make_probe(solver)
    print(f"case: {case.name}")
    print(f"  model         {element}, {p.nn} nodes, {p.n} dofs, {int(p.fixed.sum())} held")
    if args.result:
        # the full state when the checkpoint holds this frame -- materials,
        # creep, connector slips -- and the supports of the
        # step the frame is in, not step 1's
        x, frame, k, full, missing_rot, note = modal.restore_state(
            solver, case, element, args.case, args.result, args.frame)
        p = make_probe(solver)
        print(f"  state         frame {frame} of {args.result} (step {k + 1}, "
              f"{int(p.fixed.sum())} dofs held; "
              + ("full state from its checkpoint)" if full else "positions only)"))
        if note:
            print(f"  note          {note}")
        if missing_rot:
            print("  note          the result carries no `rotation` field; the points' triads "
                  "are the reference ones")
        for line in modal.connector_kinks(solver) if full else ():
            print(f"  warning       connector {line}: the finite-difference tangent there is "
                  "the mean of its two branches, which is neither state")
    else:
        x = np.zeros(p.n)
        print("  state         the reference configuration (unloaded; for a cable or a "
              "prestressed member ask at a loaded state with --result)")
    res = modal.analyse(solver, n_modes=args.modes, x=x, max_dofs=args.max_dofs)
    print(f"  mass          {res.total_mass[0]:.4g} kg in x, {res.total_mass[1]:.4g} in y, "
          f"{res.total_mass[2]:.4g} in z on the free translational dofs"
          + (f"; {res.n_condensed} massless dof(s) condensed" if res.n_condensed else "")
          + (f"; {res.n_dropped} dropped" if res.n_dropped else ""))
    print(res.table())
    if res.note:
        print(f"  note          {res.note}")
    out = args.output or os.path.join("results", os.path.splitext(os.path.basename(args.case))[0] + "_modes.h5")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    modal.write_modes(out, res, p.nn, solver.dofs.stride if hasattr(solver, "dofs") else 3)
    print(f"  written       {out}")
    if args.paraview is not None:
        from .post import modes_vtk
        pv_dir = args.paraview or os.path.splitext(out)[0] + "_paraview"
        modes_vtk.write(solver, res, pv_dir, stem="modes", frames=args.frames,
                        scale=args.mode_scale)
    return 0


def cmd_buckling(args):
    """Linear buckling factors of a step's loads (1.1)."""
    from . import buckling, modal
    from .testing.state import probe as make_probe

    case, _ = schema.load(args.case)
    mesh, material, element = build(case, args.case)
    solver = make_solver(_with_element(case, element, args.case), mesh, material)
    p = make_probe(solver)
    print(f"case: {case.name}")
    print(f"  model         {element}, {p.nn} nodes, {p.n} dofs")
    state = None
    if args.result:
        if args.step is not None:
            raise UserError("BUCKLING_STEP_AND_RESULT",
                            "--step and --result both choose the loads: the result's frame is in a step",
                            "give one: --result (the loads of the step the frame is in) or --step")
        state, frame, k, full, _missing, note = modal.restore_state(
            solver, case, element, args.case, args.result, args.frame)
        print(f"  base          frame {frame} of {args.result}"
              + (" (its whole state: materials, creep, connectors)" if full else " (positions only)")
              + f"; loads of step {k + 1} ADDED to it at full amplitude")
        if note:
            print(f"  note          {note}")
    else:
        steps = getattr(solver, "steps", None) or [{}]
        k = (args.step or 1) - 1
        if not 0 <= k < len(steps):
            raise UserError("BUCKLING_STEP_OUT_OF_RANGE", f"--step {args.step}: the case has "
                            f"{len(steps)} step(s)", "steps are counted from 1")
        modal.use_step_at(solver, k + 1 if getattr(solver, "mode", "") != "dynamic"
                          else (k + 1) * float(getattr(solver, "total_time", 1.0) or 1.0))
        print(f"  base          the reference configuration; loads of step {k + 1} at full amplitude "
              "(every load of it, self-weight included)")
    res = buckling.analyse(solver, n_modes=args.modes, state=state, max_dofs=args.max_dofs)
    base_solver = solver
    pb = make_probe(base_solver)
    unit = (case.units or {}).get("force", "")
    print(f"  load          resultant {res.load[0]:.4g}, {res.load[1]:.4g}, {res.load[2]:.4g} {unit} "
          f"in x, y, z on the free dofs; {res.n_free} free dofs")
    print(res.table())
    if res.note:
        print(f"  note          {res.note}")
    out = args.output or os.path.join("results", os.path.splitext(os.path.basename(args.case))[0] + "_buckling.h5")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    buckling.write(out, res, pb.nn, base_solver.dofs.stride if hasattr(base_solver, "dofs") else 3)
    print(f"  written       {out}")
    if args.paraview is not None and len(res.factors):
        from .post import modes_vtk
        pv_dir = args.paraview or os.path.splitext(out)[0] + "_paraview"
        rows = ["mode  load factor"] + [f"{i:4d}  {lam:.6g}" for i, lam in enumerate(res.factors, 1)]
        modes_vtk.write(base_solver, res, pv_dir, stem="buckling", frames=args.frames,
                        scale=args.mode_scale, rows=rows)
    return 0


def cmd_sweep(args):
    from .sweep import write_series
    # the result `run` wrote: the case's own output.path (relative to the
    # working directory, as `run` reads it), not results/<stem>.h5
    result = args.result or schema.load(args.case)[0].output.path
    pvd = write_series(args.case, result, out_dir=args.output, every=args.every, surface=args.surface)
    print(f"  written       {pvd} (open in ParaView; the frames are beside it)")
    return 0


def cmd_view(args):
    cases = [p for p in args.result if p.lower().endswith((".yaml", ".yml"))]
    if cases:
        # a case, before it is run: the model page
        from .core.diagnostics import UserError
        if len(args.result) != 1:
            raise UserError("VIEW_MODEL", "give one case file (the model page shows one model), or result files",
                            "vfife view bridge.yaml  |  vfife view result.h5 [more.h5 ...]")
        extra = [o for o, v in (("--case", args.case), ("--names", args.names), ("--hide", args.hide),
                                ("--fields", args.fields), ("--every", args.every != 1),
                                ("--max-frames", args.max_frames != 400)) if v]
        if extra:
            raise UserError("VIEW_MODEL", f"{', '.join(extra)} {'is' if len(extra) == 1 else 'are'} for results; "
                            "the model page of a case takes -o, --title, --thickness and --scripts",
                            "vfife view bridge.yaml [-o page.html] [--scripts DIR]")
        from .post.model_view import write as write_model
        out = write_model(cases[0], output=args.output, title=args.title, thickness=args.thickness,
                          scripts=args.scripts)
        size = os.path.getsize(out) / 1e6
        where = "three.js is in the page" if args.scripts else "three.js comes from a CDN; --scripts DIR puts it in the page"
        print(f"  written       {out} ({size:.1f} MB; open it in a browser -- {where})")
        return 0
    from .post.view import write
    # empty entries dropped, as for --hide and --fields ("a, b," is two names)
    names = [n.strip() for n in args.names.split(",") if n.strip()] if args.names else None
    hide = [n.strip() for n in args.hide.split(",") if n.strip()] if args.hide else None
    keep = [n.strip() for n in args.fields.split(",") if n.strip()] if args.fields else None
    out = write(args.result, output=args.output, every=args.every, max_frames=args.max_frames,
                thickness=args.thickness, title=args.title, case=args.case, names=names, hide=hide,
                fields=keep, scripts=args.scripts)
    size = os.path.getsize(out) / 1e6
    where = "three.js is in the page" if args.scripts else "three.js comes from a CDN; --scripts DIR puts it in the page"
    print(f"  written       {out} ({size:.1f} MB; open it in a browser -- {where})")
    return 0


def cmd_tendons(args):
    """The force in each tendon of `tendons:` at the end of each step:
    least, mean and largest over its bars, and the mean
    against the jacking force; `--bars` every bar, `--csv` to a file."""
    import csv
    from .core.diagnostics import UserError
    from .io.writer import open_result
    from .post import tendons as post_tendons
    try:
        rows = post_tendons.bars(args.result) if args.bars else post_tendons.table(args.result)
    except ValueError as exc:
        raise UserError("RESULT_INCOMPLETE", str(exc),
                        "a frame with `axial_force` in it is needed: see the run's status and `output.fields`") from None
    if not rows:
        print(post_tendons.text([]))
        return 0
    if args.csv:
        keys = [k for k in rows[0] if k != "step_index"]
        with open(args.csv, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if r[k] is None else r[k]) for k in keys})
        print(f"wrote {args.csv} ({len(rows)} rows)")
        return 0
    with open_result(args.result) as f:
        unit = str(f.attrs.get("unit_force", ""))
    if args.bars:
        print("step\ttendon\tbar\tregion\tforce\tjack" + (f"\t[{unit}]" if unit else ""))
        for r in rows:
            force = "-" if r["force"] is None else f"{r['force']:.6g}"
            print(f"{r['step']}\t{r['tendon']}\t{r['bar']}\t{r['region']}\t{force}\t{r['jack']:.6g}")
    else:
        print(post_tendons.text(rows, unit))
    return 0


def cmd_summary(args):
    """One page on a result (1.1): status, what the model weighs against the
    reactions, the largest displacement / force / stress over the frames and
    where, energy, monitors."""
    import json
    from .post.summary import summarize, text
    from .post.summary import _finite
    s = summarize(args.result, case=args.case)
    if args.case and s.get("case"):
        from .io import schema as _schema
        try:
            name = _schema.load(args.case)[0].name
        except Exception:
            name = None
        if name and name != s["case"]:
            s.setdefault("notes", []).append(
                f"--case is {name!r} but the result was run from {s['case']!r}: the mass and "
                "weight are that case's")
    if args.json:
        print(json.dumps(_finite(s), indent=1, ensure_ascii=False))
    else:
        if not (s.get("totals") or args.case):
            s.setdefault("notes", []).append(
                "this result was written before 1.1 and does not carry the model's mass and "
                "weight: add --case <case.yaml> for them")
        print(text(s))
        for n in s.get("notes", []):
            print(f"  note          {n}")
    return 0


def cmd_formfind(args):
    from . import formfind
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    from pathlib import Path
    out = args.output or str(Path(args.case).with_name(Path(args.case).stem + "_found.yaml"))
    # the folder of `-o` is made BEFORE the search: it failed with "no such
    # file" after the whole search
    if os.path.dirname(os.path.abspath(out)):
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    try:
        found = formfind.find(args.case, groups, tol=args.tol, max_iter=args.max_iter,
                              step_index=args.step)
    except UserError as exc:
        if exc.code != "FORMFIND_NOT_RELAXED":
            raise
        # an inner run that did not converge is an analysis that did not
        # converge: exit 3, as `vfife run`
        print(f"  formfind      NOT CONVERGED: {exc.message}", file=sys.stderr)
        print(f"  hint: {exc.hint}", file=sys.stderr)
        return 3
    for region, forces in found.forces.items():
        f = np.asarray(forces)
        print(f"  {region:<14} {len(f)} elements, tension {f.min():.4g} .. {f.max():.4g} N")
    formfind.write_found_case(args.case, found, out)
    status = "CONVERGED" if found.converged else "NOT CONVERGED"
    print(f"  formfind      {status} ({found.iterations} {found.solves}, max displacement "
          f"{found.max_displacement:.3e} m)")
    print(f"  written       {out}")
    return 0 if found.converged else 3


def _console_safe():
    """A character the console's code page lacks (a Chinese Windows pipe is
    GBK) is printed as an escape instead of ending the run with a
    UnicodeEncodeError halfway through a log line."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if (getattr(stream, "encoding", "") or "").lower().replace("-", "") != "utf8":
                stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass


def _named_file(args, order=("case", "result", "mesh", "manifest")) -> str:
    """The file a command was given, to name in a one-line error."""
    for attr in order:
        v = getattr(args, attr, None)
        if v:
            return ", ".join(v) if isinstance(v, list) else str(v)
    return "the input"


def _h5_name(exc) -> str | None:
    """The file an h5py OSError names in its text (`name = '...'`), if any."""
    import re
    m = re.search(r"name = '([^']+)'", str(exc))
    return m.group(1) if m else None


def _named_files(args) -> str:
    """Every file a command was given (output first), for a message that
    cannot tell which of them failed."""
    seen = []
    for attr in ("output", "result", "case", "mesh", "manifest", "contract"):
        v = getattr(args, attr, None)
        for x in (v if isinstance(v, list) else [v]):
            if x and str(x) not in seen:
                seen.append(str(x))
    return ", ".join(seen)


def main(argv=None):
    _console_safe()
    ap = argparse.ArgumentParser(prog="vfife")
    ap.add_argument("--debug", action="store_true",
                    help="let exceptions escape with their traceback")
    from . import __version__
    ap.add_argument("--version", action="version", version=f"vfife {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="solve a case")
    r.add_argument("case")
    r.add_argument("-o", "--output")
    r.add_argument("--restart", help="result file to resume from")
    r.add_argument("--progress-every", type=int, default=200)
    r.set_defaults(func=cmd_run)

    me = sub.add_parser("mesh", help="report what is in a mesh file")
    me.add_argument("mesh")
    me.add_argument("--scale", type=float, default=1.0)
    me.add_argument("--format", choices=("text", "json"), default="text")
    me.set_defaults(func=cmd_mesh)

    i = sub.add_parser("info", help="set up a case and print a summary")
    i.add_argument("case")
    i.set_defaults(func=cmd_info)

    l = sub.add_parser("list", help="list registered materials and elements")
    l.add_argument("kind", nargs="?", choices=("all", "elements", "materials"),
                   default="all")
    l.add_argument("--format", choices=("text", "json"), default="text")
    l.set_defaults(func=cmd_list)

    d = sub.add_parser("describe", help="describe one element or material")
    d.add_argument("name")
    d.add_argument("--format", choices=("text", "json"), default="text")
    d.set_defaults(func=cmd_describe)

    cap = sub.add_parser("capabilities", help="query supported analysis combinations")
    cap.add_argument("--element")
    cap.add_argument("--material")
    cap.add_argument("--backend", choices=("numpy", "numba", "warp"))
    cap.add_argument("--mode", choices=("dynamic", "relaxation", "static"))
    cap.add_argument("--show-unsupported", action="store_true",
                     help="include rejected combinations and their reasons")
    cap.add_argument("--format", choices=("text", "json"), default="text")
    cap.set_defaults(func=cmd_capabilities)

    ref = sub.add_parser("reference", help="regenerate docs/YAML参考.md and docs/vfife.schema.json "
                                           "from the input model (--api: docs/API参考.md from vfife.api)")
    ref.add_argument("-o", "--output", default=None,
                     help="write the YAML reference here instead of docs/YAML参考.md (and skip the schema)")
    ref.add_argument("--api", action="store_true", help="the Python API reference instead")
    ref.add_argument("--codes", action="store_true",
                     help="the error-code catalogue docs/error_codes.md instead")
    ref.add_argument("--check", action="store_true",
                     help="exit 1 when the file on disk differs from what would be generated")
    ref.set_defaults(func=lambda a: __import__("vfife.reference", fromlist=["cmd_reference"]).cmd_reference(a))

    sc = sub.add_parser("schema", help="write the case file's JSON Schema, for completion "
                                       "and checking in an editor (VS Code: the YAML extension)")
    sc.add_argument("-o", "--output", default=None,
                    help="file (with an extension) or folder to write; default "
                         "vfife.schema.json in the current folder")
    sc.set_defaults(func=cmd_schema)

    qualify = sub.add_parser("qualify", help="create or check library admission evidence")
    qsub = qualify.add_subparsers(dest="qualify_cmd", required=True)
    qc = qsub.add_parser("check", help="check an element/material qualification manifest")
    qc.add_argument("manifest")
    qc.add_argument("--format", choices=("text", "json"), default="text")
    qc.set_defaults(func=cmd_qualify_check)
    qt = qsub.add_parser("template", help="emit a qualification manifest template")
    qt.add_argument("kind", choices=("element", "material"))
    qt.add_argument("name")
    qt.add_argument("--level", choices=("registered", "verified", "benchmarked", "production"),
                    default="verified")
    qt.add_argument("-o", "--output")
    qt.add_argument("--force", action="store_true")
    qt.set_defaults(func=cmd_qualify_template)

    nw = sub.add_parser("new", help="a case from a scenario template (a girder, a pier on piles, "
                                    "an arch, a cable) that runs as it stands")
    nw.add_argument("template", nargs="?", help="template name (`--list` lists them)")
    nw.add_argument("case", nargs="?", help="YAML file to create (default <template>.yaml)")
    nw.add_argument("--list", action="store_true", help="list the templates")
    nw.add_argument("--overwrite", action="store_true", help="replace an existing file")
    nw.set_defaults(func=cmd_new)

    init = sub.add_parser("init", help="create a generic case from a Gmsh mesh")
    init.add_argument("case", help="YAML file to create")
    init.add_argument("--mesh", required=True, help="3-D Gmsh .msh file")
    init.add_argument("--name", help="case name; defaults to the YAML stem")
    init.add_argument("--scale", type=float, default=1.0,
                      help="mesh-coordinate scale, e.g. 0.001 for mm to m")
    init.add_argument("--fixed-region", help="Gmsh Physical group to fully fix")
    init.add_argument("--load-region", help="Gmsh Physical group receiving the load")
    init.add_argument("--load", type=float, nargs=3, default=_DEFAULT_LOAD,
                      metavar=("FX", "FY", "FZ"), help="total load vector in N")
    init.add_argument("--non-interactive", action="store_true",
                      help="require region flags instead of prompting")
    init.add_argument("--fix", nargs=2, action="append", metavar=("REGION", "DOFS"),
                      help="add a support using a 0/1 DOF mask, e.g. --fix symmetry 100")
    init.add_argument("--force", nargs=4, action="append",
                      metavar=("REGION", "FX", "FY", "FZ"),
                      help="add a total force; repeat for multiple load regions")
    # not `--force`, which `init` already spends on loads
    init.add_argument("--overwrite", action="store_true",
                      help="replace the case file if it exists (never the mesh)")
    init.set_defaults(func=cmd_init)

    check = sub.add_parser("check", help="validate model quality before solver/GPU initialization")
    check.add_argument("case")
    check.add_argument("-o", "--output", help="JSON report path")
    check.set_defaults(func=cmd_check)

    pr = sub.add_parser("probe", help="ask the model the invariants: is the force a "
                                      "gradient, is a rigid motion free, does the mass "
                                      "bound the step, is the state stable")
    pr.add_argument("case")
    pr.add_argument("--result", help="ask at the state this result reached "
                                     "(the state a failed run stalled in)")
    pr.add_argument("--frame", help="which frame of the result, by its step number (default: the last)")
    pr.add_argument("--bend", type=float, default=0.8,
                    help="also ask at an arc of this many radians (0 to skip)")
    pr.add_argument("--max-dofs", type=int, default=4000,
                    help="above this the finite-difference tangent is skipped")
    pr.add_argument("--no-materials", action="store_true",
                    help="skip the constitutive-law questions")
    pr.set_defaults(func=cmd_probe)

    mg = sub.add_parser("migrate", help="rewrite a case in the current case format: the old form to materials / sections / assign, old spellings renamed, `version` set")
    mg.add_argument("case", help="case YAML in the old form")
    mg.add_argument("-o", "--output", help="write here instead of printing")
    mg.add_argument("--in-place", action="store_true", help="overwrite the case file")
    mg.set_defaults(func=cmd_migrate)

    ex = sub.add_parser("expand", help="write what a case's high-level blocks (tendons) expand to: the case and mesh the solver reads, to look at")
    ex.add_argument("case", help="case YAML with a high-level block")
    ex.add_argument("-o", "--output", help="the expanded case (default <case>_expanded.yaml; its mesh beside it)")
    ex.set_defaults(func=cmd_expand)

    v = sub.add_parser("verify", help="check a result against its verification contract")
    v.add_argument("case", help="numbered case YAML (selects the default contract/result names)")
    v.add_argument("--result", help="HDF5 result; defaults to results/<case-stem>.h5")
    v.add_argument("--contract", help="contract YAML; defaults to verification/<case-stem>.yaml")
    v.add_argument("--report-dir", help="JSON/Markdown directory; defaults to results/verification")
    v.set_defaults(func=cmd_verify)

    pl = sub.add_parser("plot", help="plot the monitors from a result file")
    pl.add_argument("result", help="HDF5 result written by `vfife run`")
    pl.add_argument("-o", "--output", help="PNG to write; defaults to <result>.png")
    pl.add_argument("--x", help="monitor for the horizontal axis; with --y this "
                               "gives a load-displacement (or hysteresis) curve")
    pl.add_argument("--y", action="append",
                    help="monitor to plot; repeatable. Default: all of them")
    pl.set_defaults(func=cmd_plot)

    ex = sub.add_parser("export", help="write the monitors out as CSV")
    ex.add_argument("result", help="HDF5 result written by `vfife run`")
    ex.add_argument("-o", "--output", help="CSV to write; defaults to <result>.csv")
    ex.add_argument("--monitor", action="append",
                    help="monitor to include; repeatable. Default: all of them")
    ex.set_defaults(func=cmd_export)

    ff = sub.add_parser("formfind", help="find the cables' unstressed lengths that leave the "
                                          "mesh at its dead-load geometry")
    ff.add_argument("case")
    ff.add_argument("--groups", required=True,
                    help="comma-separated member regions (bar sections on line regions) to find")
    ff.add_argument("-o", "--output", help="write the found case here (default: <case>_found.yaml)")
    ff.add_argument("--tol", type=float, default=1.0e-3, help="max node displacement, m (1e-3)")
    ff.add_argument("--max-iter", type=int, default=25)
    ff.add_argument("--step", type=int, default=0,
                    help="the step that is the dead load, by its index from 0 (0)")
    ff.set_defaults(func=cmd_formfind)

    md = sub.add_parser("modal", help="natural frequencies and mode shapes at a state "
                                      "(finite-difference tangent, physical mass)")
    md.add_argument("case")
    md.add_argument("--modes", type=int, default=6, help="how many, lowest first (6)")
    md.add_argument("--result", help="ask at the state of this result (a loaded, deformed "
                                     "configuration); default: the reference configuration")
    md.add_argument("--frame", help="which frame of the result, by its step number (default: the last)")
    md.add_argument("--max-dofs", type=int, default=6000,
                    help="refuse above this many dofs (the tangent costs 2 n force evaluations)")
    md.add_argument("--paraview", nargs="?", const="", default=None, metavar="DIR",
                    help="also write the mode shapes as ParaView time series (one .pvd per "
                         "mode, one period each, plus <DIR>/modes_all.pvd); DIR defaults to "
                         "results/<case>_modes_paraview")
    md.add_argument("--frames", type=int, default=24, help="frames per period (24)")
    md.add_argument("--mode-scale", type=float, default=None,
                    help="largest drawn translation, in the case's length unit "
                         "(default 3%% of the model's largest extent)")
    md.add_argument("-o", "--output", help="HDF5 of the modes (default results/<case>_modes.h5)")
    md.set_defaults(func=cmd_modal)

    bk = sub.add_parser("buckling", help="buckling: the load factors at which the model loses its "
                                         "stiffness under a step's loads, and the shapes")
    bk.add_argument("case")
    bk.add_argument("--modes", type=int, default=6, help="how many, lowest factor first (6)")
    bk.add_argument("--step", type=int, default=None,
                    help="whose loads (counted from 1; default the first step)")
    bk.add_argument("--result", help="start from the state of this result instead (its materials, "
                                     "creep, bearings): the factor on the step's loads added to it")
    bk.add_argument("--frame", help="which frame of the result, by its step number (default: the last)")
    bk.add_argument("--max-dofs", type=int, default=6000,
                    help="refuse above this many dofs (each tangent costs 2 n force evaluations)")
    bk.add_argument("--paraview", nargs="?", const="", default=None, metavar="DIR",
                    help="also write the buckling shapes as ParaView series (as `modal --paraview`)")
    bk.add_argument("--frames", type=int, default=24, help="frames per shape (24)")
    bk.add_argument("--mode-scale", type=float, default=None,
                    help="largest drawn translation (default 3%% of the model's largest extent)")
    bk.add_argument("-o", "--output", help="HDF5 of the factors and shapes (default "
                                           "results/<case>_buckling.h5)")
    bk.set_defaults(func=cmd_buckling)

    vw = sub.add_parser("view", help="write a result as one self-contained web page that "
                                     "replays it in 3-D (members, shells, solids; fields, breaks, monitors); "
                                     "or a case (.yaml), before it is run: its members, supports and loads step by step")
    vw.add_argument("result", nargs="+", help="HDF5 result file(s) of ONE model; several become the page's acts. "
                                              "Or one case file (.yaml): the model page (frame models)")
    vw.add_argument("-o", "--output", help="the page to write (default <first result>.html, <case>.model.html)")
    vw.add_argument("--case", help="the case file: line members are then drawn at their sections' size")
    vw.add_argument("--names", help="comma-separated act names, one per result")
    vw.add_argument("--hide", help="comma-separated regions of the case not to draw (rigid arms, links)")
    vw.add_argument("--fields", help="comma-separated fields to put in the page (default: every scalar field; "
                                     "the displacement components are always there)")
    vw.add_argument("--every", type=int, default=1, help="take every k-th frame")
    vw.add_argument("--max-frames", type=int, default=400, help="at most this many frames (evenly picked)")
    vw.add_argument("--thickness", type=float, help="drawn thickness of line members (default 1/300 of the model); on the model page of a case, the beams' line width in pixels (default 2)")
    vw.add_argument("--title", help="the page's title (default the case name)")
    vw.add_argument("--scripts", metavar="DIR", help="a folder with three.js r128's three.min.js and OrbitControls.js: written "
                                      "into the page, which then needs no network (where the CDNs are not reachable)")
    vw.set_defaults(func=cmd_view)

    tn = sub.add_parser("tendons", help="the force in each tendon of `tendons:` at the end of each step: "
                                        "least, mean and largest over its bars, and the mean against the jacking force")
    tn.add_argument("result", help="HDF5 result file")
    tn.add_argument("--bars", action="store_true", help="every bar of every tendon, not one row per tendon")
    tn.add_argument("--csv", help="write the rows to this CSV file instead of printing them")
    tn.set_defaults(func=cmd_tendons)

    sm = sub.add_parser("summary", help="one page on a result: status, mass and weight against the "
                                         "reactions, the largest displacement, member force and stress "
                                         "and where, energy, monitors")
    sm.add_argument("result", help="HDF5 result file")
    sm.add_argument("--case", help="the case file, for the mass and weight of a result written before 1.1")
    sm.add_argument("--json", action="store_true", help="print it as JSON")
    sm.set_defaults(func=cmd_summary)

    sw = sub.add_parser("sweep", help="sweep the fibre sections along the deformed members "
                                      "into a surface (.pvd + .vtu per frame) for ParaView")
    sw.add_argument("case")
    sw.add_argument("--result", help="HDF5 result; defaults to the case's own output.path "
                                     "(relative to the working directory, as `run` writes it)")
    sw.add_argument("-o", "--output", help="directory for the frames (default <result>_sweep/)")
    sw.add_argument("--every", type=int, default=1, help="write every k-th frame (the last always)")
    sw.add_argument("--surface", action="store_true",
                    help="the outline surface only, instead of a solid of fibre cells")
    sw.set_defaults(func=cmd_sweep)

    args = ap.parse_args(argv)
    if getattr(args, "debug", False):
        return args.func(args)
    try:
        return args.func(args)
    except UserError as exc:
        # a mistake in the input is a sentence, not a stack trace; its code
        # (docs/error_codes.md) is the stable name a script can match on
        # (v1.0: the docs cited codes the terminal never showed)
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"  hint: {exc.hint}", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"error: {os.path.basename(str(getattr(args, 'case', '')))} "
              "is not a valid case file", file=sys.stderr)
        for e in exc.errors():
            where = ".".join(str(p) for p in e["loc"]) or "(top level)"
            print(f"  {where}: {e['msg']}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        # `exc.filename` is set by `open()` and NOT by `np.loadtxt`, which
        # raises with the name only in its message. A missing accelerogram
        # therefore reported "no such file: None" -- neither what was missing
        # nor which of a case's several paths it was.
        print(f"error: no such file: {exc.filename}" if exc.filename
              else f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        # A material or a curve refusing its own numbers -- "rho_c = 2.102 is
        # outside (0,1)" from a case written in MPa while declaring Pa -- came
        # out as a traceback, because only UserError was a sentence. These are
        # input mistakes too, and they already carry an explanation; what they
        # lacked was somewhere to be printed. `--debug` still
        # gives the stack.
        print(f"error: {exc}", file=sys.stderr)
        # `check` itself was told to run `check`; and the
        # unit hint only where the message is about a number's size -- it was
        # appended to "add both to output.fields" and "a different mesh"
        import re as _re
        about_size = bool(_re.search(r"outside|out of range|range|magnitude|too (large|small|big|stiff|soft)"
                                     r"|exceed|negative|unit|> ?[0-9]|< ?[0-9]", str(exc), _re.I))
        units = ("" if args.func is cmd_check or not about_size else
                 " If the numbers look like another unit system, `vfife check <case>` reports the "
                 "magnitudes against the units the case declares.")
        # --debug is the program's option, before the command: `vfife run
        # x.yaml --debug` was refused as an unknown argument
        cmd = getattr(args, "cmd", None) or "<command>"
        print(f"  hint: run again as `vfife --debug {cmd} ...` for the stack." + units, file=sys.stderr)
        return 1
    # three more input mistakes that came out as a traceback:
    # a case that is not YAML, a result that is not HDF5, and a prompt that
    # met the end of its input
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark is not None else ""
        what = getattr(exc, "problem", None) or str(exc).splitlines()[0]
        # the file the parser was reading, not the case the command was
        # given (a broken contract was reported as the case)
        name = getattr(mark, "name", None)
        if not name or str(name).startswith("<"):
            name = _named_file(args)
        print(f"error: {name} is not valid YAML{where}: {what}", file=sys.stderr)
        return 1
    except EOFError:
        print("error: the input ended before a prompt was answered", file=sys.stderr)
        print("  hint: give the answers as options instead, e.g. `vfife init case.yaml --mesh m.msh "
              "--non-interactive --fixed-region <group> --load-region <group>`", file=sys.stderr)
        return 1
    except BrokenPipeError:
        # the reader went away (`vfife capabilities | head`): nothing to say.
        # stdout is pointed at devnull so the interpreter's own flush at exit
        # cannot raise again (the recipe of Python's `signal` docs), and the
        # exit is 141 = 128 + SIGPIPE, what a tool killed by the broken pipe
        # reports -- not 1, which here means an input mistake
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        except (OSError, ValueError):
            pass
        return 141
    except OSError as exc:
        # "cannot read <case>" of a result it could not WRITE (locked by
        # another program, a folder that is a file): the path is the one the
        # error names, and the verb is both
        name = exc.filename or _h5_name(exc)
        if name:
            print(f"error: cannot read or write {name}: {exc.strerror or exc}", file=sys.stderr)
        else:
            print(f"error: cannot read or write a file: {exc}", file=sys.stderr)
            given = _named_files(args)
            if given:
                print(f"  hint: the files of this command are {given}", file=sys.stderr)
        if "signature" in str(exc):
            print("  hint: it is not an HDF5 file; a vfife result is the .h5 `vfife run` "
                  "writes", file=sys.stderr)
        if "lock" in str(exc):
            print("  hint: another program (ParaView, HDFView, a Python session) has the file "
                  "open; close it, or write elsewhere with -o", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # cmd_run has already closed its writer in a finally block
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
