"""The YAML reference, generated from the code.

`docs/yaml_reference.md` answers two questions a hand-written manual keeps getting
wrong as the library moves: WHAT can a case file say, and WHERE is a working
example. Both are read from the source of truth rather than remembered:

  * the fields, their types, defaults and doc comments come from the Pydantic
    input model (`vfife/io/model.py`, `vfife/io/schema.py`) -- every field's
    `#:` comment is its documentation;
  * the materials and elements come from the catalogue (`vfife.catalog`);
  * the "used by" column comes from walking every shipped `cases/case_*.yaml`,
    so a field that no case exercises is listed by name at the end -- either
    it wants an example or it is dead.

`tests/test_reference.py` regenerates the files and fails when it differs
from the one in the repository, so a new field without its documentation is
a red test, not a stale page.
"""
from __future__ import annotations

import collections
import inspect
import pathlib
import re
import typing
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from . import __version__
from .catalog import library
from .io import model as im
from .io import schema as sc
from .prep.offsets import OffsetArmSpec
from .prep.tendons import TendonSpec

ROOT = pathlib.Path(__file__).resolve().parent.parent
MATERIALS = frozenset(library()["materials"])

# ---------------------------------------------------------------- the layout
# The YAML blocks in the order a case is read, each with the class that
# validates it. A `*` level is a user-chosen name (a section, a region); a
# `[]` level is a list. Nested model classes are found from the annotations.
BLOCKS: List[Tuple[str, Optional[type], str]] = [
    ("version", None,
     "Case format version: `1` (from 1.0); left out, it is read by the same rules as `1` (old spellings read with a DEPRECATED_FIELD warning); a larger number is refused. `vfife migrate` adds it"),
    ("name", None,
     "The case's name, also the result file's default title"),
    ("units", None,
     "The units declared (`length`, `force`, `stress`, `time`; also `moment`, `mass`, `temperature`, `density`): preflight checks magnitudes by them; numbers written are not converted, defaults that carry a unit are converted to them; `stress` without `length` or `force` has them inferred from the other two (`{force: N, stress: MPa}` is mm), or UNITS_INCOMPLETE refuses; the result records them as `unit_*` attributes"),
    ("include", None,
     "Merge the materials, sections, reinforcement, groups, tendons and tendon_defaults of other files: `[path, ...]`, or `{path, override: true}` for this file's definitions to win; a repeated name is refused"),
    ("groups.*", None,
     "Member groups: name -> `[mesh groups or group names, ...]`; read as one group of the mesh made of those parts, usable wherever a region name is; lowered to `mesh.groups`"),
    ("mesh", sc.MeshSpec,
     "The mesh: a structured box, a Gmsh `.msh`, an Abaqus `.inp`, or a line mesh written in the case (`lines`, 1.1)"),
    ("materials.*", None,
     "Material table: name -> `{type, parameters...}`; each material's parameters are in part 3"),
    ("sections.*", im.SectionSpec,
     "Sections: geometry + material, assigned to regions by `assign`"),
    ("reinforcement.*", im.ReinforcementGeometry,
     "Reinforcement and tendon geometry, drawn in drawing coordinates, independent of the mesh"),
    ("regions.*", sc.RegionSpec,
     "Named node sets: by geometry, by Gmsh physical group, by number, or set operations"),
    ("assign[]", im.AssignSpec,
     "The one place a section meets geometry: region <- section (+ reinforcement, bond, prestress, fracture)"),
    ("tendon_defaults", None,
     "Settings the tendons share: a key a tendon leaves out is taken from here; mappings (`stress`, `friction`, ...) merge key by key, the tendon's own winning; a profile may be `{table: file.csv}`, rows taken by tendon name"),
    ("offset_arms", OffsetArmSpec,
     "Rigid arms of members set off their nodes: `{material, factor, limit, regions}`, one rule instead of a section per arm; each arm's section is computed on reading from the stiffness of the members it joins"),
    ("tendons[]", TendonSpec,
     "Tendons of a frame model, one entry each: profile, area, jacking, friction, draw-in, relaxation; expanded on reading into tendon nodes, rigid arms and jacked bars"),
    ("constraints[]", sc.ConstraintSpec,
     "Constraints: fixed degrees of freedom, or prescribed displacement, velocity or acceleration histories"),
    ("loads[]", sc.LoadSpec,
     "Loads: forces, moments, line loads, tractions, pressure, body force, gravity, temperature, moving loads, vehicles, ground motion"),
    ("nodal_masses[]", sc.NodalMassSpec,
     "Added nodal masses: by default in three directions and with weight; may be directional (added mass, no weight) and carry rotary inertia"),
    ("hydrodynamic_mass[]", sc.HydrodynamicMassSpec,
     "Morison added mass of submerged members: transverse, weightless"),
    ("soil_springs[]", sc.SoilSpringSpec,
     "Soil springs on piles: p-y curves by depth, nonlinear elastic (1.2)"),
    ("diaphragms[]", sc.DiaphragmSpec,
     "Rigid diaphragms: the region's nodes move as a rigid body in a plane"),
    ("amplitudes.*", None,
     "Amplitude tables: name -> `[[progress or time, factor], ...]`, or `hold`; loads and constraints refer to them by name"),
    ("steps[]", sc.StepSpec,
     "Analysis steps: each with its own constraints, loads and iteration budget, run in order"),
    ("removals[]", sc.RemovalSpec,
     "Timed removals: a region's members leave at a given time (a cable-loss scenario)"),
    ("contact", sc.ContactSpec,
     "Contact with rigid surfaces: ground, sea bed, pile cap -- a node that touches gets a penalty force, with Coulomb friction in dynamics"),
    ("response_spectrum", sc.ResponseSpectrumSpec,
     "Response spectrum: modes at the equilibrium state the run ends in, each mode's peak from the spectrum, combined (CQC/SRSS)"),
    ("solver", sc.SolverSpec,
     "Solution: dynamic relaxation (statics) or explicit dynamics; damping, time step, backend, mass scaling, initial state"),
    ("output", sc.OutputSpec,
     "Output: result file, output interval, fields, checkpoints, monitors"),
]


# Blocks the input model types as a free dict but a schema class defines:
# documented as their own tables, and the walk records their fields too.
EXTRA: Dict[str, type] = {
    "sections.*.fracture": sc.FractureSpec,
    "assign[].bond": sc.BondSpec,
    "assign[].friction": sc.TendonFrictionSpec,
    "assign[].failure": sc.FailureSpec,
}

# Internal-form blocks: written by the lowering, accepted from old case files.
INTERNAL = ["element", "material", "blocks", "members"]

CASE_GLOB = "cases/case_*.yaml"


# ---------------------------------------------------------------- doc comments
_FIELD_RE = re.compile(r"^    (\w+)\s*:")


def field_docs(cls: type) -> Dict[str, str]:
    """`#:` comments above each field of a Pydantic class, joined: the
    field's documentation."""
    out: Dict[str, str] = {}
    pending: List[str] = []
    for raw in inspect.getsource(cls).splitlines():
        line = raw.rstrip()
        s = line.strip()
        if s.startswith("#:"):
            pending.append(s[2:].strip())
            continue
        m = _FIELD_RE.match(line)
        if m and not s.startswith("model_config"):
            if pending:
                out[m.group(1)] = " ".join(pending)
            pending = []
        elif s and not s.startswith("#"):
            pending = []
    return out


def _unwrap(tp):
    """The model class inside Optional[...] / List[...] / Dict[str, ...], if any,
    and how it is reached ('', '[]' or '*')."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is typing.Union:
        for a in args:
            if a is not type(None):
                return _unwrap(a)
        return None, ""
    if origin in (list, List):
        inner, how = _unwrap(args[0]) if args else (None, "")
        return inner, "[]" if inner is not None else ""
    if origin in (dict, Dict):
        inner, how = _unwrap(args[1]) if len(args) == 2 else (None, "")
        return inner, "*" if inner is not None else ""
    if inspect.isclass(tp) and issubclass(tp, BaseModel):
        return tp, ""
    return None, ""


def type_name(tp) -> str:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is typing.Union:
        inner = [a for a in args if a is not type(None)]
        return " \\| ".join(type_name(a) for a in inner)
    if origin is typing.Literal:
        return " \\| ".join(f"`{a}`" for a in args)
    if origin in (list, List):
        return f"list[{type_name(args[0])}]" if args else "list"
    if origin in (dict, Dict):
        return f"dict[{type_name(args[0])}, {type_name(args[1])}]" if args else "dict"
    if tp is Any:
        return "any"
    if inspect.isclass(tp):
        if issubclass(tp, BaseModel):
            return f"see {tp.__name__}"
        return {"str": "str", "float": "float", "int": "int", "bool": "bool"}.get(tp.__name__, tp.__name__)
    return str(tp).replace("typing.", "")


def default_text(field) -> str:
    d = field.default
    if d is PydanticUndefined:
        if field.default_factory is not None:
            return "—"
        return "**required**"
    if isinstance(d, BaseModel):
        return "(default block)"
    if d is None:
        return "—"
    if isinstance(d, (list, dict)) and not d:
        return "empty"
    return f"`{d}`"


# ---------------------------------------------------------------- usage walk
def _walk(value, cls: Optional[type], path: str, hits: Dict[str, set], case: str):
    """Record every field of `cls` that `value` (a YAML dict) sets, then
    recurse into nested model classes."""
    if not isinstance(value, dict):
        return
    if cls is None:
        return
    fields = cls.model_fields
    by_alias = {f.alias or n: n for n, f in fields.items()}
    for key, v in value.items():
        name = by_alias.get(key, key)
        if name not in fields:
            continue
        fpath = f"{path}.{name}" if path else name
        hits[fpath].add(case)
        inner, how = _unwrap(fields[name].annotation)
        if inner is None:
            # an inline material or bond law: {type: <material>, ...}
            if isinstance(v, dict) and v.get("type") in MATERIALS:
                _walk_material(v, hits, case)
            if fpath in EXTRA:
                _walk(v, EXTRA[fpath], fpath, hits, case)
            continue
        if how == "[]" and isinstance(v, list):
            for item in v:
                _walk(item, inner, fpath + "[]", hits, case)
        elif how == "*" and isinstance(v, dict):
            for item in v.values():
                _walk(item, inner, fpath + ".*", hits, case)
        else:
            _walk(v, inner, fpath, hits, case)


def _walk_material(mat: dict, hits: Dict[str, set], case: str):
    if not isinstance(mat, dict) or "type" not in mat:
        return
    t = mat["type"]
    hits[f"material:{t}"].add(case)
    for k in mat:
        if k != "type":
            hits[f"material:{t}.{k}"].add(case)


def usage(cases_dir: pathlib.Path) -> Tuple[Dict[str, set], List[str]]:
    hits: Dict[str, set] = collections.defaultdict(set)
    names: List[str] = []
    for p in sorted(cases_dir.glob("case_*.yaml")):
        case = p.stem[5:].split("_")[0]           # case_24b_hzb_... -> 24b
        names.append(case)
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for block, cls, _ in BLOCKS:
            key = block.rstrip("*[]").rstrip(".")
            if key not in d:
                continue
            hits[key].add(case)
            v = d[key]
            if block.endswith("[]") and isinstance(v, list):
                for item in v:
                    _walk(item, cls, key + "[]", hits, case)
            elif block.endswith("*") and isinstance(v, dict):
                for item in v.values():
                    if key == "materials":
                        _walk_material(item, hits, case)
                    else:
                        _walk(item, cls, key + ".*", hits, case)
            else:
                _walk(v, cls, key, hits, case)
        # materials may also be named inline on a section or a bar
        for s in (d.get("sections") or {}).values():
            if isinstance(s, dict) and isinstance(s.get("material"), dict):
                _walk_material(s["material"], hits, case)
        for k in INTERNAL:
            if k in d:
                hits[f"internal:{k}"].add(case)
    return hits, names


def _cases(hits: Dict[str, set], key: str, limit: int = 8) -> str:
    c = sorted(hits.get(key, ()), key=lambda s: (int(re.match(r"\d+", s).group()), s))
    if not c:
        return "—"
    if len(c) > limit:
        return f"{len(c)} ({', '.join(c[:4])} ...)"
    return ", ".join(c)


# ---------------------------------------------------------------- rendering
def _table(cls: type, path: str, hits, unused: List[str], seen: set,
           undocumented: List[str]) -> List[str]:
    """One class as a table; nested classes follow as their own tables."""
    if cls in seen:
        return []
    seen.add(cls)
    docs = field_docs(cls)
    lines = [f"### `{path}` — {cls.__name__}", ""]
    if cls.__doc__:
        lines += [inspect.cleandoc(cls.__doc__).replace("\n", " "), ""]
    lines += ["| field | type | default | description | cases |", "|---|---|---|---|---|"]
    nested: List[Tuple[type, str]] = []
    for name, f in cls.model_fields.items():
        key = f.alias or name
        fpath = f"{path}.{name}"
        used = _cases(hits, fpath)
        if used == "—":
            unused.append(fpath)
        doc = docs.get(name, "").replace("|", "\\|").replace("``", "`")
        if not doc:
            undocumented.append(fpath)
        lines.append(f"| `{key}` | {type_name(f.annotation)} | {default_text(f)} | {doc} | {used} |")
        inner, how = _unwrap(f.annotation)
        if inner is not None:
            nested.append((inner, fpath + ("[]" if how == "[]" else ".*" if how == "*" else "")))
        elif fpath in EXTRA:
            nested.append((EXTRA[fpath], fpath))
    lines.append("")
    for inner, p in nested:
        lines += _table(inner, p, hits, unused, seen, undocumented)
    return lines


def generate(root: pathlib.Path = ROOT) -> str:
    """docs/yaml_reference.md."""
    hits, cases = usage(root / "cases")
    lib = library()
    out: List[str] = []
    out += [f"# vfife YAML reference (generated, {__version__})", ""]
    out += ["> Generated by `python -m vfife reference` from `vfife/io/model.py`, `vfife/io/schema.py` and the "
            "material and element catalogue; `tests/test_reference.py` keeps it in step with the code. "
            "**Do not edit** -- edit the field's `#:` comment and generate again.",
            "> Concepts, modelling methods and complete examples are in `docs/user_manual.md`. The last column "
            f"of each table lists the shipped cases that use the field (`cases/case_NN_*.yaml`, {len(cases)} of "
            "them, each with a contract under `verification/`).", ""]

    out += ["## 1 Top-level blocks of a case file", "",
            "| block | meaning | cases |", "|---|---|---|"]
    for block, cls, meaning in BLOCKS:
        key = block.rstrip("*[]").rstrip(".")
        out.append(f"| `{key}` | {meaning} | {_cases(hits, key)} |")
    used_internal = [f"`{k}` {_cases(hits, 'internal:' + k)}" for k in INTERNAL
                     if _cases(hits, 'internal:' + k) != "—"]
    out += ["", "Internal-form blocks (what `materials/sections/assign` lower to; old cases may write them "
            "directly): " + ", ".join(f"`{k}`" for k in INTERNAL) + ". " +
            ("Cases that use them: " + "; ".join(used_internal) + "." if used_internal
             else "The shipped cases all use the input format."), ""]

    out += ["## 2 The fields of each block", ""]
    unused: List[str] = []
    undocumented: List[str] = []
    seen: set = set()
    for block, cls, _ in BLOCKS:
        if cls is None:
            continue
        out += _table(cls, block, hits, unused, seen, undocumented)

    out += ["## 3 Materials", "",
            "`materials.<name>: {type: <material>, <parameters>...}`. Where a parameter has no default in the table, "
            "see the document of the same name in `docs/` or a case. "
            "Defaults that carry a unit (`rho`, `fpc` of `concrete01/02`, `E`/`Fy` of `steel02`, `E` and the grade's "
            "`fy`/`fu` of `steel_cyclic`, `tau_max`/`s1..s3` of `bond_ceb` and `bond_cyclic`) are given in SI and "
            "converted to the declared `units` when those are not SI. "
            "An alias of a parameter (`fc`=`fpc`, `eps_c0`=`epsc0`, `eps_cu`=`epscu` of `concrete01/02`; "
            "`fpcu`/`fcu2`=`fcu` of `concrete02`; `fcu`=`fpcu` of `concrete01`; `E0`=`E`, "
            "`yield_strength`=`fy`=`Fy` of `steel02`; `fy`=`yield_strength` of `rebar`, `j2`, `chaboche`) "
            "written beside its proper name with a different value is refused (`MATERIAL_ALIAS_CONFLICT`).", ""]
    for name, m in sorted(lib["materials"].items()):
        used = _cases(hits, f"material:{name}")
        if used == "—":
            unused.append(f"materials.{name}")
        out += [f"### `{name}`", "",
                f"* kinematics {', '.join(m['kinematics'])}; backends {', '.join(m['backends'])}; "
                f"analysis modes {', '.join(m['analysis_modes'])}; "
                f"{'nonlinear' if m['nonlinear'] else 'linear'}"
                f"{', regularised' if m.get('regularised') else ''}",
                "* parameters: " + ", ".join(f"`{p}`" for p in m["parameters"]),
                "* output fields: " + (", ".join(f"`{p}`" for p in m["output_fields"]) or "none"),
                ]
        if m.get("caveat"):
            out.append(f"* note: {m['caveat']}")
        out.append(f"* cases: {used}")
        params_used = sorted(k.split(".", 1)[1] for k in hits if k.startswith(f"material:{name}."))
        if params_used:
            out.append("* parameters the cases use: " + ", ".join(f"`{p}`" for p in params_used))
        out.append("")

    out += ["## 4 Elements and section types", "",
            "The element a section `type` makes: `solid` -> solids (tet4/tet10/hexa8/hexa20/wedge6/wedge15/pyramid5, "
            "from the mesh), `shell` -> shell3/shell4, `beam` -> beam2 (`theory: timoshenko`: beam2_timoshenko), "
            "`fiber` -> beam2_fiber, `bar` -> truss2, `connector` -> connector2.", "",
            "\"Features\" are the model-level features a model of the element takes (`FEATURES` in "
            "`vfife/elements/library.py`, the one table preflight, the schema and every analysis's refusals read): "
            "`static` Newton statics and arc length, `diaphragms` rigid diaphragms, `rigid_links` master-slave rigid "
            "links, `step_changes` analysis steps that build or remove members and rigid links, `buckling` linear "
            "buckling.", "",
            "| element | backends | analysis modes | features | materials | output fields |", "|---|---|---|---|---|---|"]
    for name, e in sorted(lib["elements"].items()):
        modes = list(e["analysis_modes"]) + (["static"] if "static" in e.get("features", ())
                                             and "static" not in e["analysis_modes"] else [])
        out.append(f"| `{name}` | {', '.join(e['backends'])} | {', '.join(modes)} | "
                   f"{', '.join(e.get('features', ())) or '—'} | "
                   f"{', '.join(e['compatible_materials'])} | {', '.join(e.get('output_fields', [])) or '—'} |")
    out.append("")

    out += ["## 5 Fields no shipped case uses", "",
            "These can be written and are validated, but have no example with a contract. Before using one, look "
            "for it in the tests (`grep -r <field> tests/`); a field long without an example is a candidate for "
            "removal.", ""]
    out += [f"* `{u}`" for u in unused]
    out += ["", "## 6 Fields without a description in the code", "",
            f"{len(undocumented)}. A field's description is its `#:` comment; add it and generate again.", ""]
    out += [f"* `{u}`" for u in undocumented]
    out += ["", "## 7 Deprecated spellings", "",
            "Still read, with a `DEPRECATED_FIELD` warning from preflight; `vfife migrate <case> --in-place` renames "
            "them in the text itself and adds `version`, leaving comments and layout alone, and checks that the "
            "model lowered before and after is the same.", "",
            "| block | old spelling | now |", "|---|---|---|"]
    out += [f"| `sections.*` | `{old}` | `{new}` |" for old, new in im.DEPRECATED_SECTION_KEYS.items()]
    out.append("")
    return "\n".join(out)


def generate_api() -> str:
    """docs/api_reference.md: every name in `vfife.api.__all__` with its signature
    and docstring. The docstrings are the reference; this
    only lays them out, so the page cannot say something the code does not."""
    import dataclasses
    from . import api

    def ann(a) -> str:
        # `from __future__ import annotations` keeps them as the source wrote them
        return a if isinstance(a, str) else getattr(a, "__name__", str(a))

    def sig(obj) -> str:
        try:
            sg = inspect.signature(obj)
        except (TypeError, ValueError):
            return "(...)"
        parts, star = [], False
        for p in sg.parameters.values():
            if p.kind is p.KEYWORD_ONLY and not star:
                parts.append("*"); star = True
            t = p.name if p.kind is not p.VAR_POSITIONAL else "*" + p.name
            if p.annotation is not p.empty:
                t += ": " + ann(p.annotation)
            if p.default is not p.empty:
                t += (" = " if p.annotation is not p.empty else "=") + repr(p.default)
            parts.append(t)
        ret = "" if sg.return_annotation is sg.empty else " -> " + ann(sg.return_annotation)
        return "(" + ", ".join(parts) + ")" + ret

    def doc(obj) -> List[str]:
        d = inspect.getdoc(obj) or ""
        if not d:
            return ["(no docstring)", ""]
        return ["```text", d, "```", ""]

    out = [f"# vfife Python API reference (generated, API version {api.API_VERSION}, {__version__})", "",
           "> Generated by `python -m vfife reference --api` from the signatures and docstrings of "
           "`vfife/api.py`; `tests/test_api_v1.py` keeps it in step with the code. **Do not edit** -- "
           "edit the docstring and generate again.", "",
           "The names in `vfife.api.__all__` are the API: within 1.x their signatures and meaning do not "
           "change, and parameters are only added with defaults; everything else in the package is "
           "internal and may change. Concepts and complete examples: `docs/user_manual.md` 12.6.", "",
           "## The module", ""] + doc(api)
    out += ["## Names", "", "| name | kind | summary |", "|---|---|---|"]
    for name in api.__all__:
        obj = getattr(api, name)
        kind = "class" if inspect.isclass(obj) else "function" if callable(obj) else "constant"
        para = " ".join((inspect.getdoc(obj) or "").split("\n\n")[0].split())
        first = para.split(". ")[0].rstrip(".") + "." if callable(obj) else f"`{obj!r}`"
        out.append(f"| `{name}` | {kind} | {first.replace('|', '/')} |")
    out.append("")
    for name in api.__all__:
        obj = getattr(api, name)
        if not callable(obj):
            continue
        if inspect.isclass(obj):
            out += [f"## `{name}`", ""]
            if dataclasses.is_dataclass(obj):
                out += ["| field | type | default |", "|---|---|---|"]
                for f in dataclasses.fields(obj):
                    dflt = "—" if f.default is dataclasses.MISSING else f"`{f.default!r}`"
                    typ = str(f.type).replace("|", "\\|")
                    out.append(f"| `{f.name}` | `{typ}` | {dflt} |")
                out.append("")
            else:
                out += [f"`{name}{sig(obj)}`", ""]
            out += doc(obj)
            for mname, member in sorted(vars(obj).items()):
                if mname.startswith("_"):
                    continue
                fn = member.fget if isinstance(member, property) else member
                if isinstance(fn, (staticmethod, classmethod)):
                    fn = fn.__func__
                if not callable(fn):
                    continue
                label = f"{name}.{mname}" + ("" if isinstance(member, property) else sig(fn))
                prop = " (property)" if isinstance(member, property) else ""
                out += [f"### `{label}`" + prop, ""] + doc(fn)
        else:
            out += [f"## `{name}{sig(obj)}`", ""] + doc(obj)
    return "\n".join(out).rstrip("\n") + "\n"


_CODE_PATTERNS = [
    # (regex, kind): a refusal raised as UserError, a preflight issue (its
    # severity may be an expression), a `vfife check` build failure
    (re.compile(r'UserError\(\s*"([A-Z][A-Z0-9_]+)",\s*(?:f?"((?:[^"\\]|\\.)*)")?', re.S), "refusal"),
    (re.compile(r'_issue\(\s*\w+,\s*([^,]*?),\s*"([A-Z][A-Z0-9_]+)",\s*(?:f?"((?:[^"\\]|\\.)*)")?',
                re.S), "preflight"),
    (re.compile(r'\bfail\(\s*"([A-Z][A-Z0-9_]+)",\s*(?:f?"((?:[^"\\]|\\.)*)")?', re.S), "check"),
]


def generate_codes(root: pathlib.Path = ROOT) -> str:
    """docs/error_codes.md: every code a refusal or a preflight issue carries,
    found in the source. The docs cited codes the terminal
    never printed and no page listed."""
    found: Dict[str, dict] = {}
    for path in sorted((root / "vfife").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        for rx, kind in _CODE_PATTERNS:
            for m in rx.finditer(text):
                if kind == "refusal":
                    code, msg, levels = m.group(1), m.group(2) or "", {"refused"}
                elif kind == "check":
                    code, msg, levels = m.group(1), m.group(2) or "", {"ERROR"}
                else:
                    expr, code, msg = m.group(1), m.group(2), m.group(3) or ""
                    levels = {lv for lv in ("ERROR", "WARNING", "INFO") if lv in expr} or {"ERROR"}
                entry = found.setdefault(code, {"levels": set(), "where": set(), "message": ""})
                entry["levels"] |= levels
                entry["where"].add(rel)
                if not entry["message"] and msg:
                    msg = re.sub(r"\{[^{}]*\}", "…", msg).replace("|", "/").strip()
                    entry["message"] = (msg[:110] + "…") if len(msg) > 110 else msg
    out = ["# vfife error codes (generated, " + __import__("vfife").__version__ + ")", "",
           "> Generated by `python -m vfife reference --codes` from the source; `tests/test_review6_docs.py` "
           "keeps it in step with the code. **Do not edit.**", "",
           "Every code a refusal (`UserError`, raised by the Python API and printed by the command line as "
           "`error [CODE]: …`) or a preflight issue (`vfife check`, the start of every run; ERROR stops, "
           "WARNING and INFO do not) carries. A code is a stable name a script or a front end can match on; "
           "the message beside it is the start of the text, with `…` where a value is filled in.", "",
           "Exit codes of the command line: 0 success; 1 an input mistake (a refusal, a file that cannot be "
           "read, invalid YAML or schema); 2 `check`/`info`/`run` preflight FAIL and `verify` FAIL; "
           "3 an analysis that did not converge (`run`, `formfind`); 130 stopped by Ctrl-C; 143 by SIGTERM; "
           "141 the output pipe closed.", "",
           "Analysis statuses (`analysis_status` in the result, `AnalysisOutcome.status`; see docs/diagnostics.md): "
           "`CONVERGED`, `COMPLETED_DYNAMIC` (answers); `MAX_STEPS`, `NONFINITE`, `INTERRUPTED` (kept, not "
           "answers); `PREFLIGHT_FAILED` (an outcome of `api.run_case` only, nothing written).", "",
           "| code | kind | first words of the message | source |", "|---|---|---|---|"]
    for code in sorted(found):
        e = found[code]
        kind = "/".join(sorted(e["levels"], key=lambda s: ("refused", "ERROR", "WARNING", "INFO").index(s)))
        out.append(f"| `{code}` | {kind} | {e['message'] or '—'} | {', '.join(sorted(e['where']))} |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- JSON Schema
SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_TITLE = "vfife case file, format 1"
SCHEMA_FILE = "vfife.schema.json"


def _models_in(tp) -> List[type]:
    """Every Pydantic class inside an annotation (Optional, Union, List, Dict)."""
    if inspect.isclass(tp) and issubclass(tp, BaseModel):
        return [tp]
    out: List[type] = []
    for a in typing.get_args(tp):
        out += _models_in(a)
    return out


def _all_docs(cls: type) -> Dict[str, str]:
    """`field_docs` over the class and its bases (a field documented where
    it is declared)."""
    docs: Dict[str, str] = {}
    for k in reversed(cls.__mro__):
        if inspect.isclass(k) and issubclass(k, BaseModel) and k is not BaseModel:
            try:
                docs.update(field_docs(k))
            except (OSError, TypeError):            # made by create_model: no source
                pass
    return docs


def _describe(doc: str) -> str:
    """A field's description in the editor's schema."""
    return doc.replace("``", "`")


def generate_schema() -> dict:
    """The JSON Schema of a case file as written, for an editor:
    every field's `#:` comment as its `description` (the
    same text as docs/yaml_reference.md), Literal fields as `enum`, a material's
    `type` from the catalogue, and the dict-typed sub-blocks (`fracture`,
    `bond`, `friction`, `failure`) with their own fields."""
    from . import api
    from .io.model import InputCase
    model = api._input_case_model()
    s = model.model_json_schema()
    defs = s.setdefault("$defs", {})

    # the sub-blocks the input model types as free dicts
    sub = {}
    for path, cls in EXTRA.items():
        one = cls.model_json_schema(ref_template="#/$defs/{model}")
        defs.update(one.pop("$defs", {}))
        defs.setdefault(cls.__name__, one)
        sub[path] = cls.__name__

    def ref_free_dict(prop: dict, name: str):
        branches = prop.get("anyOf") or [prop]
        for b in branches:
            if b.get("type") == "object" and "properties" not in b:
                b.clear(); b["$ref"] = f"#/$defs/{name}"
                return

    for path, name in sub.items():
        parent, field = path.rsplit(".", 1)
        owner = {"sections.*": "SectionSpec", "assign[]": "AssignSpec"}[parent]
        ref_free_dict(defs[owner]["properties"][field], name)

    # descriptions, from the classes the definitions were made from
    classes: Dict[str, type] = {}

    def visit(cls):
        if cls.__name__ in classes:
            return
        classes[cls.__name__] = cls
        for f in cls.model_fields.values():
            for inner in _models_in(f.annotation):
                visit(inner)
    for cls in [model, *EXTRA.values()]:
        visit(cls)

    def annotate(node: dict, cls: type, extra_docs: Dict[str, str] = {}):
        docs = _all_docs(cls)
        for name, f in cls.model_fields.items():
            prop = (node.get("properties") or {}).get(f.alias or name)
            if prop is None:
                continue
            text = _describe(docs[name]) if name in docs else extra_docs.get(name, "")
            if text:
                prop["description"] = text

    blocks = {b.rstrip("*[]").rstrip("."): meaning for b, _, meaning in BLOCKS}
    from .io import schema as _sc
    root_docs = {**{k: _describe(v) for k, v in _all_docs(_sc.Case).items()},
                 **{k: _describe(v) for k, v in _all_docs(InputCase).items()}}
    annotate(s, model, {k: blocks.get(k) or root_docs.get(k, "") for k in model.model_fields})
    for key, node in defs.items():
        cls = classes.get(key) or classes.get(node.get("title", ""))
        if cls is not None:
            annotate(node, cls)

    # a material is {type: <a catalogue name>, <its parameters>...}
    lib = library()["materials"]
    names = sorted(lib)
    mat = s["properties"]["materials"]
    mat["additionalProperties"] = {
        "type": "object", "required": ["type"], "additionalProperties": True,
        "properties": {"type": {
            "enum": names,
            "description": "material type; its parameters are listed in docs/yaml_reference.md part 3",
            # VS Code shows these beside each value
            "enumDescriptions": ["parameters: " + ", ".join(lib[n]["parameters"]) for n in names]}}}

    # a tendon may leave any key but its name to `tendon_defaults`, whose
    # mappings merge key by key; a profile may be a table
    if "TendonSpec" in defs:
        spec = defs["TendonSpec"]
        spec["required"] = ["name"]
        for key in ("TendonLine", "TendonRelaxation"):
            defs.get(key, {}).pop("required", None)
        table = {"type": "object", "required": ["table"], "additionalProperties": False,
                 "properties": {"table": {"type": "string", "description": "a CSV file (name, x, z|y, radius)"},
                                "name": {"type": "string", "description": "the rows of this name; none: of the tendon's own"}}}
        for key in ("profile_y", "profile_z"):
            prop = spec["properties"][key]
            rows = {k: v for k, v in prop.items() if k not in ("description", "title", "default")}
            spec["properties"][key] = {"description": prop.get("description", ""), "anyOf": [rows, table]}
        td = s["properties"].get("tendon_defaults")
        if td is not None:
            td.pop("additionalProperties", None)
            td["additionalProperties"] = False
            td["properties"] = {k: v for k, v in spec["properties"].items() if k != "name"}
    # a load or a nodal mass may be a CSV table of rows
    def table_entry(extra: Dict[str, str]) -> dict:
        props = {"table": {"type": "string", "description": "a CSV file, relative to the case (11.15)"}}
        props.update({k: {"description": v} for k, v in extra.items()})
        return {"type": "object", "required": ["table"], "additionalProperties": False, "properties": props}

    load_table = table_entry({"keep": "as a load's keep", "amplitude": "as a load's amplitude"})
    mass_table = table_entry({"weight": "as a nodal mass's weight", "components": "as a nodal mass's components"})
    for owner, key, table in ((s["properties"], "loads", load_table),
                              (defs.get("StepSpec", {}).get("properties", {}), "loads", load_table),
                              (s["properties"], "nodal_masses", mass_table)):
        prop = owner.get(key)
        if isinstance(prop, dict) and isinstance(prop.get("items"), dict) and "$ref" in prop["items"]:
            prop["items"] = {"anyOf": [prop["items"], table]}
    out = {"$schema": SCHEMA_DRAFT, "title": SCHEMA_TITLE}
    out.update({k: v for k, v in s.items() if k != "title"})
    return out


def schema_text() -> str:
    import json
    return json.dumps(generate_schema(), indent=2, ensure_ascii=False) + "\n"


def cmd_reference(args) -> int:
    if getattr(args, "codes", False):
        text = generate_codes()
        target = pathlib.Path(args.output) if args.output else ROOT / "docs" / "error_codes.md"
        return _write_or_check(text, target, args.check, "python -m vfife reference --codes")
    if getattr(args, "api", False):
        target = pathlib.Path(args.output) if args.output else ROOT / "docs" / "api_reference.md"
        return _write_or_check(generate_api(), target, args.check, "python -m vfife reference --api")
    target = pathlib.Path(args.output) if args.output else ROOT / "docs" / "yaml_reference.md"
    rc = _write_or_check(generate(), target, args.check, "python -m vfife reference")
    if args.output:
        return rc
    # and the editor's schema beside it
    return max(rc, _write_or_check(schema_text(), ROOT / "docs" / SCHEMA_FILE, args.check,
                                   "python -m vfife reference"))


def _write_or_check(text, target, check, command) -> int:
    if check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current != text:
            print(f"{target} is out of date: run `{command}`")
            return 1
        print(f"{target} is up to date")
        return 0
    target.write_text(text, encoding="utf-8")
    print(f"wrote {target}")
    return 0
