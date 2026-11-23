"""Old case form -> input form (materials / sections / reinforcement / assign).

`vfife migrate old.yaml` writes the same model in the input form and checks
its own work: the result is lowered again and compared field by field with
the original before anything is written. A migration that does not round-trip
is refused, with the first differing field named.

Names are invented where the old form had none -- `concrete`, `steel`,
`body`, `bar_1` -- and are the analyst's to rename. The output is the compact
style (io/yamlfmt.py), with a header saying where it came from.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List

from .schema import coerce_numbers


def _material_name(m: Dict[str, Any], taken: set) -> str:
    t = str(m.get("type", "material"))
    if t in ("cdp", "concrete01", "concrete02"):
        base = "concrete"
    elif t in ("rebar", "steel02", "steel_cyclic", "j2", "chaboche"):
        base = "steel"
    elif t == "linear_elastic":
        base = "steel" if float(m.get("E", 0) or 0) > 1.0e11 else "elastic"
    else:
        base = t
    name, k = base, 2
    while name in taken:
        name, k = f"{base}{k}", k + 1
    taken.add(name)
    return name


def _dedupe_material(mats: Dict[str, Dict[str, Any]], m: Dict[str, Any], taken: set) -> str:
    for name, existing in mats.items():
        if existing == m:
            return name
    name = _material_name(m, taken)
    mats[name] = m
    return name


def migrate(old: Dict[str, Any]) -> Dict[str, Any]:
    old = coerce_numbers(copy.deepcopy(old))
    element = dict(old.get("element") or {})
    material = old.get("material")
    if material is None:
        raise ValueError("the old case has no `material`")
    taken: set = set()
    materials: Dict[str, Dict[str, Any]] = {}
    sections: Dict[str, Dict[str, Any]] = {}
    reinforcement: Dict[str, Any] = {}
    assign: List[Dict[str, Any]] = []

    etype = element.get("type", "hexa8")
    main_mat = _dedupe_material(materials, material, taken)

    def line_section(entry: Dict[str, Any], name: str) -> Dict[str, Any]:
        t = entry.get("type", etype)
        if t == "truss2":
            sec = {"type": "bar", "material": main_mat, "shape": "general", "area": entry["area"]}
        elif t in ("beam2", "beam2_timoshenko"):
            sec = {"type": "beam", "material": main_mat, "shape": "general"}
            for k in ("area", "iy", "iz", "j", "ay", "az"):
                if entry.get(k) is not None:
                    sec[k] = entry[k]
            if t == "beam2_timoshenko":
                sec["theory"] = "timoshenko"
                for k in ("kappa_y", "kappa_z"):
                    if entry.get(k) is not None:
                        sec[k] = entry[k]
        elif t == "connector2":
            sec = {"type": "connector", "material": main_mat}
            for k in ("stiffness", "connector_damping", "curves"):
                if entry.get(k):
                    sec[k] = entry[k]
        else:
            raise ValueError(f"cannot migrate line element type {t!r}")
        if entry.get("formulation") not in (None, "full"):
            sec["formulation"] = entry["formulation"]
        return sec

    def line_assign(entry: Dict[str, Any], region: str, section: str) -> Dict[str, Any]:
        a = {"region": region, "section": section}
        for k in ("orientation", "releases", "coordinate_system"):
            if entry.get(k):
                a[k] = entry[k]
        return a

    if etype in ("mixed_line", "coupled"):
        if etype == "coupled":
            sections["body"] = {"type": "solid", "material": main_mat}
            if element.get("formulation") not in (None, "b-bar"):
                sections["body"]["formulation"] = element["formulation"]
            assign.append({"region": "all", "section": "body"})
        for i, mem in enumerate(old.get("members") or []):
            word = {"truss2": "bar", "beam2": "beam", "beam2_timoshenko": "beam",
                    "connector2": "connector"}.get(mem.get("type", ""), "member")
            name = word if word not in sections else f"{word}_{mem['region']}"
            sections[name] = line_section(mem, name)
            assign.append(line_assign(mem, mem["region"], name))
    elif etype in ("truss2", "beam2", "beam2_timoshenko", "connector2"):
        name = {"truss2": "bar", "beam2": "beam", "beam2_timoshenko": "beam", "connector2": "connector"}[etype]
        sections[name] = line_section(element, name)
        assign.append(line_assign(element, "all", name))
    elif etype == "beam2_fiber":
        fs = dict(element.get("fiber_section") or {})
        sec = {"type": "fiber", "material": main_mat, "shape": "rect",
               "width": fs.get("width"), "height": fs.get("height")}
        for k in ("ny", "nz", "cover"):
            if fs.get(k) is not None:
                sec[k] = fs[k]
        if fs.get("core_material") is not None:
            core = fs["core_material"]
            cname = "core" if "core" not in materials else _material_name(core, taken)
            materials[cname] = core; taken.add(cname)
            sec["core_material"] = cname
        if element.get("fiber_integration"):
            sec["integration"] = element["fiber_integration"]
        bars = []
        for rb in element.get("fiber_rebars") or []:
            mname = _dedupe_material(materials, rb["material"], taken)
            bname = None
            for sname, s in sections.items():
                if s.get("type") == "bar" and s.get("material") == mname and s.get("area") == rb["area"]:
                    bname = sname
            if bname is None:
                bname = f"bar_{len([s for s in sections.values() if s.get('type') == 'bar']) + 1}"
                sections[bname] = {"type": "bar", "material": mname, "shape": "general", "area": rb["area"]}
            bars.append({"y": rb["y"], "z": rb["z"], "section": bname})
        if bars:
            sec["bars"] = bars
        sections["rc"] = sec
        a = {"region": "all", "section": "rc"}
        if element.get("strain_penetration"):
            a["anchorage"] = element["strain_penetration"]
        if element.get("orientation"):
            a["orientation"] = element["orientation"]
        assign.append(a)
    elif etype in ("shell4", "shell3"):
        sec = {"type": "shell", "material": main_mat, "thickness": element["thickness"]}
        for k in ("shear_correction", "drilling_factor", "warpage_tolerance", "hourglass_coeff"):
            if element.get(k) is not None:
                sec[k] = element[k]
        if element.get("formulation") not in (None, "b-bar"):
            sec["formulation"] = element["formulation"]
        sections["shell"] = sec
        assign.append({"region": "all", "section": "shell"})
    else:
        sec = {"type": "solid", "material": main_mat}
        if element.get("formulation") not in (None, "b-bar"):
            sec["formulation"] = element["formulation"]
        if element.get("hourglass_coeff") not in (None, 0.1):
            sec["hourglass_coeff"] = element["hourglass_coeff"]
        sections["body"] = sec
        # a lone solid section needs no assignment line

    # embedded reinforcement
    for i, r in enumerate(old.get("reinforcement") or []):
        mname = _dedupe_material(materials, r["material"], taken)
        sec = {"type": "bar", "material": mname}
        if r.get("diameter") is not None:
            sec.update({"shape": "circle", "diameter": r["diameter"], "area": r["area"]})
        else:
            sec.update({"shape": "general", "area": r["area"]})
        bname = None
        for sname, s in sections.items():
            if s == sec:
                bname = sname
        if bname is None:
            bname = r.get("name") or f"bar_{i + 1}"
            if bname in sections:
                bname = f"bar_{i + 1}"
            sections[bname] = sec
        rname = r.get("name") or f"set_{i + 1}"
        if rname in reinforcement:
            rname = f"{rname}_{i + 1}"
        if r.get("lines"):
            reinforcement[rname] = {"lines": r["lines"]}
        else:
            raise ValueError(f"reinforcement {rname!r} uses nodes/elements, which the input form does not carry yet")
        a = {"reinforcement": rname, "section": bname, "name": r.get("name") or rname}
        if r.get("bond"):
            a["bond"] = r["bond"]
        if r.get("transverse_penalty") is not None:
            a["transverse_penalty"] = r["transverse_penalty"]
        if r.get("host_deduction"):
            a["host_deduction"] = r["host_deduction"]
        assign.append(a)

    from .schema import CASE_FORMAT
    new: Dict[str, Any] = {"version": CASE_FORMAT}
    for k in ("name", "units", "mesh"):
        if k in old:
            new[k] = old[k]
    new["materials"] = materials
    new["sections"] = sections
    if reinforcement:
        new["reinforcement"] = reinforcement
    if assign:
        new["assign"] = assign
    for k, v in old.items():
        if k not in ("name", "units", "mesh", "element", "material", "members", "reinforcement", "version"):
            new[k] = v
    return new


def check_round_trip(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    """Field-level differences between the old case and the lowered new one,
    ignoring what the input form legitimately adds (a bar's diameter)."""
    from .schema import Case
    from .model import lower

    a = Case.model_validate(coerce_numbers(copy.deepcopy(old))).model_dump(mode="python")
    b = Case.model_validate(lower(copy.deepcopy(new))).model_dump(mode="python")
    a.pop("version", None); b.pop("version", None)
    out: List[str] = []

    def walk(x, y, p):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k == "diameter" and x.get(k) is None:
                    continue
                walk(x.get(k), y.get(k), f"{p}.{k}")
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                out.append(f"{p}: {len(x)} vs {len(y)} entries")
                return
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{p}[{i}]")
        elif isinstance(x, float) and isinstance(y, (int, float)):
            if abs(x - y) > 1e-9 * max(1.0, abs(y)):
                out.append(f"{p}: {x} vs {y}")
        elif x != y:
            out.append(f"{p}: {x!r} vs {y!r}")

    walk(a, b, "")
    return out


# --------------------------------------------------------------------------- #
#  0.x input form -> case format 1
# --------------------------------------------------------------------------- #

def upgrade_text(text: str):
    """An input-form case, as TEXT, rewritten to case format 1: the old
    spellings of section keys renamed where they stand, and `version: 1`
    added. Everything else -- comments, flow or block style, key order,
    number formatting -- is the analyst's and is left byte for byte: the
    renames are made at the positions PyYAML's composer reports for the key
    scalars, not by dumping the case again.

    Returns (new_text, notes), notes one line per change."""
    import yaml as _yaml
    from .model import DEPRECATED_SECTION_KEYS
    from .schema import CASE_FORMAT

    root = _yaml.compose(text)
    if not isinstance(root, _yaml.MappingNode):
        raise ValueError("a case file is a mapping at the top level")
    edits = []                                   # (start, end, replacement)
    notes: List[str] = []
    version_node = None
    seen = set()
    for k, v in root.value:
        if k.value == "version":
            version_node = v
        if k.value == "sections" and isinstance(v, _yaml.MappingNode):
            for sk, sv in v.value:
                if not isinstance(sv, _yaml.MappingNode):
                    continue
                # an aliased section (`s2: *s1`) is the SAME node as its
                # anchor: renamed once, not once per name (twice gave
                # `widthidth` -- found in review)
                if id(sv) in seen:
                    continue
                seen.add(id(sv))
                present = {kk.value for kk, _ in sv.value}
                for kk, _vv in sv.value:
                    new = DEPRECATED_SECTION_KEYS.get(kk.value)
                    if new is None:
                        continue
                    if new in present:
                        raise ValueError(f"sections.{sk.value}: both `{kk.value}` and `{new}` are "
                                         f"given; keep one before migrating")
                    edits.append((kk.start_mark.index, kk.end_mark.index, new))
                    notes.append(f"sections.{sk.value}: `{kk.value}` -> `{new}`")
    out = text
    for start, end, rep in sorted(edits, reverse=True):
        out = out[:start] + rep + out[end:]
    if version_node is None:
        # at the first top-level key -- after any BOM, `%YAML` directive,
        # `---` and the leading comments, which a line-based insertion put it
        # in front of (found in review). The edits above all
        # lie after this point, so its index is still good.
        at = root.start_mark.index
        if root.flow_style:
            if out[at] != "{":
                raise ValueError("cannot place `version` in this flow mapping; add `version: 1` by hand")
            out = out[:at + 1] + f"version: {CASE_FORMAT}, " + out[at + 1:]
        elif root.start_mark.column == 0:
            out = out[:at] + f"version: {CASE_FORMAT}\n" + out[at:]
        else:
            raise ValueError("the top-level mapping is indented; add `version: 1` by hand")
        notes.append(f"`version: {CASE_FORMAT}` added")
    elif str(version_node.value) != str(CASE_FORMAT):
        try:
            newer = int(str(version_node.value)) > CASE_FORMAT
        except ValueError:
            newer = False
        if newer:
            raise ValueError(f"`version: {version_node.value}` is a newer case format than this vfife "
                             f"writes ({CASE_FORMAT}); it is not rewritten downwards")
        a, b = version_node.start_mark.index, version_node.end_mark.index
        lead = " " if a > 0 and out[a - 1] == ":" else ""      # an empty `version:`
        out = out[:a] + lead + str(CASE_FORMAT) + out[b:]
        notes.append(f"`version: {version_node.value or '(empty)'}` -> {CASE_FORMAT}")
    return out, notes


def same_model(text_a: str, text_b: str) -> List[str]:
    """Field-level differences between what two input-form case texts lower
    to (the `version` field aside): empty when a rewrite changed nothing the
    solver reads."""
    import yaml as _yaml
    from .schema import from_dict

    a = from_dict(_yaml.safe_load(text_a)).model_dump(mode="python")
    b = from_dict(_yaml.safe_load(text_b)).model_dump(mode="python")
    a.pop("version", None); b.pop("version", None)
    out: List[str] = []

    def walk(x, y, p):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                walk(x.get(k), y.get(k), f"{p}.{k}")
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{p}[{i}]")
        elif x != y:
            out.append(f"{p}: {x!r} vs {y!r}")

    walk(a, b, "")
    return out
