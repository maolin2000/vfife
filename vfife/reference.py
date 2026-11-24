"""The YAML reference, generated from the code.

`docs/YAML参考.md` answers two questions a hand-written manual keeps getting
wrong as the library moves: WHAT can a case file say, and WHERE is a working
example. Both are read from the source of truth rather than remembered:

  * the fields, their types, defaults and doc comments come from the Pydantic
    input model (`vfife/io/model.py`, `vfife/io/schema.py`) -- every field's
    `#:` comment is its documentation;
  * the materials and elements come from the catalogue (`vfife.catalog`);
  * the "used by" column comes from walking every shipped `cases/case_*.yaml`,
    so a field that no case exercises is listed by name at the end -- either
    it wants an example or it is dead.

`tests/test_reference.py` regenerates the file and fails when it differs
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
    ("version", None, "算例格式版本：写 `1`（1.0 版起）；不写与 `1` 按同样的规则读（旧写法读入并给 DEPRECATED_FIELD 警告），写更大的数则拒绝。`vfife migrate` 会加上"),
    ("name", None, "算例名，也是结果文件的默认标题"),
    ("units", None, "单位声明（`length`、`force`、`stress`、`time`，另收 `moment`、`mass`、`temperature`、`density`）：preflight 据此检查量级；写出的数不换算，带单位的默认值换算到声明的单位；写了 `stress` 而缺 `length` 或 `force` 时由另外两个推出（`{force: N, stress: MPa}` 即 mm），推不出则 UNITS_INCOMPLETE 拒绝；结果文件记为 `unit_*` 属性"),
    ("include", None, "并入其他文件里的 materials、sections、reinforcement、groups、tendons、tendon_defaults：`[路径, ...]`，或 `{path, override: true}` 让本文件的定义优先；重名拒绝"),
    ("groups.*", None, "构件组：名字 → `[网格的组或组名, ...]`；读入后是网格里由这几块组成的一个组，凡写区域名的地方都可以用；降级为 `mesh.groups`"),
    ("mesh", sc.MeshSpec, "网格：结构化盒子、Gmsh `.msh`、Abaqus `.inp`，或写在算例里的线单元网格（`lines`，1.1）"),
    ("materials.*", None, "材料库：名字 → `{type, 参数...}`；每种材料的参数见第二部分"),
    ("sections.*", im.SectionSpec, "截面：几何 + 材料，被 `assign` 指派到区域上"),
    ("reinforcement.*", im.ReinforcementGeometry, "钢筋/预应力束的几何：按图纸坐标画，与网格无关"),
    ("regions.*", sc.RegionSpec, "命名节点集：按几何、按 Gmsh 物理组、按编号，或集合运算"),
    ("assign[]", im.AssignSpec, "唯一让截面遇到几何的地方：区域 ← 截面（+ 钢筋、粘结、预应力、断裂）"),
    ("tendon_defaults", None, "各束钢束共用的设置：钢束没写的键从这里取，映射（`stress`、`friction` 等）逐键合并、钢束自己写的优先；剖面可写 `{table: 文件.csv}`，按钢束名取行"),
    ("offset_arms", OffsetArmSpec, "偏心构件的刚臂：`{material, factor, limit, regions}`，一条规则代替每根刚臂一个截面；读入时按所连偏心构件的刚度算出每根的截面"),
    ("tendons[]", TendonSpec, "框架模型的钢束，一束一条：线形、面积、张拉、摩擦、回缩、松弛；读入时展开成钢束节点、刚臂和张拉的杆"),
    ("constraints[]", sc.ConstraintSpec, "约束：固定自由度，或强制位移/速度/加速度时程"),
    ("loads[]", sc.LoadSpec, "荷载：力、力偶、线荷载、面力、压力、体力、重力、温度、移动荷载、车辆、地面运动"),
    ("nodal_masses[]", sc.NodalMassSpec, "附加节点质量：默认三向、有重量；可定向（附加质量，无重量）、可加转动惯量"),
    ("hydrodynamic_mass[]", sc.HydrodynamicMassSpec, "水下构件的 Morison 附加质量：横向、无重量"),
    ("soil_springs[]", sc.SoilSpringSpec, "桩的土弹簧：按深度给的 p–y 曲线，非线性弹性（1.2）"),
    ("diaphragms[]", sc.DiaphragmSpec, "刚性隔板：区域内节点在某个平面内作刚体运动"),
    ("amplitudes.*", None, "幅值表：名字 → `[[进度或时间, 系数], ...]`，或 `hold`；荷载和约束用名字引用"),
    ("steps[]", sc.StepSpec, "多分析步：每步自己的约束、荷载与迭代预算，按顺序执行"),
    ("removals[]", sc.RemovalSpec, "定时移除：某个区域的构件在给定时刻退出（断索情景）"),
    ("contact", sc.ContactSpec, "刚性面接触：地面、海底、承台——节点碰到即受罚力，动力下可带库仑摩擦"),
    # it was missing from the reference altogether
    ("response_spectrum", sc.ResponseSpectrumSpec,
     "反应谱：在计算结束时的平衡状态上做模态、按谱求各阶峰值并组合（CQC/SRSS）"),
    ("solver", sc.SolverSpec, "求解：动力松弛（静力）或显式动力，阻尼、步长、后端、质量缩放、起始状态"),
    ("output", sc.OutputSpec, "输出：结果文件、写出频率、场、检查点、监测量"),
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
_CJK = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")


def _is_chinese(line: str) -> bool:
    """A Chinese summary line, as opposed to an English line that merely
    cites a 阶段 note."""
    return len(_CJK.findall(line.replace("阶段", ""))) >= 2


def field_docs(cls: type) -> Dict[str, Tuple[str, str]]:
    """`#:` comments above each field of a Pydantic class: (Chinese, English).

    The convention: the FIRST `#:` line of a field, when it begins with a CJK
    character, is the one-line Chinese summary; the lines after it are the
    English original. A field with no Chinese line gets ("", english)."""
    out: Dict[str, Tuple[str, str]] = {}
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
                zh = pending[0] if _is_chinese(pending[0]) else ""
                en = pending[1:] if zh else pending
                out[m.group(1)] = (zh, " ".join(en))
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
            return f"见 {tp.__name__}"
        return {"str": "str", "float": "float", "int": "int", "bool": "bool"}.get(tp.__name__, tp.__name__)
    return str(tp).replace("typing.", "")


def default_text(field) -> str:
    d = field.default
    if d is PydanticUndefined:
        if field.default_factory is not None:
            return "—"
        return "**必填**"
    if isinstance(d, BaseModel):
        return "（默认块）"
    if d is None:
        return "—"
    if isinstance(d, (list, dict)) and not d:
        return "空"
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
        return f"{len(c)} 个（{', '.join(c[:4])} …）"
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
    lines += ["| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |", "|---|---|---|---|---|---|"]
    nested: List[Tuple[type, str]] = []
    for name, f in cls.model_fields.items():
        key = f.alias or name
        fpath = f"{path}.{name}"
        used = _cases(hits, fpath)
        if used == "—":
            unused.append(fpath)
        zh, en = docs.get(name, ("", ""))
        zh = zh.replace("|", "\\|"); en = en.replace("|", "\\|").replace("``", "`")
        if not zh:
            undocumented.append(fpath)
        lines.append(f"| `{key}` | {type_name(f.annotation)} | {default_text(f)} | {zh} | {en} | {used} |")
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
    hits, cases = usage(root / "cases")
    lib = library()
    out: List[str] = []
    out += [f"# vfife YAML 参考（自动生成，{__version__}）", ""]
    out += ["> 由 `python -m vfife reference` 从 `vfife/io/model.py`、`vfife/io/schema.py` 和材料/单元目录生成；"
            "`tests/test_reference.py` 保证它与代码一致。**不要手改**——改字段的 `#:` 注释，再重新生成。",
            "> 说明列是字段上方 `#:` 注释的中文第一行，英文原注是其余行。概念、建模方法和完整例子见 `docs/用户手册.md`；每个字段最后一列是",
            f"> 用到它的自带算例编号（`cases/case_NN_*.yaml`，共 {len(cases)} 个，每个都有 `verification/` 下的契约）。", ""]

    out += ["## 一 算例文件的顶层块", "",
            "| 块 | 含义 | 用到它的算例 |", "|---|---|---|"]
    for block, cls, zh in BLOCKS:
        key = block.rstrip("*[]").rstrip(".")
        out.append(f"| `{key}` | {zh} | {_cases(hits, key)} |")
    out += ["", "内部格式的块（由 `materials/sections/assign` 降解生成，旧算例可直接写）：" +
            "、".join(f"`{k}`" for k in INTERNAL) + "。" +
            ("用到的算例：" + "; ".join(f"`{k}` {_cases(hits, 'internal:' + k)}" for k in INTERNAL
                                  if _cases(hits, 'internal:' + k) != "—")
             if any(_cases(hits, 'internal:' + k) != "—" for k in INTERNAL)
             else "自带算例全部用输入格式。"), ""]

    out += ["## 二 每个块的字段", ""]
    unused: List[str] = []
    undocumented: List[str] = []
    seen: set = set()
    for block, cls, _ in BLOCKS:
        if cls is None:
            continue
        out += _table(cls, block, hits, unused, seen, undocumented)

    out += ["## 三 材料", "",
            "`materials.<名>: {type: <材料名>, <参数>...}`。参数没有默认值表的，看 `docs/` 里同名文档或算例。"
            "带单位的默认值（`rho`、`concrete01/02` 的 `fpc`、`steel02` 的 `E`/`Fy`、`steel_cyclic` 的 `E` 与牌号的 `fy`/`fu`、`bond_ceb` 与 `bond_cyclic` 的 `tau_max`/`s1..s3`）按 SI 给出，声明了非 SI 的 `units` 时换算到声明的单位。"
            "同一参数的别名（`concrete01/02` 的 `fc`=`fpc`、`eps_c0`=`epsc0`、`eps_cu`=`epscu`，`concrete02` 的 `fpcu`/`fcu2`=`fcu`、`concrete01` 的 `fcu`=`fpcu`，`steel02` 的 `E0`=`E`、`yield_strength`=`fy`=`Fy`，`rebar`、`j2`、`chaboche` 的 `fy`=`yield_strength`）与正名同时给出且不等时拒绝（`MATERIAL_ALIAS_CONFLICT`）。", ""]
    for name, m in sorted(lib["materials"].items()):
        used = _cases(hits, f"material:{name}")
        if used == "—":
            unused.append(f"materials.{name}")
        out += [f"### `{name}`", "",
                f"* 运动学 {', '.join(m['kinematics'])}；后端 {', '.join(m['backends'])}；"
                f"分析模式 {', '.join(m['analysis_modes'])}；"
                f"{'非线性' if m['nonlinear'] else '线性'}"
                f"{'，正则化' if m.get('regularised') else ''}",
                "* 参数：" + ", ".join(f"`{p}`" for p in m["parameters"]),
                "* 输出字段：" + (", ".join(f"`{p}`" for p in m["output_fields"]) or "无"),
                ]
        if m.get("caveat"):
            out.append(f"* 注意：{m['caveat']}")
        out.append(f"* 算例：{used}")
        params_used = sorted(k.split(".", 1)[1] for k in hits if k.startswith(f"material:{name}."))
        if params_used:
            out.append("* 算例里出现过的参数：" + ", ".join(f"`{p}`" for p in params_used))
        out.append("")

    out += ["## 四 单元与截面类型", "",
            "截面 `type` 对应的单元：`solid` → 实体（由网格决定 tet4/tet10/hexa8/hexa20/wedge6/wedge15/pyramid5），"
            "`shell` → shell3/shell4，`beam` → beam2（`theory: timoshenko` 为 beam2_timoshenko），"
            "`fiber` → beam2_fiber，`bar` → truss2，`connector` → connector2。", "",
            "| 单元 | 后端 | 分析模式 | 可用材料 | 输出字段 |", "|---|---|---|---|---|"]
    from .cli import _STATIC_ELEMENTS
    for name, e in sorted(lib["elements"].items()):
        # `mode: static` runs these, as `describe` says
        modes = list(e["analysis_modes"]) + (["static"] if name in _STATIC_ELEMENTS
                                             and "static" not in e["analysis_modes"] else [])
        out.append(f"| `{name}` | {', '.join(e['backends'])} | {', '.join(modes)} | "
                   f"{', '.join(e['compatible_materials'])} | {', '.join(e.get('output_fields', [])) or '—'} |")
    out.append("")

    out += ["## 五 没有任何自带算例用到的字段", "",
            "这些字段能写、会被校验，但没有带契约的例子。用之前先在测试里找它（`grep -r 字段名 tests/`）；"
            "长期没有例子的字段是删除候选。", ""]
    out += [f"* `{u}`" for u in unused]
    out += ["", "## 六 代码里没有中文说明的字段", "",
            f"{len(undocumented)} 个。字段上方第一行 `#:` 注释以中文开头即为中文说明，其后是英文原注；补上再生成即可。", ""]
    out += [f"* `{u}`" for u in undocumented]
    out += ["", "## 七 弃用的写法", "",
            "仍然读，preflight 给一条 `DEPRECATED_FIELD` 警告；`vfife migrate <算例> --in-place` 在原文里就地改名、"
            "加上 `version`，注释和格式不动，并核对改写前后降解出的模型相同。", "",
            "| 块 | 旧写法 | 现在的写法 |", "|---|---|---|"]
    out += [f"| `sections.*` | `{old}` | `{new}` |" for old, new in im.DEPRECATED_SECTION_KEYS.items()]
    out.append("")
    return "\n".join(out)


def generate_api() -> str:
    """docs/API参考.md: every name in `vfife.api.__all__` with its signature
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
            return ["（无说明）", ""]
        return ["```text", d, "```", ""]

    out = [f"# vfife Python API 参考（自动生成，API 版本 {api.API_VERSION}，{__version__}）", "",
           "> 由 `python -m vfife reference --api` 从 `vfife/api.py` 的签名和文档字符串生成；"
           "`tests/test_api_v1.py` 保证它与代码一致。**不要手改**——改文档字符串，再重新生成。", "",
           "`vfife.api.__all__` 里的名字就是 API：1.x 之内签名和含义不变，只会加带默认值的参数；"
           "包里其余的一切都是内部实现，可能改变。概念和完整例子见 `docs/用户手册.md` 12.6。", "",
           "## 模块说明", ""] + doc(api)
    out += ["## 名字一览", "", "| 名字 | 种类 | 一句话 |", "|---|---|---|"]
    for name in api.__all__:
        obj = getattr(api, name)
        kind = ("类" if inspect.isclass(obj) else "函数" if callable(obj) else "常量")
        para = " ".join((inspect.getdoc(obj) or "").split("\n\n")[0].split())
        first = para.split(". ")[0].rstrip(".") + "." if kind != "常量" else f"`{obj!r}`"
        out.append(f"| `{name}` | {kind} | {first.replace('|', '/')} |")
    out.append("")
    for name in api.__all__:
        obj = getattr(api, name)
        if not callable(obj):
            continue
        if inspect.isclass(obj):
            out += [f"## `{name}`", ""]
            if dataclasses.is_dataclass(obj):
                out += ["| 字段 | 类型 | 默认 |", "|---|---|---|"]
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
                out += [f"### `{label}`" + ("（属性）" if isinstance(member, property) else ""), ""] + doc(fn)
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
    out = ["# vfife 错误代码 / error codes（自动生成，" + __import__("vfife").__version__ + "）", "",
           "> 由 `python -m vfife reference --codes` 从源码生成；`tests/test_review6_docs.py` 保证它与代码一致。"
           "**不要手改。**", "",
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


def _all_docs(cls: type) -> Dict[str, Tuple[str, str]]:
    """`field_docs` over the class and its bases (a field documented where
    it is declared)."""
    docs: Dict[str, Tuple[str, str]] = {}
    for k in reversed(cls.__mro__):
        if inspect.isclass(k) and issubclass(k, BaseModel) and k is not BaseModel:
            try:
                docs.update(field_docs(k))
            except (OSError, TypeError):            # made by create_model: no source
                pass
    return docs


def _describe(zh: str, en: str) -> str:
    en = en.replace("``", "`")
    return "\n\n".join(s for s in (zh, en) if s)


def generate_schema() -> dict:
    """The JSON Schema of a case file as written, for an editor:
    every field's `#:` comment as its `description` (the
    same text as docs/YAML参考.md), Literal fields as `enum`, a material's
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
            text = _describe(*docs[name]) if name in docs else extra_docs.get(name, "")
            if text:
                prop["description"] = text

    blocks = {b.rstrip("*[]").rstrip("."): zh for b, _, zh in BLOCKS}
    from .io import schema as _sc
    root_docs = {**{k: _describe(*v) for k, v in _all_docs(_sc.Case).items()},
                 **{k: _describe(*v) for k, v in _all_docs(InputCase).items()}}
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
            "description": "材料类型；参数见 docs/YAML参考.md 第三部分\n\nmaterial type; its "
                           "parameters are listed in docs/YAML参考.md part 3",
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
                 "properties": {"table": {"type": "string", "description": "CSV 文件（name, x, z|y, radius）\n\na CSV file (name, x, z|y, radius)"},
                                "name": {"type": "string", "description": "取这个名字的行；不写则取钢束自己名字的行\n\nthe rows of this name; none: of the tendon's own"}}}
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
        props = {"table": {"type": "string", "description": "CSV 文件，路径相对算例（11.15 节）\n\na CSV file, "
                                                               "relative to the case (11.15)"}}
        props.update({k: {"description": v} for k, v in extra.items()})
        return {"type": "object", "required": ["table"], "additionalProperties": False, "properties": props}

    load_table = table_entry({"keep": "与荷载的 keep 相同\n\nas a load's keep",
                              "amplitude": "与荷载的 amplitude 相同\n\nas a load's amplitude"})
    mass_table = table_entry({"weight": "与节点质量的 weight 相同\n\nas a nodal mass's weight",
                              "components": "与节点质量的 components 相同\n\nas a nodal mass's components"})
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
        text = generate_api()
        target = pathlib.Path(args.output) if args.output else ROOT / "docs" / "API参考.md"
        return _write_or_check(text, target, args.check, "python -m vfife reference --api")
    text = generate()
    target = pathlib.Path(args.output) if args.output else ROOT / "docs" / "YAML参考.md"
    rc = _write_or_check(text, target, args.check, "python -m vfife reference")
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
