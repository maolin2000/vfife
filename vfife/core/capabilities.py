"""What runs where: one table, read by the code that refuses and by the
manuals (P1 of the architecture review of 1.17).

The review counted 50 places where the compiled step and the compiled
relaxation gave a reason for taking the step-by-step loop instead (40
different reasons), each a string literal at the place it was decided, and the manuals listing them again by hand -- three
places that drifted (the manual said "rigid-surface contact" where the code
said "the step is hooked (rigid contact)"). Every reason is a `Limit` here
now: the code asks `refuse(key, ...)` for its words, the manuals' tables are
`table(...)` (tools/capability_tables.py writes them, and
tests/test_capabilities.py fails while they are stale), and a reason the
code gives that is not in the table is a test failure too.

The paths, the columns of the table:

    frame   the compiled dynamic step of frame models (`beam2`, `beam2_timoshenko`,
            `mixed_line`: frames, bars and connectors), under `backend: numba`
    fibre   the compiled dynamic step of fibre-beam models (`beam2_fiber`), when
            the fused kernel runs
    relax   the compiled staged relaxation (1.9): frame models under
            `backend: numba`; fibre models' relaxation is not compiled

A `Limit` with no `feature` is an internal condition (a setting, a state
between two steps, an array the solver laid out differently): it is in the
table, so its words come from here, but not in the manuals' list.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

PATHS = ("frame", "fibre", "relax")
PATH_NAMES = {"en": {"frame": "frames, dynamic", "fibre": "fibre beams, dynamic",
                     "relax": "staged relaxation"},
              "zh": {"frame": "框架，动力", "fibre": "纤维梁，动力", "relax": "分级松弛"}}


@dataclass(frozen=True)
class Limit:
    key: str
    #: the reason as the solver reports it (`compiled_step_refusal`,
    #: `compiled_relax_refusal`); a format string where it names a value
    message: str
    #: the compiled paths that hand such a step to the loop
    paths: Tuple[str, ...]
    #: what a user sees it as, for the manuals' table; None: internal
    feature: Optional[str] = None
    feature_zh: Optional[str] = None
    #: the paths on which the feature exists at all (the table's "—"
    #: elsewhere): a fibre model has no plastic hinges, a frame no fibres
    applies: Tuple[str, ...] = PATHS


LIMITS = (
    # -- settings and the software
    Limit("off", "VFIFE_COMPILED_STEP is off", PATHS),
    Limit("no_numba", "numba is not installed", PATHS,
          "numba not installed", "未安装 numba"),
    Limit("backend", "backend is not numba", ("frame", "relax"),
          "`backend: numpy` (a fibre model takes its fused kernel all the same)",
          "`backend: numpy`（纤维梁模型照样用它的融合内核）"),
    Limit("short_fibre", "a short fibre run ({left} steps): compiling the step would take longer than it saves "
          "(VFIFE_COMPILED_STEP=1 asks for it)", ("fibre",),
          "a fibre run of fewer than 20000 steps (`VFIFE_COMPILED_STEP=1` forces the compiled step)",
          "不到 20000 步的纤维梁计算（`VFIFE_COMPILED_STEP=1` 强制走编译）",
          applies=("fibre",)),
    # -- the model
    Limit("own_forces", "{kind} has its own member forces", ("frame", "relax"),
          "shell, solid, coupled and shell-line models, a standalone `truss2` or `connector2` model "
          "(no compiled step at all)",
          "壳、实体、耦合和壳线模型，单独的 `truss2` 或 `connector2` 模型（根本没有编译的整步）",
          applies=("frame", "relax")),
    Limit("fibre_relax", "{kind} has its own member forces", ("relax",),
          "a fibre model's relaxation (`solver.threads` speeds its kernel up)",
          "纤维梁模型的松弛（`solver.threads` 让它的内核变快）",
          applies=("fibre", "relax")),
    Limit("other_members", "members other than frames, bars and connectors", ("frame", "relax"),
          "members other than frames, bars and connectors", "框架、杆、连接器以外的构件",
          applies=("frame", "relax")),
    Limit("six_dofs", "not six dofs per point", PATHS),
    Limit("bars_dofs", "bars without three translations", ("frame", "relax")),
    Limit("bars_creep", "bars that creep", ("frame", "relax"),
          "bars that creep", "会徐变的杆",
          applies=("frame", "relax")),
    Limit("connectors", "connectors without the compiled path", PATHS,
          "connectors whose law has no compiled path", "本构没有编译版本的连接器"),
    Limit("diaphragms", "rigid diaphragms", PATHS,
          "rigid diaphragms", "刚性楼板"),
    Limit("soil", "soil springs", PATHS, "soil springs", "土弹簧"),
    Limit("rigid_links", "rigid links", PATHS,
          "master-slave rigid links (`rigid_links`, 1.21)", "主从刚性连接（`rigid_links`，1.21）"),
    Limit("damping_regions", "Rayleigh damping by region", ("frame", "fibre"),
          "Rayleigh damping by region (`damping.regions`, 1.22)", "分区 Rayleigh 阻尼（`damping.regions`，1.22）",
          applies=("frame", "fibre")),
    Limit("end_offsets", "members with end offsets", PATHS,
          "members with end offsets (`end_offsets`, 1.21)", "带梁端偏移（`end_offsets`，1.21）的构件"),
    Limit("hinges", "plastic hinges", ("relax",), "plastic hinges", "塑性铰",
          applies=("frame", "relax")),
    Limit("fibre_section", "the fibre section runs on the NumPy reference ({status})", ("fibre",),
          "a fibre section the fused kernel does not take (`solver.kernel_status` says why)",
          "融合内核不接纳的纤维截面（`solver.kernel_status` 说明原因）",
          applies=("fibre",)),
    Limit("fibre_imposed", "an imposed fibre strain other than the prestrain", ("fibre",),
          "an imposed fibre strain other than the prestrain", "预应变以外的纤维给定应变",
          applies=("fibre",)),
    Limit("fibre_fracture", "fracture on a law the compiled step cannot read (steel02 bars only)", ("fibre",),
          "`fracture` on a law other than steel02 bars", "steel02 钢筋以外本构上的 `fracture`",
          applies=("fibre",)),
    # -- the loads and the supports
    Limit("rotation", "a prescribed rotation", PATHS, "a prescribed rotation", "给定转动"),
    Limit("motion_kind", "a prescribed motion of kind {motion!r}", ("frame", "fibre")),
    Limit("relax_motion", "a prescribed {motion}", ("relax",),
          "a prescribed velocity or acceleration", "给定速度或加速度"),
    Limit("relax_start", "the level's start does not match its prescriptions", ("relax",)),
    Limit("callable", "a load group that is a callable of its own", ("frame", "fibre"),
          "a script's own callable in a load group (a relaxation level takes the load it gives once)",
          "脚本自己放进荷载组的可调用对象（松弛的一级只取它给的荷载一次）"),
    Limit("thermal", "temperatures or time-dependent strains", PATHS,
          "temperatures, creep, shrinkage and the relaxation of steel", "温度、徐变、收缩和钢材松弛"),
    # (the relaxation says "temperatures or time-dependent strains" for it)
    Limit("imposed", "an imposed deformation", ("frame", "relax"),
          "an imposed deformation of the frame members", "框架构件的给定变形",
          applies=("frame", "relax")),
    # -- the run
    Limit("not_dynamic", "not a dynamic step", ("frame", "fibre")),
    Limit("steps", "analysis steps", PATHS,
          "analysis steps (`steps` with procedures, 1.2)", "分析步（带 procedure 的 `steps`，1.2）"),
    # (the frame paths refuse their analysis steps first)
    Limit("born", "members born in a step", PATHS,
          "members born in a step", "在某一步出生的构件"),
    Limit("split_waiting", "a split waiting to be applied", ("frame", "fibre")),
    Limit("trial", "a trial state", ("fibre",)),
    Limit("grown", "a grown model", ("relax",)),
    Limit("ledger", "no energy ledger", ("relax",)),
    Limit("csr", "the stiffness is not a CSR matrix", ("frame", "fibre")),
    Limit("arrays", "{name} is not a contiguous float array of shape {shape}", PATHS),
    # -- what a component or a script put on the solver
    Limit("hooked", "the step is hooked ({what})", PATHS,
          "a script's hook on the solver; rigid-surface contact (`contact.rigid`); in the relaxation any contact",
          "脚本挂在求解器上的钩子；刚性面接触（`contact.rigid`）；松弛时任何接触"),
    Limit("two_contacts", "more than one contact between members", ("frame", "fibre")),
    Limit("rigid_contact", "rigid contact", ("frame", "fibre")),
    Limit("component", "{name}", ("frame", "fibre")),
    Limit("contact_component_hooked", "the contact's component is hooked", ("frame", "fibre")),
    Limit("foreign_contact", "a contact between members that is not the solver's own", ("frame", "fibre")),
    Limit("contact_hooked", "the contact's {method} is hooked", ("frame", "fibre")),
)

BY_KEY = {lim.key: lim for lim in LIMITS}
assert len(BY_KEY) == len(LIMITS), "two limits share a key"
assert all(set(lim.paths) <= set(lim.applies) <= set(PATHS) for lim in LIMITS)


def refuse(key: str, **values) -> str:
    """The words of limit `key`, its values filled in."""
    return BY_KEY[key].message.format(**values)


def table(lang: str = "en") -> str:
    """The manuals' table: every feature, and on each compiled path whether
    it is taken there or keeps the step-by-step loop."""
    names = PATH_NAMES[lang]
    yes, no = ("compiled", "loop") if lang == "en" else ("编译", "逐步循环")
    head = ("what" if lang == "en" else "情形")
    rows = [f"| {head} | " + " | ".join(names[p] for p in PATHS) + " |",
            "|---|" + "---|" * len(PATHS)]
    for lim in LIMITS:
        feature = lim.feature if lang == "en" else lim.feature_zh
        if feature is None:
            continue
        cell = lambda p: no if p in lim.paths else (yes if p in lim.applies else "—")  # noqa: E731
        rows.append(f"| {feature} | " + " | ".join(cell(p) for p in PATHS) + " |")
    return "\n".join(rows)


#: the markers between which a manual holds the generated table
BEGIN = "<!-- capabilities:compiled (generated by tools/capability_tables.py; do not edit) -->"
END = "<!-- /capabilities:compiled -->"
