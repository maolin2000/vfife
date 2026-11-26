# vfife YAML 参考（自动生成，1.21.0）

> 由 `python -m vfife reference` 从 `vfife/io/model.py`、`vfife/io/schema.py` 和材料/单元目录生成；`tests/test_reference.py` 保证它与代码一致。**不要手改**——改字段的 `#:` 注释，再重新生成。
> 说明列是字段上方 `#:` 注释的中文第一行，英文原注是其余行。概念、建模方法和完整例子见 `docs/用户手册.md`；每个字段最后一列是
> 用到它的自带算例编号（`cases/case_NN_*.yaml`，共 61 个，每个都有 `verification/` 下的契约）。

## 一 算例文件的顶层块

| 块 | 含义 | 用到它的算例 |
|---|---|---|
| `version` | 算例格式版本：写 `1`（1.0 版起）；不写与 `1` 按同样的规则读（旧写法读入并给 DEPRECATED_FIELD 警告），写更大的数则拒绝。`vfife migrate` 会加上 | 61 个（01, 02, 03, 04 …） |
| `name` | 算例名，也是结果文件的默认标题 | 61 个（01, 02, 03, 04 …） |
| `units` | 单位声明（`length`、`force`、`stress`、`time`，另收 `moment`、`mass`、`temperature`、`density`）：preflight 据此检查量级；写出的数不换算，带单位的默认值换算到声明的单位；写了 `stress` 而缺 `length` 或 `force` 时由另外两个推出（`{force: N, stress: MPa}` 即 mm），推不出则 UNITS_INCOMPLETE 拒绝；结果文件记为 `unit_*` 属性 | 61 个（01, 02, 03, 04 …） |
| `include` | 并入其他文件里的 materials、sections、reinforcement、groups、tendons、tendon_defaults：`[路径, ...]`，或 `{path, override: true}` 让本文件的定义优先；重名拒绝 | — |
| `groups` | 构件组：名字 → `[网格的组或组名, ...]`；读入后是网格里由这几块组成的一个组，凡写区域名的地方都可以用；降级为 `mesh.groups` | — |
| `mesh` | 网格：结构化盒子、Gmsh `.msh`、Abaqus `.inp`，或写在算例里的线单元网格（`lines`，1.1） | 61 个（01, 02, 03, 04 …） |
| `materials` | 材料库：名字 → `{type, 参数...}`；每种材料的参数见第二部分 | 61 个（01, 02, 03, 04 …） |
| `sections` | 截面：几何 + 材料，被 `assign` 指派到区域上 | 61 个（01, 02, 03, 04 …） |
| `reinforcement` | 钢筋/预应力束的几何：按图纸坐标画，与网格无关 | 06, 12, 12b, 12c, 13, 14, 15 |
| `regions` | 命名节点集：按几何、按 Gmsh 物理组、按编号，或集合运算 | 31 个（06, 10, 11, 12 …） |
| `assign` | 唯一让截面遇到几何的地方：区域 ← 截面（+ 钢筋、粘结、预应力、断裂） | 58 个（01, 02, 03, 05 …） |
| `tendon_defaults` | 各束钢束共用的设置：钢束没写的键从这里取，映射（`stress`、`friction` 等）逐键合并、钢束自己写的优先；剖面可写 `{table: 文件.csv}`，按钢束名取行 | — |
| `offset_arms` | 偏心构件的刚臂：`{material, factor, limit, regions}`，一条规则代替每根刚臂一个截面；读入时按所连偏心构件的刚度算出每根的截面 | — |
| `tendons` | 框架模型的钢束，一束一条：线形、面积、张拉、摩擦、回缩、松弛；读入时展开成钢束节点、刚臂和张拉的杆 | — |
| `constraints` | 约束：固定自由度，或强制位移/速度/加速度时程 | 60 个（01, 02, 03, 04 …） |
| `loads` | 荷载：力、力偶、线荷载、面力、压力、体力、重力、温度、移动荷载、车辆、地面运动 | 46 个（01, 02, 03, 04 …） |
| `nodal_masses` | 附加节点质量：默认三向、有重量；可定向（附加质量，无重量）、可加转动惯量 | 05, 36, 37 |
| `hydrodynamic_mass` | 水下构件的 Morison 附加质量：横向、无重量 | 37 |
| `soil_springs` | 桩的土弹簧：按深度给的 p–y 曲线，非线性弹性（1.2） | — |
| `diaphragms` | 刚性隔板：区域内节点在某个平面内作刚体运动 | — |
| `amplitudes` | 幅值表：名字 → `[[进度或时间, 系数], ...]`，或 `hold`；荷载和约束用名字引用 | 9 个（06, 11, 19, 24c …） |
| `steps` | 多分析步：每步自己的约束、荷载与迭代预算，按顺序执行 | 34 个（06, 11, 12, 12b …） |
| `removals` | 定时移除：某个区域的构件在给定时刻退出（断索情景） | 33 |
| `contact` | 刚性面接触：地面、海底、承台——节点碰到即受罚力，动力下可带库仑摩擦 | 33 |
| `response_spectrum` | 反应谱：在计算结束时的平衡状态上做模态、按谱求各阶峰值并组合（CQC/SRSS） | 35, 37 |
| `solver` | 求解：动力松弛（静力）或显式动力，阻尼、步长、后端、质量缩放、起始状态 | 61 个（01, 02, 03, 04 …） |
| `output` | 输出：结果文件、写出频率、场、检查点、监测量 | 61 个（01, 02, 03, 04 …） |

内部格式的块（由 `materials/sections/assign` 降解生成，旧算例可直接写）：`element`、`material`、`blocks`、`members`。自带算例全部用输入格式。

## 二 每个块的字段

### `mesh` — MeshSpec

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `type` | `box` \| `gmsh` \| `abaqus` \| `lines` | `box` | 网格来源：`box` 结构化盒子（dimensions/divisions/origin）、`gmsh` 读 .msh、`abaqus` 读 .inp、`lines` 在算例里写线单元网格（points/lines，1.1） | `box` (dimensions/divisions/origin), `gmsh` (path to .msh), `abaqus` (path to .inp), or `lines`: a line-member mesh written in the case (points/lines, 1.1) | 55 个（01, 02, 03, 04 …） |
| `points` | dict[str, list[float]] | — | lines：命名的点 {名字: [x, y, z]}；每个名字是一个只含该节点的区域；两个点不能重合（支座两侧要错开一点，用连接器相连） | lines: named points {name: [x, y, z]}; each name is a region of that one node. No two at one place: a bearing's two sides are two points a little apart, joined by a connector | — |
| `lines` | dict[str, 见 LineGroupSpec] | — | lines：命名的构件组 {名字: LineGroupSpec}；名字是这些单元和它们节点的区域 | lines: named groups of line members {name: LineGroupSpec}; the name is a region of those members and of their nodes | — |
| `groups` | dict[str, list[str]] | — | lines：几条线合成的组 {名字: [线名, ...]}，与 Gmsh 物理组一样是**单元**区域（供 assign、removals）；`regions:` 里的 union 只是节点集。gmsh：几个物理组合成的组（顶层 `groups:` 降级到这里） | lines: groups of lines {name: [line names]}, an ELEMENT region like a Gmsh physical group of several pieces (for `assign` and `removals`); a `regions:` union is a set of nodes only, and a member whose two nodes are in it belongs to it whichever line it is on. gmsh: groups of the file's physical groups, the same kind of region (the top-level `groups:` lowers to this) | — |
| `dimensions` | list[float] | — | 盒子网格的三个边长 [Lx, Ly, Lz]（m） | box only | 06, 12, 12b, 12c, 13, 14, 19, 20 |
| `divisions` | list[int] | — | 盒子网格沿 x、y、z 的单元数 | box only: elements along x, y, z | 06, 12, 12b, 12c, 13, 14, 19, 20 |
| `origin` | list[float] | `[0.0, 0.0, 0.0]` | 盒子网格 x、y、z 最小的那个角点 | box only: corner with the smallest x, y, z | — |
| `path` | str | — | 网格文件路径，相对算例文件 | gmsh only -- path to a .msh, relative to the case file | 53 个（01, 02, 03, 04 …） |
| `scale` | float | `1.0` | 坐标统一乘以它：网格按 mm 画、分析按 m 算时写 0.001 | multiplies every coordinate. Meshing in mm and analysing in m is the normal case, and a silent factor of 1000 in the geometry is not the kind of mistake that announces itself. | — |

### `mesh.lines.*` — LineGroupSpec

One named line of members, straight or an arc, generated (1.1). Its name is a region: the members' (for `assign`) and their nodes' (for constraints, loads, monitors), as a Gmsh physical line group's is.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `through` | list[str \| list[float]] | **必填** | 依次经过的点：`points` 里的名字或坐标 [x, y, z]；每两个相邻点之间是一段 | the points the line passes through, in order: names from `points` or coordinates [x, y, z]; each pair of neighbours is one segment | — |
| `divisions` | int | — | 每一段分成的单元数（与 `size` 二选一；都不写为 1） | elements per segment (or `size`; neither: 1) | — |
| `size` | float | — | 单元的目标长度：每一段分成 ceil(段长 / size) 个单元 | the target element length: each segment is divided into ceil(length / size) | — |
| `arc` | 见 ArcSpec | — | 画成曲线（只用于两点之间）：抛物线或圆弧，见 ArcSpec | a curve instead of a straight line (two points only): see ArcSpec | — |

### `mesh.lines.*.arc` — ArcSpec

A line drawn as a curve between its two end points (1.1).

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `rise` | float | **必填** | 矢高：弦中点处曲线离弦的距离，沿 `up` 量（负值向下） | the rise: how far the curve is from the chord at its middle, along `up` (negative: below) | — |
| `shape` | `parabola` \| `circle` | `parabola` | `parabola`（按弦等分，桥梁拱轴的常用形状）或 `circle`（按圆心角等分） | `parabola` (nodes evenly along the chord, the usual arch axis) or `circle` (evenly by angle) | — |
| `up` | list[float] | `[0.0, 0.0, 1.0]` | 矢高的方向（与弦不能平行；默认整体 +z） | the direction of the rise (not along the chord; default global +z) | — |

### `sections.*` — SectionSpec

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `type` | `solid` \| `shell` \| `beam` \| `bar` \| `fiber` \| `connector` | **必填** | 截面种类，决定承载它的单元：solid 实体、shell 壳、beam 梁（beam2）、fiber 纤维梁（beam2_fiber）、bar 杆/索（truss2）、connector 连接器（connector2） | what the section is, and so which element carries it: solid (the meshed continuum), shell, beam (beam2), fiber (beam2_fiber), bar (truss2: bars, cables), connector (connector2) | 61 个（01, 02, 03, 04 …） |
| `material` | str | — | `materials` 里的名字；连接器可省；纤维截面必填（patch 不写材料时用它） | a name in `materials`; optional for a connector; required on a fiber section (the law of every patch that names none of its own) | 61 个（01, 02, 03, 04 …） |
| `formulation` | `full` \| `reduced` \| `b-bar` \| `selective` | — | 实体：hexa8 的体积锁死处理（默认 `b-bar`；`selective`、`reduced` 加沙漏控制、`full`） | solid: volumetric locking treatment for hexa8 (`b-bar` default; `selective`, `reduced` with hourglass control, `full`) | 11 个（07, 10, 12, 12b …） |
| `hourglass_coeff` | float | — | 实体、只在 `reduced` 下：沙漏稳定系数 | solid, `reduced` only: hourglass stabilisation coefficient | 10, 17 |
| `thickness` | float | — | 壳：板厚（m） | shell: plate thickness (m) | 10, 17 |
| `shear_correction` | float | — | 壳：横向剪切修正系数（默认 5/6） | shell: transverse shear correction factor (5/6 default) | 10, 17 |
| `drilling_factor` | float | — | 壳：面内转动（钻转）刚度系数 | shell: drilling-rotation stiffness factor | 10, 17 |
| `warpage_tolerance` | float | — | 壳：shell4 四个角偏离平面超过这个比例时拒绝 | shell: a shell4 whose corners leave a plane by more than this fraction is refused | — |
| `shape` | `circle` \| `rect` \| `I` \| `general` \| `tube` \| `box` \| `patches` \| `box_girder` \| `t_girder` | — | 形状：圆 `circle`、矩形 `rect`、工字 `I`、直接给性质 `general`；纤维截面另有圆管 `tube`、箱 `box`、四边形块 `patches` | `tube` (circular hollow) and `box` (rectangular/square hollow) are fiber sections only: a steel spatial structure is built of tubes, and a tube discretised as a solid puts fibres where there is no steel. | 55 个（01, 02, 03, 05 …） |
| `diameter` | float | — | 圆（梁、杆）与纤维圆管：外径；圆加 `t` 即圆管（旧写法 `d`，1.0 版起弃用） | circle (beam, bar) and fiber tube: outer diameter; a circle with `t` is a tube (the old spelling `d` is deprecated since v1.0) | 16 个（12, 12b, 12c, 13 …） |
| `t` | float | — | 空心形状的壁厚（circle + t、rect + t；纤维 tube/box） | wall thickness of a hollow shape (circle + t, rect + t; fiber tube/box) | — |
| `width` | float | — | 矩形（梁、纤维）与纤维箱：沿截面 y 轴的宽，箱为外宽（旧写法 `b`，1.0 版起弃用） | rect (beam, fiber) and fiber box: width along the section's y axis, outer for a box (the old spelling `b` is deprecated since v1.0) | 10 个（11, 21, 21b, 21c …） |
| `height` | float | — | 矩形（梁、纤维）与纤维箱：沿 z 轴的高，箱为外高（旧写法 `h`，1.0 版起弃用） | rect (beam, fiber) and fiber box: depth along z, outer for a box (the old spelling `h` is deprecated since v1.0) | 10 个（11, 21, 21b, 21c …） |
| `top_width` | float | — | 箱梁、T 梁：桥面全宽（翼缘尖到尖） | box_girder, t_girder: the whole deck, tip to tip | — |
| `bottom_width` | float | — | 箱梁：底板外宽；T 梁：下翼缘宽（给了就有下翼缘） | box_girder: the bottom slab, outside to outside; t_girder: the bottom flange's width (an I or a bulb-T) | — |
| `cantilever` | float | — | 箱梁：悬臂长，从翼缘尖到腹板外侧面，在桥面顶量 | box_girder: from a tip to the web's outer face, measured at the top of the deck | — |
| `top_thickness` | float | — | 箱梁：两腹板间顶板厚；T 梁：翼缘厚 | box_girder: the top slab between the webs; t_girder: the flange | — |
| `bottom_thickness` | float | — | 箱梁：底板厚；T 梁：下翼缘厚 | box_girder: the bottom slab; t_girder: the bottom flange | — |
| `web_thickness` | float | — | 箱梁、T 梁：腹板厚，斜腹板按水平方向量 | box_girder, t_girder: each web, measured HORIZONTALLY on an inclined web | — |
| `cantilever_thickness` | float \| list[float] | — | 悬臂（翼缘）厚：一个值，或 [根部, 尖端]；默认 top_thickness | the cantilever (flange outstand): one value or [at the web, at the tip]; default top_thickness | — |
| `top_haunch` | list[float] | — | 箱梁：箱内顶板与腹板相交处的承托 [宽, 高] | box_girder: the fillet in each inside top corner of the cell, [width, height] | — |
| `bottom_haunch` | list[float] | — | 箱梁：箱内底板与腹板相交处的承托 [宽, 高] | box_girder: the fillet in each inside bottom corner, [width, height] | — |
| `cells` | int | — | 箱梁：箱室数（默认 1）；中腹板竖直，均分在两外腹板中线之间 | box_girder: the number of cells (1 default); interior webs vertical, evenly between the outer webs' centre lines | — |
| `layers` | int | — | 箱梁、T 梁纤维截面：每块板沿厚度的纤维层数（默认 3） | box_girder, t_girder fibre sections: fibres through each wall's thickness (3 default) | — |
| `fiber_size` | float | — | 箱梁、T 梁纤维截面：沿板长方向的纤维最大尺寸（默认 height/8） | box_girder, t_girder fibre sections: the longest a fibre runs along a wall (default height / 8) | — |
| `origin` | `centroid` \| `top` \| `bottom` | — | 箱梁、T 梁纤维截面：构件轴线穿过的点——形心 `centroid`（默认）、桥面顶 `top`、梁底 `bottom`；钢筋与预应力束的坐标都相对它 | box_girder, t_girder fibre sections: where the member line passes -- `centroid` (default), the top of the deck, the bottom of the girder; bars and tendons are written about it | — |
| `bf` | float | — | 工字：翼缘宽（沿 y） | I: flange width (along y) | — |
| `tf` | float | — | 工字：翼缘厚 | I: flange thickness | — |
| `hw` | float | — | 工字：两翼缘之间的腹板净高（总高 = hw + 2 tf） | I: clear web height between the flanges (total depth = hw + 2 tf) | — |
| `tw` | float | — | 工字：腹板厚 | I: web thickness | — |
| `area` | float | — | `general`：截面面积（m²）；其他形状上给了就覆盖推算值 | general: cross-section area (m^2); on a shape, overrides the derived value | 48 个（01, 02, 03, 05 …） |
| `iy` | float | — | 绕截面 y 轴的惯性矩（纤维沿 z 分布），m⁴ | second moment about the section y axis (fibres vary in z), m^4 | 22 个（01, 03, 08, 16 …） |
| `iz` | float | — | 绕 z 轴的惯性矩，m⁴ | second moment about z, m^4 | 22 个（01, 03, 08, 16 …） |
| `j` | float | — | 圣维南扭转常数，m⁴；纤维截面不给时用纤维的极惯矩（箱梁会偏高）（纤维截面的旧写法 `torsion_constant`，1.0 版起弃用） | Saint-Venant torsion constant, m^4; on a fibre section the polar moment of the fibres when absent (the fibre sections' old spelling `torsion_constant` is deprecated since v1.0) | 30 个（01, 03, 08, 16 …） |
| `ay` | float | — | Timoshenko：y 向剪切面积（默认由形状推算） | Timoshenko: shear area for y-shear (default from the shape) | 08 |
| `az` | float | — | Timoshenko：z 向剪切面积 | Timoshenko: shear area for z-shear | 08 |
| `perimeter` | float | — | 杆：粘结用的接触周长（默认由形状推算） | bar: contact perimeter for bond (default from the shape) | — |
| `theory` | `euler` \| `timoshenko` | — | 梁理论：Euler–Bernoulli（默认）或 Timoshenko（需要 ay、az） | beam theory: Euler-Bernoulli (default) or Timoshenko (needs ay, az) | 08 |
| `kappa_y` | float | — | Timoshenko：ay 上的剪切修正系数（1 = 按 ay 原值） | Timoshenko shear correction factor on ay (1 = use ay as given) | 08 |
| `kappa_z` | float | — | Timoshenko：az 上的剪切修正系数 | Timoshenko shear correction factor on az | 08 |
| `ny` | int | — | 纤维：沿宽度的纤维数（rect），或沿壁的纤维数（box） | fiber: number of fibres across the width (rect), or along the walls (box) | 11, 21, 21b, 21c, 29d, 29e, 29f, 29g |
| `nz` | int | — | 纤维：沿高度的纤维数 | fiber: number of fibres through the depth | 11, 21, 21b, 21c, 29d, 29e, 29f, 29g |
| `nt` | int | — | 纤维圆管：沿环向的纤维数 | hollow fiber sections: fibres round the ring (`tube`) and through the wall thickness (both). `ny`/`nz` count along the walls of a `box`. | — |
| `nr` | int | — | 纤维圆管/箱：沿壁厚的纤维数 | fibres through the wall thickness of a tube or box | — |
| `cover` | float | — | 纤维 rect：保护层厚度；保护层内的纤维用 `material`，核心用 `core_material` | fiber rect: cover thickness; the fibres inside it carry `material`, the core `core_material` | 11, 21, 21b, 21c |
| `core_material` | str | — | 纤维 rect：`cover` 以内箍筋约束核心的材料（`materials` 里的名字） | fiber rect: material of the stirrup-confined core inside `cover` (a name in `materials`) | 11, 21, 21b, 21c |
| `integration` | str | — | 纤维：沿构件的积分站——gauss2（默认）或 lobatto3/4/5（含端点，铰在端部形成时用） | fiber: stations along the member -- gauss2 (default) or lobatto3/4/5 (end points sampled; a hinge forms at the end) | 19 个（11, 21, 21b, 21c …） |
| `patches` | list[dict[str, any]] | 空 | `shape: patches`：由四边形块拼成的纤维截面 [{corners: 四个 [y, z], ny, nz, material?, holes?}]，可有斜腹板、承托、孔洞 | `shape: patches`: a fibre section built of quadrilateral patches [{corners: [[y, z] x 4], ny, nz}], each subdivided bilinearly -- a box girder with inclined webs, haunches and tapered cantilevers. Coordinates in the section axes, as `bars`. | 11 个（24, 24b, 24c, 24d …） |
| `shear_stiffness` | float | — | 纤维截面的有效剪切刚度 κGA_s（N）：给了即按 Timoshenko 插值并加常剪应变；纤维不参与剪切，是分析者给定的弹性（或开裂折减）剪切柔度 | Effective shear stiffness kappa G A_s [N] of the fibre section. Absent, the member is Euler-Bernoulli; given, the bending interpolation becomes Timoshenko's (Reddy's interdependent element, phi = 12 E I / (GA_s L^2) per member) and a constant shear strain gamma = V / GA_s is added. The fibres do not see shear: this is the elastic (or cracked-effective, e.g. ASCE 41's 0.4 E_c A_w) member shear flexibility, stated by the analyst. | 21, 21c |
| `shear` | 见 FiberShearSpec | — | 纤维截面的开裂剪切律：未开裂 κGA → 开裂（弯曲开裂触发）后按给定或箍筋桁架模型的刚度 → 可选箍筋屈服；与 `shear_stiffness` 二选一；1.7 起也进了 numba 内核 | a cracking shear law for a fibre section, in series with the bending; exclusive with `shear_stiffness`; in the numba kernel since 1.7 | 21b |
| `bars` | list[dict[str, any]] | 空 | 截面内的钢筋 [{y, z, section: 杆截面名, count?}]，`count` 把 n 根相同钢筋合在一点 | [{y, z, section: <bar section name>, count?}] -- OpenSees `layer` per bar; `count` lumps n identical bars at the point | 13 个（11, 21, 21b, 21c …） |
| `bar_layers` | list[dict[str, any]] | 空 | 纤维截面内成排的钢筋（1.2）：[{section, from: [y, z], to: [y, z], count 或 spacing, z_from: origin\|top\|bottom}]，两端都放，按间距时取不大于它的等间距 | bars laid in straight layers (1.2): [{section, from: [y, z], to: [y, z], count \| spacing, z_from}] -- both ends included; `spacing` puts as many bars as fit at that spacing or closer (d16 @ 150 over 1 m is 8 bars); `z_from: top \| bottom` measures z from that face, downward from the top | — |
| `tendons` | list[dict[str, any]] | 空 | 纤维截面内的预应力束：{section, profile: [[s, y, z], ...], prestress 或 jack + friction, elastic_shortening: none\|compensate}；束按各单元中点处的 profile 位置成为一根带初应变的粘结钢纤维 | PRESTRESSING TENDONS of a fibre section: each is {section: <bar section>, profile: [[s, y, z], ...], interpolation: linear \| quadratic, normalised: false, prestress: <Pa> \| [[s, <Pa>], ...]  \|  jack: <Pa>, friction: {mu, k, jack, anchor_set}, elastic_shortening: none \| compensate} `profile` is the tendon's line: `s` the distance along the member from its start (the end with the smaller global x, then y, then z; a fraction of the member length when `normalised`), `y`/`z` the offsets in the section axes, as for `bars`; `quadratic` is one parabola through each three consecutive points (0-1-2, 2-3-4, ...: an odd count), and the curve as placed must stay inside the section. The tendon becomes a bonded steel fibre whose position is the profile's value at each element's midpoint -- a draped tendon is a polyline of one eccentricity per element -- carrying an initial strain: `prestress` is the stress the tendon holds with the section unstrained, or `jack` is the stress at the jack and `friction` is the post-tensioning loss profile the embedded tendons already use. What the relaxation then finds is the elastic shortening; `elastic_shortening: compensate` adds the linear-elastic estimate of it back (the gross section's 3x3 stiffness about the origin, so a T or a box need not be written about its centroid), so the stated stress is what the tendon carries against the hardened, already-shortened concrete. | 11 个（24, 24c, 24d, 24f …） |
| `fracture` | dict[str, any] | — | `{threshold: 1.0}`：截面内失效纤维达到这个比例时构件断开（写在截面上，因为截面才知道有多少纤维） | `fracture: {threshold: 1.0}` -- the member is severed when that fraction of its fibres has failed. On the SECTION and not on the assignment because it is the section that says how many fibres there are and which laws they carry; what makes a fibre fail is in the material block. | — |
| `stiffness` | any | — | 连接器：六个对角刚度 [ux, uy, uz, rx, ry, rz]（N/m、N·m/rad）或对称 6×6 矩阵（docs/connector2.md） | connector: six diagonal values [ux, uy, uz, rx, ry, rz] (N/m, N m/rad) or a symmetric 6x6 matrix (docs/connector2.md) | 09, 24e, 24f, 32 |
| `connector_damping` | any | — | 连接器：粘滞阻尼，写法同 `stiffness`（N·s/m、N·m·s/rad） | connector: viscous damping in the same form as `stiffness` (N s/m, N m s/rad) | — |
| `curves` | dict[str, list[list[float]]] | 空 | 连接器：某分量的分段线性力–位移骨架 [[u, F], ...]，代替该分量的刚度（可逆、单调不减） | connector: per component a piecewise-linear force-displacement skeleton [[u, F], ...] replacing that component's stiffness (reversible, non-decreasing) | 09 |
| `laws` | dict[str, 见 ConnectorLawSpec] | 空 | 连接器：某分量的路径相关律——`gap` 间隙/限位、`bilinear` 双线性滞回（隔震支座）、`friction` 库仑摩擦滑动；见 ConnectorLawSpec | connector: a path-dependent law per component -- gap, bilinear hysteresis, Coulomb friction (schema.ConnectorLawSpec) | 32 |

### `sections.*.shear` — FiberShearSpec

A cracking shear law for a fibre section, in series with the fibres' bending: V = GA0 gamma until the section cracks, then the cracked stiffness, then (optionally) a stirrup-yield plateau with a small hardening; unloading and reloading run to the origin once cracked.  The cracked stiffness is given, or taken from the 45-degree truss of the stirrups (Park & Paulay 1975, sec. 7.3): GA_cr = rho_v / (1 + 4 n rho_v) E_s b_w d, rho_v = A_v / (b_w s), n = E_s / E_c. The section cracks in shear when its concrete first cracks in flexure (`cracking: flexural`, the flexure-shear cracks a frame member actually has), or at a stated shear force.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `direction` | `y` \| `z` | `z` | 剪力方向：`z`（截面高度方向，绕 y 弯曲的那个平面）或 `y` | the direction the shear acts along: `z` (the section's depth, the plane of bending about y) or `y` | — |
| `stiffness` | float | **必填** | 未开裂 κGA（N） | uncracked kappa G A (N) | 21b |
| `cracked_stiffness` | float | — | 开裂后 GA（N）；或给 `stirrups` 按桁架模型算 | cracked GA (N); or give `stirrups` for the truss model | — |
| `stirrups` | 见 StirrupSpec | — | 箍筋（桁架模型的开裂剪切刚度） | stirrups (the truss model's cracked stiffness) | 21b |
| `web_width` | float | — | 桁架模型的腹板宽 b_w（m）；默认截面在垂直于剪力方向的宽度 | the web width b_w for the truss model (m); by default the section's width across the shear | — |
| `cracking` | `flexural` \| float | `flexural` | 何时开裂：`flexural`（该构件的混凝土纤维第一次受拉开裂时）或一个剪力值（N） | when it cracks: `flexural` (when the member's concrete first cracks in tension) or a shear force (N) | — |
| `yield_force` | float | — | 箍筋屈服时的剪力（N），之后按 `hardening` 的刚度比继续 | the shear at which the stirrups yield (N); beyond it the stiffness is `hardening` times the cracked one | — |
| `hardening` | float | `0.01` | 屈服后刚度比 | post-yield stiffness ratio | — |

### `sections.*.shear.stirrups` — StirrupSpec

Stirrups for the truss-model cracked shear stiffness.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `area` | float | **必填** | 一道箍筋的总肢面积（m²，各肢之和） | area of one stirrup set, all legs (m^2) | 21b |
| `spacing` | float | **必填** | 箍筋间距（m） | spacing along the member (m) | 21b |
| `depth` | float | **必填** | 有效高度 d（m） | effective depth d (m) | 21b |
| `E` | float | `200000000000.0` | 箍筋弹性模量：默认 2e11 Pa，声明了非 SI 的 `units` 时换算（MPa 下为 2e5） | stirrup modulus: 2e11 Pa by default, converted to the declared `units` when they are not SI (2e5 in MPa) | 21b |

### `sections.*.fracture` — FractureSpec

When an element stops carrying load, and what happens then.  ONE NUMBER, because everything else about fracture is decided elsewhere on purpose (core/fracture.py): WHEN a material point is finished is the material's business and is written in the material block (a bar's `rupture_strain`, a damage law's own limit), and HOW the points of this element family add up is the element's and is not a choice at all. What is left for the analyst is how much of the section has to be gone before the member is treated as severed, and the honest default is all of it.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `threshold` | float | `1.0` | 截面纤维（或实体的高斯点）失效的比例达到它即断；1.0 = 整个截面 | fraction of the section's fibres (or a solid's Gauss points) that must have failed. 1.0 = the whole section; lower cuts the member earlier, which is a modelling shortcut and should be said out loud in a report. | — |
| `split` | bool | `False` | 断开时把构件从框架上分离出来：它的质量转到自己的粒子上、碎片下落（需要 mode: dynamic；没有接触） | separate the member from the frame when it fractures, instead of only switching its forces off. The member's share of the mass moves onto particles of its own and the fragment falls. Off by default because it is a different analysis: it needs solver.mode = dynamic, it reserves two particles per element, and what the debris then lands on is not modelled yet -- there is no contact. | — |

### `sections.*.laws.*` — ConnectorLawSpec

A path-dependent law on ONE component of a connector: `gap` (free travel, then a stop), `bilinear` (elastic-plastic with kinematic hardening: an isolator, a yielding device) or `friction` (Coulomb: sticks, then slides at mu times the compression another component of the same connector carries). elements/connector.py has the mechanics; one scalar of history per component, the slipped deformation.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `type` | `gap` \| `bilinear` \| `friction` \| `damper` \| `pendulum` \| `material` | **必填** | 律的种类：`gap` 间隙/限位，`bilinear` 双线性滞回（隔震支座、屈服耗能装置），`friction` 库仑摩擦滑动，`damper` 非线性粘滞阻尼器 F = c·sign(v)·\|v\|^α（1.10），`pendulum` 摩擦摆支座（1.10），`material` 材料表里的一维材料，F = area·σ(d/length)（1.21） | `gap`, `bilinear`, `friction`, `damper` (a fluid viscous damper, F = c sign(v) \|v\|^alpha; 1.10), `pendulum` (a friction pendulum bearing; 1.10) or `material` (a one-dimensional law of the material table, F = area sigma(d / length); 1.21) | 32 |
| `lo` | float | — | gap：自由行程的下限（m 或 rad），不给即该侧无限位（friction 的粘着刚度是 `stiffness`，不读 `lo`） | gap: lower bound of the free travel (None = no stop that side); friction does not read it (its stick stiffness is `stiffness`) | 32 |
| `hi` | float | — | gap：自由行程的上限；在局部 ux（伸长为正）上，`{lo: 0}` 是只受压的支承（压紧受力、拉开自由），`{hi: 0}` 是只受拉的 | gap: upper bound of the free travel; on the local ux (elongation positive) `{lo: 0}` is a compression-only seat and `{hi: 0}` a tension-only tie | — |
| `stiffness` | float | — | gap：碰到限位后的刚度（N/m 或 N·m/rad）；friction、pendulum：粘着刚度；damper：与阻尼器并联的弹簧（可不写） | gap: the stop's stiffness once a bound is reached; friction, pendulum: the stick stiffness before sliding; damper: a spring beside the damper (optional) | 32 |
| `k1` | float | — | bilinear：初始刚度 | bilinear: initial (elastic) stiffness | — |
| `k2` | float | — | bilinear：屈服后刚度（0 = 理想塑性；须小于 k1） | bilinear: post-yield stiffness (0 = perfectly plastic; must be below k1) | — |
| `fy` | float | — | bilinear：屈服力（N 或 N·m） | bilinear: yield force | — |
| `mu` | float | — | friction、pendulum：摩擦系数 | friction, pendulum: coefficient of friction | 32 |
| `radius` | float | — | pendulum：滑动球面的等效曲率半径（m）；恢复刚度是 N/radius，N 为支座当时的压力 | pendulum: effective radius of the sliding surface; the restoring stiffness is N / radius, N the compression the bearing carries at that instant | — |
| `c` | float | — | damper：阻尼系数 c（N·(s/m)^α），F = c·sign(v)·\|v\|^α | damper: the coefficient c of F = c sign(v) \|v\|^alpha (force per velocity^alpha) | — |
| `exponent` | float | — | damper：速度指数 α，0 < α ≤ 1（1 即线性阻尼器；桥梁常用 0.2–0.5） | damper: the velocity exponent alpha, 0 < alpha <= 1 (1 is the linear dashpot; bridges use 0.2-0.5) | — |
| `linear_below` | float | — | damper：低于这个速度（m/s）时按线性 F = c·v0^(α-1)·v；α < 1 时必须写（v → 0 时幂律的斜率无穷大，显式积分会在零速附近抖动）。取设计速度的 1% 左右 | damper: below this velocity the law is linear, F = c v0^(alpha-1) v; required when alpha < 1 (the power law's slope is infinite at v = 0 and an explicit step chatters there). About 1% of the design velocity | — |
| `normal` | str \| float | — | friction：压紧力从哪来——`ux`（同一连接器局部 ux 线性弹簧当前的压力；只有它的正负与节点顺序无关），或一个常数（N） | friction: where the compression comes from -- `ux` (the current compression of this connector's local ux spring, the one component whose sign does not depend on the node order) or a constant force | 32 |
| `material` | str \| dict[str, any] | — | material：`materials` 里一维材料的名字（可以是 parallel、series、gap） | material: the name of a one-dimensional law in `materials` (parallel, series and gap included) | — |
| `area` | float | — | material：力 = area × 应力；不写为 1（材料表即力–变形关系） | material: force = area x stress; 1 when left out (the table is then the force-deformation relation) | — |
| `length` | float | — | material：应变 = 变形 / length；不写为 1 | material: strain = deformation / length; 1 when left out | — |

### `reinforcement.*` — ReinforcementGeometry

One named bar set: exactly one primitive, or a plain list of lines.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `line` | 见 LinePrim | — | 一根直钢筋（LinePrim） | one straight bar (LinePrim) | 13 |
| `lines` | list[见 LinePrim] | — | 一组里的多根直钢筋 | several straight bars in one set | 06 |
| `layer` | 见 LayerPrim | — | `count` 根平行钢筋（LayerPrim） | `count` parallel bars (LayerPrim) | 12, 12b, 12c, 15 |
| `hoop` | 见 HoopPrim | — | 沿某轴重复的封闭矩形箍筋（HoopPrim） | closed rectangular ties along an axis (HoopPrim) | 12, 12b, 12c |
| `path` | 见 PathPrim | — | 过控制点的弯起或曲线钢筋/束（PathPrim） | a bent or curved bar through control points (PathPrim) | 14, 15 |

### `reinforcement.*.line` — LinePrim

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `start` | list[float] | **必填** | 钢筋起点 [x, y, z]，网格坐标 | first end of the bar, [x, y, z] in mesh coordinates | 13 |
| `end` | list[float] | **必填** | 终点 | second end | 13 |
| `divisions` | int | — | 沿钢筋的分段数；会开裂的构件里优先用 `segment_length`（见 schema.BarLineSpec） | number of segments along the bar; prefer `segment_length` in a member that cracks (see schema.BarLineSpec) | — |
| `segment_length` | float | — | 分段长度（m），对粘结钢筋是物理的弥散长度；取整到整段 | segment length in metres (a physical smearing length for a bonded bar), rounded to whole segments | 13 |
| `repeat` | dict[str, any] | — | `{count, offset: [dx, dy, dz]}`：复制 `count` 根，每根比上一根平移 `offset` | `{count, offset: [dx, dy, dz]}`: `count` copies of the line, each shifted by `offset` from the last | 13 |

### `reinforcement.*.layer` — LayerPrim

OpenSees `layer straight`: `count` parallel bars from the first at `from`..`to`, each the next one `spacing` away.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `from` | list[float] | **必填** | 第一根钢筋的起点 [x, y, z] | the first bar, [x, y, z] of its start | 12, 12b, 12c, 15 |
| `to` | list[float] | **必填** | 第一根钢筋的终点 | the first bar's end | 12, 12b, 12c, 15 |
| `count` | int | **必填** | 平行钢筋的根数 | how many parallel bars | 12, 12b, 12c, 15 |
| `spacing` | list[float] | `[0.0, 0.0, 0.0]` | 相邻两根的偏移 [dx, dy, dz] | offset from one bar to the next, [dx, dy, dz] | 12, 12b, 12c, 15 |
| `divisions` | int | — | 每根的分段数（或 `segment_length`） | segments per bar (or `segment_length`) | — |
| `segment_length` | float | — | 分段长度（m） | segment length in metres | — |

### `reinforcement.*.hoop` — HoopPrim

A closed rectangular tie in the plane normal to `axis`, repeated along it from `from` to `to` at `spacing`. `corners` are the two opposite corners in the plane's own two coordinates (the other two axes in x, y, z order).

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `corners` | list[list[float]] | **必填** | 箍筋在垂直于 `axis` 的平面内的两个对角，用该平面自己的两个坐标 | two opposite corners of the tie in the plane normal to `axis`, in that plane's own two coordinates | 12, 12b, 12c |
| `axis` | `x` \| `y` \| `z` | **必填** | 箍筋沿哪个轴堆叠 | the axis the ties are stacked along | 12, 12b, 12c |
| `from` | float | **必填** | 第一道箍筋沿 `axis` 的坐标 | coordinate along `axis` of the first tie | 12, 12b, 12c |
| `to` | float | **必填** | 最后一道箍筋的坐标 | coordinate of the last tie | 12, 12b, 12c |
| `spacing` | float | **必填** | 沿 `axis` 的箍距 | tie spacing along `axis` | 12, 12b, 12c |
| `divisions` | int | — | 每道箍筋每边的分段数（或 `segment_length`） | segments per side of each tie (or `segment_length`) | 12, 12b, 12c |
| `segment_length` | float | — | 分段长度（m） | segment length in metres | — |

### `reinforcement.*.path` — PathPrim

A bar route through control points, rather than a single straight run.  Two shapes, and the choice between them is not a smoothing preference:  `polyline` (the default) lays a straight leg between each pair of points. It is exact -- the bar goes where the points say and nowhere else -- and it is what a bent-up bar, an L-bar, a hook and a HARPED pretensioned tendon actually are. Each leg is subdivided by `segment_length` (or `divisions`) the way an ordinary `line` is.  `parabola` takes exactly three points and draws the unique quadratic through them, with the middle point at the PARAMETER midpoint -- so for the usual symmetric tendon (anchorage, low point at midspan, anchorage) the middle point is the low point, and for an asymmetric profile it is not what you meant and `polyline` is the honest way to say it. The curve is sampled at EQUAL ARC LENGTH, so `segment_length` means the same physical thing here as it does on a straight bar (see BarLineSpec: it is a smearing length, not a discretisation choice), and each chord becomes one bar segment.  A path can `repeat`, exactly as a line can: several parallel tendons or a row of identical bent-up bars is one entry plus an offset.  WHAT A CURVED TENDON DOES NOT BUY YOU. The geometry is right and the prestress acts along it -- so the vertical component of a draped tendon balances load, which is the whole point of draping one. Friction is still not computed: `prestress` remains the effective stress after losses. What preflight can now give you is the number a friction calculation needs, the tendon's total angular change, because a path has one.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `points` | list[list[float]] | — | 至少两个点，每个 [x, y, z]；`parabola` 正好三个 | at least two points, each [x, y, z]; three exactly, for `parabola` | 14, 15 |
| `shape` | `polyline` \| `parabola` | `polyline` | `polyline` 各点之间直线段（默认）或 `parabola` 过三点的抛物线 | `polyline` (straight legs between the points) or `parabola` (three points) | 14, 15 |
| `group` | str | — | 把 Gmsh 曲线当钢筋路径——保留字段，目前拒绝 | a Gmsh curve as a bar route -- reserved, and refused with a message | — |
| `divisions` | int | — | `polyline` 每段各分、`parabola` 整条曲线分这么多段 | `polyline` subdivides each leg; `parabola` divides the whole curve | — |
| `segment_length` | float | — | 分段长度（m）：polyline 按每段、parabola 按整条弧长 | segment length in metres along each leg (polyline) or the whole curve (parabola) | 14, 15 |
| `repeat` | dict[str, any] | — | `{count, offset}`，同 `line` | `{count, offset}` as for a `line` | 14, 15 |

### `regions.*` — RegionSpec

One primitive, or a boolean combination of other regions.  See core/selectors.py. Exactly one selector per region; `tol` is an absolute length and defaults to 1e-6 of the mesh diagonal, so the same case works whether the mesh is in millimetres or metres.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `box` | list[float] | — | [xmin, xmax, ymin, ymax, zmin, zmax]：盒子内的节点（按轴成对给） | [xmin, xmax, ymin, ymax, zmin, zmax]: nodes inside the box, the bounds in pairs per axis (it was documented as all minima, then all maxima, which selected something else) | 16 个（11, 12, 12c, 13 …） |
| `where` | dict[str, float \| dict[str, float]] | — | 逐轴给 {min, max}（可只给一个）或一个数（该轴取值在 `tol` 内）：{x: 0.1, z: 0} 是一条线，三个轴都给是一点 | per axis either {min, max} (either optional) or a bare number, which pins that axis within `tol`: {x: 0.1, z: 0} is a line, three axes a point | 12, 12b, 12c |
| `plane` | dict[str, any] | — | `{point: [x,y,z], normal: [nx,ny,nz]}`：离平面 `tol` 内的节点 | `{point: [x,y,z], normal: [nx,ny,nz]}`: nodes within `tol` of the plane | 06, 20 |
| `line` | dict[str, any] | — | `{from: [x,y,z], to: [x,y,z]}`：离线段 `tol` 内的节点 | {from: [x,y,z], to: [x,y,z]}: nodes within `tol` of the segment | — |
| `cylinder` | dict[str, any] | — | `{point, axis, radius, rmin?, length?}`：圆柱内的节点（`point` 在轴上，`length` 从它沿 `axis` 量） | `{point, axis, radius, rmin?, length?}`: nodes inside the cylinder (`point` on the axis, `length` measured from it along `axis`) | — |
| `sphere` | dict[str, any] | — | `{centre, radius, rmin?}`：球内的节点（`center` 也收） | `{centre, radius, rmin?}`: nodes inside the sphere (`center` is taken too) | — |
| `group` | str | — | Gmsh 物理组名（该组单元的节点） | a Gmsh physical group name (nodes of its elements) | — |
| `node_ids` | list[int] | — | 直接给节点编号（从 0 起） | explicit 0-based node numbers | — |
| `nearest` | list[float] | — | 离 [x, y, z] 最近的那一个节点 | the one node nearest [x, y, z] | 10, 19 |
| `all` | bool | — | `true`：全部节点 | `true`: every node | — |
| `union` | list[str] | — | 并集：要合并的区域名 | names of regions to unite | 18 个（21, 21b, 21c, 22 …） |
| `intersect` | list[str] | — | 交集：区域名 | names of regions to intersect | — |
| `difference` | list[str] | — | [a, b, ...]：第一个区域减去其余 | [a, b, ...]: the first region minus the others | — |
| `tol` | float | — | 几何选择器的绝对容差（默认网格对角线的 1e-6） | absolute tolerance for the geometric selectors (default 1e-6 of the mesh diagonal) | — |

### `assign[]` — AssignSpec

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `section` | str | **必填** | `sections` 里的名字 | a name in `sections` | 58 个（01, 02, 03, 05 …） |
| `region` | str | — | 目标二选一：网格区域，或一个/多个钢筋组 | exactly one target: a mesh region, or one or more reinforcement sets | 55 个（01, 02, 03, 05 …） |
| `reinforcement` | str \| list[str] | — | `reinforcement` 里的一个或多个名字：这个（杆）截面描述的钢筋 | one or more names in `reinforcement`: the bars this (bar) section describes | 06, 12, 12b, 12c, 13, 14, 15 |
| `orientation` | list[float] | — | 梁/连接器：确定截面局部 y 轴的向量（不能与构件平行） | beam/connector: a vector fixing the section's local y axis (not parallel to the member) | 38 个（01, 03, 08, 09 …） |
| `releases` | list[str] | 空 | 梁端释放，取自 i_rx, i_ry, i_rz, j_rx, j_ry, j_rz（i 为第一个节点）；两端铰接写 [i_ry, i_rz, j_ry, j_rz] | beam end releases, from i_rx, i_ry, i_rz, j_rx, j_ry, j_rz (i = first node); a pin at both ends is [i_ry, i_rz, j_ry, j_rz] | — |
| `end_offsets` | dict[str, any] | — | 梁端刚性偏移（1.21）：{i: [x, y, z], j: [x, y, z], axes: local\|global, per: line\|member, axial: elastic\|full}，弹性段两端离节点的向量（`local` 是构件按节点连线和 `orientation` 定的轴）。梁端刚域沿构件给，截面偏心（形心在节点下方）横向给。`axial: full` 让框架梁的轴向和扭转刚度按节点间的长度算；默认 `elastic` 按弹性段 | rigid end offsets (1.21): {i: [x, y, z], j: [x, y, z], axes: local\|global, per: line\|member, axial: elastic\|full}, the vectors from the nodes to the ends of the member's elastic part (`local`: the member's axes from its nodes' chord and `orientation`). A rigid end zone is an offset along the member, a section whose centroid is below its nodes one across it. `axial: full` takes a frame beam's axial and torsional stiffness over the length between the nodes; `elastic`, the default, over the elastic part | — |
| `coordinate_system` | `local` \| `global` | — | 连接器：`local`（x 沿弦、y 由 `orientation` 定）或 `global` 分量 | connector: `local` (x along the chord, y from `orientation`) or `global` components | 09, 24e, 24f |
| `bond` | dict[str, any] | — | 嵌入钢筋：`{type: bond_ceb \| bond_cyclic, ...}` 让钢筋有自己的自由度并按粘结–滑移律传力；不给即完全粘结 | embedded bars: `{type: bond_ceb \| bond_cyclic, ...}` gives the bar its own dofs and a bond-slip law; absent = perfect bond | 12b |
| `segment_length` | float | — | 嵌入钢筋：几何没给时，这组每根线的分段长度（m） | embedded bars: segment length (m) for every line of the set, when the geometry gives none | 12, 12b, 12c, 15 |
| `divisions` | int | — | 嵌入钢筋：几何没给时，每根线的分段数 | embedded bars: segments per line, when the geometry gives none | — |
| `transverse_penalty` | float | — | 粘结钢筋：横向约束刚度（N/m），默认取钢筋自身 EA/L_trib。数值罚函数，不是销栓作用 | bonded bars: stiffness holding the bar across its axis (N/m); default the bar's own E A / L_trib. A numerical device, not a dowel | — |
| `host_deduction` | float | — | 从钢筋应力里扣除混凝土基体的 E·ε（只对弹性基体；基体会损伤时求解器拒绝） | subtract the host's E*eps from the bar (elastic hosts only; the solver refuses it with a damaging host -- see elements/truss.py) | — |
| `anchorage` | str \| dict[str, any] | — | 纤维梁：应变渗透 {region, bar_diameter}；粘结钢筋：`ends`/`start`/`end` 把钢筋端节点绑到基体上 | fiber beams: strain penetration, {region, bar_diameter}; bonded bars: `ends` / `start` / `end`, the bar nodes tied to the host | 11, 21, 21b, 21c |
| `prestress` | float | — | 先张：扣除损失后的**有效**预应力（Pa，受拉为正）。程序只算弹性压缩损失——写 σ_pe，不是 σ_con | pretensioning: the EFFECTIVE prestress in the bar after losses, Pa, positive in tension. The program computes the elastic-shortening loss (it is what equilibrium at transfer gives) and nothing else -- write sigma_pe here, not sigma_con. | 13, 14, 15, 33 |
| `friction` | dict[str, any] | — | 后张摩擦 {mu, k, jack, anchor_set}：给了则 `prestress` 是张拉端应力，程序沿束算损失剖面；不给则整根束都是 `prestress` | post-tensioning friction {mu, k, jack, anchor_set}. With it, `prestress` is the JACKING stress and the solver computes the profile along the tendon; without it, `prestress` is what the whole tendon carries. | — |
| `unstressed_length` | float | — | 线区域上的 `bar` 截面作为索：总无应力长度（代替 `prestress`） | a `bar` section on a line region as a CABLE: its total unstressed length (an alternative to `prestress`), and whether it carries no compression | 22 |
| `unstressed_lengths` | list[float] | — | 区域内每个单元一个无应力长度，按单元顺序（找形的输出） | one per element of the region, in element order (form-finding output) | 23 |
| `jack` | float | — | 索按力张拉（N）：在建立它的那个分析步（从头就建的是第一步）结束时，每根构件的轴力正好是这个值，不管结构在张拉中让了多少；此后它是一根普通的杆，无应力长度就是那一步结束时定下的。只用于框架模型（梁、杆、连接器） | the force (N) a cable is JACKED to: at the end of the analysis step that builds it (the first, when built from the start) each member carries exactly this, whatever the structure gave under it; from then on it is a bar of the unstressed length it ended with. Frame models (beams, bars, connectors) only | — |
| `tension_only` | bool | `False` | 索：不受压（松弛后力为零） | cable: the member carries no compression (goes slack) | 22, 23 |
| `failure` | dict[str, any] | — | 线构件的弹脆性断裂：见 schema.FailureSpec | elastic-brittle breaking of a line member: see schema.FailureSpec | 34 |
| `name` | str | — | 钢筋组的名字，供监测量（`of:`）和结果字段用；默认取截面名 | the reinforcement set's name for monitors (`of:`) and result fields; defaults to the section name | 06, 12, 12b, 12c, 13, 14, 15 |

### `assign[].bond` — BondSpec

Bond stress-slip between an embedded bar and its host.  Present means the bar gets its OWN degrees of freedom and may slip; absent means perfect bond, which is cheaper, runs on every backend, and is right whenever the member is not expected to crack. See materials/bond.py for the law and v0.1 for why the choice matters more than it looks.  TWO LAWS SHIP, AND THE DEFAULT IS THE MONOTONIC ONE     `bond_ceb`     the MC2010 envelope with a linear unloading rule. Right                    for a pushover; under a long cyclic history it                    overstates the bond that survives, which is the unsafe                    direction.     `bond_cyclic`  the same envelope with Eligehausen-type degradation and                    a pinched reversal. Identical to `bond_ceb` on a                    monotonic path, bit for bit, so switching costs nothing                    but the extra state.      The default stays `bond_ceb` because changing it would silently     re-rate every model that already names a bond block.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `type` | str | `bond_ceb` | 粘结–滑移律：`bond_ceb`（MC2010 单调）或 `bond_cyclic`（往复退化）；其余键是该材料的参数 | the bond-slip law: `bond_ceb` (MC2010 monotonic) or `bond_cyclic`; every other key is that material's parameter | 12b |

### `assign[].friction` — TendonFrictionSpec

Post-tensioning friction: sigma(s) = sigma_con exp(-(mu theta + k s)).  With this present, `prestress` is sigma_con AT THE JACK and the solver derives the stress in every segment; without it, `prestress` is the value the whole tendon carries. That is the one place the meaning of a field depends on another, and preflight says which reading is in force.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `mu` | float | **必填** | 曲率摩擦系数（每弧度角度变化） | curvature coefficient, per radian of accumulated angle change | — |
| `k` | float | `0.0` | 摆动摩擦系数（每米束长） | wobble coefficient, per metre of tendon | — |
| `jack` | `start` \| `end` \| `both` | `start` | 张拉端在哪头；`both` 两端张拉，每点取较小的损失 | which end the jack is at; `both` takes the smaller loss at each point | — |
| `anchor_set` | float | — | 锚固回缩量，**米**（6 mm 写 0.006） | draw-in as the wedges seat, in METRES (6 mm is 0.006) | — |

### `assign[].failure` — FailureSpec

When a line member breaks: ELASTIC-BRITTLE -- linear until the criterion, then gone. Dynamic runs only; per member group in a mixed_line model (`assign[].failure`), for every member of a single- section beam2/beam2_timoshenko/truss2 model (`element.failure`).  A bar (cable) breaks when its tension reaches `tensile_strength` x area. A beam breaks when, at either end, |N|/(A f_y) + |M_y|/M_y,cap + |M_z|/M_z,cap >= 1 with M_cap = shape_factor f_y I / c: `extreme_fibre` is [c_y, c_z], the distances from the member axis to the extreme fibres along the section's local y and z. Shear and torsion are not checked against a strength; but a beam twisted a quarter turn end to end is removed (the co-rotational frame cannot follow half a turn), and a piece broken at both ends is removed once its demand reaches twice its capacity (it is still steel).  Why not a strain limit: in an elastic member the strain at which steel actually tears (15%) is reached only at 30 000 MPa, so the member would never break first; a CAPACITY is what an elastic member can be checked against, and it needs only A, I and the depth of the section.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `tensile_strength` | float | — | 杆/索：拉应力达到它即断（Pa） | bars and cables: the tension at which the member breaks, Pa | — |
| `yield_strength` | float | — | 梁：承载力判据 \|N\|/N_p + \|M_y\|/M_y,cap + \|M_z\|/M_z,cap ≥ 1 用的屈服强度（Pa） | beams: the yield strength (Pa) behind N_p = f_y A and M_cap = shape_factor f_y I / c | 34 |
| `extreme_fibre` | list[float] | — | [c_y, c_z]：构件轴到局部 y、z 方向最外纤维的距离（m），M_cap = 形状系数·f_y·I/c | beams: [c_y, c_z], the distances (m) from the member axis to the extreme fibres along the section's local y and z | 34 |
| `shape_factor` | float | `1.0` | 塑性/弹性弯矩之比（1 = 首次屈服；箱形约 1.12） | plastic over elastic moment (1 = first yield; a thin-walled box about 1.12) | — |
| `rotation_capacity` | float | — | 梁（mixed_line）：给了它，梁端就是塑性铰——端弯矩按 \|N\|/N_p + \|M_y\|/M_y,p + \|M_z\|/M_z,p ≤ 1 封顶，超出的转成塑性转角，累计塑性转角（rad）到它才在该端断开；不写就是弹脆性，到承载力立即断开 | beams (mixed_line): given, every end is a PLASTIC HINGE -- the end moments are capped on \|N\|/N_p + \|M_y\|/M_y,p + \|M_z\|/M_z,p <= 1, what would exceed it becomes plastic rotation, and the end breaks when its accumulated plastic rotation (rad) reaches this; left out, elastic-brittle: the end breaks the moment the capacity is reached | — |
| `split` | bool | — | 断了之后：true 质点分裂——构件两端移到自己的备用质点上、带着质量脱开、继续作为一段自由的弹性构件；false 只是不再传力、质量留在节点。不写：梁/mixed_line 分裂，独立桁架不分裂 | after the break: true splits the member off onto its own particles with its mass (a free elastic segment), false only deactivates it (mass stays on the nodes); left out, frames and mixed_line split, the standalone truss deactivates | — |

### `offset_arms` — OffsetArmSpec

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `material` | str | **必填** | 刚臂的材料（取它的 E） | the arms' material (its E) | — |
| `factor` | float | `30.0` | 刚臂的刚度：所连偏心构件的多少倍（沿轴向和转动） | an arm's stiffness: this times what the members set off carry, along and in turning | — |
| `limit` | float | — | 一个构件传给刚臂的刚度上限（力/长度，不写为 1e13 N/m 换算到 `units`；转动取 limit L²/3） | the most a member hands an arm (force/length; none: 1e13 N/m in the case's `units`; in turning, limit L^2/3) | — |
| `regions` | list[str] | **必填** | 刚臂：每个区域一根杆，从画出的节点到形心节点 | the arms: one member per region, from the drawn node to the centroid node | — |

### `tendons[]` — TendonSpec

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `name` | str | **必填** | 钢束名：结果里按它报告，展开出的区域以它开头（`名字/bar1`、`名字/arm1`） | its name: what the results report it by, and the start of the regions it expands to (`name/bar1`, `name/arm1`) | — |
| `nodes` | str | **必填** | 它粘结的主梁节点：网格的一个节点组；用它长度范围内的节点，以及离两个锚点最近的节点 | the girder nodes it is bonded to: a group of the mesh; the ones along its length are used, and the nearest to each anchor | — |
| `section` | str | **必填** | 杆截面：一根钢绞线（或钢筋）的面积和材料 | a bar section: the area of one strand or bar, and its material | — |
| `count` | int | `1` | 孔道里的根数，面积乘它 | strands or bars in the duct: the area is times this | — |
| `line` | 见 TendonLine | **必填** | 参考线 | its reference line | — |
| `offset` | list[float] | `[0.0, 0.0]` | 处处加在剖面上的 [oy, oz] | [oy, oz] added to the profile everywhere | — |
| `profile_y` | list[list[float]] | 空 | 平弯：[x, y, 半径] 点列；不写则在参考线上 | across the line: [x, y, radius] points; none: on the line | — |
| `profile_z` | list[list[float]] | **必填** | 竖弯：[x, z, 半径] 点列，折角按半径倒圆 | its drape: [x, z, radius] points, each corner rounded to its radius | — |
| `stress` | 见 TendonStress | **必填** | 张拉 | its jacking | — |
| `friction` | 见 TendonFriction | （默认块） | 摩擦损失 | its friction losses | — |
| `draw_in` | 见 TendonDrawIn | （默认块） | 锚具回缩 | its anchorages' draw-in | — |
| `relaxation` | 见 TendonRelaxation | — | 松弛（不写则不松弛） | its relaxation (none: it does not relax) | — |
| `arm_factor` | float | `30.0` | 刚臂的刚度：钢束最刚的一段杆（4EA/L）的多少倍 | an arm's stiffness: this times the tendon's stiffest bar (4 E A / L) | — |

### `tendons[].line` — TendonLine

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `origin` | list[float] | **必填** | 参考线上剖面 x = 0 的点 | a point of the tendon's reference line, where its profile's x is 0 | — |
| `direction` | list[float] | **必填** | 参考线方向（x）；y 水平、垂直于它，z 是第三根轴（参考线不竖直时朝上） | the reference line's direction (x); its y is horizontal, across it, and z the third axis (up for a line that is not vertical) | — |
| `turn` | float | `0.0` | y、z 两轴绕 x 转过的角度（度） | the y and z axes turned about x by this angle (degrees) | — |

### `tendons[].stress` — TendonStress

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `step` | str | — | 张拉它的分析步（步名）；不写则是它粘结的构件最后建成的那一步 | the step that stresses it (its name); none: the step in which the last of the members it is bonded to is built | — |
| `ends` | `begin` \| `end` \| `both` | `begin` | 张拉端：`begin`、`end`、`both` | the end or ends it is jacked at: `begin`, `end`, `both` | — |
| `stress` | list[float] | — | 起点端和终点端的张拉控制应力（应力单位），或…… | the stress at the jack, at its begin and its end (stress unit), or ... | — |
| `force` | list[float] | — | ……整束的张拉力（力单位），两者写一个 | ... the whole tendon's force at the jack (force unit); one of the two | — |

### `tendons[].friction` — TendonFriction

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `mu` | float | `0.0` | 摩擦系数 μ，每弧度转角 | the friction coefficient mu, per radian of the line's turning | — |
| `k` | float | `0.0` | 偏差系数 k，每单位长度 | the wobble coefficient k, per unit of length | — |

### `tendons[].draw_in` — TendonDrawIn

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `begin` | float | `0.0` | 起点端锚具回缩（长度单位） | the anchorage's draw-in at the begin (length unit) | — |
| `end` | float | `0.0` | 终点端锚具回缩（长度单位） | the anchorage's draw-in at the end (length unit) | — |

### `tendons[].relaxation` — TendonRelaxation

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `code` | `jtg3362` | `jtg3362` | 按哪本规范算松弛（`vfife.codes`）：目前 `jtg3362` | the code whose loss it follows (`vfife.codes`): `jtg3362` for now | — |
| `fpk` | float | **必填** | 抗拉强度标准值 fpk（应力单位） | the characteristic strength fpk (stress unit) | — |
| `zeta` | float | **必填** | 规范的松弛系数 ζ | the code's relaxation coefficient zeta | — |
| `overstressed` | bool | `False` | 超张拉：ψ 取 0.9（否则 1.0） | over-stressed: psi 0.9 instead of 1.0 | — |

### `constraints[]` — ConstraintSpec

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `region` | str | **必填** | 约束作用的节点 | the nodes the constraint applies to | 60 个（01, 02, 03, 04 …） |
| `dofs` | list[int] | **必填** | 1 = 约束/强制，0 = 自由，顺序 [ux, uy, uz, rx, ry, rz]；只有平动的模型给三个 | 1 = held/prescribed, 0 = free, in the order [ux, uy, uz, rx, ry, rz]; three entries on a translation-only model | 60 个（01, 02, 03, 04 …） |
| `value` | any | — | 强制位移；标量或三分量向量。不给 = 固定为零 | prescribed displacement; scalar or 3-vector. None = held at zero. | 29b, 29e |
| `motion` | `displacement` \| `velocity` \| `acceleration` | `displacement` | 强制的运动量 `value × amplitude(progress)`：位移、速度或加速度；后两者只用于动力分析 | Kinematic quantity prescribed by `value * amplitude(progress)`. Velocity and acceleration are dynamic-analysis conditions. | — |
| `keep` | bool | `False` | 只用于步内约束（1.2）：以后各步继续保持（解除见步的 `release`）；不写则只在本步有效 | a step's constraint only (1.2): it holds in every later step, until a step's `release` frees its nodes; without it a step's constraint is that step's alone | — |
| `amplitude` | str | — | 幅值曲线名；默认：分级松弛的步里 `ramp`（一步内 0 到 1），连续松弛与动力分析里 `ramp_hold`（前段升到 1 后保持），后续步里的模型级荷载和带 `to_age` 的时间步里的一切 `hold` | name of an amplitude curve; by default `ramp` (0 to 1 over the step) in a staged relaxation, `ramp_hold` (up to 1, then held) in a continuous relaxation or a dynamic run, and `hold` for a model-level load in a later step and for everything in a time step (`to_age`) | — |
| `record` | 见 RecordMotionSpec | — | 支承自己的地面运动记录（多点激励）：被标记的平动自由度按记录随时间运动 | the support's own ground-motion record (multiple-support excitation): the flagged translational dofs follow it in time | 36 |

### `constraints[].record` — RecordMotionSpec

A support's own ground motion: the flagged translational dofs of the constraint's region follow a record in time, in ABSOLUTE coordinates -- the multiple-support form, in which every support can move differently and the members between them are strained by the difference.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `path` | str | **必填** | 两列记录（时间 s、数值），相对算例文件 | a two-column record (time s, value), relative to the case file | 36 |
| `quantity` | `displacement` \| `velocity` \| `acceleration` | `displacement` | 记录的量：`displacement`（推荐：多点激励要的是位移，且已做基线校正）、`velocity` 或 `acceleration` | what the record is: `displacement` (recommended -- a multiple-support run needs the displacement, baseline-corrected), `velocity` or `acceleration` | 36 |
| `scale` | float | `1.0` | 乘在记录上的系数（记录以 g 给出时写 9.80665） | multiplies the record (9.80665 for an acceleration in g) | — |
| `delay` | float | `0.0` | 这个支承收到记录的时延（s） | how late this support receives the record (s) | — |
| `wave` | 见 WaveSpec | — | 行波：区域里每个节点按位置再加时延 | a travelling wave: each node of the region gets a further delay from its position | 36 |

### `constraints[].record.wave` — WaveSpec

A travelling wave: each node of the region gets the record late by (x - origin) . direction / speed.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `speed` | float | **必填** | 视波速（m/s） | the apparent wave speed (m/s) | 36 |
| `direction` | list[float] | **必填** | 传播方向（整体坐标，自动归一化） | the direction the wave travels in (global, normalised) | 36 |
| `origin` | list[float] | `[0.0, 0.0, 0.0]` | 零时延的点 | the point the record arrives at first (zero delay) | 36 |

### `loads[]` — LoadSpec

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `region` | str | `all` | 荷载作用的节点、面（traction/pressure）、体（body_force/gravity）或构件（line_load/moving_load） | the nodes, faces (traction/pressure), volume (body_force/gravity) or members (line_load/moving_load) the load acts on | 43 个（01, 02, 03, 04 …） |
| `keep` | bool | `False` | 只用于步内荷载（1.2）：以后各步继续作用，取本步结束时的值；不写则只在本步有效 | a step's load only (1.2): it stays on in every later step at the value it ended this one with; without it a step's load is that step's alone | — |
| `vector` | list[float] | — | 默认是区域上的**总**力，分摊到各节点；`per_node` 时是每个节点上的力 | by default the TOTAL force on the region, spread over its nodes; with per_node it is the force applied at each node instead | 20 个（01, 02, 03, 06 …） |
| `moment` | list[float] | — | 总力矩（N·m）：梁节点上的力偶，或实体表面上按 Mz/I 分布的弯曲面力（可随动） | Total moment (N m): a couple on the nodes of a beam2 model, or on a solid FACE a couple about M's axis with a potential -- in the reference configuration the linear bending traction M z / I (and, for the component along the normal, Saint-Venant's torsion shear) as nodal forces; see core/mesh.FaceMoment. | — |
| `line_load` | list[float] | — | 沿构件单位长度的均布力（N/m），一致地化为梁端力和力矩；默认全局分量，`coordinate_system: local` 时为局部 | Uniform force per unit member length (N/m), consistently converted to beam end forces and moments. Components are global unless `coordinate_system: local` is selected. | 24, 24c, 24d, 24e, 24f, 24g, 28d |
| `coordinate_system` | `global` \| `local` | `global` | line_load：`local` 用构件自身坐标（x 沿构件）给分量 | line_load: `local` gives the components in the member's own axes (x along it) | — |
| `with_mass` | bool | `False` | line_load：这个荷载是某样东西的重量（二期恒载、铺装、栏杆），也带上它的质量 \|q\|/g，按构件长度分到两端节点、三个平动方向；这份质量不再称重（重量就是这个荷载本身），地震、模态和反应谱里有惯性。g 为 9.80665 m/s²，按声明的单位换算。只能写在模型级的 `loads` 里（1.1） | line_load: this load is the weight of something (superimposed dead load, surfacing, parapets), so carry its mass \|q\|/g too, lumped to the members' end nodes in all three translations. The mass weighs nothing (the load IS its weight), so `gravity` does not count it twice; it gives the load inertia in a dynamic run, a modal analysis and a response spectrum. g is 9.80665 m/s² in the declared units. Model-level `loads` only (1.1) | — |
| `traction` | list[float] | — | 常面力（N/m²），在区域的外表面上积分；与 `vector` 不同，不随表面节点密度变 | constant traction (N/m²) integrated over exterior faces in `region`. Unlike `vector`, this is independent of surface-node density. | — |
| `pressure` | float | — | 正标量压力（Pa），沿外法线向内作用 | positive scalar pressure (Pa), acting inward along the exterior normal. | 04, 10, 19 |
| `follower` | bool | `False` | 压力：每步在当前位置的面上重新积分，始终垂直于表面；面力矩：力偶随面转动并保持大小。默认按参考构形 | pressure: re-integrate it over the faces at their CURRENT positions every step, so it stays normal to the surface it acts on. Face moment: the couple turns with the face and keeps its magnitude through any rotation (the potential -\|M\| theta of the face's angle about M's axis); without it the reference distribution is a dead load whose couple decays with the cosine of the rotation. The default keeps the reference geometry. Solid elements, numpy/numba. | 19 |
| `body_force` | list[float] | — | 常体力密度（N/m³），在体上积分 | constant force density (N/m³), consistently integrated over volume. | — |
| `gravity` | list[float] | — | 重力加速度（m/s²），在所选体上化为 ρg | acceleration (m/s²); converted to rho * gravity in the selected volume. | 15, 22, 25, 25b, 30, 31, 32, 33 |
| `temperature` | 见 TemperatureSpec | — | 截面上的温度变化（见 TemperatureSpec） | A temperature change over the section (see TemperatureSpec). | 27, 27b, 28b, 28c |
| `moving_load` | 见 MovingLoadSpec | — | 沿区域构件行驶的轴重组（见 MovingLoadSpec） | Axles travelling along the members of `region` (see MovingLoadSpec). | 26, 26b |
| `vehicle` | 见 VehicleSpec | — | 有质量、悬挂和轮胎的车辆（见 VehicleSpec） | a vehicle with mass, suspension and tyres (see VehicleSpec) | 38 |
| `ground_motion` | dict[str, any] | — | 整个模型的一致地面加速度（动力；框架、纤维梁、桁架、壳、实体）：`path` 两列（秒，加速度），`scale` 换算到算例的加速度单位（SI 下记录以 g 给出写 9.80665），`direction` 给激振方向；只收这三个键 | Uniform base acceleration of the whole model in a dynamic run (frames, fibre beams, trusses, shells, solids): `path` holds two columns (seconds, acceleration), `scale` brings it to the case's acceleration unit (9.80665 for a record in g in an SI case) and `direction` gives the excitation axis. Only these three keys; no amplitude, region `all`. | 05 |
| `per_node` | bool | `False` | `vector`/`moment` 作为**每个**节点上的力，而不是区域上的总和 | `vector`/`moment` as the force at EACH node rather than the total over the region | — |
| `amplitude` | str | — | `amplitudes` 里的名字，或 `hold`（从第一步起全量）；默认：分级松弛 `ramp`（一步内 0 到 1），连续松弛与动力 `ramp_hold`，后续步里的模型级荷载和带 `to_age` 的时间步里的荷载 `hold` | name in `amplitudes`, or `hold` (full from the first step); by default `ramp` (0 to 1 over the step) in a staged relaxation, `ramp_hold` in a continuous relaxation or a dynamic run, and `hold` for a model-level load in a later step and for a load in a time step (`to_age`) | 19, 24c, 24d, 24g, 32, 33, 34 |

### `loads[].temperature` — TemperatureSpec

A temperature change over a member's section.  Temperature is not a force. It is a FREE STRAIN the section would take on if nothing held it: eps_th(y, z) = alpha * T(y, z). What that produces depends entirely on what is in the way --    * nothing in the way (a determinate member, a linear T): the member     simply changes shape. It elongates by alpha dT L, it bows to a     curvature alpha dT/dy, and it carries NO stress at all.   * something in the way (a restrained end, a redundant support): the part     of the free strain the structure refuses becomes force -- an axial     N = EA alpha dT, a continuous beam's secondary moments.   * a NONLINEAR T through the depth (which is what a bridge code's design     profile is): even a simply supported beam cannot take that shape,     because plane sections must stay plane. The section keeps the linear     part as elongation and curvature and turns the rest into     self-equilibrated stresses -- zero net force, zero net moment, and     real stress. Only a fibre section can show them: a `profile` on a     `beam2` / `beam2_timoshenko` section (shaped or `general`) is refused,     not reduced -- reducing it needs the area distribution a beam section     does not carry.  The field is `uniform` + `gradient`, or a `profile` sampled through the section. `gradient` is K per metre of section depth rather than "top minus bottom", because a `general` beam section states area and second moments and has no depth to divide by.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `uniform` | float | `0.0` | 整体温差（K），正 = 比无应力状态热 | K, the whole section (positive = warmer than the stress-free state) | 27, 28b, 28c |
| `gradient` | any | — | [dT/dy, dT/dz]（K/m），沿截面自身坐标；给标量即沿 `axis`（默认 y）的梯度 | [dT/dy, dT/dz] in K/m along the section's own axes; a scalar is the gradient along `axis` (y by default: [g, 0]; `axis: z` gives [0, g]). | 27, 28b |
| `profile` | list[list[float]] | — | [[s, T], ...] 沿截面高度的温度分布（m，K），点间线性、两端外常数——规范的设计温度分布 | [[s, T], ...] through the section (m, K), linear between the points and constant beyond the ends -- a code's design profile written literally. Mutually exclusive with uniform/gradient. | 27b |
| `axis` | `y` \| `z` | `y` | profile 或标量 gradient 沿截面哪个轴：`rect`/`box` 的高度在 z，`patches` 在角点坐标所放的那个轴 | which of the section's own axes the profile, or a scalar gradient, runs along. A `rect`/`box` has its depth on z (width along y); a `patches` section on whichever axis its corners put it -- so a deck's vertical profile names the axis rather than assuming one. Refused with a two-component gradient, which names its axes already. | 27b |

### `loads[].moving_load` — MovingLoadSpec

A group of axles travelling along the members of a region.  The region's members must form one open chain; `s` is arc length along it from the end that comes first in (x, y, z), so a bridge laid out along +x is entered at its left end by a positive `speed`.  `axles` are `[offset, force]` pairs: `offset` is the axle's position along the path RELATIVE to the group's reference point (so a trailing axle 4.2 m behind a lead axle is -4.2, whichever way the group travels), and `force` is its magnitude along `direction`, which defaults to straight down. An axle whose station is off the path carries nothing, which is how a vehicle drives on at one end and off at the other.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `axles` | list[list[float]] | **必填** | [[沿路径的偏移 (m), 轴重 (N)], ...]；偏移沿**行进方向**量，后轴为负 | [[offset along the path (m), force (N)], ...] | 26, 26b |
| `start` | float | `0.0` | 步开始时参考点所在的里程 | station of the group's reference point at the start of the step | 26, 26b |
| `speed` | float | — | 沿路径的速度（m/s）：s = start + speed·t，动力过桥 | m/s along the path: s = start + speed * t. Dynamic crossings. | 26 |
| `stop` | float | — | 代替速度：在步的进度内从 `start` 扫到 `stop`——配 `load_steps: n` 就是 n 个静力位置，即影响线 | instead of a speed, sweep s from `start` to `stop` over the step's progress -- with `load_steps: n`, n static positions, an influence line. | 26b |
| `direction` | list[float] | `[0.0, 0.0, -1.0]` | 力的方向（单位化）；默认 −z | unit vector the force pushes along (it is normalised); default is -z | — |

### `loads[].vehicle` — VehicleSpec

A vehicle with mass, suspension and tyres crossing the members of the region: a rigid body (heave, and pitch with more than one axle) on per-axle suspensions, unsprung masses and tyres that touch the deck plus a roughness profile. Dynamic only; the same carriageway rules as `moving_load`.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `name` | str | — | 名字（监测器用 `of` 引用它）；不写则每一步里依次为 vehicle、vehicle_2……，同一步里不得重名 | its name (a monitor's `of`); unnamed ones are vehicle, vehicle_2, ... in each step, and two in one step may not share one | 38 |
| `body` | 见 VehicleBodySpec | **必填** | 车身 | the body | 38 |
| `axles` | list[见 VehicleAxleSpec] | **必填** | 车轴 | the axles | 38 |
| `speed` | float | **必填** | 沿路径的速度（m/s） | speed along the path (m/s) | 38 |
| `start` | float | `0.0` | 步开始时参考点的里程（负值在桥外的引道上） | station of the reference point at the start of the step (negative: on the approach) | 38 |
| `direction` | list[float] | `[0.0, 0.0, -1.0]` | 重力方向（车辆向下压的方向） | the direction gravity pulls (the way the vehicle presses) | — |
| `gravity` | float | `9.80665` | 重力加速度：默认 9.80665 m/s²，声明了非 SI 的 `units` 时换算（mm 下为 9806.65；1.0 版前一律 9.80665） | gravitational acceleration: 9.80665 m/s^2 by default, converted to the declared `units` when they are not SI (9806.65 in mm; it was 9.80665 whatever the units before v1.0) | — |
| `roughness` | 见 RoughnessSpec | — | 路面不平度 | road roughness | — |

### `loads[].vehicle.body` — VehicleBodySpec

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `mass` | float | **必填** | 簧上质量（kg） | sprung mass (kg) | 38 |
| `pitch_inertia` | float | — | 俯仰转动惯量（kg·m²），两轴以上必填 | pitch inertia (kg m^2); needed with more than one axle | — |
| `cg` | float | `0.0` | 质心相对参考点的偏移（m，沿行进方向） | offset of the centre of mass from the reference point (m, along travel) | — |

### `loads[].vehicle.axles[]` — VehicleAxleSpec

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `offset` | float | **必填** | 沿行进方向相对参考点的偏移（m），后轴为负 | offset along the direction of travel from the reference point (m); a trailing axle is negative | 38 |
| `unsprung_mass` | float | — | 簧下质量（kg）：车轮和车轴；有轮胎时必填，无轮胎的刚性轴不给 | unsprung mass (kg): the wheel and the axle; with a tyre only | — |
| `suspension` | 见 SpringDashpotSpec | **必填** | 悬挂 | the suspension, between the body and the axle | 38 |
| `tyre` | 见 SpringDashpotSpec | — | 轮胎，车轴与路面之间，只压不拉；不给则车轴刚性贴着路面（文献里的"移动簧上质量"） | the tyre, between the axle and the road; it pushes and never pulls. Without one the axle rides the road rigidly (the literature's moving sprung mass) | — |

### `loads[].vehicle.axles[].suspension` — SpringDashpotSpec

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `stiffness` | float | **必填** | 刚度（N/m） | stiffness (N/m) | 38 |
| `damping` | float | `0.0` | 阻尼（N·s/m） | damping (N s/m) | — |

### `loads[].vehicle.roughness` — RoughnessSpec

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `profile` | str | — | 两列文件（里程 m，高程 m），线性插值，范围外为零 | a two-column file (station m, elevation m), linear, zero outside | — |
| `scale` | float | `1.0` | 高程的乘数（profile 或 psd 都乘） | multiplies the elevation (of a profile or a psd) | — |
| `psd` | 见 RoughnessPsdSpec | — | 位移功率谱 G_d(n) = g0 (n/n0)^-w（见 RoughnessPsdSpec） | a displacement PSD G_d(n) = g0 (n/n0)^-w (see RoughnessPsdSpec) | — |

### `loads[].vehicle.roughness.psd` — RoughnessPsdSpec

The displacement PSD G_d(n) = g0 (n/n0)^-w of a road, sampled as a sum of `count` cosines with random phases. Its own block so a misspelt key is refused (v1.0: `sead` and `nmax` were read as nothing).

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `g0` | float | **必填** | n0 处的功率谱密度（m³），必填；等级表不在库里 | the PSD at n0 (m^3); required. No class table lives here | — |
| `n0` | float | `0.1` | 参考空间频率（周/长度单位）（默认值按 SI 给出，声明了非 SI 的 `units` 时换算到声明的单位） | reference spatial frequency (cycles per length unit) (the default is SI, converted to the declared `units` when they are not SI) | — |
| `w` | float | `2.0` | 频率指数 | the frequency exponent | — |
| `n_min` | float | `0.01` | 最低空间频率（周/长度单位）（默认值按 SI 给出，声明了非 SI 的 `units` 时换算到声明的单位） | lowest spatial frequency (cycles per length unit) (the default is SI, converted to the declared `units` when they are not SI) | — |
| `n_max` | float | `3.0` | 最高空间频率（周/长度单位）（默认值按 SI 给出，声明了非 SI 的 `units` 时换算到声明的单位） | highest spatial frequency (cycles per length unit) (the default is SI, converted to the declared `units` when they are not SI) | — |
| `count` | int | `400` | 余弦项数 | number of cosines | — |
| `seed` | int | `0` | 相位的随机种子 | seed of the random phases | — |

### `nodal_masses[]` — NodalMassSpec

Lumped mass attached to nodes in a named region.  Isotropic and weighing by default, as before v1.0. `components` makes it directional (added mass: a fluid's, or a deck's along one axis), `weight: false` keeps it out of `gravity`, `inertia` adds a rotary inertia about the global axes on the six-dof families.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `region` | str | `all` | 带质量的节点 | the nodes carrying the mass | 05, 36, 37 |
| `mass` | float | `0.0` | 区域上的总质量（kg），`per_node` 时为每个节点的质量 | kg total over the region, or kg at every node with `per_node`. | 05, 36, 37 |
| `per_node` | bool | `False` | `mass` 是每个节点的，而不是区域总量分摊 | `mass` at each node rather than the total spread over the region | 05 |
| `components` | list[int] | `[1, 1, 1]` | 质量作用在哪些平动分量上 [x, y, z]；默认三向 | which translations carry the mass, [x, y, z]; all three by default | — |
| `weight` | bool | — | 重力是否看见它；默认三向质量有重量、定向质量（附加质量）没有 | whether gravity sees it; by default a mass on all three translations weighs and a directional (added) one does not | — |
| `inertia` | list[float] | — | 绕整体轴的转动惯量（kg·m²，区域总量或 `per_node` 每节点），只用于六自由度的梁、壳 | rotary inertia about the global axes (kg m^2, over the region or per node with per_node); six-dof families only | — |

### `hydrodynamic_mass[]` — HydrodynamicMassSpec

The Morison added mass of members under water: C_a rho_w A per unit length of the submerged part, across the member and not along it, weighing nothing. For piers and piles: the members must lie within 5 degrees of a global axis (a lumped mass is diagonal).

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `region` | str | **必填** | 构件区域（两端都在区域内的构件） | the members (both ends in the region) | 37 |
| `diameter` | float | — | 圆截面外径（m）；或给 `area` | outer diameter of a round section (m); or give `area` | 37 |
| `area` | float | — | 排开水的截面积（m²），非圆截面用 | the displaced cross-section (m^2), for a section that is not round | — |
| `coefficient` | float | `1.0` | 附加质量系数 C_a = C_M − 1；圆柱 1.0 | added-mass coefficient C_a = C_M - 1; 1.0 for a circular cylinder | 37 |
| `density` | float | `1025.0` | 水的密度：默认 1025 kg/m³，声明了非 SI 的 `units` 时换算（mm/N/s 下为 1.025e-9 t/mm³） | water density: 1025 kg/m^3 by default, converted to the declared `units` when they are not SI (1.025e-9 t/mm^3 in mm/N/s) | 37 |
| `water_level` | float | `0.0` | 水面高程：竖向坐标低于它的构件部分计入 | the free surface: the part of a member below it counts | 37 |
| `vertical` | `x` \| `y` \| `z` | `z` | 竖直轴 | the vertical axis | — |

### `soil_springs[]` — SoilSpringSpec

Soil springs on a pile: p-y curves by depth, nonlinear elastic (1.2). Each node of the region below the ground gets a horizontal spring of the curve at its depth times the pile length it stands for.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `region` | str | **必填** | 桩的构件区域（两端都在区域内的构件） | the pile's members (both ends in the region) | — |
| `ground` | float | `0.0` | 地面高程：沿 `vertical` 的坐标，其下的桩段受土弹簧 | the ground surface, as a coordinate on `vertical`: the pile below it carries springs | — |
| `vertical` | `x` \| `y` \| `z` | `z` | 竖直轴 | the vertical axis | — |
| `direction` | `horizontal` \| `x` \| `y` \| `z` | `horizontal` | `horizontal`：水平面内径向（按两个水平位移的合量取抗力）；`x`、`y`（或 `z`）：只沿一个轴，平面模型用 | `horizontal`: radial in the horizontal plane (the resistance follows the resultant of the two horizontal displacements); `x`, `y` (or `z`): one axis only, for a plane model | — |
| `curves` | list[见 PYCurveSpec] | **必填** | 各深度的 p–y 曲线；深度之间按同一 y 线性插值，浅于第一条取第一条，深于最后一条取最后一条 | the p-y curves by depth: between two depths interpolated at the same y; above the first the first, below the last the last | — |

### `soil_springs[].curves[]` — PYCurveSpec

One p-y curve, at one depth (1.2).

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `depth` | float | **必填** | 深度：地面以下的距离，沿 `vertical` 向下量 | depth below the ground surface, measured down along `vertical` | — |
| `p_y` | list[list[float]] | **必填** | [[y, p], ...]：桩的水平位移 y 与单位桩长的土抗力 p（按算例的单位，SI 下 m 与 N/m）；从 [0, 0] 起，y 递增，第二点 p > 0；最后一点之后 p 不再增加 | [[y, p], ...]: the pile's horizontal displacement y and the soil's resistance per unit length of pile p (in the case's units: m and N/m in SI); from [0, 0], y increasing, p > 0 at the second point; past the last point p stays at its last value | — |

### `diaphragms[]` — DiaphragmSpec

A floor that is rigid in its own plane.  The nodes of `region` move together as one rigid body in the plane normal to `normal`: two translations and the rotation about `normal`. Everything else -- the out-of-plane translation and the two bending rotations -- stays free, which is what makes this a diaphragm and not a rigid body.  It is how a building's floor slab is written down when the slab itself is not meshed, and it is what an OpenSees model means by `rigidDiaphragm`.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `region` | str | **必填** | 绑在一起的节点 | the nodes tied together | — |
| `normal` | `x` \| `y` \| `z` | `z` | 平面法向：节点在垂直于它的平面内作刚体运动 | the plane's normal: the nodes move as a rigid body within the plane normal to it | — |

### `steps[]` — StepSpec

One analysis step. Steps run in sequence and share the model state.  The model-level `constraints` and `loads` are the base and apply in every step; these are added to them, and where both prescribe the same (node, dof) the step wins.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `name` | str | — | 显示在运行日志里，并随该步结果保存 | shown in the run log and stored with the step's results | 34 个（06, 11, 12, 12b …） |
| `constraints` | list[见 ConstraintSpec] | 空 | 本步的约束，叠加在模型级约束上（冲突时本步优先） | this step's constraints, added to the model-level ones (the step wins on a conflict) | 15 个（06, 11, 12, 12b …） |
| `loads` | list[见 LoadSpec] | 空 | 本步的荷载，叠加在模型级荷载上 | this step's loads, added to the model-level ones | 10 个（13, 14, 18, 20 …） |
| `n_steps` | int | — | 逐步覆盖求解器里「做多少」的设置；mode、mass、backend、damping 属于模型，不能逐步改 | per-step overrides of the solver settings that describe HOW MUCH to do; mode, mass, backend and damping are properties of the model, not the step | 06 |
| `load_steps` | int | — | 松弛：本步内分这么多荷载级，逐级保持到平衡 | relaxation: hold this many load levels to equilibrium within the step | 33 个（06, 11, 12, 12b …） |
| `max_steps_per_load` | int | — | 松弛：每级的迭代预算 | relaxation: iteration budget per load level | 25 个（06, 11, 12, 12b …） |
| `stage_ramp_iterations` | int | — | 每个新荷载级渐入所用的迭代数（默认 max_steps_per_load/5） | iterations over which each new load level is blended in (default max_steps_per_load / 5) | — |
| `tolerance` | float | — | 本步的残差容差 | residual tolerance for this step | 11 个（06, 12, 12b, 12c …） |
| `age` | float | — | 本步起点的混凝土龄期（天）；默认接上一步结束处 | The step's time window in days of concrete age: `age` defaults to where the last step left off, and `to_age` makes the step a stretch of TIME whose `load_steps` are its time stations -- log-spaced, because that is the axis creep happens on. | 28d, 29, 29b, 29d, 29e, 29f, 29g |
| `to_age` | float | — | 本步终点的混凝土龄期（天）：给了本步就是一段时间，`load_steps` 是时间站 | concrete age in days at the end of the step: makes it a stretch of time | 28d, 29, 29b, 29c, 29d, 29e, 29f, 29g |
| `time_spacing` | `log` \| `linear` | `log` | 时间站的分布：`log`（默认，徐变的轴）或 `linear` | `log` (default, the creep axis) or `linear` spacing of the time stations | — |
| `procedure` | `relaxation` \| `dynamic` \| `static` | — | 本步的分析类型（1.2）：relaxation、dynamic 或 static；不写则用 solver.mode。线单元模型（beam、fiber、mixed_line） | this step's procedure (1.2): relaxation, dynamic or static; left out, solver.mode's. Line models (beam, fibre, mixed_line) | — |
| `total_time` | float | — | 动力步的时长（s）；不写则用 solver.total_time | a dynamic step's duration (s); left out, solver.total_time | — |
| `activate` | list[str] | 空 | 本步出生的构件（所有节点都在区域内的）：在当时的变形位置无应力出生，自重与线荷载在本步内加上 | members born in this step (every node inside the region): stress-free where the structure has moved to, their weight and line loads coming on over the step | — |
| `birth` | `tangent` \| `design` | `tangent` | 只连着出生构件的节点放在哪里：tangent（默认，随相邻已有节点做刚体平移与转动——悬臂施工的切线安装）或 design（留在设计坐标） | where the nodes that only the born members reach are put: tangent (default: carried rigidly by the neighbouring node that already stands -- a cantilever's tangential installation) or design (left at their design coordinates) | — |
| `activate_age` | float | — | 出生构件的混凝土龄期（天）；不写则用 solver.age | the concrete age (days) of the members born in this step; left out, solver.age | — |
| `deactivate` | list[str] | 空 | 本步移除的构件：它们对其余结构的作用力（含自重）在本步内卸到零 | members removed in this step: what they did to the rest of the structure (their weight included) is taken off over the step | — |
| `release` | list[str] | 空 | 解除这些区域节点上继承下来的约束（模型级与 keep 的）：支反力在本步内卸到零；本步自己的约束照常施加。也可以写主从刚性连接的 `name`（1.21）：连接拆除，它的约束力在本步内卸到零 | frees the inherited constraints (model-level and kept ones) on these regions' nodes: the reaction is taken off over the step; the step's own constraints still apply. A rigid link's `name` frees that link (1.21): the force it carried is taken off over the step | — |
| `rigid_links` | list[见 RigidLinkSpec] | 空 | 本步开始时建立的主从刚性连接（1.21），按这时节点所在的位置和转角建立，建立时不受力；写法同模型级的 `rigid_links` | master-slave rigid links made at this step's start (1.21), where their nodes then stand and free of force; written as the model's `rigid_links` | — |

### `steps[].rigid_links[]` — RigidLinkSpec

A master-slave rigid link (1.21): the slaves' tied components follow the master as a rigid body does, in large rotation (core/constraints). Midas' *RIGIDLINK, OpenSees' rigidLink and (with `arm: false`) equalDOF.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `master` | str | **必填** | 主节点：恰好一个节点的区域 | the master: a region of exactly one node | — |
| `slaves` | str | **必填** | 从节点：一个或多个节点的区域 | the slaves: a region of one or more nodes | — |
| `dofs` | list[int] | — | 跟着主节点的分量，整体轴的 ux uy uz rx ry rz，1 为跟随（Midas 的 6 位标记） | the components that follow the master, in the global axes ux uy uz rx ry rz, 1 = tied (Midas' six flags) | — |
| `arm` | bool | `True` | true：平动按主从之间的力臂随主节点转动（rigidLink）；false：从节点平动等于主节点平动（equalDOF） | true: the translations carry the arm between master and slave, turning with the master (rigidLink); false: the slave's translation is the master's (equalDOF) | — |
| `name` | str | — | 名字（报错和输出里用） | a name (for messages) | — |

### `removals[]` — RemovalSpec

A member taken out at a given time: the members wholly in `region` stop carrying load at `time` seconds after the run starts -- a cut cable, a burnt hanger. Dynamic runs of any line model (mixed_line, beam2, beam2_timoshenko, truss2, beam2_fiber).

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `region` | str | **必填** | 要移除的构件（所有节点都在区域内的） | the members (every node inside the region) to remove | 33 |
| `split` | bool | — | 移除后是否质点分裂（同 FailureSpec.split） | whether the removed member splits off onto its own particles (as FailureSpec.split) | 33 |
| `time` | float | **必填** | 它们停止传力的时刻（s） | the instant (s) they stop carrying load | 33 |

### `contact` — ContactSpec

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `rigid` | list[见 RigidSurfaceSpec] | 空 | 刚性面列表（地面、海底、承台顶面……） | the rigid surfaces (ground, sea floor, a pile cap ...) | 33 |
| `members` | 见 MemberContactSpec | — | 线单元构件之间的接触（1.11）：落梁、梁体碰撞、倒塌堆积 | contact between line members (1.11): a falling span, pounding girders, the pile a collapse leaves | — |

### `contact.rigid[]` — RigidSurfaceSpec

One rigid, immovable surface the nodes of `region` cannot pass through: a penalty spring on the penetration along the normal, and Coulomb friction against the tangential velocity in a dynamic run. core/rigid_contact.py has the mechanics and the sizing of `auto`.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `plane` | 见 PlaneSpec | **必填** | 平面：一点加法向；法向指向允许的一侧 | the plane, a point and the normal; the normal points to the allowed side | 33 |
| `region` | str | `all` | 哪些节点受它约束（默认全部） | the nodes it holds back (default: all) | 33 |
| `stiffness` | `auto` \| float | `auto` | 罚刚度（N/m）：`auto` 按节点质量和步长取，不缩小稳定步长（core/rigid_contact.py） | penalty stiffness (N/m): `auto` sizes it per node from its mass and the step so the step never shrinks for it | — |
| `friction` | float | `0.0` | 库仑摩擦系数（只用于动力计算；松弛拒绝） | Coulomb friction coefficient (dynamic runs only; refused in relaxation) | — |
| `damping` | float | `0.0` | 接触阻尼比 ζ（只用于动力计算）：节点在面内时并联一个阻尼器 2ζ√(km)，碰撞不再完全弹性，碎片会停下来；0.5 时恢复系数约 0.16 | contact damping ratio zeta (dynamic only): a dashpot 2 zeta sqrt(k m) while the node is inside, so an impact loses energy and debris settles; restitution about 0.16 at 0.5 | 33 |
| `name` | str | — | 名字，用于报错和摘要 | a name for messages and the summary | 33 |

### `contact.rigid[].plane` — PlaneSpec

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `point` | list[float] | **必填** | 平面上一点 [x, y, z] | a point on the plane | 33 |
| `normal` | list[float] | **必填** | 法向 [nx, ny, nz]，指向结构所在的一侧（允许的一侧）；自动单位化 | the normal, pointing to the side the structure lives on (the allowed side); normalised | 33 |

### `contact.members` — MemberContactSpec

Contact between line members (1.11): each member a capsule about its axis; two that overlap push each other apart, with a dashpot and Coulomb friction in a dynamic step. Pairs that overlap in the reference configuration keep that overlap as an allowance until they have separated. core/member_contact.py has the mechanics.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `bodies` | list[见 ContactBodySpec] | **必填** | 参与接触的构件组，每组一个区域和一个半径；各组之间、组内（不共节点的构件）都检查 | the bodies, a region and a radius each; every member is tested against every other, its own body's too (not the ones it shares a point with) | — |
| `stiffness` | `auto` \| float | `auto` | 罚刚度（N/m）：`auto` 按两构件的折算质量和步长取，不缩小稳定步长；写数时以它为上限 | penalty stiffness: `auto` sizes it from the pair's reduced mass and the step so the step never shrinks for it; a number is capped there | — |
| `friction` | float | `0.0` | 库仑摩擦系数（只在动力步里起作用） | Coulomb friction coefficient (acts in dynamic steps only) | — |
| `damping` | float | `0.0` | 接触阻尼比 ζ（只在动力步里起作用）：碰撞不再完全弹性 | contact damping ratio zeta (dynamic steps only): an impact loses energy | — |
| `slack` | float | `0.05` | 初始就重叠的构件对（支座上下的梁与盖梁、节点处的相邻构件）在接触起作用之前还能再靠近多少：较小半径的倍数（默认 5%） | how much closer a pair that overlaps at the start (a girder and the cap under its bearing, neighbours at a joint) may come before the contact acts, as a fraction of the smaller of the two radii (default 5%) | — |

### `contact.members.bodies[]` — ContactBodySpec

A set of line members that take part in the contact between members, and the radius of the capsule each of them is (1.11).

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `region` | str | `all` | 参与接触的构件所在的区域（两端节点都在区域内的构件；连接器不参与） | the region whose members take part (a member is in when both its points are; connectors take no part) | — |
| `radius` | `auto` \| float | `auto` | 胶囊体半径（m）：构件表面到轴线的距离。`auto` 只用于杆和索（取与截面等面积的圆）；梁必须写 | the capsule's radius, the distance of the member's surface from its axis. `auto` is for bars and cables alone (the circle of their area); a beam must say | — |

### `response_spectrum` — ResponseSpectrumSpec

Response-spectrum analysis at the state the run ends in: the modes of the tangent there (as `vfife modal`), each mode's peak from the spectrum at its period, S_a / omega^2 times its participation, and the modal peaks combined -- per mode pair by CQC (Der Kiureghian, one damping ratio) or SRSS, then the directions by SRSS. Displacements, member end actions and support reactions are combined component by component; the peaks go to the result's `response_spectrum` group.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `spectra` | list[见 SpectrumSpec] | **必填** | 谱与方向（每个方向一张谱；多个方向的结果按 SRSS 组合） | the spectra and their directions (one per direction; directions combine by SRSS) | 35, 37 |
| `combination` | `cqc` \| `srss` | `cqc` | 振型组合：`cqc`（默认）或 `srss` | how the modes combine: `cqc` (default) or `srss` | 35, 37 |
| `damping` | float | `0.05` | CQC 用的振型阻尼比（谱本身应是按同一阻尼比给的） | the modal damping ratio CQC's correlation uses (the spectrum should be for the same ratio) | — |
| `modes` | int | — | 取多少阶；不写：取到每个受激方向的有效质量累计达到 `mass_fraction` | how many modes, lowest first; left out, as many as bring every excited direction to `mass_fraction` | 35, 37 |
| `mass_fraction` | float | `0.9` | 不写 `modes` 时要达到的有效质量比 | the effective-mass fraction `modes` left out has to reach | — |
| `beyond_table` | `refuse` \| `hold` | `refuse` | 周期超出谱表时：`refuse`（默认，报出是哪几阶）或 `hold`（取端值） | a period outside the table: `refuse` (default, naming the modes) or `hold` (the end value) | — |
| `max_dofs` | int | `6000` | 有限差分切线的自由度上限（2n 次内力计算） | refuse above this many dofs (the finite-difference tangent costs 2n force evaluations) | — |

### `response_spectrum.spectra[]` — SpectrumSpec

One design spectrum and the direction it acts in. The table is the user's: the library holds no code's spectrum.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `direction` | list[float] | **必填** | 地震作用的方向（整体坐标，自动归一化） | the direction the ground acceleration acts in (global, normalised) | 35, 37 |
| `table` | list[list[float]] | — | 谱表 [[周期 s, 加速度], ...]，周期递增；与 `path` 二选一 | the spectrum as [[period s, pseudo-acceleration], ...], periods increasing; or `path` | 35, 37 |
| `path` | str | — | 文本文件，前两列为周期、加速度（空格、制表符或逗号分隔，其余列不读），相对算例文件；与 `table` 二选一 | a text file with period and acceleration in its first two columns (spaces, tabs or commas; further columns are not read), relative to the case file; or `table` | — |
| `scale` | float | `1.0` | 谱值乘的系数（谱以 g 给出时写 9.80665） | multiplies the spectral values (9.80665 when the table is in g) | — |

### `solver` — SolverSpec

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `mode` | `dynamic` \| `relaxation` \| `static` | `relaxation` | `relaxation` 动力松弛求静力解，`static` 隐式静力（Newton），或 `dynamic` 显式时程 | `relaxation` (dynamic relaxation to a static state), `static` (implicit, Newton) or `dynamic` (explicit time history) | 61 个（01, 02, 03, 04 …） |
| `static` | 见 StaticSpec | （默认块） | 隐式静力的设置（见 StaticSpec） | the implicit static solver's settings (see StaticSpec) | 16b, 24g |
| `mass` | `physical` \| `underwood` \| `eigenvalue` | — | 松弛：`underwood`（参考刚度的 Gershgorin 行和，默认）或 `eigenvalue`（各单元最大特征值，大转动下仍成立）；动力：`physical` | relaxation: `underwood` (Gershgorin row sums of the reference stiffness, the default) or `eigenvalue` (each element's largest eigenvalue, a bound that survives large rotations -- see elements/continuum.stiffness_max_eigenvalues); dynamic: `physical` | 06, 12, 12b, 12c, 13, 14, 15 |
| `damping` | 见 DampingSpec | （默认块） | 见 DampingSpec；松弛默认 kinetic，动力默认无阻尼 | see DampingSpec; kinetic by default under relaxation, none under dynamic | 36 个（01, 02, 03, 04 …） |
| `total_time` | float | — | 动力：模拟时长（s），必填 | dynamic: simulated duration (s); required | 9 个（05, 19, 24c, 24d …） |
| `dt_safety` | float | `0.6` | 实际步长占稳定步长的比例 | fraction of the stable explicit step actually used | 05, 19, 36 |
| `target_dt` | float | — | 动力、连续体：均匀放大质量把步长抬到这个值（选择性的见 `mass_scaling`） | dynamic, continuum: raise the step to this by uniformly scaling the mass (see `mass_scaling` for the selective form) | — |
| `n_steps` | int | — | 松弛：迭代预算；每个分析步都自带预算时可省 | relaxation: iteration budget; may be left out when every step carries its own | 35 个（01, 02, 03, 04 …） |
| `tolerance` | float | `0.0001` | 松弛：残差与荷载之比低于它即收敛 | relaxation: converged when the residual-to-load ratio falls below this | 52 个（01, 03, 04, 06 …） |
| `kinetic_tolerance` | float | `1e-10` | 松弛：同时动能比要低于它 | relaxation: and the kinetic-energy ratio below this | 11 个（11, 21b, 22, 23 …） |
| `relaxation_safety` | float | `0.6` | 松弛：稳定步长的比例，同动力的 `dt_safety` | relaxation: fraction of the stable step, as `dt_safety` for dynamic | 11 个（03, 06, 12, 12b …） |
| `erosion` | 见 ErosionSpec | — | 单元删除；不写即不删 | omit entirely to disable element deletion | — |
| `load_steps` | int | — | 分级：每一级保持到平衡，每个输出帧都是荷载–位移曲线上的真实点 | hold each load level until equilibrium; every output frame is then a genuine point on the load-displacement curve | 12 个（04, 09, 10, 11 …） |
| `max_steps_per_load` | int | `5000` | 带 `load_steps` 的松弛：每级迭代预算 | relaxation with `load_steps`: iteration budget per level | 13 个（04, 09, 10, 11 …） |
| `stage_ramp_iterations` | int | — | 每级荷载渐入所用的迭代数（默认 max_steps_per_load/5） | iterations over which each load level is blended in (default max_steps_per_load / 5) | 04, 22, 23, 24e, 24f |
| `age` | float | — | 开始时的混凝土龄期（天）；只有材料声明了 creep/shrinkage/ageing 才读 | concrete age in days at the start of the run. Only a model whose material declares `creep`, `shrinkage` or `ageing` reads it. | 28d, 29, 29b, 29c, 29d, 29e, 29f, 29g |
| `mass_scaling` | 见 MassScalingSpec | — | 选择性质量缩放：只在 beam2、beam2_timoshenko、beam2_fiber、mixed_line 上；其他拒绝 | selective mass scaling: beam2, beam2_timoshenko, beam2_fiber and mixed_line only; refused elsewhere | — |
| `initial_state` | str | — | 从本模型早先一次结果的终态起步——位置、转角、材料状态，静止、t = 0。典型用法：先松弛得恒载状态，再从它做动力。相对算例文件；结果的网格必须是本算例的网格 | start from the final state of an earlier result of THIS model -- its positions, rotations and material state, at rest, at t = 0 -- instead of from the mesh. The typical pair: a relaxation run for the dead state, then a dynamic run from it. Relative to the case file; the result's mesh must be this case's mesh. | — |
| `rotation_iterations` | int | `2` | 实体单元：每步从变形梯度提取转动的极分解迭代次数；梁、壳、桁架不读它 | solid elements: polar-decomposition iterations that extract the rotation from the deformation gradient each step. Line, shell and truss models do not read it (it was documented as a beam2 setting) | — |
| `backend` | `numpy` \| `numba` \| `warp` | `numpy` | `numpy`（参考实现）、`numba`（编译内核，同样答案）或 `warp`（GPU） | 'numpy' (reference) or 'numba' (fused scalar kernel, same answer) | 61 个（01, 02, 03, 04 …） |
| `threads` | int | `1` | numba 纤维梁与框架内核的线程数：1 串行（默认），0 用全部核，n 用 n 个；结果与串行逐位相同 | threads for the numba fibre-beam and frame kernels: 1 serial (the default), 0 all cores, n that many; the answer is the serial one to the bit | — |
| `device` | str | `cuda` | 只对 warp："cuda"、"cuda:0"、"cpu" 等 | warp only: "cuda", "cuda:0", "cpu" ... | — |
| `precision` | `mixed` \| `fp64` | `mixed` | 只对 warp：`mixed`（单元 fp32、节点 fp64）或 `fp64`（验证用） | warp only: "mixed" (fp32 elements, fp64 nodal) or "fp64" (verification) | — |

### `solver.static` — StaticSpec

The implicit static solver's settings.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `arc_length` | 见 ArcLengthSpec | — | 弧长法：荷载因子成为未知量，能越过极限点 | arc length: the load factor becomes an unknown and the path passes limit points | 16b, 24g |
| `max_iterations` | int | `30` | 每个增量的 Newton 迭代上限 | Newton iterations per increment | — |
| `cutbacks` | int | `10` | 不收敛时增量减半的次数上限 | how many times an increment may be halved when it does not converge | — |
| `line_search` | bool | `True` | 回溯线搜索（只用于荷载控制的增量；弧长法的校正步没有线搜索） | backtracking line search on the residual norm (load-controlled increments only; the arc-length corrector has none) | — |
| `tangent` | `full` \| `modified` | `full` | `full`：每次迭代重算切线；`modified`：每个增量开头算一次，收敛变慢（残差没减半）时重算 | `full`: a new tangent every iteration; `modified`: once per increment, renewed when an iteration fails to halve the residual | — |
| `perturbation` | float | `1e-07` | 有限差分切线的步长：平动为它乘单元长度的中位数，转动为它乘 10（弧度） | the finite-difference step of the tangent: this times the median element length for a translation, ten times it (radians) for a rotation | — |
| `fallback` | `relaxation` \| `none` | `none` | Newton 用尽减半次数仍不收敛时怎么办（1.7）：`none`（默认）报极限点停下；`relaxation` 从上一个收敛状态起，这一级和本步余下的各级都改用动力松弛（虚拟质量、动能阻尼，与 `mode: relaxation` 同一算法；下一步再从 Newton 开始）——越过跳跃之后落在哪一支由这段虚拟的动力过程决定，软化截面上未必是 `mode: relaxation` 走的那支 | what happens when Newton has used up its cutbacks (1.7): `none` (the default) stops at the limit point; `relaxation` takes this level and the rest of the step by dynamic relaxation from the last converged state (fictitious mass, kinetic damping, the algorithm of `mode: relaxation`; the next step starts with Newton again) -- which branch it lands on past a jump is the fictitious transient's choice, and on a softening section not necessarily the one `mode: relaxation` follows | — |
| `fallback_iterations` | int | — | 回落松弛每级的迭代上限（默认 solver.max_steps_per_load） | the iteration budget of each relaxed level (default solver.max_steps_per_load) | — |

### `solver.static.arc_length` — ArcLengthSpec

Crisfield's cylindrical arc length: the load factor becomes an unknown, so the path can pass a limit point -- a pushover's peak, a snap-through, a snap-back.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `initial` | float | `0.1` | 第一个增量的荷载因子（据此定弧长） | the first increment's load factor (which sets the arc length) | 16b, 24g |
| `max_increments` | int | `200` | 每步增量数上限 | increments per step, at most | 16b, 24g |
| `target_iterations` | int | `5` | 每个增量期望的迭代次数：弧长按 sqrt(期望/实际) 伸缩 | iterations an increment should take: the arc scales by sqrt(desired / taken) | 16b, 24g |
| `stop` | 见 ArcStopSpec | — | 荷载因子到不了 1 时的停止条件 | where to stop when the load factor does not reach 1 | 24g |
| `steps` | list[str] | — | 用弧长法的分析步名；不给则所有步都用（其余步用荷载控制的 Newton） | the steps that use it, by name; all of them when left out (the others take load-controlled Newton) | 24g |
| `control` | 见 ArcControlSpec | — | 弧长只量这一个位移（1.3）：位移控制，荷载因子是未知量；被控位移不会往回走，跟不了它自己的回跳；不给则量整个状态 | measure the arc on this one displacement (1.3): displacement control with the load factor unknown; the controlled displacement never goes back, and cannot follow its own snap-back; the whole state's norm when left out | — |

### `solver.static.arc_length.stop` — ArcStopSpec

Where an arc-length step ends when the load factor never reaches 1: the mean displacement of a region along a direction reaching `value`.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `region` | str | **必填** | 节点区域 | the nodes | 24g |
| `dof` | `x` \| `y` \| `z` | `z` | 方向 | the direction | 24g |
| `value` | float | **必填** | 到这个位移（带符号）就停 | stop when it reaches this displacement (signed) | 24g |

### `solver.static.arc_length.control` — ArcControlSpec

The displacement an arc is measured on (1.3): one node's translation, instead of the norm of the whole state. Displacement control with the load factor as the unknown -- OpenSees' DisplacementControl integrator -- so the pushed point moves on at every increment and the path cannot turn back down an unloading branch; it cannot follow a snap-back of that displacement.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `region` | str | **必填** | 一个节点的区域 | a region of ONE node | — |
| `dof` | `x` \| `y` \| `z` | `z` | 方向 | the direction | — |

### `solver.damping` — DampingSpec

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `type` | `viscous` \| `kinetic` \| `rayleigh` \| `none` | — | 阻尼类型：viscous、kinetic、rayleigh、none。不给时按模式定——松弛 kinetic，动力 none（动力分析不要用 kinetic，它会抹掉动力响应） | NO DEFAULT HERE, DELIBERATELY, AND IT IS THE ONE FIELD THAT NEEDS NONE. The right default depends on the analysis mode -- kinetic for relaxation, undamped for dynamic -- which this file cannot see. It used to default to "kinetic", and because a pydantic model always emits its fields, that made the mode-dependent expression in SIX solvers dead code: every `mode: dynamic` run got kinetic damping, which zeroes the velocity at every kinetic-energy peak and so removes the dynamics it is applied to.  `None` here, resolved in the solver with `or` -- never with a `get` default, which a present-but-None key walks straight past. See `solver_setting` above for the general rule. | 36 个（01, 02, 03, 04 …） |
| `coeff` | float | `0.0` | viscous：单位质量的阻尼系数（1/s） | viscous: damping ratio-like coefficient per unit mass (1/s) | 19, 24c, 24d, 34 |
| `alpha` | float | `0.0` | 质量比例阻尼（1/s）。用于 `type: rayleigh`，以及松弛里与 `kinetic` 叠加（0.1 版为软化问题设计的组合）；`viscous`、`none` 下写了会被拒绝。软化问题上它会选择解的分支 | Mass-proportional damping, 1/s. Applied under `type: rayleigh`, and ALONGSIDE `type: kinetic` in a relaxation -- refused under `viscous` and `none`, where the frame solvers ignored it. `type: kinetic` with a small alpha is the combination it exists for: kinetic damping only fires when the kinetic energy peaks, so a mode whose energy grows monotonically -- a softening element -- is never touched by it, while alpha*M*v opposes velocity whatever the energy is doing. It is not free: on a softening problem it selects a solution branch (v0.1 has the measurements). | 05 |
| `beta` | float | `0.0` | 刚度比例阻尼（s）；只用于动力分析的 `type: rayleigh`，会缩小稳定步长 | stiffness-proportional damping, s; `type: rayleigh` in a dynamic analysis only; makes the stable step smaller | 05 |
| `ratio` | any | — | Rayleigh 的另一种写法：在 `frequencies` 的两个频率上各给阻尼比——一个数（两处相同）或 [ζ1, ζ2]；alpha、beta 由此算出 | Rayleigh stated by its damping ratio at the two `frequencies` (one number, or [zeta_1, zeta_2]); alpha and beta are computed from it | — |
| `frequencies` | list[float] | — | 给 `ratio` 的两个频率（Hz）：通常取第一阶和参与显著的最高阶 | the two frequencies (Hz) `ratio` is stated at: usually the first mode and the highest one that matters | — |

### `solver.erosion` — ErosionSpec

Element deletion, after Abaqus `ELEMENT DELETION` / LS-DYNA `ERODE`.  A softening model has no answer for a fully degraded element: it still has nodes and still carries whatever residual force they carry, but no stiffness to resist it.  Both commercial codes answer by removing the element.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `damage` | float | `0.99` | 损伤达到或超过它时积分点算失效（Abaqus 保留单元时 MAX DEGRADATION 为 0.99） | damage at or above which the point counts as failed (Abaqus MAX DEGRADATION is 0.99 when elements are kept, 1.0 when they are deleted; MAT_CSCM erodes at d > 0.99) | — |
| `measure` | `d` \| `dt` \| `dc` | `d` | 读哪个损伤变量：`d`（应力用的刚度退化）、`dt` 受拉、`dc` 受压 | which damage variable `damage` is read from: `d`, the stiffness degradation the stress carries (capped at the material's damage_max, so a threshold above the cap never fires); or `dt` / `dc`, the tensile and compressive variables on their own (CDP's DAMAGET / DAMAGEC, uncapped). `dc` is the spalling rule: crushed concrete goes, cracked concrete stays and keeps holding its bars -- on the Alberta beam a rule on `d` removed the cracked tension zone first. | — |
| `strain` | float | `0.05` | 同时必须超过的最大主 Green–Lagrange 应变，取单元中心 | max principal Green-Lagrange strain that must ALSO be exceeded, measured at the element centre. MAT_CSCM spells this as ERODE - 1 and FHWA's own deck uses 0.05. It is a KINEMATIC measure on purpose -- a plastic strain is rescaled by the crack band, so a threshold set for one element size would not carry to another (see continuum.strain_measures). | — |
| `points` | int | — | 单元内要有多少个积分点达到 `damage`——LS-DYNA 的 NUMFIP，默认 1 | how many of the element's integration points must reach `damage` -- LS-DYNA NUMFIP, whose default of 1 means one failed point out of eight deletes a fully integrated hexahedron. Defaults here to half the points, rounded up. | — |
| `min_volume_ratio` | float | `0.2` | 无关判据的保险：体积塌缩或翻转到低于这个比例的单元无论材料怎么说都删 | the criterion-free safety net: an element whose volume has collapsed or inverted goes whatever the material says (LS-DYNA *CONTROL_TIMESTEP ERODE with a non-zero DTMIN; Abaqus distortion control exists for this too) | — |
| `max_volume_ratio` | float | `5.0` | 体积膨胀超过这个倍数的单元删除（纠缠） | an element whose volume has grown by more than this factor is deleted (tangling) | — |
| `every` | int | `250` | 每隔多少次迭代检查一次 | iterations between checks -- LS-DYNA checks DR convergence every NRCYCK = 250 cycles for the same reason | — |

### `solver.mass_scaling` — MassScalingSpec

Selective mass scaling to a target explicit step: only the DOFs whose own stable step is below `target_dt` get mass, and exactly as much as lifts them to it. The mass is real -- gravity and a ground motion see it -- and more than `max_added_fraction` of the model's translational mass is refused. See vfife/core/mass_scaling.py.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `target_dt` | float | **必填** | 要达到的显式步长（s） | the explicit step to reach (s) | — |
| `max_added_fraction` | float | `0.01` | 补的平动质量超过全模型的这个比例时拒绝 | refuse when the added translational mass exceeds this fraction of the model's | — |

### `output` — OutputSpec

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `path` | str | `results.h5` | HDF5 结果文件；相对路径按当前工作目录（运行命令的目录）解析，不是算例文件所在目录；旁边生成 .xdmf 供 ParaView | the HDF5 result file; a relative path is relative to the working directory (where the command is run), not to the case file; an .xdmf beside it for ParaView | 61 个（01, 02, 03, 04 …） |
| `every` | int | `200` | 每隔多少步写一帧；动力与松弛分析的监测量在同样的帧上记录，静力（mode: static）分析的监测量在每个增量都记录、只有帧被稀疏（`time` 总与每条监测量一样长）；静力分析不写 `every` 时每个增量一帧 | write a frame every this many steps; in dynamic and relaxation runs the monitors are recorded at the same frames, in static runs (mode: static) at every increment and only the frames are thinned (`time` is always as long as each monitor); a static run that does not write `every` writes a frame at every increment | 61 个（01, 02, 03, 04 …） |
| `fields` | list[str] | `['displacement', 'von_mises']` | 要写的节点/单元场；`vfife describe <单元>` 列出模型能给的；写出来的名字模型给不了即拒绝（UNKNOWN_OUTPUT_FIELD）。不写时按模型取默认（1.1）：位移、转角、反力、实体与杆的 von_mises、壳的 von_mises_top/bottom、构件的带符号内力，有塑性铰、连接器、可断构件时另加它们的场；模型给不了的不写、也不报 | nodal/element fields to write; `vfife describe <element>` lists what the model can produce, and a name written here that the model cannot produce is refused (UNKNOWN_OUTPUT_FIELD). Left out, the default is the model's (1.1): displacement, rotation, reactions, von_mises on solids and bars, von_mises_top/bottom on shells, the members' signed forces, and the fields of its plastic hinges, connectors and breakable members where it has them; a default name the model does not produce is skipped without a word | 59 个（01, 02, 03, 04 …） |
| `checkpoint_every` | int | `0` | 每隔多少步写一个重启检查点到结果旁的 `<结果>.ckpt.h5`（在帧或进度检查处取，0 = 只在结束时） | write a restart checkpoint to `<result>.ckpt.h5` beside the result every this many steps (taken at a frame or progress check; 0 = only at the end) | 06 |
| `monitors` | dict[str, 见 MonitorSpec] | 空 | 命名的标量时程，见 MonitorSpec；`time`、`residual`、`kinetic_energy`、`age` 是保留名 | named scalar time histories, see MonitorSpec; `time`, `residual`, `kinetic_energy` and `age` are reserved | 58 个（01, 02, 04, 05 …） |

### `output.monitors.*` — MonitorSpec

One scalar time history, written to `monitors/<name>` in the result.  Four kinds, and the split is which THING the monitor reduces over:    reaction, displacement   the NODES of a region   element                  the ELEMENTS a region owns -- every node of the                            element inside it, the rule a section                            assignment already uses   bar                      the SEGMENTS of one reinforcement set  The last two exist because a verification contract can only read monitors and whole-model field extrema. Before them, case_15 could not assert the tendon stress it is built around (bar stress lives in `rebar_steps/`, which contracts cannot read) and had to stand in the camber for it, and it could not say "the constant-moment region has not crushed" -- only "no element anywhere has", which the anchorage zone decides. Both of those limits are recorded in that case's contract; both are now expressible.

| 字段 | 类型 | 默认 | 说明 | 英文原注 | 算例 |
|---|---|---|---|---|---|
| `kind` | `reaction` \| `displacement` \| `element` \| `bar` \| `fiber` \| `vehicle` | **必填** | reaction / displacement（区域的节点）、element（区域的单元）、bar（钢筋组的段）、fiber（截面上一点）、vehicle（车辆，`of` 为车名） | reaction / displacement (nodes of a region), element (elements of a region), bar (segments of a reinforcement set), fiber (one section point), vehicle (a vehicle named by `of`) | 58 个（01, 02, 04, 05 …） |
| `region` | str | — | 读的区域（节点类必填；element/bar 是缩小范围） | the region read (nodal kinds: required; element/bar: narrows to it) | 58 个（01, 02, 04, 05 …） |
| `regions` | list[str] | — | 节点类：几个区域一起读 | nodal kinds: several regions read together | 28d |
| `dof` | `x` \| `y` \| `z` | `z` | reaction/displacement：哪个分量 | reaction/displacement: the component | 58 个（01, 02, 04, 05 …） |
| `field` | str | — | element：实体是材料的任一单元输出（`dt`、`dc`、`d`……）加 `stress`、`von_mises`；线单元模型是任一构件场（`axial_force`、`bending_y`、`*_max_abs`、`*_i/_j`、`capacity_ratio`……）；bar：`axial_stress`、`axial_force`、`plastic_strain` | element: on solids any per-element output of the material (`dt`, `dc`, `d`, ...) plus `stress` and `von_mises`; on line models any member field (`axial_force`, `bending_y`, `*_max_abs`, `*_i/_j`, `capacity_ratio`, ..., v1.0). bar: `axial_stress`, `axial_force`, `plastic_strain`. | 18 个（12, 13, 14, 15 …） |
| `component` | `xx` \| `yy` \| `zz` \| `xy` \| `yz` \| `zx` \| `xz` | — | 张量场取哪个分量（本库统一的 Voigt 顺序 xx, yy, zz, xy, yz, zx；`xz` 作 `zx` 读） | which component of a tensor field, in this package's one Voigt order xx, yy, zz, xy, yz, zx (`xz` is read as `zx`) | 13, 15 |
| `of` | str | — | bar：钢筋组的名字，即其 assign 上的 `name` | bar: the reinforcement set's name, i.e. the `name` on its assignment | 12, 13, 14, 15, 38 |
| `axle` | int | `0` | vehicle：第几根轴（contact_force、wheel、station），从 0 起 | vehicle: which axle (contact_force, wheel, station), from 0 | — |
| `reduce` | `max` \| `min` \| `mean` \| `absmax` | — | element 监测量（实体与线单元模型）和实体的 bar 监测量：单元或钢筋段上的值怎么归约（max 默认、min、mean、absmax=绝对值最大者带符号）；其余监测量不接受 reduce | element monitors (solids and line models) and bar monitors on solids: how the values over the elements/segments are reduced (max by default; absmax is the signed value of largest magnitude); the other kinds refuse it | 12, 13, 14, 15 |
| `relative_to` | str | — | displacement：相对这个区域的平均位移报告——一个部件相对另一个的运动 | displacement: report it relative to the mean displacement of this region -- the motion of one part of a body seen from another | 19 |
| `at` | list[float] | — | fiber（只对 beam2_fiber）：离区域节点最近的积分站的截面，读 [y, z] 处：`field: strain` 是该点的平截面应变，`field: stress` 是最近纤维的应力——桥面或底缘应变片读到的量 | fiber (beam2_fiber only): the section at the station nearest the region's node, read at the point [y, z] of the section: `field: strain` is the plane-section strain eps0 - y kz + z ky there (the concrete's, no tendon prestrain), `field: stress` the stress of the fibre closest to the point. What a strain gauge on the deck or the soffit reads, and what a thesis reports as 上缘/下缘. | 13 个（24, 24c, 24d, 24f …） |
| `fiber_kind` | `concrete` \| `steel` | — | fiber、`field: stress`：只在混凝土纤维（`concrete`）或钢筋/预应力束（`steel`）里找最近的纤维；不给则在全部纤维里找。读的是哪根纤维（坐标、材料、种类）记在结果的监测量属性里 | fiber, `field: stress`: look for the nearest fibre among the concrete fibres only (`concrete`) or the bars and tendons only (`steel`); left out, among all of them. Which fibre was read (coordinates, material, kind) is recorded on the monitor in the result. | — |

## 三 材料

`materials.<名>: {type: <材料名>, <参数>...}`。参数没有默认值表的，看 `docs/` 里同名文档或算例。带单位的默认值（`rho`、`concrete01/02` 的 `fpc`、`steel02` 的 `E`/`Fy`、`steel_cyclic` 的 `E` 与牌号的 `fy`/`fu`、`bond_ceb` 与 `bond_cyclic` 的 `tau_max`/`s1..s3`）按 SI 给出，声明了非 SI 的 `units` 时换算到声明的单位。同一参数的别名（`concrete01/02` 的 `fc`=`fpc`、`eps_c0`=`epsc0`、`eps_cu`=`epscu`，`concrete02` 的 `fpcu`/`fcu2`=`fcu`、`concrete01` 的 `fcu`=`fpcu`，`steel02` 的 `E0`=`E`、`yield_strength`=`fy`=`Fy`，`rebar`、`j2`、`chaboche` 的 `fy`=`yield_strength`）与正名同时给出且不等时拒绝（`MATERIAL_ALIAS_CONFLICT`）。

### `bond_ceb`

* 运动学 bond；后端 numpy, numba, warp；分析模式 dynamic, relaxation；非线性
* 参数：`fc`, `condition`, `failure_mode`, `confinement`, `tau_max`, `tau_max_coeff`, `s1`, `s2`, `s3`, `alpha`, `tau_f_ratio`, `s_elastic`, `rho`
* 输出字段：`bond_stress`, `slip`, `slip_max`
* 注意：MC2010 monotonic envelope with a linear unloading rule; not Eligehausen cyclic degradation -- use bond_cyclic for that, which reproduces this law bit for bit on a monotonic path.
* 算例：12b
* 算例里出现过的参数：`condition`, `fc`

### `bond_cyclic`

* 运动学 bond；后端 numpy, numba, warp；分析模式 dynamic, relaxation；非线性
* 参数：`damage_coeff`, `damage_exponent`, `friction_ratio`, `s_reload`
* 输出字段：`bond_stress`, `slip`, `slip_max`, `bond_damage`, `bond_energy`
* 注意：Eligehausen-type cyclic degradation on the MC2010 envelope. The shape is standard; damage_coeff, damage_exponent, friction_ratio and s_reload are exposed defaults to calibrate, not citations.
* 算例：—

### `cdp`

* 运动学 solid_3d；后端 numpy, numba, warp；分析模式 dynamic, relaxation；非线性，正则化
* 参数：`E`, `nu`, `rho`, `compression`, `tension`, `dilation_angle`, `eccentricity`, `fb0_fc0`, `Kc`, `wt`, `wc`, `viscosity`, `damage_max`, `n_return`, `crack_band`, `l_ref`
* 输出字段：`stress`, `strain`, `von_mises`, `eps_p`, `ept`, `epc`, `d`
* 算例：06, 12, 12b, 12c, 15
* 算例里出现过的参数：`E`, `Kc`, `compression`, `crack_band`, `damage_max`, `dilation_angle`, `eccentricity`, `fb0_fc0`, `l_ref`, `n_return`, `nu`, `rho`, `tension`, `viscosity`, `wc`, `wt`

### `chaboche`

* 运动学 solid_3d；后端 numpy, numba, warp；分析模式 dynamic, relaxation；非线性
* 参数：`E`, `nu`, `rho`, `yield_strength`, `Q_inf`, `b`, `H_iso`, `C1`, `gamma1`, `C2`, `gamma2`, `damage_initiation_strain`, `damage_evolution_rate`, `damage_max`, `damage_viscosity`, `n_newton`
* 输出字段：`stress`, `strain`, `von_mises`, `eps_p`, `alpha`, `X1`, `X2`, `D`, `D_bar`
* 算例：—

### `concrete01`

* 运动学 uniaxial；后端 numpy, numba；分析模式 dynamic, relaxation；非线性
* 参数：`fpc`, `epsc0`, `fpcu`, `epscu`, `fc`, `eps_c0`, `fcu`, `eps_cu`, `E`, `rho`, `nu`
* 输出字段：`axial_stress`, `axial_strain`
* 注意：Kent-Scott-Park envelope, Karsan-Jirsa unloading, no tension.
* 算例：—

### `concrete02`

* 运动学 uniaxial；后端 numpy, numba；分析模式 dynamic, relaxation；非线性
* 参数：`fpc`, `epsc0`, `fcu`, `epscu`, `rat`, `ft`, `Ets`, `fc`, `eps_c0`, `fcu2`, `eps_cu`, `rho`, `nu`, `strain_scale`, `fpcu`
* 输出字段：`axial_stress`, `axial_strain`
* 注意：Yassin concrete with linear tension softening.
* 算例：16 个（11, 21, 21b, 21c …）
* 算例里出现过的参数：`Ets`, `creep`, `epsc0`, `epscu`, `fcu`, `fpc`, `ft`, `rat`, `rho`, `shrinkage`

### `gap`

* 运动学 uniaxial；后端 numpy；分析模式 dynamic, relaxation；非线性
* 参数：`E`, `rho`, `lo`, `hi`
* 输出字段：`stress`
* 算例：—

### `j2`

* 运动学 solid_3d；后端 numpy, numba, warp；分析模式 dynamic, relaxation；非线性
* 参数：`E`, `nu`, `rho`, `hardening`, `yield_strength`, `kinematic`, `n_newton`
* 输出字段：`stress`, `strain`, `von_mises`, `eps_p`, `alpha`, `X`
* 算例：—

### `linear_elastic`

* 运动学 solid_3d, uniaxial；后端 numpy, numba, warp；分析模式 dynamic, relaxation；线性
* 参数：`E`, `nu`, `rho`
* 输出字段：`stress`, `strain`, `von_mises`
* 算例：40 个（01, 02, 03, 04 …）
* 算例里出现过的参数：`E`, `creep`, `nu`, `rho`, `shrinkage`, `thermal_expansion`

### `orthotropic`

* 运动学 solid_3d；后端 numpy；分析模式 dynamic, relaxation；线性
* 参数：`E1`, `E2`, `E3`, `nu12`, `nu13`, `nu23`, `G12`, `G13`, `G23`, `rho`
* 输出字段：`stress`, `strain`, `von_mises`
* 注意：Local material axes coincide with global axes; orientation rotation is not yet implemented.
* 算例：—

### `parallel`

* 运动学 uniaxial；后端 numpy；分析模式 dynamic, relaxation；非线性
* 参数：`of`, `factors`, `rho`
* 输出字段：`stress`
* 注意：no compiled form: a fibre section with it runs on the NumPy reference
* 算例：—

### `piecewise_uniaxial`

* 运动学 uniaxial；后端 numpy；分析模式 dynamic, relaxation；非线性
* 参数：`E`, `rho`, `tension`, `compression`, `hardening_mode`, `kinematic_fraction`, `nu`
* 输出字段：`axial_stress`, `axial_strain`, `eps_p`, `alpha`, `backstress`, `external_work`, `elastic_energy`, `hysteretic_energy`
* 注意：Piecewise-linear rate-independent law; tables use positive stress magnitudes.
* 算例：—

### `rebar`

* 运动学 uniaxial；后端 numpy, numba, warp；分析模式 dynamic, relaxation；非线性
* 参数：`E`, `rho`, `yield_strength`, `hardening`, `kinematic_fraction`, `nu`, `rupture_strain`
* 输出字段：`axial_stress`, `axial_strain`, `eps_p`, `alpha`, `X`
* 算例：06, 12, 12b, 12c, 13, 14, 15
* 算例里出现过的参数：`E`, `hardening`, `kinematic_fraction`, `rho`, `yield_strength`

### `series`

* 运动学 uniaxial；后端 numpy；分析模式 dynamic, relaxation；非线性
* 参数：`of`, `max_iterations`, `tolerance`, `rho`
* 输出字段：`stress`
* 注意：no compiled form: a fibre section with it runs on the NumPy reference; the split is iterated at each point and a split that does not converge is an error
* 算例：—

### `steel02`

* 运动学 uniaxial；后端 numpy, numba；分析模式 dynamic, relaxation；非线性
* 参数：`E`, `E0`, `rho`, `Fy`, `yield_strength`, `b`, `R0`, `cR1`, `cR2`, `a1`, `a2`, `a3`, `a4`, `nu`, `strain_scale`, `rupture_strain`
* 输出字段：`axial_stress`, `axial_strain`, `eps_rev`, `sig_rev`, `hysteretic_energy`, `eps_max`
* 算例：15 个（11, 21, 21b, 21c …）
* 算例里出现过的参数：`E`, `Fy`, `R0`, `b`, `relaxation`, `rho`, `rupture_strain`

### `steel_cyclic`

* 运动学 uniaxial；后端 numpy, numba；分析模式 dynamic, relaxation；非线性
* 参数：`E`, `rho`, `nu`, `grade`, `fy`, `fu`, `f_uf`, `k1`, `k2`, `k3`, `a`, `b0`, `b1`, `b2`, `eps_cri`, `lambda_dsps`, `skeleton_fit`
* 输出字段：`axial_stress`, `axial_strain`, `eps_max`, `eps_acc_c`, `eps_uf`, `failed`
* 注意：fracture strain from the DSPS model; f_uf is not in the paper's Table 1 and defaults to 0.5 f_u
* 算例：—

### `viscoelastic`

* 运动学 solid_3d；后端 numpy；分析模式 dynamic；非线性
* 参数：`E`, `nu`, `rho`, `g`, `tau`
* 输出字段：`stress`, `strain`, `von_mises`
* 注意：Shear Prony only; bulk elastic. Dynamic analysis only (physical time).
* 算例：—

## 四 单元与截面类型

截面 `type` 对应的单元：`solid` → 实体（由网格决定 tet4/tet10/hexa8/hexa20/wedge6/wedge15/pyramid5），`shell` → shell3/shell4，`beam` → beam2（`theory: timoshenko` 为 beam2_timoshenko），`fiber` → beam2_fiber，`bar` → truss2，`connector` → connector2。

| 单元 | 后端 | 分析模式 | 可用材料 | 输出字段 |
|---|---|---|---|---|
| `beam2` | numpy, numba | dynamic, relaxation, static | linear_elastic | displacement, rotation, end_actions, axial_force, shear_y, shear_z, torsion, bending_y, bending_z, axial_force_max_abs, shear_y_max_abs, shear_z_max_abs, torsion_max_abs, bending_y_max_abs, bending_z_max_abs, capacity_ratio, member_failed, member_built, reaction_force, reaction_moment |
| `beam2_fiber` | numpy, numba | dynamic, relaxation, static | linear_elastic, rebar, steel02, steel_cyclic, piecewise_uniaxial, concrete01, concrete02, parallel, series, gap | displacement, rotation, end_actions, reaction_force, reaction_moment, axial_force, shear_y, shear_z, torsion, bending_y, bending_z, axial_force_max_abs, shear_y_max_abs, shear_z_max_abs, torsion_max_abs, bending_y_max_abs, bending_z_max_abs, member_failed, member_built, fiber_stress, fiber_damage, connector_flag, connector_force, connector_moment, connector_deformation, connector_slip |
| `beam2_timoshenko` | numpy, numba | dynamic, relaxation, static | linear_elastic | displacement, rotation, end_actions, axial_force, shear_y, shear_z, torsion, bending_y, bending_z, axial_force_max_abs, shear_y_max_abs, shear_z_max_abs, torsion_max_abs, bending_y_max_abs, bending_z_max_abs, capacity_ratio, member_failed, member_built, reaction_force, reaction_moment |
| `connector2` | numpy | dynamic, relaxation | linear_elastic | displacement, rotation, connector_force, connector_moment, connector_deformation, reaction_force, reaction_moment |
| `coupled` | numpy, numba | dynamic, relaxation | linear_elastic | displacement, stress, strain, von_mises, stress_nodal, reaction_force, rotation, member_type, beam_formulation, connector_flag, connector_force, connector_moment, connector_deformation, end_actions, reaction_moment |
| `hexa20` | numpy, numba, warp | dynamic, relaxation | linear_elastic, orthotropic, viscoelastic, j2, chaboche, cdp | displacement, stress, strain, von_mises, stress_nodal, reaction_force |
| `hexa8` | numpy, numba, warp | dynamic, relaxation | linear_elastic, orthotropic, viscoelastic, j2, chaboche, cdp | displacement, stress, strain, von_mises, stress_nodal, reaction_force |
| `mixed_line` | numpy, numba | dynamic, relaxation, static | linear_elastic | displacement, rotation, member_type, beam_formulation, connector_flag, connector_force, connector_moment, connector_deformation, connector_slip, end_actions, axial_force, shear_y, shear_z, torsion, bending_y, bending_z, axial_force_max_abs, shear_y_max_abs, shear_z_max_abs, torsion_max_abs, bending_y_max_abs, bending_z_max_abs, capacity_ratio, member_failed, member_built, reaction_force, reaction_moment |
| `pyramid5` | numpy, numba, warp | dynamic, relaxation | linear_elastic, orthotropic, viscoelastic, j2, chaboche, cdp | displacement, stress, strain, von_mises, stress_nodal, reaction_force |
| `shell3` | numpy | dynamic, relaxation | linear_elastic | displacement, rotation, membrane_resultants, bending_resultants, shear_resultants, shell_stress_top, shell_stress_bottom, von_mises_top, von_mises_bottom, reaction_force, reaction_moment |
| `shell4` | numpy, numba, warp | dynamic, relaxation | linear_elastic | displacement, rotation, membrane_resultants, bending_resultants, shear_resultants, shell_stress_top, shell_stress_bottom, von_mises_top, von_mises_bottom, reaction_force, reaction_moment |
| `shell_line` | numpy, numba | dynamic, relaxation | linear_elastic | displacement, rotation, shell_stress_top, shell_stress_bottom, end_actions, reaction_force, reaction_moment |
| `tet10` | numpy, numba, warp | dynamic, relaxation | linear_elastic, orthotropic, viscoelastic, j2, chaboche, cdp | displacement, stress, strain, von_mises, stress_nodal, reaction_force |
| `tet4` | numpy, numba, warp | dynamic, relaxation | linear_elastic, orthotropic, viscoelastic, j2, chaboche, cdp | displacement, stress, strain, von_mises, stress_nodal, reaction_force |
| `truss2` | numpy | dynamic, relaxation | linear_elastic, rebar, piecewise_uniaxial, steel02, steel_cyclic, concrete01, concrete02, parallel, series, gap | displacement, axial_force, axial_force_max_abs, axial_stress, von_mises, capacity_ratio, member_failed, reaction_force |
| `wedge15` | numpy, numba, warp | dynamic, relaxation | linear_elastic, orthotropic, viscoelastic, j2, chaboche, cdp | displacement, stress, strain, von_mises, stress_nodal, reaction_force |
| `wedge6` | numpy, numba, warp | dynamic, relaxation | linear_elastic, orthotropic, viscoelastic, j2, chaboche, cdp | displacement, stress, strain, von_mises, stress_nodal, reaction_force |

## 五 没有任何自带算例用到的字段

这些字段能写、会被校验，但没有带契约的例子。用之前先在测试里找它（`grep -r 字段名 tests/`）；长期没有例子的字段是删除候选。

* `mesh.points`
* `mesh.lines`
* `mesh.groups`
* `mesh.origin`
* `mesh.scale`
* `mesh.lines.*.through`
* `mesh.lines.*.divisions`
* `mesh.lines.*.size`
* `mesh.lines.*.arc`
* `mesh.lines.*.arc.rise`
* `mesh.lines.*.arc.shape`
* `mesh.lines.*.arc.up`
* `sections.*.warpage_tolerance`
* `sections.*.t`
* `sections.*.top_width`
* `sections.*.bottom_width`
* `sections.*.cantilever`
* `sections.*.top_thickness`
* `sections.*.bottom_thickness`
* `sections.*.web_thickness`
* `sections.*.cantilever_thickness`
* `sections.*.top_haunch`
* `sections.*.bottom_haunch`
* `sections.*.cells`
* `sections.*.layers`
* `sections.*.fiber_size`
* `sections.*.origin`
* `sections.*.bf`
* `sections.*.tf`
* `sections.*.hw`
* `sections.*.tw`
* `sections.*.perimeter`
* `sections.*.nt`
* `sections.*.nr`
* `sections.*.bar_layers`
* `sections.*.fracture`
* `sections.*.connector_damping`
* `sections.*.shear.direction`
* `sections.*.shear.cracked_stiffness`
* `sections.*.shear.web_width`
* `sections.*.shear.cracking`
* `sections.*.shear.yield_force`
* `sections.*.shear.hardening`
* `sections.*.fracture.threshold`
* `sections.*.fracture.split`
* `sections.*.laws.*.hi`
* `sections.*.laws.*.k1`
* `sections.*.laws.*.k2`
* `sections.*.laws.*.fy`
* `sections.*.laws.*.radius`
* `sections.*.laws.*.c`
* `sections.*.laws.*.exponent`
* `sections.*.laws.*.linear_below`
* `sections.*.laws.*.material`
* `sections.*.laws.*.area`
* `sections.*.laws.*.length`
* `reinforcement.*.line.divisions`
* `reinforcement.*.layer.divisions`
* `reinforcement.*.layer.segment_length`
* `reinforcement.*.hoop.segment_length`
* `reinforcement.*.path.group`
* `reinforcement.*.path.divisions`
* `regions.*.line`
* `regions.*.cylinder`
* `regions.*.sphere`
* `regions.*.group`
* `regions.*.node_ids`
* `regions.*.all`
* `regions.*.intersect`
* `regions.*.difference`
* `regions.*.tol`
* `assign[].releases`
* `assign[].end_offsets`
* `assign[].divisions`
* `assign[].transverse_penalty`
* `assign[].host_deduction`
* `assign[].friction`
* `assign[].jack`
* `assign[].friction.mu`
* `assign[].friction.k`
* `assign[].friction.jack`
* `assign[].friction.anchor_set`
* `assign[].failure.tensile_strength`
* `assign[].failure.shape_factor`
* `assign[].failure.rotation_capacity`
* `assign[].failure.split`
* `offset_arms.material`
* `offset_arms.factor`
* `offset_arms.limit`
* `offset_arms.regions`
* `tendons[].name`
* `tendons[].nodes`
* `tendons[].section`
* `tendons[].count`
* `tendons[].line`
* `tendons[].offset`
* `tendons[].profile_y`
* `tendons[].profile_z`
* `tendons[].stress`
* `tendons[].friction`
* `tendons[].draw_in`
* `tendons[].relaxation`
* `tendons[].arm_factor`
* `tendons[].line.origin`
* `tendons[].line.direction`
* `tendons[].line.turn`
* `tendons[].stress.step`
* `tendons[].stress.ends`
* `tendons[].stress.stress`
* `tendons[].stress.force`
* `tendons[].friction.mu`
* `tendons[].friction.k`
* `tendons[].draw_in.begin`
* `tendons[].draw_in.end`
* `tendons[].relaxation.code`
* `tendons[].relaxation.fpk`
* `tendons[].relaxation.zeta`
* `tendons[].relaxation.overstressed`
* `constraints[].motion`
* `constraints[].keep`
* `constraints[].amplitude`
* `constraints[].record.scale`
* `constraints[].record.delay`
* `loads[].keep`
* `loads[].moment`
* `loads[].coordinate_system`
* `loads[].with_mass`
* `loads[].traction`
* `loads[].body_force`
* `loads[].per_node`
* `loads[].moving_load.direction`
* `loads[].vehicle.direction`
* `loads[].vehicle.gravity`
* `loads[].vehicle.roughness`
* `loads[].vehicle.body.pitch_inertia`
* `loads[].vehicle.body.cg`
* `loads[].vehicle.axles[].unsprung_mass`
* `loads[].vehicle.axles[].tyre`
* `loads[].vehicle.axles[].suspension.damping`
* `loads[].vehicle.roughness.profile`
* `loads[].vehicle.roughness.scale`
* `loads[].vehicle.roughness.psd`
* `loads[].vehicle.roughness.psd.g0`
* `loads[].vehicle.roughness.psd.n0`
* `loads[].vehicle.roughness.psd.w`
* `loads[].vehicle.roughness.psd.n_min`
* `loads[].vehicle.roughness.psd.n_max`
* `loads[].vehicle.roughness.psd.count`
* `loads[].vehicle.roughness.psd.seed`
* `nodal_masses[].components`
* `nodal_masses[].weight`
* `nodal_masses[].inertia`
* `hydrodynamic_mass[].area`
* `hydrodynamic_mass[].vertical`
* `soil_springs[].region`
* `soil_springs[].ground`
* `soil_springs[].vertical`
* `soil_springs[].direction`
* `soil_springs[].curves`
* `soil_springs[].curves[].depth`
* `soil_springs[].curves[].p_y`
* `diaphragms[].region`
* `diaphragms[].normal`
* `steps[].stage_ramp_iterations`
* `steps[].time_spacing`
* `steps[].procedure`
* `steps[].total_time`
* `steps[].activate`
* `steps[].birth`
* `steps[].activate_age`
* `steps[].deactivate`
* `steps[].release`
* `steps[].rigid_links`
* `steps[].rigid_links[].master`
* `steps[].rigid_links[].slaves`
* `steps[].rigid_links[].dofs`
* `steps[].rigid_links[].arm`
* `steps[].rigid_links[].name`
* `contact.members`
* `contact.rigid[].stiffness`
* `contact.rigid[].friction`
* `contact.members.bodies`
* `contact.members.stiffness`
* `contact.members.friction`
* `contact.members.damping`
* `contact.members.slack`
* `contact.members.bodies[].region`
* `contact.members.bodies[].radius`
* `response_spectrum.damping`
* `response_spectrum.mass_fraction`
* `response_spectrum.beyond_table`
* `response_spectrum.max_dofs`
* `response_spectrum.spectra[].path`
* `response_spectrum.spectra[].scale`
* `solver.target_dt`
* `solver.erosion`
* `solver.mass_scaling`
* `solver.initial_state`
* `solver.rotation_iterations`
* `solver.threads`
* `solver.device`
* `solver.precision`
* `solver.static.max_iterations`
* `solver.static.cutbacks`
* `solver.static.line_search`
* `solver.static.tangent`
* `solver.static.perturbation`
* `solver.static.fallback`
* `solver.static.fallback_iterations`
* `solver.static.arc_length.control`
* `solver.static.arc_length.control.region`
* `solver.static.arc_length.control.dof`
* `solver.damping.ratio`
* `solver.damping.frequencies`
* `solver.erosion.damage`
* `solver.erosion.measure`
* `solver.erosion.strain`
* `solver.erosion.points`
* `solver.erosion.min_volume_ratio`
* `solver.erosion.max_volume_ratio`
* `solver.erosion.every`
* `solver.mass_scaling.target_dt`
* `solver.mass_scaling.max_added_fraction`
* `output.monitors.*.axle`
* `output.monitors.*.fiber_kind`
* `materials.bond_cyclic`
* `materials.chaboche`
* `materials.concrete01`
* `materials.gap`
* `materials.j2`
* `materials.orthotropic`
* `materials.parallel`
* `materials.piecewise_uniaxial`
* `materials.series`
* `materials.steel_cyclic`
* `materials.viscoelastic`

## 六 代码里没有中文说明的字段

0 个。字段上方第一行 `#:` 注释以中文开头即为中文说明，其后是英文原注；补上再生成即可。


## 七 弃用的写法

仍然读，preflight 给一条 `DEPRECATED_FIELD` 警告；`vfife migrate <算例> --in-place` 在原文里就地改名、加上 `version`，注释和格式不动，并核对改写前后降解出的模型相同。

| 块 | 旧写法 | 现在的写法 |
|---|---|---|
| `sections.*` | `b` | `width` |
| `sections.*` | `h` | `height` |
| `sections.*` | `d` | `diameter` |
| `sections.*` | `torsion_constant` | `j` |
