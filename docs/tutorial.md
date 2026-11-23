# 上手指引：从一块几何到一张 ParaView 图

这份文档手把手走完一个完整算例：**Gmsh 建网格 → 写 YAML → 运行 → ParaView 看结果 → 校核**。

例子是一根钢悬臂托架，端部受集中力。它足够简单，能用梁理论手算校核；也足够真实，会撞上一个所有人第一次都会撞的坑（四面体锁死）。

**文中每一条命令和每一段输出都是实际跑出来的**，你照着做应该看到同样的数字。全程约 15 分钟，其中求解占 3 分钟。

> 约定：命令写作 `python -m vfife ...`。若你已 `pip install -e .`，可直接用 `vfife ...`。

---

## 0. 准备

```bash
conda activate vfife          # 或你自己的环境
cd <项目根目录>
python -m vfife list          # 能列出单元和材料就说明装好了
```

另外需要 **Gmsh**（建网格）和 **ParaView**（看结果），两者都是独立软件，各自下载安装即可。

---

## 1. 在 Gmsh 里建几何和网格

新建 `bracket.geo`：

```c
SetFactory("OpenCASCADE");
L = 1.2; b = 0.15; h = 0.20;          // 悬臂长 1.2 m，截面 150 × 200 mm
Box(1) = {0, 0, 0, L, b, h};

Physical Volume("steel")   = {1};      // 实体本身
Physical Surface("fixed")  = {1};      // x = 0 的固定端
Physical Surface("tip")    = {2};      // x = L 的加载端

Mesh.MeshSizeMax = 0.05;               // 目标单元尺寸 50 mm
```

### 这里唯一要紧的事：Physical 组

**没有 Physical 组，网格就没法用。** 求解器靠这些名字知道哪里固定、哪里加载 —— YAML 里写的 `region: fixed` 指的就是这里的 `Physical Surface("fixed")`。

面的编号（上面的 `{1}`、`{2}`）**不能靠猜**。在 Gmsh GUI 里点选面即可在状态栏看到编号；或者用一小段脚本列出每个面的形心：

```python
import gmsh
gmsh.initialize(); gmsh.open("bracket.geo"); gmsh.model.occ.synchronize()
for dim, tag in gmsh.model.getEntities(2):
    c = gmsh.model.occ.getCenterOfMass(2, tag)
    print(f"面 {tag}: 形心 x={c[0]:.3f}")
gmsh.finalize()
```

```
面 1: 形心 x=0.000      ← 固定端
面 2: 形心 x=1.200      ← 加载端
```

编号错了不会报错，只会把边界条件加到别的面上，然后你得到一个看起来很合理的错答案。**第 2 步就是用来当场发现这件事的。**

### 生成网格

```bash
gmsh -3 bracket.geo -o bracket.msh
```

`-3` 表示生成三维单元。忘了它就只有面网格，求解器会告诉你。

---

## 2. 先看看这个网格里有什么

**不要跳过这一步。** 它花两秒，能挡住后面一半的问题：

```bash
python -m vfife mesh bracket.msh
```

```
mesh: /tmp/tut/bracket.msh
  element       tet4 (continuum), 1791 elements, 565 nodes
  formulations  full
  backends      numpy, numba, warp
  extent        x [0, 1.2]  y [0, 0.15]  z [0, 0.2]
  physical groups (use these names in `region:`)
    fixed                25 nodes   x=0, y [0, 0.15], z [0, 0.2]
    steel               565 nodes   x [0, 1.2], y [0, 0.15], z [0, 0.2]
    tip                  25 nodes   x=1.2, y [0, 0.15], z [0, 0.2]
  next          python -m vfife init <case>.yaml --mesh bracket.msh
```

三件事值得看清楚：

| 看什么 | 为什么 |
|---|---|
| `fixed` 的范围是 `x=0` | 证明你选对了面。如果这里显示 `x=1.2`，说明 `.geo` 里的编号写反了 —— 现在改，比跑完再改便宜 |
| `element tet4` | Gmsh 三维默认就是四面体。**记住这一点，第 8 步会用到** |
| `formulations full` | tet4 **只支持** `full`。写别的会被拦下来 |
| 三维单元有几种 | 一种最省事。若显示 `element  mixed: tet4 (18253) + pyramid5 (2957)`，那也能直接算 —— 各类单元共用节点，作为一个整体求解。金字塔是 Gmsh 的过渡单元，`.geo` 里一句 `Recombine Surface` 就会产生它们。代价见第 8 节：混合网格目前只能用 `numpy` 后端，`formulations` 也只剩各块的交集 |

---

## 3. 生成 YAML 骨架

```bash
python -m vfife init bracket.yaml --mesh bracket.msh
```

它会问三个问题（组名就是第 2 步看到的那些）：

```
Imported Gmsh Physical groups: fixed, steel, tip
Fully fixed support region [fixed, steel, tip]: fixed
Loaded region [fixed, steel, tip]: tip
Total load on 'tip' as Fx Fy Fz in N [0 0 -1000]: 0 0 -10000
  created       bracket.yaml
  next          python -m vfife info bracket.yaml
```

生成的文件：

```yaml
version: 1
name: bracket
units: {length: m, force: N, stress: Pa}

mesh:
  type: gmsh
  path: bracket.msh
  scale: 1.0                 # 网格若用 mm 建，这里写 0.001

materials:
  steel: {type: linear_elastic, E: 2.1e+11, nu: 0.3, rho: 7850.0}   # 默认是钢，按需要改

sections:
  body: {type: solid, material: steel, formulation: full}   # formulation 按网格里的单元自动选；
                                                            # 唯一的 solid 截面自动赋给整个网格

constraints:
  - region: fixed
    dofs: [1, 1, 1]          # 1 = 约束，0 = 自由，顺序 [ux, uy, uz]

loads:
  - region: tip
    vector: [0.0, 0.0, -10000.0]   # 作用在该区域上的**总力**，自动分摊到节点

solver:
  backend: numpy
  mode: relaxation
  n_steps: 40000
  tolerance: 1.0e-06
  damping: {type: kinetic}

output:
  path: results/bracket.h5
  every: 100000
  fields: [displacement, von_mises]
  monitors:
    u: {kind: displacement, region: tip, dof: z}     # 加载区域沿荷载方向的位移
    R: {kind: reaction, region: fixed, dof: z}       # 支座在同一方向的反力
```

### 改一改监测量

骨架已经带了两条监测量：加载区域的位移 `u` 和支座反力 `R`（沿荷载的方向）。把它们改成读得懂的名字，
再让结果多写几帧，你就能不开 ParaView 知道答案对不对：

```yaml
output:
  path: results/bracket.h5
  every: 2000                      # 每 2000 步写一帧（供 ParaView 动画）
  fields: [displacement, von_mises]
  monitors:
    tip_uz:   {kind: displacement, region: tip,   dof: z}
    reaction: {kind: reaction,     region: fixed, dof: z}
```

顺手把 `tolerance` 收紧到 `1.0e-8`（这是个线弹性小算例，收敛很快）。

> **`reaction` 是最有价值的一个监测量。** 它应该与你施加的荷载**等值反号** ——
> 支座要托住向下的荷载，它给出的力就是向上的。若量值对不上，说明约束或荷载加错了
> 地方；这是最省事的自查。

`reaction` 和 `displacement` 读的是区域里的**节点**。另外两种读别的东西：

```yaml
  monitors:
    # 单元场，限制在一个区域里。归约: max | min | mean | absmax
    dc_span:   {kind: element, field: dc, reduce: max, region: span_block}
    # 应力是张量，要给 component（xx yy zz xy yz zx，本包只有一套 Voigt 顺序）
    sigma_bot: {kind: element, field: stress, component: xx, reduce: mean,
                region: span_soffit}
    # 钢筋/预应力筋。`of` 是**配筋指派上的 name**，不是配筋几何的键
    sigma_p:   {kind: bar, of: tendons, field: axial_stress, reduce: mean,
                region: tendon_flat}
```

- `field` 对 `element` 来说是材料能输出的任意逐单元量（CDP 是 `d`/`dt`/`dc`），
  外加 `stress` 和 `von_mises`；对 `bar` 是 `axial_stress`、`axial_force`、
  `plastic_strain`。名字写错会在**建模阶段**报错并列出可用的。
- `element` 的区域归属规则和截面指派完全一致：**单元的每个节点**都在区域里。
  所以一个只有一层节点的面区域不拥有任何单元，这会当场报错，而不是安静地
  归约出一个 0。
- `displacement` 可加 `relative_to: <区域>`：报告的是本区域减去那个区域的平均
  位移。一个自由飞行的物体，质心在走、两端在绕它转——`{kind: displacement,
  region: A, dof: z, relative_to: C}` 读出来的就是那个转动（0.1 版的 case_19）。
- `bar` 的区域是**几何**判据：钢筋有自己的节点，区域解析出的节点集与它无关，
  所以判据是段的两端都落在区域的 `box` 里。不是 box 的区域（Gmsh 物理组、
  显式节点表）会被拒绝，而不是用包围盒近似。
- 两者的 `region` 都可以省略，那就是整个模型。
- 被侵蚀删掉的单元不参与 `element` 归约：它的损伤停在离场那一刻，让它决定
  `max` 等于让监测量报告历史而不是状态。

**为什么要有这两种。** 验证契约只能读 `monitor`、整模型的 `field` 极值和
`diagnostic`。在它们之前，`cases/case_15` 无法断言它整个建立于其上的
预应力筋应力（筋应力写在 `rebar_steps/` 里，契约读不到），只能用拱度当替身；
也无法说"常弯段没有压碎"，只能说"全模型没有任何单元压碎"，而全局最大值由
锚固区决定。两件事都是监测量形状的，所以它们现在是监测量。

---

## 4. 求解前检查

```bash
python -m vfife info bracket.yaml
```

```
case: bracket
  preflight     PASS (0 errors, 0 warnings)
  mesh          565 nodes, 1791 elements
  element       tet4 / full (1 integration point)
  material      linear_elastic (0 state vars/point)
  mode          relaxation, mass = underwood, damping = kinetic
  backend       numpy
  time step     dt = 1.0000e+00, 40000 steps
  regions       fixed(25), tip(25), steel(565), all(565), surface(available)
```

`info` 不求解，只把模型建起来再打印一遍。**preflight 会检查刚体位移、孤立节点、单元与材料是否兼容等**，它报 FAIL 时不要硬跑。

除了你自己命名的区域，还有两个总是可用：`all`（全部节点）和 `surface`（网格外表面，用到才计算）。

---

## 5. 运行

```bash
python -m vfife run bracket.yaml
```

运行时的进度行：

```
  step     8000/40000   20.0%  resid 4.71e-06  KE 1.203e-12  resets 62    458 steps/s  eta   70s
```

- **`resid`** 是不平衡力的相对残差 —— 它降到 `tolerance` 以下就是收敛。**这是唯一要盯的数**
- `KE` 是动能，`resets` 是动能阻尼的重启次数（求解器在耗散振荡，正常）

结束时：

```
  analysis      CONVERGED (residual …, analysis step 1 of 1)
  wall time     54.54 s (458 steps/s, 0.82 M element-steps/s)
  max |u|       2.351146e-04
  tip_uz        final -2.349903e-04
  reaction      final +1.000000e+04
  written       results/bracket.h5 (+ .xdmf time series, _last.xdmf fallback for ParaView; .ckpt.h5 checkpoint)
```

**先看 `analysis` 这一行**：`CONVERGED` 才是答案；`MAX_STEPS` 等失败状态照样写出结果文件（供诊断），但命令以退出码 3 结束。

**先看 `reaction`：+1.0000e+04 N，与施加的 −10 kN 等值反号。** 平衡成立，边界条件加对了。

> 想中途停下：按 **Ctrl-C**。结果文件会正常关闭并在旁边保留检查点 `results/bracket.ckpt.h5`，之后用
> `python -m vfife run bracket.yaml --restart results/bracket.h5 -o results/bracket_rest.h5` 从断点续跑——
> 续算写进**新文件**（写回原文件会被拒绝：`RESTART_OVERWRITES`），两个文件的帧接起来就是一次完整的计算。

---

## 6b. 曲线：出图与导出

ParaView 看的是场，**监视器的时程曲线**要用另外两条命令 —— 不需要写任何 Python：

```bash
python -m vfife plot results/bracket.h5                       # 全部监视器 vs 时间，写成 results/bracket.png
python -m vfife plot results/bracket.h5 --y tip_uz            # 只画一条
python -m vfife plot results/bracket.h5 --x tip_uz --y reaction   # 一个对另一个
python -m vfife export results/bracket.h5                     # 全部监视器 -> results/bracket.csv
python -m vfife export results/bracket.h5 --monitor tip_uz --monitor reaction
```

**`--x` 加 `--y` 是关键的一条**：反力对位移画出来就是荷载-位移曲线，
配上循环幅值就是**滞回环**。

坐标轴标签是自动的 —— 监视器把自己的 `kind`/`region`/`dof` 存在结果文件里，
所以 `--x tip_uz` 的横轴会标成 `tip_uz  (displacement z on tip) [m]`（单位取自算例声明的 `units`），不用你去记
当初写的名字是什么意思。

`plot` 需要 matplotlib（`pip install vfife[plot]`）。没装也不会报栈，会告诉你
装什么，以及 `export` 不需要任何额外依赖。

---

## 6. 在 ParaView 里看

运行产出三个文件：

| 文件 | 用途 |
|---|---|
| `bracket.h5` | 全部数据 |
| `bracket.xdmf` | **在 ParaView 里打开这个** —— 完整时间序列 |
| `bracket_last.xdmf` | 只含最后一帧的备用文件，某些 ParaView 版本读时间序列有问题时用它 |

操作步骤：

1. **File → Open → `results/bracket.xdmf`**，读取器选 **XDMF Reader**，点 **Apply**
2. 左上角下拉框把 `Solid Color` 改成 **`von_mises`**，右边的下拉改成 **Cell Data**
3. 点 **Filters → Alphabetical → Warp By Vector**，Vectors 选 `displacement`，
   Scale Factor 填 **500**（位移只有 0.24 mm，不放大看不见），**Apply**
4. 点工具栏的 **▶** 播放，能看到加载过程

看到的应该是：根部上下表面应力最高、向端部递减 —— 经典的悬臂弯曲分布。

---

## 7. 校核 —— 这一步不能省

跑出数不等于跑对数。这个例子可以用梁理论手算：

```
截面    b = 0.15 m, h = 0.20 m,  I = b·h³/12 = 1.0e-4 m⁴
弯曲    δ = P·L³/(3·E·I) = 1e4 × 1.728 / (3 × 2.1e11 × 1e-4) = 0.2743 mm
剪切    δ = P·L/(κ·G·A),  κ = 5/6                            = 0.0059 mm
合计                                                          = 0.2802 mm
```

而计算给出 **0.2350 mm** —— 只有理论值的 **83.9%**，也就是**偏刚 16%**。

### 这不是 bug，是四面体锁死

`tet4` 是常应变单元：整个单元内应变是常数，没有足够的自由度表达弯曲，于是表现得比真实结构硬。网格加密**不能**有效解决（收敛极慢）。

正确做法是**换二次单元**：

```bash
gmsh -3 -order 2 bracket.geo -o bracket_o2.msh
```

```bash
python -m vfife mesh bracket_o2.msh
```
```
  element       tet10 (continuum), 1791 elements, 3399 nodes
  formulations  full, b-bar
```

单元数没变，节点数从 565 涨到 3399（多了棱中点）。而且 tet10 支持 `b-bar`。改两行 YAML：

```yaml
mesh:
  path: bracket_o2.msh       # ← 换网格

sections:
  body: {type: solid, material: steel, formulation: b-bar}   # ← tet10 支持，抗锁死更好
```

重跑：

```
  max |u|       2.811409e-04
  tip_uz        final -2.808897e-04
  reaction      final -9.999995e+03
```

**0.2809 mm，是梁理论的 100.2%。**

| 单元 | 结果 | 对梁理论 |
|---|---|---|
| tet4 / full | 0.2350 mm | 83.9%（锁死） |
| tet10 / b-bar | 0.2809 mm | **100.2%** |

### 记住这条

> **实体四面体做受弯构件，用 tet10，不要用 tet4。**
> tet4 只适合填充和以受压为主的部位。若必须用一阶单元，改用六面体（`gmsh` 里 recombine），六面体配 `b-bar` 没有这个问题。

这也是为什么**每个算例都值得找一个能手算的量去对**。这次差 16%，是单元选择问题；如果你没校核，这 16% 会一路带进结论里。

---

## 8. 常见问题对照表

| 你看到 | 含义 | 怎么办 |
|---|---|---|
| `error: unknown material 'linear_elasitc'` | 材料名拼错 | 提示里会给"你是不是想写"和全部可用名字 |
| `element.Iz: Extra inputs are not permitted` | YAML 字段名不对 | 冒号前那串就是出错的字段路径 |
| `FORMULATION_UNSUPPORTED: element 'tet4' does not support 'b-bar'` | 单元与格式不匹配 | 用 `vfife mesh` 看该单元支持哪些 |
| `UNKNOWN_REGION: region 'fixd' is referenced but unavailable` | 区域名写错 | 用 `vfife mesh` 对一遍名字。**注意它通常连带报出下面那条 `RIGID_BODY_MODE`** —— 那是后果不是第二个毛病，修好名字两条一起消失 |
| `RIGID_BODY_MODE: ... suppresses only N of 6 ...` | 结构没被约束住，能整体平动或转动 | 若同时有 `UNKNOWN_REGION`，先修那个。否则检查约束是否足够 |
| `MAX_STEPS: ... residual=... > tolerance=...` | 迭代用完了还没收敛 | 先看 `resid` 是在缓慢下降还是卡住。缓慢下降就加 `n_steps`；卡住通常是模型有机构，或非线性材料软化。**梁 / 壳 / 纤维梁的算例若是以前能收敛、现在不能**，多半是半步起步修复带来的预算变化，见 `docs/convergence_debugging.md` |
| `MIXED_MESH_BACKEND: ... runs on the numpy backend only` | 网格里不止一种三维单元，但你要求了 `numba` / `warp` | 混合网格**能算**，只是暂时只有 `numpy` 后端。要用融合内核或 GPU，就把网格统一成一种单元：`.geo` 里加 `Mesh.SubdivisionAlgorithm = 2;` → 全六面体（hexa8 本来就是更好的单元，支持 `b-bar` 且不锁死）；或删掉 `Recombine Surface` / `Transfinite` 行 → 全四面体。注意 `Mesh.RecombineAll = 0` **覆盖不了**显式的 Recombine 行，必须删 |
| `FORMULATION_UNSUPPORTED: element 'tet4' ...`（网格是混合的） | 一个 `formulation` 设置要同时满足所有单元块。`b-bar` 对 hexa8 合法、对 tet4 不合法 | 提示里给的就是各块的交集，通常是 `full`。`vfife mesh` 的 `formulations` 一行也只列交集 |
| `[WARNING] REPEATED_ELEMENTS_IN_ONE_VOLUME: N tet4 cells were written more than once inside a single volume` | **Gmsh 划分失败了，而且它报了错你没看见。** 最常见的原因是 `.geo` 里的 `Recombine Surface {:}` 把**内部界面**也重组了 —— 那个面两侧都是体，属于 non-manifold quad boundary，Gmsh 明确不支持，会刷出成百上千行 `Error : ... non-manifold quad boundaries not supported yet`，**然后照样把文件写出来** | 读取时已经自动合并（节点集相同就是同一个单元，留着会让那一带刚度和质量翻倍），所以**能继续算**。但根子在 `.geo`：只重组外表面，别碰内部界面。见下方"只重组外表面"写法 |
| `[INFO] MERGED_SHARED_ELEMENTS: N cells were listed under more than one entity` | 同一个单元同时属于两个 Physical 组（比如界面上的单元既算外层又算芯部）。MSH 2.2 更是**每个 Physical 组各写一遍** | 不用管。已合并成一个单元，并且**同时属于所有点名过它的组**，区域选择照常可用 |
| `DUPLICATE_NODE` + 混合网格 | 两块单元在界面上**各用各的节点**，坐标重合但编号不同 —— 也就是界面不协调。这种模型能读能算，算出来是**两个几乎独立的物体** | 检查两块的接触面：六面体侧是四边形、四面体侧是三角形时天然不协调。要么让界面两侧都不重组（靠金字塔过渡），要么用 `BooleanFragments` + 共享面保证共节点 |
| 不收敛，而且看不出是模型的哪里 | 可能根本不是模型：荷载不保守、框架不客观、虚拟质量兜不住、或这个状态本身没有稳定平衡 | `python -m vfife probe 算例.yaml --result 结果.h5` 在**停滞的那个状态**上把这四个与单元无关的问题各问一遍，几千自由度以内几秒钟。详见 `docs/convergence_debugging.md` |
| `NONFINITE` | 出现 inf/nan，发散 | 检查是否有约束不足的部位。也可以试着减小 `relaxation_safety`（默认 0.6，可试 0.3，也就是加大虚拟质量）——**但这不是万灵药，两个方向都可能更差**：在材料软化的问题上实测加大质量反而**更早**失败。详见 `docs/convergence_debugging.md` |
| 跑完了但 ParaView 里没有某个场 | `output.fields` 里的名字拼错了，会被静默忽略 | 对照 `vfife describe <单元名>` 里的 `output_fields` |

任何命令加 `--debug` 可以看到完整的 Python 栈，用于排查工具本身的问题。

---

## 9. 再往前一步

这个算例是线弹性、单一荷载步。真正的分析通常需要：

**非线性材料** —— 把 `material` 换掉即可，其余不动：

```yaml
materials:
  concrete:
    type: cdp                          # 混凝土塑性损伤
    E: 3.0e10
    nu: 0.2
    rho: 2400.0
    compression: {source: gb50010, fc: 30.0e6}
    tension:     {source: fracture_energy, ft: 3.0e6, Gf: 120.0}
sections:
  body: {type: solid, material: concrete}
```

用 `python -m vfife list materials` 看全部，`python -m vfife describe cdp` 看某一个的参数和适用范围。

**分级加载**（要画荷载-位移曲线时必须用）：

```yaml
solver:
  load_steps: 20                     # 20 个平衡点，每个都收敛到 tolerance
  max_steps_per_load: 5000
```

**多个分析步 + 幅值曲线**（先自重、再推覆；或往复加载）：

```yaml
amplitudes:
  drift: {cyclic: {peaks: [0.5, 1.0], repeats: 2}}

steps:
  - name: gravity                    # 模型级 loads 在这一步加上，之后保持
  - name: push
    constraints:
      - {region: top, dofs: [1,0,0], value: 0.02, amplitude: drift}
    load_steps: 16
```

**实体面上的弯矩**——实体没有转动自由度，`moment` 加在一个面区域上时
是一个有势的定轴力偶：势能 −|M|·θ(x)，θ 是面上节点最佳拟合刚体转动绕 M 轴转过的
角度。参考构形下它的梯度就是教科书的线性弯曲应力 M z / I（法向分量则是圣维南扭转
的线性剪应力）按节点从属面积化成的节点力；`follower: true` 让这对力偶随面一起转、
大小始终是 M，整圈平滑、无状态；不写 `follower` 就是参考分布冻结的死荷载，力偶随
转角按余弦衰减。两者都是保守的。三维里定向不变的力偶本身不保守（绕略微不同的轴
转一整圈做功不同），这个势只在面转到与 M 轴平行（弯曲卷圆碰不到）时付账：

```yaml
regions:
  tip: {plane: {point: [1, 0, 0], normal: [1, 0, 0]}}
steps:
  - loads:
      - {region: tip, moment: [0.0, 39.27, 0.0], follower: true}   # N m，绕 y
    load_steps: 40
```

**自定义区域**（当 Gmsh 物理组不够用时）：

```yaml
regions:
  base:      {plane: {point: [0,0,0], normal: [0,0,1]}}
  mid_span:  {where: {x: {min: 0.5, max: 0.7}}}
  corner:    {intersect: [base, mid_span]}
  interior:  {difference: [all, surface]}
```

**只重组外表面** —— 想要外表面规整、内部随意时，不要写 `Recombine Surface {:}`：
它会把 `BooleanFragments` 生成的内部界面也重组，Gmsh 直接报错并输出坏网格。
把内部界面减掉：

```gmsh
v() = BooleanFragments { Volume{1}; Delete; }{ Volume{2}; Delete; };
sh() = Abs(Boundary{ Volume{v(0)}; });
co() = Abs(Boundary{ Volume{v(1)}; });
sh() -= co();              // 去掉两块共享的内部界面
Recombine Surface{ sh() };
```

另外要清楚：**`Recombine Surface` 不会把四面体变成六面体**。它只重组*面*，
体网格仍是四面体，只是在每个四边形面下面塞一个金字塔过渡单元。真正的六面体
边界层要用 `Extrude { ... } Layers{n}; Recombine;`。

**混合网格** —— 规则区域用六面体、异形区域用四面体，是真实构件的常规做法，
Gmsh 也几乎必然在两者交界处塞金字塔。这些单元块共用节点，作为**一个**整体求解：
质量、内力按块求和，时间步取各块的最小值，ParaView 里也是一个网格一套场。
两处代价：**(a)** 目前只能用 `numpy` 后端；**(b)** `formulation` 只有一个设置，
必须同时满足所有块，hex+tet 就意味着 `full`。`vfife mesh` 两行都会直接告诉你。

**哪些单元能混？** 七种实体单元的**任意组合**都能建、能算 —— 求解器按块处理，不认单元名。
但两块要真正"焊在一起"，得在界面上提供**同一种面型**：

| | tet4 | hexa8 | wedge6 | pyramid5 | tet10 | hexa20 | wedge15 |
|---|---|---|---|---|---|---|---|
| **tet4** | · | ✗ | 三角3 | 三角3 | ✗ | ✗ | ✗ |
| **hexa8** | ✗ | · | 四边4 | 四边4 | ✗ | ✗ | ✗ |
| **wedge6** | 三角3 | 四边4 | · | 三角3 | ✗ | ✗ | ✗ |
| **pyramid5** | 三角3 | 四边4 | 三角3 | · | ✗ | ✗ | ✗ |
| **tet10** | ✗ | ✗ | ✗ | ✗ | · | ✗ | 三角6 |
| **hexa20** | ✗ | ✗ | ✗ | ✗ | ✗ | · | 四边8 |
| **wedge15** | ✗ | ✗ | ✗ | ✗ | 三角6 | 四边8 | · |

两条规律：**线性单元和二次单元永远不能相接**（二次面上多出的边中点，线性单元不认识）；
**hexa8 和 tet4 直接相接也不协调** —— 一个给四边形面、一个给三角形面。
金字塔和楔形两头都能接，所以它们是标准的过渡单元 —— Gmsh 自动塞金字塔正是干这个。

实测过 hexa8+tet4 直接相接（共用四个角节点、四边形面被两个三角形盖住）：
它**通过分片试验**（残差 2.5e-12），界面位移跳跃约为挠度的 **0.1%**，且随网格加密线性消失
（h 减半，跳跃减半）。也就是说能算、会收敛，但界面在任何有限网格上都不是严格连续的。
要严格协调就让 Gmsh 在中间插金字塔。

**更快的后端** —— 单一单元类型的大模型换 `backend: numba`（CPU 融合内核）或
`backend: warp`（GPU）。支持三后端的只有 `hexa8` / `tet4` / `tet10` / `wedge6`；
`hexa20` / `wedge15` / `pyramid5` 以及全部梁、壳、桁架单元目前仅 `numpy`。
`vfife mesh` 的 `backends` 一行就会告诉你当前网格能用哪些。

参考算例都在 `cases/` 目录，`cases/README.md` 有一张对照表。

---

## 附：完整命令清单

```bash
gmsh -3 -order 2 bracket.geo -o bracket.msh    # 1. 建网格（二次单元）
python -m vfife mesh bracket.msh               # 2. 看网格里有什么
python -m vfife init bracket.yaml --mesh bracket.msh   # 3. 生成骨架
#    编辑 bracket.yaml：材料、监测量
python -m vfife info bracket.yaml              # 4. 检查
python -m vfife run  bracket.yaml              # 5. 求解
#    ParaView 打开 results/bracket.xdmf        # 6. 看场
python -m vfife plot   results/bracket.h5      # 6b. 看曲线
python -m vfife export results/bracket.h5      #     或导出 CSV
#    对反力和一个可手算的量做校核              # 7. 校核
```
