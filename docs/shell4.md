# E03 四节点 Mindlin–Reissner 共旋转壳

> **实验性，1.0 不作承诺。** 第三轮审阅量出：默认 `hourglass_coeff: 0.1` 的稳定项
> 使弯曲偏硬，加密网格消不掉（固支板收敛停在 0.980；纯弯条带单元长宽比 1、5、10 时转角只有
> 精确值的 0.965、0.828、0.706），输出的弯矩也同样偏低；扭曲四边形在薄板上自锁，弯曲分片检验
> 不过（本文后面"分片检验与扭曲网格已覆盖"的说法不成立）。`hourglass_coeff: 0` 时大转动完全精确，
> 规则网格收敛到 1.00。单元公式要改写成 MITC4 才能解决，排在 1.0 之后。三角形 `shell3` 剪切
> 自锁，不要用。

`shell4` 是均质线弹性四边形壳，节点自由度为 `[ux, uy, uz, rx, ry, rz]`。单元在当前局部坐标中扣除整体平移和有限刚体转动，再计算膜、弯曲和横向剪切变形，保持 VFIFE 的运动分解思想。

## 输入

```yaml
materials:
  steel: {type: linear_elastic, E: 2.0e11, nu: 0.3, rho: 7850}
sections:
  plate:
    type: shell
    material: steel
    formulation: selective
    thickness: 0.01
    shear_correction: 0.8333333333333333
    drilling_factor: 1.0
    hourglass_coeff: 0.1
    warpage_tolerance: 0.05
```

- 膜和弯曲采用 2×2 积分；
- 横向剪切采用选择性单点积分；
- 剪切零能模式由 `t²/A` 缩放的 full-minus-reduced 能量项控制，其量级随弯曲刚度 `Et³`，不会成为薄壳中的残余剪切弹簧；
- drilling 约束惩罚“节点法向转角−面内连续体自旋”，刚体面内转动的该差值严格为零；
- 参考面翘曲超过单元平面尺度的 `warpage_tolerance` 时拒绝分析，应细化或重建网格。

支持节点力/矩、均布面压、面力、自重、附加节点质量、地震基底运动、dynamic/relaxation 和 checkpoint。材料目前只有 `linear_elastic`。

## 后端

`solver.backend` 取 `numpy | numba | warp`：

- `numpy`：所有单元一次批处理（einsum），可读的参考实现；
- `numba`：逐单元串行融合核，256 单元屋盖约 2500 迭代/秒（numpy 约 900）。两核上 `prange` 比串行慢 20 倍——每个单元只有几微秒的工作量，线程调度比它贵——所以是串行；
- `warp`：整段分级松弛驻留在设备上。六自由度状态按 (2·节点数, 3) 数组看待，实体后端的节点积分器、边界条件和归约核原样复用；每次迭代只有六个标量回到主机（动能、收敛判据的四个范数、被拒绝的翘曲面数）。`precision: mixed` 把 24×24 局部刚度存为 fp32，几何全部 fp64；`fp64` 用于校验。只接受 `mode: relaxation` 下的位移约束和荷载；基底运动、速度/加速度约束和 `mode: dynamic` 走 `numba`。

节点的局部转角是其取向相对面坐标系的相对转动 log(R·T_n·R₀ᵀ)，取向按每步自旋乘法更新。三者互相钉住（`tests/test_shell_at_scale.py`）：numba 与 warp-fp64 的内力与 numpy 相差 1e-9 以内，warp-mixed 1e-6 以内；设备上的分级松弛与主机在同一迭代收敛、曲线相同。`shell3` 仍走逐单元参考循环。

编号 17 案例（Sze, Liu & Lo 2004 的铰支圆柱壳屋盖，256 单元，位移控制穿过整个 snap-through）是第一个需要这个速度的基准：极限荷载 0.7485 P_max 对文献 0.7420，最终上升段 1.0224 对 1.0208。

## 输出与 ParaView

XDMF 使用 `Quadrilateral` 拓扑。节点输出包括 `displacement`、`rotation`、反力和反力矩；单元输出包括膜力、弯矩、剪力、上下表面 `Tensor6` 应力，以及上下表面 von Mises。工程 Voigt 顺序保持 `[xx, yy, zz, xy, yz, xz]`。

打开 `results/case_10_clamped_square_shell.xdmf` 查看时程，旧版 ParaView 可打开 `_last.xdmf`。用 `Warp By Vector` 显示位移；上下表面应力为壳局部坐标分量，不应误解为全局应力方向。

## 壳—梁混合

`shell_line` 从同一 Gmsh 文件读取 Quad4 壳和 `members` 引用的 Physical Curve。两者共享全部六个节点自由度，不使用插值或惩罚连接。未被成员分组引用的边界 Line2 不会误建成梁。当前混合路径支持梁/杆/连接的节点等效载荷；梁均布 `line_load` 尚未接入，应先换算为节点载荷。

## 验证边界

测试覆盖膜/弯曲 patch、有限刚体运动、恰好六个刚体零模态、畸变四边形、面载荷平衡、薄板锁死、4×4/8×8 网格收敛、简支方板自由振动和壳—梁共享装配。编号 10 案例中心挠度与夹固方板经典解误差 2.73%。当前资格等级为 `verified`，不是复合层合壳、非线性壳、屈曲特征值或接触壳。
