# 统一自由度与混合装配协议

`vfife/base/dof.py`（1.22 之前是 `vfife/core/dof.py`） 是节点物理分量到全局方程编号的唯一映射层。单元不得自行假定 `6*node+dof`，而应通过 `DofLayout` 请求自由度映射。

## 布局

```python
solid = DofLayout.translations(n_nodes)  # ux, uy, uz
frame = DofLayout.frame(n_nodes)         # ux, uy, uz, rx, ry, rz

member_map = frame.element_dofs([node_i, node_j])
translation_map = frame.node_dofs(nodes, TRANSLATION)
rotation = frame.field(global_vector, ROTATION)
```

该协议允许：

- 实体和独立杆保留紧凑的三自由度节点状态；
- 梁和混合杆系使用六自由度状态；
- coupled 模型只把实体力、质量装入共享平动分量；
- 无梁节点的旋转槽保持非活动，而不是赋予虚假微小质量；
- 新壳、弹簧、连接和界面单元按名称请求所需分量。

`SparseAssembler` 通过单元显式 equation map 装配矩阵，并校验局部矩阵尺寸。`DofLayout.add_nodal()` 负责按命名分量装配节点载荷或耦合力。

## VFIFE 几何原则

本协议只统一自由度所有权和装配，不改变任何单元的变形计算。梁仍先扣除刚体平移和转动，再由局部变形求等效内力；实体与杆同样保持现有共旋转/VFIFE 路径。因此混合装配不会把传统小位移全局刚度假设引入单元内力更新。

梁的 12×12 局部—整体变换使用 `repeated_rotation()`。它生成确定的稠密块旋转矩阵，替代 SciPy 正在改变返回类型的 `sparse.block_diag`，消除了原专项测试中的 36,303 条弃用警告。

## 当前接入状态

- 实体 `Solver`：登记三平动布局；
- `TrussSolver`：登记三平动布局；
- `BeamSolver`：自由度、节点载荷、质量访问、监视器及矩阵装配使用统一布局；
- `MixedStructuralSolver`：梁/杆共用 equation map 和稀疏装配器；
- `CoupledSolver`：实体质量与内力只装配到命名平动分量，旋转活动性通过命名旋转分量确定。

## 新单元准入

新增单元必须：

1. 在能力注册中声明物理自由度名称；
2. 使用 `DofLayout` 生成 element equation map；
3. 使用统一装配接口，不得复制 stride 算术；
4. 对不拥有的自由度保持非活动，不得用微小质量掩盖机构；
5. 通过局部矩阵尺寸、节点力平衡、刚体运动、混合装配和经典解测试。
