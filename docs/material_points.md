# 统一材料—积分点协议

`vfife/materials/points.py` 管理材料从“用户参数”进入“单元积分点”的共同生命周期。本构公式仍位于各材料模块中；该协议只统一运动学、状态、网格绑定、后端和重启动契约。

## 初始化入口

```python
state, contract = initialise_points(
    material,
    n_points,
    kinematics="solid_3d",       # 或 uniaxial
    backend="numpy",
    char_length=point_lengths,
)
```

初始化会在建立计算后端前检查：

- 材料是否支持指定运动学和后端；
- 积分点数量是否合法；
- 状态字段 slice 是否完整、无重叠且不越界；
- `init_state()` 是否返回准确的 `(n_points, n_state)`；
- 特征长度是否逐点、有限且为正；
- 正则化材料是否确实获得了特征长度；
- 材料是否提供有限且为正的波速。

## 分量约定

三维实体使用工程 Voigt 应变：

```text
[exx, eyy, ezz, gxy, gyz, gxz]
```

及对称 Cauchy 应力：

```text
[sxx, syy, szz, sxy, syz, sxz]
```

一维杆/钢筋声明 `uniaxial`，物理分量为 `axial_strain` 和 `axial_stress`。现有底层设备缓冲仍可使用六分量打包，但该实现细节不再被误认为三维材料能力。

## 状态与重启动

`MaterialPointContract` 记录：

- 材料名称与运动学；
- 应力、应变分量；
- 状态宽度和字段；
- 后端；
- 特征长度策略；
- 波速和稳定时间步换算；
- 状态布局签名。

实体和独立杆 checkpoint 现在保存 `material_signature`。恢复时会验证签名、状态数组尺寸和有限性；不同材料状态布局不能再被静默装入当前模型。旧 checkpoint 没有签名时仍可读取，但必须通过状态尺寸检查。

## 当前接入路径

- 所有实体积分点通过统一入口初始化；
- 独立杆积分点以 `uniaxial` 运动学初始化；
- 嵌入钢筋积分点以 `uniaxial` 运动学初始化；
- CDP 的 crack-band 特征长度由统一入口绑定；
- NumPy、Numba 和 Warp 的能力在分配状态/启动设备前检查；
- `stable_time_step(length)` 统一由材料波速给出局部上限信息。

梁当前使用线弹性截面刚度，不调用三维材料积分点更新；其材料兼容性由能力注册表限制为 `linear_elastic`。未来纤维梁截面应由截面层持有多个 `uniaxial` 材料积分点，而不是让梁直接调用三维本构。

## 新材料准入

新增材料必须：

1. 明确声明 `solid_3d`、`uniaxial` 或未来新增的运动学；
2. 完整命名所有状态列；
3. 对正则化模型声明 `regularised=True` 并实现逐点特征长度绑定；
4. 提供波速或等价稳定性信息；
5. 对声明的每个后端通过同一路径和步长的一致性测试；
6. 验证单调、卸载、反向、多轴、能量和重启动路径；
7. 不得通过 `**ignored` 静默宣称尚未实现的能力。
