# M02 通用一维分段线性材料

`piecewise_uniaxial` 是杆、嵌入钢筋和未来纤维梁共用的一维材料点。它不属于三维连续体材料，不能赋给实体单元。

## 输入约定

```yaml
materials:
  steel:
    type: piecewise_uniaxial
    E: 2.0e11
    rho: 7850
    tension: [[0.0, 2.0e8], [0.005, 2.6e8], [0.02, 3.2e8]]
    compression: [[0.0, 1.8e8], [0.005, 2.4e8], [0.02, 3.0e8]]
    hardening_mode: combined       # isotropic | kinematic | combined
    kinematic_fraction: 0.5        # 仅 combined 使用
```

两张表均采用 `[累积塑性应变, 屈服应力绝对值]`，首点塑性应变必须为零，横坐标严格递增，应力为正且不递减。`compression` 可省略，此时复制拉伸表。超过末点后按末段斜率外推。

`E` 同时是初始弹性模量和卸载刚度。`isotropic` 扩张拉压屈服边界，不产生 Bauschinger 效应；`kinematic` 平移屈服面；`combined` 按 `kinematic_fraction` 分配两种机制。当前模型为率无关塑性，不包含黏性、强度退化、捏缩、断裂或负刚度软化。

## 状态、能量与重启动

每个材料点保存塑性应变、累积塑性应变、背应力、上一应变/应力和外功密度。输出字段包括：

- `axial_stress`、`axial_strain`；
- `eps_p`、`alpha`、`backstress`；
- `external_work`、`elastic_energy`、`hysteretic_energy`。

`hysteretic_energy = max(external_work - elastic_energy, 0)`，单位为 J/m³；闭合循环末端等于应力—应变环面积。所有六个状态量随 `truss2` checkpoint 保存，恢复时验证材料点签名和数组布局。

## 当前接入范围

- 已接入独立共旋转 `truss2`（NumPy，dynamic/relaxation）；
- 底层 `EmbeddedTruss` 已按统一一维材料点协议调用，可供嵌入钢筋使用；
- `mixed_line` 仍只有一个全局材料块，尚不能安全表达“梁截面弹性、不同杆组分别非线性”。在建立成员级材料分配前不以算例特判放行；
- 弹簧/连接与未来纤维梁可复用本模型，但各自需要先建立材料点所有权和状态输出协议。

资格清单位于 `qualification/piecewise_uniaxial.yaml`，当前等级为 `verified`。

可用专用环境重复生成三种硬化规则的循环路径、CSV 和 SVG：

```powershell
python benchmarks/m02_path.py
```
