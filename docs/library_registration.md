# 单元与材料注册协议

单元库、材料库、预检、CLI 和未来 GUI 共享一套机器可读的工程能力声明。声明的目的不是替代数值实现，而是让一个新模型在进入求解器之前就能回答“它能和什么组合、需要什么参数、产生什么结果、验证到什么程度”。

## 材料注册

材料继续使用 `@register`，同时声明工程元数据：

```python
@register(
    "my_material",
    kinematics=("solid_3d",),
    backends=("numpy",),
    analysis_modes=("dynamic", "relaxation"),
    output_fields=("stress", "strain", "history"),
    nonlinear=True,
    regularised=False,
    verification="constitutive_path",
    caveat="...",
)
class MyMaterial(Material):
    ...
```

构造函数参数会自动进入 `MaterialDef.parameters`；`state_names` 自动进入状态字段。不得在 CLI 或 GUI 中另建材料名称列表。

当前登记材料：`linear_elastic`、`j2`、`chaboche`、`cdp`、`rebar`、`piecewise_uniaxial`。

## 单元注册

`ElementCapability` 统一描述实体和杆系单元：

- family 与网格拓扑；
- 节点自由度；
- formulation；
- 材料运动学与当前兼容材料；
- 后端和分析模式；
- 载荷与输出字段；
- 几何非线性方法；
- 验证等级和已知限制。

连续体的插值与积分仍由 `ElementDef` 定义，能力由该定义生成；杆、梁、混合及耦合单元使用同一个 `ElementCapability` 协议。

当前登记单元：`hexa8`、`tet4`、`tet10`、`wedge6`、`truss2`、`beam2`、`beam2_timoshenko`、`shell4`、`shell_line`、`connector2`、`mixed_line`、`coupled`。

所有现有单元均明确声明 `geometric_nonlinearity: co_rotational_vfife`，避免未来新增单元偏离 VFIFE 的刚体运动扣除思想而未被发现。

## 公共访问与组合校验

```python
from vfife.catalog import library, compatibility

catalog = library()  # 可直接 JSON 序列化，schema_version = 1
ok, reasons = compatibility("beam2", "j2", backend="warp")
```

`preflight.check()` 已调用同一 `compatibility()`，不支持的单元—材料—后端—分析模式组合会产生 `CAPABILITY_MISMATCH`，无需初始化求解器或 CUDA。

CLI 直接公开同一目录：

```powershell
vfife list
vfife list materials --format json
vfife describe beam2
vfife describe cdp --format json
vfife capabilities --element hexa8 --material cdp --backend warp
vfife capabilities --element beam2 --backend warp --show-unsupported
```

`capabilities` 默认只显示支持的组合；`--show-unsupported` 会同时给出每个拒绝原因。所有三个命令都只读取注册信息，不导入网格、不构造求解器，也不会触发 CUDA 初始化。`--format json` 的顶层携带目录 `schema_version`，供脚本和未来 GUI 使用。

## 新模型准入规则

新增单元或材料时必须：

1. 注册完整元数据，不允许只修改 schema 的 Literal；
2. 为声明的每个后端提供实现和一致性测试；
3. 给出验证等级，不得把只有单元测试的模型标为 benchmark；
4. 明确输出字段、状态字段及其物理位置；
5. 在已知限制中记录尚未支持的组合；
6. 增加目录完整性和非法组合测试。

目前兼容材料仍由单元能力显式列出。这是刻意的保守策略：新材料注册后不会自动获得所有实体或杆系单元的支持，必须在完成对应验证后显式放行。

材料进入单元后的积分点生命周期见 [material_points.md](material_points.md)。注册表说明“允许什么组合”，积分点协议保证“该组合如何获得一致的状态、特征长度、稳定性信息和重启动身份”。

模型从“登记”升级为“已验证/基准化/生产可用”的规则见 [qualification_gates.md](qualification_gates.md)。注册表中的文字标签不再单独构成工程准入证据。
