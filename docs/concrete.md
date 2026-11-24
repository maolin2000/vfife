# Concrete01 / Concrete02 一维混凝土

两者沿用 OpenSees 同名材料的名字和参数，按原论文实现（`concrete01`：Kent–Scott–Park 包络线加 Karsan–Jirsa 卸载；`concrete02`：Yassin 模型），与 OpenSees 程序的输出在舍入误差内一致（`tests/test_opensees_reference.py`）；仅用于 `beam2_fiber` 纤维（`uniaxial`），不可赋给连续体单元。

| 模型 | 来源 | 受拉 | 关键参数 | 状态 |
|---|---|---|---|---|
| `concrete01` | Kent-Scott-Park 包络线，Karsan-Jirsa 卸载 | 无（`>epsc0` 后拉应力恒 0） | `fpc/epsc0/fcu/epscu`（`fpc<0, epsc0<0` 为压） | `n_state=5` |
| `concrete02` | Yassin（1994），Sze & Filippou | 有（`ft/Ets` 线性软化） | `fpc/epsc0/fcu/epscu/rat/ft/Ets` | `n_state=5` |

## 输入

```yaml
materials:
  c01: {type: concrete01, fpc: 35e6, epsc0: 0.002, fcu: 7e6, epscu: 0.0035, rho: 2500}
  c02: {type: concrete02, fpc: 35e6, epsc0: 0.002, fcu: 7e6, epscu: 0.0035, ft: 3.2e6, Ets: 1.2e9, rat: 0.1, rho: 2500}
```

均兼容旧别名 `fc/eps_c0/fcu/eps_cu/E`。`concrete01` 包络为 Hognestad 抛物线 + 压软化平台，卸载按 `CunloadSlope` 退化；`concrete02` 另含受拉上升 `ft` 与 `Ets` 软化及 `rat` 卸载刚度比。两者率无关，`checkpoint` 安全，混合截面中与 `steel02` 按 `fiber_group` 分组积分。

资格均为 `verified`（`qualification/concrete0*.yaml`），`case_11`（Tanaka & Park 1990 试件 1）采用 `concrete02`。**曾据该算例判定其卸载偏陡（耗能仅 0.32 倍），那个结论是错的**——真因是求解器把纤维材料状态逐迭代提交；修正后耗能为 1.30 倍。本构没有被证伪。

两者的 `update()` 均走编译点循环（模块级 `@njit(cache=True)`，numba 不可用时回退到 `_update_reference`，包络函数一并内联）。实测 384 个纤维点 581 → 4.6 µs（126 倍），与参照实现**位级一致**；等价性由 `tests/test_uniaxial_numba.py` 沿含压碎与开裂反复的历程核对。
