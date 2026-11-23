# M01 正交各向异性线弹性

`orthotropic` 为三维正交线弹性材料，适用于木材、轧制钢板、单层复合材料等方向性构件。仅支持 `hexa8/tet4/tet10/wedge6/pyramid5/hexa20/wedge15` 等连续体单元，`numpy` 后端。

## 输入

```yaml
materials:
  laminate:
    type: orthotropic
    E1: 12.0e9
    E2: 8.0e9
    E3: 10.0e9
    G12: 4.5e9
    G23: 3.8e9
    G13: 4.0e9
    nu12: 0.3
    nu13: 0.25
    nu23: 0.28
    rho: 800
```

9 个独立常数（`E1-3, G12/G23/G13, nu12/13/23`），对称柔度矩阵自动补全，刚度矩阵求逆前做正定性检查（`E>0, G>0, |nu|<0.5, 行列式>0`）。材料局部坐标暂与全局一致，后续通过 `orientation` 支持方向旋转。

各向同性退化 `E1=E2=E3, G=E/2(1+nu), nu12=nu` 时与 `linear_elastic` 在机器精度内一致（`test_isotropic_limit_matches_linear_elastic`）。

资格等级 `verified`（`qualification/orthotropic.yaml`），`numpy` 仅，暂不含失效准则与层合壳。
