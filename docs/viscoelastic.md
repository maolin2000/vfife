# M04 黏弹性 Prony

`viscoelastic` 为线性黏弹性 Prony 级数，仅支持 `dynamic` 模式（物理时间积分），`numpy` 后端，不可在 `relaxation` 伪时间内使用。

## 输入

```yaml
materials:
  polymer:
    type: viscoelastic
    E0: 3.0e9
    nu: 0.35
    rho: 1200
    prony: [[0.2, 0.05], [0.15, 0.5], [0.1, 5.0]]  # [g_i, tau_i] 每行
```

`E(t)=E0*(1 - sum g_i*(1-exp(-t/tau_i)))`，`g_i` 为剪切松弛权重，`tau_i` 为松弛时间。`E0` 为瞬时模量，长时间模量 `E_inf=E0*(1-sum g_i)` 必须为正。

- 弹性极限 `g_i=0` 时退化为 `linear_elastic`；
- `dynamic` 模式下按物理 `dt` 演化，`relaxation` 模式（`dt=1` 伪时间）拒绝使用以免误解释为真实松弛。

状态每点保存 `n_prony` 个历史应变，`checkpoint` 随 `state` 一并保存。资格 `verified`（`qualification/viscoelastic.yaml`）。
