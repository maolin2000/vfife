# M04 Viscoelastic (Prony series)

`viscoelastic` is a linear viscoelastic Prony series, for `dynamic` mode only (integration in physical time), `numpy` backend; it cannot be used in the pseudo-time of `relaxation`.

## Input

```yaml
materials:
  polymer:
    type: viscoelastic
    E0: 3.0e9
    nu: 0.35
    rho: 1200
    prony: [[0.2, 0.05], [0.15, 0.5], [0.1, 5.0]]  # one [g_i, tau_i] per row
```

`E(t)=E0*(1 - sum g_i*(1-exp(-t/tau_i)))`: `g_i` are the shear relaxation weights and `tau_i` the relaxation times. `E0` is the instantaneous modulus; the long-term modulus `E_inf=E0*(1-sum g_i)` must be positive.

- With `g_i=0` it reduces to `linear_elastic`;
- in `dynamic` mode it evolves with the physical `dt`; `relaxation` mode (pseudo-time `dt=1`) is refused, so that pseudo-time is not mistaken for real relaxation.

Each point keeps `n_prony` history strains, which `checkpoint` saves with `state`. Qualified `verified` (`qualification/viscoelastic.yaml`).
