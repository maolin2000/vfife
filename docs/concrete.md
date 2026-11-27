# Concrete01 / Concrete02: uniaxial concrete

Both keep the names and parameters of the OpenSees materials of the same name and are implemented from the original papers (`concrete01`: the Kent–Scott–Park envelope with Karsan–Jirsa unloading; `concrete02`: Yassin's model). They agree with the output of the OpenSees program to round-off (`tests/test_opensees_reference.py`). They are for the fibres of `beam2_fiber` (`uniaxial`) only and cannot be assigned to continuum elements.

| model | source | tension | key parameters | state |
|---|---|---|---|---|
| `concrete01` | Kent–Scott–Park envelope, Karsan–Jirsa unloading | none (zero tensile stress past `epsc0`) | `fpc/epsc0/fcu/epscu` (`fpc<0, epsc0<0` in compression) | `n_state=5` |
| `concrete02` | Yassin (1994), Sze & Filippou | yes (`ft/Ets` linear softening) | `fpc/epsc0/fcu/epscu/rat/ft/Ets` | `n_state=5` |

## Input

```yaml
materials:
  c01: {type: concrete01, fpc: 35e6, epsc0: 0.002, fcu: 7e6, epscu: 0.0035, rho: 2500}
  c02: {type: concrete02, fpc: 35e6, epsc0: 0.002, fcu: 7e6, epscu: 0.0035, ft: 3.2e6, Ets: 1.2e9, rat: 0.1, rho: 2500}
```

Both accept the old aliases `fc/eps_c0/fcu/eps_cu/E`. The envelope of `concrete01` is Hognestad's parabola with a softening plateau in compression, and its unloading slope degrades as `CunloadSlope`; `concrete02` adds the tensile branch `ft` with `Ets` softening and the unloading stiffness ratio `rat`. Both are rate-independent and safe to checkpoint; in a mixed section they are integrated with `steel02` group by group (`fiber_group`).

Both are qualified `verified` (`qualification/concrete0*.yaml`); `case_11` (Tanaka & Park 1990, specimen 1) uses `concrete02`. **That case was once taken to show that its unloading is too steep (0.32 times the dissipated energy); the conclusion was wrong.** The cause was the solver committing the fibres' material state at every iteration; with that fixed the energy is 1.30 times. The law was not disproved.

`update()` of both runs through a compiled point loop (module-level `@njit(cache=True)`, falling back to `_update_reference` without numba, with the envelope functions inlined). Measured on 384 fibre points: 581 → 4.6 µs (126 times), **bit-identical** to the reference implementation; `tests/test_uniaxial_numba.py` checks the equivalence along a history with crushing and repeated cracking.
