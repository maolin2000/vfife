"""The hexa8 kernel cache: it must hit on sameness and MISS on difference.

WHY THIS FILE EXISTS
`Material.numba_point_fn()` returns a freshly built njit device function every
call, so before this cache every solver built from the same material recompiled
the element kernel -- 8-12 s, paid again on every `run_case`. Caching it is
worth about 15-20% of a test run (measured: 33 kernel builds, 27 distinct keys
across five test files).

The cache is keyed on a hash of the point function's CLOSURE, because that is
by construction exactly the set of values the kernel can see. A hand-listed
parameter key would drift the moment someone captured a new variable, and its
failure mode is the one that matters: two DIFFERENT materials silently sharing
a kernel and producing wrong stresses with no error anywhere.

So the tests below are weighted accordingly. One asserts the cache hits. The
rest assert it misses -- on the material, on the formulation, on the quadrature
and on the element -- because a false hit is a silent wrong answer while a
false miss only costs a compile.
"""
import numpy as np
import pytest

from vfife.materials import create

numba = pytest.importorskip("numba")

from vfife.kernels import numba_hexa8 as H          # noqa: E402


def _kernel(mat, formulation="b-bar", n_gp=8, n_nodes=8):
    return H.make_kernel(mat.numba_point_fn(), formulation, n_gp, n_nodes)


def _elastic(E=3.0e10, nu=0.2):
    return create("linear_elastic", E=E, nu=nu, rho=2400.0)


def test_the_same_material_twice_reuses_one_compiled_kernel():
    """The whole point: two solvers, one compile.

    Note that the two materials here are DIFFERENT OBJECTS built from the same
    numbers, which is exactly the case that used to miss -- a solver never sees
    the previous solver's material.
    """
    a, b = _elastic(), _elastic()
    assert a is not b
    assert _kernel(a) is _kernel(b)


def test_a_different_stiffness_does_not_reuse_the_kernel():
    """The failure this cache must never have: a silent wrong answer.

    E is closed over by the point function, so a kernel built for E = 3e10 is
    wrong for E = 6e10. If these two ever share a kernel the model is quietly
    solving a different structure, with nothing raised anywhere.
    """
    assert _kernel(_elastic(E=3.0e10)) is not _kernel(_elastic(E=6.0e10))


def test_a_different_poisson_ratio_does_not_reuse_the_kernel():
    """nu reaches the kernel through lam and G, not as itself -- a key written
    from a parameter list could plausibly have missed it."""
    assert _kernel(_elastic(nu=0.2)) is not _kernel(_elastic(nu=0.35))


@pytest.mark.parametrize("formulation", ["full", "reduced"])
def test_a_different_formulation_does_not_reuse_the_kernel(formulation):
    """Formulation is baked in as a compile-time constant, not read at run
    time, so the kernels are genuinely different code."""
    mat = _elastic()
    assert _kernel(mat, "b-bar") is not _kernel(mat, formulation)


def test_a_different_quadrature_count_does_not_reuse_the_kernel():
    """n_gp is baked in; a reduced-integration kernel cannot serve 8 points."""
    mat = _elastic()
    assert _kernel(mat, "full", n_gp=8) is not _kernel(mat, "full", n_gp=1)


def test_a_different_element_size_does_not_reuse_the_kernel():
    """n_nodes distinguishes hexa8 from tet4/tet10/wedge6, which share this
    source. Sharing a kernel across them would index past the end of the
    element arrays."""
    mat = _elastic()
    assert _kernel(mat, "full", 8, n_nodes=8) is not _kernel(mat, "full", 8,
                                                             n_nodes=4)


def test_a_cached_kernel_gives_the_same_answer_as_a_fresh_one():
    """Sameness of object is not the property that matters -- sameness of
    ANSWER is. This solves the same model twice, the second time on a kernel
    that came out of the cache, and requires the displacement field to agree to
    the last bit rather than merely to a tolerance.
    """
    from vfife.core.mesh import box_mesh
    from vfife.core.solver import Solver

    def solve():
        mesh = box_mesh([0.4, 0.4, 1.0], [2, 2, 4])
        x = mesh.nodes
        mesh.regions["root"] = np.flatnonzero(np.abs(x[:, 2]) < 1e-12)
        mesh.regions["tip"] = np.flatnonzero(np.abs(x[:, 2] - 1.0) < 1e-12)
        problem = {
            "element": {"type": "hexa8", "formulation": "b-bar"},
            "constraints": [{"region": "root", "dofs": [1, 1, 1]}],
            "loads": [{"region": "tip", "vector": [1.0e5, 0.0, 0.0]}],
            "solver": {"mode": "relaxation", "n_steps": 200,
                       "backend": "numba", "tolerance": 1.0e-30},
            "output": {"monitors": {}},
        }
        s = Solver(problem, mesh, _elastic(), log=lambda *_: None)
        try:
            s.run()
        except Exception:                    # tolerance 1e-30 never converges
            pass
        return s.u.copy()

    first, second = solve(), solve()
    assert np.array_equal(first, second)


def test_an_array_valued_capture_is_part_of_the_key():
    """CDP closes over four curve arrays per curve. Two materials differing
    only inside those arrays must not share a kernel, and only a key that
    hashes array CONTENTS can tell them apart -- identity would not, because
    the arrays are rebuilt per material.
    """
    def cdp(fc):
        return create("cdp", E=3.0e10, nu=0.2, rho=2400.0, l_ref=0.1,
                      compression={"source": "gb50010", "fc": fc},
                      tension={"source": "fracture_energy", "ft": 3.0e6,
                               "Gf": 120.0})
    a, b = cdp(30.0e6), cdp(40.0e6)
    assert _kernel(a, "full", 8) is not _kernel(b, "full", 8)
    assert _kernel(a, "full", 8) is _kernel(cdp(30.0e6), "full", 8)
