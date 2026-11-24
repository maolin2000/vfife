"""Concrete02: Yassin (tension + compression)."""
import numpy as np
import pytest
from vfife.materials.base import create
from vfife.catalog import library

def _follow(mat, history):
    state = mat.init_state(1)
    stresses=[]
    for eps in history:
        strain=np.zeros((1,6)); strain[0,0]=eps
        stresses.append(mat.update(strain, state, 1.0)[0,0])
    return np.array(stresses), state

def test_concrete02_envelope():
    m=create("concrete02", fpc=30e6, epsc0=0.002, fcu=6e6, epscu=0.0035, ft=3e6, Ets=1.5e9)
    s,_=_follow(m, [0.00005])
    assert s[0] > 0
    s2,_=_follow(create("concrete02", fpc=30e6, epsc0=0.002, fcu=6e6, epscu=0.0035, ft=3e6, Ets=1.5e9), [-0.002])
    assert abs(s2[0]+30e6) < 1e6

def test_concrete02_tension_softening():
    m=create("concrete02", fpc=30e6, epsc0=0.002, fcu=6e6, epscu=0.0035, ft=3e6, Ets=1.0e9)
    # ft=3MPa, Ec0=30GPa, eps0=0.0001, epsu=0.0031
    s,_=_follow(m, [0.0001])
    assert abs(s[0]-3e6) < 0.2e6
    s2,_=_follow(create("concrete02", fpc=30e6, epsc0=0.002, fcu=6e6, epscu=0.0035, ft=3e6, Ets=1.0e9), [0.002])
    assert 0 < s2[0] < 2e6

def test_concrete02_catalog():
    lib=library()
    assert "concrete02" in lib["materials"]
    assert "concrete02" in lib["elements"]["beam2_fiber"]["compatible_materials"]

def test_concrete02_unload_reload():
    m=create("concrete02", fpc=30e6, epsc0=0.002, fcu=6e6, epscu=0.0035, rat=0.1, ft=3e6, Ets=1e9)
    hist=[0, -0.003, -0.001, -0.002, 0.0005, 0]
    s,_=_follow(m, hist)
    # after compression, tension reloading should give positive stress
    assert s[4] >= 0

def test_concrete02_step_convergence():
    target=-0.002
    m1=create("concrete02", fpc=30e6, epsc0=0.002, fcu=6e6, epscu=0.0035)
    a,_=_follow(m1, [target])
    m2=create("concrete02", fpc=30e6, epsc0=0.002, fcu=6e6, epscu=0.0035)
    b,_=_follow(m2, np.linspace(0,target,51))
    assert a[-1]==pytest.approx(b[-1], rel=1e-12)
