"""The package's numba cache is one thing (kernels/_cache.py).

Numba (before 0.68) keys a cached function on its own source file alone: a
cached kernel that calls a function of another file keeps that function's
old machine code when the other file changes. The compiled step ran the
connectors of the day before on the user's machine -- the new damper law
fell through to friction, 0.1 m off, no error. A stamp of every kernel and
material source is kept beside the cache; a mismatch removes every cache
file before a kernel is imported.
"""
import os

import pytest

from vfife.kernels import _cache


@pytest.fixture
def package(tmp_path, monkeypatch):
    k = tmp_path / "kernels"; m = tmp_path / "materials"
    for d in (k, m):
        (d / "__pycache__").mkdir(parents=True)
    (k / "numba_step.py").write_text("a = 1\n")
    (k / "numba_connector.py").write_text("b = 1\n")
    (m / "rebar.py").write_text("c = 1\n")
    monkeypatch.setattr(_cache, "_folders", lambda: [str(k), str(m)])
    monkeypatch.delenv("VFIFE_KEEP_NUMBA_CACHE", raising=False)
    return k, m


def _fill(k, m):
    for d, names in ((k, ("numba_step.dynamic_chunk-1.py312.nbi", "numba_step.dynamic_chunk-1.py312.1.nbc")),
                     (m, ("rebar.point-9.py312.nbi",))):
        for n in names:
            (d / "__pycache__" / n).write_bytes(b"x")
    (k / "__pycache__" / "numba_step.cpython-312.pyc").write_bytes(b"p")


def _cached(k, m):
    return sorted(n for d in (k, m) for n in os.listdir(d / "__pycache__") if n.endswith((".nbi", ".nbc")))


def test_a_cache_from_other_sources_is_removed_once(package):
    pytest.importorskip("numba")
    k, m = package
    _fill(k, m)
    assert _cache.refresh() is True                    # no stamp: whose cache is this?
    assert _cached(k, m) == []
    assert (k / "__pycache__" / "numba_step.cpython-312.pyc").exists()     # Python's own is not ours
    _fill(k, m)
    assert _cache.refresh() is False and len(_cached(k, m)) == 3           # same sources: kept


def test_a_change_in_a_file_another_kernel_calls_removes_them_all(package):
    pytest.importorskip("numba")
    k, m = package
    _cache.refresh()
    _fill(k, m)
    (k / "numba_connector.py").write_text("b = 2  # a new law\n")           # NOT the cached function's file
    assert _cache.refresh() is True
    assert _cached(k, m) == []
    _fill(k, m)
    (m / "rebar.py").write_text("c = 22\n")
    assert _cache.refresh() is True and _cached(k, m) == []


def test_it_can_be_told_to_keep_them(package, monkeypatch):
    pytest.importorskip("numba")
    k, m = package
    _fill(k, m)
    monkeypatch.setenv("VFIFE_KEEP_NUMBA_CACHE", "1")
    assert _cache.refresh() is False and len(_cached(k, m)) == 3


def test_the_real_package_is_stamped():
    pytest.importorskip("numba")
    import vfife.kernels  # noqa: F401  (its import refreshes)
    stamp = os.path.join(os.path.dirname(_cache.__file__), "__pycache__", _cache._STAMP)
    assert os.path.exists(stamp)
    assert open(stamp, encoding="ascii").read().strip() == _cache._signature()
