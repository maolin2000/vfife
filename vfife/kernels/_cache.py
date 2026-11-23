"""Numba's on-disk cache, and the kernels that call one another.

Numba keys a cached function on the source file IT is defined in (its size
and modification time). A cached kernel that calls a function of another
file -- the compiled step (`numba_step`) calls the connectors'
(`numba_connector`), the frames', the fibres' -- carries that function's
machine code inside its own cache entry, and numba versions before 0.68 do
not look at the other file: edit `numba_connector.py`, and the compiled step
goes on running the connectors as they were. Found on the user's machine
(numba 0.67) the day the damper and the pendulum laws went into the
connectors: the loop ran the new laws, the compiled step a
cache from the day before, in which their codes fell through to friction --
a 0.1 m difference with no error.

So the cache of the whole package is one thing: a stamp of every source
file of `kernels/` and `materials/` (name, size, mtime) and of the numba
version is kept beside the cache, and when it does not match, every cache
file of the two folders is removed before any kernel is imported. An
installed wheel never changes, so this costs a directory listing; a
checkout that moves recompiles once.
"""
from __future__ import annotations

import hashlib
import os

_STAMP = "vfife_kernels.stamp"


def _folders():
    here = os.path.dirname(os.path.abspath(__file__))
    return [here, os.path.join(os.path.dirname(here), "materials")]


def _signature() -> str:
    h = hashlib.blake2b(digest_size=16)
    try:
        import numba
        h.update(str(numba.__version__).encode())
    except Exception:                                   # pragma: no cover - no numba: nothing is cached
        return ""
    for folder in _folders():
        try:
            entries = [e for e in os.scandir(folder) if e.name.endswith(".py")]
        except OSError:
            continue
        for e in sorted(entries, key=lambda x: x.name):
            st = e.stat()
            h.update(f"{os.path.basename(folder)}/{e.name}:{st.st_size}:{st.st_mtime_ns};".encode())
    return h.hexdigest()


def refresh() -> bool:
    """Remove the package's numba cache files when its sources have changed
    since they were written. True when a wipe was made. Never raises: a
    folder that cannot be written is one numba does not cache into either
    (it falls back to a per-user folder keyed by this path, which a new
    install does not share)."""
    if os.environ.get("VFIFE_KEEP_NUMBA_CACHE", "") == "1":
        return False
    sig = _signature()
    if not sig:
        return False
    stamp = os.path.join(_folders()[0], "__pycache__", _STAMP)
    try:
        with open(stamp, "r", encoding="ascii") as f:
            if f.read().strip() == sig:
                return False
    except OSError:
        pass
    clean = True
    wiped = False
    for folder in _folders():
        cache = os.path.join(folder, "__pycache__")
        try:
            names = os.listdir(cache)
        except OSError:
            continue
        for name in names:
            if name.endswith((".nbi", ".nbc")):
                try:
                    os.remove(os.path.join(cache, name))
                    wiped = True
                except FileNotFoundError:
                    pass                                # another process got there first
                except OSError:
                    clean = False                       # open elsewhere (Windows): try again next time
    if clean:
        try:
            os.makedirs(os.path.dirname(stamp), exist_ok=True)
            tmp = f"{stamp}.{os.getpid()}"
            with open(tmp, "w", encoding="ascii") as f:
                f.write(sig)
            os.replace(tmp, stamp)
        except OSError:
            pass
    return wiped
