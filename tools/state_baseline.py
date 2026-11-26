"""The shipped cases' answers to the bit, kept to compare a refactoring with
(before P2 of the architecture review).

The contracts say whether an answer is still right to its tolerance; a
refactoring that is meant to change nothing must give the same answer to
the bit, and a contract within 1e-3 does not see a change of 1e-9. This
solves every shipped case as `tools/check_all.py` solves it (one process
per case, single-threaded but THREADED_CASES, the backend the case declares
mapped to the CPU) and keeps, for each, the SHA-256 of every dataset and
every attribute of its result file and of its checkpoint (the final state
in float64):

    python tools/state_baseline.py --write 1.18.0      # results/state_baseline/1.18.0/
    python tools/state_baseline.py --compare 1.18.0    # solve again, say what differs
    python tools/state_baseline.py --compare 1.18.0 --cases case_24g case_36

Not hashed: the attributes that name the run rather than its answer (SKIP:
its id, its date, the version, the files' paths) and a dataset holding a
wall time (WALL; there is none today). `--write` keeps a copy of each
checkpoint, and of each result file up to 50 MB, so that `--compare` can
say by how much a dataset that differs differs. A baseline is not
overwritten without `--force`; it is written beside and renamed at the end.

`--compare` gives each case the threads the baseline recorded for it (the
threaded kernels give the serial answer to the bit, but that is what the
comparison is for). Bitwise holds on one machine with one numba cache:
compare on the machine that wrote the baseline. A cold numba cache once
gave 1e-12 on case_04 -- compare twice before believing a
difference that small. The report is
results/state_baseline/<name>_compare_<time>.md; the exit code is non-zero
when a case differs or fails.

A change that is MEANT to change answers is compared with
`--tolerance`: a case that differs is then judged on its final state alone
-- the last frame of the result, each monitor's last value and the
checkpoint's `q` -- each quantity against the row of TOLERANCES its name
matches (a dataset by its name, a monitor by its `kind`): |a - b| <= rtol
|b| + floor S, S the largest value of that class of quantity
(displacements, rotations, forces, stresses) in the baseline's final
state, so that a value near zero is held to an absolute tolerance set by
the size of what the model carries. A case that relaxes is held to no less
than RELAXED (10) times the residual it relaxes to (its `tolerance`): its
answer is defined to that, and carries it through its history. The path a relaxation took to that state (its iterations, its
residual and kinetic series, its fictitious mass) is not judged. A case
within tolerance is reported WITHIN, and does not fail the compare.

A case loaded to collapse (COLLAPSES) is judged on its history up to the
collapse instead: every monitor at every frame up to that time, against the
same rows. What follows the collapse -- a structure falling apart -- moves
under any change of round-off and is not judged (case_24d: 1e-11 up to
t = 54 s, 1% after). This needs the baseline's monitors: `--write` keeps
them for every case (`<case>.monitors.h5`, small even when the result file
is not kept); for a baseline written before that, `--backfill-monitors NAME
--tree DIR --cases ...` solves the cases again with a checkout of the
baseline's commit in DIR, checks that the result and the checkpoint hash
to the baseline's, and keeps the monitors.

Every compare also reports the cases that ran more than SLOWER (20 %) and
more than SLOWER_S (5 s) longer than the baseline did: the performance
gate. It fails the compare only with `--perf-gate` (a shared machine's
times wander; on the machine that wrote the baseline, use it).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import glob
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import conftest  # noqa: E402,F401  (pins the OpenBLAS kernel as the tests do, before NumPy)
import check_all  # noqa: E402  (its case list, its threads, its environment)

#: datasets that hold a wall time, by the last part of their name
WALL = {"wall_time", "wall_s", "elapsed", "elapsed_s"}
#: attributes that name the run, not its answer
SKIP = {"run_id", "created_utc", "vfife_version", "restarted_from", "checkpoint_file", "result", "case_yaml"}
#: result files larger than this are hashed, not copied
COPY_LIMIT = 50 * 2**20

#: The per-quantity tolerances of `--tolerance`: (class, patterns, rtol,
#: floor); the first row whose pattern is in the quantity's name wins -- the
#: last part of a dataset's name, a monitor's `kind`. The floor is a
#: fraction of the largest finite value of the CLASS in the baseline's final
#: state -- a reaction of 3 N at one support of a 1e7 N bridge is zero, to a
#: relaxation converged to 1e-6 of its forces. "classes" (fibre damage, a
#: crack's state) are counted instead: the share of entries that changed; an
#: integer or boolean dataset (a member failed, the connectivity) is held
#: exactly. Set for an answer that moved to within what its relaxation
#: converged to, carried through its history; a change meant to change
#: nothing is still held to the bit.
TOLERANCES = [
    ("rotation", ("rotation", "theta", "drift"), 1e-5, 1e-5),
    ("displacement", ("displacement", "checkpoint/q", "deflection", "velocity", "acceleration"),
     1e-5, 1e-5),
    ("classes", ("damage", "crack"), 1e-3, None),
    # strain before stress, and its own class: a strain held to a floor set by the
    # largest STRESS (1e7 Pa) would never be judged
    ("strain", ("strain", "eps"), 1e-3, 1e-3),
    ("stress", ("stress", "von_mises", "sig"), 1e-3, 1e-3),
    ("force", ("force", "moment", "shear", "bending", "axial", "torsion", "reaction", "section"),
     1e-4, 1e-4),
    ("energy", ("energy", "work"), 1e-3, 1e-3),
    ("other", ("",), 1e-5, 1e-5),
]
#: cases loaded to collapse: {case: the time up to which they are judged}.
#: The time is the last frame before the collapse, read off the run (the
#: midspan deflection jumps within a second of it); what follows is a
#: structure falling apart, which no change of round-off leaves in place.
COLLAPSES = {
    "case_24c_hzb_girder_collapse": 46.0,   # w_mid_middle 0.29 m at 46 s, 0.81 m at 47 s
    "case_24d_hzb_girder_piers": 54.0,   # w_mid_middle 0.71 m at 54 s, 3.9 m at 55 s
}
#: the performance gate: slower than the baseline by this fraction AND these seconds
SLOWER, SLOWER_S = 0.20, 5.0


def _cpu_backend(declared):
    # as tests/test_cases.py::_cpu_backend
    if declared == "warp":
        declared = "numba"
    if declared == "numba":
        try:
            import numba  # noqa: F401
        except ImportError:
            return "numpy"
    return declared


def solve_one(stem, out):
    """Solve one case as tests/test_cases.py does; print its status."""
    from vfife import api
    source = api.load_case(str(ROOT / "cases" / (stem + ".yaml")))
    source.set("solver.backend", _cpu_backend(source.get("solver").get("backend", "numpy")))
    source.get("solver").pop("device", None)
    threads = os.environ.get("VFIFE_CASE_THREADS")
    if threads:
        source.set("solver.threads", int(threads))
    outcome = api.run_case(source, output=out)
    print("STATUS", outcome.status, outcome.converged)
    return 0 if outcome.converged else 1


def _bytes(a):
    import numpy as np
    if isinstance(a, np.ndarray):
        if a.dtype.kind == "O":                 # variable-length strings: their values, not pointers
            return repr(a.tolist()).encode()
        return np.ascontiguousarray(a).tobytes()
    if isinstance(a, (bytes, str)):
        return a.encode() if isinstance(a, str) else a
    if isinstance(a, np.generic):
        return a.tobytes()
    return repr(a).encode()


def digests(path):
    """{dataset or object@attribute: sha256} of an HDF5 file."""
    import h5py
    out = {}
    if not os.path.exists(path):
        return out

    def attrs(name, obj):
        for k in obj.attrs:
            if k not in SKIP:
                v = obj.attrs[k]
                out[f"{name}@{k}"] = hashlib.sha256(f"{type(v).__name__}".encode() + _bytes(v)).hexdigest()

    with h5py.File(path, "r") as f:
        attrs("", f)

        def visit(name, obj):
            attrs(name, obj)
            if isinstance(obj, h5py.Dataset) and name.rsplit("/", 1)[-1] not in WALL:
                h = hashlib.sha256(f"{obj.dtype.str}{obj.shape}".encode())
                h.update(_bytes(obj[()]))
                out[name] = h.hexdigest()
        f.visititems(visit)
    return dict(sorted(out.items()))


def how_much(path_a, path_b, names):
    """{name: what differs} for datasets (by how much, when numeric) and
    attributes in both files."""
    import h5py
    import numpy as np
    out = {}
    with h5py.File(path_a, "r") as fa, h5py.File(path_b, "r") as fb:
        for n in names:
            if "@" in n:
                obj, key = n.split("@", 1)
                obj = obj or "/"
                va = fa[obj].attrs.get(key) if obj in fa else None
                vb = fb[obj].attrs.get(key) if obj in fb else None
                out[n] = f"attribute {va!r} -> {vb!r}"[:300]
                continue
            if n not in fa or n not in fb:
                out[n] = "only in one of the files"
                continue
            a, b = np.asarray(fa[n][()]), np.asarray(fb[n][()])
            if a.shape != b.shape:
                out[n] = f"shape {a.shape} -> {b.shape}"
            elif a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
                a, b = a.astype(float), b.astype(float)
                nan = np.isnan(a) | np.isnan(b)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    d = float(np.max(np.abs(a - b)[~nan])) if (~nan).any() else 0.0
                    s = float(np.max(np.abs(a)[~nan])) if (~nan).any() else 0.0
                moved = int(np.count_nonzero(np.isnan(a) != np.isnan(b)))
                out[n] = (f"max |diff| {d:.3e} (max |value| {s:.3e})" + (f", NaN at {moved} more or fewer places"
                                                                         if moved else "")
                          + (" -- bits only (a signed zero, a NaN's payload)" if d == 0.0 and not moved else ""))
            else:
                out[n] = "differs (not numeric)"
    return out


def _tolerance(what):
    for row in TOLERANCES:
        if any(p in what for p in row[1]):
            return row
    return TOLERANCES[-1]


def final_state(path, ckpt):
    """{name: (array, what)}: the last frame of a result file (its member and
    rebar frames too), each monitor's last value, and the checkpoint's q --
    what `--tolerance` judges. `what` names the quantity for TOLERANCES: a
    dataset's last name, a monitor's `kind` (its name is the user's)."""
    import h5py
    import numpy as np
    out = {}
    if path and os.path.exists(path):
        with h5py.File(path, "r") as f:
            for group in ("steps", "member_steps", "rebar_steps"):
                if group in f and len(f[group]):
                    last = sorted(f[group])[-1]             # %08d: in order
                    for k, d in f[group][last].items():
                        if isinstance(d, h5py.Dataset) and d.dtype.kind in "fiub":
                            out[f"{group}/<last>/{k}"] = (np.asarray(d[()]), k)
            if "monitors" in f:
                for k, d in f["monitors"].items():
                    if (isinstance(d, h5py.Dataset) and d.dtype.kind in "fiub" and d.size
                            and k not in ("residual", "kinetic_energy", "time", "step")):
                        kind = d.attrs.get("kind", "")
                        kind = kind.decode() if isinstance(kind, bytes) else str(kind)
                        if kind in ("fiber", "fibre", "section"):    # a fibre monitor: its field, else its name
                            field = d.attrs.get("field", "")
                            kind = field.decode() if isinstance(field, bytes) else str(field or k)
                        out[f"monitors/{k}"] = (np.asarray(d[()]).reshape(len(d), -1)[-1], kind or k)
    if ckpt and os.path.exists(ckpt):
        with h5py.File(ckpt, "r") as f:
            if "checkpoint/q" in f:
                out["checkpoint/q"] = (np.asarray(f["checkpoint/q"][()], float), "checkpoint/q")
    return out


#: a relaxed answer is held to this many times the tolerance it relaxed to
#: (its relative residual), carried through its history
RELAXED = 10.0


def relaxed_to(stem):
    """The largest relative residual a case's relaxation stops at (its
    solver's and its steps' `tolerance`), or 0.0 for a case that does not
    relax (a dynamic run, Newton alone)."""
    import yaml
    from vfife.io import schema
    try:
        d = yaml.safe_load((ROOT / "cases" / f"{stem}.yaml").read_text(encoding="utf-8")) or {}
        sol = d.get("solver") or {}
        steps = [s for s in d.get("steps") or [] if isinstance(s, dict)]
        modes = ({sol.get("mode", schema.SolverSpec.model_fields["mode"].default)}
                 | {s.get("procedure") for s in steps})
        if not modes & {"relaxation", "static"}:
            return 0.0
        tol = [sol.get("tolerance", schema.SolverSpec.model_fields["tolerance"].default)]
        tol += [s.get("tolerance") for s in steps]
        return max(float(t) for t in tol if t is not None)
    except Exception:                       # an unreadable case is judged without the floor
        return 0.0


def within_tolerance(a_files, b_files, relaxed=0.0):
    """(ok, lines): the final state of run `a` against the baseline's `b`
    (each a (result, checkpoint) pair), quantity by quantity. `relaxed`: the
    tolerance the case relaxes to (`relaxed_to`); no row is then tighter
    than RELAXED times it. A baseline that kept no copy of its result file
    (larger than COPY_LIMIT) is judged on q alone, and says so."""
    b_result = b_files[0] if b_files[0] and os.path.exists(b_files[0]) else None
    a = final_state(a_files[0] if b_result else None, a_files[1])
    b = final_state(b_result, b_files[1])
    lines = [] if b_result else ["- (the baseline kept no copy of the result file: q alone is judged)"]
    return _judge(a, b, relaxed, lines)


def monitor_history(path, until, frames=None):
    """{name: (array, what)} like `final_state`, but each monitor's values at
    every frame up to the time `until` (the frames of `monitors/time`), and
    nothing else: what a collapse case is judged on. `frames`: take this many
    frames instead (the baseline's count, so that a run whose frame at
    `until` reads 1e-12 later is cut at the same frame and its time line is
    judged by value)."""
    import h5py
    import numpy as np
    out = {}
    if not path or not os.path.exists(path):
        return out
    with h5py.File(path, "r") as f:
        m = f.get("monitors")
        if m is None or "time" not in m:
            return out
        t = np.asarray(m["time"][()], float)
        n = int(np.searchsorted(t, until, side="right")) if frames is None else min(int(frames), len(t))
        for k, d in m.items():
            if (isinstance(d, h5py.Dataset) and d.dtype.kind in "fiub" and len(d) == len(t)
                    and k not in ("residual", "kinetic_energy", "time", "step")):
                kind = d.attrs.get("kind", "")
                kind = kind.decode() if isinstance(kind, bytes) else str(kind)
                if kind in ("fiber", "fibre", "section"):
                    field = d.attrs.get("field", "")
                    kind = field.decode() if isinstance(field, bytes) else str(field or k)
                out[f"monitors/{k}[t<={until:g}]"] = (np.asarray(d[()])[:n], kind or k)
        out["monitors/time[frames]"] = (t[:n], "time")
    return out


def within_tolerance_until(a_result, b_monitors, until, relaxed=0.0):
    """(ok, lines): a collapse case's monitors up to `until` against the
    baseline's (its kept `<case>.monitors.h5`), quantity by quantity."""
    b = monitor_history(b_monitors, until)
    if "monitors/time[frames]" not in b:
        return False, ["- **OUT**: loaded to collapse, but the baseline's monitors have no time line"]
    n = len(b["monitors/time[frames]"][0])
    a = monitor_history(a_result, until, frames=n)
    lines = [f"- (loaded to collapse: every monitor at each of the {n} frames up to t = {until:g} s is judged, "
             "the state after the collapse is not)"]
    return _judge(a, b, relaxed, lines)


def _judge(a, b, relaxed, lines):
    import numpy as np
    size = {}
    for n, (y, what) in b.items():
        c = _tolerance(what)[0]
        fin = np.abs(y[np.isfinite(y)]) if y.dtype.kind == "f" else np.abs(y.astype(float))
        if fin.size:
            size[c] = max(size.get(c, 0.0), float(fin.max()))
    ok = True
    for n in sorted(set(a) | set(b)):
        if n not in a or n not in b:
            ok = False
            lines.append(f"- **OUT** `{n}`: only in one of the two")
            continue
        (x, _w), (y, what) = a[n], b[n]
        if x.shape != y.shape:
            ok = False
            lines.append(f"- **OUT** `{n}`: shape {y.shape} -> {x.shape}")
            continue
        if not y.size:
            continue
        if y.dtype.kind in "iub":                    # a state, held exactly
            changed = int(np.count_nonzero(x != y))
            if changed:
                ok = False
                lines.append(f"- **OUT** `{n}`: {changed} of {y.size} entries changed (held exactly)")
            continue
        x, y = x.astype(float), y.astype(float)
        cls, _p, rtol, floor = _tolerance(what)
        rtol = max(rtol, RELAXED * relaxed)
        floor = None if floor is None else max(floor, RELAXED * relaxed)
        nan = np.isnan(x) != np.isnan(y)
        err = np.abs(x - y)
        err[np.isnan(err)] = 0.0
        if floor is None:                              # classes: the share that changed
            share = float(np.count_nonzero(err) + np.count_nonzero(nan)) / y.size
            bad = share > rtol
            if share:
                lines.append(f"- {'**OUT**' if bad else 'within'} `{n}`: {share:.2e} of the entries changed "
                             f"(at most {rtol:g})")
        else:
            bound = rtol * np.nan_to_num(np.abs(y), posinf=0.0, neginf=0.0) + floor * size.get(cls, 0.0)
            bad = bool(nan.any() or np.any(err > bound))
            worst = float(err.max())
            if bad or worst > 0.0:
                lines.append(f"- {'**OUT**' if bad else 'within'} `{n}`: max |diff| {worst:.3e} "
                             f"({cls}: largest {size.get(cls, 0.0):.3e}; rtol {rtol:g}, floor {floor:g})")
        ok = ok and not bad
    return ok, lines


def run_case(stem, tmp, env, timeout):
    h5 = os.path.join(tmp, stem + ".h5")
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", stem, "--out", h5],
                           cwd=ROOT, env={**env, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        ok, text = p.returncode == 0, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        tail = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        ok, text = False, tail[-4000:] + f"\nTIMEOUT after {timeout} s"
    t = time.perf_counter() - t0
    ckpt = os.path.splitext(h5)[0] + ".ckpt.h5"
    return {"ok": ok, "seconds": round(t, 1), "tail": check_all._tail(text, 3 if ok else 20),
            "result": digests(h5), "checkpoint": digests(ckpt), "_h5": h5, "_ckpt": ckpt}


def keep_monitors(result, dest):
    """Copy the `monitors` group of a result file to `dest` (a small file,
    kept for every case even when the result file is not). False when the
    result has no monitors."""
    import h5py
    if not os.path.exists(result):
        return False
    with h5py.File(result, "r") as f:
        if "monitors" not in f:
            return False
        tmp = str(dest) + ".part"
        with h5py.File(tmp, "w") as g:
            f.copy("monitors", g)
    os.replace(tmp, dest)
    return True


def backfill(name, tree, stems, home, timeout):
    """Keep the monitors of a baseline written before `--write` kept them:
    solve each case with the checkout of the baseline's commit in `tree`
    (its own `tools/state_baseline.py --one`), check that its result and
    checkpoint hash to what the baseline recorded -- the same answer, to
    the bit -- and keep its monitors beside the baseline's files."""
    base = home / name
    old = json.loads((base / "index.json").read_text(encoding="utf-8"))
    tree = pathlib.Path(tree).resolve()
    tool = tree / "tools" / "state_baseline.py"
    if not tool.exists():
        print(f"no {tool}: --tree is a checkout of the baseline's commit ({old['commit']})")
        return 2
    env = check_all._single_threaded()
    bad = 0
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for stem in stems:
            o = old["cases"][stem]
            n = int(o.get("threads", 1))
            h5 = os.path.join(tmp, stem + ".h5")
            ckpt = os.path.splitext(h5)[0] + ".ckpt.h5"
            t0 = time.perf_counter()
            for attempt in (1, 2):           # a cold numba cache in a fresh checkout: once more, warm
                try:
                    p = subprocess.run([sys.executable, str(tool), "--one", stem, "--out", h5], cwd=tree,
                                       env={**(check_all._threaded(env, n) if n > 1 else env),
                                            "PYTHONPATH": str(tree), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
                                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                                       timeout=timeout)
                except subprocess.TimeoutExpired:
                    p = subprocess.CompletedProcess([], 1, "", f"TIMEOUT after {timeout} s")
                same = (p.returncode == 0 and digests(h5) == o["result"] and digests(ckpt) == o["checkpoint"])
                if same or p.returncode:
                    break
            dest = base / "files" / (stem + ".monitors.h5")
            if same and keep_monitors(h5, dest):
                print(f"  {stem}: the same as the baseline to the bit; kept {dest.name} "
                      f"({time.perf_counter() - t0:.0f} s)", flush=True)
            else:
                bad += 1
                why = "failed" if p.returncode else "not the baseline's answer (another machine or commit?)"
                print(f"  {stem}: {why}, nothing kept\n{check_all._tail((p.stdout or '') + (p.stderr or ''), 20)}",
                      flush=True)
            _drop(h5, ckpt)
    return 1 if bad else 0


def _drop(*paths):
    for p in paths:
        for q in (p, os.path.splitext(p)[0] + ".xdmf", os.path.splitext(p)[0] + "_last.xdmf"):
            try:
                os.remove(q)
            except OSError:
                pass


def _differences(r, o, kept):
    """The lines of the report for a case that differs from the baseline."""
    lines = []
    for part, key, ext in (("result", "_h5", ".h5"), ("checkpoint", "_ckpt", ".ckpt.h5")):
        diff = sorted(k for k in set(r[part]) | set(o[part]) if r[part].get(k) != o[part].get(k))
        if not diff:
            continue
        copy = kept / (r["stem"] + ext)
        what = how_much(str(copy), r[key], diff) if copy.exists() and os.path.exists(r[key]) else {}
        lines.append(f"{part}: {len(diff)} of {len(o[part])} datasets and attributes differ")
        lines += [f"- `{n}`: {what.get(n, '(no copy kept)')}" for n in diff[:40]]
        if len(diff) > 40:
            lines.append(f"- ... and {len(diff) - 40} more")
        lines.append("")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", metavar="NAME", help="solve and keep the baseline NAME")
    g.add_argument("--compare", metavar="NAME", help="solve and compare with the baseline NAME")
    g.add_argument("--backfill-monitors", metavar="NAME",
                   help="keep the monitors of baseline NAME, solving --cases with the checkout in --tree")
    g.add_argument("--one", metavar="CASE", help=argparse.SUPPRESS)
    ap.add_argument("--tree", help="--backfill-monitors: a checkout of the baseline's commit")
    ap.add_argument("--out", help=argparse.SUPPRESS)
    ap.add_argument("--force", action="store_true", help="--write over an existing baseline NAME")
    ap.add_argument("--cases", nargs="+", default=None, help="only these cases (a prefix such as case_24g works)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--case-threads", type=int, default=None)
    ap.add_argument("--case-timeout", type=float, default=3 * 3600.0)
    ap.add_argument("--include-slow-cases", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tolerance", action="store_true",
                    help="judge a case that differs on its final state, quantity by quantity (TOLERANCES)")
    ap.add_argument("--perf-gate", action="store_true",
                    help="fail the compare when a case is slower than the baseline by the gate (SLOWER)")
    args = ap.parse_args(argv)
    if args.one:
        return solve_one(args.one, args.out)

    home = ROOT / "results" / "state_baseline"
    if args.backfill_monitors:
        name = args.backfill_monitors
        if not (home / name / "index.json").exists() or not args.tree or not args.cases:
            print("--backfill-monitors NAME needs the baseline NAME, --tree DIR and --cases")
            return 2
        known = json.loads((home / name / "index.json").read_text(encoding="utf-8"))["cases"]
        unknown = [c for c in args.cases if not any(s == c or s.startswith(c + "_") for s in known)]
        if unknown:
            print("no such case in the baseline: " + ", ".join(unknown))
            return 2
        stems = sorted(s for s in known if any(s == c or s.startswith(c + "_") for c in args.cases))
        return backfill(name, args.tree, stems, home, args.case_timeout)
    if args.tree:
        print("--tree goes with --backfill-monitors")
        return 2
    name = args.write or args.compare
    base = home / name
    index = base / "index.json"
    old = None
    if args.compare:
        if not index.exists():
            print(f"no baseline {name}: {index} (python tools/state_baseline.py --write {name})")
            return 2
        old = json.loads(index.read_text(encoding="utf-8"))
    elif base.exists() and not args.force:
        print(f"the baseline {name} exists ({base}); --force writes over it, or give another name")
        return 2

    every = sorted(pathlib.Path(p).stem for p in glob.glob(str(ROOT / "cases" / "case_*.yaml")))
    if args.cases:
        unknown = [c for c in args.cases if not any(s == c or s.startswith(c + "_") for s in every)]
        if unknown:
            print("no such case: " + ", ".join(unknown))
            return 2
        stems = [s for s in every if any(s == c or s.startswith(c + "_") for c in args.cases)]
    else:
        stems = [s for s in every if args.include_slow_cases or s not in check_all.LONG_CASES]
    last = check_all._last_times()
    if args.quick and not args.cases:
        stems = [s for s in stems if last.get(s, 0.0) <= 600.0]
    notes = []
    if old is not None:
        new = [s for s in stems if s not in old["cases"]]
        gone = [s for s in old["cases"] if s not in every]
        if new:
            notes.append("not in the baseline, not solved: " + ", ".join(new))
        if gone:
            notes.append("in the baseline, no longer a case: " + ", ".join(gone))
        stems = [s for s in stems if s in old["cases"]]
        for n in notes:
            print(n)
    if not stems:
        print("no case to solve")
        return 2
    stems.sort(key=lambda s: -last.get(s, float("inf")))
    env = check_all._single_threaded()
    n_threads = check_all._case_threads(args)

    def threads_of(stem):
        if old is not None:
            return int(old["cases"][stem].get("threads", 1))
        if stem in check_all.THREADED_CASES:
            return min(n_threads, check_all.THREADED_CASES.get(stem) or n_threads)
        return 1

    from vfife import __version__
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "log", "--oneline", "-1"], capture_output=True,
                              text=True, encoding="utf-8", errors="replace").stdout.strip()
    except OSError:
        head = ""
    started = dt.datetime.now()
    print(f"[{check_all._now()}] {len(stems)} cases, {args.jobs} at a time ({'write' if args.write else 'compare'} "
          f"{name}; vfife {__version__}, {head})", flush=True)
    staging = home / (name + ".writing")
    if args.write:
        if staging.exists():
            shutil.rmtree(staging)
        (staging / "files").mkdir(parents=True)
    kept = base / "files"
    results, details = {}, {}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        def one(stem):
            n = threads_of(stem)
            r = run_case(stem, tmp, check_all._threaded(env, n) if n > 1 else env, args.case_timeout)
            r["threads"], r["stem"] = n, stem
            if args.write:
                if os.path.exists(r["_ckpt"]):
                    shutil.copy(r["_ckpt"], staging / "files" / (stem + ".ckpt.h5"))
                if os.path.exists(r["_h5"]) and os.path.getsize(r["_h5"]) <= COPY_LIMIT:
                    shutil.copy(r["_h5"], staging / "files" / (stem + ".h5"))
                keep_monitors(r["_h5"], staging / "files" / (stem + ".monitors.h5"))
            else:
                o = old["cases"][stem]
                r["same"] = (r["ok"] and o.get("ok", True)
                             and (r["result"], r["checkpoint"]) == (o["result"], o["checkpoint"]))
                if r["ok"] and not r["same"]:
                    details[stem] = _differences(r, o, kept)
                    if args.tolerance:
                        tol = relaxed_to(stem)
                        mon = kept / (stem + ".monitors.h5")
                        if not mon.exists():               # a kept result file has its monitors too
                            mon = kept / (stem + ".h5")
                        if stem in COLLAPSES and mon.exists():
                            ok, lines = within_tolerance_until(r["_h5"], str(mon), COLLAPSES[stem], tol)
                        else:
                            ok, lines = within_tolerance(
                                (r["_h5"], r["_ckpt"]),
                                (str(kept / (stem + ".h5")), str(kept / (stem + ".ckpt.h5"))), tol)
                            if stem in COLLAPSES:
                                lines.insert(0, f"- (loaded to collapse at t = {COLLAPSES[stem]:g} s, but the "
                                                "baseline kept no monitors: its final state is judged; "
                                                "--backfill-monitors keeps them)")
                        if tol:
                            lines.insert(0, f"- (relaxed to {tol:g}: no row tighter than {RELAXED * tol:g})")
                        r["within"] = ok
                        details[stem] = (["final state, per quantity "
                                          f"({'within tolerance' if ok else 'OUT OF TOLERANCE'}):"]
                                         + lines + [""] + details[stem])
            _drop(r["_h5"], r["_ckpt"])            # %TEMP% does not keep sixty result files
            return r

        with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for k, f in enumerate(cf.as_completed([pool.submit(one, s) for s in stems]), 1):
                r = f.result()
                results[r["stem"]] = r
                same = "" if old is None else ("  same" if r["same"] else "  within" if r.get("within")
                                               else "  DIFFERS" if r["ok"] else "")
                th = f" ({r['threads']} threads)" if r["threads"] > 1 else ""
                print(f"  [{check_all._now()}] {k:2d}/{len(stems)}  {'ok  ' if r['ok'] else 'FAIL'}  {r['stem']}"
                      f"{th}  {r['seconds']:7.1f} s{same}", flush=True)

    minutes = (dt.datetime.now() - started).total_seconds() / 60
    if args.write:
        failed = sorted(s for s, r in results.items() if not r["ok"])
        keep = {s: {k: v for k, v in r.items() if not k.startswith("_") and k != "stem"}
                for s, r in sorted(results.items()) if r["ok"]}
        (staging / "index.json").write_text(json.dumps(
            {"vfife": __version__, "commit": head, "written": f"{started:%Y-%m-%d %H:%M}",
             "python": sys.version.split()[0], "cores": os.cpu_count(), "cases": keep}, indent=1), encoding="utf-8")
        if base.exists():
            shutil.rmtree(base)
        staging.rename(base)
        print(f"written: {index} ({len(keep)} cases, {minutes:.1f} min)")
        for s in failed:
            print(f"  FAILED, not in the baseline: {s}\n{results[s]['tail']}")
        return 1 if failed else 0

    lines = [f"# state_baseline compare {started:%Y-%m-%d %H:%M}", "",
             f"baseline {name}: vfife {old['vfife']}, {old['commit']}, written {old['written']}",
             f"now: vfife {__version__}, {head}; {minutes:.1f} min", ""] + [f"- {n}" for n in notes]
    bad, within, rows, more = 0, 0, [], []
    slow = []
    for stem in sorted(results):
        r = results[stem]
        was = float(old["cases"][stem].get("seconds", 0.0))
        if r["ok"] and was and r["seconds"] > was * (1 + SLOWER) and r["seconds"] - was > SLOWER_S:
            slow.append(f"| {stem} | {was} | {r['seconds']} | {r['seconds'] / was - 1:+.0%} |")
        if r["same"]:
            rows.append(f"| same | {stem} | {r['seconds']} |")
            continue
        if r.get("within"):
            within += 1
            rows.append(f"| within | {stem} | {r['seconds']} |")
        else:
            bad += 1
            rows.append(f"| **{'DIFFERS' if r['ok'] else 'FAIL'}** | {stem} | {r['seconds']} |")
        more += [f"### {stem}", ""] + (details.get(stem, []) if r["ok"] else ["```text", r["tail"], "```", ""])
    head_line = ("ALL THE SAME" if not bad and not within else
                 f"{within} WITHIN TOLERANCE, THE REST THE SAME" if not bad else f"{bad} DIFFER OR FAIL"
                 + (f", {within} within tolerance" if within else ""))
    lines += ["", f"**{head_line}** ({len(results)} cases)", "",
              "| | case | s |", "|---|---|---|"] + rows + [""]
    lines += ([f"**{len(slow)} SLOWER than the baseline** (by more than {SLOWER:.0%} and {SLOWER_S:g} s"
               + ("; --perf-gate fails the compare" if args.perf_gate else "") + ")", "",
               "| case | baseline s | now s | |", "|---|---|---|---|"] + slow + [""]) if slow else []
    lines += more
    report = "\n".join(lines)
    path = home / f"{name}_compare_{started:%Y%m%d_%H%M%S}.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"report: {path}")
    return 1 if bad or (args.perf_gate and slow) else 0


if __name__ == "__main__":
    sys.exit(main())
