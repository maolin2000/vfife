"""Everything the nightly CI layer did, on your own machine.

The GitHub layer that solved every shipped case every night was sixty jobs
of up to two hours each; on a private repository that is the month's
Actions minutes in days. The nightly schedule is off now -- CI runs the fast
layer on push and the verification only when started by hand -- and this
script is the full check instead, run where the cores are:

    python tools/check_all.py                 # everything, cores/2 cases at once
    python tools/check_all.py --jobs 8        # eight cases at once
    python tools/check_all.py --only fast     # the default pytest layer
    python tools/check_all.py --only cases --cases case_24g case_36
    python tools/check_all.py --include-slow-cases   # case_15 too (~1-2 h)
    python tools/check_all.py --quick         # skip the cases that took > 10 min last time

The two pytest layers run on `--jobs` processes when pytest-xdist is
installed (`pip install -e .[dev]`), and the cases are started longest first
by the times of the last report, so the run is not left waiting on case_24d
started last (v1.0: 120 min on 16 cores, 70 of them the default
layer on one core).

Four parts, each its own section of the report:

  fast      `pytest` (the default layer: everything not marked slow)
  slow      `pytest -m slow`, except the case contracts
  cases     every shipped case solved and checked against its contract, one
            process per case, `--jobs` at a time, each single-threaded but
            the longest (THREADED_CASES, `--case-threads`)
  package   the wheel built, installed into a fresh virtual environment
            outside the repository, and a case run from it (needs `build`)

The report is written to results/check_all/<time>.md (and printed); the
exit code is non-zero when anything failed.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import glob
import os
import pathlib
import shutil
import subprocess
import sys
import re
import tempfile
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # this checkout's vfife, not an installed one
#: solved by hand only: an hour or two each (see .github/workflows/ci.yml)
LONG_CASES = {"case_15_chalmers_prc"}
#: cases run with threads of their own (1.7): the one that sets the length of
#: the cases part runs alone for most of it, on one core of sixteen. Its
#: threaded kernels give the serial answer to the bit.
#: How many: the cores the other jobs leave, at most 8 (on a hybrid CPU 16
#: threads were slower than 8)
#: threads=2 for the four fibre pushovers (1.9): their relaxation is the
#: fused fibre kernel, which threads to the serial answer to the bit; two
#: each, as they start together with case_24d among the longest. NOT the
#: three U of Alberta beams: given two threads each the
#: longest cases asked for 22 threads on 16 cores, and case_24d went from
#: 272 to 481 s, case_31 from 209 to 382 s -- 25.5 min in all against 22.9.
#: With their bars compiled they take 150-190 s on one thread's worth anyway
THREADED_CASES = {"case_24d_hzb_girder_piers": None, "case_25_chen_hollow_slab": 2,
                  "case_25b_chen_hollow_slab": 2, "case_30_kuang_control_beam": 2,
                  "case_31_li_control_beam": 2}
#: the whole output of each part that failed, written beside the report: the
#: report keeps pytest's one-line list, which said WHICH test failed on the
#: Windows machine but not why
FAILED_OUTPUT: dict = {}


def _run(cmd, cwd=ROOT, env=None, timeout=None):
    t0 = time.perf_counter()
    try:
        # UTF-8 both ways: on a Chinese Windows the pipes default to GBK, and
        # a child printing a character GBK lacks would die of it
        env = {**(env or os.environ), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        ok, out = p.returncode == 0, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        ok, out = False, f"TIMEOUT after {timeout} s\n" + str(e.stdout or "")[-4000:]
    return ok, time.perf_counter() - t0, out


def _now():
    return dt.datetime.now().strftime("%H:%M:%S")


def _run_pytest(cmd, label, env=None):
    """`_run` for a pytest layer, telling the console how far it has got:
    pytest's own "[ 45%]" every 5 percent, and a line a minute while it
    compiles or collects (the first minutes after a commit, when numba
    builds its cache again, print nothing else)."""
    t0 = time.perf_counter()
    env = {**(env or os.environ), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    out, shown, last = [], [-5], [time.perf_counter()]
    done = threading.Event()

    def tick():
        while not done.wait(60.0):
            if time.perf_counter() - last[0] >= 60.0:
                print(f"  [{_now()}] {label}: {shown[0] if shown[0] >= 0 else 0}% "
                      f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
                last[0] = time.perf_counter()

    th = threading.Thread(target=tick, daemon=True)
    th.start()
    try:
        for line in p.stdout:
            out.append(line)
            m = re.match(r"^[.sFExX]*\s*\[\s*(\d+)%\]\s*$", line)
            if m and int(m.group(1)) >= shown[0] + 5:
                shown[0] = int(m.group(1))
                last[0] = time.perf_counter()
                print(f"  [{_now()}] {label}: {shown[0]}% ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
        p.wait()
    finally:
        done.set()
        if p.poll() is None:              # interrupted: the layer and its workers go too
            p.kill()
            p.wait()
        p.stdout.close()
    text = "".join(out)
    ok = p.returncode == 0
    print(f"  [{_now()}] {label}: {'PASS' if ok else 'FAIL'} -- {_tail(text, 1)}", flush=True)
    return ok, time.perf_counter() - t0, text


class _Tee:
    """What check_all prints, to the console and to the log at once."""

    def __init__(self, *streams):
        self.streams = streams
        self.lock = threading.Lock()      # the heartbeat thread prints too

    def write(self, text):
        with self.lock:
            for st in self.streams:
                st.write(text)
                st.flush()
        return len(text)

    def flush(self):
        for st in self.streams:
            st.flush()


def _tail(text, n=25):
    lines = [l for l in text.splitlines() if l.strip()]
    return "\n".join(lines[-n:])


def _summary(text, ok, n=60):
    """pytest's own list of what failed (-rfE), then its last line; the
    last lines alone showed 40 of 176 problems once, and none of the six
    failures."""
    if ok:
        return _tail(text, 3)
    lines = [l for l in text.splitlines() if l.startswith(("FAILED ", "ERROR "))]
    last = _tail(text, 1)
    head = lines[:n] + ([f"... and {len(lines) - n} more"] if len(lines) > n else [])
    return "\n".join(head + [last]) if lines else _tail(text, 40)


def _xdist(args):
    """`-n <jobs>` for pytest when pytest-xdist is there."""
    try:
        import xdist  # noqa: F401
    except ImportError:
        print("  (pytest-xdist not installed: the pytest layers run on one core; "
              "`pip install pytest-xdist`)", flush=True)
        return []
    return ["-n", str(args.jobs)]


def _last_times():
    """{label: seconds} from the reports, the newest time of each, to start
    the longest first (a partial run updates only what it ran)."""
    out = ROOT / "results" / "check_all"
    reports = sorted(out.glob("*.md")) if out.is_dir() else []
    times = {}
    for path in reports:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            cells = [c.strip() for c in line.split("|")]
            if len(cells) >= 5 and cells[1] in ("PASS", "**FAIL**"):
                try:
                    # a threaded case is labelled "stem (n threads)"
                    times[cells[2].split(" (")[0]] = float(cells[3])
                except ValueError:
                    pass
    return times


def _single_threaded():
    env = dict(os.environ)
    for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = "1"
    env.pop("VFIFE_CASE_THREADS", None)
    return env


def _case_threads(args):
    """The threads a THREADED_CASES case gets: the cores the other jobs leave
    free, at most 8; 1 (serial) on a machine without them."""
    if args.case_threads is not None:
        return max(1, int(args.case_threads))
    return max(1, min(8, (os.cpu_count() or 1) - (args.jobs - 1)))


def _threaded(env, n):
    """`env` with numba allowed `n` threads and the case told to use them
    (the BLAS libraries stay at one: the kernels are numba's)."""
    env = dict(env)
    if n > 1:
        env["NUMBA_NUM_THREADS"] = str(n)
        env["VFIFE_CASE_THREADS"] = str(n)
    return env


def part_fast(args):
    ok, t, out = _run_pytest([sys.executable, "-m", "pytest", "-q", "-rfE", "--tb=short", "-p", "no:cacheprovider"]
                             + _xdist(args), "default layer")
    if not ok:
        FAILED_OUTPUT["fast"] = out
    return [("pytest (default layer)", ok, t, _summary(out, ok))]


def part_slow(args):
    ok, t, out = _run_pytest([sys.executable, "-m", "pytest", "-q", "-rfE", "--tb=short", "-m", "slow",
                              "-p", "no:cacheprovider", "--ignore=tests/test_cases.py"] + _xdist(args), "slow layer")
    if not ok:
        FAILED_OUTPUT["slow"] = out
    return [("pytest -m slow (not the case contracts)", ok, t, _summary(out, ok))]


def part_cases(args):
    stems = sorted(pathlib.Path(p).stem for p in glob.glob(str(ROOT / "cases" / "case_*.yaml")))
    if args.cases:
        stems = [s for s in stems if any(s == c or s.startswith(c + "_") for c in args.cases)]
    elif not args.include_slow_cases:
        stems = [s for s in stems if s not in LONG_CASES]
    last = _last_times()
    if args.quick and not args.cases:
        long = sorted(s for s in stems if last.get(s, 0.0) > 600.0)
        if long:
            print(f"  --quick: skipping {', '.join(long)} (over 10 min last time)", flush=True)
        stems = [s for s in stems if last.get(s, 0.0) <= 600.0]
    # longest first, so the slowest case does not start last (unknown: first)
    stems.sort(key=lambda s: -last.get(s, float("inf")))
    env = _single_threaded()
    n_threads = _case_threads(args)
    threaded = sorted(s for s in stems if s in THREADED_CASES)
    if threaded and n_threads > 1:
        print("  " + ", ".join(f"{t}: {min(n_threads, THREADED_CASES[t] or n_threads)} threads"
                               for t in threaded), flush=True)

    def one(stem):
        node = f"tests/test_cases.py::test_a_case_solved_satisfies_its_contract[{stem}]"
        n = min(n_threads, THREADED_CASES.get(stem) or n_threads) if stem in THREADED_CASES else 1
        ok, t, out = _run([sys.executable, "-m", "pytest", "-q", "-m", "slow", "-p", "no:cacheprovider",
                           node], env=_threaded(env, n) if n > 1 else env,
                          timeout=args.case_timeout)
        label = stem if n == 1 else f"{stem} ({n} threads)"
        return (label, ok, t, _tail(out, 3 if ok else 30))

    rows = []
    print(f"  [{_now()}] {len(stems)} cases, {args.jobs} at a time, longest first", flush=True)
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(one, s): s for s in stems}
        for f in cf.as_completed(futures):
            r = f.result()
            rows.append(r)
            print(f"  [{_now()}] {len(rows):2d}/{len(stems)}  {'PASS' if r[1] else 'FAIL'}  {r[0]}  {r[2]:7.1f} s",
                  flush=True)
    return sorted(rows)


def part_package(args):
    try:
        import build  # noqa: F401
    except ImportError:
        return [("package", None, 0.0, "skipped: `pip install build` to check the wheel")]
    rows = []
    print(f"  [{_now()}] building the wheel and installing it into a fresh environment", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        ok, t, out = _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp / "dist"), str(ROOT)])
        rows.append(("wheel builds", ok, t, _tail(out, 3 if ok else 30)))
        if not ok:
            return rows
        wheel = next((tmp / "dist").glob("*.whl"))
        ok, t, out = _run([sys.executable, "-m", "venv", str(tmp / "venv")])
        py = tmp / "venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
        if ok:
            ok, t2, out = _run([str(py), "-m", "pip", "install", "-q", str(wheel)])
            t += t2
        rows.append(("installs into a fresh environment", ok, t, _tail(out, 3 if ok else 30)))
        if not ok:
            return rows
        run = tmp / "run"
        run.mkdir()
        for f in ("cantilever.yaml", "cantilever.msh"):
            shutil.copy(ROOT / "tests" / "data" / "result_format_2" / f, run / f)
        script = ("from vfife import api\n"
                  "o = api.run_case(api.load_case('cantilever.yaml'), output='out.h5')\n"
                  "assert o.status == 'COMPLETED_DYNAMIC', o.status\n"
                  "w = o.monitors['w_tip'][-1]\n"
                  "assert abs(w + 2.16961e-3) < 1e-7, w\n"
                  "print('w_tip', w)\n")
        ok, t, out = _run([str(py), "-c", script], cwd=run)
        rows.append(("the installed package runs a case (outside the repository)", ok, t, _tail(out, 3 if ok else 30)))
        ok, t, out = _run([str(py), "-m", "vfife", "list"], cwd=run)
        rows.append(("`vfife list` from the installed package", ok, t, _tail(out, 2 if ok else 20)))
    return rows


PARTS = {"fast": part_fast, "slow": part_slow, "cases": part_cases, "package": part_package}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(PARTS), default=list(PARTS))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="cases solved at once (default: half the cores)")
    ap.add_argument("--cases", nargs="+", default=None, help="only these cases (a prefix such as case_24g works)")
    ap.add_argument("--include-slow-cases", action="store_true", help="also " + ", ".join(sorted(LONG_CASES)))
    ap.add_argument("--quick", action="store_true",
                    help="skip the cases that took more than 10 min in the last report")
    ap.add_argument("--case-timeout", type=float, default=3 * 3600.0, help="seconds per case (default 3 h)")
    ap.add_argument("--case-threads", type=int, default=None,
                    help="threads for " + ", ".join(sorted(THREADED_CASES)) +
                         " (default: the cores the other jobs leave, at most 8, and at most 2 for the"
                         " fibre pushovers; 1 runs them serially)")
    ap.add_argument("--log", default=None,
                    help="also append everything this prints to this file (run_full_check.bat: the console "
                         "shows the progress, the file keeps it)")
    args = ap.parse_args(argv)
    if args.log:
        for st in (sys.__stdout__, sys.__stderr__):
            try:
                st.reconfigure(errors="replace")      # a GBK console must not stop the run
            except (AttributeError, ValueError):
                pass
        logf = open(args.log, "a", encoding="utf-8")
        sys.stdout = _Tee(sys.__stdout__, logf)
        sys.stderr = _Tee(sys.__stderr__, logf)

    started = dt.datetime.now()
    sections, failed = [], 0
    for name in args.only:
        print(f"[{_now()}] == {name}", flush=True)
        rows = PARTS[name](args)
        sections.append((name, rows))
        failed += sum(1 for r in rows if r[1] is False)

    from vfife import __version__
    lines = [f"# check_all {started:%Y-%m-%d %H:%M}", "",
             f"vfife {__version__}, Python {sys.version.split()[0]}, {os.cpu_count()} cores, "
             f"--jobs {args.jobs}; {(dt.datetime.now() - started).total_seconds() / 60:.1f} min in all.", "",
             f"**{'ALL PASSED' if failed == 0 else f'{failed} FAILED'}**", ""]
    for name, rows in sections:
        lines += [f"## {name}", "", "| | 项 | 用时 s |", "|---|---|---|"]
        for label, ok, t, _ in rows:
            mark = "PASS" if ok else ("skip" if ok is None else "**FAIL**")
            lines.append(f"| {mark} | {label} | {t:.1f} |")
        lines.append("")
        for label, ok, t, text in rows:
            if ok is not True and text:
                lines += [f"### {label}", "", "```text", text, "```", ""]
    report = "\n".join(lines)
    out = ROOT / "results" / "check_all"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{started:%Y%m%d_%H%M}.md"
    for part, text in FAILED_OUTPUT.items():
        log = out / f"{started:%Y%m%d_%H%M}_{part}.log"
        log.write_text(text, encoding="utf-8")
        report += f"\n\nthe whole output of `{part}`: {log.name}\n"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"report: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
