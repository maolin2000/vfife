"""Getting the answer back out of the result file.

The solver wrote every monitor into an HDF5 file and stopped there. Seeing a
load-displacement curve -- or the hysteresis loop that is the whole point of a
cyclic analysis -- meant opening h5py and writing the plot yourself. The chain
broke one step before the thing anyone actually wanted, and it broke for real:
someone had to be handed hand-written Python to look at an RC column result.

These tests are fast on purpose. None of them solves anything; the result files
are synthesised, because what is under test is the reading and drawing, not the
solver.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from vfife.core.diagnostics import UserError
from vfife.post import Series, plot, read_monitors, to_csv


def make_result(path, *, monitors=True, meta=True, n=64):
    """A result file with cyclic monitors, without running an analysis."""
    import h5py

    t = np.linspace(0.0, 4 * np.pi, n)
    d = 0.02 * np.sin(t)
    with h5py.File(path, "w") as f:
        if not monitors:
            f.create_group("steps")
            return path
        g = f.create_group("monitors")
        g.create_dataset("time", data=t)
        for name, vals, kind, region, dof in (
            ("tip_disp", d, "displacement", "head", "z"),
            ("base_shear", -3.0e5 * d, "reaction", "base", "z"),
        ):
            ds = g.create_dataset(name, data=vals)
            if meta:
                ds.attrs["kind"] = kind
                ds.attrs["regions"] = region
                ds.attrs["dof"] = dof
        g.create_dataset("residual", data=np.geomspace(1.0, 1e-7, n))
    return path


def cli(*args, cwd=None):
    return subprocess.run([sys.executable, "-m", "vfife", *args],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8",
                               "PYTHONPATH": os.path.dirname(os.path.dirname(
                                   os.path.abspath(__file__)))})


# --------------------------------------------------------------------------- #
#  reading
# --------------------------------------------------------------------------- #


def test_a_monitor_carries_what_it_is_not_just_its_numbers(tmp_path):
    """`tip_disp` means nothing to a reader; "displacement z on head" does.

    The writer used to store a bare column, so nothing downstream could label
    an axis without being told separately what it was looking at.
    """
    s = read_monitors(make_result(tmp_path / "r.h5"))
    assert s["tip_disp"].kind == "displacement"
    assert s["tip_disp"].regions == "head"
    assert s["tip_disp"].dof == "z"
    assert "displacement" in s["tip_disp"].label and "head" in s["tip_disp"].label
    assert s["base_shear"].units == "N"
    assert s["tip_disp"].units == "m"


def test_a_result_written_before_the_metadata_existed_still_reads(tmp_path):
    """Older files have no attrs. They must not become unreadable."""
    s = read_monitors(make_result(tmp_path / "old.h5", meta=False))
    assert set(s) >= {"tip_disp", "base_shear"}
    assert s["tip_disp"].label == "tip_disp"     # falls back to the bare name


def test_a_file_with_no_monitors_says_what_to_add(tmp_path):
    with pytest.raises(UserError) as exc:
        read_monitors(make_result(tmp_path / "bare.h5", monitors=False))
    assert exc.value.code == "NO_MONITORS"
    assert "output.monitors" in exc.value.hint


def test_a_missing_file_is_a_sentence(tmp_path):
    with pytest.raises(UserError) as exc:
        read_monitors(str(tmp_path / "nope.h5"))
    assert exc.value.code == "NO_SUCH_RESULT"


def test_a_mistyped_monitor_name_suggests_the_right_one(tmp_path):
    p = make_result(tmp_path / "r.h5")
    with pytest.raises(UserError) as exc:
        to_csv(p, str(tmp_path / "o.csv"), monitors=["tip_dsip"])
    assert exc.value.code == "UNKNOWN_MONITOR"
    assert "tip_disp" in exc.value.hint          # did-you-mean
    assert "base_shear" in exc.value.hint        # and the full list


# --------------------------------------------------------------------------- #
#  CSV
# --------------------------------------------------------------------------- #


def test_export_writes_every_monitor_with_time_first(tmp_path):
    out = to_csv(make_result(tmp_path / "r.h5"), str(tmp_path / "o.csv"))
    lines = open(out, encoding="utf-8").read().splitlines()
    assert lines[0].split(",")[0] == "time"
    # `residual` is a convergence series, sampled at progress checks: left
    # out by default and exported on its own
    assert set(lines[0].split(",")) == {"time", "tip_disp", "base_shear"}
    assert len(lines) == 65                      # header + 64 rows


def test_export_can_pick_columns_and_still_carries_time(tmp_path):
    """Two columns are what a spreadsheet needs for a curve; time is what makes
    the row order meaningful, so it is never dropped."""
    out = to_csv(make_result(tmp_path / "r.h5"), str(tmp_path / "o.csv"),
                 monitors=["tip_disp", "base_shear"])
    header = open(out, encoding="utf-8").readline().strip().split(",")
    assert header == ["time", "tip_disp", "base_shear"]


def test_exported_numbers_round_trip_exactly(tmp_path):
    """A CSV that loses the last digits is not a way to take the result
    elsewhere."""
    p = make_result(tmp_path / "r.h5")
    out = to_csv(p, str(tmp_path / "o.csv"))
    original = read_monitors(p)["tip_disp"].values
    got = np.loadtxt(out, delimiter=",", skiprows=1,
                     usecols=open(out, encoding="utf-8").readline().strip().split(",").index("tip_disp"), encoding="utf-8")
    assert np.array_equal(got, original)


# --------------------------------------------------------------------------- #
#  plots
# --------------------------------------------------------------------------- #


def test_plotting_all_monitors_writes_a_png(tmp_path):
    pytest.importorskip("matplotlib")
    out = plot(make_result(tmp_path / "r.h5"), str(tmp_path / "a.png"))
    assert os.path.getsize(out) > 5000
    assert open(out, "rb").read(8) == b"\x89PNG\r\n\x1a\n"


def test_one_monitor_against_another_is_the_hysteresis_loop(tmp_path):
    """The plot this project was missing. With a cyclic amplitude, reaction
    against displacement IS the hysteresis loop, and it is the reason someone
    had to write h5py by hand to look at an RC column."""
    pytest.importorskip("matplotlib")
    out = plot(make_result(tmp_path / "r.h5"), str(tmp_path / "h.png"),
               x="tip_disp", y=["base_shear"])
    assert os.path.getsize(out) > 5000


def test_x_without_y_says_what_is_missing_and_what_exists(tmp_path):
    pytest.importorskip("matplotlib")
    with pytest.raises(UserError) as exc:
        plot(make_result(tmp_path / "r.h5"), str(tmp_path / "h.png"), x="tip_disp")
    assert exc.value.code == "PLOT_NEEDS_Y"
    assert "base_shear" in exc.value.hint


def test_plotting_without_matplotlib_is_a_sentence_not_a_traceback(tmp_path,
                                                                   monkeypatch):
    """It is an optional dependency, so its absence is a normal state that has
    to read as advice -- including the escape hatch that needs nothing."""
    import builtins

    real = builtins.__import__

    def no_matplotlib(name, *a, **k):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("no matplotlib")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    with pytest.raises(UserError) as exc:
        plot(make_result(tmp_path / "r.h5"), str(tmp_path / "x.png"))
    assert exc.value.code == "NO_MATPLOTLIB"
    assert "pip install matplotlib" in exc.value.hint
    assert "vfife export" in exc.value.hint      # the way out that always works


# --------------------------------------------------------------------------- #
#  through the CLI
# --------------------------------------------------------------------------- #


def test_the_cli_plots_and_names_the_monitors_it_found(tmp_path):
    pytest.importorskip("matplotlib")
    p = make_result(tmp_path / "r.h5")
    r = cli("plot", str(p), "-o", str(tmp_path / "o.png"))
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "o.png").exists()
    # it tells you what is in there, so the next command can be specific
    assert "tip_disp" in r.stdout and "base_shear" in r.stdout
    assert "--x" in r.stdout                     # and points at the x-y form


def test_the_cli_defaults_the_output_name_to_the_result(tmp_path):
    pytest.importorskip("matplotlib")
    p = make_result(tmp_path / "run.h5")
    assert cli("plot", str(p)).returncode == 0
    assert (tmp_path / "run.png").exists()
    assert cli("export", str(p)).returncode == 0
    assert (tmp_path / "run.csv").exists()


def test_a_cli_mistake_is_a_sentence_with_the_available_names(tmp_path):
    r = cli("export", str(make_result(tmp_path / "r.h5")), "--monitor", "nope")
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "no monitor called" in r.stderr
    assert "base_shear" in r.stderr
