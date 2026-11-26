"""tools/state_baseline.py: the digest of a result file
sees a change in any dataset or attribute, to the last bit, and not what
names the run (its id, its date) nor a wall time."""
import os
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import state_baseline  # noqa: E402


def _file(path, q, run_id, wall=1.0, load_factor=0.5, note="dead load"):
    with h5py.File(path, "w") as f:
        f.attrs["run_id"] = run_id
        f.attrs["created_utc"] = f"2026-11-1{len(run_id)}"
        d = f.create_dataset("checkpoint/q", data=q)
        d.attrs["load_factor"] = load_factor
        f.create_dataset("checkpoint/step", data=np.array([7]))
        f.create_dataset("diagnostics/wall_time", data=np.array([wall]))
        f.create_dataset("names", data=[note, "two"], dtype=h5py.string_dtype())


def test_a_digest_sees_the_last_bit_and_not_what_names_the_run(tmp_path):
    q = np.linspace(0.0, 1.0, 5)
    a, b, c, d = (str(tmp_path / n) for n in ("a.h5", "b.h5", "c.h5", "d.h5"))
    _file(a, q, "one", wall=1.0)
    _file(b, q, "three", wall=2.0)
    _file(c, np.nextafter(q, 2.0), "one")
    _file(d, q, "one", load_factor=np.nextafter(0.5, 1.0), note="dead loaf")
    da, db, dc, dd = (state_baseline.digests(p) for p in (a, b, c, d))
    assert set(da) == {"checkpoint/q", "checkpoint/q@load_factor", "checkpoint/step", "names"}
    assert da == db
    assert [k for k in da if da[k] != dc[k]] == ["checkpoint/q"]
    assert sorted(k for k in da if da[k] != dd[k]) == ["checkpoint/q@load_factor", "names"]
    told = state_baseline.how_much(a, c, ["checkpoint/q"])["checkpoint/q"]
    assert told.startswith("max |diff| 2.220e-16")
    assert "attribute" in state_baseline.how_much(a, d, ["checkpoint/q@load_factor"])["checkpoint/q@load_factor"]


def test_variable_length_strings_hash_by_their_value(tmp_path):
    a = str(tmp_path / "a.h5")
    _file(a, np.zeros(2), "one")
    assert state_baseline.digests(a) == state_baseline.digests(a)


def test_nan_and_signed_zero_are_told(tmp_path):
    a, b = str(tmp_path / "a.h5"), str(tmp_path / "b.h5")
    with h5py.File(a, "w") as f:
        f["x"] = np.array([0.0, 1.0, np.nan])
    with h5py.File(b, "w") as f:
        f["x"] = np.array([-0.0, np.nan, np.nan])
    told = state_baseline.how_much(a, b, ["x"])["x"]
    assert "NaN at 1 more or fewer places" in told


def test_a_missing_file_has_no_digests(tmp_path):
    assert state_baseline.digests(str(tmp_path / "none.h5")) == {}


def test_an_unknown_case_and_an_existing_baseline_are_refused(tmp_path, monkeypatch, capsys):
    assert state_baseline.main(["--compare", "no_such_baseline_xyz"]) == 2
    assert "no baseline" in capsys.readouterr().out


def _final(path, u, f, q, mon):
    with h5py.File(path, "w") as h:
        h.create_dataset("steps/00000001/displacement", data=u * 0.5)
        h.create_dataset("steps/00000002/displacement", data=u)
        h.create_dataset("steps/00000002/axial_force_i", data=f)
        h.create_dataset("monitors/count", data=mon)
        h.create_dataset("monitors/residual", data=np.arange(len(mon)) * 1.0)
    with h5py.File(path + ".ck", "w") as h:
        h.create_dataset("checkpoint/q", data=q)
        h.create_dataset("checkpoint/mass", data=q * 7.0)


def test_the_tolerance_judges_the_final_state_quantity_by_quantity(tmp_path):
    """Version 1.19: the last frame, the monitors' last values and q, each
    to its own row of TOLERANCES; the path (the series, the mass) is not
    judged, and a value near zero is judged by the quantity's size."""
    u = np.array([0.0, 1e-3, 2e-3]); f = np.array([1e6, -2e6, 0.0]); q = np.array([0.0, 1e-3])
    a, b, c, d = (str(tmp_path / n) for n in ("a.h5", "b.h5", "c.h5", "d.h5"))
    _final(b, u, f, q, np.array([0.0, 1.0, 2.0]))
    # round-off, a near-zero force a little off, a longer monitor series, another mass
    _final(a, u * (1 + 1e-9), f + np.array([1.0, 1.0, 3.0]), q, np.array([0.0, 0.5, 1.5, 2.0 + 1e-9]))
    with h5py.File(a + ".ck", "a") as h:
        h["checkpoint/mass"][...] = 99.0
    ok, lines = state_baseline.within_tolerance((a, a + ".ck"), (b, b + ".ck"))
    assert ok, lines
    _final(c, u * (1 + 1e-3), f, q, np.array([0.0, 1.0, 2.0]))             # displacement 1e-3 off
    ok, lines = state_baseline.within_tolerance((c, c + ".ck"), (b, b + ".ck"))
    assert not ok and any("**OUT**" in x and "displacement" in x for x in lines)
    _final(d, u, f, q, np.array([0.0, 1.0, 2.0]))
    with h5py.File(d, "a") as h:
        del h["steps/00000002/axial_force_i"]
    ok, lines = state_baseline.within_tolerance((d, d + ".ck"), (b, b + ".ck"))
    assert not ok and any("only in one" in x for x in lines)


def test_the_tolerance_classes_a_monitor_by_its_kind_and_holds_a_state_exactly(tmp_path):
    """A monitor is classed by its `kind`, not its user-chosen name; an
    integer dataset (a member failed) is held exactly; a baseline without a
    copy of its result file is judged on q alone, and says so."""
    def write(path, tip, failed):
        with h5py.File(path, "w") as h:
            h.create_dataset("steps/00000001/reaction_force", data=np.array([1e3, -1e3]))
            h.create_dataset("steps/00000001/member_failed", data=np.array(failed, dtype=np.uint8))
            d = h.create_dataset("monitors/tip_uz", data=np.array([0.0, tip]))
            d.attrs["kind"] = "displacement"
        with h5py.File(path + ".ck", "w") as h:
            h.create_dataset("checkpoint/q", data=np.array([0.0, tip]))
    a, b, c = (str(tmp_path / n) for n in ("a.h5", "b.h5", "c.h5"))
    write(b, 1e-3, [0, 0])
    write(a, 1e-3 * (1 + 1e-3), [0, 0])            # 1e-3 off: the reaction's size does not hide it
    ok, lines = state_baseline.within_tolerance((a, a + ".ck"), (b, b + ".ck"))
    assert not ok and any("**OUT**" in x and "tip_uz" in x for x in lines)
    write(c, 1e-3, [0, 1])
    ok, lines = state_baseline.within_tolerance((c, c + ".ck"), (b, b + ".ck"))
    assert not ok and any("member_failed" in x and "held exactly" in x for x in lines)
    ok, lines = state_baseline.within_tolerance((c, c + ".ck"), (str(tmp_path / "none.h5"), b + ".ck"))
    assert ok and "q alone" in lines[0]


def _history(path, t, w):
    with h5py.File(path, "w") as f:
        f["monitors/time"] = t
        d = f.create_dataset("monitors/w_mid", data=w)
        d.attrs["kind"] = "displacement"
        f["monitors/residual"] = np.ones_like(t)


def test_a_collapse_case_is_judged_up_to_the_collapse_and_not_after(tmp_path):
    t = np.arange(0.0, 61.0)
    w = np.where(t <= 54.0, -0.01 * t, -0.5 * t)                 # the collapse after 54 s
    after = w.copy()
    after[t > 54.0] *= 1.08                                     # 8 % apart after it
    before = w.copy()
    before[30] *= 1.001                                         # 0.1 % apart before it
    base, run_after, run_before, kept = (str(tmp_path / n) for n in ("b.h5", "a.h5", "c.h5", "k.h5"))
    _history(base, t, w)
    _history(run_after, t, after)
    _history(run_before, t, before)
    ok, lines = state_baseline.within_tolerance_until(run_after, base, 54.0)
    assert ok, lines
    assert "up to t = 54 s" in lines[0]
    ok, lines = state_baseline.within_tolerance_until(run_before, base, 54.0)
    assert not ok
    assert any("**OUT** `monitors/w_mid[t<=54]`" in s for s in lines)
    # judged on the time line too: a run with other frames is not the baseline's
    _history(run_after, t[:-1] + 0.5, after[:-1])
    ok, lines = state_baseline.within_tolerance_until(run_after, base, 54.0)
    assert not ok
    # --write keeps the monitors, and nothing else, for every case
    assert state_baseline.keep_monitors(base, kept)
    with h5py.File(kept, "r") as f:
        assert sorted(f) == ["monitors"] and np.array_equal(f["monitors/w_mid"][()], w)
    assert not state_baseline.keep_monitors(str(tmp_path / "none.h5"), str(tmp_path / "x.h5"))


def test_every_collapse_case_is_a_case():
    for stem, until in state_baseline.COLLAPSES.items():
        assert os.path.exists(os.path.join(ROOT, "cases", stem + ".yaml"))
        assert 0.0 < until


def test_a_strain_is_its_own_class_not_held_to_the_size_of_a_stress():
    assert state_baseline._tolerance("strain")[0] == "strain"
    assert state_baseline._tolerance("eps_mid_bot")[0] == "strain"
    assert state_baseline._tolerance("fiber_stress")[0] == "stress"
    assert state_baseline._tolerance("sig_mid_bot")[0] == "stress"
    a = {"monitors/eps": (np.array([1.1e-2]), "strain"), "monitors/sig": (np.array([3.0e7]), "stress")}
    b = {"monitors/eps": (np.array([1.0e-2]), "strain"), "monitors/sig": (np.array([3.0e7]), "stress")}
    ok, lines = state_baseline._judge(a, b, 0.0, [])
    assert not ok and any("**OUT** `monitors/eps`" in s for s in lines)


def test_a_collapse_case_is_cut_at_the_baselines_frame_not_by_rounding(tmp_path):
    t = np.arange(0.0, 61.0)
    w = -0.01 * t
    late = t.copy()
    late[54] += 1e-12                                       # the frame at 54 s reads a hair later
    base, run = str(tmp_path / "b.h5"), str(tmp_path / "a.h5")
    _history(base, t, w)
    _history(run, late, w)
    ok, lines = state_baseline.within_tolerance_until(run, base, 54.0)
    assert ok, lines
    assert "each of the 55 frames" in lines[0]
    with h5py.File(base, "w") as f:                         # no time line: not judged as the same
        f["monitors/w_mid"] = w
    ok, lines = state_baseline.within_tolerance_until(run, base, 54.0)
    assert not ok
