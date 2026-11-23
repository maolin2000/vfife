"""Getting the answer back out.

The solver produced an HDF5 file and stopped there. Every monitor -- the whole
point of asking for one -- lived inside it as a bare column of numbers, so
seeing a load-displacement curve or a hysteresis loop meant writing h5py by
hand. That is a break in the chain: the tool that ran the analysis could not
show you what it found.

This module is the other end. It reads the monitors back with the metadata the
writer now stores beside them, so a plot can label its own axes, and it offers
the two things anyone actually wants: a picture, and a CSV to take elsewhere.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core.diagnostics import UserError
from ..io.writer import open_result

#: Series the solver always records, as opposed to the ones the case asked for.
BUILTIN = ("time", "residual", "kinetic_energy")
#: ...of which these two are sampled at the PROGRESS checks (every
#: `progress_every` steps), not at the output frames `time` belongs to. They
#: were paired with `time` as if they were: the export
#: refused the file as "ragged", the plot squeezed them into part of the axis.
CONVERGENCE = ("residual", "kinetic_energy")


class Series:
    """One monitor: its values, and what it is."""

    __slots__ = ("name", "values", "kind", "regions", "dof", "unit")

    def __init__(self, name, values, kind="", regions="", dof="", unit=None):
        self.name = name
        self.values = np.asarray(values, dtype=float)
        self.kind = kind
        self.regions = regions
        self.dof = dof
        #: the unit as the result recorded it (the dataset's `units`
        #: attribute); None = not recorded
        self.unit = unit

    @property
    def label(self) -> str:
        """An axis label, from the metadata rather than the user's chosen name.

        `tip_w` means nothing to a reader; "displacement z on Tip" does.
        """
        if not self.kind:
            return self.name
        where = f" on {self.regions}" if self.regions else ""
        axis = f" {self.dof}" if self.dof else ""
        return f"{self.name}  ({self.kind}{axis}{where})"

    @property
    def units(self) -> str:
        # the case's own units, as the result recorded them:
        # this said [m] and [N] whatever the case was in
        return self.unit or ""


def read_monitors(path: str) -> Dict[str, Series]:
    """Every recorded time history in a result file, metadata included."""
    import h5py

    if not os.path.exists(path):
        raise UserError("NO_SUCH_RESULT", f"no such result file: {path}",
                        "run the case first, or check the path")
    with open_result(path, "r") as f:
        if "monitors" not in f and "analysis_status" not in f.attrs and "steps" in f \
                and int(f.attrs.get("result_schema_version", 0) or 0) >= 3:
            # a run killed before its end (SIGKILL, a power cut): the frames
            # were flushed as they came, the monitors and the diagnostics are
            # written at the end. It was told to add monitors
            raise UserError(
                "RESULT_NOT_CLOSED",
                f"{path} was not closed (the run was killed?): its {len(f['steps'])} frame(s) are "
                "readable, but the monitors and the diagnostics are written at the end of a run "
                "and are missing",
                "resume it with `vfife run <case> --restart <this result> -o <new.h5>` if a "
                "checkpoint was written beside it, or run the case again; `vfife view` shows the frames")
        if "monitors" not in f:
            raise UserError(
                "NO_MONITORS", f"{path} contains no monitors",
                "add an `output.monitors` block to the case and re-run; "
                "`reaction` on the supported region is the cheapest self-check")
        declared = _declared_units(f)
        out = {}
        for name in f["monitors"]:
            d = f["monitors"][name]
            a = dict(d.attrs)
            kind = str(a.get("kind", ""))
            unit = a.get("units")
            if unit is None:
                # not recorded (a result of format 2, or a case that declared
                # no units): by kind, in the units its case declared, else the
                # SI the manual recommends
                unit = {"displacement": declared.get("length", "" if declared else "m"),
                        "reaction": declared.get("force", "" if declared else "N")}.get(kind, "")
            out[name] = Series(name, d[...], kind=kind,
                               regions=str(a.get("regions", "")),
                               dof=str(a.get("dof", "")), unit=str(unit))
    return out


def _declared_units(f) -> Dict[str, str]:
    """The units a result's case declared: the root `unit_*` attributes
    (format 3), or the embedded case's `units` block for an older file."""
    out = {k[5:]: str(v) for k, v in f.attrs.items() if k.startswith("unit_")}
    if out:
        return out
    try:
        import yaml
        case = yaml.safe_load(f.attrs.get("case_yaml", "") or "") or {}
        return {str(k): str(v) for k, v in (case.get("units") or {}).items()}
    except Exception:
        return {}


def _pick(series: Dict[str, Series], name: str) -> Series:
    if name in series:
        return series[name]
    import difflib
    close = difflib.get_close_matches(name, series, n=1)
    raise UserError(
        "UNKNOWN_MONITOR", f"no monitor called {name!r} in this result",
        (f"did you mean {close[0]!r}? " if close else "")
        + "available: " + ", ".join(sorted(series)))


def to_csv(path: str, out: str, monitors: Optional[Sequence[str]] = None) -> str:
    """Monitors as one CSV, time first, one column per series."""
    series = read_monitors(path)
    names = list(monitors) if monitors else sorted(n for n in series if n not in CONVERGENCE)
    if monitors:
        for n in names:
            _pick(series, n)
        conv = [n for n in names if n in CONVERGENCE]
        if conv and len(conv) != len(names):
            raise UserError(
                "MIXED_SAMPLING",
                f"{', '.join(conv)} {'is' if len(conv) == 1 else 'are'} sampled at the progress "
                "checks and the other monitors at the output frames; they have no common rows",
                "export them separately: --monitor " + " ".join(conv))
        if conv:
            # their own axis: the check number
            os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
            n_rows = max(len(series[n].values) for n in conv)
            with open(out, "w", newline="", encoding="utf-8") as fh:
                fh.write(",".join(["check"] + conv) + "\n")
                for i in range(n_rows):
                    row = [str(i + 1)] + [repr(float(series[n].values[i])) if i < len(series[n].values)
                                          else "" for n in conv]
                    fh.write(",".join(row) + "\n")
            return out
    # time leads, always: it is what makes the row order mean something, and a
    # column order that depends on the alphabet is not an ordering anyone asked
    # for. Selecting columns never drops it.
    names = [n for n in names if n != "time"]
    if "time" in series:
        names = ["time"] + names

    columns = [series[n] for n in names]
    n_rows = min(len(c.values) for c in columns)
    if any(len(c.values) != n_rows for c in columns):
        # honest rather than silently truncating an inconsistent file
        raise UserError(
            "RAGGED_MONITORS",
            "the monitors in this file have different lengths: "
            + ", ".join(f"{c.name}={len(c.values)}" for c in columns),
            "this should not happen; keep the file and report it")

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write(",".join(c.name for c in columns) + "\n")
        for i in range(n_rows):
            fh.write(",".join(repr(float(c.values[i])) for c in columns) + "\n")
    return out


def _pyplot():
    """matplotlib, or a sentence saying how to get it."""
    try:
        import matplotlib
    except ImportError:
        raise UserError(
            "NO_MATPLOTLIB", "plotting needs matplotlib, which is not installed",
            "pip install matplotlib   (or `pip install vfife[plot]`). "
            "`vfife export` needs nothing extra and writes a CSV you can plot "
            "anywhere.") from None
    matplotlib.use("Agg")          # a CLI has no display
    import matplotlib.pyplot as plt
    return plt


def plot(path: str, out: str, *, x: Optional[str] = None,
         y: Optional[Sequence[str]] = None) -> str:
    """Write a PNG.

    Three shapes, because three questions get asked:

    * nothing named -> every monitor against time, stacked. "Did it do
      anything sensible?"
    * `y` named -> those monitors against time. "What did the tip do?"
    * `x` and `y` named -> one against the other. That is the load-displacement
      curve, and with a cyclic amplitude it is the hysteresis loop -- the plot
      this project was missing, and the reason someone had to write h5py by
      hand to look at an RC column.
    """
    plt = _pyplot()
    series = read_monitors(path)
    if x is not None:
        xs = _pick(series, x)
        bad = [n for n in [x] + list(y or []) if n in CONVERGENCE]
        if bad:
            raise UserError("MIXED_SAMPLING",
                            f"{', '.join(bad)} is sampled at the progress checks, not at the "
                            "frames, so it cannot be plotted against another monitor",
                            f"plot it on its own: vfife plot <result> --y {bad[0]}")
        if not y:
            raise UserError(
                "PLOT_NEEDS_Y", "--x names the horizontal axis, so --y must "
                "name what to plot against it",
                "e.g. --x tip_disp --y reaction for a load-displacement curve. "
                "available: " + ", ".join(sorted(series)))
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for name in y:
            s = _pick(series, name)
            n = min(len(xs.values), len(s.values))
            # a marker when there are few points: a one-record result (a
            # single relaxation) drew an empty axes
            ax.plot(xs.values[:n], s.values[:n], lw=1.4, marker="o" if n < 3 else None,
                    label=s.label if len(y) > 1 else None)
        ax.set_xlabel(_axis(xs))
        ax.set_ylabel(_axis(_pick(series, y[0])) if len(y) == 1 else "")
        ax.axhline(0, color="0.75", lw=0.8, zorder=0)
        ax.axvline(0, color="0.75", lw=0.8, zorder=0)
        if len(y) > 1:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        names = list(y) if y else [n for n in sorted(series) if n != "time"]
        for n in names:
            _pick(series, n)
        if not names:
            raise UserError("NO_MONITORS", f"{path} records no series to plot",
                            "add an `output.monitors` block to the case")
        t = series.get("time")
        fig, axes = plt.subplots(len(names), 1, sharex=True,
                                 figsize=(7, 1.9 * len(names) + 1.0),
                                 squeeze=False)
        for ax, n in zip(axes[:, 0], names):
            s = series[n]
            if n in CONVERGENCE or t is None or len(t.values) != len(s.values):
                # on its own axis, the progress check: it is not sampled at
                # the frame times
                xv = np.arange(1, len(s.values) + 1, dtype=float)
                ax.set_xlabel("progress check", fontsize=7)
            else:
                xv = t.values
            if n in ("residual", "kinetic_energy") and np.all(s.values > 0):
                ax.semilogy(xv, s.values, lw=1.2, marker="o" if len(xv) < 3 else None)
            else:
                ax.plot(xv, s.values, lw=1.2, marker="o" if len(xv) < 3 else None)
            # stacked panels: the short name and unit on the axis, the long
            # description as the panel's title -- the long labels ran into
            # each other
            ax.set_ylabel(s.name + (f" [{s.units}]" if s.units else ""), fontsize=8)
            if s.label != s.name:
                ax.set_title(s.label, fontsize=7, loc="left")
            ax.grid(alpha=0.3)
        axes[-1, 0].set_xlabel(_time_axis(t) if t is not None else "output frame")

    fig.suptitle(os.path.basename(path), fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _axis(s: Series) -> str:
    if s.name == "time" and s.kind:
        return _time_axis(s)
    u = f" [{s.units}]" if s.units else ""
    return s.label + u


def _time_axis(t: Series) -> str:
    """The horizontal axis of a time plot, with what it measures: seconds in
    a dynamic run, the load factor (completed steps + lambda) in a
    relaxation or static one -- it read a bare "time"."""
    if t.kind == "load_factor":
        return "load factor (completed steps + λ)"
    return f"time [{t.units or 's'}]"
