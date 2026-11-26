"""The errors a caller sees (1.22: from `core.diagnostics`, which still
exports them): a mistake in the input as a sentence with a code and a hint
(`UserError`), the ways an analysis ends without its answer
(`AnalysisFailure` and its kinds), and a run stopped on purpose."""
from __future__ import annotations


class AnalysisFailure(RuntimeError):
    def __init__(self, status, message, *, location=None):
        super().__init__(message); self.status = status; self.location = location or {}


class ConvergenceFailure(AnalysisFailure):
    def __init__(self, message, **kw): super().__init__("MAX_STEPS", message, **kw)


class NonFiniteFailure(AnalysisFailure):
    def __init__(self, message, **kw): super().__init__("NONFINITE", message, **kw)


class UnstableFailure(AnalysisFailure):
    def __init__(self, message, **kw): super().__init__("UNSTABLE", message, **kw)


class UserError(ValueError):
    """A mistake in the input, reported as a sentence rather than a traceback.

    A `ValueError` subclass on purpose: the input paths already raised plain
    ValueErrors and several callers catch them, so inheriting keeps every
    existing handler working while adding the code/hint a caller can act on.

    Carries the three things a caller needs to act, whether that caller is a
    terminal or a dialog box: a stable `code` to branch on, a `message` saying
    what is wrong, and a `hint` saying what to do about it.  A GUI can render
    the same object; before this, `api.prepare_case` raised a bare `KeyError`
    and the CLI printed twenty lines of stack for a typo.
    """

    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "hint": self.hint}

    def __str__(self) -> str:
        return self.message


class StopAnalysis(Exception):
    """Raised inside the time loop when a caller asks the solve to stop.

    `on_progress` returning False means stop: that is the cancel token a GUI
    needs, and it costs the solver one comparison per progress interval.  The
    partial result is still written -- a cancelled run is not a lost one.
    """


class AnalysisInterrupted(Exception):
    """A run that was stopped on purpose, after its output was preserved.

    `signal` is the signal that stopped it (`signal.SIGINT`, `signal.SIGTERM`),
    or None when `on_progress` returned False. A batch script that catches
    `Exception` around `run_case` should look at it: a SIGTERM is the
    machine's time limit or a shutdown asking the PROCESS to end, and the
    library does not end it for the script; `vfife run`
    exits 143 on it, 130 on Ctrl-C."""

    def __init__(self, result_path, sidecars, report, signal=None):
        super().__init__("analysis interrupted" + (f" by signal {int(signal)}" if signal else ""))
        self.result_path = result_path
        self.sidecars = sidecars
        self.report = report
        self.signal = signal
