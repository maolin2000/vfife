"""`ruff check vfife` is clean (pyproject.toml, [tool.ruff]): no name that
does not exist, no import nobody uses. Skipped where ruff is not installed
(it is in the `dev` extra, so CI and the local full check run it)."""
import shutil
import subprocess
import sys

import pytest


def test_the_package_is_lint_clean():
    if shutil.which("ruff") is None:
        try:
            import ruff  # noqa: F401
        except ImportError:
            pytest.skip("ruff is not installed (pip install -e .[dev])")
    out = subprocess.run([sys.executable, "-m", "ruff", "check", "vfife", "--output-format", "concise"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-1000:]
