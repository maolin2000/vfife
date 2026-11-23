"""Write the manuals' table of what the compiled step takes.

    python tools/capability_tables.py          # rewrite the tables in place
    python tools/capability_tables.py --check  # exit 1 if one is stale

The table is `vfife.core.capabilities.table()`, the same rows the compiled
step and the compiled relaxation refuse with; each manual holds it between
`capabilities.BEGIN` and `capabilities.END`.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vfife.core import capabilities  # noqa: E402

MANUALS = {"docs/user_manual.md": "en", "docs/用户手册.md": "zh"}


def rendered(text, lang):
    """`text` with its table rewritten, or None when it has no markers."""
    i = text.find(capabilities.BEGIN)
    j = text.find(capabilities.END)
    if i < 0 or j < i:
        return None
    # the file's own line endings (a Windows checkout may have CRLF)
    nl = "\r\n" if "\r\n" in text else "\n"
    block = nl + nl + capabilities.table(lang).replace("\n", nl) + nl + nl
    return text[:i + len(capabilities.BEGIN)] + block + text[j:]


def main(argv):
    # the manuals' names are not ASCII, and a console may not be UTF-8
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    stale = []
    for rel, lang in MANUALS.items():
        path = os.path.join(ROOT, rel)
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        new = rendered(text, lang)
        if new is None:
            stale.append(f"{rel}: no table markers")
            continue
        if new != text:
            stale.append(rel)
            if "--check" not in argv:
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write(new)
    if stale:
        print(("stale: " if "--check" in argv else "written: ") + ", ".join(stale))
    return 1 if (stale and "--check" in argv) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
