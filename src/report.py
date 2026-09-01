"""
src/report.py

Shared helper for writing marked sections into reports/results.md.

Several scripts write into the same report file. Each owns exactly one
marker-delimited block and must leave every other block untouched, so that
re-running one script never silently discards another's output. Keeping the
splice logic in one place means the scripts can't drift apart in how they
do it.

All reads/writes pass encoding="utf-8" explicitly: the platform default is
locale-dependent on Windows and will mangle non-ASCII characters (em dashes
in the report tables) on a round trip.
"""
from __future__ import annotations

from pathlib import Path


def markers(name: str) -> tuple[str, str]:
    """HTML-comment marker pair for a named section."""
    tag = name.upper()
    return f"<!-- {tag}_START -->", f"<!-- {tag}_END -->"


def update_section(path: Path, name: str, section_text: str) -> None:
    """Replace the named section of a markdown file with section_text,
    appending it if the file has no such section yet. Any content outside
    this section's markers is preserved exactly."""
    start, end = markers(name)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    block = f"{start}\n{section_text}\n{end}"

    if start in existing and end in existing:
        pre = existing.split(start)[0]
        post = existing.split(end)[1]
        new_content = pre + block + post
    elif existing:
        sep = "\n" if existing.endswith("\n") else "\n\n"
        new_content = existing + sep + block + "\n"
    else:
        new_content = block + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_content, encoding="utf-8")
