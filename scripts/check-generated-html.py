#!/usr/bin/env python3
"""Fail a site build if generated HTML references a forbidden remote origin."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


FORBIDDEN_URLS = {
    "polyfill.io": re.compile(
        r"(?:https?:)?//(?:[a-z0-9-]+\.)*polyfill\.io(?:[/:?#]|$)",
        re.IGNORECASE,
    ),
    "an unpinned KaTeX release": re.compile(
        r"(?:https?:)?//cdn\.jsdelivr\.net/npm/katex@latest(?:[/:?#]|$)",
        re.IGNORECASE,
    ),
}


def check_site(site_dir: Path) -> list[tuple[Path, int, str]]:
    findings: list[tuple[Path, int, str]] = []

    for html_file in sorted(site_dir.rglob("*.html")):
        contents = html_file.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(contents.splitlines(), start=1):
            for origin, pattern in FORBIDDEN_URLS.items():
                if pattern.search(line):
                    findings.append((html_file, line_number, origin))

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check generated HTML for forbidden remote origins."
    )
    parser.add_argument(
        "site_dir",
        nargs="?",
        type=Path,
        default=Path("_site"),
        help="generated site directory (default: _site)",
    )
    args = parser.parse_args()

    if not args.site_dir.is_dir():
        parser.error(f"site directory does not exist: {args.site_dir}")

    html_files = list(args.site_dir.rglob("*.html"))
    if not html_files:
        parser.error(f"no HTML files found under: {args.site_dir}")

    findings = check_site(args.site_dir)
    if findings:
        print("Unsafe generated HTML detected:")
        for path, line_number, origin in findings:
            print(f"  {path}:{line_number}: references {origin}")
        print("\nThe site was not accepted. Remove the unsafe origin and render again.")
        return 1

    print(f"Checked {len(html_files)} HTML files: no forbidden origins found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
