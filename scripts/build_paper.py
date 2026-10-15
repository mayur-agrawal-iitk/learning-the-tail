"""Step 6: copy the latest tables/figures into paper/ and compile paper/main.pdf.

Runs pdflatex -> bibtex -> pdflatex -> pdflatex (needs MiKTeX or TeX Live on PATH).
Usage: python scripts/build_paper.py [--config main]
"""

import argparse
import re
import subprocess
import sys

import _bootstrap  # noqa: F401

from tailrisk.config import ROOT

PAPER = ROOT / "paper"


def run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, cwd=PAPER, capture_output=True, text=True, errors="replace")
    if res.returncode != 0:
        errors = [l for l in res.stdout.splitlines() if l.startswith("!")]
        sys.exit(f"{cmd[0]} failed:\n" + "\n".join(errors[:10] or res.stdout.splitlines()[-20:]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="main")
    args = ap.parse_args()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "export_paper.py"), "--config", args.config],
                   check=True)
    latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    run(latex)
    run(["bibtex", "main"])
    run(latex)
    run(latex)
    log = (PAPER / "main.log").read_text(encoding="latin-1")
    pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    undefined = sorted(set(re.findall(r"(?:Citation|Reference) `([^']+)' .*undefined", log)))
    print(f"built {PAPER / 'main.pdf'} ({pages.group(1) if pages else '?'} pages)")
    if undefined:
        print("undefined citations/references:", ", ".join(undefined))


if __name__ == "__main__":
    main()
