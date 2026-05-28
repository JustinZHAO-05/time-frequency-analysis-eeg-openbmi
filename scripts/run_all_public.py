from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from config import ROOT, ensure_dirs


STEPS = [
    "experiment1.py",
    "experiment2.py",
    "experiment3.py",
    "experiment4_eeg.py",
    "english_figures.py",
    "report_writer_en.py",
    "build_report_en.py",
]


def run_script(script: str) -> None:
    path = Path(__file__).resolve().parent / script
    print(f"\n[run_all_public] running {script}")
    start = time.time()
    subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)
    print(f"[run_all_public] finished {script} in {time.time() - start:.1f}s")


def main() -> None:
    ensure_dirs()
    for script in STEPS:
        run_script(script)


if __name__ == "__main__":
    main()
