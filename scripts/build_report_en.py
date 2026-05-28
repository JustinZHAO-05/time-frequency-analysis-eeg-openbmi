from __future__ import annotations

import shutil
import subprocess

from config import REPORT_DIR, ROOT


def main() -> None:
    tex = REPORT_DIR / "main_en.tex"
    if not tex.exists():
        raise FileNotFoundError("report/main_en.tex does not exist; run scripts/report_writer_en.py first")
    latexmk = shutil.which("latexmk")
    if not latexmk:
        raise RuntimeError("latexmk was not found; cannot compile the English PDF")
    cmd = [
        latexmk,
        "-xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "main_en.tex",
    ]
    print("[build_report_en] compiling report/main_en.tex")
    subprocess.run(cmd, cwd=REPORT_DIR, check=True)
    out = REPORT_DIR / "main_en.pdf"
    final = ROOT / "Experiment_3_Time_Frequency_Analysis_Report_EN.pdf"
    if out.exists():
        shutil.copy2(out, final)
        print(f"[build_report_en] PDF copied to {final}")


if __name__ == "__main__":
    main()
