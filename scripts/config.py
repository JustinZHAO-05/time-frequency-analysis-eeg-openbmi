from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "实验三-教学使用"
REPORT_DIR = ROOT / "report"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
DATA_OUT_DIR = RESULTS_DIR / "data"
CACHE_DIR = RESULTS_DIR / "cache"
RENDER_DIR = RESULTS_DIR / "rendered_pages"


def ensure_dirs() -> None:
    for path in [REPORT_DIR, RESULTS_DIR, FIG_DIR, TABLE_DIR, DATA_OUT_DIR, CACHE_DIR, RENDER_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def task_docx_path() -> Path:
    matches = list(ROOT.glob("*.docx"))
    if not matches:
        raise FileNotFoundError("未找到实验任务书 .docx")
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def mat_path() -> Path:
    matches = list(DATA_DIR.glob("**/*.mat"))
    if not matches:
        raise FileNotFoundError("未找到 EEG .mat 数据文件")
    return matches[0]


def paper_pdf_path() -> Path:
    matches = list(DATA_DIR.glob("**/*.pdf"))
    if not matches:
        raise FileNotFoundError("未找到数据集论文 PDF")
    return matches[0]


def rel_to_report(path: Path) -> str:
    return path.resolve().relative_to(REPORT_DIR.resolve()).as_posix()
