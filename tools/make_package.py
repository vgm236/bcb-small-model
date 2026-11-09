"""
Build a single zip file with everything needed to rebuild the whole project from scratch (code, notebooks,
documentation, launchers, requirements and setup scripts), without the downloaded data, the reference PDFs, the
outputs or the virtual environment. Usage:  python tools/make_package.py  [output path]
The archive is written to BCB_model_source.zip in the project folder by default.
"""
import os
import sys
import pathlib
import zipfile
import datetime as dt

ROOT = pathlib.Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ["src", "notebooks", "app", "docs", "tools"]
INCLUDE_FILES = ["README.md", "INSTALL.md", "requirements.txt", "setup.bat", "build_model.bat", "BCB model app.bat", "run_app.bat"]
EXCLUDE_SUFFIX = (".pyc", ".log", ".png", ".csv", ".npy", ".json", ".pdf", ".txt")
KEEP_TXT = {"requirements.txt"}


def wanted(path):
    rel = path.relative_to(ROOT)
    if any(part in ("__pycache__", ".venv", "data", "output", "refs", ".claude") for part in rel.parts):
        return False
    if path.name in KEEP_TXT:
        return True
    if path.suffix in EXCLUDE_SUFFIX:
        return False
    if path.name.startswith("_patch"):
        return False
    return True


def build(out_path=None):
    out = pathlib.Path(out_path) if out_path else ROOT / "BCB_model_source.zip"
    files = []
    for d in INCLUDE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and wanted(p):
                files.append(p)
    for f in INCLUDE_FILES:
        p = ROOT / f
        if p.exists():
            files.append(p)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, "BCB model/" + p.relative_to(ROOT).as_posix())
        manifest = "\n".join(p.relative_to(ROOT).as_posix() for p in files)
        z.writestr("BCB model/MANIFEST.txt", "Package built %s\n%d files\n\n%s\n" % (dt.datetime.now().isoformat(timespec="seconds"), len(files), manifest))
    size = out.stat().st_size
    print("wrote %s (%d files, %.0f KB)" % (out, len(files), size / 1024))
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else None)
