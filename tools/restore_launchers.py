"""
After unzipping an e-mail package: gives the launchers back their original names (*.bat.txt -> *.bat), which were
changed so that mail servers do not block the archive. Usage:  python tools/restore_launchers.py
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
n = 0
for p in list(ROOT.glob("*.bat.txt")) + list(ROOT.glob("*/*.bat.txt")):
    target = p.with_name(p.name[:-4])
    p.rename(target)
    print("restored", target.relative_to(ROOT))
    n += 1
print("%d launcher(s) restored" % n if n else "nothing to restore (launchers already have their .bat names)")
