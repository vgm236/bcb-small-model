"""
Download the BCB reference documents used by the project (boxes of the Inflation Report / Monetary Policy
Report and the full reports of 2024-2026) into refs/ and extract their text into refs/txt (pdfminer), and fetch the
minutes of the transcribed Copom meetings. Run once on a fresh installation:  python src/download_refs.py

The files are public: https://www.bcb.gov.br/content/ri/relatorioinflacao/<yyyymm>/<file>.pdf. Files already
present are kept; failures are reported and do not stop the script (the code has fallbacks for the two inputs it
reads from these texts: the published output-gap estimates and the minutes' consistency checks).
"""
import re
import sys
import pathlib

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REFS = ROOT / "refs"
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}

# boxes (refs/) and full reports (refs/full/); the text is extracted for the files marked with *
BOXES = ["ri201706b7p.pdf", "ri202009b7p.pdf", "ri202103b7p.pdf", "ri202103b9p.pdf", "ri202112b7p.pdf*", "ri202303b6p.pdf", "ri202306b6p.pdf",
         "ri202403b4p.pdf", "ri202406b11p.pdf", "ri202406b12p.pdf*", "rpm202506b8p.pdf*", "rpm202506b9p.pdf*", "rpm202509b2p.pdf",
         "rpm202603b8p.pdf", "rpm202606b6p.pdf"]
FULL = ["ri202406p.pdf", "ri202409p.pdf", "ri202412p.pdf", "rpm202503p.pdf*", "rpm202506p.pdf*", "rpm202509p.pdf*", "rpm202512p.pdf*",
        "rpm202603p.pdf*", "rpm202606p.pdf*", "rpm202609p.pdf*"]


def url_for(name):
    m = re.match(r"(ri|rpm)(\d{6})", name)
    return "https://www.bcb.gov.br/content/ri/relatorioinflacao/%s/%s" % (m.group(2), name)


def get(url, dest):
    if dest.exists() and dest.stat().st_size > 1000:
        return "kept"
    try:
        r = requests.get(url, headers=H, timeout=180)
        if r.status_code == 200 and b"%PDF" in r.content[:1024]:
            dest.write_bytes(r.content)
            return "downloaded (%d KB)" % (len(r.content) // 1024)
        return "FAILED (HTTP %s)" % r.status_code
    except Exception as e:  # noqa: BLE001
        return "FAILED (%s)" % e


def extract(pdf, txt):
    if txt.exists():
        return "kept"
    from pdfminer.high_level import extract_text
    txt.write_text(extract_text(str(pdf)), encoding="utf-8")
    return "extracted"


def main():
    (REFS / "full").mkdir(parents=True, exist_ok=True)
    (REFS / "txt").mkdir(exist_ok=True)
    for lst, folder in [(BOXES, REFS), (FULL, REFS / "full")]:
        for item in lst:
            name = item.rstrip("*")
            dest = folder / name
            print("%-22s %s" % (name, get(url_for(name), dest)))
            if item.endswith("*") and dest.exists():
                print("%-22s text %s" % ("", extract(dest, REFS / "txt" / (name[:-4] + ".txt"))))
    try:
        import copom_meetings as cm
        print("minutes of the transcribed meetings:", cm.download_known_minutes(verbose=False), "downloaded")
    except Exception as e:  # noqa: BLE001
        print("minutes: could not download (%s)" % e)


if __name__ == "__main__":
    main()
