"""
Build an e-mail-safe package of the project.

Mail servers (Gmail, Outlook/Exchange, corporate gateways) block executable attachments such as .bat files (Outlook
also blocks .py), look for them inside zip archives whatever the archive's extension, refuse password-protected
archives, and cap the size of a message (commonly 10, 20 or 25 MB, counted after the mail system has re-encoded the
attachments, which adds about 37%). This tool therefore:
  1. builds the source zip (same contents as tools/make_package.py) with the launchers renamed *.bat -> *.bat.txt
     and a restore script inside (tools/restore_launchers.py);
  2. encodes the zip as plain text (base64, 76-character lines), which every mail system accepts and which Windows
     decodes with the built-in `certutil -decode` command (macOS/Linux: `base64 -d`; any system: Python);
  3. splits the text into parts of at most --part-mb MB (default 10, about 14 MB per message once re-encoded)
     when needed, to be sent in several messages and joined with `copy /b` (Windows) or `cat` before decoding;
  4. writes DECODE_INSTRUCTIONS.txt (also printed) with the SHA-256 of the zip to check the transfer.

Usage:
    python tools/make_email_package.py                       # source only (about 0.3 MB of text, one part)
    python tools/make_email_package.py --include-data        # also data/raw, data/processed and output (several parts)
    python tools/make_email_package.py --part-mb 7 --out some/folder
"""
import sys
import base64
import hashlib
import pathlib
import argparse
import zipfile
import datetime as dt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_package  # noqa: E402

DATA_DIRS = ["data/raw", "data/processed", "output"]
MIME_FACTOR = 1.37          # size of an attachment inside the message after the mail system's base64 re-encoding


def build_zip(out, include_data=False):
    """Zip with the launchers renamed to .bat.txt (no blocked file types) and optionally the data and results."""
    files = []
    for d in make_package.INCLUDE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and make_package.wanted(p):
                files.append(p)
    for f in make_package.INCLUDE_FILES:
        p = ROOT / f
        if p.exists():
            files.append(p)
    extra = []
    if include_data:
        for d in DATA_DIRS:
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    extra.append(p)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files + extra:
            arc = "BCB model/" + p.relative_to(ROOT).as_posix()
            if arc.endswith(".bat"):
                arc += ".txt"
            z.write(p, arc)
        z.writestr("BCB model/READ_ME_FIRST.txt",
                   "This archive was prepared for e-mail: the launchers are stored as *.bat.txt.\n"
                   "After unzipping, run  python tools/restore_launchers.py  (or rename them by hand) to get\n"
                   "BCB model app.bat, run_app.bat, setup.bat and build_model.bat back. Then follow INSTALL.md.\n"
                   "Package built %s; %d files%s.\n" % (dt.datetime.now().isoformat(timespec="seconds"), len(files) + len(extra),
                                                        " including data and results" if include_data else " (source only; setup.bat downloads the data)"))
    return out, len(files) + len(extra)


def encode_text(zip_path, out_dir, part_mb):
    """Base64 text of the zip (CRLF lines, as a Windows text file), in one file or in parts of at most part_mb MB."""
    data = zip_path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    text = base64.encodebytes(data).decode("ascii").replace("\n", "\r\n")
    limit = int(part_mb * 1e6)
    if len(text) <= limit:
        p = out_dir / (zip_path.stem + ".b64.txt")
        p.write_bytes(text.encode("ascii"))
        return sha, [p]
    chunks, chunk, size = [], [], 0
    for ln in text.splitlines(keepends=True):
        if size + len(ln) > limit and chunk:
            chunks.append("".join(chunk)); chunk, size = [], 0
        chunk.append(ln); size += len(ln)
    if chunk:
        chunks.append("".join(chunk))
    parts = []
    for k, c in enumerate(chunks, 1):
        p = out_dir / ("%s.part%02dof%02d.txt" % (zip_path.stem, k, len(chunks)))
        p.write_bytes(c.encode("ascii")); parts.append(p)
    return sha, parts


def instructions(zip_name, sha, parts, n_files, part_mb):
    b64 = zip_name + ".b64.txt" if len(parts) > 1 else parts[0].name
    largest = max(p.stat().st_size for p in parts) / 1e6
    lines = ["HOW TO RECOVER THE PROJECT FROM THE E-MAIL ATTACHMENT(S)", "",
             "The attachment(s) are the zip archive '%s' encoded as plain text (base64, .txt), the form that mail" % zip_name,
             "servers accept. Archive: %d files. SHA-256 of the zip: %s" % (n_files, sha), ""]
    step = 1
    if len(parts) > 1:
        names = [p.name for p in parts]
        lines += ["%d. Save all %d parts in one folder and join them in order (Windows, command prompt or PowerShell):" % (step, len(parts)),
                  "       cmd /c copy /b " + "+".join('"%s"' % n for n in names) + ' "%s"' % b64,
                  "   (macOS/Linux:  cat %s > %s)" % (" ".join(names), b64), ""]
        step += 1
    lines += ["%d. Decode the text back into the zip (the zip must not exist yet in that folder):" % step,
              '       certutil -decode "%s" "%s"' % (b64, zip_name),
              "   (macOS/Linux:  base64 -d %s > %s" % (b64, zip_name),
              "    any system with Python:  python -c \"import base64; open('%s', 'wb').write(base64.b64decode(open('%s').read()))\")" % (zip_name, b64), "",
              "%d. Check the transfer (optional):  certutil -hashfile \"%s\" SHA256   must print the SHA-256 above." % (step + 1, zip_name), "",
              "%d. Unzip, open the folder 'BCB model' and run  python tools/restore_launchers.py  to rename the launchers" % (step + 2),
              "   (*.bat.txt -> *.bat). Then follow INSTALL.md: setup.bat (environment, references, data) and build_model.bat.", "",
              "Why this form: Gmail, Outlook and corporate gateways block .bat, .py and other executable attachments, look for them",
              "inside zip archives whatever the archive's extension, and refuse password-protected archives; a plain-text attachment",
              "passes everywhere. The largest attachment is %.1f MB on disk, about %.0f MB inside the message after the mail system's" % (largest, largest * MIME_FACTOR),
              "own encoding (parts are capped at %g MB; build with --part-mb to change the cap if the server's limit is lower)." % part_mb]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-data", action="store_true", help="include data/raw, data/processed and output (large)")
    ap.add_argument("--part-mb", type=float, default=10.0, help="maximum size of each text part in MB (default 10)")
    ap.add_argument("--out", default=str(ROOT / "email_package"), help="output folder")
    a = ap.parse_args()
    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*"):
        if old.is_file():
            old.unlink()
    zip_name = "BCB_model_%s.zip" % ("full" if a.include_data else "source")
    zip_path, n_files = build_zip(out_dir / zip_name, include_data=a.include_data)
    sha, parts = encode_text(zip_path, out_dir, a.part_mb)
    txt = instructions(zip_name, sha, parts, n_files, a.part_mb)
    (out_dir / "DECODE_INSTRUCTIONS.txt").write_text(txt, encoding="utf-8")
    print("zip: %s (%.2f MB, %d files, sha256 %s...)" % (zip_path.name, zip_path.stat().st_size / 1e6, n_files, sha[:16]))
    for p in parts:
        mb = p.stat().st_size / 1e6
        print("attach: %s (%.2f MB on disk, about %.1f MB inside the message)" % (p.name, mb, mb * MIME_FACTOR))
    print("instructions: %s (paste into the e-mail body)" % (out_dir / "DECODE_INSTRUCTIONS.txt"))
    print()
    print(txt)


if __name__ == "__main__":
    main()
