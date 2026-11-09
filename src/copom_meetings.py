"""
Copom meetings used in the projection backtest: dates, reference-scenario assumptions and the projections published
by the Copom (statement and minutes, Table 1 "Projeções de inflação no cenário de referência").

The first ten meetings (272nd to 281st) are transcribed from the minutes (refs/copom/atas/*.pdf, text in
refs/copom/atas/txt); the statements carry the same figures. `check_against_minutes()` re-reads the minutes and
verifies the transcription. Later meetings are added automatically by `update_history()`, which downloads the new
minutes from the BCB site, parses Table 1 and the assumptions, and stores them in data/processed/copom_history.json;
`check_parser()` verifies that the parser reproduces the transcribed table for the known meetings.

Conventions of the Copom reference scenario (minutes, section "Cenários e análise de riscos"):
* Selic path: medians of the Focus survey (per-meeting expectations) of the Friday before the meeting week;
* exchange rate: starts at the average PTAX of the ten business days ending on the last day of the week before the
  meeting, rounded to R$0.05 (footnote of the minutes, procedure adopted at the 258th meeting), then follows PPP;
* oil: approximately the futures curve for six months, +2% a year afterwards;
* electricity tariff flag ("bandeira tarifária") in December as stated;
* neutral real interest rate 5.00% (Monetary Policy Reports, 2025-2026, "Projeções condicionais");
* relevant horizon: six quarters after the quarter of the meeting.
"""
import re
import json
import pathlib

import pandas as pd

from config import REFS, DATA_PROC

# projections: {"years": {year: (IPCA, IPCA livres, IPCA administrados)}, "horizon": (quarter, IPCA, livres, adm)}
MEETINGS = [
    {"n": 272, "decision": "2025-07-30", "focus": "2025-07-25", "fx_start": 5.55, "flag": "verde (dez/2025 e dez/2026)",
     "years": {2025: (4.9, 5.1, 4.4), 2026: (3.6, 3.5, 4.0)}, "horizon": ("2027Q1", 3.4, 3.3, 3.9),
     "minutes": "Copom272-not20250730272"},
    {"n": 273, "decision": "2025-09-17", "focus": "2025-09-12", "fx_start": 5.40, "flag": "verde (dez/2025 e dez/2026)",
     "years": {2025: (4.8, 5.0, 4.3), 2026: (3.6, 3.5, 3.8)}, "horizon": ("2027Q1", 3.4, 3.3, 3.8),
     "minutes": "Copom273-not20250917273"},
    {"n": 274, "decision": "2025-11-05", "focus": "2025-10-31", "fx_start": 5.40, "flag": "amarela (dez/2025 e dez/2026)",
     "years": {2025: (4.6, 4.5, 5.0), 2026: (3.6, 3.6, 3.4)}, "horizon": ("2027Q2", 3.3, 3.2, 3.5),
     "minutes": "Copom274-not20251105274"},
    {"n": 275, "decision": "2025-12-10", "focus": "2025-12-05", "fx_start": 5.35, "flag": "amarela (dez/2025 e dez/2026)",
     "years": {2025: (4.4, 4.0, 5.3), 2026: (3.5, 3.6, 3.2)}, "horizon": ("2027Q2", 3.2, 3.2, 3.4),
     "minutes": "Copom275-not20251210275"},
    {"n": 276, "decision": "2026-01-28", "focus": "2026-01-23", "fx_start": 5.35, "flag": "amarela (dez/2026)",
     "years": {2026: (3.4, 3.5, 3.0)}, "horizon": ("2027Q3", 3.2, 3.1, 3.3),
     "minutes": "Copom276-not20260128276"},
    {"n": 277, "decision": "2026-03-18", "focus": "2026-03-13", "fx_start": 5.20, "flag": "amarela (dez/2026)",
     "years": {2026: (3.9, 3.7, 4.3)}, "horizon": ("2027Q3", 3.3, 3.3, 3.2),
     "minutes": "Copom277-not20260318277"},
    {"n": 278, "decision": "2026-04-29", "focus": "2026-04-24", "fx_start": 5.00, "flag": "amarela (dez/2026)",
     "years": {2026: (4.6, 4.5, 4.8)}, "horizon": ("2027Q4", 3.5, 3.5, 3.6),
     "minutes": "Copom278-not20260429278"},
    {"n": 279, "decision": "2026-06-17", "focus": "2026-06-12", "fx_start": 5.10, "flag": "amarela (dez/2026)",
     "years": {2026: (5.2, 5.3, 4.7)}, "horizon": ("2027Q4", 3.7, 3.7, 3.9),
     "minutes": "Copom279-not20260617279"},
    {"n": 280, "decision": "2026-08-05", "focus": "2026-07-31", "fx_start": 5.10, "flag": "amarela (dez/2026)",
     "years": {2026: (5.1, 5.1, 4.9), 2027: (3.8, 3.9, 3.5)}, "horizon": ("2028Q1", 3.2, 3.1, 3.5),
     "minutes": "Copom280-not20260805280"},
    {"n": 281, "decision": "2026-09-16", "focus": "2026-09-11", "fx_start": 5.15, "flag": "amarela (dez/2026)",
     "years": {2026: (5.2, 5.2, 5.3), 2027: (3.9, 4.1, 3.5)}, "horizon": ("2028Q1", 3.2, 3.2, 3.2),
     "minutes": "Copom281-not20260916281"},
]

# EIA Short-Term Energy Outlook vintages available locally (forecast dates); the vintage used for a meeting is the
# latest one published (forecast date + 6 days) on or before the Focus reference date. Later months are tried as
# downloads by projection.steo_vintage_for.
STEO_VINTAGES = ["jun25", "jul25", "aug25", "sep25", "oct25", "nov25", "dec25", "jan26", "feb26", "mar26", "apr26",
                 "may26", "jun26", "jul26", "aug26", "sep26", "oct26"]

HISTORY = DATA_PROC / "copom_history.json"
ATAS_DIR = REFS / "copom" / "atas"
LIST_URL = "https://www.bcb.gov.br/api/servico/sitebcb/atascopom/ultimas?quantidade=%d&filtro="
# scheduled meetings (decision dates -> meeting number) from the calendars published by the BCB (2026: June 2025;
# 2027: June 2026); completed by the history API as meetings take place
CALENDAR = {"2026-11-04": 282, "2026-12-09": 283, "2027-01-27": 284, "2027-03-17": 285, "2027-04-28": 286, "2027-06-16": 287,
            "2027-08-04": 288, "2027-09-22": 289, "2027-10-27": 290, "2027-12-08": 291}
HORIZON_QUARTERS = 6      # relevant horizon: quarters after the quarter of the meeting
_PT_QUARTER = {"primeiro": 1, "segundo": 2, "terceiro": 3, "quarto": 4}


def meeting(n):
    for m in all_meetings():
        if m["n"] == n:
            return m
    raise KeyError(n)


def horizon_quarter(q):
    return pd.Period(q, freq="Q")


def relevant_horizon(decision_date):
    return str(pd.Timestamp(decision_date).to_period("Q") + HORIZON_QUARTERS)


def _num(s):
    return float(s.replace(",", "."))


def focus_date_for(decision_date):
    """Friday before the week of the (two-day) meeting."""
    d = pd.Timestamp(decision_date) - pd.Timedelta(days=1)
    back = (d.weekday() - 4) % 7
    back = 7 if back == 0 else back
    return (d - pd.Timedelta(days=back)).strftime("%Y-%m-%d")


def check_against_minutes(verbose=True):
    """Re-read the minutes and compare the transcribed IPCA projections with the sentence
    'as projeções ... são, respectivamente, x% e y%' and the exchange-rate start value."""
    ok = True
    for m in MEETINGS:
        t = (REFS / "copom" / "atas" / "txt" / (m["minutes"] + ".txt")).read_text(encoding="utf-8")
        t1 = re.sub(r"\s+", " ", t)
        sent = re.search(r"as projeções para a inflação acumulada em quatro trimestres para ([^.]*?)são, respectivamente, ([^.]*?)\.", t1)
        nums = [_num(x) for x in re.findall(r"(\d,\d)%", sent.group(2))] if sent else []
        expected = [v[0] for _, v in sorted(m["years"].items())] + [m["horizon"][1]]
        if sent and "trimestre" not in sent.group(1) and "horizonte" not in sent.group(1):
            expected = [v[0] for _, v in sorted(m["years"].items())]   # the horizon is reported in another sentence
        fx = re.search(r"taxa de câmbio parte de (?:\d bcb\.gov\.br \d+ª Reunião COPOM [^R]*)?R\$(\d,\d\d)/US\$", t1)
        fx_ok = fx is not None and abs(_num(fx.group(1)) - m["fx_start"]) < 1e-9
        this_ok = (nums == expected) and fx_ok
        ok = ok and this_ok
        if verbose:
            print("%d %s | minutes: %s | table: %s | fx %s | %s" % (m["n"], m["decision"], nums, expected,
                  fx.group(1) if fx else None, "OK" if this_ok else "MISMATCH"))
    return ok


# ---------------------------------------------------------------------------
# Automatic update of the meeting history from the minutes published by the BCB
# ---------------------------------------------------------------------------
def minutes_list(n=20):
    import requests
    r = requests.get(LIST_URL % n, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    return r.json()["conteudo"]


def download_minutes(entry):
    """Download one minutes PDF (if absent) and return the path of its extracted text."""
    import requests
    from pdfminer.high_level import extract_text
    ATAS_DIR.mkdir(parents=True, exist_ok=True)
    (ATAS_DIR / "txt").mkdir(exist_ok=True)
    pdf = ATAS_DIR / pathlib.Path(entry["Url"]).name
    if not pdf.exists():
        r = requests.get("https://www.bcb.gov.br" + entry["Url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
        r.raise_for_status()
        pdf.write_bytes(r.content)
    txt = ATAS_DIR / "txt" / (pdf.stem + ".txt")
    if not txt.exists():
        txt.write_text(extract_text(str(pdf)), encoding="utf-8")
    return txt


def _quarter_from_words(word, year):
    return "%dQ%d" % (int(year), _PT_QUARTER[word])


def parse_minutes(text, decision_date, n):
    """Parse a minutes text into a meeting record (same structure as MEETINGS). Raises ValueError when the
    projections cannot be identified."""
    t1 = re.sub(r"\s+", " ", text)
    # 1) headline sentence with the IPCA projections by target
    m = re.search(r"as projeções para a inflação acumulada em quatro trimestres para (.*?) são, respectivamente, (.*?)\.", t1)
    if not m:
        raise ValueError("projection sentence not found")
    tgt_txt, num_txt = m.group(1), m.group(2)
    targets = []
    for tok in re.finditer(r"(primeiro|segundo|terceiro|quarto) trimestre de (20\d\d)|(?<!de )(20\d\d)", tgt_txt):
        if tok.group(1):
            targets.append(_quarter_from_words(tok.group(1), tok.group(2)))
        elif tok.group(3):
            targets.append(int(tok.group(3)))
    nums = [_num(x) for x in re.findall(r"(-?\d,\d)%", num_txt)]
    if len(nums) != len(targets):
        raise ValueError("targets %s and numbers %s do not match" % (targets, nums))
    ipca = dict(zip(targets, nums))
    horizon_q = next((x for x in targets if isinstance(x, str)), None)
    if horizon_q is None:  # horizon reported in a separate sentence (2025 phrasing), possibly cut by a footnote
        h = re.search(r"horizonte relevante de política monetária, (primeiro|segundo|terceiro|quarto) trimestre de (20\d\d), a projeção", t1)
        if h:
            v = re.search(r"em (-?\d,\d)%", t1[h.end():h.end() + 700])
            if v:
                horizon_q = _quarter_from_words(h.group(1), h.group(2))
                ipca[horizon_q] = _num(v.group(1))
    years = [x for x in ipca if isinstance(x, int)]
    cols = years + ([horizon_q] if horizon_q else [])
    # 2) Table 1 block with the free/administered split
    i = t1.find("Projeções de inflação no cenário de referência")
    block = t1[i:i + 400] if i >= 0 else ""
    j = block.find("IPCA")
    vals = [_num(x) for x in re.findall(r"(?<![\d,])(-?\d,\d)(?![\d,])", block[j:])] if j >= 0 else []
    nc = len(cols)
    rows = None
    if len(vals) >= 3 * nc:
        vals = vals[:3 * nc]
        cand = {"column-wise": {c: (vals[k * 3], vals[k * 3 + 1], vals[k * 3 + 2]) for k, c in enumerate(cols)}}
        ny = len(years)
        if horizon_q and ny >= 1:
            lay = {c: (vals[k], vals[ny + k], vals[2 * ny + k]) for k, c in enumerate(years)}
            lay[horizon_q] = (vals[3 * ny], vals[3 * ny + 1], vals[3 * ny + 2])
            cand["row-wise"] = lay
        for name, lay in cand.items():
            ok = all(abs(lay[c][0] - ipca[c]) < 1e-9 for c in cols)
            ok = ok and all(abs(0.74 * lay[c][1] + 0.26 * lay[c][2] - lay[c][0]) < 0.25 for c in cols)
            if ok:
                rows = lay
                break
    if rows is None:  # split not identified: keep the headline numbers only
        rows = {c: (ipca[c], None, None) for c in cols}
    fx = re.search(r"taxa de câmbio parte de (?:\d bcb\.gov\.br \d+ª Reunião COPOM [^R]*)?R\$(\d,\d\d)/US\$", t1)
    flag = re.search(r"bandeira tarifária \W?(\w+)", t1)
    rec = {"n": int(n), "decision": str(pd.Timestamp(decision_date).date()), "focus": focus_date_for(decision_date),
           "fx_start": _num(fx.group(1)) if fx else None, "flag": flag.group(1) if flag else None,
           "years": {int(y): tuple(rows[y]) for y in years},
           "horizon": (horizon_q,) + tuple(rows[horizon_q]) if horizon_q else None, "minutes": None, "parsed": True}
    return rec


def _entry_number(entry):
    m = re.search(r"(\d+)ª Reuni", entry["Titulo"])
    return int(m.group(1)) if m else None


def update_history(n_list=20, verbose=True):
    """Fetch the list of minutes from the BCB, parse the meetings not yet in MEETINGS or in the saved history, and
    save data/processed/copom_history.json. Returns the full list of meetings (transcribed + parsed)."""
    hist = json.loads(HISTORY.read_text(encoding="utf-8")) if HISTORY.exists() else {}
    known = {m["n"] for m in MEETINGS} | {int(k) for k in hist}
    try:
        entries = minutes_list(n_list)
    except Exception as e:  # noqa: BLE001
        if verbose:
            print("could not fetch the minutes list:", e)
        entries = []
    for e in entries:
        n = _entry_number(e)
        if n is None or n in known or n < MEETINGS[0]["n"]:
            continue
        try:
            txt = download_minutes(e)
            rec = parse_minutes(txt.read_text(encoding="utf-8"), e["DataReferencia"][:10], n)
            rec["minutes"] = txt.stem
            hist[str(n)] = rec
            if verbose:
                print("parsed minutes of meeting %d (%s): %s" % (n, rec["decision"], rec["years"]))
        except Exception as ex:  # noqa: BLE001
            if verbose:
                print("meeting %d: could not parse the minutes (%s)" % (n, ex))
    HISTORY.write_text(json.dumps(hist, indent=1, ensure_ascii=False), encoding="utf-8")
    return all_meetings()


def download_known_minutes(n_list=20, verbose=True):
    """Download the minutes (PDF + text) of the transcribed meetings when they are not in refs/copom/atas, so that
    check_against_minutes() and check_parser() can run on a fresh installation."""
    try:
        entries = minutes_list(n_list)
    except Exception as e:  # noqa: BLE001
        if verbose:
            print("could not fetch the minutes list:", e)
        return 0
    got = 0
    for e in entries:
        n = _entry_number(e)
        if n is None or n not in {m["n"] for m in MEETINGS}:
            continue
        txt = ATAS_DIR / "txt" / (pathlib.Path(e["Url"]).stem + ".txt")
        if not txt.exists():
            download_minutes(e)
            got += 1
    if verbose:
        print("minutes downloaded:", got)
    return got


def all_meetings():
    """Transcribed meetings plus the parsed ones from the saved history, sorted by number."""
    out = [dict(m) for m in MEETINGS]
    if HISTORY.exists():
        hist = json.loads(HISTORY.read_text(encoding="utf-8"))
        for k, rec in hist.items():
            rec = dict(rec)
            rec["years"] = {int(y): tuple(v) for y, v in rec["years"].items()}
            rec["horizon"] = tuple(rec["horizon"]) if rec.get("horizon") else None
            if rec["n"] not in {m["n"] for m in out}:
                out.append(rec)
    return sorted(out, key=lambda m: m["n"])


def next_meeting(today=None):
    """(number, decision date) of the next scheduled meeting on or after `today`."""
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    past = {m["decision"] for m in all_meetings()}
    for d, n in sorted((pd.Timestamp(d), n) for d, n in CALENDAR.items()):
        if d >= today and d.strftime("%Y-%m-%d") not in past:
            return n, d
    return None, None


def check_parser(verbose=True):
    """The parser must reproduce the transcribed table for every meeting whose minutes are available."""
    ok = True
    for m in MEETINGS:
        txt = REFS / "copom" / "atas" / "txt" / (m["minutes"] + ".txt")
        rec = parse_minutes(txt.read_text(encoding="utf-8"), m["decision"], m["n"])
        same = rec["years"] == m["years"] and rec["horizon"] == m["horizon"] and rec["fx_start"] == m["fx_start"] and rec["focus"] == m["focus"]
        ok = ok and same
        if verbose:
            print("%d parser %s %s %s fx %s focus %s" % (m["n"], "OK" if same else "MISMATCH", rec["years"], rec["horizon"], rec["fx_start"], rec["focus"]))
    return ok


if __name__ == "__main__":
    print("transcription consistent:", check_against_minutes())
    print("parser consistent:", check_parser())
