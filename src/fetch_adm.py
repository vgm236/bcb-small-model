"""
Download layer for the administered-price model: IPCA subitems (IBGE SIDRA), electricity tariff flags (ANEEL open
data) and the US CPI (BLS). Cached in data/raw like fetch.py; pass refresh=True to re-download.
"""
import io
import json
import datetime as dt

import pandas as pd
import requests

from config import DATA_RAW
from fetch import _get, _save_meta, HEADERS

# IPCA subitems of the administered-price model (June-2025 box): IBGE subitem code -> SIDRA c315 category id
# (the same ids in tables 7060 (from 2020) and 1419 (2012-2019)); two subitems exist only from 2020.
SUBITEMS = {
    "5104001": ("7657", "Gasolina"), "6203": ("7695", "Plano de saúde"), "2202003": ("7485", "Energia elétrica residencial"),
    "6101": ("7662", "Produtos farmacêuticos"), "5102004": ("7642", "Emplacamento e licença"), "2101004": ("7451", "Taxa de água e esgoto"),
    "2201004": ("7482", "Gás de botijão"), "5101001": ("7628", "Ônibus urbano"), "7201063": ("107668", "Jogos de azar"),
    "5101006": ("7631", "Ônibus intermunicipal"), "5104003": ("7659", "Óleo diesel"), "9101002": ("7790", "Plano de telefonia fixa"),
    "5101002": ("7629", "Táxi"), "2201005": ("7483", "Gás encanado"), "5101007": ("7632", "Ônibus interestadual"),
    "5102006": ("107653", "Multa"), "5102015": ("7649", "Pedágio"), "5104005": ("107657", "Gás veicular"),
    "5101011": ("7635", "Metrô"), "9101001": ("7789", "Correio"), "5101053": ("47650", "Integração transporte público"),
    "7101090": ("7728", "Conselho de classe"), "5101004": ("7630", "Trem"), "7101034": ("7723", "Cartório"),
}
SIDRA_IDS_7060 = {k: (("47668" if k == "9101002" else v[0]), v[1]) for k, v in SUBITEMS.items()}   # telephone plan id differs in 7060
GENERAL = "7169"
_PT_MONTHS = {"janeiro": 1, "fevereiro": 2, "março": 3, "abril": 4, "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
              "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12}


def _parse_sidra(js):
    rows = []
    for r in js[1:]:
        mon, year = r["D3N"].split()
        rows.append({"month": pd.Period(year=int(year), month=_PT_MONTHS[mon.lower()], freq="M"), "cat": r["D4C"], "name": r["D4N"],
                     "variable": "var" if r["D2C"] == "63" else "weight", "value": pd.to_numeric(r["V"], errors="coerce")})
    return pd.DataFrame(rows)


def sidra_subitems(refresh=False):
    """Monthly variation (%) and weight (%) of the administered subitems and of the general index, 2012 onwards
    (SIDRA tables 1419 for 2012-2019 and 7060 from 2020). Returns a long DataFrame (month, code, name, variable, value)."""
    name = "sidra_ipca_subitems_adm"
    path = DATA_RAW / (name + ".csv")
    if path.exists() and not refresh:
        df = pd.read_csv(path, dtype={"code": str, "cat": str})
        df["month"] = pd.PeriodIndex(df["month"], freq="M")
        return df
    frames = []
    for table, ids in [("1419", {k: v[0] for k, v in SUBITEMS.items()}), ("7060", {k: v[0] for k, v in SIDRA_IDS_7060.items()})]:
        cats = ",".join(sorted(set(list(ids.values()) + [GENERAL])))
        url = "https://apisidra.ibge.gov.br/values/t/%s/n1/all/v/63,66/p/all/c315/%s" % (table, cats)
        r = _get(url, timeout=300)
        df = _parse_sidra(r.json())
        inv = {v: k for k, v in ids.items()}
        inv[GENERAL] = "geral"
        df["code"] = df["cat"].map(inv)
        df["table"] = table
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).dropna(subset=["code"])
    out = out.sort_values(["code", "variable", "month"]).drop_duplicates(["code", "variable", "month"], keep="last")
    out.to_csv(path, index=False)
    _save_meta(name, {"source": "IBGE SIDRA tables 1419 and 7060, variables 63 (monthly variation) and 66 (monthly weight)"})
    out["month"] = pd.PeriodIndex(out["month"], freq="M")
    return out


ANEEL_ACIONAMENTO = ("https://dadosabertos.aneel.gov.br/dataset/7f43a020-6dc5-44b8-80b4-d97eaa94436c/resource/"
                     "0591b8f6-fe54-437b-b72b-1aa2efd46e42/download/bandeira-tarifaria-acionamento.csv")
ANEEL_ADICIONAL = ("https://dadosabertos.aneel.gov.br/dataset/7f43a020-6dc5-44b8-80b4-d97eaa94436c/resource/"
                   "5879ca80-b3bd-45b1-a135-d9b77c1d5b36/download/bandeira-tarifaria-adicional.csv")


def _read_aneel_csv(content):
    txt = content.decode("utf-8-sig") if content[:3] == b"\xef\xbb\xbf" else content.decode("latin-1")
    return pd.read_csv(io.StringIO(txt), sep=";")


def aneel_flags(refresh=False):
    """Monthly electricity tariff flag in force (ANEEL open data, 'Bandeira Tarifária - Acionamento'): flag name
    and surcharge in R$/MWh for each competence month since 2015. Returns a DataFrame indexed by month with
    columns flag, surcharge (R$/MWh)."""
    name = "aneel_bandeiras"
    path = DATA_RAW / (name + ".csv")
    if path.exists() and not refresh:
        df = pd.read_csv(path)
        df["month"] = pd.PeriodIndex(df["month"], freq="M")
        return df.set_index("month")
    r = _get(ANEEL_ACIONAMENTO, timeout=300)
    d = _read_aneel_csv(r.content)
    out = pd.DataFrame({"month": pd.to_datetime(d["DatCompetencia"].astype(str).str[:10]).dt.to_period("M"),
                        "flag": d["NomBandeiraAcionada"].astype(str).str.strip().str.lower(),
                        "surcharge": pd.to_numeric(d["VlrAdicionalBandeira"].astype(str).str.replace(",", "."), errors="coerce").fillna(0.0)})
    out = out.drop_duplicates("month", keep="last").sort_values("month").set_index("month")
    out.reset_index().to_csv(path, index=False)
    _save_meta(name, {"source": "ANEEL dados abertos, bandeiras tarifárias (acionamento)", "url": ANEEL_ACIONAMENTO})
    return out


def aneel_flag_values(refresh=False):
    """Surcharge of each flag (R$/MWh) by resolution and start of validity ('Bandeira Tarifária - Adicional')."""
    name = "aneel_bandeiras_valores"
    path = DATA_RAW / (name + ".csv")
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["start"])
    r = _get(ANEEL_ADICIONAL, timeout=300)
    d = _read_aneel_csv(r.content)
    out = pd.DataFrame({"start": pd.to_datetime(d["DatVigencia"].astype(str).str[:10]), "flag": d["NomBandeiraAcionada"].astype(str).str.strip().str.lower(),
                        "surcharge": pd.to_numeric(d["VlrAdicionalBandeiraRSMWh"].astype(str).str.replace(",", "."), errors="coerce"),
                        "resolution": d["DscResolucao"].astype(str)})
    out = out.sort_values(["start", "flag"])
    out.to_csv(path, index=False)
    _save_meta(name, {"source": "ANEEL dados abertos, bandeiras tarifárias (adicional)", "url": ANEEL_ADICIONAL})
    return out


def current_flag_values():
    """Latest surcharge (R$/MWh) by flag name: {'verde': 0, 'amarela': x, 'vermelha p1': y, 'vermelha p2': z}."""
    v = aneel_flag_values()
    last = v[v["start"] == v["start"].max()]
    out = {"verde": 0.0}
    out.update({f: float(s) for f, s in zip(last["flag"], last["surcharge"])})
    return out


def bls_cpi(refresh=False):
    """US CPI-U, all items, NSA (BLS series CUUR0000SA0), monthly index, 2005 onwards."""
    name = "bls_cpi_u"
    path = DATA_RAW / (name + ".csv")
    if path.exists() and not refresh:
        s = pd.read_csv(path, index_col=0).iloc[:, 0]
        s.index = pd.PeriodIndex(s.index, freq="M")
        return s
    rows = []
    year = dt.date.today().year
    for y0, y1 in [(2005, 2014), (2015, 2024), (2025, year)]:
        r = _get("https://api.bls.gov/publicAPI/v2/timeseries/data/CUUR0000SA0?startyear=%d&endyear=%d" % (y0, y1), timeout=120)
        for d in r.json()["Results"]["series"][0]["data"]:
            if d["period"].startswith("M") and d["period"] != "M13":
                v = pd.to_numeric(d["value"], errors="coerce")
                if pd.notna(v):
                    rows.append((pd.Period(year=int(d["year"]), month=int(d["period"][1:]), freq="M"), float(v)))
    s = pd.Series(dict(rows)).sort_index()
    s.name = "cpi_us"
    s.to_csv(path, header=True)
    _save_meta(name, {"source": "BLS public API v2, CUUR0000SA0"})
    return s
