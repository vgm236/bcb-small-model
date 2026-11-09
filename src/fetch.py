"""
Download layer for all public data used by the model, with local caching and retries.

Sources
-------
* BCB SGS time-series API            https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados
* BCB Focus survey (Olinda OData)    https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/
* BCB Copom meeting history          https://www.bcb.gov.br/api/servico/sitebcb/historicotaxasjuros
* IBGE SIDRA API                     https://apisidra.ibge.gov.br/values/...
* Ipeadata OData                     http://www.ipeadata.gov.br/api/odata4/
* NOAA CPC Oceanic Nino Index        https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
* FRED (CSV, no key)                 https://fred.stlouisfed.org/graph/fredgraph.csv?id=...
* OECD SDMX (quarterly national accounts)

Every function returns a pandas object and caches the raw payload in data/raw as CSV; pass
refresh=True to force a new download.
"""
import io
import json
import time
import datetime as dt
import urllib.parse as up

import pandas as pd
import requests

from config import DATA_RAW

HEADERS = {"User-Agent": "Mozilla/5.0 (research script; pandas/requests)",
           "Accept": "application/json, text/csv, */*"}


def _get(url, tries=6, timeout=120, headers=None, **kw):
    """GET with exponential back-off; BCB endpoints occasionally reset TLS connections."""
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=headers or HEADERS, timeout=timeout, **kw)
            if r.status_code == 200:
                return r
            last = RuntimeError("HTTP %s for %s" % (r.status_code, url[:120]))
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(1.5 * (2 ** k))
    raise RuntimeError("download failed after %d tries: %s :: %s" % (tries, url[:150], last))


def _cache_path(name):
    return DATA_RAW / (name + ".csv")


def _save_meta(name, info):
    meta = DATA_RAW / (name + ".meta.json")
    info = dict(info, downloaded_at=dt.datetime.now().isoformat(timespec="seconds"))
    meta.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# BCB SGS
# ---------------------------------------------------------------------------
def sgs(code, start="1990-01-01", end=None, name=None, daily=False, refresh=False):
    """Return a pd.Series indexed by date for SGS series `code`.

    Daily series are limited to 10-year windows by the API, so they are fetched in 5-year chunks.
    """
    name = name or ("sgs_%s" % code)
    path = _cache_path(name)
    if path.exists() and not refresh:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = name
        return s
    end = end or dt.date.today().strftime("%Y-%m-%d")
    frames = []
    starts = [pd.Timestamp(start)]
    if daily:
        starts = list(pd.date_range(start, end, freq="5YS"))
        if not starts or starts[0] > pd.Timestamp(start):
            starts = [pd.Timestamp(start)] + starts
    url = ""
    for i, s0 in enumerate(starts):
        s1 = (starts[i + 1] - pd.Timedelta(days=1)) if i + 1 < len(starts) else pd.Timestamp(end)
        url = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.%s/dados?formato=json"
               "&dataInicial=%s&dataFinal=%s" % (code, s0.strftime("%d/%m/%Y"), s1.strftime("%d/%m/%Y")))
        r = _get(url)
        if not r.text.startswith("["):
            raise RuntimeError("unexpected SGS payload for %s: %s" % (code, r.text[:100]))
        js = r.json()
        if js:
            frames.append(pd.DataFrame(js))
    df = pd.concat(frames, ignore_index=True)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    s = df.drop_duplicates("data").set_index("data")["valor"].sort_index()
    s.name = name
    s.to_csv(path, header=True)
    _save_meta(name, {"source": "BCB SGS", "code": code, "url": url})
    return s


# ---------------------------------------------------------------------------
# BCB Focus survey (Olinda OData)
# ---------------------------------------------------------------------------
OLINDA = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/"
_SAFE = "'(),=/"


def _odata_all(entity, flt=None, select=None, page=20000):
    """Fetch all rows of an OData entity set using $top/$skip paging (spaces must be %20-encoded)."""
    rows, skip = [], 0
    while True:
        parts = ["$top=%d" % page, "$skip=%d" % skip, "$format=json"]
        if flt:
            parts.append("$filter=" + up.quote(flt, safe=_SAFE))
        if select:
            parts.append("$select=" + up.quote(select, safe=_SAFE))
        url = OLINDA + entity + "?" + "&".join(parts)
        r = _get(url, timeout=300)
        try:
            v = r.json()["value"]
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("Olinda error: %s" % r.text[:200]) from e
        rows.extend(v)
        if len(v) < page:
            break
        skip += page
    return pd.DataFrame(rows)


def focus_inflacao_12m(suavizada="S", indicador="IPCA", refresh=False):
    """Daily Focus statistics for inflation expected over the next 12 months (smoothed: 'S', raw: 'N')."""
    name = "focus_%s_12m_%s" % (indicador.lower(), suavizada)
    path = _cache_path(name)
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["Data"])
    df = _odata_all("ExpectativasMercadoInflacao12Meses",
                    flt="Indicador eq '%s' and Suavizada eq '%s'" % (indicador, suavizada),
                    select="Indicador,Data,Suavizada,Media,Mediana,DesvioPadrao,numeroRespondentes,baseCalculo")
    df["Data"] = pd.to_datetime(df["Data"])
    df = df.sort_values(["Data", "baseCalculo"])
    df.to_csv(path, index=False)
    _save_meta(name, {"source": "BCB Focus (Olinda)", "entity": "ExpectativasMercadoInflacao12Meses"})
    return df


def focus_selic_reunioes(refresh=False):
    """Daily Focus expectations for the Selic target after each Copom meeting (from Nov/2004)."""
    name = "focus_selic_reunioes"
    path = _cache_path(name)
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["Data"])
    df = _odata_all("ExpectativasMercadoSelic",
                    select="Indicador,Data,Reuniao,Media,Mediana,DesvioPadrao,numeroRespondentes,baseCalculo")
    df["Data"] = pd.to_datetime(df["Data"])
    df.to_csv(path, index=False)
    _save_meta(name, {"source": "BCB Focus (Olinda)", "entity": "ExpectativasMercadoSelic"})
    return df


def focus_mensais(indicador="Selic", refresh=False):
    """Daily Focus expectations for monthly reference dates (Selic end-of-month available Nov/2001-Dec/2005)."""
    name = "focus_mensais_%s" % indicador.lower().replace(" ", "_")
    path = _cache_path(name)
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["Data"])
    df = _odata_all("ExpectativaMercadoMensais", flt="Indicador eq '%s'" % indicador,
                    select="Indicador,Data,DataReferencia,Media,Mediana,DesvioPadrao,numeroRespondentes,baseCalculo")
    df["Data"] = pd.to_datetime(df["Data"])
    df.to_csv(path, index=False)
    _save_meta(name, {"source": "BCB Focus (Olinda)", "entity": "ExpectativaMercadoMensais"})
    return df


def focus_anuais(indicador="Selic", refresh=False):
    """Daily Focus expectations for calendar-year reference (e.g. Selic end of year, from 2000)."""
    name = "focus_anuais_%s" % indicador.lower().replace(" ", "_")
    path = _cache_path(name)
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["Data"])
    df = _odata_all("ExpectativasMercadoAnuais", flt="Indicador eq '%s'" % indicador,
                    select="Indicador,Data,DataReferencia,Media,Mediana,DesvioPadrao,numeroRespondentes,baseCalculo")
    df["Data"] = pd.to_datetime(df["Data"])
    df.to_csv(path, index=False)
    _save_meta(name, {"source": "BCB Focus (Olinda)", "entity": "ExpectativasMercadoAnuais"})
    return df


def copom_meetings(refresh=False):
    """History of Copom meetings (date, meeting number, Selic target decided)."""
    name = "copom_meetings"
    path = _cache_path(name)
    if not path.exists() or refresh:
        r = _get("https://www.bcb.gov.br/api/servico/sitebcb/historicotaxasjuros")
        js = r.json()
        df = pd.DataFrame(js.get("conteudo", js))
        df.to_csv(path, index=False)
        _save_meta(name, {"source": "BCB site API historicotaxasjuros"})
    df = pd.read_csv(path)
    df["DataReuniaoCopom"] = pd.to_datetime(df["DataReuniaoCopom"].str[:10])
    df["DataInicioVigencia"] = pd.to_datetime(df["DataInicioVigencia"].str[:10])
    return df.sort_values("DataReuniaoCopom").reset_index(drop=True)


# ---------------------------------------------------------------------------
# IBGE SIDRA
# ---------------------------------------------------------------------------
def sidra(url, name, refresh=False):
    path = _cache_path(name)
    if path.exists() and not refresh:
        return pd.read_csv(path)
    r = _get(url, timeout=180)
    js = r.json()
    df = pd.DataFrame(js[1:])
    df.to_csv(path, index=False)
    _save_meta(name, {"source": "IBGE SIDRA", "url": url})
    return df


# ---------------------------------------------------------------------------
# Ipeadata
# ---------------------------------------------------------------------------
def ipeadata(sercodigo, name=None, refresh=False):
    name = name or ("ipea_%s" % sercodigo)
    path = _cache_path(name)
    if path.exists() and not refresh:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = name
        return s
    url = "http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='%s')" % sercodigo
    r = _get(url, timeout=180)
    df = pd.DataFrame(r.json()["value"])
    df["VALDATA"] = pd.to_datetime(df["VALDATA"].str[:10])
    s = df.set_index("VALDATA")["VALVALOR"].astype(float).sort_index()
    s.name = name
    s.to_csv(path, header=True)
    _save_meta(name, {"source": "Ipeadata", "url": url})
    return s


# ---------------------------------------------------------------------------
# NOAA ONI
# ---------------------------------------------------------------------------
def noaa_oni(refresh=False):
    name = "noaa_oni"
    path = _cache_path(name)
    if path.exists() and not refresh:
        return pd.read_csv(path)
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    r = _get(url, timeout=120)
    df = pd.read_csv(io.StringIO(r.text), sep=r"\s+")
    df.to_csv(path, index=False)
    _save_meta(name, {"source": "NOAA CPC ONI v5", "url": url})
    return df


# ---------------------------------------------------------------------------
# FRED and OECD (external block)
# ---------------------------------------------------------------------------
def fred(series_id, refresh=False):
    name = "fred_%s" % series_id
    path = _cache_path(name)
    if path.exists() and not refresh:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = series_id
        return s
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=%s" % series_id
    r = _get(url, timeout=180)
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    s = pd.to_numeric(df.set_index("date")[series_id], errors="coerce")
    s.name = series_id
    s.to_csv(path, header=True)
    _save_meta(name, {"source": "FRED", "url": url})
    return s


def oecd_qna_gdp(countries, refresh=False):
    """Quarterly real GDP (volume, seasonally adjusted) from the OECD QNA SDMX API (long format)."""
    name = "oecd_qna_gdp"
    path = _cache_path(name)
    if path.exists() and not refresh:
        return pd.read_csv(path)
    geo = "+".join(countries)
    # key has 13 dimensions: FREQ.ADJUSTMENT.REF_AREA.SECTOR.COUNTERPART_SECTOR.TRANSACTION.INSTR_ASSET.
    # ACTIVITY.EXPENDITURE.UNIT_MEASURE.PRICE_BASE.TRANSFORMATION.TABLE_IDENTIFIER
    url = ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,1.1/"
           "Q..%s.S1..B1GQ.......?startPeriod=1995-Q1&format=csvfilewithlabels" % geo)
    r = _get(url, timeout=300, headers={"User-Agent": HEADERS["User-Agent"], "Accept": "text/csv"})
    df = pd.read_csv(io.StringIO(r.text), low_memory=False)
    keep = ["REF_AREA", "Reference area", "ADJUSTMENT", "UNIT_MEASURE", "PRICE_BASE", "TRANSFORMATION",
            "TABLE_IDENTIFIER", "TIME_PERIOD", "OBS_VALUE"]
    df = df[[c for c in keep if c in df.columns]]
    df.to_csv(path, index=False)
    _save_meta(name, {"source": "OECD SDMX QNA", "url": url})
    return df


def oecd_fed_funds(refresh=False):
    """US immediate interest rate (federal funds effective rate, % p.a., monthly) from OECD STES/MEI."""
    name = "oecd_stes_usa_irstci"
    path = _cache_path(name)
    if path.exists() and not refresh:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = "fed_funds"
        return s
    url = ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/"
           "USA.M.IRSTCI.PA.....?startPeriod=1995-01&format=csvfilewithlabels")
    r = _get(url, timeout=300, headers={"User-Agent": HEADERS["User-Agent"], "Accept": "text/csv"})
    df = pd.read_csv(io.StringIO(r.text), low_memory=False)
    df["date"] = pd.to_datetime(df["TIME_PERIOD"])
    s = df.set_index("date")["OBS_VALUE"].astype(float).sort_index()
    s.name = "fed_funds"
    s.to_csv(path, header=True)
    _save_meta(name, {"source": "OECD SDMX STES (MEI financial indicators)", "url": url})
    return s


# ---------------------------------------------------------------------------
# Inputs for the conditional projections (added with the projection module)
# ---------------------------------------------------------------------------
def focus_monthly(indicador="IPCA", since="2024-06-01", refresh=False):
    """Daily Focus medians for monthly reference dates (e.g. IPCA by month), from `since` onwards.

    Used for the short-term (nowcast) part of the projections: the months of the current quarter whose IPCA is
    not yet published are filled with the Focus median of the survey date.
    """
    name = "focus_mensais_%s_%son" % (indicador.lower().replace(" ", "_"), since[:4])
    path = _cache_path(name)
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["Data"])
    df = _odata_all("ExpectativaMercadoMensais", flt="Indicador eq '%s' and Data ge '%s'" % (indicador, since),
                    select="Indicador,Data,DataReferencia,Media,Mediana,DesvioPadrao,numeroRespondentes,baseCalculo")
    df["Data"] = pd.to_datetime(df["Data"])
    df.to_csv(path, index=False)
    _save_meta(name, {"source": "BCB Focus (Olinda)", "entity": "ExpectativaMercadoMensais", "filter": "%s, Data ge %s" % (indicador, since)})
    return df


def eia_brent_daily(refresh=False):
    """Europe Brent spot price FOB (US$/bbl), daily, from the EIA (file RBRTEd.xls)."""
    name = "eia_brent_daily"
    path = _cache_path(name)
    if path.exists() and not refresh:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = "brent"
        return s
    url = "https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls"
    xls = DATA_RAW / "eia" / "RBRTEd.xls"
    xls.parent.mkdir(exist_ok=True)
    r = _get(url, timeout=180)
    xls.write_bytes(r.content)
    x = pd.read_excel(xls, sheet_name="Data 1", skiprows=2)
    x.columns = ["date", "brent"]
    s = x.dropna().set_index("date")["brent"].astype(float).sort_index()
    s.index = pd.to_datetime(s.index)
    s.name = "brent"
    s.to_csv(path, header=True)
    _save_meta(name, {"source": "EIA, Europe Brent Spot Price FOB", "url": url})
    return s


def eia_steo_brent(vintage, refresh=False):
    """Monthly Brent price path (US$/bbl, history + forecast) from an EIA Short-Term Energy Outlook archive.

    `vintage` is e.g. 'sep26' (the STEO of September 2026). Returns (forecast_date, monthly Series). The STEO's
    near-term price path is used as the public proxy for the Brent futures curve assumed by the Copom.
    """
    xls = DATA_RAW / "eia" / "steo" / ("%s_base.xlsx" % vintage)
    if not xls.exists() or refresh:
        xls.parent.mkdir(parents=True, exist_ok=True)
        r = _get("https://www.eia.gov/outlooks/steo/archives/%s_base.xlsx" % vintage, timeout=180)
        xls.write_bytes(r.content)
    x = pd.read_excel(xls, sheet_name="2tab", header=None)
    fdate = pd.to_datetime(str(x.iloc[3, 0]).split(",", 1)[1].strip())
    years, months = x.iloc[2].tolist(), x.iloc[3].tolist()
    row = x[x.iloc[:, 0].astype(str) == "BREPUUS"].iloc[0].tolist()
    out, year = {}, None
    for k in range(2, len(months)):
        if isinstance(years[k], (int, float)) and not pd.isna(years[k]):
            year = int(years[k])
        m = months[k]
        if year is None or not isinstance(m, str):
            continue
        try:
            out[pd.Period("%s %d" % (m, year), freq="M")] = float(row[k])
        except (ValueError, TypeError):
            continue
    return fdate, pd.Series(out).sort_index()


def noaa_nino34_monthly(refresh=False):
    """Monthly Niño 3.4 SST anomaly (ERSSTv5, 1991-2020 base) from NOAA CPC; the ONI is its 3-month mean."""
    name = "noaa_nino_monthly"
    path = DATA_RAW / (name + ".txt")
    if not path.exists() or refresh:
        r = _get("https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii", timeout=120)
        path.write_text(r.text, encoding="utf-8")
        _save_meta(name, {"source": "NOAA CPC ERSSTv5 monthly Nino indices"})
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = ["YR", "MON", "NINO12", "ANOM12", "NINO3", "ANOM3", "NINO4", "ANOM4", "NINO34", "ANOM34"]
    idx = pd.PeriodIndex([pd.Period(year=int(y), month=int(m), freq="M") for y, m in zip(df["YR"], df["MON"])], freq="M")
    return pd.Series(df["ANOM34"].astype(float).values, index=idx, name="nino34_anom")


def noaa_nino34_weekly(refresh=False):
    """Monthly Niño 3.4 SST anomaly from the OISST-based CPC file sstoi.indices (updated within days of month end),
    the most timely public ENSO reading; the ONI (ERSSTv5, 3-month mean) follows with a longer delay."""
    name = "noaa_nino_weekly"
    path = DATA_RAW / (name + ".txt")
    if not path.exists() or refresh:
        r = _get("https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices", timeout=120)
        path.write_text(r.text, encoding="utf-8")
        _save_meta(name, {"source": "NOAA CPC OISST monthly Nino indices (sstoi.indices)"})
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = ["YR", "MON", "NINO12", "ANOM12", "NINO3", "ANOM3", "NINO4", "ANOM4", "NINO34", "ANOM34"]
    idx = pd.PeriodIndex([pd.Period(year=int(y), month=int(m), freq="M") for y, m in zip(df["YR"], df["MON"])], freq="M")
    return pd.Series(df["ANOM34"].astype(float).values, index=idx, name="nino34_anom_oisst")


def noaa_roni(refresh=False):
    """Relative Oceanic Niño Index (NOAA CPC, RONI.ascii.txt): 3-month seasons, columns SEAS YR ANOM. Adopted by the
    BCB in 2026 in place of the ONI (Monetary Policy Report, September 2026)."""
    name = "noaa_roni"
    path = DATA_RAW / (name + ".txt")
    if not path.exists() or refresh:
        r = _get("https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt", timeout=120)
        path.write_text(r.text, encoding="utf-8")
        _save_meta(name, {"source": "NOAA CPC RONI"})
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = ["SEAS", "YR", "ANOM"]
    return df
