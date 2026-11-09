# %% [markdown]
# # 01 – Download of all raw data
#
# This script (VS Code "interactive" cells, `# %%`) downloads every public series used to build the
# BCB small-scale aggregate model dataset and caches them under `data/raw/`.
# All sources are public and key-free: BCB SGS, BCB Focus survey (Olinda), BCB Copom history,
# IBGE SIDRA, Ipeadata (EMBI+), NOAA (ONI) and OECD (quarterly GDP of trading partners).
#
# Re-running the script uses the cache; set `REFRESH = True` to force new downloads.

# %%
import sys, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import fetch
from config import SGS, SIDRA_PIB_SA, SIDRA_PIB_NSA, SIDRA_PIB_NOMINAL, SIDRA_DESOCUP_PNADC, IPEA_EMBI

REFRESH = False
t0 = time.time()

# %% [markdown]
# ## BCB SGS series
# Monthly series are fetched in one call; the daily Selic target is fetched in 5-year chunks.

# %%
raw = {}
for key, code in SGS.items():
    daily = key in ("selic_meta_daily",)
    s = fetch.sgs(code, start="1995-01-01", name="sgs_%d_%s" % (code, key), daily=daily, refresh=REFRESH)
    raw[key] = s
    print("%-24s code=%-6d n=%5d  %s .. %s" % (key, code, len(s), s.index[0].date(), s.index[-1].date()))

# %% [markdown]
# ## IBGE SIDRA: quarterly GDP (volume index SA and NSA, current prices)

# %%
pib_sa = fetch.sidra(SIDRA_PIB_SA, "sidra_1621_pib_sa", refresh=REFRESH)
pib_nsa = fetch.sidra(SIDRA_PIB_NSA, "sidra_1620_pib_nsa", refresh=REFRESH)
pib_nom = fetch.sidra(SIDRA_PIB_NOMINAL, "sidra_1846_pib_nominal", refresh=REFRESH)
pnadc = fetch.sidra(SIDRA_DESOCUP_PNADC, "sidra_6381_desocupacao", refresh=REFRESH)
for name, d in [("PIB SA", pib_sa), ("PIB NSA", pib_nsa), ("PIB nominal", pib_nom), ("PNADC", pnadc)]:
    print(name, len(d), d.iloc[0]["D3N"] if "D3N" in d else "", "..", d.iloc[-1]["D3N"] if "D3N" in d else "")

# %% [markdown]
# ## Focus survey (Olinda OData)
# * 12-month-ahead IPCA expectations, smoothed ("suavizada") and raw.
# * Selic expectations per Copom meeting (from Nov/2004), monthly end-of-month Selic (2001-2005) and end-of-year Selic.

# %%
f12s = fetch.focus_inflacao_12m("S", refresh=REFRESH)
f12n = fetch.focus_inflacao_12m("N", refresh=REFRESH)
print("Focus 12m S:", len(f12s), f12s["Data"].min().date(), f12s["Data"].max().date())
print("Focus 12m N:", len(f12n), f12n["Data"].min().date(), f12n["Data"].max().date())
fsel = fetch.focus_selic_reunioes(refresh=REFRESH)
print("Focus Selic meetings:", len(fsel), fsel["Data"].min().date(), fsel["Data"].max().date())
fmen = fetch.focus_mensais("Selic", refresh=REFRESH)
print("Focus Selic monthly:", len(fmen), fmen["Data"].min().date(), fmen["Data"].max().date())
fanu = fetch.focus_anuais("Selic", refresh=REFRESH)
print("Focus Selic annual:", len(fanu), fanu["Data"].min().date(), fanu["Data"].max().date())
fipca_anu = fetch.focus_anuais("IPCA", refresh=REFRESH)
print("Focus IPCA annual:", len(fipca_anu))
cop = fetch.copom_meetings(refresh=REFRESH)
print("Copom meetings:", len(cop), cop["DataReuniaoCopom"].min().date(), cop["DataReuniaoCopom"].max().date())

# %% [markdown]
# ## Risk premium (EMBI+ Brazil, Ipeadata) and climate (ONI, NOAA)

# %%
embi = fetch.ipeadata(IPEA_EMBI, refresh=REFRESH)
print("EMBI+:", len(embi), embi.index[0].date(), embi.index[-1].date())
oni = fetch.noaa_oni(refresh=REFRESH)
print("ONI:", len(oni), oni.iloc[0].tolist(), oni.iloc[-1].tolist())

# %% [markdown]
# ## External block: quarterly real GDP of trading partners (OECD QNA)

# %%
PARTNERS = ["USA", "CHN", "ARG", "JPN", "GBR", "DEU", "CAN", "MEX", "CHL", "KOR", "IND", "NLD", "ESP", "ITA", "FRA", "BEL"]
oecd = fetch.oecd_qna_gdp(PARTNERS, refresh=REFRESH)
print("OECD QNA rows:", len(oecd))
ff = fetch.oecd_fed_funds(refresh=REFRESH)
print("Fed funds (OECD MEI):", len(ff), ff.index[0].date(), ff.index[-1].date())
print("done in %.0fs" % (time.time() - t0))


# %% [markdown]
# ## Inputs of the projections, the Copom backtest and the administered-price model
# Daily PTAX, Focus annual medians (Selic, IPCA, administered and free prices, GDP, exchange rate) and monthly IPCA
# medians, Brent (EIA spot and the Short-Term Energy Outlook archives used as the futures-curve proxy), NOAA Niño
# indices (ONI, RONI, monthly OISST), IBGE subitems of the administered items, ANEEL tariff flags, US CPI (BLS).

# %%
import fetch_adm
from copom_meetings import STEO_VINTAGES
fetch.sgs(1, start="2023-01-01", name="sgs_1_ptax_venda_daily", daily=True, refresh=REFRESH)
for ind in ["Selic", "IPCA", "IPCA Administrados", "IPCA Livres", "PIB Total", "Câmbio"]:
    fetch.focus_anuais(ind, refresh=REFRESH)
fetch.focus_monthly("IPCA", refresh=REFRESH)
fetch.eia_brent_daily(refresh=REFRESH)
for v in STEO_VINTAGES:
    try:
        fetch.eia_steo_brent(v, refresh=REFRESH)
    except Exception as e:  # noqa: BLE001
        print("STEO", v, "not available:", e)
fetch.noaa_nino34_weekly(refresh=REFRESH)
fetch.noaa_roni(refresh=REFRESH)
fetch_adm.sidra_subitems(refresh=REFRESH)
fetch_adm.aneel_flags(refresh=REFRESH)
fetch_adm.aneel_flag_values(refresh=REFRESH)
fetch_adm.bls_cpi(refresh=REFRESH)
print("projection inputs downloaded")
