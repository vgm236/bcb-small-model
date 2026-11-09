"""
Project configuration: paths, data-series identifiers and sample settings.

Model reference: Banco Central do Brasil (BCB), Inflation Report, June 2024, box
"Atualização dos modelos semiestruturais de pequeno porte" (aggregate version),
which the 2025-2026 Monetary Policy Reports still cite as the operative small-scale model.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUTPUT = ROOT / "output"
REFS = ROOT / "refs"
for _p in (DATA_RAW, DATA_PROC, OUTPUT):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# BCB SGS (Sistema Gerenciador de Séries Temporais) codes
# ---------------------------------------------------------------------------
SGS = {
    # prices (monthly % change)
    "ipca": 433,              # IPCA - monthly % change
    "ipca_livres": 11428,     # IPCA - free prices (itens livres) - monthly %
    "ipca_monit": 4449,       # IPCA - administered/monitored prices - monthly %
    "ipca_12m": 13522,        # IPCA accumulated 12 months (%)
    "meta_inflacao": 13521,   # annual inflation target (% p.a.), one obs per year
    # policy rate
    "selic_meta_daily": 432,  # Selic target (% p.a.), daily
    "selic_efetiva_aa": 4189, # Selic effective, monthly, annualised (% p.a.)
    # exchange rate (BRL/USD, PTAX sell, monthly average)
    "ptax_venda_mensal": 3698,
    # commodities (IC-Br, BRL, index)
    "icbr": 27574, "icbr_agro": 27575, "icbr_metal": 27576, "icbr_energia": 27577,
    # activity / factor utilisation
    "pib_sa_idx": 22109,      # quarterly GDP, chained volume index, seasonally adjusted (IBGE) - mirror of SIDRA 1621
    "nuci_fgv_sa": 28561,     # FGV capacity utilisation, manufacturing, seasonally adjusted (monthly, from 2001)
    "nuci_fgv_nsa": 24352,    # FGV capacity utilisation, not seasonally adjusted (monthly)
    "desocup_pnadc": 24369,   # PNAD Contínua unemployment rate (monthly, rolling quarter, from Mar/2012)
    "desocup_pme": 10777,     # PME unemployment rate (Nov/2001 - Feb/2016), metropolitan areas
    "caged_estoque": 28763,   # stock of formal jobs (Caged / Novo Caged), monthly, from 1992
    # fiscal (NFSP, "below the line"; sign convention: + = deficit (financing need))
    "prim_gov_central_rs": 4639,  # primary result, central government (Federal Gov. + BCB), monthly flow, BRL million
    "prim_gov_central_pib12": 5783, # same, 12-month flow, % GDP
    "prim_consolidado_rs": 4649,    # primary result, consolidated public sector, monthly flow, BRL million
    "pib_nominal_mensal": 4380,     # monthly nominal GDP (BCB estimate), BRL million
}

# IBGE SIDRA
SIDRA_PIB_SA = "https://apisidra.ibge.gov.br/values/t/1621/n1/all/v/584/p/all/c11255/90707"   # quarterly GDP volume index, SA (1995=100)
SIDRA_PIB_NSA = "https://apisidra.ibge.gov.br/values/t/1620/n1/all/v/583/p/all/c11255/90707"  # quarterly GDP volume index, NSA
SIDRA_PIB_NOMINAL = "https://apisidra.ibge.gov.br/values/t/1846/n1/all/v/585/p/all/c11255/90707"  # quarterly GDP, current prices, BRL million
SIDRA_DESOCUP_PNADC = "https://apisidra.ibge.gov.br/values/t/6381/n1/all/v/4099/p/all"        # PNADC unemployment, monthly (rolling quarter)

# Ipeadata
IPEA_EMBI = "JPM366_EMBI366"   # JP Morgan EMBI+ Brazil spread (bps), daily

# NOAA Oceanic Niño Index
NOAA_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# ---------------------------------------------------------------------------
# Sample and calibration settings (BCB June-2024 box: 2003Q4-2023Q4)
# ---------------------------------------------------------------------------
SAMPLE_START = "2003Q4"
SAMPLE_END = "2023Q4"      # replication sample; data are downloaded through the latest available quarter
PI_STAR_SS = 2.0           # long-run foreign inflation (% p.a.) used in the PPP term of the UIP equation
HP_LAMBDA = 1600           # HP filter smoothing parameter for quarterly data
PRE_COVID_END = "2019Q4"   # NUCI and unemployment cyclical components: deviation from mean through 2019Q4
CRISIS_2008 = ("2008Q4", "2009Q4")
CRISIS_2020 = ("2020Q1", "2020Q4")
PHILLIPS_HIGH_VAR = ("2020Q2", "2022Q4")

# Calibrated quantities not reported in the box (see docs/model_specification.md, section 4, and the sweeps in
# notebooks/03b, 03c): standard-deviation multiples of the crisis shocks in the IS curve and of the Phillips-curve
# shock in 2020Q2-2022Q4, and the s.d. of the random-walk components of the neutral/equilibrium real rates.
CRISIS_SD_MULT = 20.0
HV_SD_MULT = 3.0
SD_RR_RW = 0.15
