# %% [markdown]
# # 02 – Construction of the quarterly model dataset
#
# Builds every variable of the BCB aggregate small-scale model (June-2024 specification) from the raw
# downloads, following the definitions in the box (see `docs/model_specification.md`):
#
# | Model variable | Construction |
# |---|---|
# | $\pi^{L,sa}_t$, $\pi^{IPCA,sa}_t$ | quarterly log-inflation of IPCA free prices / headline, seasonally adjusted (STL) |
# | $\pi^e_{t,t+4\|t}$ | quarterly mean of the daily Focus median for IPCA 12 months ahead (smoothed) |
# | $i^e_{t,t+4\|t}$ | Focus-implied expected average Selic over the next 12 months (per-meeting expectations) |
# | $\overline{rr}^{trend}_t$ | HP(1600) trend of the Focus ex-ante real rate $i^e-\pi^e$ |
# | $\hat{\pi}^*_t$ | 100·Δlog(IC-Br, BRL) − target/4 |
# | $\widehat{\Delta e}_t$ | 100·Δlog(PTAX quarterly mean) − (target − 2)/4 |
# | $\widetilde{rp}_t$ | central-government primary surplus (4q, %GDP, net of one-offs, cyclically adjusted) − HP trend |
# | $h^*_t$ | export-weighted HP cycles of partners' real GDP (OECD QNA) |
# | $fpib, fnuci, femp, fcaged$ | HP cycle of log GDP; NUCI − pre-Covid mean; −(u − pre-Covid mean); HP cycle of log Caged stock |
# | $Clima^2$, $d^{el}$, $d^{la}$ | ONI anomaly squared and sign dummies (NOAA) |
# | $i^{dif}_t$ | Selic − (Fed funds + EMBI+) |

# %%
import sys, pathlib, json, warnings
ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import build_dataset as bd
from config import DATA_PROC, OUTPUT

pd.set_option("display.width", 220)
df = bd.assemble(save=True)
print(df.shape, df.index[0], df.index[-1])

# %% [markdown]
# ## Data dictionary and build information

# %%
dd = pd.Series(bd.DATA_DICTIONARY, name="definition")
dd.to_csv(DATA_PROC / "model_data_dictionary.csv", header=True)
info = json.loads((DATA_PROC / "model_data_build_info.json").read_text(encoding="utf-8"))
print("Focus Selic splice (meetings minus monthly, overlap mean):", round(info["splice_diff_focus_selic_meetings_minus_monthly"], 3))
print(dd.to_string())

# %% [markdown]
# ## Consistency checks
# * the expected-Selic series from per-meeting expectations and from monthly expectations should agree in the overlap (2004Q4–2005Q4);
# * the implicit free-price weight should be ~0.70–0.76;
# * the ex-ante real rate should be close to BCB's published "Selic real ex-ante" (e.g. ~ 3.6% neutral in 2021Q4 per the Dec-2021 box).

# %%
daily = pd.read_csv(DATA_PROC / "focus_expected_selic_12m_daily.csv", index_col=0, parse_dates=True)
ov = daily.dropna()
print("overlap days:", len(ov), " mean diff (meetings - monthly): %.3f pp, corr %.3f" % ((ov.iloc[:, 0] - ov.iloc[:, 1]).mean(), ov.corr().iloc[0, 1]))
print(df.loc["2003Q4":"2023Q4", ["w_livres", "r_focus", "rr_trend"]].describe().round(2))

# %% [markdown]
# ## Figures of all model series

# %%
(OUTPUT / "figs").mkdir(exist_ok=True)
panels = [("pi_livres_sa", "Free-price inflation, SA (q, log-%)"), ("pi_ipca_sa", "IPCA inflation, SA (q, log-%)"),
          ("pi_e_12m", "Focus IPCA 12m ahead (% p.a.)"), ("i_e_12m", "Expected Selic next 12m (% p.a.)"),
          ("selic", "Selic target (% p.a.)"), ("r_focus", "Ex-ante real rate (log, % p.a.) and HP trend"),
          ("fpib", "GDP cycle, HP (%)"), ("fnuci", "NUCI deviation (pp)"), ("femp", "-(unemployment deviation) (pp)"),
          ("fcaged", "Caged stock cycle, HP (%)"), ("rp_tilde", "Fiscal: CAB deviation from trend (% GDP)"),
          ("h_star", "World gap (%)"), ("de_hat", "Exchange-rate change minus PPP drift (%)"),
          ("pi_icbr_hat", "IC-Br inflation minus target (q, %)"), ("i_dif", "Interest differential (log, pp)"),
          ("clim_el", "Climate term El Nino / La Nina")]
fig, axes = plt.subplots(8, 2, figsize=(14, 22))
x = df.index.to_timestamp()
for ax, (col, title) in zip(axes.ravel(), panels):
    ax.plot(x, df[col], lw=1.2)
    if col == "r_focus":
        ax.plot(x, df["rr_trend"], lw=1.2, ls="--")
    if col == "clim_el":
        ax.plot(x, df["clim_la"], lw=1.2, ls="--")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvspan(pd.Timestamp("2003-10-01"), pd.Timestamp("2023-12-31"), color="grey", alpha=0.08)
    ax.set_title(title, fontsize=10)
fig.tight_layout()
fig.savefig(OUTPUT / "figs" / "02_model_series.png", dpi=110)
print("saved", OUTPUT / "figs" / "02_model_series.png")

# %%
world = pd.read_csv(DATA_PROC / "world_gdp_cycles.csv", index_col=0)
world.index = pd.PeriodIndex(world.index, freq="Q")
fig, ax = plt.subplots(figsize=(10, 4))
for c in ["USA", "CHN", "ARG", "DEU", "h_star"]:
    if c in world:
        ax.plot(world.index.to_timestamp(), world[c], lw=2 if c == "h_star" else 0.8, label=c)
ax.legend(); ax.set_title("HP cycles of partners' real GDP and the trade-weighted world gap (%)")
fig.tight_layout(); fig.savefig(OUTPUT / "figs" / "02_world_gap.png", dpi=110)
print(df.loc["2003Q4":"2023Q4"].describe().T.round(3).to_string())
