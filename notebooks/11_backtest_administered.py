# %% [markdown]
# # 11 – Backtest of the administered-price model at the Copom meetings
#
# The administered-price model (administered.py, the BCB's June-2025 box) replaces the Copom's administered
# figures as the input of the projections, with the Copom's tariff-flag assumption of each meeting, the reference
# scenario's oil and exchange-rate paths and the feedback with free prices. Its calendar-year and relevant-horizon
# projections of administered inflation are compared with the Copom's published numbers, and the resulting IPCA
# projections with the Copom's IPCA, for the BCB's calibration and for the parameters re-estimated at each meeting.

# %%
import sys, pathlib, warnings, time, json
ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import projection as pj, copom_meetings as cm, adjust
from config import OUTPUT

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
model = pj.load_model()
meetings = cm.all_meetings()
VARIANTS = {"input: Copom administered": {"adm_source": "input"},
            "model: BCB calibration": {"adm_source": "model", "adm_theta": "bcb"},
            "model: estimated parameters": {"adm_source": "model", "adm_theta": "estimated"},
            "model: BCB calibration, handover after 4 quarters": {"adm_source": "model", "adm_theta": "bcb", "adm_handover": 4},
            "model: estimated, handover after 4 quarters": {"adm_source": "model", "adm_theta": "estimated", "adm_handover": 4}}
rows, items = [], []
for name, kw in VARIANTS.items():
    t0 = time.time()
    for m in meetings:
        flag = {y: ("verde" if "verde" in str(m.get("flag", "")) else "amarela") for y in range(int(m["decision"][:4]), int(m["decision"][:4]) + 3)}
        # the annual administered figures used for the current quarter's nowcast (and the handover quarters) are
        # the Copom's own, so that the model's contribution starts where the Copom's specialists' would
        adm_years = {y: v[2] for y, v in m["years"].items()}
        adm_h = (m["horizon"][0], m["horizon"][3])
        if kw["adm_source"] == "input":
            out = pj.project(m["decision"], m["focus"], m["fx_start"], adm_years, adm_h, H=12, model=model, **dict(pj.ADOPTED, **kw))
        else:
            out = pj.project(m["decision"], m["focus"], m["fx_start"], adm_years, adm_h, H=12, model=model, flag_december=flag, **dict(pj.ADOPTED, **kw))
        for r in adjust.target_rows(m, out):
            r["variant"] = name; r["meeting"] = m["n"]; rows.append(r)
        if out.get("adm_model") is not None:
            qd = out["adm_model"]["quarterly"]
            for q, r in qd.iterrows():
                items.append(dict(variant=name, meeting=m["n"], quarter=str(q), **{c: r[c] for c in qd.columns}))
    print("variant '%s' done (%.0fs)" % (name, time.time() - t0))
tab = pd.DataFrame(rows)
for c in ["ipca", "livres", "adm"]:
    tab["diff_" + c] = tab["model_" + c] - tab["copom_" + c]
tab.to_csv(OUTPUT / "11_backtest_administered.csv", index=False)
pd.DataFrame(items).to_csv(OUTPUT / "11_backtest_administered_items.csv", index=False)
summ = tab.groupby(["variant", "type"]).agg(mean_diff_adm=("diff_adm", "mean"), mae_adm=("diff_adm", lambda x: x.abs().mean()),
                                            mean_diff_ipca=("diff_ipca", "mean"), mae_ipca=("diff_ipca", lambda x: x.abs().mean()),
                                            mean_diff_livres=("diff_livres", "mean"), mae_livres=("diff_livres", lambda x: x.abs().mean())).round(2)
print(summ.to_string())
summ.to_csv(OUTPUT / "11_backtest_administered_summary.csv")
piv = tab.pivot_table(index=["meeting", "target"], columns="variant", values="model_adm").round(2)
piv["Copom"] = tab.groupby(["meeting", "target"])["copom_adm"].first()
print(piv.to_string())
piv.to_csv(OUTPUT / "11_backtest_administered_by_meeting.csv")
