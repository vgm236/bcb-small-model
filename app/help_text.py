"""Text of the 'How it works' tab of the desktop application, and the tab itself."""
import tkinter as tk
from tkinter import ttk

HELP_TEXT = """
HOW THIS APPLICATION WORKS

1. What the model is
The engine is a from-scratch reconstruction of the Banco Central do Brasil's aggregate small-scale semi-structural
model (Inflation Report, June 2024 box): a Phillips curve for free prices, an IS curve with an unobserved output gap
and neutral rate, a Taylor rule, uncovered interest parity, a hybrid expectations equation and four activity
indicators that identify the gap. It is estimated by Bayesian methods on 2003Q4-2023Q4 with public data (BCB, IBGE,
Focus, OECD, EIA, NOAA); its coefficients are compared with the BCB's published table in docs/results.md. The
administered-price block is the BCB's medium-term model of June 2025 (24 item equations), re-estimated at every run.
Projections are made the way the Copom builds its reference scenario, and the whole thing is checked against the
projections the Copom published at each meeting since July 2025.

2. Tab "Next meeting"
The application finds the next scheduled Copom meeting (calendar built in; completed by the BCB's history), the
latest Focus survey, the exchange-rate start value (10-business-day PTAX average ending on the Friday before the
meeting week, rounded to R$0.05, the Copom's convention) and the Focus medians for administered prices. Press
"Run projection". The current quarter is nowcast with the Focus monthly medians for the headline IPCA and with the
administered figure; the quarters ahead follow the Focus Selic path, the PPP exchange-rate path, the Brent path
(spot plus the shape of the latest EIA outlook for six months, then +2% a year), the administered assumption, the
climate term and the target. The Kalman filter gives the state at the origin (output gap, expectations, neutral
rate); the paths are imposed through anticipated shocks so that expectations are consistent with them.
Two outputs are shown: the unadjusted model output, and the output adjusted by the model's error (model minus
Copom) at the previous meeting, by type of target (current year, next year, relevant horizon). The relevant
horizon is the quarter six quarters after the meeting's quarter, as the Copom defines it. The bands come from
stochastic simulations of the model's shocks around the central path, with the Selic, exchange-rate and
administered paths kept fixed; they are the model's own uncertainty, not a forecast-error history.
Model settings: the neutral real rate (5% is the Copom's conditioning value); the climate assumption (real-time
Nino 3.4 readings with an AR(1) decay, no effect, or the realized values); the output-gap anchor on the BCB's
published estimates (the single most important choice for matching the Copom; see point 7); the coefficients
(ours or the BCB's published modes); the source of administered prices (annual figures, or the BCB's administered
model with the annual figures kept for four quarters, the specialists' horizon, before the model takes over); the
December tariff flag; the number of simulation draws; the weight of the previous-meeting error in the adjusted
output.

2b. Tab "Output gap"
The output gap is the unobserved state of the model, identified by four indicators (the HP cycle of GDP, capacity
utilisation, unemployment and formal jobs, each divided by its estimated loading) through the IS curve. The chart
shows the smoothed estimate (using the whole sample), the filtered estimate (using only past data, which is what
a real-time user would have had), the indicator-implied gaps, the estimates the BCB publishes in the Monetary
Policy Report (red squares; the projection is anchored on them when the option is on) and the projection ahead
under the scenario's real-rate path with 50% and 90% bands from the model's shocks. The table gives the numbers
for the last eight quarters and the projection quarters; "Save gap table" writes them to a CSV file. The tab is
filled after each run of the "Next meeting" tab (or press Run on the tab itself). Watch: the labour-market
indicator (unemployment far below its 2003-2019 mean) pulls the model's own estimate up; the smoothed and the
filtered estimates can differ by several tenths at the end of the sample; the projection depends on the neutral
rate assumed (5%).

3. Tab "Scenario builder"
Press "Load reference scenario" first: the reference paths of the Selic and the exchange rate appear, quarter by
quarter. Then type your own values where you want to deviate: a Selic level (% p.a., quarterly average) applies
from the day after the meeting; a BRL/USD level applies from the Focus date; blank cells keep the reference
value and the last edited value is carried forward. Administered inflation by year, the exchange-rate start and
the 12-month inflation expectation can be changed too.
Shocks to inflation expectations are relative to the Focus survey: the "origin" shock shifts the 12-month
expectation the model starts from (the expectations equation then drives it back over time); the quarterly shocks
keep the model's expectations that many percentage points above or below their own path in those quarters (they
are not carried forward). Internally they are imposed as anticipated expectation shocks, so the model-consistent
component of expectations, the Phillips curve and the IS curve all respond. The result is shown next to the
reference scenario, unadjusted and adjusted.

4. Tab "Administered prices"
The BCB's administered-price model is calibrated by the BCB from institutional rules (fuel price composition,
ICMS, electricity tariffs with the Itaipu share and the tariff flag, the ANS rule for health plans, the CMED rule
for pharmaceuticals, indexation to past inflation with seasonal weights for the rest). Every equation is also
estimated by OLS on the same regressors; the table shows both sets with the distance in standard errors. The
re-estimation and the data download happen when you press the buttons. The ANS and CMED readjustments of the
current year are read from the IPCA items as soon as they appear; later years follow the rules of the box.

5. Tab "Backtest"
For every meeting since July 2025 the model is run with the information available at the time (publication lags
of each series are respected; data revisions cannot be undone) and the Copom's own assumptions, and compared with
Table 1 of the minutes. "Update Copom history and re-run the backtest" downloads any new minutes, parses the
projections and the assumptions, re-runs all meetings and stores the results; the adjusted output of the next
projection then uses the latest meeting's error automatically. Re-run it after each meeting's minutes are out
(the Tuesday after the meeting).

6. Tab "Data and updates"
"Refresh the market and macro inputs" downloads the Focus survey, the Selic, the PTAX, the IPCA and its items,
activity data, Brent, the tariff flags and the climate indices again. Do it before running a projection for a
new meeting. The table shows the last observation and the download time of each input.

7. Why it is done this way, and what to watch
- Conventions are the Copom's: the Focus Selic path, the PTAX start and PPP drift, the six-month oil curve, the
  stated December flag, the 5% neutral rate, the relevant horizon six quarters ahead. Each was verified against
  the minutes (the start value is reproduced at all ten meetings).
- The current quarter uses the Focus monthly medians and the annual administered figures because the BCB uses
  specialists' bottom-up projections for the short term; a quarterly model is not a nowcasting tool.
- The output gap is anchored on the BCB's published estimates because the diagnostics showed it to be the main
  source of differences from the Copom: the labour-market indicator (unemployment far below its 2003-2019 mean)
  pushes the model's own gap about half a point above the BCB's in 2025-2026.
- The HP trend of the ex-ante real rate, which the model uses as the neutral-rate trend in the estimation sample,
  reaches 8-9% in 2025-2026 because the policy stance was restrictive; projections use the Copom's 5% instead.
- The climate term is large in 2026-2027 (a strong El Nino). A real-time rule cannot reproduce NOAA's outlook,
  which the BCB uses; the climate options show how much depends on it.
- The administered model is a medium-term tool: started from the next quarter it departs from the Copom by one to
  two points (fuel pass-through with the Petrobras pricing policy); after the specialists' four quarters it is
  within 0.2-0.3 points. The annual figures remain the default input.
- The previous-meeting adjustment removes the average bias and helps at the relevant horizon, but it adds noise
  when the sign of the error changes between meetings; a weight of 0.25-0.5 minimises the mean absolute
  difference overall, the full weight at the horizon.
- The model's coefficients were estimated once (2003Q4-2023Q4); only the administered model is re-estimated at
  each run. The risk-premium series (EMBI+) ends in July 2024; the exchange-rate path is imposed, so this does not
  affect the projections. The Brent futures curve is proxied by the EIA outlook's shape. Data revisions of GDP,
  Caged and PNAD are not undone in the backtest.
- Errors during a computation are written to app/desktop_app.log. The window stays responsive while the model
  runs (10-20 seconds per projection, more with many simulation draws).

Documentation: README.md, docs/model_specification.md, docs/discussion.md, docs/results.md (estimation),
docs/projection.md and docs/results_backtest.md (projections and backtest), docs/administered.md and
docs/results_administered.md (administered prices).
"""


def reflow(text):
    """Join the hard-wrapped lines of each paragraph (bullets stay on their own line) so the Text widget wraps
    them to the window width."""
    out, para = [], []

    def flush():
        if para:
            out.append(" ".join(para)); para.clear()
    for line in text.splitlines():
        s = line.strip()
        if not s:
            flush(); out.append("")
        elif s.startswith("- ") or s[:1].isdigit() and ". " in s[:4] or s.isupper() or s.startswith("Documentation:"):
            flush(); para.append(s)
            if s.isupper() or (s[:1].isdigit() and ". " in s[:4]):
                flush()
        else:
            para.append(s)
    flush()
    return "\n".join(out)


class HelpTab(ttk.Frame):
    def __init__(self, master, app=None):
        super().__init__(master)
        txt = tk.Text(self, wrap="word", font=("Segoe UI", 10), padx=12, pady=10)
        vsb = ttk.Scrollbar(self, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        txt.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        body = reflow(HELP_TEXT.strip()) + "\n"
        txt.insert("1.0", body)
        txt.tag_configure("h", font=("Segoe UI", 11, "bold"))
        for line_no, line in enumerate(body.splitlines(), start=1):
            head = line.split(" ")[0].rstrip(".")
            if (head.isdigit() and line[len(head) + 1:len(head) + 2] == " ") or (line.isupper() and line.strip()):
                txt.tag_add("h", "%d.0" % line_no, "%d.end" % line_no)
        txt.configure(state="disabled")
