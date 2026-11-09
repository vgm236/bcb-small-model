"""
Error adjustment of the projections with the model's error at the previous Copom meeting.

For a meeting k the adjusted projection is the unadjusted projection minus the error (model minus Copom) observed
at meeting k-1 for the same type of target: current calendar year, next calendar year and the relevant horizon
(four quarters). When the previous meeting did not publish a given type (the next-year figure appears only from
the August meeting onwards), the horizon error is used in its place. The quarterly four-quarter path is adjusted
with the same errors, interpolated linearly between the target quarters (year-ends and the horizon quarter) and
held constant afterwards. The factor `damping` scales the correction (1 = full previous error, the default).
"""
import json
import numpy as np
import pandas as pd

from config import OUTPUT

CACHE = OUTPUT / "backtest_cache"
TYPES = ["current year", "next year", "relevant horizon"]


def target_rows(meeting, out):
    """Rows (target, type, Copom, model) for a meeting's published targets, from a projection output."""
    rows = []
    year0 = int(meeting["decision"][:4])
    for y, (ipca, liv, adm) in meeting["years"].items():
        o = out["years"][y]
        rows.append({"target": str(y), "type": "current year" if y == year0 else "next year", "copom_ipca": ipca, "model_ipca": o["pi"],
                     "copom_livres": liv, "model_livres": o["piL"], "copom_adm": adm, "model_adm": o["piA"]})
    hq, ipca, liv, adm = meeting["horizon"]
    t = out["table"]
    hp = pd.Period(hq, freq="Q")
    rows.append({"target": hq, "type": "relevant horizon", "copom_ipca": ipca, "model_ipca": float(t.loc[hp, "pi_4q"]),
                 "copom_livres": liv, "model_livres": float(t.loc[hp, "piL_4q"]), "copom_adm": adm, "model_adm": float(t.loc[hp, "piA_4q"])})
    return rows


def save_meeting_result(meeting, out, settings=None):
    """Store the comparison of a backtest meeting (used later as 'previous meeting error')."""
    CACHE.mkdir(exist_ok=True)
    rec = {"meeting": meeting["n"], "decision": meeting["decision"], "focus": meeting["focus"], "settings": settings or out.get("settings", {}),
           "targets": target_rows(meeting, out),
           "path_4q": {str(q): float(v) for q, v in out["table"]["pi_4q"].dropna().items()},
           "path_4q_livres": {str(q): float(v) for q, v in out["table"]["piL_4q"].dropna().items()}}
    (CACHE / ("%d.json" % meeting["n"])).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def load_meeting_result(n):
    p = CACHE / ("%d.json" % n)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def errors_of(rec):
    """{type: {'ipca': model-Copom, 'livres': ...}} of a stored meeting result."""
    e = {}
    for r in rec["targets"]:
        e[r["type"]] = {"ipca": r["model_ipca"] - r["copom_ipca"], "livres": r["model_livres"] - r["copom_livres"], "target": r["target"]}
    return e


def previous_errors(n, meetings_sorted):
    """Errors of the latest meeting before n with a stored result (normally the previous meeting)."""
    prev = [m for m in meetings_sorted if m["n"] < n]
    for m in reversed(prev):
        rec = load_meeting_result(m["n"])
        if rec is not None:
            return errors_of(rec), m["n"]
    return None, None


def error_for_type(errors, typ, var="ipca"):
    if errors is None:
        return 0.0
    if typ in errors:
        return float(errors[typ][var])
    if "relevant horizon" in errors:        # fallback when the previous meeting did not publish this type
        return float(errors["relevant horizon"][var])
    return 0.0


def adjust_targets(rows, errors, damping=1.0):
    """Adjusted model numbers for target rows (list of dicts as in target_rows)."""
    out = []
    for r in rows:
        r2 = dict(r)
        for var in ("ipca", "livres"):
            r2["adjusted_" + var] = r["model_" + var] - damping * error_for_type(errors, r["type"], var)
        r2["adjusted_adm"] = r["model_adm"]
        out.append(r2)
    return out


def adjust_path(table, q0, errors, damping=1.0, horizon_q=None, var="pi_4q"):
    """Adjusted four-quarter path: the error is interpolated between the target quarters (end of the current year:
    current-year error; end of the next year: next-year error; horizon quarter: horizon error), zero at the current
    quarter, and constant after the last target quarter."""
    if errors is None:
        return table[var].copy()
    pts = {q0: 0.0}
    y0 = q0.year
    pts[pd.Period(year=y0, quarter=4, freq="Q")] = error_for_type(errors, "current year", "ipca" if var.startswith("pi") else "livres")
    if "next year" in errors or "relevant horizon" in errors:
        pts[pd.Period(year=y0 + 1, quarter=4, freq="Q")] = error_for_type(errors, "next year", "ipca" if var.startswith("pi") else "livres")
    hq = pd.Period(horizon_q, freq="Q") if horizon_q is not None else q0 + 6
    pts[hq] = error_for_type(errors, "relevant horizon", "ipca" if var.startswith("pi") else "livres")
    pts = pd.Series(pts).sort_index()
    pts = pts[~pts.index.duplicated(keep="last")]
    idx = table.index
    x = np.array([(q - q0).n for q in idx], dtype=float)
    xp = np.array([(q - q0).n for q in pts.index], dtype=float)
    corr = np.interp(x, xp, pts.values)
    corr[x <= 0] = 0.0
    adj = table[var] - damping * pd.Series(corr, index=idx)
    return adj


def evaluate_damping(records, dampings=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Mean absolute difference from the Copom of the adjusted projections, by damping factor and target type, using
    each meeting's previous stored result."""
    recs = sorted(records, key=lambda r: r["meeting"])
    rows = []
    for k in range(1, len(recs)):
        prev, cur = recs[k - 1], recs[k]
        e = errors_of(prev)
        for r in cur["targets"]:
            for d in dampings:
                rows.append({"meeting": cur["meeting"], "type": r["type"], "damping": d,
                             "abs_diff": abs(r["model_ipca"] - d * error_for_type(e, r["type"], "ipca") - r["copom_ipca"]),
                             "abs_diff_livres": abs(r["model_livres"] - d * error_for_type(e, r["type"], "livres") - r["copom_livres"])})
    df = pd.DataFrame(rows)
    by_type = df.groupby(["damping", "type"])[["abs_diff", "abs_diff_livres"]].mean().unstack("type")
    overall = df.groupby("damping")[["abs_diff", "abs_diff_livres"]].mean()
    return by_type.round(3), overall.round(3), df
