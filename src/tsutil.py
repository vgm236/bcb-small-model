"""Time-series helpers: quarterly aggregation, HP filter, seasonal adjustment, splicing."""
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import STL


def to_quarterly_mean(s):
    """Monthly/daily series -> quarterly mean (PeriodIndex 'Q')."""
    s = s.dropna()
    return s.groupby(s.index.to_period("Q")).mean()


def to_quarterly_last(s):
    s = s.dropna()
    return s.groupby(s.index.to_period("Q")).last()


def to_quarterly_sum(s, require_full=True):
    s = s.dropna()
    g = s.groupby(s.index.to_period("Q"))
    out = g.sum()
    if require_full:
        out = out[g.count() == 3]
    return out


def quarterly_log_inflation(monthly_pct):
    """Monthly % changes -> quarterly inflation in log-percent: 100*sum(log(1+x/100)) over complete quarters."""
    lp = 100.0 * np.log1p(monthly_pct.dropna() / 100.0)
    return to_quarterly_sum(lp, require_full=True)


def hp_cycle(s, lamb=1600):
    """HP cycle and trend of a (quarterly) series without NaNs."""
    s = s.dropna()
    cycle, trend = hpfilter(s.values, lamb=lamb)
    return pd.Series(cycle, index=s.index), pd.Series(trend, index=s.index)


def seasonal_adjust(s, method="stl", period=4, robust=True):
    """Seasonally adjust a quarterly series.

    method='stl'     : STL decomposition (LOESS), robust to outliers, seasonal component removed.
    method='dummies' : regression on quarter dummies (constant seasonal pattern).
    """
    s = s.dropna()
    if method == "stl":
        res = STL(s.values, period=period, robust=robust).fit()
        sa = s.values - res.seasonal
        return pd.Series(sa, index=s.index)
    if method == "dummies":
        q = np.asarray([p.quarter for p in s.index])
        X = np.column_stack([(q == k).astype(float) for k in (1, 2, 3, 4)])
        beta = np.linalg.lstsq(X, s.values, rcond=None)[0]
        seas = X @ beta - beta.mean()
        return pd.Series(s.values - seas, index=s.index)
    raise ValueError(method)


def splice_by_differences(new, old):
    """Backcast `new` (level series) with the first differences of `old` where `new` is missing.

    Both are quarterly Series on PeriodIndex. The result equals `new` on its support and extends
    backwards using changes of `old`, so the level is anchored on `new` and the dynamics on `old`.
    """
    out = new.copy()
    idx = sorted(set(new.index) | set(old.index))
    out = out.reindex(idx)
    first = new.first_valid_index()
    pos = idx.index(first)
    for i in range(pos - 1, -1, -1):
        p, pn = idx[i], idx[i + 1]
        if pd.isna(out.loc[p]) and (p in old.index) and (pn in old.index):
            out.loc[p] = out.loc[pn] - (old.loc[pn] - old.loc[p])
    return out


def period_range(a, b):
    return pd.period_range(a, b, freq="Q")
