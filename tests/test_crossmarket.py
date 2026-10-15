"""Tests for the cross-market information set: timing, no look-ahead, standardisation."""

import numpy as np
import pandas as pd

from tailrisk.crossmarket import overnight_news, session_utc


def _toy_returns():
    dates = pd.bdate_range("2019-01-01", "2020-06-30")
    rng = np.random.default_rng(0)
    return {m: pd.Series(rng.standard_normal(len(dates)), index=dates, name=m)
            for m in ("SPX", "N225", "FTSE", "DAX")}


def test_tokyo_sees_previous_us_session_and_us_sees_same_day_tokyo():
    rets = _toy_returns()
    news = overnight_news(rets, scale_window=50)
    nk, nk_av = news["N225"]
    us, us_av = news["SPX"]
    d = pd.Timestamp("2020-03-17")      # a Tuesday
    prev = pd.Timestamp("2020-03-16")
    # Tokyo on D: New York's session of D-1 has closed before Tokyo opens
    assert nk_av.loc[d, "SPX"] == 1
    s = np.sqrt((rets["SPX"] ** 2).rolling(50, min_periods=20).mean().shift(1))
    assert np.isclose(nk.loc[d, "SPX"], rets["SPX"][prev] / s[prev])
    # New York on D: Tokyo's session of D (same date) has closed; London's D session has not
    assert us_av.loc[d, "N225"] == 1 and us_av.loc[d, "FTSE"] == 0
    # London on D: New York's D-1 session is new information; Tokyo D closes too close to the open
    assert news["FTSE"][1].loc[d, "SPX"] == 1


def test_no_source_session_closes_after_target_open():
    rets = _toy_returns()
    news = overnight_news(rets, scale_window=50)
    for tgt, (nw, av) in news.items():
        open_t, _ = session_utc(tgt, nw.index)
        for src in nw.columns:
            if src == tgt:
                assert av[src].sum() == 0
                continue
            _, close_s = session_utc(src, rets[src].index)
            # for every target day with news from src, the latest src close used is before the open
            used = av[src].to_numpy() == 1
            latest = np.searchsorted(close_s, open_t[used], side="right") - 1
            assert np.all(close_s[latest] < open_t[used])


def test_dst_handled():
    """New York opens at 13:30 UTC in summer and 14:30 UTC in winter."""
    o, _ = session_utc("SPX", pd.DatetimeIndex(["2020-07-01", "2020-01-15"]))
    hours = pd.to_datetime(o, utc=True).hour + pd.to_datetime(o, utc=True).minute / 60
    np.testing.assert_allclose(hours, [13.5, 14.5])
