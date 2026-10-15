r"""Cross-market ("overnight") information for pre-open tail-risk forecasts.

A risk manager in market i sets the VaR for trading day D just before i opens. By then, other
markets j may already have closed sessions that i has not yet reflected in its own close-to-close
return (e.g. Tokyo sees New York's previous session; New York sees Tokyo's same-day session).
For target i and day D we collect, for every other market j, the sum of j's log returns over all
sessions whose close lies in

        ( C_i(D_prev),  O_i(D) - 2 * margin ]

where C_i(D_prev) is i's previous close and O_i(D) its open. To rule out look-ahead despite
historical schedule changes, close times are the *latest* plausible times (closing auctions
included) and open times the *earliest* plausible times (pre-open auctions included), and a further
safety margin is applied on each side. Daylight saving is handled per exchange via zoneinfo.

News values are standardised by the source market's own slow scale (250-day RMS up to the
previous session), so that x_{j} is "the size of j's overnight move in units of its usual
volatility".
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

# exchange time zone, earliest plausible open (incl. pre-open), latest plausible close (incl. auction)
SESSIONS: dict[str, tuple[str, time, time]] = {
    # development markets
    "SPX": ("America/New_York", time(9, 30), time(16, 15)),
    "NASDAQ": ("America/New_York", time(9, 30), time(16, 15)),
    "RUT": ("America/New_York", time(9, 30), time(16, 15)),
    "FTSE": ("Europe/London", time(7, 50), time(16, 40)),
    "DAX": ("Europe/Berlin", time(8, 50), time(17, 40)),
    "CAC": ("Europe/Paris", time(8, 50), time(17, 40)),
    "N225": ("Asia/Tokyo", time(8, 55), time(15, 35)),
    "HSI": ("Asia/Hong_Kong", time(9, 0), time(16, 15)),
    "SENSEX": ("Asia/Kolkata", time(9, 0), time(15, 45)),
    "ASX": ("Australia/Sydney", time(9, 55), time(16, 20)),
    "KOSPI": ("Asia/Seoul", time(8, 55), time(15, 45)),
    "IBOV": ("America/Sao_Paulo", time(9, 55), time(18, 15)),
    # holdout markets
    "TSX": ("America/Toronto", time(9, 30), time(16, 15)),
    "SMI": ("Europe/Zurich", time(8, 50), time(17, 40)),
    "TAIEX": ("Asia/Taipei", time(8, 55), time(13, 45)),
    "SHCOMP": ("Asia/Shanghai", time(9, 10), time(15, 15)),
    "IPC": ("America/Mexico_City", time(8, 25), time(15, 15)),
    "AEX": ("Europe/Amsterdam", time(8, 50), time(17, 40)),
    "IBEX": ("Europe/Madrid", time(8, 50), time(17, 40)),
    "KLCI": ("Asia/Kuala_Lumpur", time(8, 25), time(17, 15)),
    "JKSE": ("Asia/Jakarta", time(8, 45), time(16, 30)),
    "STI": ("Asia/Singapore", time(8, 25), time(17, 25)),
    "ATX": ("Europe/Vienna", time(8, 50), time(17, 40)),
    "BEL20": ("Europe/Brussels", time(8, 50), time(17, 40)),
}


def session_utc(market: str, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """UTC open and close timestamps (int64 ns) of a market's sessions on its local dates."""
    tz, t_open, t_close = SESSIONS[market]
    d = pd.DatetimeIndex(dates).normalize()

    def at(t: time) -> np.ndarray:
        local = (d + pd.Timedelta(hours=t.hour, minutes=t.minute)).tz_localize(
            tz, ambiguous="NaT", nonexistent="shift_forward")
        return local.tz_convert("UTC").asi8

    return at(t_open), at(t_close)


def overnight_news(rets: dict[str, pd.Series], scale_window: int = 250,
                   margin_minutes: int = 30) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """For every target market: (news, available), each a (target dates x markets) frame.

    news[j] is the sum of source j's standardised log returns over sessions that closed after the
    target's previous close and at least 2*margin before the target's open; available[j] is 1 if
    at least one such session exists. The target's own column is always 0 / unavailable.
    """
    margin = np.int64(margin_minutes) * 60 * 10**9
    markets = list(rets)
    src = {}
    for j in markets:
        r = rets[j]
        s = np.sqrt(pd.Series(r.to_numpy() ** 2).rolling(scale_window, min_periods=20).mean().shift(1))
        x = (r.to_numpy() / s.to_numpy())
        _, close = session_utc(j, r.index)
        ok = np.isfinite(x)
        src[j] = (close[ok], np.cumsum(np.concatenate([[0.0], np.clip(x[ok], -20, 20)])))
    out = {}
    for i in markets:
        dates = rets[i].index
        open_i, close_i = session_utc(i, dates)
        lo = np.concatenate([[np.iinfo(np.int64).min], close_i[:-1]])    # previous close of i
        hi = open_i - 2 * margin                                          # latest usable source close
        news = pd.DataFrame(0.0, index=dates, columns=markets)
        avail = pd.DataFrame(0.0, index=dates, columns=markets)
        for j in markets:
            if j == i:
                continue
            c, cs = src[j]
            a = np.searchsorted(c, lo, side="right")      # first source session closing after lo
            b = np.searchsorted(c, hi, side="right")      # sessions closing at or before hi
            b = np.maximum(a, b)
            assert np.all(c[np.maximum(b - 1, 0)][b > a] <= hi[b > a]), "look-ahead in news window"
            news[j] = cs[b] - cs[a]
            avail[j] = (b > a).astype(float)
        out[i] = (news, avail)
    return out
