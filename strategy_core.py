"""
Shared stock-selection logic used by Final_cv.py (live picks) and backtest.py.

Everything here is causal: the picks for week t only ever look at rows <= t
of the price matrix and at the fundamentals inside week t's own file, so the
same code can be replayed week by week in a backtest without look-ahead.
"""
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd

# ==========================================
# ⚙️ STRATEGY CONFIGURATIONS
# ==========================================
# "current" reproduces the original Final_cv.py selection exactly.
CURRENT_LOGIC = {
    "name": "current",
    "weights": {20: 0.5, 8: 0.3, 4: 0.15, 2: 0.05},
    "rank_normalize": False,     # raw mom/vol summed (20w term dominates ~76%)
    "ffill_limit": None,         # stale prices carried forward forever
    "require_fresh_price": False,
    "min_vol": 0.0,              # no floor on weekly volatility
    "max_flat_weeks_pct": 1.0,   # no check on weeks with zero price change
    "min_mcap": 500,
    "min_sales_growth": 0,
    "min_profit_growth": 0,
    "ma_weeks": 30,
    "ma_rising_weeks": 0,        # 0 = don't require a rising 30W MA
    "high_lookback": 40,
    "near_high": 0.85,
    "top_n": 20,
}

# "new" = current logic + the changes that held up in the walk-forward
# backtest on 77 weeks of real data (weeks 40-76 tested, Oct 2026):
#   sales growth > 10%     best single change on every metric, both halves
#   stale/illiquid guards  neutral on returns, kept so untradeable stocks
#                          can't reach the list
# Tested and NOT adopted (switches kept so they can be re-tested later):
#   rank_normalize=True          much worse (Sharpe 2.33 -> 1.27)
#   weights without 2-week term  worse short term
#   ma_rising_weeks=4            no effect
NEW_LOGIC = {
    **CURRENT_LOGIC,
    "name": "new",
    "ffill_limit": 2,            # a price older than 2 weeks is treated as missing
    "require_fresh_price": True, # must have an actual price in the signal week
    "min_vol": 0.01,             # >= 1% weekly vol: blocks stale/illiquid charts
    "max_flat_weeks_pct": 0.20,  # <= 20% of last 20 weeks with zero price change
    "min_sales_growth": 10,      # real growth, not 0.09%
}

LOGICS = {"current": CURRENT_LOGIC, "new": NEW_LOGIC}

FUND_COLS = ["Market Capitalization", "YOY Quarterly sales growth", "YOY Quarterly profit growth"]


# ==========================================
# 1️⃣ DATA LOADING
# ==========================================
def get_week_number(filename):
    match = re.search(r"Week\s*(\d+)", os.path.basename(filename), re.IGNORECASE)
    return int(match.group(1)) if match else -1


def load_all_weekly_data(folder, verbose=True):
    """Load every weekly file into one long DataFrame.

    Fails loudly on problems that silently corrupt results in the old loader:
    files without a week number and two files with the same week number.
    """
    files = (glob.glob(os.path.join(folder, "*.xlsx")) + glob.glob(os.path.join(folder, "*.xls"))
             + glob.glob(os.path.join(folder, "*.csv")))
    files = [f for f in files if not os.path.basename(f).startswith("~$")]  # Excel lock files
    if not files:
        raise FileNotFoundError(f"❌ No weekly files in '{folder}'")

    no_week = [f for f in files if get_week_number(f) < 0]
    if no_week:
        raise ValueError(f"❌ These files have no 'Week <n>' in their name: {no_week}")
    weeks = pd.Series([get_week_number(f) for f in files])
    dup = weeks[weeks.duplicated()].unique()
    if len(dup):
        raise ValueError(f"❌ Duplicate week numbers {sorted(dup)} - one file per week only")

    if verbose:
        print(f"📂 Found {len(files)} files. Loading...")
    all_data = []
    for f in files:
        df = pd.read_csv(f) if f.lower().endswith(".csv") else pd.read_excel(f)
        df.columns = df.columns.str.strip()
        if "Name" not in df.columns or "Current Price" not in df.columns:
            raise ValueError(f"❌ {f} is missing 'Name' or 'Current Price'")
        df = df[["Name", "Current Price"] + [c for c in FUND_COLS if c in df.columns]].copy()
        df["Name"] = df["Name"].astype(str).str.strip()
        df = df[df["Name"].ne("") & df["Name"].ne("nan")]
        n_dup = df["Name"].duplicated().sum()
        if n_dup and verbose:
            print(f"⚠️ {os.path.basename(f)}: {n_dup} duplicate names, keeping first")
        df = df.drop_duplicates("Name")
        for c in ["Current Price"] + FUND_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["Week_Index"] = get_week_number(f)
        all_data.append(df)

    data = pd.concat(all_data, ignore_index=True).sort_values("Week_Index")
    wk = np.sort(data["Week_Index"].unique())
    gaps = wk[1:][np.diff(wk) > 1]
    if len(gaps) and verbose:
        print(f"⚠️ Missing week numbers before weeks {list(gaps)} - lookbacks count files, not calendar weeks")
    return data


# ==========================================
# 2️⃣ PANEL (all indicators, computed once)
# ==========================================
class Panel:
    """Price/fundamental matrices (index = week, columns = stock) plus cached indicators."""

    def __init__(self, master_df):
        self.raw_price = master_df.pivot(index="Week_Index", columns="Name", values="Current Price")
        self.weeks = list(self.raw_price.index)
        self.fund = {c: master_df.pivot(index="Week_Index", columns="Name", values=c)
                     .reindex(index=self.raw_price.index, columns=self.raw_price.columns)
                     for c in FUND_COLS if c in master_df.columns}
        # listed[t, s] = stock s appears in week t's file (with or without a price)
        self.listed = (master_df.assign(v=True).pivot(index="Week_Index", columns="Name", values="v")
                       .reindex(index=self.raw_price.index, columns=self.raw_price.columns)
                       .notna())
        self._cache = {}

    def prices(self, ffill_limit):
        key = ("px", ffill_limit)
        if key not in self._cache:
            self._cache[key] = self.raw_price.ffill(limit=ffill_limit)
        return self._cache[key]

    def indicators(self, cfg):
        lim = cfg["ffill_limit"]
        key = ("ind", lim, tuple(sorted(cfg["weights"])), cfg["ma_weeks"], cfg["high_lookback"])
        if key in self._cache:
            return self._cache[key]
        px = self.prices(lim)
        ret1 = px.pct_change(1, fill_method=None)
        ind = {
            "ma": px.rolling(cfg["ma_weeks"]).mean(),
            "high": px.rolling(cfg["high_lookback"]).max(),
            "vol20": ret1.rolling(20).std(),
            "flat20": (ret1 == 0).astype(float).where(ret1.notna()).rolling(20).mean(),
            "cv": {},
        }
        for w in cfg["weights"]:
            mom = px.pct_change(w, fill_method=None)
            vol = ret1.rolling(w).std()
            ind["cv"][w] = mom / (vol + 1e-6)
        self._cache[key] = ind
        return ind


# ==========================================
# 3️⃣ SELECTION FOR ONE WEEK
# ==========================================
def select(panel, t, cfg):
    """Score and filter every stock for week position t (0-based row of the panel).

    Returns a DataFrame for every stock listed in week t with its indicators, a
    Drop_Reason ('' when it passed every filter), Composite_Score and Rank
    (1 = best, NaN when not eligible). Sorted best first.
    """
    ind = panel.indicators(cfg)
    px = panel.prices(cfg["ffill_limit"])
    listed = panel.listed.iloc[t]
    names = listed.index[listed.values]

    def row(m):
        return m.iloc[t].reindex(names)

    df = pd.DataFrame({
        "Price": row(px),
        "MA_30": row(ind["ma"]),
        "High_40": row(ind["high"]),
        "Vol_20W": row(ind["vol20"]),
        "Flat_Weeks_Pct": row(ind["flat20"]),
    })
    for c in FUND_COLS:
        df[c] = row(panel.fund[c]) if c in panel.fund else np.nan
    t_back = t - cfg["ma_rising_weeks"]
    df["MA_30_Prev"] = ind["ma"].iloc[t_back].reindex(names) if cfg["ma_rising_weeks"] and t_back >= 0 else np.nan

    cv = pd.DataFrame({w: row(ind["cv"][w]) for w in cfg["weights"]})

    # --- filters, first failing reason wins ---
    reason = pd.Series("", index=names, dtype=object)

    def drop(mask, why):
        # comparisons against NaN are False, so ~(x > y) also drops missing data
        reason[(reason == "") & mask.astype(bool)] = why

    if cfg["require_fresh_price"]:
        drop(panel.raw_price.iloc[t].reindex(names).isna(), "No price this week")
    drop(df["Price"].isna() | cv.isna().any(axis=1), "Too little price history")
    drop(~(df["Market Capitalization"] >= cfg["min_mcap"]), "Market cap too small/missing")
    drop(~(df["YOY Quarterly sales growth"] > cfg["min_sales_growth"]), "Sales growth too low/missing")
    drop(~(df["YOY Quarterly profit growth"] > cfg["min_profit_growth"]), "Profit growth too low/missing")
    if cfg["min_vol"] > 0:
        drop(~(df["Vol_20W"] >= cfg["min_vol"]), "Price barely moves (stale/illiquid)")
    if cfg["max_flat_weeks_pct"] < 1:
        drop(~(df["Flat_Weeks_Pct"] <= cfg["max_flat_weeks_pct"]), "Too many flat weeks (illiquid)")
    drop(~(df["Price"] >= cfg["near_high"] * df["High_40"]), "Not near 40W high")
    drop(~(df["Price"] > df["MA_30"]), "Below 30W MA")
    if cfg["ma_rising_weeks"]:
        drop(~(df["MA_30"] > df["MA_30_Prev"]), "30W MA not rising")
    df["Drop_Reason"] = reason
    ok = reason == ""

    # --- score ---
    w = pd.Series(cfg["weights"], dtype=float)
    if cfg["rank_normalize"]:
        pct = cv[ok].rank(pct=True) * 100           # 0-100 within eligible stocks
        score = (pct * w).sum(axis=1) / w.sum()
    else:
        score = (cv * w).sum(axis=1, min_count=len(w))
    df["Composite_Score"] = score
    df["Rank"] = df["Composite_Score"].where(ok).rank(ascending=False, method="first")
    return df.sort_values(["Rank", "Composite_Score"], ascending=[True, False], na_position="last")


def top_picks(sel, top_n):
    return sel[sel["Rank"] <= top_n]


def recommendation_streaks(panel, t, cfg, names):
    """Weeks_Recommended: consecutive weeks before t each name was in the top N.

    Recomputed from the weekly data (not from saved Results files), so running
    the script twice in one week or skipping a week can't distort it.
    """
    streak = pd.Series(0, index=names, dtype=int)
    alive = pd.Series(True, index=names)
    for back in range(t - 1, -1, -1):
        prev = set(top_picks(select(panel, back, cfg), cfg["top_n"]).index)
        alive &= streak.index.isin(prev)
        if not alive.any():
            break
        streak[alive] += 1
    return streak


# Entry timing by Weeks_Recommended (backtest, 77 weeks, both logics):
#   1-3 weeks: ~2% better over the next 4 weeks than brand-new picks (~96% confidence)
#   4-5 weeks: median 4w return negative, only 40-44% up, 18% lost >10%
#   6+ weeks: too few cases, results flip between runs
BUY_ZONE_WEEKS = (1, 3)


def entry_note(weeks_recommended, buy_zone=BUY_ZONE_WEEKS):
    lo, hi = buy_zone
    if weeks_recommended < lo:
        return "WATCH - new, wait 1 week"
    return "BUY ZONE" if weeks_recommended <= hi else "LATE - hold, no fresh buy"


def market_breadth(panel, t, cfg):
    px = panel.prices(cfg["ffill_limit"]).iloc[t]
    ma = panel.indicators(cfg)["ma"].iloc[t]
    valid = px.notna() & ma.notna()
    return float((px[valid] > ma[valid]).mean()) if valid.any() else float("nan")
