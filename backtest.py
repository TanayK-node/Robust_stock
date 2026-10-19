"""
Walk-forward backtest of the stock-selection logic.

For every past week t it runs the exact same selection as Final_cv.py using only
data available up to week t, then measures what the picks did afterwards.

    python backtest.py "D:\\weekly_data"

Writes Backtest_Report.xlsx with:
  Summary         current vs new logic (+ each change on its own)
  Streak_<logic>  forward returns by Weeks_Recommended (0, 1, 2, ...)
  Weekly_<logic>  week-by-week top-N vs market returns
  Picks_<logic>   every pick with its streak and forward returns
"""
import os
import sys

import numpy as np
import pandas as pd

from strategy_core import CURRENT_LOGIC, NEW_LOGIC, Panel, load_all_weekly_data, select

DATA_FOLDER = r"D:\weekly_data"
OUTPUT_FILE = "Backtest_Report.xlsx"
HORIZONS = [1, 4, 8, 12]       # weeks ahead to measure
MAIN_HORIZON = 4               # used for hit rates / streak verdicts
ROUND_TRIP_COST = 0.005        # 0.5% per buy+sell (brokerage, STT, slippage)
MIN_BUCKET_OBS = 30            # fewer observations than this = not reliable
STREAK_BUCKETS = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"),
                  (6, 8, "6-8"), (9, 12, "9-12"), (13, 10**6, "13+")]
STREAK_GROUPS = [(0, 0, "0 (new pick)"), (1, 3, "1-3"), (4, 5, "4-5"), (6, 8, "6-8"), (9, 10**6, "9+")]

# Each change from NEW_LOGIC applied alone on top of CURRENT_LOGIC,
# so you can see which change helps and which doesn't.
ABLATIONS = {
    "current + rank-normalised score": {"rank_normalize": True},
    "current + no 2-week lookback": {"weights": {20: 0.5, 8: 0.3, 4: 0.2}},
    "current + stale/illiquid guards": {k: NEW_LOGIC[k] for k in
                                        ["ffill_limit", "require_fresh_price", "min_vol", "max_flat_weeks_pct"]},
    "current + sales growth > 10%": {"min_sales_growth": NEW_LOGIC["min_sales_growth"]},
    "current + rising 30W MA": {"ma_rising_weeks": 4},
}


def forward_returns(panel):
    """fwd[h].iloc[t, s] = return of stock s from week t to week t+h.

    Prices are carried forward without limit, so a stock that disappears keeps
    its last price (counted as 0% after it vanishes - real delistings are
    often worse, so results for picks that later delist are optimistic).
    """
    px = panel.prices(None)
    fwd = {h: px.shift(-h) / px - 1 for h in HORIZONS}
    # worst close over the next MAIN_HORIZON weeks vs entry (how far it fell first)
    future_min = pd.concat([px.shift(-k) for k in range(1, MAIN_HORIZON + 1)]).groupby(level=0).min()
    fwd["min"] = (future_min / px - 1).reindex(px.index)
    return fwd


def universe_mask(panel, t, min_mcap):
    """Benchmark = every stock listed in week t with a price and market cap >= min_mcap."""
    mcap = panel.fund["Market Capitalization"].iloc[t] if "Market Capitalization" in panel.fund else None
    m = panel.listed.iloc[t] & panel.raw_price.iloc[t].notna()
    if mcap is not None:
        m &= mcap >= min_mcap
    return m


def max_drawdown(curve):
    return float((curve / curve.cummax() - 1).min()) if len(curve) else np.nan


def run_logic(panel, cfg, fwd, first_t):
    """Replay the selection every week; return per-pick rows and per-week rows.

    Selection starts as early as this logic allows (so streaks are already
    counted correctly by first_t) but results are only recorded from first_t,
    so every logic is scored on the same weeks.
    """
    n = cfg["top_n"]
    picks_rows, week_rows = [], []
    prev_top, streak = [], {}
    T = len(panel.weeks)
    for t in range(first_testable_week([cfg]), T - 1):
        sel = select(panel, t, cfg)
        eligible = sel[sel["Drop_Reason"] == ""]
        top = eligible.head(n)
        top_names = list(top.index)

        # streak = consecutive previous weeks in the top N (same meaning as Weeks_Recommended)
        streak = {s: (streak.get(s, -1) + 1 if s in prev_top else 0) for s in top_names}
        prev_top = top_names
        if t < first_t:
            continue

        uni = universe_mask(panel, t, cfg["min_mcap"])
        uni_names = uni.index[uni.values]
        week = panel.weeks[t]
        wrow = {"Week": week, "Eligible": len(eligible), "Picks": len(top_names)}
        for h in HORIZONS:
            if t + h >= T:
                continue
            f = fwd[h].iloc[t]
            wrow[f"Top_{h}w"] = f.reindex(top_names).mean()
            wrow[f"Eligible_{h}w"] = f.reindex(eligible.index).mean()
            wrow[f"Market_{h}w"] = f.reindex(uni_names).mean()
            if h == MAIN_HORIZON and len(eligible) >= 5:
                s = eligible["Composite_Score"]
                r = f.reindex(eligible.index)
                ok = r.notna()
                wrow["IC"] = s[ok].rank().corr(r[ok].rank()) if ok.sum() >= 5 else np.nan
        week_rows.append(wrow)

        mkt_main = wrow.get(f"Market_{MAIN_HORIZON}w", np.nan)
        for s in top_names:
            prow = {"Week": week, "Name": s, "Rank": int(top.at[s, "Rank"]),
                    "Weeks_Recommended": streak[s], "Composite_Score": top.at[s, "Composite_Score"]}
            for h in HORIZONS:
                prow[f"Ret_{h}w"] = fwd[h].iat[t, fwd[h].columns.get_loc(s)] if t + h < T else np.nan
            prow[f"Worst_Dip_{MAIN_HORIZON}w"] = (fwd["min"].iat[t, fwd["min"].columns.get_loc(s)]
                                                  if t + MAIN_HORIZON < T else np.nan)
            prow[f"Market_{MAIN_HORIZON}w"] = mkt_main
            picks_rows.append(prow)

    return pd.DataFrame(picks_rows), pd.DataFrame(week_rows)


def portfolio_stats(picks, weeks):
    """Equal-weight top-N bought every week and held 1 week, with trading costs."""
    if weeks.empty or "Top_1w" not in weeks:
        return {}
    w = weeks.dropna(subset=["Market_1w"]).copy()
    w["Top_1w"] = w["Top_1w"].where(w["Picks"] > 0, 0.0)   # no picks that week = sit in cash
    held = picks.groupby("Week")["Name"].apply(set)
    turnover = []
    prev = set()
    for wk in w["Week"]:
        cur = held.get(wk, set())
        turnover.append(len(cur - prev) / len(cur) if cur else 0.0)
        prev = cur
    w["Turnover"] = turnover
    net = w["Top_1w"] - w["Turnover"] * ROUND_TRIP_COST
    mkt = w["Market_1w"]
    years = len(w) / 52
    excess = net - mkt
    half = len(w) // 2
    return {
        "Weeks tested": len(w),
        "Strategy CAGR (net of costs)": (1 + net).prod() ** (1 / years) - 1 if years else np.nan,
        "Market CAGR": (1 + mkt).prod() ** (1 / years) - 1 if years else np.nan,
        "Strategy max drawdown": max_drawdown((1 + net).cumprod()),
        "Market max drawdown": max_drawdown((1 + mkt).cumprod()),
        "Sharpe (weekly, annualised)": net.mean() / net.std() * np.sqrt(52) if net.std() > 0 else np.nan,
        "Avg weekly turnover": w["Turnover"].mean(),
        "Weekly excess return t-stat": excess.mean() / (excess.std() / np.sqrt(len(excess)))
        if excess.std() > 0 else np.nan,
        "Excess/week 1st half": excess.iloc[:half].mean(),
        "Excess/week 2nd half": excess.iloc[half:].mean(),
    }


def summarise(name, picks, weeks):
    h = MAIN_HORIZON
    p = picks.dropna(subset=[f"Ret_{h}w"])
    out = {"Logic": name,
           "Avg eligible stocks/week": weeks["Eligible"].mean(),
           "Avg picks/week": weeks["Picks"].mean()}
    for hz in HORIZONS:
        if f"Top_{hz}w" in weeks:
            out[f"Top-N avg {hz}w return"] = weeks[f"Top_{hz}w"].mean()
            out[f"Market avg {hz}w return"] = weeks[f"Market_{hz}w"].mean()
            out[f"Top-N excess {hz}w"] = (weeks[f"Top_{hz}w"] - weeks[f"Market_{hz}w"]).mean()
    out[f"Picks up after {h}w (%)"] = (p[f"Ret_{h}w"] > 0).mean()
    out[f"Picks beat market after {h}w (%)"] = (p[f"Ret_{h}w"] > p[f"Market_{h}w"]).mean()
    out[f"Avg worst dip within {h}w"] = p[f"Worst_Dip_{h}w"].mean()
    out[f"Avg rank IC ({h}w)"] = weeks["IC"].mean() if "IC" in weeks else np.nan
    out.update(portfolio_stats(picks, weeks))
    return out


def streak_table(picks, buckets=STREAK_BUCKETS):
    """Forward returns of picks grouped by Weeks_Recommended.

    'vs other picks' compares each pick with the average of the same week's
    top-N, which removes market ups and downs: it answers "within this week's
    list, are the ones with streak k better or worse?".
    """
    h = MAIN_HORIZON
    picks = picks.copy()
    for hz in HORIZONS:
        picks[f"Peer_{hz}w"] = picks[f"Ret_{hz}w"] - picks.groupby("Week")[f"Ret_{hz}w"].transform("mean")
    rows = []
    for lo, hi, label in buckets:
        b = picks[(picks["Weeks_Recommended"] >= lo) & (picks["Weeks_Recommended"] <= hi)]
        bh = b.dropna(subset=[f"Ret_{h}w"])
        row = {"Weeks_Recommended": label, "Observations": len(bh)}
        for hz in HORIZONS:
            row[f"Avg {hz}w return"] = b[f"Ret_{hz}w"].mean()
            row[f"Median {hz}w return"] = b[f"Ret_{hz}w"].median()
        row[f"Up after {h}w (%)"] = (bh[f"Ret_{h}w"] > 0).mean() if len(bh) else np.nan
        row[f"Beat market {h}w (%)"] = (bh[f"Ret_{h}w"] > bh[f"Market_{h}w"]).mean() if len(bh) else np.nan
        row[f"Avg excess {h}w"] = (bh[f"Ret_{h}w"] - bh[f"Market_{h}w"]).mean() if len(bh) else np.nan
        for hz in HORIZONS[1:]:
            row[f"vs other picks {hz}w"] = b[f"Peer_{hz}w"].mean()
        row[f"Avg worst dip {h}w"] = bh[f"Worst_Dip_{h}w"].mean() if len(bh) else np.nan
        row[f"Lost >10% in {h}w (%)"] = (bh[f"Ret_{h}w"] < -0.10).mean() if len(bh) else np.nan
        row["Reliable"] = "yes" if len(bh) >= MIN_BUCKET_OBS else "too few - ignore"
        rows.append(row)
    return pd.DataFrame(rows)


def first_testable_week(cfgs):
    need = max(max(max(c["weights"]) + 1, c["ma_weeks"] + c["ma_rising_weeks"], c["high_lookback"], 21)
               for c in cfgs)
    return need - 1


def run(folder, output=OUTPUT_FILE):
    panel = Panel(load_all_weekly_data(folder))
    configs = {"current": CURRENT_LOGIC, "new": NEW_LOGIC}
    configs.update({k: {**CURRENT_LOGIC, **v, "name": k} for k, v in ABLATIONS.items()})
    first_t = first_testable_week(configs.values())
    T = len(panel.weeks)
    if T - 1 - first_t < 5:
        raise ValueError(f"❌ Need at least {first_t + 6} weekly files to backtest, found {T}")
    print(f"📅 {T} weeks loaded; testing weeks {panel.weeks[first_t]}..{panel.weeks[T - 2]}")

    fwd = forward_returns(panel)
    summary, detail = [], {}
    for name, cfg in configs.items():
        print(f"🔁 Backtesting: {name}")
        picks, weeks = run_logic(panel, cfg, fwd, first_t)
        summary.append(summarise(name, picks, weeks))
        if name in ("current", "new"):
            detail[name] = (picks, weeks)

    summary = pd.DataFrame(summary).set_index("Logic").T
    with pd.ExcelWriter(output) as xw:
        summary.to_excel(xw, sheet_name="Summary")
        for name, (picks, weeks) in detail.items():
            streak_table(picks).to_excel(xw, sheet_name=f"Streak_{name}", index=False)
            streak_table(picks, STREAK_GROUPS).to_excel(xw, sheet_name=f"StreakGroups_{name}", index=False)
            weeks.to_excel(xw, sheet_name=f"Weekly_{name}", index=False)
            picks.to_excel(xw, sheet_name=f"Picks_{name}", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 80 + "\n📊 SUMMARY\n" + "=" * 80)
    print(summary.map(lambda v: f"{v:.4f}" if isinstance(v, float) else v).to_string())
    for name, (picks, _) in detail.items():
        print("\n" + "=" * 80 + f"\n⏳ RETURNS BY WEEKS_RECOMMENDED ({name} logic)\n" + "=" * 80)
        st = streak_table(picks, STREAK_GROUPS)
        cols = ["Weeks_Recommended", "Observations", f"Avg {MAIN_HORIZON}w return",
                f"vs other picks {MAIN_HORIZON}w", "vs other picks 8w",
                f"Up after {MAIN_HORIZON}w (%)", f"Beat market {MAIN_HORIZON}w (%)",
                f"Avg worst dip {MAIN_HORIZON}w", f"Lost >10% in {MAIN_HORIZON}w (%)", "Reliable"]
        print(st[cols].round(3).to_string(index=False))
    print(f"\n✅ Full report saved to '{output}'")
    return summary, detail


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("WEEKLY_DATA", DATA_FOLDER))
