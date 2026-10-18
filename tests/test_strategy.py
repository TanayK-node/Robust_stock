"""Checks on synthetic weekly files: run with `python -m pytest tests`."""
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest  # noqa: E402
import legacy_final_cv  # noqa: E402  (original Final_cv.py, untouched)
from strategy_core import (CURRENT_LOGIC, NEW_LOGIC, Panel, load_all_weekly_data,  # noqa: E402
                           recommendation_streaks, select)

N_STOCKS, N_WEEKS = 300, 70


def make_market(folder, seed=0):
    """Write one CSV per week with realistic mess: listings, delistings, stale
    prices, missing prices and missing fundamentals. Drift is persistent, so
    momentum genuinely predicts returns in this fake market."""
    rng = np.random.default_rng(seed)
    drift = rng.normal(0.002, 0.006, N_STOCKS)
    vol = rng.uniform(0.02, 0.08, N_STOCKS)
    rets = rng.normal(drift, vol, (N_WEEKS, N_STOCKS))
    px = 100 * np.exp(np.cumsum(rets, axis=0)) * rng.uniform(0.2, 20, N_STOCKS)
    start = np.where(rng.random(N_STOCKS) < 0.1, rng.integers(5, 40, N_STOCKS), 0)
    end = np.where(rng.random(N_STOCKS) < 0.05, rng.integers(45, N_WEEKS, N_STOCKS), N_WEEKS)
    stale = rng.random(N_STOCKS) < 0.05            # illiquid: price rarely changes
    for s in np.where(stale)[0]:
        for w in range(1, N_WEEKS):
            if rng.random() < 0.7:
                px[w, s] = px[w - 1, s]
    mcap = rng.lognormal(7, 1.5, N_STOCKS)
    names = [f"Stock {i:03d}" for i in range(N_STOCKS)]
    os.makedirs(folder, exist_ok=True)
    for w in range(N_WEEKS):
        live = (start <= w) & (w < end)
        df = pd.DataFrame({
            "Name": np.array(names)[live],
            "Current Price": px[w, live].round(2),
            "Market Capitalization": (mcap[live] * px[w, live] / px[0, live]).round(2),
            "YOY Quarterly sales growth": rng.normal(15, 25, live.sum()).round(2),
            "YOY Quarterly profit growth": rng.normal(20, 60, live.sum()).round(2),
        })
        df.loc[rng.random(len(df)) < 0.02, "Current Price"] = np.nan
        df.loc[rng.random(len(df)) < 0.05, "YOY Quarterly sales growth"] = np.nan
        df.to_csv(os.path.join(folder, f"Data Week {w + 1}.csv"), index=False)
    return names, stale


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    folder = str(tmp_path_factory.mktemp("market"))
    names, stale = make_market(folder)
    data = load_all_weekly_data(folder, verbose=False)
    return folder, data, Panel(data), names, stale


def test_current_logic_matches_original_final_cv(market):
    folder, data, panel, _, _ = market
    with contextlib.redirect_stdout(io.StringIO()):
        legacy_final, legacy_full = legacy_final_cv.apply_advanced_strategy(
            legacy_final_cv.load_all_weekly_data(folder))
    sel = select(panel, len(panel.weeks) - 1, CURRENT_LOGIC)
    eligible = sel[sel["Drop_Reason"] == ""]
    assert len(legacy_final) > 20
    assert list(eligible.index) == list(legacy_final.index)
    np.testing.assert_allclose(eligible["Composite_Score"].values, legacy_final["Composite_Score"].values)
    # Full_System_Status (filtered list incl. SELLs) = everything that only failed the MA test
    old_full = set(legacy_full.index)
    new_full = set(sel.index[sel["Drop_Reason"].isin(["", "Below 30W MA"])])
    assert old_full == new_full


@pytest.mark.parametrize("cfg", [CURRENT_LOGIC, NEW_LOGIC], ids=["current", "new"])
def test_no_lookahead(market, cfg):
    """Picks for week t must not change when future weeks are deleted."""
    _, data, panel, _, _ = market
    for t in [45, 55, 65]:
        week = panel.weeks[t]
        truncated = Panel(data[data["Week_Index"] <= week])
        a = select(panel, t, cfg)
        b = select(truncated, t, cfg)
        pd.testing.assert_frame_equal(a, b)


def test_new_logic_filters(market):
    _, _, panel, names, stale = market
    t = len(panel.weeks) - 1
    sel = select(panel, t, NEW_LOGIC)
    ok = sel[sel["Drop_Reason"] == ""]
    assert len(ok) > 0
    assert (ok["Price"] > ok["MA_30"]).all()
    assert (ok["MA_30"] > ok["MA_30_Prev"]).all()
    assert (ok["Price"] >= 0.85 * ok["High_40"]).all()
    assert (ok["YOY Quarterly sales growth"] > 10).all()
    assert (ok["Market Capitalization"] >= 500).all()
    assert (ok["Vol_20W"] >= 0.01).all()
    assert ok["Composite_Score"].between(0, 100).all()
    assert list(ok["Rank"]) == list(range(1, len(ok) + 1))
    assert sel.loc[sel["Drop_Reason"] != "", "Rank"].isna().all()
    stale_names = {n for n, s in zip(names, stale) if s}
    assert not stale_names & set(ok.index), "stale-price stocks should be filtered"
    assert panel.raw_price.iloc[t].reindex(ok.index).notna().all()


def test_streaks_match_backtest(market):
    _, _, panel, _, _ = market
    fwd = backtest.forward_returns(panel)
    first_t = backtest.first_testable_week([CURRENT_LOGIC, NEW_LOGIC])
    picks, _ = backtest.run_logic(panel, NEW_LOGIC, fwd, first_t)
    t = len(panel.weeks) - 2
    last = picks[picks["Week"] == panel.weeks[t]].set_index("Name")["Weeks_Recommended"]
    live = recommendation_streaks(panel, t, NEW_LOGIC, last.index)
    pd.testing.assert_series_equal(last.sort_index(), live.sort_index(), check_names=False, check_dtype=False)
    assert last.max() > 0


def test_forward_returns_are_future_prices(market):
    _, _, panel, _, _ = market
    fwd = backtest.forward_returns(panel)
    px = panel.prices(None)
    s = px.columns[0]
    t = 40
    assert fwd[4].iloc[t][s] == pytest.approx(px.iloc[t + 4][s] / px.iloc[t][s] - 1)
    assert fwd["min"].iloc[t][s] == pytest.approx(px.iloc[t + 1:t + 5][s].min() / px.iloc[t][s] - 1)


def test_backtest_end_to_end(market, tmp_path):
    folder = market[0]
    out = str(tmp_path / "report.xlsx")
    with contextlib.redirect_stdout(io.StringIO()):
        summary, detail = backtest.run(folder, out)
    assert {"current", "new"} <= set(summary.columns)
    sheets = pd.read_excel(out, sheet_name=None)
    assert {"Summary", "Streak_current", "Streak_new", "Weekly_new", "Picks_new"} <= set(sheets)
    picks = detail["new"][0]
    assert picks.groupby("Week").size().max() <= 20
    # momentum is real in the fake market, so the ranking should have positive IC
    assert summary.loc["Avg rank IC (4w)", "new"] > 0


def test_loader_rejects_bad_files(tmp_path):
    pd.DataFrame({"Name": ["A"], "Current Price": [1]}).to_csv(tmp_path / "Week 1.csv", index=False)
    pd.DataFrame({"Name": ["A"], "Current Price": [1]}).to_csv(tmp_path / "Week 1 copy.csv", index=False)
    with pytest.raises(ValueError, match="Duplicate week"):
        load_all_weekly_data(str(tmp_path), verbose=False)
    os.remove(tmp_path / "Week 1 copy.csv")
    pd.DataFrame({"Name": ["A"], "Current Price": [1]}).to_csv(tmp_path / "latest.csv", index=False)
    with pytest.raises(ValueError, match="no 'Week"):
        load_all_weekly_data(str(tmp_path), verbose=False)
