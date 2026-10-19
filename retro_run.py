"""
Retrospective run: replay the model as if it had been run every week, with a
different market-cap cut-off, without touching Final_cv.py or Results/.

    python retro_run.py "D:\\weekly_data"                      # mcap 1000, weeks 65+
    python retro_run.py "D:\\weekly_data" --min-mcap 2000 --from-week 60

Writes into Retro_mcap<N>/ :
  Results_Week<w>.xlsx   what Final_cv.py would have shown that week
                         (Top_Picks + Full_System_Status, same columns)
  Retro_Summary.xlsx     All_Picks       every pick, its streak, entry note and what
                                         it actually did afterwards (1/4/8/12 weeks)
                         Weekly          top-N vs market each week
                         Mcap_Comparison mcap 500 vs N, all tested weeks and weeks >= from-week
                         Streak_Groups   returns by Weeks_Recommended at mcap N
"""
import argparse
import os

import pandas as pd

import backtest
from strategy_core import LOGICS, Panel, entry_note, load_all_weekly_data, select


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default=os.environ.get("WEEKLY_DATA", backtest.DATA_FOLDER))
    ap.add_argument("--min-mcap", type=float, default=1000)
    ap.add_argument("--from-week", type=int, default=65)
    ap.add_argument("--logic", choices=sorted(LOGICS), default="new")
    ap.add_argument("--out", default=None, help="output folder (default Retro_mcap<N>)")
    args = ap.parse_args()

    base = LOGICS[args.logic]
    cfg = {**base, "min_mcap": args.min_mcap, "name": f"{args.logic} @ mcap {args.min_mcap:g}"}
    out = args.out or f"Retro_mcap{args.min_mcap:g}"
    os.makedirs(out, exist_ok=True)

    panel = Panel(load_all_weekly_data(args.folder))
    T = len(panel.weeks)
    retro_ts = [t for t in range(T) if panel.weeks[t] >= args.from_week]
    if not retro_ts:
        raise ValueError(f"❌ No weeks >= {args.from_week} (data has weeks {panel.weeks[0]}..{panel.weeks[-1]})")
    if retro_ts[0] < backtest.first_testable_week([cfg]):
        print(f"⚠️ Weeks before {panel.weeks[backtest.first_testable_week([cfg])]} lack enough history and give no picks")
    fwd = backtest.forward_returns(panel)

    # --- 1. picks every week (with streaks + what happened next) ---
    picks, weeks = backtest.run_logic(panel, cfg, fwd, retro_ts[0], include_last=True)
    picks["Entry_Note"] = picks["Weeks_Recommended"].map(entry_note)

    # --- 2. one Results file per week, same layout as Final_cv.py ---
    for t in retro_ts:
        week = panel.weeks[t]
        full = select(panel, t, cfg)
        full["Action_Signal"] = (full["Price"] > full["MA_30"]).map({True: "BUY/HOLD", False: "SELL"})
        full["Stop_Loss"] = full["MA_30"]
        wk_picks = picks[picks["Week"] == week].set_index("Name")
        top = full[full["Rank"] <= cfg["top_n"]].copy()
        top["Weeks_Recommended"] = wk_picks["Weeks_Recommended"].reindex(top.index)
        top["Entry_Note"] = wk_picks["Entry_Note"].reindex(top.index)
        full["Weeks_Recommended"] = top["Weeks_Recommended"].reindex(full.index).fillna(0).astype(int)
        with pd.ExcelWriter(os.path.join(out, f"Results_Week{week}.xlsx")) as xw:
            top.to_excel(xw, sheet_name="Top_Picks", index_label="Name")
            full.to_excel(xw, sheet_name="Full_System_Status", index_label="Name")

    # --- 3. how much the mcap cut-off changes results ---
    first_t = backtest.first_testable_week([base, cfg])
    comparison = {}
    for label, c in [(f"{args.logic} @ mcap {base['min_mcap']:g}", base), (cfg["name"], cfg)]:
        p_all, w_all = backtest.run_logic(panel, c, fwd, first_t)
        comparison[f"{label} | weeks {panel.weeks[first_t]}+"] = backtest.summarise(label, p_all, w_all)
        recent = w_all["Week"] >= args.from_week
        if recent.sum() >= 2:
            p_rec = p_all[p_all["Week"] >= args.from_week]
            comparison[f"{label} | weeks {args.from_week}+"] = backtest.summarise(label, p_rec, w_all[recent])
    comparison = pd.DataFrame(comparison).drop(index="Logic", errors="ignore")

    summary_file = os.path.join(out, "Retro_Summary.xlsx")
    with pd.ExcelWriter(summary_file) as xw:
        picks.to_excel(xw, sheet_name="All_Picks", index=False)
        weeks.to_excel(xw, sheet_name="Weekly", index=False)
        comparison.to_excel(xw, sheet_name="Mcap_Comparison")
        backtest.streak_table(picks, backtest.STREAK_GROUPS).to_excel(xw, sheet_name="Streak_Groups", index=False)

    # --- console summary ---
    pd.set_option("display.width", 220)
    h = backtest.MAIN_HORIZON
    print("\n" + "=" * 80 + f"\n📅 WEEKLY PICKS ({cfg['name']}), weeks {panel.weeks[retro_ts[0]]}..{panel.weeks[-1]}\n" + "=" * 80)
    cols = ["Week", "Eligible", "Picks", f"Top_{h}w", f"Market_{h}w"]
    print(weeks[[c for c in cols if c in weeks]].round(3).to_string(index=False))
    print(f"(Top_{h}w / Market_{h}w = what this week's picks / the market did over the next {h} weeks; "
          f"blank = not enough weeks yet)")
    print("\n" + "=" * 80 + "\n📊 MCAP COMPARISON\n" + "=" * 80)
    rows = [f"Avg eligible stocks/week", f"Top-N avg {h}w return", f"Market avg {h}w return", f"Top-N excess {h}w",
            "Top-N excess 12w", f"Picks up after {h}w (%)", f"Picks beat market after {h}w (%)",
            f"Avg worst dip within {h}w", "Strategy CAGR (net of costs)", "Strategy max drawdown",
            "Sharpe (weekly, annualised)", "Weekly excess return t-stat"]
    print(comparison.reindex([r for r in rows if r in comparison.index]).map(
        lambda v: f"{v:.4f}" if isinstance(v, float) else v).to_string())
    print(f"\n✅ {len(retro_ts)} weekly Results files + Retro_Summary.xlsx saved in '{out}/'")


if __name__ == "__main__":
    main()
