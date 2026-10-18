from datetime import datetime
import os
import sys

import pandas as pd

from strategy_core import LOGICS, Panel, load_all_weekly_data, market_breadth, recommendation_streaks, select

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
DATA_FOLDER = r"D:\weekly_data"   # or: python Final_cv.py "path\to\weekly_data"
ACTIVE_LOGIC = "new"              # "new" or "current" - run backtest.py before switching
RESULTS_FOLDER = "Results"

# Market Regime Settings (information only - does not change the picks)
BREADTH_STRONG = 0.50
BREADTH_WEAK = 0.30

# All strategy parameters (weights, filters, top N) live in strategy_core.py


# ==========================================
# 3️⃣ EXECUTION & OUTPUT
# ==========================================
if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("WEEKLY_DATA", DATA_FOLDER)
    cfg = LOGICS[ACTIVE_LOGIC]
    top_n = cfg["top_n"]

    panel = Panel(load_all_weekly_data(folder))
    t = len(panel.weeks) - 1
    print(f"🔄 Selecting with '{ACTIVE_LOGIC}' logic for week {panel.weeks[t]}...")

    breadth = market_breadth(panel, t, cfg)
    print(f"\n🚦 MARKET BREADTH: {breadth:.1%} stocks above 30W MA")
    if breadth >= BREADTH_STRONG:
        print("🟢 STRONG MARKET")
    elif breadth >= BREADTH_WEAK:
        print("🟡 CAUTION MARKET")
    else:
        print("🔴 WEAK MARKET: momentum picks fail more often in weak markets")

    full_report = select(panel, t, cfg)
    full_report["Action_Signal"] = (full_report["Price"] > full_report["MA_30"]).map(
        {True: "BUY/HOLD", False: "SELL"})
    full_report["Stop_Loss"] = full_report["MA_30"]   # weekly close below 30W MA = exit

    top_buys = full_report[full_report["Rank"] <= top_n].copy()
    top_buys["Weeks_Recommended"] = recommendation_streaks(panel, t, cfg, top_buys.index)
    full_report["Weeks_Recommended"] = top_buys["Weeks_Recommended"].reindex(full_report.index).fillna(0).astype(int)

    print("\n" + "=" * 80)
    print(f"🚀 TOP {top_n} STOCKS ({ACTIVE_LOGIC} logic) - weights {cfg['weights']}")
    print("=" * 80)
    display_cols = ["Rank", "Price", "Weeks_Recommended", "Stop_Loss", "Composite_Score"]
    print(top_buys[display_cols].round(2).to_string())

    print("\n📉 Why the other stocks were excluded:")
    print(full_report["Drop_Reason"].replace("", "✅ Passed all filters").value_counts().to_string())

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    filename = os.path.join(RESULTS_FOLDER, f"Results_{datetime.today().strftime('%Y-%m-%d')}.xlsx")
    with pd.ExcelWriter(filename) as writer:
        top_buys.to_excel(writer, sheet_name="Top_Picks", index_label="Name")
        full_report.to_excel(writer, sheet_name="Full_System_Status", index_label="Name")
    print(f"\n✅ Saved to '{filename}' (Full_System_Status now lists every stock with its Drop_Reason)")
