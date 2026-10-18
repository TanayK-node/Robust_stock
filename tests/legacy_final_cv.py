from datetime import datetime
import pandas as pd
import numpy as np
import glob
import re
import os

# ==========================================
# ⚙️ CONFIGURATION & RISK PARAMETERS
# ==========================================
DATA_FOLDER = r"D:\weekly_data"
TOTAL_CAPITAL = 100000       # Total cash available for this strategy
RISK_PER_TRADE_PCT = 0.01    # Risk 1% of total capital per trade
TOP_N_RESULTS = 20           


# Market Regime Settings
BREADTH_STRONG = 0.50
BREADTH_WEAK = 0.30

# Multi-Timeframe Weights (Must sum to 1.0)
# We prioritize the 20-week trend (40%) but verify it with shorter windows
WEIGHTS = {20: 0.5, 8: 0.3, 4: 0.15, 2: 0.05}
LOOKBACKS = list(WEIGHTS.keys())

# Strategy Parameters
TREND_MA_WEEKS = 30          # Sell signal if price falls below this
HIGH_LOOKBACK = 40           
NEAR_HIGH_THRESHOLD = 0.85   

# Fundamental Filters
USE_FUNDAMENTALS = True      
MIN_SALES_GROWTH = 0         
MIN_PROFIT_GROWTH = 0        
RESULTS_FOLDER = "Results"
# ==========================================
def get_week_number(filename):
    match = re.search(r"Week\s*(\d+)", filename, re.IGNORECASE)
    return int(match.group(1)) if match else -1


# ==========================================
# 2️⃣ ADVANCED QUANT STRATEGY ENGINE (MULTI-TIMEFRAME)
# ==========================================
def apply_advanced_strategy(master_df):
    print("🔄 Processing Multi-Timeframe Quantitative Signals...")
    
    # Pivot Prices for matrix calculations
    price_matrix = master_df.pivot_table(index="Week_Index", columns="Name", values="Current Price").ffill()
        # --- 🚦 MARKET BREADTH ---
    ma_30_all = price_matrix.rolling(window=TREND_MA_WEEKS).mean()
    stocks_above_ma = (price_matrix > ma_30_all).sum(axis=1)
    total_stocks = price_matrix.count(axis=1)
    market_breadth = (stocks_above_ma / total_stocks).iloc[-1]

    print(f"\n🚦 MARKET BREADTH: {market_breadth:.1%} stocks above 30W MA")

    # --- 📈 COMPOSITE SCORE CALCULATION ---
    # We initialize a score series with zeros
    composite_score = pd.Series(0.0, index=price_matrix.columns)
    
    print(f"   Calculated Weights: {WEIGHTS}")
    
    for weeks in LOOKBACKS:
        # Calculate Momentum (Total Return over N weeks)
        mom = price_matrix.pct_change(periods=weeks).iloc[-1]
        
        # Calculate Volatility (Std Dev of 1-week returns over N weeks)
        vol = price_matrix.pct_change(1).rolling(window=weeks).std().iloc[-1]
        
        # Inverse CV for this timeframe: Higher is better (Stable Growth)
        # We add 1e-6 to avoid division by zero
        cv_score = mom / (vol + 1e-6)
        
        # Add to the total weighted score
        composite_score += cv_score * WEIGHTS[weeks]

    # --- TECHNICAL INDICATORS ---
    ma_30 = price_matrix.rolling(window=TREND_MA_WEEKS).mean().iloc[-1]
    high_40 = price_matrix.rolling(window=HIGH_LOOKBACK).max().iloc[-1]
    current_prices = price_matrix.iloc[-1]

    # Get Latest Fundamentals
    latest_week = master_df["Week_Index"].max()
    latest_fund = master_df[master_df["Week_Index"] == latest_week].set_index("Name")
    
    # Build Master DataFrame
    results = pd.DataFrame({
        "Price": current_prices,
        "MA_30": ma_30,
        "High_40": high_40,
        "Composite_Score": composite_score
    })
    
    results = results.join(latest_fund[["Market Capitalization", "YOY Quarterly sales growth", "YOY Quarterly profit growth"]], how="inner")
    
    # Drop rows where critical data is missing (Price or Score)
    results = results.dropna(subset=["Price", "Composite_Score"])

    # --- 🚦 SIGNALS & FILTERS ---
    # 1. Fundamental & Liquidity Filters
    if USE_FUNDAMENTALS:
        results = results[
            (results["Market Capitalization"] >= 500) & 
            (results["YOY Quarterly sales growth"] > MIN_SALES_GROWTH) & 
            (results["YOY Quarterly profit growth"] > MIN_PROFIT_GROWTH)
        ]

    # 2. Sell Signal: If Price < 30-week MA, signal is SELL
    results["Action_Signal"] = np.where(results["Price"] > results["MA_30"], "BUY/HOLD", "SELL")
    
    # 3. Near Highs Filter (Must be within 15% of 40-week High)
    results = results[results["Price"] >= (NEAR_HIGH_THRESHOLD * results["High_40"])]

   # --- 📊 POSITION SIZING & DYNAMIC RISK ---
    results["Stop_Loss"] = results["MA_30"]
    results["Risk_Per_Share"] = results["Price"] - results["Stop_Loss"]

    # 1. Determine Market Regime Multiplier First
    if market_breadth >= BREADTH_STRONG:
        regime_multiplier = 1.0
        print("🟢 STRONG MARKET: Full risk allocation allowed.")
    elif market_breadth >= BREADTH_WEAK:
        regime_multiplier = 0.5
        print("🟡 CAUTION MARKET: Risk reduced to 50%.")
    else:
        regime_multiplier = 0.25  # Fixed: Changed from 0.25 to 0.0 to match your "No new positions" logic
        print("🔴 WEAK MARKET: No new positions allowed.")

    cash_to_risk = TOTAL_CAPITAL * RISK_PER_TRADE_PCT * regime_multiplier

    # 2. Calculate Shares Safely (Vectorized)
    risk = results["Risk_Per_Share"]
    
    # We substitute np.inf for 0 or negative risk to prevent division-by-zero warnings
    safe_risk = np.where(risk > 0, risk, np.inf) 
    
    # Calculate raw shares
    raw_shares = np.where(risk > 0, cash_to_risk / safe_risk, 0)
    
    # Filter out any remaining NaNs or Infs just to be perfectly safe, then convert to int
    valid_shares = np.where(np.isfinite(raw_shares), raw_shares, 0)
    results["Shares_to_Buy"] = valid_shares.astype(int)

    results["Suggested_Allocation"] = results["Shares_to_Buy"] * results["Price"]

    # --- 🚀 FINAL RANKING ---
    # Filter for BUY signals and sort by our new Weighted Composite Score
    if regime_multiplier == 0.0:
        final_list = results.iloc[0:0]  # Empty DataFrame
    else:
        final_list = results[results["Action_Signal"] == "BUY/HOLD"]
        final_list = final_list.sort_values(by="Composite_Score", ascending=False)

    
    return final_list, results


def load_all_weekly_data(folder):
    files = (
        glob.glob(f"{folder}/*.xlsx")
        + glob.glob(f"{folder}/*.xls")
        + glob.glob(f"{folder}/*.csv")
    )
    if not files: raise FileNotFoundError(f"❌ No Excel files in '{folder}'")
    
    all_data = []
    print(f"📂 Found {len(files)} files. Loading...")

    for i, f in enumerate(files):
        try:
            df = pd.read_csv(f) if f.lower().endswith(".csv") else pd.read_excel(f)
            df.columns = df.columns.str.strip() 
            cols = ["Name", "Current Price", "Market Capitalization"]
            fund_cols = ["YOY Quarterly sales growth", "YOY Quarterly profit growth"]
            
            if USE_FUNDAMENTALS:
                cols += [c for c in fund_cols if c in df.columns]

            df = df[[c for c in cols if c in df.columns]].copy()
            df["Week_Index"] = get_week_number(f)
            all_data.append(df)
        except Exception as e: print(f"❌ Error {f}: {e}")

    return pd.concat(all_data, ignore_index=True).sort_values("Week_Index")



def get_recommendation_streak(current_top_buys, results_folder):
    """
    Counts how many consecutive previous weeks each stock
    has appeared in the Top_Picks sheet.

    0 = New recommendation
    1 = Also recommended last week
    2 = Last 2 weeks
    etc.
    """

    current_top_buys = current_top_buys.copy()
    current_top_buys["Weeks_Recommended"] = 0

    files = sorted(glob.glob(os.path.join(results_folder, "Results_*.xlsx")))

    today = datetime.today().strftime("%Y-%m-%d")
    files = [
        f for f in files
        if today not in os.path.basename(f)
    ]

    previous_files = files[::-1]

    if len(files) == 0:
        return current_top_buys

    previous_files = files[::-1]      # newest → oldest

    for stock in current_top_buys.index:
        streak = 0

        for file in previous_files:
            try:
                prev = pd.read_excel(file, sheet_name="Top_Picks")

                if "Name" in prev.columns:
                    names = prev["Name"].astype(str).str.strip().values
                else:
                    # If stock names are saved as index
                    names = prev.iloc[:,0].astype(str).str.strip().values

                if stock in names:
                    streak += 1
                else:
                    break

            except:
                break

        current_top_buys.loc[stock, "Weeks_Recommended"] = streak

    return current_top_buys



# ==========================================
# 3️⃣ EXECUTION & OUTPUT
# ==========================================
if __name__ == "__main__":
    try:
        raw_data = load_all_weekly_data(DATA_FOLDER)
        top_buys, full_report = apply_advanced_strategy(raw_data)
        top_buys = get_recommendation_streak(top_buys, RESULTS_FOLDER)
        
        print("\n" + "="*80)
        print(f"🚀 TOP {TOP_N_RESULTS} STOCKS TO BUY (Multi-Timeframe Weighted)")
        print(f"Weights Used: {WEIGHTS}")
        print("="*80)
        
        display_cols = ["Price", "Action_Signal", "Weeks_Recommended", "Shares_to_Buy", "Stop_Loss", "Composite_Score" ]
        # Rounding for cleaner display
        disp = top_buys[display_cols].copy()
        disp["Composite_Score"] = disp["Composite_Score"].round(2)
        disp["Price"] = disp["Price"].round(2)
        disp["Stop_Loss"] = disp["Stop_Loss"].round(2)
        
        print(disp.head(TOP_N_RESULTS).to_string())
        today = datetime.today().strftime('%Y-%m-%d')
        
        # 2. Create the dynamic filename
        os.makedirs("Results", exist_ok=True)
        filename = os.path.join("Results", f"Results_{today}.xlsx")
        # Save output
        full_report = full_report.join(
            top_buys["Weeks_Recommended"],
            how="left"
            )

        full_report["Weeks_Recommended"] = (
                full_report["Weeks_Recommended"]
                .fillna(0)
                .astype(int)
            )
        with pd.ExcelWriter(filename) as writer:
            top_buys.head(TOP_N_RESULTS).to_excel(writer, sheet_name="Top_Picks")
            full_report.to_excel(writer, sheet_name="Full_System_Status")
            
        print(f"\n✅ Saved actionable signals to '{filename}'")
        print("💡 TIP: The 'Composite_Score' now reflects stability across 2, 4, 8, and 20 weeks.")
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")