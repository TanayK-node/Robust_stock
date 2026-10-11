import pandas as pd
import numpy as np
import glob
import re
import os

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
DATA_FOLDER = "weekly_data"
MIN_MARKET_CAP = 500         
TOP_N_RESULTS = 20           

# Strategy Parameters
TREND_MA_WEEKS = 30          
MOMENTUM_WEEKS = 20          
HIGH_LOOKBACK = 40           
NEAR_HIGH_THRESHOLD = 0.85   

# 🔹 NEW: FUNDAMENTAL FILTERS
USE_FUNDAMENTALS = True      # Set to False if columns are missing
MIN_SALES_GROWTH = 0         # YOY Quarterly Sales Growth must be positive
MIN_PROFIT_GROWTH = 0        # YOY Quarterly Profit Growth must be positive

# ==========================================
# 1️⃣ DATA LOADING
# ==========================================
def get_week_number(filename):
    match = re.search(r"Week\s*(\d+)", filename, re.IGNORECASE)
    return int(match.group(1)) if match else -1

def load_all_weekly_data(folder):
    files = glob.glob(f"{folder}/*.xlsx") + glob.glob(f"{folder}/*.xls")
    if not files: raise FileNotFoundError(f"❌ No Excel files in '{folder}'")
    
    all_data = []
    print(f"📂 Found {len(files)} files. Loading...")

    for i, f in enumerate(files):
        try:
            if i % 10 == 0: print(f"   Loading {i+1}/{len(files)}...")
            df = pd.read_excel(f)
            df.columns = df.columns.str.strip() 
            
            # Select Columns
            cols = ["Name", "Current Price", "Market Capitalization"]
            
            # Add Fundamental Columns if needed
            fund_cols = ["YOY Quarterly sales growth", "YOY Quarterly profit growth"]
            if USE_FUNDAMENTALS:
                cols += [c for c in fund_cols if c in df.columns]

            # Filter and add Week Index
            df = df[[c for c in cols if c in df.columns]].copy()
            df["Week_Index"] = get_week_number(f)
            all_data.append(df)
            
        except Exception as e: print(f"❌ Error {f}: {e}")

    return pd.concat(all_data, ignore_index=True).sort_values("Week_Index")

# ==========================================
# 2️⃣ STRATEGY CALCULATION
# ==========================================
def apply_strategy(master_df):
    print("🔄 Building Matrices...")
    
    # Pivot Prices
    price_matrix = master_df.pivot_table(index="Week_Index", columns="Name", values="Current Price").ffill()
    
    # --- 🚦 1. MARKET TRAFFIC LIGHT ---
    # Count how many stocks are above their 30-week MA for each week
    ma_30_all = price_matrix.rolling(window=TREND_MA_WEEKS).mean()
    stocks_above_ma = (price_matrix > ma_30_all).sum(axis=1)
    total_stocks = price_matrix.count(axis=1)
    market_breadth = (stocks_above_ma / total_stocks).iloc[-1]
    
    print(f"\n🚦 MARKET STATUS (Week {price_matrix.index.max()}): {market_breadth:.1%} of stocks are in an Uptrend.")
    
    if market_breadth < 0.50:
        print("⚠️ WARNING: Market Breath is < 50%. The market is weak. Be careful!")
    else:
        print("✅ GREEN LIGHT: Market is healthy (>50% stocks in uptrend).")

    # --- 📈 STOCK INDICATORS ---
    ma_30 = price_matrix.rolling(window=TREND_MA_WEEKS).mean()
    mom_20 = price_matrix.pct_change(periods=MOMENTUM_WEEKS)
    high_40 = price_matrix.rolling(window=HIGH_LOOKBACK).max()
    
    # 🔹 NEW: Volatility (Standard Deviation of weekly returns over 20 weeks)
    weekly_returns = price_matrix.pct_change(1)
    vol_20 = weekly_returns.rolling(window=MOMENTUM_WEEKS).std()

    # Get Latest Data
    current_prices = price_matrix.iloc[-1]
    latest_week = master_df["Week_Index"].max()
    latest_fund = master_df[master_df["Week_Index"] == latest_week].set_index("Name")
    
    # Build Results
    results = pd.DataFrame({
        "Price": current_prices,
        "MA_30": ma_30.iloc[-1],
        "Mom_20": mom_20.iloc[-1],
        "Vol_20": vol_20.iloc[-1],
        "High_40": high_40.iloc[-1],
    })
    
    # Join Fundamentals
    results = results.join(latest_fund, how="inner")
    results = results.dropna(subset=["Price", "MA_30", "Mom_20"]) # Drop if tech data missing

    # --- 🛑 FILTERS ---
    
    # 1. Liquidity
    results = results[results["Market Capitalization"] >= MIN_MARKET_CAP]
    
    # 2. Fundamentals (The Junk Filter)
    if USE_FUNDAMENTALS:
        # Fill NaN with 0 so we don't crash, but maybe exclude them
        results["YOY Quarterly sales growth"] = results.get("YOY Quarterly sales growth", pd.Series(0)).fillna(0)
        results["YOY Quarterly profit growth"] = results.get("YOY Quarterly profit growth", pd.Series(0)).fillna(0)
        
        # The Rule: Must be Growing
        results = results[
            (results["YOY Quarterly sales growth"] > MIN_SALES_GROWTH) & 
            (results["YOY Quarterly profit growth"] > MIN_PROFIT_GROWTH)
        ]

    # 3. Stage 2 Trend
    results = results[results["Price"] > results["MA_30"]]

    # 4. Near Highs
    results = results[results["Price"] >= (NEAR_HIGH_THRESHOLD * results["High_40"])]

    # --- 🚀 RANKING (Risk Adjusted) ---
    # Score = Momentum / Volatility (Higher is better)
    # We add a small number (1e-6) to volatility to avoid division by zero
    results["Score"] = results["Mom_20"] / (results["Vol_20"] + 1e-6)
    
    results = results.sort_values(by="Score", ascending=False)
    
    return results, market_breadth

# ==========================================
# 3️⃣ EXECUTION
# ==========================================
if __name__ == "__main__":
    df = load_all_weekly_data(DATA_FOLDER)
    final_list, breadth = apply_strategy(df)
    
    print("\n" + "="*60)
    print(f"🚀 TOP {TOP_N_RESULTS} QUALITY MOMENTUM STOCKS")
    print("="*60)
    
    display_cols = ["Price", "Mom_20", "Score", "YOY Quarterly sales growth", "YOY Quarterly profit growth"]
    # Clean up for display
    disp = final_list.copy()
    disp["Mom_20"] = (disp["Mom_20"]*100).round(1).astype(str) + "%"
    disp["Score"] = disp["Score"].round(2)
    disp["Price"] = disp["Price"].round(1)
    
    # Check if cols exist before printing
    valid_disp_cols = [c for c in display_cols if c in disp.columns]
    
    print(disp[valid_disp_cols].head(TOP_N_RESULTS).to_string())
    disp.head(TOP_N_RESULTS).to_excel("quality_momentum_results.xlsx")
    print("\n✅ Saved to quality_momentum_results.xlsx")