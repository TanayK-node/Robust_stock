import pandas as pd
import numpy as np
import glob
import re

# ===============================
# CONFIG
# ===============================
DATA_FOLDER = "weekly_data"
TOP_N = 10
INITIAL_CAPITAL = 100000

LOOKBACKS = [20, 8, 4, 2]

# Weight combinations to test
WEIGHT_SETS = [
    {20:0.4, 8:0.3, 4:0.2, 2:0.1},  # Current
    {20:0.6, 8:0.2, 4:0.1, 2:0.1},  # Long-term heavy
    {20:0.25,8:0.25,4:0.25,2:0.25}, # Equal
    {20:0.5, 8:0.3, 4:0.15,2:0.05}, # Suggested refined
    {20:0.3, 8:0.3, 4:0.2, 2:0.2}   # Short-term balanced
]

# ===============================
# DATA LOADING
# ===============================
def get_week_number(filename):
    match = re.search(r"Week\s*(\d+)", filename, re.IGNORECASE)
    return int(match.group(1)) if match else -1

def load_data(folder):
    files = glob.glob(f"{folder}/*.xlsx")
    all_data = []

    for f in files:
        df = pd.read_excel(f)
        df.columns = df.columns.str.strip()
        df = df[["Name", "Current Price"]]
        df["Week_Index"] = get_week_number(f)
        all_data.append(df)

    return pd.concat(all_data).sort_values("Week_Index")

# ===============================
# PERFORMANCE METRICS
# ===============================
def calculate_metrics(returns):

    returns = returns.dropna()
    if len(returns) < 10:
        return None

    cumulative = (1 + returns).cumprod()

    total_return = cumulative.iloc[-1] - 1
    years = len(returns) / 52
    cagr = (1 + total_return) ** (1/years) - 1

    sharpe = (returns.mean() / returns.std()) * np.sqrt(52)

    rolling_max = cumulative.cummax()
    drawdown = (cumulative / rolling_max) - 1
    max_dd = drawdown.min()

    return cagr, sharpe, max_dd

# ===============================
# BACKTEST ENGINE
# ===============================
def run_backtest(price_matrix, weights):

    portfolio_returns = []

    max_lb = max(LOOKBACKS)

    for i in range(max_lb + 1, len(price_matrix)-1):

        data_slice = price_matrix.iloc[:i]

        composite_score = pd.Series(0.0, index=data_slice.columns)

        for lb in LOOKBACKS:
            mom = data_slice.pct_change(lb).iloc[-1]
            vol = data_slice.pct_change(1).rolling(lb).std().iloc[-1]
            cv = mom / (vol + 1e-6)
            composite_score += cv * weights[lb]

        top_stocks = composite_score.sort_values(ascending=False).head(TOP_N).index

        next_week_prices = price_matrix.iloc[i+1][top_stocks]
        current_prices = price_matrix.iloc[i][top_stocks]

        weekly_return = (next_week_prices / current_prices - 1).mean()
        portfolio_returns.append(weekly_return)

    return pd.Series(portfolio_returns)

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    raw_data = load_data(DATA_FOLDER)

    price_matrix = raw_data.pivot_table(
        index="Week_Index",
        columns="Name",
        values="Current Price"
    ).ffill()

    results = []

    for weights in WEIGHT_SETS:

        print(f"\nTesting Weights: {weights}")

        returns = run_backtest(price_matrix, weights)

        metrics = calculate_metrics(returns)

        if metrics:
            cagr, sharpe, max_dd = metrics
            results.append({
                "Weights": str(weights),
                "CAGR": round(cagr,4),
                "Sharpe": round(sharpe,2),
                "Max_Drawdown": round(max_dd,4)
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Sharpe", ascending=False)

    print("\n==== WEIGHT VALIDATION RESULTS ====\n")
    print(results_df.to_string(index=False))

    results_df.to_excel("weight_validation_results.xlsx", index=False)
    print("\nSaved to weight_validation_results.xlsx")
