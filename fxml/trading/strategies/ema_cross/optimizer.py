from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fxml.trading.strategies.ema_cross.ema_cross import ema_cross

if __name__ == "__main__":

    data = pd.read_pickle("data/resampled/EURUSD-60m-20240101-20241231.pkl")
    # Define parameter ranges to test
    fast_range = range(2, 55, 2)  # 8, 10, 12, 14, 16, 18
    slow_range = range(24, 169, 2)  # 5, 7, 9, 11, 13

    # Store results
    results = []

    # Grid search optimization
    for fast, slow in product(fast_range, slow_range):
        # Skip invalid combinations where fast >= slow
        if fast >= slow:
            continue

        df_test = data.copy()

        df_test["signal"] = ema_cross(df_test, fast, slow)
        df_test["r"] = np.log(df_test["close"]).diff().shift(-1)
        df_test["strat_r"] = df_test["signal"] * df_test["r"]

        r = df_test["strat_r"].dropna()

        if len(r) > 0 and r[r < 0].abs().sum() != 0:
            profit_factor = r[r > 0].sum() / r[r < 0].abs().sum()
            sharpe_ratio = r.mean() / r.std() if r.std() != 0 else 0
            total_return = r.sum()
            win_rate = len(r[r > 0]) / len(r)

            results.append(
                {
                    "fast": fast,
                    "slow": slow,
                    "profit_factor": profit_factor,
                    "sharpe_ratio": sharpe_ratio,
                    "total_return": total_return,
                    "win_rate": win_rate,
                    "num_trades": len(r),
                }
            )

    results_df = pd.DataFrame(results)
    # Sort by Sharpe Ratio (or any metric you prefer)
    results_df = results_df.sort_values("sharpe_ratio", ascending=False)
    # Display top 10 parameter combinations
    print("Top 10 Parameter Combinations by Sharpe Ratio:")
    print(results_df.head(10).to_string(index=False))

    print("\n" + "=" * 80 + "\n")

    # Display top 10 by Profit Factor
    print("Top 10 Parameter Combinations by Profit Factor:")
    print(
        results_df.sort_values("profit_factor", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    # Get the best parameters
    best_params = results_df.iloc[0]
    print("\n" + "=" * 80 + "\n")
    print("Best Parameters (by Sharpe Ratio):")
    print(f"FAST: {int(best_params['fast'])}")
    print(f"SLOW: {int(best_params['slow'])}")
    print(f"Sharpe Ratio: {best_params['sharpe_ratio']:.4f}")
    print(f"Profit Factor: {best_params['profit_factor']:.4f}")
    print(f"Total Return: {best_params['total_return']:.4f}")
    print(f"Win Rate: {best_params['win_rate']:.2%}")

    # 2. Heatmap: Profit Factor vs Fast/Slow
    plt.figure(figsize=(25, 25))
    pivot_pf = results_df.pivot(index="slow", columns="fast", values="profit_factor")
    sns.heatmap(pivot_pf, annot=True, fmt=".3f", cmap="RdYlGn")
    plt.show()
