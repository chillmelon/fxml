import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fxml.trading.strategies.macd_cross.macd_cross import macd_cross_strategy


def main():
    data = pd.read_pickle("data/resampled/EURUSD-60m-20240101-20241231.pkl")
    data["timestamp"] = data["timestamp"].astype("datetime64[s]")
    data = data.set_index("timestamp")
    data = data.dropna()

    ## Comprehensive Parameter Sweep
    fast_periods = list(range(5, 21, 2))
    slow_periods = list(range(20, 50, 2))
    signal_periods = list(range(6, 15, 2))

    results = []

    print("Running parameter optimization...")
    print(f"Fast periods: {fast_periods}")
    print(f"Slow periods: {slow_periods}")
    print(f"Signal periods: {signal_periods}")
    print(
        f"Total combinations: {len(fast_periods) * len(slow_periods) * len(signal_periods)}\n"
    )

    for fp in fast_periods:
        for sp in slow_periods:
            for sigp in signal_periods:
                # Skip invalid combinations (fast must be < slow)
                if fp >= sp:
                    continue

                macd_line, signal_line, histogram, signal = macd_cross_strategy(
                    data["close"].to_numpy(), fp, sp, sigp
                )
                data["signal"] = signal

                data["r"] = np.log(data["close"]).diff().shift(-1)
                strat_r = data["signal"] * data["r"]

                # Calculate metrics
                pf = strat_r[strat_r > 0].sum() / strat_r[strat_r < 0].abs().sum()
                total_return = strat_r.sum()
                sharpe = (
                    strat_r.mean() / strat_r.std() * np.sqrt(252 * 24)
                )  # Hourly data
                win_rate = (strat_r > 0).sum() / (strat_r != 0).sum()

                results.append(
                    {
                        "fast": fp,
                        "slow": sp,
                        "signal": sigp,
                        "profit_factor": pf,
                        "total_return": total_return,
                        "sharpe_ratio": sharpe,
                        "win_rate": win_rate,
                    }
                )

                print(
                    f"({fp:2d}, {sp:2d}, {sigp:2d}) | PF: {pf:6.4f} | Ret: {total_return:7.4f} | "
                    f"Sharpe: {sharpe:6.2f} | WR: {win_rate:5.2%}"
                )

    results_df = pd.DataFrame(results)

    # Find best parameters
    best_pf = results_df.loc[results_df["profit_factor"].idxmax()]
    best_sharpe = results_df.loc[results_df["sharpe_ratio"].idxmax()]
    best_return = results_df.loc[results_df["total_return"].idxmax()]

    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"\nBest Profit Factor: {best_pf['profit_factor']:.4f}")
    print(
        f"  Parameters: Fast={best_pf['fast']:.0f}, Slow={best_pf['slow']:.0f}, Signal={best_pf['signal']:.0f}"
    )
    print(
        f"  Return: {best_pf['total_return']:.4f}, Sharpe: {best_pf['sharpe_ratio']:.2f}"
    )

    print(f"\nBest Sharpe Ratio: {best_sharpe['sharpe_ratio']:.2f}")
    print(
        f"  Parameters: Fast={best_sharpe['fast']:.0f}, Slow={best_sharpe['slow']:.0f}, Signal={best_sharpe['signal']:.0f}"
    )
    print(
        f"  PF: {best_sharpe['profit_factor']:.4f}, Return: {best_sharpe['total_return']:.4f}"
    )

    print(f"\nBest Total Return: {best_return['total_return']:.4f}")
    print(
        f"  Parameters: Fast={best_return['fast']:.0f}, Slow={best_return['slow']:.0f}, Signal={best_return['signal']:.0f}"
    )
    print(
        f"  PF: {best_return['profit_factor']:.4f}, Sharpe: {best_return['sharpe_ratio']:.2f}"
    )
    print("=" * 80)

    # Save results
    results_df.to_csv("macd_optimization_results.csv", index=False)
    print(f"\nResults saved to macd_optimization_results.csv")

    ## Visualizations
    plt.style.use("dark_background")

    # 1. Heatmap: Fast vs Slow (averaged over signal periods)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for sigp in signal_periods[:4]:  # Show first 4 signal periods
        ax_idx = signal_periods.index(sigp)
        if ax_idx >= 4:
            break
        ax = axes[ax_idx // 2, ax_idx % 2]

        pivot_data = results_df[results_df["signal"] == sigp].pivot(
            index="slow", columns="fast", values="profit_factor"
        )

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=1.0,
            ax=ax,
            cbar_kws={"label": "Profit Factor"},
        )
        ax.set_title(f"Signal Period = {sigp}")
        ax.set_xlabel("Fast Period")
        ax.set_ylabel("Slow Period")

    plt.suptitle("MACD Optimization: Profit Factor Heatmaps", fontsize=16, y=1.00)
    plt.tight_layout()
    plt.show()

    # 2. 3D scatter plot
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        results_df["fast"],
        results_df["slow"],
        results_df["signal"],
        c=results_df["profit_factor"],
        cmap="RdYlGn",
        s=100,
        alpha=0.6,
    )

    ax.set_xlabel("Fast Period")
    ax.set_ylabel("Slow Period")
    ax.set_zlabel("Signal Period")
    ax.set_title("MACD Parameter Space (colored by Profit Factor)")
    fig.colorbar(scatter, ax=ax, label="Profit Factor", shrink=0.5)
    plt.show()

    # 3. Metric comparison for top 10 strategies
    top10 = results_df.nlargest(10, "profit_factor")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Profit Factor
    ax = axes[0, 0]
    top10.plot(
        x="fast",
        y="profit_factor",
        kind="bar",
        ax=ax,
        color="cyan",
        legend=False,
    )
    ax.set_title("Top 10: Profit Factor")
    ax.set_ylabel("Profit Factor")
    ax.axhline(1.0, color="white", linestyle="--", alpha=0.5)
    ax.set_xticklabels(
        [f"({r.fast},{r.slow},{r.signal})" for _, r in top10.iterrows()], rotation=45
    )

    # Sharpe Ratio
    ax = axes[0, 1]
    top10.plot(
        x="fast", y="sharpe_ratio", kind="bar", ax=ax, color="orange", legend=False
    )
    ax.set_title("Top 10: Sharpe Ratio")
    ax.set_ylabel("Sharpe Ratio")
    ax.axhline(0, color="white", linestyle="--", alpha=0.5)
    ax.set_xticklabels(
        [f"({r.fast},{r.slow},{r.signal})" for _, r in top10.iterrows()], rotation=45
    )

    # Win Rate
    ax = axes[1, 0]
    top10.plot(x="fast", y="win_rate", kind="bar", ax=ax, color="green", legend=False)
    ax.set_title("Top 10: Win Rate")
    ax.set_ylabel("Win Rate")
    ax.axhline(0.5, color="white", linestyle="--", alpha=0.5)
    ax.set_xticklabels(
        [f"({r.fast},{r.slow},{r.signal})" for _, r in top10.iterrows()], rotation=45
    )

    # 4. Parameter distribution for profitable strategies
    profitable = results_df[results_df["profit_factor"] > 1.0]

    if len(profitable) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].hist(
            profitable["fast"], bins=len(fast_periods), color="cyan", alpha=0.7
        )
        axes[0].set_xlabel("Fast Period")
        axes[0].set_ylabel("Count")
        axes[0].set_title(
            f"Profitable Strategies: Fast Period Distribution (n={len(profitable)})"
        )

        axes[1].hist(
            profitable["slow"], bins=len(slow_periods), color="orange", alpha=0.7
        )
        axes[1].set_xlabel("Slow Period")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Profitable Strategies: Slow Period Distribution")

        axes[2].hist(
            profitable["signal"], bins=len(signal_periods), color="green", alpha=0.7
        )
        axes[2].set_xlabel("Signal Period")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Profitable Strategies: Signal Period Distribution")

        plt.tight_layout()
        plt.show()
    else:
        print("\nNo profitable strategies found (PF > 1.0)")


if __name__ == "__main__":
    main()
