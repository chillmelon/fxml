import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from fxml.trading.strategies.bar_permute import get_permutation
from fxml.trading.strategies.trendline_breakout.trendline_breakout import (
    optimize_trendline,
)

data = pd.read_pickle("data/resampled/EURUSD-15m-20240101-20241231.pkl")
data["timestamp"] = data["timestamp"].astype("datetime64[s]")
data = data.set_index("timestamp")

print(data.head())

train_data = data

best_lookback, best_real_pf = optimize_trendline(train_data)
print("In-sample PF", best_real_pf, "Best Lookback", best_lookback)

n_permutations = 1000
perm_better_count = 1
permuted_pfs = []
print("In-Sample MCPT")
for perm_i in tqdm(range(1, n_permutations)):
    train_perm = get_permutation(train_data)
    best_perm_lookback, best_perm_pf = optimize_trendline(train_perm)

    if best_perm_pf >= best_real_pf:
        perm_better_count += 1

    permuted_pfs.append(best_perm_pf)

insample_mcpt_pval = perm_better_count / n_permutations
print(f"In-sample MCPT P-Value: {insample_mcpt_pval}")

plt.style.use("dark_background")
pd.Series(permuted_pfs).hist(color="blue", label="Permutations")
plt.axvline(best_real_pf, color="red", label="Real")
plt.xlabel("Profit Factor")
plt.title(f"In-sample MCPT. P-Value: {insample_mcpt_pval}")
plt.grid(False)
plt.legend()
plt.show()
