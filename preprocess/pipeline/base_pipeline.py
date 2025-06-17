import numpy as np
import pandas as pd
from tqdm import tqdm


class EventPipeline:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.close = data["close"]

    def get_t_events(self, threshold):
        # events timestamps extraction
        values = self.data.values
        timestamps = self.data.index

        s_pos = np.zeros_like(values)
        s_neg = np.zeros_like(values)
        t_events_mask = np.zeros_like(values, dtype=bool)

        cum_pos, cum_neg = 0.0, 0.0

        for i in tqdm(range(len(values))):
            cum_pos = max(0.0, cum_pos + values[i])
            cum_neg = min(0.0, cum_neg + values[i])
            s_pos[i] = cum_pos
            s_neg[i] = cum_neg

            if cum_pos > threshold:
                t_events_mask[i] = True
                cum_pos = 0.0
            if cum_neg < -threshold:
                t_events_mask[i] = True
                cum_pos = 0.0
        self.t_events = timestamps[t_events_mask]

    def get_trgt(self):
        pass

    def get_min_ret(self):
        pass

    def get_t1(self):
        pass

    def get_events(self):
        pass

    def get_bins(self):
        pass

    def run(self):
        pass

    def _get_daily_volatility(self, span0=100):
        # daily vol reindexed to close
        df0 = self.close.index.searchsorted(self.close.index - pd.Timedelta(days=1))
        # bp()
        df0 = df0[df0 > 0]
        # bp()
        df0 = pd.Series(
            self.close.index[df0 - 1],
            index=self.close.index[self.close.shape[0] - df0.shape[0] :],
        )
        # bp()
        try:
            df0 = (
                self.close.loc[df0.index] / self.close.loc[df0.values].values - 1
            )  # daily rets
        except Exception as e:
            print(e)
            print("adjusting shape of self.close.loc[df0.index]")
            cut = (
                self.close.loc[df0.index].shape[0] - self.close.loc[df0.values].shape[0]
            )
            df0 = (
                self.close.loc[df0.index].iloc[:-cut]
                / self.close.loc[df0.values].values
                - 1
            )
        df0 = df0.ewm(span=span0).std().rename("dailyVol")
        return df0
