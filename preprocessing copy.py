from preprocess.feature import add_delta, add_direction, add_technical_indicators
import pandas as pd

DATA_PATH = './data/compressed/usdjpy-bar-2020-01-01-2024-12-31.csv'
PROCESSED_PATH = r'data\processed\usd-jpy-2024.csv'

def main():
    df = pd.read_csv(DATA_PATH)
    df['dt'] = pd.to_datetime(df['timestamp'], format='mixed')
    df_2024 = df[df['dt'].dt.year == 2024]
    df_2024.reset_index(drop=True, inplace=True)


    data = add_delta(df_2024)
    data = add_direction(data, delta_columns=['close'], threshold=5e-05)
    # data = add_technical_indicators(data)
    data.to_csv(PROCESSED_PATH)
    # data.to_pickle(PROCESSED_PATH)

if __name__ == '__main__':
    main()
