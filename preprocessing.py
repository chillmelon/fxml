from preprocess.feature import add_delta, add_direction, add_technical_indicators
import pandas as pd

DATA_PATH = './data/compressed/usdjpy-bar-2020-01-01-2024-12-31.csv'
PROCESSED_PATH = './data/processed/usdjpy-20200101-20241231.csv'

def main():
    data = pd.read_csv(DATA_PATH)
    data = add_delta(data)
    data = add_direction(data, delta_columns=['close'], threshold=0.002)
    data = add_technical_indicators(data)
    print(data.head)
    data.to_csv(PROCESSED_PATH)

if __name__ == '__main__':
    main()
