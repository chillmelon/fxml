from dataset.data import ForexData
from preprocess.feature import add_delta, add_technical_indicators
import pandas as pd

DATA_PATH = './data/compressed/usdjpy-bar-2020-01-01-2024-12-31.csv'
PROCESSED_PATH = './data/processed/usdjpy-20200101-20241231.csv'

def main():
    data = pd.read_csv(DATA_PATH)
    data = add_delta(data)
    data = add_technical_indicators(data)
    data.to_csv(PROCESSED_PATH)

if __name__ == '__main__':
    main()
