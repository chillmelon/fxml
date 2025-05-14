# Machine learning forex trading strategy

## Prequesties
- install conda

## How to start
1. clone this project
1. put data in `PROJECT_DIR/data/`
1. create conda environment
  ```
  conda env create -f environment.yml
  conda activate fxml
  ```

1. run the scripts
  - preprocessing: `preprocessing.py`
  - training: `train.py`
  - testing: `test.py`
  - backtesting: `backtest.py`
  - mt5 trading: `trade.py`

1. exploration
  - run `jupyter notebook`
  - go to `notebooks` folder
  - open the notebooks to explore
