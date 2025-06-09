# Machine learning forex trading strategy

## Prequesties
- install uv ([installation guide](https://docs.astral.sh/uv/getting-started/installation/) )

## 📦 Setup
```bash
# Clone the project
git clone https://github.com/yourusername/fxml.git
cd fxml
# Create a virtual environment
uv venv .venv
source .venv/bin/activate
```
## Install dependencies (uses lockfile if present)
```uv pip install -r requirements.lock || uv pip install -r requirements.in```


## Run this project
1. Preprocessing
```python preprocessing.py```
1. Training
```python training.py```
1. Testing
```python testing.py```
1. Backtesting
```python backtesting.py```
1. Exploration
```jupyter notebook```
