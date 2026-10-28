# House Price Prediction By SRM

Backend for AI Applications Lab (26ECP-102). This project compares Linear Regression and Random Forest Regressor using Kaggle's **House Prices: Advanced Regression Techniques** dataset (`train.csv`) and predicts `SalePrice`.

## 1. Project files

- `train_model.py` — EDA summary, preprocessing pipelines, model training/evaluation, plots, and saved model.
- `app.py` — Flask API with `POST /predict`, `GET /metrics`, and `GET /health`.
- `requirements.txt` — Python dependencies.
- `outputs/` — generated model, metrics JSON, neighborhood list, and charts.
- `data/` — optional place to keep the dataset.

## 2. Dataset setup

Download `train.csv` from Kaggle's **House Prices: Advanced Regression Techniques** competition. Place it either at `house_price_prediction_srm/train.csv` or `house_price_prediction_srm/data/train.csv`. Do not use Kaggle's `test.csv`: it does not contain `SalePrice`.

## 3. Install and train

Run these commands from inside the `house_price_prediction_srm` folder.

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
```

The training script uses an 80/20 split with `random_state=42`, trains on `log1p(SalePrice)`, and reports MAE/RMSE in original SalePrice dollar units. It selects the best model using the lowest held-out RMSE; actual metrics are generated from the dataset at training time.

## 4. Start the API

After successful training, open a terminal in this folder, activate the same virtual environment, and run:

```bash
python app.py
```

The API starts at `http://127.0.0.1:5000`. Open `http://127.0.0.1:5000/health` to check readiness and `http://127.0.0.1:5000/metrics` to view the model comparison.

## 5. Example prediction request

Send a `POST` request to `http://127.0.0.1:5000/predict` with `Content-Type: application/json`. Use a `Neighborhood` string exactly as returned by `GET /metrics`.

```json
{
  "GrLivArea": 1500,
  "TotRmsAbvGrd": 7,
  "BedroomAbvGr": 3,
  "Neighborhood": "CollgCr",
  "YearBuilt": 2005,
  "OverallQual": 7,
  "GarageCars": 2,
  "TotalBsmtSF": 900,
  "FullBath": 2
}
```

This JSON is only an example input, not a claim about the prediction. The returned price comes from the trained model.

## 6. Generated outputs

After `python train_model.py`, the `outputs/` directory contains:

- `best_model.joblib`
- `metrics.json`
- `neighborhoods.json`
- `price_distribution.png`
- `correlation_heatmap.png`
- `actual_vs_predicted.png`
- `feature_importance.png`
- `feature_importance.csv`

If the API reports that the model or metrics are missing, train first and restart `app.py`.
