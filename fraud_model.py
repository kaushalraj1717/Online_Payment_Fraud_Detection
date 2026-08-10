from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).resolve().with_name('new_data.csv')
MODEL_PATH = Path(__file__).resolve().with_name('fraud_model.joblib')
DEFAULT_DATASET_URL = os.environ.get('DATASET_URL')


def download_dataset(url: str, path: str | Path = DATA_PATH) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    return str(destination)


def build_synthetic_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(500):
        step = int(rng.integers(1, 31))
        transaction_type = rng.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])
        amount = float(rng.uniform(10, 20000))
        old_balance_org = float(rng.uniform(100, 100000))
        new_balance_orig = old_balance_org - amount if transaction_type in {'PAYMENT', 'CASH_OUT'} else float(rng.uniform(0, 100000))
        old_balance_dest = float(rng.uniform(0, 100000))
        new_balance_dest = old_balance_dest + amount if transaction_type in {'TRANSFER', 'CASH_IN'} else float(rng.uniform(0, 100000))
        is_fraud = 1 if transaction_type in {'TRANSFER', 'CASH_OUT'} and amount > 500 and rng.random() < 0.35 else 0
        rows.append({
            'step': step,
            'type': transaction_type,
            'amount': amount,
            'oldbalanceOrg': old_balance_org,
            'newbalanceOrig': new_balance_orig,
            'oldbalanceDest': old_balance_dest,
            'newbalanceDest': new_balance_dest,
            'isFraud': is_fraud,
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def ensure_dataset_exists(path: str | Path = DATA_PATH, dataset_url: str | None = None) -> str:
    data_path = Path(path)
    if data_path.exists():
        return str(data_path)

    url = dataset_url or DEFAULT_DATASET_URL
    if url:
        return download_dataset(url, data_path)

    build_synthetic_dataset(data_path)
    return str(data_path)


def load_dataset(path: str | Path = DATA_PATH, dataset_url: str | None = None) -> pd.DataFrame:
    resolved_path = ensure_dataset_exists(path, dataset_url)
    df = pd.read_csv(resolved_path)
    required_cols = [
        'step',
        'type',
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest',
        'isFraud',
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    return df[required_cols].copy()


def build_pipeline() -> Pipeline:
    categorical_cols = ['type']
    numerical_cols = [
        'step',
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest',
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ]
    )

    model = HistGradientBoostingClassifier(
        max_depth=8,
        max_iter=200,
        random_state=42,
    )

    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])


def train_and_save_model(path: str | Path = DATA_PATH, model_path: str | Path = MODEL_PATH, dataset_url: str | None = None) -> dict:
    df = load_dataset(path, dataset_url=dataset_url)
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)

    joblib.dump({'pipeline': pipeline, 'accuracy': accuracy}, model_path)
    return {'accuracy': float(accuracy), 'model_path': str(model_path)}


def ensure_model_exists(path: str | Path = DATA_PATH, model_path: str | Path = MODEL_PATH, dataset_url: str | None = None):
    if not Path(model_path).exists():
        return train_and_save_model(path, model_path, dataset_url=dataset_url)
    bundle = joblib.load(model_path)
    return {'accuracy': float(bundle.get('accuracy', 0.0)), 'model_path': str(model_path)}


def predict_fraud(data: dict | pd.DataFrame):
    model_path = MODEL_PATH
    if not model_path.exists():
        ensure_model_exists()

    bundle = joblib.load(model_path)
    pipeline = bundle['pipeline']

    if isinstance(data, dict):
        row = pd.DataFrame([data], columns=[
            'step',
            'type',
            'amount',
            'oldbalanceOrg',
            'newbalanceOrig',
            'oldbalanceDest',
            'newbalanceDest',
        ])
    else:
        row = data.copy()

    probability = float(pipeline.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)
    return prediction, probability
