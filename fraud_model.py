from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).resolve().with_name('new_data.csv')
MODEL_PATH = Path(__file__).resolve().with_name('fraud_model.joblib')


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
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


def train_and_save_model(path: str | Path = DATA_PATH, model_path: str | Path = MODEL_PATH) -> dict:
    df = load_dataset(path)
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


def ensure_model_exists(path: str | Path = DATA_PATH, model_path: str | Path = MODEL_PATH):
    if not Path(model_path).exists():
        return train_and_save_model(path, model_path)
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
