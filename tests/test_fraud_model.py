import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'fraud_model.py'


def test_model_module_exists():
    assert MODULE_PATH.exists(), 'fraud_model.py should exist for the website app'

    spec = importlib.util.spec_from_file_location('fraud_model', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, 'train_and_save_model')
    assert hasattr(module, 'predict_fraud')


def test_flask_app_exists():
    import app as flask_app_module

    assert hasattr(flask_app_module, 'app')
    client = flask_app_module.app.test_client()

    home = client.get('/')
    predict = client.get('/predict')

    assert home.status_code == 200
    assert predict.status_code == 200
