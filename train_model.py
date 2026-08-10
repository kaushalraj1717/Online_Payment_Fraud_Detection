from fraud_model import ensure_model_exists


if __name__ == '__main__':
    result = ensure_model_exists()
    print(f"Model ready: {result['model_path']} | accuracy={result['accuracy']:.4f}")
