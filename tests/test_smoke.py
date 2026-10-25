import os
import json
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def test_credit_card_model_and_metadata_load():
    model_path = os.path.join(PROJECT_ROOT, 'credit_card_best_model.pkl')
    meta_path = os.path.join(PROJECT_ROOT, 'credit_card_metadata.json')
    assert os.path.exists(model_path), 'credit_card_best_model.pkl missing'
    assert os.path.exists(meta_path), 'credit_card_metadata.json missing'

    pipeline = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    assert 'feature_names' in metadata
    features = metadata['feature_names']
    # Create a dummy input using median-like values
    sample = {f: 0.0 for f in features}
    df = pd.DataFrame([sample])
    preds = pipeline.predict(df)
    assert preds.shape[0] == 1
def test_breast_cancer_model_and_metadata_load():
    model_path = os.path.join(PROJECT_ROOT, 'breast_cancer_best_model.pkl')
    meta_path = os.path.join(PROJECT_ROOT, 'breast_cancer_metadata.json')
    assert os.path.exists(model_path), 'breast_cancer_best_model.pkl missing'
    assert os.path.exists(meta_path), 'breast_cancer_metadata.json missing'

    pipeline = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    assert 'feature_names' in metadata
    features = metadata['feature_names']
    sample = {f: 0.0 for f in features}
    df = pd.DataFrame([sample])
    preds = pipeline.predict(df)
    assert preds.shape[0] == 1


def test_home_entrypoint_exists_and_dataset_apps_expose_main():
    home_path = os.path.join(PROJECT_ROOT, 'home.py')
    assert os.path.exists(home_path), 'home.py missing'

    import importlib.util

    card_spec = importlib.util.spec_from_file_location('credit_card_app_module', os.path.join(PROJECT_ROOT, 'credit_card_app.py'))
    bc_spec = importlib.util.spec_from_file_location('breast_cancer_app_module', os.path.join(PROJECT_ROOT, 'breast_cancer_app.py'))

    assert card_spec is not None and card_spec.loader is not None
    assert bc_spec is not None and bc_spec.loader is not None

    card_module = importlib.util.module_from_spec(card_spec)
    bc_module = importlib.util.module_from_spec(bc_spec)

    card_spec.loader.exec_module(card_module)
    bc_spec.loader.exec_module(bc_module)

    assert callable(getattr(card_module, 'main', None))
    assert callable(getattr(bc_module, 'main', None))


def test_breast_cancer_app_uses_credit_card_dashboard_page_names():
    app_path = os.path.join(PROJECT_ROOT, 'breast_cancer_app.py')
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()

    expected_labels = [
        'Overview',
        'Model Comparison',
        'Cross-Validation',
        'Diagnostics',
        'Hyperparameter Tuning',
        'Default Prediction',
    ]

    for label in expected_labels:
        assert label in content, f'Breast Cancer app missing dashboard label: {label}'


def test_project_data_directory_contains_dataset_files():
    data_dir = os.path.join(PROJECT_ROOT, 'datasets')
    assert os.path.isdir(data_dir), 'datasets directory is missing'

    expected_files = [
        'breast_cancer_dataset.csv',
        'breast_cancer_test_results.csv',
        'credit_card_dataset.csv',
        'credit_card_test_results.csv',
        'credit_card_before_preprocessing.csv',
        'breast_cancer_before_preprocessing.xlsx',
    ]

    for filename in expected_files:
        assert os.path.exists(os.path.join(data_dir, filename)), f'{filename} missing from datasets/'
 