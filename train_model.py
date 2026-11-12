"""
Script for training and saving the logistic regression pipeline
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from model import MODEL_PATH, RAW_FEATURES, TARGET, build_pipeline, save_model


def load_data(path: str = 'train.csv'):
    df = pd.read_csv(path)
    return df[RAW_FEATURES], df[TARGET]


def best_threshold(y_true, proba) -> float:
    """Threshold on P(high risk) that maximizes F1 for the high-risk class."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    scores = [f1_score(y_true, proba >= t) for t in thresholds]
    return float(round(thresholds[int(np.argmax(scores))], 2))


def evaluate(y_true, proba, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, proba),
    }


def train_and_save_model(data_path: str = 'train.csv', model_path: str = MODEL_PATH) -> dict:
    X, y = load_data(data_path)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline().fit(X_train, y_train)
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    threshold = best_threshold(y_val, val_proba)
    metrics = evaluate(y_val, val_proba, threshold)

    print(f"Validation Results (threshold = {threshold:.2f}):")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"{name:<10} {value:.4f}")
    print("=" * 50)

    # Refit on all labeled data; the threshold chosen on validation carries over
    final_pipeline = build_pipeline().fit(X, y)
    model_data = {'pipeline': final_pipeline, 'threshold': threshold, 'metrics': metrics}
    save_model(model_data, model_path)
    print(f"\nModel saved to {model_path}")
    return model_data


if __name__ == '__main__':
    train_and_save_model()
