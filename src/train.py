import pandas as pd
import numpy as np
import joblib
import os
import sys
sys.path.append('.')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.db import get_engine, get_training_data
from src.features import engineer_features, get_X_y


def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified split — preserves class ratio in train and test."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # critical for imbalanced data
    )


def build_pipelines():
    """
    Three pipelines — each is scaler + model.
    Putting scaler inside pipeline prevents data leakage in CV.
    """
    pipelines = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                class_weight='balanced',  # handles imbalance
                max_iter=1000,
                random_state=42
            ))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=200,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBClassifier(
                n_estimators=200,
                scale_pos_weight=5174/1869,  # handles imbalance
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            ))
        ])
    }
    return pipelines


def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    """Train, cross-validate, and evaluate one model."""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    # Cross-validation on training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv, scoring='roc_auc', n_jobs=-1
    )
    print(f"CV AUC-ROC:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Train on full training set
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)

    print(f"Test AUC-ROC: {test_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Not Churned', 'Churned']))

    return pipeline, test_auc, y_pred, y_prob


def plot_results(results, X_test, y_test):
    """Plot confusion matrices and ROC curves for all models."""
    n_models = len(results)
    fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))

    for i, (name, (pipeline, auc, y_pred, y_prob)) in enumerate(results.items()):
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm,
                display_labels=['Not Churned', 'Churned'])
        disp.plot(ax=axes[0, i], colorbar=False, cmap='Blues')
        axes[0, i].set_title(f'{name}\nAUC: {auc:.4f}',
                              fontweight='bold')

        # ROC curve
        RocCurveDisplay.from_predictions(
            y_test, y_prob, ax=axes[1, i],
            name=name, color='#D4537E'
        )
        axes[1, i].plot([0,1],[0,1],'--',color='gray',alpha=0.5)
        axes[1, i].set_title(f'ROC Curve — {name}', fontweight='bold')
        axes[1, i].grid(alpha=0.3)

    plt.suptitle('Model Comparison', fontsize=16,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('data/processed/model_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved to data/processed/model_comparison.png")


def save_best_model(results, X_train):
    """Save the best model pipeline and feature column order to disk."""
    best_name = max(results, key=lambda k: results[k][1])
    best_pipeline = results[best_name][0]
    best_auc = results[best_name][1]

    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(best_pipeline, 'models/best_model.pkl')
    
    # Save feature column order — critical for inference
    joblib.dump(list(X_train.columns), 'models/feature_columns.pkl')

    print(f"\n{'='*50}")
    print(f"BEST MODEL: {best_name}")
    print(f"Test AUC-ROC: {best_auc:.4f}")
    print(f"Saved to: models/best_model.pkl")
    print(f"Feature columns saved to: models/feature_columns.pkl")
    print(f"{'='*50}")

    return best_name, best_pipeline


def main():
    print("Loading data from PostgreSQL...")
    engine = get_engine()
    raw_df = get_training_data(engine)

    print("Engineering features...")
    df = engineer_features(raw_df)
    X, y = get_X_y(df)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Churn rate: {y.mean()*100:.1f}%")

    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")

    print("\nTraining and evaluating 3 models...")
    pipelines = build_pipelines()
    results = {}

    for name, pipeline in pipelines.items():
        pipeline, auc, y_pred, y_prob = evaluate_model(
            name, pipeline,
            X_train, X_test, y_train, y_test
        )
        results[name] = (pipeline, auc, y_pred, y_prob)

    plot_results(results, X_test, y_test)
    best_name, best_pipeline = save_best_model(results, X_train)

    print("\nPhase 5 complete!")
    print("Best model saved — ready for API serving in Phase 6.")


if __name__ == "__main__":
    main()