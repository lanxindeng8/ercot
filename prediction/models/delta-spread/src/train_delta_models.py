#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RTM-DAM Delta Model Training Script
=====================================
Train three models for arbitrage decision-making

Models:
1. Regression model: Predict specific spread values
2. Binary classification model: Predict spread direction (RTM > DAM?)
3. Multi-class classification model: Predict spread interval

Usage:
    python train_delta_models.py --input ../data/train_features.csv --output-dir ../models
"""

import argparse
import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from catboost import CatBoostRegressor, CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath: str) -> pd.DataFrame:
    """Load feature data"""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = pd.to_datetime(df['date'])
    return df


def prepare_features_targets(df: pd.DataFrame) -> dict:
    """
    Prepare features and target variables

    Returns:
        dict with X, y_reg, y_bin, y_multi
    """
    # Feature columns (exclude time and target columns)
    exclude_cols = [
        'timestamp', 'date',
        'target_spread', 'target_spread_last', 'target_direction', 'target_class',
        'naive_pred'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].copy()
    y_reg = df['target_spread'].copy()
    y_bin = df['target_direction'].copy()
    y_multi = df['target_class'].copy()

    # Record naive prediction baseline
    naive_pred = df['spread_same_hour_hist'].copy()

    return {
        'X': X,
        'y_reg': y_reg,
        'y_bin': y_bin,
        'y_multi': y_multi,
        'feature_cols': feature_cols,
        'naive_pred': naive_pred,
        'timestamps': df['timestamp']
    }


def split_train_test(data: dict, test_ratio: float = 0.2) -> dict:
    """
    Time-series split into training/test sets
    """
    n = len(data['X'])
    train_size = int(n * (1 - test_ratio))

    result = {
        'X_train': data['X'].iloc[:train_size],
        'X_test': data['X'].iloc[train_size:],
        'y_reg_train': data['y_reg'].iloc[:train_size],
        'y_reg_test': data['y_reg'].iloc[train_size:],
        'y_bin_train': data['y_bin'].iloc[:train_size],
        'y_bin_test': data['y_bin'].iloc[train_size:],
        'y_multi_train': data['y_multi'].iloc[:train_size],
        'y_multi_test': data['y_multi'].iloc[train_size:],
        'naive_train': data['naive_pred'].iloc[:train_size],
        'naive_test': data['naive_pred'].iloc[train_size:],
        'timestamps_test': data['timestamps'].iloc[train_size:],
        'feature_cols': data['feature_cols']
    }

    return result


def train_regression_model(X_train, y_train, X_test, y_test, naive_test) -> dict:
    """
    Train regression model
    """
    print("\n" + "=" * 50)
    print("Training Regression Model (Predict Spread Value)")
    print("=" * 50)

    # Define categorical features
    cat_features = ['target_hour', 'target_dow', 'target_month', 'target_is_weekend',
                    'target_is_peak', 'target_is_summer', 'target_day_of_month', 'target_week']
    cat_indices = [i for i, col in enumerate(X_train.columns) if col in cat_features]

    model = CatBoostRegressor(
        iterations=1000,
        depth=7,
        learning_rate=0.05,
        loss_function='MAE',
        early_stopping_rounds=100,
        verbose=100,
        random_seed=42,
        cat_features=cat_indices
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True
    )

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Compare with naive baseline
    naive_mae = mean_absolute_error(y_test, naive_test)
    improvement = (naive_mae - mae) / naive_mae * 100

    print(f"\nModel performance:")
    print(f"  MAE: ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"\nBaseline comparison:")
    print(f"  Naive MAE: ${naive_mae:.2f}")
    print(f"  Relative improvement: {improvement:.1f}%")

    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'model': model,
        'predictions': y_pred,
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'naive_mae': naive_mae,
            'improvement': improvement
        },
        'feature_importance': importance
    }


def train_binary_model(X_train, y_train, X_test, y_test) -> dict:
    """
    Train binary classification model (Predict Spread Direction)
    """
    print("\n" + "=" * 50)
    print("Training Binary Classification Model (Predict Spread Direction)")
    print("=" * 50)

    cat_features = ['target_hour', 'target_dow', 'target_month', 'target_is_weekend',
                    'target_is_peak', 'target_is_summer', 'target_day_of_month', 'target_week']
    cat_indices = [i for i, col in enumerate(X_train.columns) if col in cat_features]

    model = CatBoostClassifier(
        iterations=1000,
        depth=7,
        learning_rate=0.05,
        loss_function='Logloss',
        early_stopping_rounds=100,
        verbose=100,
        random_seed=42,
        cat_features=cat_indices,
        auto_class_weights='Balanced'  # Handle class imbalance
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True
    )

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Baseline: predict majority class
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())

    print(f"\nModel performance:")
    print(f"  Accuracy: {accuracy*100:.1f}%")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Recall: {recall*100:.1f}%")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"\nBaseline comparison:")
    print(f"  Majority class accuracy: {baseline_acc*100:.1f}%")
    print(f"\nConfusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  [[TN={cm[0,0]:5d}, FP={cm[0,1]:5d}]")
    print(f"   [FN={cm[1,0]:5d}, TP={cm[1,1]:5d}]]")

    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_prob,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'baseline_accuracy': baseline_acc
        },
        'confusion_matrix': cm,
        'feature_importance': importance
    }


def train_multiclass_model(X_train, y_train, X_test, y_test) -> dict:
    """
    Train multi-class classification model (Predict Spread Interval)
    """
    print("\n" + "=" * 50)
    print("Training Multi-class Classification Model (Predict Spread Interval)")
    print("=" * 50)

    cat_features = ['target_hour', 'target_dow', 'target_month', 'target_is_weekend',
                    'target_is_peak', 'target_is_summer', 'target_day_of_month', 'target_week']
    cat_indices = [i for i, col in enumerate(X_train.columns) if col in cat_features]

    model = CatBoostClassifier(
        iterations=1000,
        depth=7,
        learning_rate=0.05,
        loss_function='MultiClass',
        early_stopping_rounds=100,
        verbose=100,
        random_seed=42,
        cat_features=cat_indices,
        auto_class_weights='Balanced'
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True
    )

    # Predict
    y_pred = model.predict(X_test).flatten()
    y_prob = model.predict_proba(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    # Baseline: predict most frequent class
    most_common = y_train.value_counts().idxmax()
    baseline_acc = (y_test == most_common).mean()

    print(f"\nModel performance:")
    print(f"  Accuracy: {accuracy*100:.1f}%")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(f"  Weighted F1: {f1_weighted:.4f}")
    print(f"\nBaseline comparison:")
    print(f"  Most frequent class accuracy: {baseline_acc*100:.1f}%")

    print(f"\nClassification report:")
    class_labels = ['< -$20', '-$20~-$5', '-$5~$5', '$5~$20', '>= $20']
    print(classification_report(y_test, y_pred, target_names=class_labels))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)

    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_prob,
        'metrics': {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'baseline_accuracy': baseline_acc
        },
        'confusion_matrix': cm,
        'feature_importance': importance
    }


def save_results(
    output_dir: str,
    reg_result: dict,
    bin_result: dict,
    multi_result: dict,
    split_data: dict
) -> None:
    """Save models and results"""
    os.makedirs(output_dir, exist_ok=True)

    # Save models
    reg_result['model'].save_model(os.path.join(output_dir, 'regression_model.cbm'))
    bin_result['model'].save_model(os.path.join(output_dir, 'binary_model.cbm'))
    multi_result['model'].save_model(os.path.join(output_dir, 'multiclass_model.cbm'))

    print(f"\nModels saved to: {output_dir}")

    # Save evaluation results
    results_summary = {
        'regression': reg_result['metrics'],
        'binary': bin_result['metrics'],
        'multiclass': multi_result['metrics'],
        'training_samples': len(split_data['X_train']),
        'test_samples': len(split_data['X_test']),
        'feature_count': len(split_data['feature_cols']),
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    # Save feature importance
    reg_result['feature_importance'].to_csv(
        os.path.join(output_dir, 'feature_importance_regression.csv'), index=False)
    bin_result['feature_importance'].to_csv(
        os.path.join(output_dir, 'feature_importance_binary.csv'), index=False)
    multi_result['feature_importance'].to_csv(
        os.path.join(output_dir, 'feature_importance_multiclass.csv'), index=False)

    # Save prediction results (for backtesting)
    predictions_df = pd.DataFrame({
        'timestamp': split_data['timestamps_test'].values,
        'actual_spread': split_data['y_reg_test'].values,
        'pred_spread': reg_result['predictions'],
        'actual_direction': split_data['y_bin_test'].values,
        'pred_direction': bin_result['predictions'],
        'pred_prob': bin_result['probabilities'],
        'actual_class': split_data['y_multi_test'].values,
        'pred_class': multi_result['predictions']
    })
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

    print(f"Results saved")


def train_all_models(
    input_path: str,
    output_dir: str,
    test_ratio: float = 0.2
) -> None:
    """
    Main function: Train all models
    """
    print("=" * 60)
    print("RTM-DAM Delta Model Training")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading data...")
    df = load_data(input_path)
    print(f"   Total samples: {len(df):,}")

    # 2. Prepare features and targets
    print("\n2. Preparing features and target variables...")
    data = prepare_features_targets(df)
    print(f"   Feature count: {len(data['feature_cols'])}")

    # 3. Split dataset
    print("\n3. Splitting training/test sets...")
    split_data = split_train_test(data, test_ratio)
    print(f"   Training set: {len(split_data['X_train']):,}")
    print(f"   Test set: {len(split_data['X_test']):,}")

    # 4. Train regression model
    reg_result = train_regression_model(
        split_data['X_train'], split_data['y_reg_train'],
        split_data['X_test'], split_data['y_reg_test'],
        split_data['naive_test']
    )

    # 5. Train binary classification model
    bin_result = train_binary_model(
        split_data['X_train'], split_data['y_bin_train'],
        split_data['X_test'], split_data['y_bin_test']
    )

    # 6. Train multi-class classification model
    multi_result = train_multiclass_model(
        split_data['X_train'], split_data['y_multi_train'],
        split_data['X_test'], split_data['y_multi_test']
    )

    # 7. Save results
    print("\n" + "=" * 50)
    print("Saving Models and Results")
    print("=" * 50)
    save_results(output_dir, reg_result, bin_result, multi_result, split_data)

    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete - Results Summary")
    print("=" * 60)
    print(f"\nRegression model:")
    print(f"  MAE: ${reg_result['metrics']['mae']:.2f} (baseline: ${reg_result['metrics']['naive_mae']:.2f})")
    print(f"  Improvement: {reg_result['metrics']['improvement']:.1f}%")
    print(f"\nBinary classification model:")
    print(f"  Accuracy: {bin_result['metrics']['accuracy']*100:.1f}%")
    print(f"  AUC: {bin_result['metrics']['auc']:.4f}")
    print(f"\nMulti-class classification model:")
    print(f"  Accuracy: {multi_result['metrics']['accuracy']*100:.1f}%")
    print(f"  Macro F1: {multi_result['metrics']['f1_macro']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Delta prediction models')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input feature CSV')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                        help='Output directory for models')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                        help='Test set ratio (default: 0.2)')
    args = parser.parse_args()

    train_all_models(
        input_path=args.input,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio
    )


if __name__ == "__main__":
    main()
