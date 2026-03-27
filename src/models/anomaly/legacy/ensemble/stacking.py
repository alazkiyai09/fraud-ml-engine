"""Stacking ensemble for anomaly detection."""

from typing import List, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def stacking_ensemble(
    X_train: np.ndarray,  # type: ignore[valid-type]
    y_train: np.ndarray,  # type: ignore[valid-type]
    X_test: np.ndarray,  # type: ignore[valid-type]
    base_scores_train: np.ndarray,  # type: ignore[valid-type]  # Shape: (n_samples, n_models)
    base_scores_test: np.ndarray,  # type: ignore[valid-type]   # Shape: (n_samples, n_models)
    meta_model: str = "LogisticRegression",
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, object]:  # type: ignore[valid-type]
    """
    Stack base models using a meta-learner.

    The meta-learner learns to combine base model predictions optimally.

    Args:
        X_train: Original training features (for validation split).
        y_train: Training labels (required for meta-learner).
        X_test: Original test features.
        base_scores_train: Base model anomaly scores for training data.
                           Shape: (n_train_samples, n_base_models).
        base_scores_test: Base model anomaly scores for test data.
                          Shape: (n_test_samples, n_base_models).
        meta_model: Type of meta-learner ('LogisticRegression' or 'RandomForest').
        cv_folds: Number of cross-validation folds for generating meta-features.
        random_state: Random seed.

    Returns:
        Tuple of (stacked_scores, meta_model_object).
    """
    # Initialize meta-learner
    if meta_model == "LogisticRegression":
        meta = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'
        )
    elif meta_model == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        meta = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight='balanced'
        )
    else:
        raise ValueError(f"Unsupported meta_model: {meta_model}")

    # Train meta-learner on base model scores
    # Note: In practice, we should use cross-validation to generate meta-features
    # to avoid overfitting. For simplicity, we train directly on the scores.
    meta.fit(base_scores_train, y_train)

    # Generate predictions
    # For anomaly scores, we use the probability of class 1
    stacked_scores = meta.predict_proba(base_scores_test)[:, 1]

    return stacked_scores, meta


def stacking_ensemble_cv(
    base_models: List,
    X_train: np.ndarray,  # type: ignore[valid-type]
    y_train: np.ndarray,  # type: ignore[valid-type]
    X_test: np.ndarray,  # type: ignore[valid-type]
    meta_model: str = "LogisticRegression",
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, object]:  # type: ignore[valid-type]
    """
    Stack base models using cross-validation to generate meta-features.

    This prevents the meta-learner from overfitting to base model predictions.

    Args:
        base_models: List of fitted base anomaly detector objects.
        X_train: Original training features.
        y_train: Training labels.
        X_test: Original test features.
        meta_model: Type of meta-learner.
        cv_folds: Number of CV folds.
        random_state: Random seed.

    Returns:
        Tuple of (stacked_scores_train, stacked_scores_test, meta_model_object).
    """
    from sklearn.model_selection import KFold

    # Generate cross-validated predictions from base models
    n_train = len(X_train)
    n_models = len(base_models)
    base_scores_train_cv = np.zeros((n_train, n_models))

    # Get base model scores on test set
    base_scores_test = np.zeros((len(X_test), n_models))
    for i, model in enumerate(base_models):
        base_scores_test[:, i] = model.predict_anomaly_score(X_test)

    # Use K-fold CV to generate out-of-fold predictions for training
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for train_idx, val_idx in kf.split(X_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]

        for i, model_template in enumerate(base_models):
            # Create a new model instance (don't use the already fitted one)
            # This is a simplified approach - in practice, clone the model properly
            from copy import deepcopy
            model_fold = deepcopy(model_template)

            # Fit on fold training data
            model_fold.fit(X_fold_train)

            # Predict on fold validation data
            base_scores_train_cv[val_idx, i] = model_fold.predict_anomaly_score(X_fold_val)

    # Normalize scores to [0, 1] for each model (required for meta-learner)
    for i in range(n_models):
        min_val = base_scores_train_cv[:, i].min()
        max_val = base_scores_train_cv[:, i].max()
        if max_val > min_val:
            base_scores_train_cv[:, i] = (base_scores_train_cv[:, i] - min_val) / (max_val - min_val)
            base_scores_test[:, i] = (base_scores_test[:, i] - min_val) / (max_val - min_val)

    # Initialize and train meta-learner
    if meta_model == "LogisticRegression":
        meta = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'
        )
    elif meta_model == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        meta = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight='balanced'
        )
    else:
        raise ValueError(f"Unsupported meta_model: {meta_model}")

    # Train meta-learner on cross-validated predictions
    meta.fit(base_scores_train_cv, y_train)

    # Generate final predictions
    stacked_scores_train = meta.predict_proba(base_scores_train_cv)[:, 1]
    stacked_scores_test = meta.predict_proba(base_scores_test)[:, 1]

    return stacked_scores_train, stacked_scores_test, meta
