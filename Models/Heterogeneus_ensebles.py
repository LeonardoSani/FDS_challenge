import pandas as pd
import numpy as np

# --- Preprocessing & Pipeline ---
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- Base Models ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# --- Ensemble Methods ---
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

# --- Model Selection & Evaluation ---
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV


def create_base_models(random_state=0):
    rf_pipe = Pipeline([
        ('scaler', 'passthrough'),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1))
    ])

    lgbm_pipe = Pipeline([
        ('scaler', 'passthrough'),
        ('lgbm', LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1))
    ])

    xgb_pipe = Pipeline([
        ('scaler', 'passthrough'),
        ('xgb', XGBClassifier(random_state=random_state, n_jobs=-1,
                              eval_metric='logloss', use_label_encoder=False))
    ])

    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=random_state))
    ])

    return [
        ('rf', rf_pipe),
        ('lgbm', lgbm_pipe),
        ('xgb', xgb_pipe),
        ('svm', svm_pipe)
    ]


def create_voting_classifier(base_models, voting_type='soft'):
    """
    Creates a VotingClassifier.

    Args:
        base_models (list): List of (name, model) tuples from create_base_models.
        voting_type (str): 'soft' (uses predicted probabilities) or 
                           'hard' (uses majority vote).

    Returns:
        sklearn.ensemble.VotingClassifier: The voting classifier.
    """
    
    print(f"Creating VotingClassifier (type='{voting_type}')...")
    
    voting_clf = VotingClassifier(
        estimators=base_models,
        voting=voting_type,
        n_jobs=-1
    )
    
    return voting_clf


def create_stacking_classifier(base_models, meta_model=None, cv_splits=5, random_state=0):
    """
    Creates a StackingClassifier.

    Args:
        base_models (list): List of (name, model) tuples from create_base_models.
        meta_model (estimator, optional): The meta-model (L1). 
                                         Defaults to LogisticRegression as requested.
        cv_splits (int): Number of folds for the internal CV used 
                         to train the meta-model.
        random_state (int): Seed for reproducibility.

    Returns:
        sklearn.ensemble.StackingClassifier: The stacking classifier.
    """
    
    # Default to LogisticRegression as the meta-model (final_estimator)
    if meta_model is None:
        meta_model = LogisticRegression(C=1.0, random_state=random_state)
        
    print(f"Creating StackingClassifier with meta-model: {meta_model.__class__.__name__}")
    
    # Define the CV scheme for training the meta-model
    stacking_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=stacking_cv,
        n_jobs=-1,
        passthrough=False # False = meta-model trains only on base-model predictions
                          # True = meta-model trains on predictions + original features
    )
    
    return stacking_clf
    """
    Performs a grid search to find the best hyperparameters for the pipeline.

    Args:
        pipeline (sklearn.pipeline.Pipeline): The base pipeline to tune.
        param_grid (dict): The dictionary of parameters to search.
        X_train (pd.DataFrame): The training features.
        Y_train (pd.Series or np.array): The training target.
        cv_splits (int): Number of cross-validation splits.

    Returns:
        sklearn.pipeline.Pipeline: The best-performing pipeline found.
    """
    
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
    
    print("Starting Grid Search...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=1  # Set to 1 or 2 to see search progress
    )
    
    grid_search.fit(X_train, Y_train)
    
    best_index = grid_search.best_index_
    best_std = grid_search.cv_results_['std_test_score'][best_index]

    print("\nGrid Search Complete.")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.8f} "
          f"(+/- {best_std:.8f})")
    
    return grid_search.best_estimator_