import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def train_and_predict(pipeline, X_train, Y_train, X_test, test_battle_ids):
    """
    Trains the pipeline and generates predictions for the test set.

    Args:
        pipeline (sklearn.pipeline.Pipeline): The model pipeline to train.
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series or np.array): The training target.
        X_test (pd.DataFrame): The test features.
        test_battle_ids (pd.Series): The battle_ids from the test set.

    Returns:
        pd.DataFrame: A DataFrame formatted for submission.
    """
    print("Training the pipeline...")
    pipeline.fit(X_train, Y_train)
    print("Training complete.")
    
    print("Making predictions...")
    Y_pred = pipeline.predict(X_test)
    print("Predictions complete.")
    submission = pd.DataFrame({
        'battle_id': test_battle_ids,
        'player_won': Y_pred
    })
    
    return submission


def perform_grid_search(pipeline, param_grid, X_train, Y_train, cv_splits=5):
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
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, Y_train)
    
    best_index = grid_search.best_index_
    best_std = grid_search.cv_results_['std_test_score'][best_index]

    print("\nGrid Search Complete.")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.8f}")
    print(f"Best CV accuracy std dev: {best_std:.8f}")
    
    
    return grid_search.best_estimator_


def evaluate_model(pipeline, X_train, Y_train, cv_splits=5):
    """
    Evaluates the model using cross-validation.

    Args:
        pipeline (sklearn.pipeline.Pipeline): The model pipeline to evaluate.
        X (pd.DataFrame): The features.
        y (pd.Series or np.array): The target.
        cv_splits (int): Number of cross-validation splits.     
    Returns:
        The mean +_ std_dev in cross-validation accuracy.
    """

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
    scores = cross_val_score(pipeline, X_train, Y_train, cv=cv, scoring='accuracy')
    mean_score = np.mean(scores)
    std_dev = np.std(scores)
    
    return mean_score, std_dev


def top_correlated_features(X: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Return the top n pairs of most correlated features (by absolute value).

    Args:
        X (pd.DataFrame): Input dataframe with numerical features.
        n (int): Number of top correlated pairs to return (default=20).

    Returns:
        pd.DataFrame: Columns: ['Feature 1', 'Feature 2', 'Correlation']
    """
    corr_matrix = X.corr(numeric_only=True)
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_pairs = corr_matrix.where(mask)

    corr_unstacked = corr_pairs.unstack().dropna().reset_index()
    corr_unstacked.columns = ['Feature 1', 'Feature 2', 'Correlation']

    corr_unstacked['AbsCorr'] = corr_unstacked['Correlation'].abs()
    corr_unstacked = corr_unstacked.sort_values(by='AbsCorr', ascending=False)
    return corr_unstacked.head(n)[['Feature 1', 'Feature 2', 'Correlation']]


def make_submission(pipeline, X_train, Y_train, X_test, test_battle_ids, name):
    submission = train_and_predict(pipeline, X_train, Y_train, X_test, test_battle_ids)

    submission.to_csv(f"submission_{name}.csv", index=False)

    print(f"Submission saved to submission_{name}.csv")