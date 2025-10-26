import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

def create_model_pipeline_PCA(n_components=11, random_state=0):
    """
    Creates a scikit-learn pipeline that bundles preprocessing and the model.

    This pipeline will:
    1. Scale the data using StandardScaler.
    2. Perform PCA, reducing to `n_components`.
    3. Fit a Logistic Regression model.

    Args:
        n_components (int): The number of principal components to keep.
        random_state (int): A random state for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: The unfitted model pipeline.
    """
    
    # Create a list of (name, transformer) tuples
    steps = [
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=random_state)),
        ('model', LogisticRegression(random_state=random_state))
    ]
    
    # Create the pipeline
    pipeline = Pipeline(steps)
    
    return pipeline


def create_model_pipeline(random_state=0):
    """
    Creates a scikit-learn pipeline that bundles preprocessing and the model.

    This pipeline will:
    1. Scale the data using StandardScaler.
    2. Fit a Logistic Regression model.

    Args:
        random_state (int): A random state for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: The unfitted model pipeline.
    """
    
    # Create a list of (name, transformer) tuples
    steps = [
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=random_state))
    ]
    
    # Create the pipeline
    pipeline = Pipeline(steps)
    
    return pipeline


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
    
    # --- 1. Train the Pipeline ---
    # This single call fits the scaler, fits the PCA,
    # transforms the data, and fits the model.
    print("Training the pipeline...")
    pipeline.fit(X_train, Y_train)
    print("Training complete.")
    
    # --- 2. Make Predictions ---
    # This single call transforms the X_test data using the
    # *already-fitted* scaler and PCA, then makes predictions.
    print("Making predictions...")
    Y_pred = pipeline.predict(X_test)
    print("Predictions complete.")
    
    # --- 3. Generate Submission File ---
    submission = pd.DataFrame({
        'battle_id': test_battle_ids,
        'player_won': Y_pred
    })
    
    return submission


def plot_pca_variance(X):
    """
    Plots the explained variance ratio of the first `n_components` principal components.

    Args:
        X (pd.DataFrame): The input features.
    """
    n_components = X.shape[1]
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    pca.fit(X_std)
    
    evr = pca.explained_variance_ratio_
    evr_cumsum = np.cumsum(evr)
    
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_components + 1), evr_cumsum, marker='o', linestyle='--')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(range(1, n_components + 1))
    plt.ylim(0, 1.05)
    plt.grid()
    plt.show()
    
    return evr_cumsum


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

