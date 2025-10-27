import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV

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


def create_model_pipeline_PCA(n_components=11, c_value=1.0, random_state=0):
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
        ('model', LogisticRegression(C=c_value, random_state=random_state))
    ]
    
    # Create the pipeline
    pipeline = Pipeline(steps)
    
    return pipeline


def create_model_pipeline(c_value=1.0, random_state=0):
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
        ('model', LogisticRegression(C=c_value,random_state=random_state))
    ]
    
    # Create the pipeline
    pipeline = Pipeline(steps)
    
    return pipeline


def create_model_pipeline_poly(c_value=1.0, degree=2, max_iter=1000, random_state=0):
    """
    Creates a scikit-learn pipeline that bundles preprocessing and the model.

    This pipeline will:
    1. Scale the data using StandardScaler.
    2. Generate polynomial features.
    3. Fit a Logistic Regression model.

    Args:
        random_state (int): A random state for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: The unfitted model pipeline.
    """
    
    from sklearn.preprocessing import PolynomialFeatures

    # Create a list of (name, transformer) tuples
    steps = [
        ('scaler', StandardScaler()),
        ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),
        ('model', LogisticRegression(C=c_value, random_state=random_state, max_iter=max_iter))
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
    
    # Use the same StratifiedKFold as in your evaluate_model function
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
    
    # Create the GridSearchCV object
    # scoring='accuracy' is used by default, but we'll be explicit.
    # n_jobs=-1 uses all available CPU cores to speed up the search.
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=1  # Set to 1 or 2 to see search progress
    )
    
    # Fit the grid search to the data
    grid_search.fit(X_train, Y_train)
    
    print("\nGrid Search Complete.")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.8f}")
    
    # Return the best model found
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

