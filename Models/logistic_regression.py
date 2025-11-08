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